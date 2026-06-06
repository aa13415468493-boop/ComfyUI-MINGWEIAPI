import os
import json
import time
import uuid
import base64
import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import folder_paths
from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile


def _tensor_to_pil(image):
    i = 255.0 * image.cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    return Image.fromarray(i)


def tensor2pil(image):
    return [_tensor_to_pil(image[i]) for i in range(image.shape[0])]


def pil2tensor(image):
    if isinstance(image, list):
        images = [np.array(img).astype(np.float32) / 255.0 for img in image]
        return torch.from_numpy(np.stack(images))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _empty_image_tensor():
    arr = np.zeros((1, 1, 1, 3), dtype=np.float32)
    return torch.from_numpy(arr)


def _empty_video():
    class _EmptyVideo:
        def __init__(self):
            self.is_empty = True

        def get_dimensions(self):
            return 1, 1

        def save_to(self, output_path, format="auto", codec="auto", metadata=None):
            try:
                import cv2
                frame = np.zeros((1, 1, 3), dtype=np.uint8)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, 1.0, (1, 1))
                out.write(frame)
                out.release()
                return True
            except Exception:
                return False

    return _EmptyVideo()


def _read_local_config():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_api_key(input_key):
    env_key = os.environ.get("KIE_API_KEY", "").strip()
    if env_key:
        return env_key
    local_key = _read_local_config().get("api_key", "")
    if isinstance(local_key, str) and local_key.strip():
        return local_key.strip()
    if isinstance(input_key, str) and input_key.strip():
        return input_key.strip()
    return ""


def _image_to_data_url(image):
    pil_image = tensor2pil(image)[0]
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def _download_to_temp(url, suffix):
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"kie_mj_{uuid.uuid4().hex[:8]}{suffix}"
    path = os.path.join(temp_dir, filename)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


class _KieMJClient:
    def __init__(self, api_key, base_url="https://api.kie.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def create_task(self, payload):
        url = f"{self.base_url}/api/v1/jobs/createTask"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def fetch_task(self, task_id):
        url = f"{self.base_url}/api/v1/mj/record-info"
        resp = requests.get(url, headers=self._headers(), params={"taskId": task_id}, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def poll_task(self, task_id, poll_interval, max_poll):
        for _ in range(max_poll):
            data = self.fetch_task(task_id)
            state = str(data.get("data", {}).get("state", "")).lower()
            if state in {"success", "succeeded", "complete", "completed"}:
                return data
            if state in {"fail", "failed", "error"}:
                return data
            time.sleep(poll_interval)
        return data


def _extract_result_urls(data):
    if not data:
        return []
    data_obj = data.get("data", {})
    result_info = data_obj.get("resultInfoJson")
    if isinstance(result_info, str):
        try:
            result_info = json.loads(result_info)
        except Exception:
            result_info = {}
    if isinstance(result_info, dict):
        urls = result_info.get("resultUrls") or result_info.get("result_urls")
        if isinstance(urls, list):
            out = []
            for item in urls:
                if isinstance(item, dict) and "resultUrl" in item:
                    out.append(item["resultUrl"])
                elif isinstance(item, str):
                    out.append(item)
            return out
        if "resultUrl" in result_info:
            return [result_info["resultUrl"]]
    return []


class MJ_kie_MW_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_type": (["mj_txt2img", "mj_img2img"], {"default": "mj_txt2img"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "speed": (["Relax", "Fast", "Turbo"], {"default": "Fast"}),
                "model_version": (["7", "6.1", "6.0", "5.2", "5.1", "niji6", "niji7"], {"default": "7"}),
                "aspect_ratio": (["1:2", "9:16", "2:3", "3:4", "5:6", "6:5", "4:3", "3:2", "1:1", "16:9", "2:1"], {"default": "1:1"}),
                "stylization": ("INT", {"default": 100, "min": 0, "max": 1000}),
                "weirdness": ("INT", {"default": 0, "min": 0, "max": 3000}),
                "watermark": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_poll": ("INT", {"default": 60, "min": 1, "max": 600}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "result_url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-MJ"

    def generate(
        self,
        task_type,
        prompt,
        speed,
        model_version,
        aspect_ratio,
        stylization,
        weirdness,
        watermark,
        poll_interval,
        max_poll,
        api_key,
        image=None,
        image_url="",
    ):
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            return (_empty_image_tensor(), "", "missing_api_key")
        client = _KieMJClient(resolved_key)
        file_url = ""
        if isinstance(image_url, str) and image_url.strip():
            file_url = image_url.strip()
        elif image is not None:
            file_url = _image_to_data_url(image)
        payload = {
            "model": task_type,
            "input": {
                "prompt": prompt,
                "speed": speed,
                "version": model_version,
                "aspectRatio": aspect_ratio,
                "stylization": stylization,
                "weirdness": weirdness,
                "waterMark": watermark,
                "fileUrl": file_url,
                "taskType": task_type,
            },
        }
        resp = client.create_task(payload)
        task_id = resp.get("task_id") or resp.get("data", {}).get("taskId") or resp.get("taskId") or ""
        if not task_id:
            return (_empty_image_tensor(), "", "task_id_missing")
        result = client.poll_task(task_id, poll_interval, max_poll)
        urls = _extract_result_urls(result)
        if not urls:
            return (_empty_image_tensor(), "", task_id)
        result_url = urls[0]
        image_path = _download_to_temp(result_url, ".png")
        pil_image = Image.open(image_path).convert("RGB")
        return (pil2tensor(pil_image), result_url, task_id)


class MJ_kie_MW_Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "speed": (["Relax", "Fast", "Turbo"], {"default": "Fast"}),
                "model_version": (["7", "6.1", "6.0", "5.2", "5.1", "niji6", "niji7"], {"default": "7"}),
                "aspect_ratio": (["1:2", "9:16", "2:3", "3:4", "5:6", "6:5", "4:3", "3:2", "1:1", "16:9", "2:1"], {"default": "1:1"}),
                "stylization": ("INT", {"default": 100, "min": 0, "max": 1000}),
                "weirdness": ("INT", {"default": 0, "min": 0, "max": 3000}),
                "watermark": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_poll": ("INT", {"default": 60, "min": 1, "max": 600}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "result_url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-MJ"

    def generate(
        self,
        prompt,
        speed,
        model_version,
        aspect_ratio,
        stylization,
        weirdness,
        watermark,
        poll_interval,
        max_poll,
        api_key,
        image=None,
        image_url="",
    ):
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            return (_empty_video(), "", "missing_api_key")
        client = _KieMJClient(resolved_key)
        file_url = ""
        if isinstance(image_url, str) and image_url.strip():
            file_url = image_url.strip()
        elif image is not None:
            file_url = _image_to_data_url(image)
        payload = {
            "model": "mj_video",
            "input": {
                "prompt": prompt,
                "speed": speed,
                "version": model_version,
                "aspectRatio": aspect_ratio,
                "stylization": stylization,
                "weirdness": weirdness,
                "waterMark": watermark,
                "fileUrl": file_url,
                "taskType": "mj_video",
            },
        }
        resp = client.create_task(payload)
        task_id = resp.get("task_id") or resp.get("data", {}).get("taskId") or resp.get("taskId") or ""
        if not task_id:
            return (_empty_video(), "", "task_id_missing")
        result = client.poll_task(task_id, poll_interval, max_poll)
        urls = _extract_result_urls(result)
        if not urls:
            return (_empty_video(), "", task_id)
        result_url = urls[0]
        video_path = _download_to_temp(result_url, ".mp4")
        return (VideoFromFile(video_path), result_url, task_id)


NODE_CLASS_MAPPINGS = {
    "MJ-kie-MW-图片": MJ_kie_MW_Image,
    "MJ-kie-MW-视频": MJ_kie_MW_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MJ-kie-MW-图片": "MJ-kie-MW-图片",
    "MJ-kie-MW-视频": "MJ-kie-MW-视频",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
