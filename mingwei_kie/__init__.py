import json
import os
import time
import uuid
import base64
from io import BytesIO
import threading

import folder_paths
from PIL import Image


try:
    import requests as _requests
except Exception:
    _requests = None

try:
    from urllib.parse import urlencode
except Exception:
    from urllib import urlencode

try:
    from urllib.request import Request, urlopen
except Exception:
    from urllib2 import Request, urlopen


try:
    from comfy.comfy_types import IO
except Exception:
    class _FallbackIO:
        VIDEO = "VIDEO"
    IO = _FallbackIO()

try:
    _string_types = (str, unicode)
except Exception:
    _string_types = (str,)


def _is_nonempty_string(v):
    try:
        return isinstance(v, _string_types) and bool(v.strip())
    except Exception:
        return False


def tensor2pil(image):
    import numpy as np
    import torch

    if isinstance(image, torch.Tensor):
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(tensor2pil(image[i]))
            return out

        numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return [Image.fromarray(numpy_image)]

    raise TypeError("image must be a torch.Tensor")


class _LocalOrUrlVideo:
    def __init__(self, video_path_or_url):
        self._value = video_path_or_url

    def get_dimensions(self):
        if not self._value:
            return 1280, 720
        if self._value.startswith("http"):
            return 1280, 720
        try:
            import cv2

            cap = cv2.VideoCapture(self._value)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception:
            return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if not self._value:
            return False
        if self._value.startswith("http"):
            _http_download_to_file(self._value, output_path, headers=None, timeout=300)
            return True

        import shutil

        shutil.copyfile(self._value, output_path)
        return True


def _read_local_config_api_key():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    try:
        try:
            f = open(config_path, "r", encoding="utf-8")
        except Exception:
            f = open(config_path, "r")
        try:
            data = json.load(f)
        finally:
            try:
                f.close()
            except Exception:
                pass
        for k in ("kie_api_key", "api_key", "KIE_API_KEY"):
            v = data.get(k)
            if _is_nonempty_string(v):
                return v.strip()
    except Exception:
        return ""
    return ""


def _resolve_api_key(widget_value):
    for k in ("KIE_API_KEY", "KIEAI_API_KEY"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    cfg = _read_local_config_api_key()
    if cfg:
        return cfg
    if widget_value and widget_value.strip():
        return widget_value.strip()
    return ""

def _resolve_first_image_url(
    image_url,
    image_tensor,
    image_url_2,
    image_tensor_2,
    image_url_3,
    image_tensor_3,
    api_key,
    insecure_ssl=False,
):
    for u in (image_url, image_url_2, image_url_3):
        u = (u or "").strip()
        if u:
            return u
    for t in (image_tensor, image_tensor_2, image_tensor_3):
        if t is not None:
            return _kie_file_base64_upload(t, api_key, insecure_ssl=bool(insecure_ssl))
    return ""


def _split_text_items(value):
    if not _is_nonempty_string(value):
        return []
    items = []
    for part in value.replace("\r", "\n").replace(",", "\n").split("\n"):
        item = part.strip()
        if item:
            items.append(item)
    return items


def _build_url(url, params):
    if not params:
        return url
    query = urlencode(params)
    if "?" in url:
        return url + "&" + query
    return url + "?" + query


def _http_json(method, url, headers=None, params=None, json_body=None, timeout=60, insecure_ssl=False):
    headers = headers or {}
    final_url = _build_url(url, params)
    body_bytes = None
    if json_body is not None:
        body_bytes = json.dumps(json_body).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    if _requests is not None:
        try:
            req_headers = dict(headers)
            if "Connection" not in req_headers:
                req_headers["Connection"] = "close"
            resp = _requests.request(
                method,
                final_url,
                headers=req_headers,
                data=body_bytes,
                timeout=timeout,
                verify=(not insecure_ssl),
            )
            status = int(getattr(resp, "status_code", 0) or 0)
            text = getattr(resp, "text", "")
            if status < 200 or status >= 300:
                raise ValueError("HTTP {}: {}".format(status, text))
            try:
                return resp.json()
            except Exception:
                return json.loads(text)
        except Exception:
            pass

    req = Request(final_url, data=body_bytes, headers=headers)
    try:
        req.get_method = lambda: method
    except Exception:
        pass
    try:
        import ssl

        if insecure_ssl:
            ctx = ssl._create_unverified_context()
        else:
            try:
                ctx = ssl.create_default_context()
                try:
                    if hasattr(ssl, "TLSVersion") and hasattr(ctx, "minimum_version"):
                        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                except Exception:
                    pass
            except Exception:
                ctx = None
        if ctx is not None:
            resp = urlopen(req, timeout=timeout, context=ctx)
        else:
            resp = urlopen(req, timeout=timeout)
    except Exception:
        resp = urlopen(req, timeout=timeout)
    status = int(getattr(resp, "getcode", lambda: 200)() or 200)
    raw = resp.read()
    if status < 200 or status >= 300:
        raise ValueError("HTTP {}: {}".format(status, raw[:500]))
    return json.loads(raw.decode("utf-8"))


def _http_download_to_file(url, output_path, headers=None, timeout=300, insecure_ssl=False):
    headers = headers or {}
    if _requests is not None:
        try:
            req_headers = dict(headers)
            if "Connection" not in req_headers:
                req_headers["Connection"] = "close"
            resp = _requests.get(url, headers=req_headers, stream=True, timeout=timeout, verify=(not insecure_ssl))
            status = int(getattr(resp, "status_code", 0) or 0)
            if status < 200 or status >= 300:
                raise ValueError("HTTP {}: {}".format(status, getattr(resp, "text", "")))
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return
        except Exception:
            pass

    req = Request(url, headers=headers)
    try:
        import ssl

        if insecure_ssl:
            ctx = ssl._create_unverified_context()
        else:
            try:
                ctx = ssl.create_default_context()
                try:
                    if hasattr(ssl, "TLSVersion") and hasattr(ctx, "minimum_version"):
                        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                except Exception:
                    pass
            except Exception:
                ctx = None
        if ctx is not None:
            resp = urlopen(req, timeout=timeout, context=ctx)
        else:
            resp = urlopen(req, timeout=timeout)
    except Exception:
        resp = urlopen(req, timeout=timeout)
    status = int(getattr(resp, "getcode", lambda: 200)() or 200)
    if status < 200 or status >= 300:
        raise ValueError("HTTP {}".format(status))
    with open(output_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _kie_file_base64_upload(image_tensor, api_key, insecure_ssl=False):
    pil_image = tensor2pil(image_tensor)[0]
    if getattr(pil_image, "mode", "") not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGB")

    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format="PNG", optimize=True)
    img_byte_arr.seek(0)

    file_name = "sora2_{}.png".format(uuid.uuid4().hex[:10])
    data_url = "data:image/png;base64," + base64.b64encode(img_byte_arr.read()).decode("ascii")

    data = _http_json(
        "POST",
        "https://kieai.redpandaai.co/api/file-base64-upload",
        headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
        json_body={"base64Data": data_url, "uploadPath": "images/user-uploads", "fileName": file_name},
        timeout=300,
        insecure_ssl=insecure_ssl,
    )
    file_data = (data or {}).get("data") or {}
    for k in ("downloadUrl", "fileUrl"):
        v = file_data.get(k)
        if _is_nonempty_string(v):
            return v.strip()
    raise ValueError("上传图片失败: {}".format(data))

def _kie_local_file_upload(file_path, api_key, upload_path, file_name=None, insecure_ssl=False):
    file_path = (file_path or "").strip().replace('"', "").replace("'", "")
    if not file_path:
        raise ValueError("video_path 不能为空")
    if not os.path.exists(file_path):
        raise ValueError("文件不存在: {}".format(file_path))

    base_url = "https://kieai.redpandaai.co"
    upload_path = (upload_path or "").strip()
    file_name = (file_name or "").strip() or os.path.basename(file_path)

    if _requests is not None:
        headers = {"Authorization": "Bearer {}".format(api_key)}
        data = {}
        if upload_path:
            data["uploadPath"] = upload_path
        if file_name:
            data["fileName"] = file_name
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            resp = _requests.post(
                "{}/api/file-stream-upload".format(base_url),
                headers=headers,
                data=data,
                files=files,
                timeout=600,
                verify=(not insecure_ssl),
            )
        status = int(getattr(resp, "status_code", 0) or 0)
        text = getattr(resp, "text", "")
        if status < 200 or status >= 300:
            raise ValueError("HTTP {}: {}".format(status, text))
        try:
            payload = resp.json()
        except Exception:
            payload = json.loads(text)
        file_data = (payload or {}).get("data") or {}
        for k in ("downloadUrl", "fileUrl"):
            v = file_data.get(k)
            if _is_nonempty_string(v):
                return v.strip()
        raise ValueError("上传文件失败: {}".format(payload))

    with open(file_path, "rb") as f:
        raw = f.read()
    data_url = "data:application/octet-stream;base64," + base64.b64encode(raw).decode("ascii")
    data = _http_json(
        "POST",
        "{}/api/file-base64-upload".format(base_url),
        headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
        json_body={"base64Data": data_url, "uploadPath": upload_path, "fileName": file_name},
        timeout=600,
        insecure_ssl=insecure_ssl,
    )
    file_data = (data or {}).get("data") or {}
    for k in ("downloadUrl", "fileUrl"):
        v = file_data.get(k)
        if _is_nonempty_string(v):
            return v.strip()
    raise ValueError("上传文件失败: {}".format(data))


def _kie_create_task(api_key, model, input_payload, insecure_ssl=False):
    last_err = None
    for attempt in range(2):
        try:
            result = _http_json(
                "POST",
                "https://api.kie.ai/api/v1/jobs/createTask",
                headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
                json_body={"model": model, "input": input_payload},
                timeout=300,
                insecure_ssl=insecure_ssl,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            msg = "{}".format(e)
            if attempt == 0 and ("HTTP 500" in msg) and ("upstream API service timed out" in msg):
                time.sleep(2.0)
                continue
            raise
    if last_err is not None:
        raise last_err
    task_id = (((result or {}).get("data") or {}).get("taskId")) or ""
    if not task_id:
        raise ValueError("createTask 未返回 taskId: {}".format(result))
    return task_id


def _kie_poll_result(api_key, task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=False):
    started = time.time()
    while True:
        result = _http_json(
            "GET",
            "https://api.kie.ai/api/v1/jobs/recordInfo",
            headers={"Authorization": "Bearer {}".format(api_key)},
            params={"taskId": task_id},
            timeout=60,
            insecure_ssl=insecure_ssl,
        )
        data = (result or {}).get("data") or {}
        state = (data.get("state") or "").lower()

        if state == "success":
            return data

        if state in ("fail", "failed", "error"):
            fail_msg = data.get("failMsg") or ""
            raise ValueError("任务失败: {}".format(fail_msg or data))

        if time.time() - started >= max_wait_s:
            raise TimeoutError("任务超时未完成: {}".format(task_id))

        time.sleep(poll_interval_s)


def _extract_video_url(task_data):
    result_json = task_data.get("resultJson") or ""
    if _is_nonempty_string(result_json):
        try:
            parsed = json.loads(result_json)
        except Exception:
            parsed = {}
    else:
        parsed = {}

    candidates = []
    if isinstance(parsed, dict):
        urls = parsed.get("resultUrls")
        if isinstance(urls, list):
            candidates.extend([u for u in urls if _is_nonempty_string(u)])
        for k in ("resultUrl", "videoUrl", "video_url", "url"):
            v = parsed.get(k)
            if _is_nonempty_string(v):
                candidates.append(v)

    for u in candidates:
        u = u.strip()
        if u.startswith("http"):
            return u
    return ""

def _extract_character_id(task_data):
    result_json = task_data.get("resultJson") or ""
    if _is_nonempty_string(result_json):
        try:
            parsed = json.loads(result_json)
        except Exception:
            parsed = {}
    else:
        parsed = {}

    if isinstance(parsed, dict):
        for k in ("character_id", "characterId", "id"):
            v = parsed.get(k)
            if _is_nonempty_string(v):
                return v.strip()
    return ""


def _download_video_to_temp(video_url, insecure_ssl=False):
    out_dir = os.path.join(folder_paths.get_temp_directory(), "kie_sora2")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "sora2_{}.mp4".format(uuid.uuid4().hex[:10]))

    _http_download_to_file(video_url, out_path, headers=None, timeout=600, insecure_ssl=insecure_ssl)
    return out_path


_TASKS_LOCK = threading.Lock()


def _tasks_dir():
    d = os.path.join(folder_paths.get_temp_directory(), "kie_sora2_async")
    if not os.path.isdir(d):
        try:
            os.makedirs(d)
        except Exception:
            pass
    return d


def _tasks_file_path():
    return os.path.join(_tasks_dir(), "tasks.json")


def _read_tasks():
    path = _tasks_file_path()
    if not os.path.exists(path):
        return {}
    try:
        try:
            f = open(path, "r", encoding="utf-8")
        except Exception:
            f = open(path, "r")
        try:
            data = json.load(f)
        finally:
            try:
                f.close()
            except Exception:
                pass
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _write_tasks(tasks):
    path = _tasks_file_path()
    try:
        try:
            f = open(path, "w", encoding="utf-8")
        except Exception:
            f = open(path, "w")
        try:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        finally:
            try:
                f.close()
            except Exception:
                pass
    except Exception:
        return


def _format_task_line(task_id, tinfo):
    status = (tinfo.get("state") or tinfo.get("status") or "unknown")
    prompt = (tinfo.get("prompt") or "")[:25]
    if status == "running":
        p = tinfo.get("progress")
        if p is not None:
            status = "running {}%".format(p)
    return "[{}] {}... - {}...".format(status, task_id[:8], prompt)


class Sora2StableKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["文生视频", "图生视频"], {"default": "文生视频"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["横版 16:9", "竖版 9:16"], {"default": "横版 16:9"}),
                "seconds": (["10", "15"], {"default": "10"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "insecure_ssl": ("BOOLEAN", {"default": False}),
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response_json")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie"

    def _map_aspect_ratio(self, v):
        if "竖" in v or "9:16" in v:
            return "portrait"
        return "landscape"

    def generate(
        self,
        mode,
        prompt,
        aspect_ratio,
        seconds,
        seed,
        insecure_ssl,
        remove_watermark,
        api_key,
        image=None,
        image_url="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        aspect_ratio = self._map_aspect_ratio(aspect_ratio)
        n_frames = str(seconds).strip() or "10"
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        is_i2v = (mode == "图生视频")
        model = "sora-2-image-to-video-stable" if is_i2v else "sora-2-text-to-video-stable"

        input_payload = {
            "prompt": (prompt or "").strip(),
            "aspect_ratio": aspect_ratio,
            "n_frames": n_frames,
            "seed": seed_value,
            "remove_watermark": bool(remove_watermark),
            "upload_method": "s3",
        }

        if is_i2v:
            image_url = (image_url or "").strip()
            if not image_url:
                if image is None:
                    raise ValueError("图生视频模式需要提供 image 或 image_url")
                image_url = _kie_file_base64_upload(image, resolved_api_key, insecure_ssl=bool(insecure_ssl))
            input_payload["image_urls"] = [image_url]

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(5)

        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(20)

        task_data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(85)

        video_url = _extract_video_url(task_data)
        if not video_url:
            raise ValueError("任务完成但未返回视频地址: {}".format(task_data))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "input": input_payload,
                "state": task_data.get("state"),
                "resultJson": task_data.get("resultJson"),
                "video_url": video_url,
                "video_path": video_path,
                "seed": seed_value,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json)


class Sora2BasicKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["文生视频", "图生视频"], {"default": "文生视频"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["横版 16:9", "竖版 9:16"], {"default": "横版 16:9"}),
                "seconds": (["10", "15"], {"default": "10"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "insecure_ssl": ("BOOLEAN", {"default": False}),
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response_json")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie"

    def _map_aspect_ratio(self, v):
        if "竖" in v or "9:16" in v:
            return "portrait"
        return "landscape"

    def generate(
        self,
        mode,
        prompt,
        aspect_ratio,
        seconds,
        seed,
        insecure_ssl,
        remove_watermark,
        api_key,
        image=None,
        image_url="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        aspect_ratio = self._map_aspect_ratio(aspect_ratio)
        n_frames = str(seconds).strip() or "10"
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        is_i2v = (mode == "图生视频")
        model = "sora-2-image-to-video" if is_i2v else "sora-2-text-to-video"

        input_payload = {
            "prompt": (prompt or "").strip(),
            "aspect_ratio": aspect_ratio,
            "n_frames": n_frames,
            "seed": seed_value,
            "remove_watermark": bool(remove_watermark),
            "upload_method": "s3",
        }

        if is_i2v:
            image_url = (image_url or "").strip()
            if not image_url:
                if image is None:
                    raise ValueError("图生视频模式需要提供 image 或 image_url")
                image_url = _kie_file_base64_upload(image, resolved_api_key, insecure_ssl=bool(insecure_ssl))
            input_payload["image_urls"] = [image_url]

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(5)

        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(20)

        task_data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(85)

        video_url = _extract_video_url(task_data)
        if not video_url:
            raise ValueError("任务完成但未返回视频地址: {}".format(task_data))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "input": input_payload,
                "state": task_data.get("state"),
                "resultJson": task_data.get("resultJson"),
                "video_url": video_url,
                "video_path": video_path,
                "seed": seed_value,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json)


class Sora2Kie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["基础版", "稳定版"], {"default": "基础版", "display_name": "🧩 模型版本"}),
                "mode": (["文生视频", "图生视频"], {"default": "文生视频", "display_name": "🎬 生成模式"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "aspect_ratio": (["横版 16:9", "竖版 9:16"], {"default": "横版 16:9", "display_name": "🖼️ 视频比例"}),
                "seconds": (["10", "15"], {"default": "10", "display_name": "⏱️ 秒数"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "display_name": "🎲 随机种子"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "remove_watermark": ("BOOLEAN", {"default": True, "display_name": "🚫 去水印"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL"}),
                "image_2": ("IMAGE",),
                "image_url_2": ("STRING", {"default": "", "display_name": "🔗 图片URL2"}),
                "image_3": ("IMAGE",),
                "image_url_3": ("STRING", {"default": "", "display_name": "🔗 图片URL3"}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "🧾 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"

    def _map_aspect_ratio(self, v):
        if "竖" in v or "9:16" in v:
            return "portrait"
        return "landscape"

    def generate(
        self,
        model_version,
        mode,
        prompt,
        aspect_ratio,
        seconds,
        seed,
        insecure_ssl,
        remove_watermark,
        api_key,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        aspect_ratio = self._map_aspect_ratio(aspect_ratio)
        n_frames = str(seconds).strip() or "10"
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        is_i2v = (mode == "图生视频")
        is_stable = (model_version == "稳定版")
        if is_i2v:
            model = "sora-2-image-to-video-stable" if is_stable else "sora-2-image-to-video"
        else:
            model = "sora-2-text-to-video-stable" if is_stable else "sora-2-text-to-video"

        input_payload = {
            "prompt": (prompt or "").strip(),
            "aspect_ratio": aspect_ratio,
            "n_frames": n_frames,
            "seed": seed_value,
            "remove_watermark": bool(remove_watermark),
            "upload_method": "s3",
        }

        if is_i2v:
            resolved_image_url = _resolve_first_image_url(
                image_url=image_url,
                image_tensor=image,
                image_url_2=image_url_2,
                image_tensor_2=image_2,
                image_url_3=image_url_3,
                image_tensor_3=image_3,
                api_key=resolved_api_key,
                insecure_ssl=bool(insecure_ssl),
            )
            if not resolved_image_url:
                raise ValueError("图生视频模式需要提供 image/image_2/image_3 或 image_url/image_url_2/image_url_3")
            input_payload["image_urls"] = [resolved_image_url]

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(5)

        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(20)

        task_data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(85)

        video_url = _extract_video_url(task_data)
        if not video_url:
            raise ValueError("任务完成但未返回视频地址: {}".format(task_data))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "input": input_payload,
                "state": task_data.get("state"),
                "resultJson": task_data.get("resultJson"),
                "video_url": video_url,
                "video_path": video_path,
                "seed": seed_value,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json, task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json, task_id)


class Sora2SubmitTaskKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["基础版", "稳定版"], {"default": "基础版", "display_name": "🧩 模型版本"}),
                "mode": (["文生视频", "图生视频"], {"default": "文生视频", "display_name": "🎬 生成模式"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "aspect_ratio": (["横版 16:9", "竖版 9:16"], {"default": "横版 16:9", "display_name": "🖼️ 视频比例"}),
                "seconds": (["10", "15"], {"default": "10", "display_name": "⏱️ 秒数"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "display_name": "🎲 随机种子"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "remove_watermark": ("BOOLEAN", {"default": True, "display_name": "🚫 去水印"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL"}),
                "image_2": ("IMAGE",),
                "image_url_2": ("STRING", {"default": "", "display_name": "🔗 图片URL2"}),
                "image_3": ("IMAGE",),
                "image_url_3": ("STRING", {"default": "", "display_name": "🔗 图片URL3"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🧾 response_json", "📋 report", "🆔 task_id")
    FUNCTION = "submit"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def _map_aspect_ratio(self, v):
        if "竖" in v or "9:16" in v:
            return "portrait"
        return "landscape"

    def submit(
        self,
        model_version,
        mode,
        prompt,
        aspect_ratio,
        seconds,
        seed,
        insecure_ssl,
        remove_watermark,
        api_key,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        aspect_ratio_mapped = self._map_aspect_ratio(aspect_ratio)
        n_frames = str(seconds).strip() or "10"
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        is_i2v = (mode == "图生视频")
        is_stable = (model_version == "稳定版")
        if is_i2v:
            model = "sora-2-image-to-video-stable" if is_stable else "sora-2-image-to-video"
        else:
            model = "sora-2-text-to-video-stable" if is_stable else "sora-2-text-to-video"

        input_payload = {
            "prompt": (prompt or "").strip(),
            "aspect_ratio": aspect_ratio_mapped,
            "n_frames": n_frames,
            "seed": seed_value,
            "remove_watermark": bool(remove_watermark),
            "upload_method": "s3",
        }

        if is_i2v:
            resolved_image_url = _resolve_first_image_url(
                image_url=image_url,
                image_tensor=image,
                image_url_2=image_url_2,
                image_tensor_2=image_2,
                image_url_3=image_url_3,
                image_tensor_3=image_3,
                api_key=resolved_api_key,
                insecure_ssl=bool(insecure_ssl),
            )
            if not resolved_image_url:
                raise ValueError("图生视频模式需要提供 image/image_2/image_3 或 image_url/image_url_2/image_url_3")
            input_payload["image_urls"] = [resolved_image_url]

        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))

        with _TASKS_LOCK:
            tasks = _read_tasks()
            tasks[task_id] = {
                "taskId": task_id,
                "model_version": model_version,
                "mode": mode,
                "model": model,
                "prompt": input_payload.get("prompt", ""),
                "input": input_payload,
                "state": "pending",
                "created_at": time.time(),
                "video_url": "",
                "video_path": "",
                "downloaded": False,
            }
            _write_tasks(tasks)

        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "input": input_payload,
            },
            ensure_ascii=False,
        )
        report = "已提交任务: {}...（离线排队）".format(task_id[:8])
        return (response_json, report, task_id)


class Sora2QueryTasksKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("📋 report",)
    FUNCTION = "query"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def query(self, api_key="", insecure_ssl=False):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        updated = False
        for task_id, tinfo in list(tasks.items()):
            state = (tinfo.get("state") or "").lower()
            if state in ("success", "fail", "failed", "error"):
                continue
            if tinfo.get("downloaded") is True:
                continue
            try:
                data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=0.0, max_wait_s=0.0, insecure_ssl=bool(insecure_ssl))
                if isinstance(data, dict):
                    tasks[task_id]["state"] = data.get("state") or tasks[task_id].get("state") or "unknown"
                    tasks[task_id]["resultJson"] = data.get("resultJson")
                    video_url = _extract_video_url(data)
                    if video_url:
                        tasks[task_id]["video_url"] = video_url
                    updated = True
            except TimeoutError:
                continue
            except Exception:
                continue

        if updated:
            with _TASKS_LOCK:
                _write_tasks(tasks)

        sorted_tasks = sorted(tasks.items(), key=lambda kv: kv[1].get("created_at", 0.0), reverse=True)
        lines = ["--- 任务队列总览 ---"]
        for task_id, tinfo in sorted_tasks[:50]:
            lines.append(_format_task_line(task_id, tinfo))
        return ("\n".join(lines),)


class Sora2GetNextVideoKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "📋 report", "🧾 response_json", "🆔 task_id")
    FUNCTION = "get_next"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def get_next(self, api_key="", insecure_ssl=False):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        sorted_tasks = sorted(tasks.items(), key=lambda kv: kv[1].get("created_at", 0.0))
        selected_id = None
        for task_id, tinfo in sorted_tasks:
            if tinfo.get("downloaded") is True:
                continue
            state = (tinfo.get("state") or "").lower()
            if state == "success" or state == "succeeded":
                selected_id = task_id
                break

        if not selected_id:
            return (_LocalOrUrlVideo(""), "当前无已完成任务可下载。", json.dumps({}, ensure_ascii=False), "")

        tinfo = tasks.get(selected_id) or {}
        video_url = (tinfo.get("video_url") or "").strip()
        if not video_url:
            try:
                data = _kie_poll_result(api_key=resolved_api_key, task_id=selected_id, poll_interval_s=0.0, max_wait_s=0.0, insecure_ssl=bool(insecure_ssl))
                video_url = _extract_video_url(data)
            except Exception:
                video_url = ""

        if not video_url:
            return (
                _LocalOrUrlVideo(""),
                "任务已完成但未找到视频地址: {}...".format(selected_id[:8]),
                json.dumps({"taskId": selected_id}, ensure_ascii=False),
                selected_id,
            )

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        tasks[selected_id]["downloaded"] = True
        tasks[selected_id]["video_path"] = video_path
        tasks[selected_id]["video_url"] = video_url

        with _TASKS_LOCK:
            _write_tasks(tasks)

        response_json = json.dumps(
            {
                "taskId": selected_id,
                "video_url": video_url,
                "video_path": video_path,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), "下载成功: {}...".format(selected_id[:8]), response_json, selected_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), "下载成功: {}...".format(selected_id[:8]), response_json, selected_id)


class Sora2FromOriginTaskCharacterKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "origin_task_id": ("STRING", {"default": "", "display_name": "🎬 原视频 taskId"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01, "display_name": "⏱️ 开始秒"}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10000.0, "step": 0.01, "display_name": "⏱️ 结束秒"}),
                "character_prompt": ("STRING", {"multiline": True, "default": "", "display_name": "🧍 角色描述"}),
                "character_user_name": ("STRING", {"default": "", "display_name": "🏷️ 角色名称"}),
                "safety_instruction": ("STRING", {"multiline": True, "default": "", "display_name": "🛡️ 安全说明"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🧬 character_id", "📋 report", "🧾 response_json")
    FUNCTION = "create_character"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"

    def create_character(
        self,
        origin_task_id,
        start_time,
        end_time,
        character_prompt,
        character_user_name,
        safety_instruction,
        insecure_ssl,
        api_key,
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        origin_task_id = (origin_task_id or "").strip()
        if not origin_task_id:
            raise ValueError("缺少 origin_task_id")

        try:
            st = float(start_time)
        except Exception:
            st = 0.0
        try:
            et = float(end_time)
        except Exception:
            et = 0.0
        if et <= st:
            raise ValueError("end_time 必须大于 start_time")

        timestamps = "{:g},{:g}".format(st, et)

        input_payload = {
            "origin_task_id": origin_task_id,
            "timestamps": timestamps,
            "character_prompt": (character_prompt or "").strip(),
        }
        if _is_nonempty_string(character_user_name):
            input_payload["character_user_name"] = character_user_name.strip()
        if _is_nonempty_string(safety_instruction):
            input_payload["safety_instruction"] = safety_instruction.strip()

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(5)

        model = "sora-2-characters-pro"
        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(25)

        task_data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(95)

        character_id = _extract_character_id(task_data)
        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "input": input_payload,
                "state": task_data.get("state"),
                "resultJson": task_data.get("resultJson"),
                "character_id": character_id,
            },
            ensure_ascii=False,
        )
        report = "创建成功: {}...".format(character_id[:8]) if character_id else "创建完成，但未返回 character_id"
        pbar.update_absolute(100)
        return (character_id, report, response_json)

class Sora2UploadCharacterKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "请填入本地视频绝对路径，例如：E:\\\\video\\\\1.mp4", "display_name": "🎞️ 本地视频路径"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01, "display_name": "⏱️ 开始秒"}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10000.0, "step": 0.01, "display_name": "⏱️ 结束秒"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "character_prompt": ("STRING", {"multiline": True, "default": "", "display_name": "🧍 角色描述"}),
                "safety_instruction": ("STRING", {"multiline": True, "default": "", "display_name": "🛡️ 安全说明"}),
                "character_user_name": ("STRING", {"default": "", "display_name": "🏷️ 角色名称"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🧬 character_id", "📋 report", "🧾 response_json")
    FUNCTION = "upload_create"
    CATEGORY = "🤖MINGWEI-API/MINGWEI-kie/sora2"

    def upload_create(
        self,
        video_path,
        start_time,
        end_time,
        insecure_ssl,
        api_key,
        character_prompt="",
        safety_instruction="",
        character_user_name="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        clean_path = (video_path or "").strip().replace('"', "").replace("'", "")
        if not clean_path:
            raise ValueError("video_path 不能为空")
        if not os.path.exists(clean_path):
            raise ValueError("文件不存在: {}".format(clean_path))

        try:
            st = float(start_time)
        except Exception:
            st = 0.0
        try:
            et = float(end_time)
        except Exception:
            et = 0.0
        if et <= st:
            raise ValueError("end_time 必须大于 start_time")

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(5)

        video_url = _kie_local_file_upload(
            file_path=clean_path,
            api_key=resolved_api_key,
            upload_path="videos/user-uploads",
            file_name="sora2_{}.{}".format(uuid.uuid4().hex[:10], os.path.basename(clean_path).split(".")[-1] if "." in os.path.basename(clean_path) else "mp4"),
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(25)

        input_payload = {"character_file_url": [video_url]}
        if _is_nonempty_string(character_prompt):
            input_payload["character_prompt"] = character_prompt.strip()
        if _is_nonempty_string(safety_instruction):
            input_payload["safety_instruction"] = safety_instruction.strip()
        if _is_nonempty_string(character_user_name):
            input_payload["character_user_name"] = character_user_name.strip()

        model = "sora-2-characters"
        task_id = _kie_create_task(api_key=resolved_api_key, model=model, input_payload=input_payload, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(55)

        task_data = _kie_poll_result(api_key=resolved_api_key, task_id=task_id, poll_interval_s=5.0, max_wait_s=1800.0, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(95)

        character_id = _extract_character_id(task_data)
        response_json = json.dumps(
            {
                "taskId": task_id,
                "model": model,
                "video_path": clean_path,
                "timestamps": "{:g},{:g}".format(st, et),
                "uploaded_video_url": video_url,
                "input": input_payload,
                "state": task_data.get("state"),
                "resultJson": task_data.get("resultJson"),
                "character_id": character_id,
            },
            ensure_ascii=False,
        )
        report = "创建成功: {}...".format(character_id[:8]) if character_id else "创建完成，但未返回 character_id"
        pbar.update_absolute(100)
        return (character_id, report, response_json)


class Veo31Kie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generationType": (["TEXT_2_VIDEO", "FIRST_AND_LAST_FRAMES_2_VIDEO", "REFERENCE_2_VIDEO"], {"default": "TEXT_2_VIDEO", "display_name": "🎬 generationType"}),
                "model": (["veo3_fast", "veo3_lite", "veo3"], {"default": "veo3_fast", "display_name": "🧠 模型"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "aspect_ratio": (["16:9", "9:16", "Auto"], {"default": "16:9", "display_name": "🖼️ 视频比例"}),
                "resolution": (["720p", "1080p", "4k"], {"default": "720p", "display_name": "📺 分辨率"}),
                "duration": (["4s", "6s", "8s"], {"default": "8s", "display_name": "⏱️ 秒数"}),
                "seed": ("INT", {"default": 12345, "min": 10000, "max": 99999, "display_name": "🎲 随机种子"}),
                "watermark": ("STRING", {"default": "", "display_name": "🏷️ 水印"}),
                "call_back_url": ("STRING", {"default": "", "display_name": "🔔 回调地址"}),
                "enable_fallback": ("BOOLEAN", {"default": False, "display_name": "🛟 启用回退"}),
                "enable_translation": ("BOOLEAN", {"default": True, "display_name": "🌍 启用翻译"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL1"}),
                "image_2": ("IMAGE",),
                "image_url_2": ("STRING", {"default": "", "display_name": "🔗 图片URL2"}),
                "image_3": ("IMAGE",),
                "image_url_3": ("STRING", {"default": "", "display_name": "🔗 图片URL3"}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "🧾 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-VEO/kie-veo"

    def _collect_image_urls(
        self,
        generation_type,
        resolved_api_key,
        insecure_ssl,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        urls = []
        for u in (image_url, image_url_2, image_url_3):
            u = (u or "").strip()
            if u:
                urls.append(u)
        for t in (image, image_2, image_3):
            if t is not None:
                urls.append(_kie_file_base64_upload(t, resolved_api_key, insecure_ssl=bool(insecure_ssl)))
        if generation_type == "FIRST_AND_LAST_FRAMES_2_VIDEO":
            if not urls:
                raise ValueError("图像转视频模式需要至少提供 1 张图片（image/image_url）")
            return urls[:2]
        if generation_type == "REFERENCE_2_VIDEO":
            if not urls:
                raise ValueError("视频参考模式需要至少提供 1 张参考图（支持最多 3 张）")
            return urls[:3]
        return []

    def _extract_task_id(self, submit_result):
        if not isinstance(submit_result, dict):
            return ""
        data = submit_result.get("data")
        if not isinstance(data, dict):
            return ""
        task_id = data.get("taskId")
        if _is_nonempty_string(task_id):
            return task_id.strip()
        return ""

    def _extract_veo_video_url_from_record(self, record_result):
        if not isinstance(record_result, dict):
            return ""
        data = record_result.get("data")
        if not isinstance(data, dict):
            return ""
        response = data.get("response")
        candidates = []
        for k in ("resultUrls", "fullResultUrls", "originUrls", "videoUrls"):
            urls = data.get(k)
            if isinstance(urls, list):
                for u in urls:
                    if _is_nonempty_string(u):
                        candidates.append(u.strip())
        for k in ("resultUrl", "videoUrl", "video_url", "url"):
            v = data.get(k)
            if _is_nonempty_string(v):
                candidates.append(v.strip())
        if isinstance(response, dict):
            for k in ("resultUrls", "fullResultUrls", "originUrls", "videoUrls"):
                urls = response.get(k)
                if isinstance(urls, list):
                    for u in urls:
                        if _is_nonempty_string(u):
                            candidates.append(u.strip())
            for k in ("resultUrl", "videoUrl", "video_url", "url"):
                v = response.get(k)
                if _is_nonempty_string(v):
                    candidates.append(v.strip())
        for u in candidates:
            if u.startswith("http"):
                return u
        return ""

    def _poll_veo_record(self, api_key, task_id, insecure_ssl=False, poll_interval_s=6.0, max_wait_s=1800.0):
        started = time.time()
        while True:
            record_result = _http_json(
                "GET",
                "https://api.kie.ai/api/v1/veo/record-info",
                headers={"Authorization": "Bearer {}".format(api_key)},
                params={"taskId": task_id},
                timeout=60,
                insecure_ssl=bool(insecure_ssl),
            )
            code = (record_result or {}).get("code")
            msg = "{}".format((record_result or {}).get("msg") or "")
            if code is not None and int(code) != 200:
                if int(code) == 422 and ("record status is not success" in msg or "record result data is blank" in msg or "record result data is empty" in msg):
                    if time.time() - started >= max_wait_s:
                        raise TimeoutError("Veo 3.1 任务超时未完成: {}".format(task_id))
                    time.sleep(poll_interval_s)
                    continue
                raise ValueError("Veo 3.1 查询失败: {}".format((record_result or {}).get("msg") or record_result))

            data = (record_result or {}).get("data") or {}
            response = (data.get("response") or {}) if isinstance(data, dict) else {}
            success_flag = response.get("successFlag")
            if success_flag is None and isinstance(data, dict):
                success_flag = data.get("successFlag")

            if success_flag in (1, "1"):
                return record_result
            if success_flag in (2, "2", 3, "3"):
                fail_msg = response.get("errorMessage") or data.get("errorMessage") or (record_result or {}).get("msg") or ""
                raise ValueError("Veo 3.1 任务失败: {}".format(fail_msg))

            if time.time() - started >= max_wait_s:
                raise TimeoutError("Veo 3.1 任务超时未完成: {}".format(task_id))
            time.sleep(poll_interval_s)

    def generate(
        self,
        generationType,
        model,
        prompt,
        aspect_ratio,
        resolution,
        duration,
        seed,
        watermark,
        call_back_url,
        enable_fallback,
        enable_translation,
        insecure_ssl,
        api_key,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        model_code = (model or "").strip() or "veo3_fast"
        generation_type = (generationType or "").strip() or "TEXT_2_VIDEO"
        if generation_type == "REFERENCE_2_VIDEO":
            model_code = "veo3_fast"
        image_urls = self._collect_image_urls(
            generation_type=generation_type,
            resolved_api_key=resolved_api_key,
            insecure_ssl=insecure_ssl,
            image=image,
            image_url=image_url,
            image_2=image_2,
            image_url_2=image_url_2,
            image_3=image_3,
            image_url_3=image_url_3,
        )

        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 12345
        if seed_value < 10000:
            seed_value = 10000
        if seed_value > 99999:
            seed_value = 99999

        request_body = {
            "prompt": (prompt or "").strip(),
            "model": model_code,
            "generationType": generation_type,
            "aspect_ratio": (aspect_ratio or "16:9").strip() or "16:9",
            "resolution": (resolution or "720p").strip() or "720p",
            "duration": int(str(duration or "8s").strip().lower().replace("s", "") or 8),
            "seeds": seed_value,
            "enableFallback": bool(enable_fallback),
            "enableTranslation": bool(enable_translation),
        }
        if image_urls:
            request_body["imageUrls"] = image_urls
        if _is_nonempty_string(watermark):
            request_body["watermark"] = watermark.strip()
        if _is_nonempty_string(call_back_url):
            request_body["callBackUrl"] = call_back_url.strip()

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(10)

        result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/veo/generate",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(35)

        result_code = (result or {}).get("code")
        if result_code is not None and int(result_code) != 200:
            raise ValueError("Veo 3.1 提交失败: {}".format((result or {}).get("msg") or result))

        task_id = self._extract_task_id(result)
        if not task_id:
            raise ValueError("Veo 3.1 提交成功但未返回 taskId: {}".format(result))

        task_data = self._poll_veo_record(
            api_key=resolved_api_key,
            task_id=task_id,
            poll_interval_s=6.0,
            max_wait_s=1800.0,
            insecure_ssl=bool(insecure_ssl),
        )
        video_url = self._extract_veo_video_url_from_record(task_data)
        pbar.update_absolute(85)

        if not video_url:
            raise ValueError("Veo 3.1 任务未返回视频地址: {}".format(result))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "request": request_body,
                "submit_response": result,
                "taskId": task_id,
                "task_data": task_data,
                "video_url": video_url,
                "video_path": video_path,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json, task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json, task_id)


class Veo31ExtendKie(Veo31Kie):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "display_name": "🆔 原视频任务ID"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 扩展提示词"}),
                "model": (["fast", "quality", "lite"], {"default": "fast", "display_name": "🧠 模型"}),
                "seed": ("INT", {"default": 12345, "min": 10000, "max": 99999, "display_name": "🎲 随机种子"}),
                "watermark": ("STRING", {"default": "", "display_name": "🏷️ 水印"}),
                "call_back_url": ("STRING", {"default": "", "display_name": "🔔 回调地址"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "📄 response_json", "🆔 task_id")
    FUNCTION = "extend"
    CATEGORY = "🤖MINGWEI-API/MW-VEO/kie-veo"

    def extend(
        self,
        task_id,
        prompt,
        model,
        seed,
        watermark,
        call_back_url,
        insecure_ssl,
        api_key,
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议用环境变量 KIE_API_KEY）")

        original_task_id = (task_id or "").strip()
        if not original_task_id:
            raise ValueError("扩展视频需要填写原 Veo3.1 生成任务的 task_id")

        prompt_text = (prompt or "").strip()
        if not prompt_text:
            raise ValueError("扩展提示词不能为空，建议使用英文描述延续动作")

        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 12345
        if seed_value < 10000:
            seed_value = 10000
        if seed_value > 99999:
            seed_value = 99999

        request_body = {
            "taskId": original_task_id,
            "prompt": prompt_text,
            "seeds": seed_value,
            "model": (model or "fast").strip() or "fast",
        }
        if _is_nonempty_string(watermark):
            request_body["watermark"] = watermark.strip()
        if _is_nonempty_string(call_back_url):
            request_body["callBackUrl"] = call_back_url.strip()

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(10)

        result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/veo/extend",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(35)

        result_code = (result or {}).get("code")
        if result_code is not None and int(result_code) != 200:
            raise ValueError("Veo 3.1 扩展提交失败: {}".format((result or {}).get("msg") or result))

        extend_task_id = self._extract_task_id(result)
        if not extend_task_id:
            raise ValueError("Veo 3.1 扩展提交成功但未返回 taskId: {}".format(result))

        task_data = self._poll_veo_record(
            api_key=resolved_api_key,
            task_id=extend_task_id,
            poll_interval_s=6.0,
            max_wait_s=1800.0,
            insecure_ssl=bool(insecure_ssl),
        )
        video_url = self._extract_veo_video_url_from_record(task_data)
        pbar.update_absolute(85)

        if not video_url:
            raise ValueError("Veo 3.1 扩展任务未返回视频地址: {}".format(task_data))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "request": request_body,
                "submit_response": result,
                "original_task_id": original_task_id,
                "taskId": extend_task_id,
                "task_data": task_data,
                "video_url": video_url,
                "video_path": video_path,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json, extend_task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json, extend_task_id)


class KuaiVeo3VideoKie:
    MODELS = [
        "veo_3_1_components_vip",
        "veo_3_1_fast_components_vip",
        "veo_3_1_fast_vip",
        "veo_3_1_lite_vip",
        "veo_3_1_vip",
    ]
    TEMP_IMAGE_UPLOAD_URL = "https://imageproxy.zhongzhuan.chat/api/upload"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generationType": (["TEXT_2_VIDEO", "FIRST_AND_LAST_FRAMES_2_VIDEO", "REFERENCE_2_VIDEO"], {"default": "TEXT_2_VIDEO", "display_name": "🎬 生成类型"}),
                "model": (cls.MODELS, {"default": "veo_3_1_fast_vip", "display_name": "🧠 模型"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9", "display_name": "🖼️ 比例"}),
                "duration": (["8s"], {"default": "8s", "display_name": "⏱️ 秒数"}),
                "enhance_prompt": ("BOOLEAN", {"default": True, "display_name": "🌍 启用翻译"}),
                "enable_upsample": ("BOOLEAN", {"default": True, "display_name": "📺 启用超分"}),
                "veo_fl_close": ("BOOLEAN", {"default": False, "display_name": "🧩 关闭自动首尾帧"}),
                "base_url": ("STRING", {"default": "https://api.kuai.host", "display_name": "🌐 BaseURL"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 跳过SSL验证"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API密钥"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL1"}),
                "image_2": ("IMAGE",),
                "image_url_2": ("STRING", {"default": "", "display_name": "🔗 图片URL2"}),
                "image_3": ("IMAGE",),
                "image_url_3": ("STRING", {"default": "", "display_name": "🔗 图片URL3"}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "📄 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-VEO/kuai-veo"

    def _resolve_kuai_api_key(self, widget_value):
        for key in ("KUAI_API_KEY", "KUAI_HOST_API_KEY", "MW_KUAI_API_KEY"):
            value = os.environ.get(key)
            if _is_nonempty_string(value):
                return value.strip()
        return (widget_value or "").strip()

    def _headers(self, api_key):
        return {
            "Authorization": "Bearer {}".format(api_key),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _api_url(self, base_url, endpoint):
        clean_base = (base_url or "https://api.kuai.host").strip().rstrip("/")
        if not clean_base:
            clean_base = "https://api.kuai.host"
        return clean_base + endpoint

    def _find_http_url(self, value):
        if isinstance(value, dict):
            for key in ("url", "URL", "image_url", "imageUrl", "download_url", "downloadUrl", "link", "src"):
                if key in value:
                    found = self._find_http_url(value.get(key))
                    if found:
                        return found
            for item in value.values():
                found = self._find_http_url(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_http_url(item)
                if found:
                    return found
        else:
            import re

            match = re.search(r"https?://[^\s\"'<>]+", str(value or ""))
            if match:
                return match.group(0).rstrip(",.;)]}")
        return ""

    def _upload_image_to_temp_url(self, image_tensor, insecure_ssl=False):
        if _requests is None:
            raise ValueError("当前环境缺少 requests，无法自动上传图片为临时 URL")

        pil_image = tensor2pil(image_tensor)[0]
        if getattr(pil_image, "mode", "") not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")

        buf = BytesIO()
        pil_image.save(buf, format="PNG", optimize=True)
        file_bytes = buf.getvalue()
        file_name = "kuai_veo_{}.png".format(uuid.uuid4().hex[:10])

        def post_file(field_name):
            return _requests.post(
                self.TEMP_IMAGE_UPLOAD_URL,
                files={field_name: (file_name, file_bytes, "image/png")},
                timeout=120,
                verify=(not bool(insecure_ssl)),
            )

        try:
            response = post_file("file")
            if int(getattr(response, "status_code", 0) or 0) >= 400:
                response = post_file("image")
        except Exception as error:
            raise ValueError("上传图片到临时图床失败: {}".format(error))

        text = getattr(response, "text", "")
        if int(getattr(response, "status_code", 0) or 0) >= 400:
            raise ValueError("上传图片到临时图床失败 HTTP {}: {}".format(response.status_code, text))

        try:
            result = response.json()
        except Exception:
            result = text

        image_url = self._find_http_url(result)
        if not image_url:
            raise ValueError("上传图片成功但未返回图片 URL: {}".format(result))
        return image_url

    def _collect_image_urls(
        self,
        generation_type,
        insecure_ssl,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        urls = []
        for url_value, image_value in (
            (image_url, image),
            (image_url_2, image_2),
            (image_url_3, image_3),
        ):
            clean_url = (url_value or "").strip()
            if clean_url:
                urls.append(clean_url)
            elif image_value is not None:
                urls.append(self._upload_image_to_temp_url(image_value, insecure_ssl=bool(insecure_ssl)))

        if generation_type == "TEXT_2_VIDEO":
            return []
        if generation_type == "FIRST_AND_LAST_FRAMES_2_VIDEO":
            if not urls:
                raise ValueError("图生视频模式需要至少提供 1 张图片或图片URL")
            return urls[:2]
        if generation_type == "REFERENCE_2_VIDEO":
            if not urls:
                raise ValueError("三图参考模式需要至少提供 1 张参考图或图片URL")
            return urls[:3]
        return []

    def _extract_task_id(self, result):
        if not isinstance(result, dict):
            return ""
        for key in ("id", "task_id", "taskId"):
            value = result.get(key)
            if _is_nonempty_string(value):
                return value.strip()
        data = result.get("data")
        if isinstance(data, dict):
            for key in ("id", "task_id", "taskId"):
                value = data.get(key)
                if _is_nonempty_string(value):
                    return value.strip()
        return ""

    def _extract_video_url(self, result):
        if not isinstance(result, dict):
            return ""
        candidates = []
        for container in (result, result.get("data"), result.get("detail")):
            if not isinstance(container, dict):
                continue
            for key in ("upsample_video_url", "video_url", "videoUrl", "url", "result_url", "resultUrl"):
                value = container.get(key)
                if _is_nonempty_string(value):
                    candidates.append(value.strip())
        for value in candidates:
            if value.startswith("http"):
                return value
        return ""

    def _poll_result(self, base_url, api_key, task_id, insecure_ssl=False, poll_interval_s=5.0, max_wait_s=1800.0):
        started = time.time()
        while True:
            result = _http_json(
                "GET",
                self._api_url(base_url, "/v1/video/query"),
                headers=self._headers(api_key),
                params={"id": task_id},
                timeout=60,
                insecure_ssl=bool(insecure_ssl),
            )
            data = result.get("data") if isinstance(result, dict) else None
            detail = result.get("detail") if isinstance(result, dict) else None
            status = ""
            for container in (result, data, detail):
                if isinstance(container, dict) and _is_nonempty_string(container.get("status")):
                    status = container.get("status").strip().lower()
                    break

            if status in ("completed", "success", "succeeded", "finished"):
                return result
            if status in ("failed", "fail", "error", "cancelled", "canceled"):
                raise ValueError("kuai-veo3 任务失败: {}".format(result))

            video_url = self._extract_video_url(result)
            if video_url and not status:
                return result

            if time.time() - started >= max_wait_s:
                raise TimeoutError("kuai-veo3 任务超时未完成: {}".format(task_id))
            time.sleep(poll_interval_s)

    def generate(
        self,
        generationType,
        model,
        prompt,
        aspect_ratio,
        duration,
        enhance_prompt,
        enable_upsample,
        veo_fl_close,
        base_url,
        insecure_ssl,
        api_key,
        image=None,
        image_url="",
        image_2=None,
        image_url_2="",
        image_3=None,
        image_url_3="",
    ):
        resolved_api_key = self._resolve_kuai_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 Kuai API 密钥，请在节点内填写 api_key")

        prompt_text = (prompt or "").strip()
        if not prompt_text:
            raise ValueError("提示词不能为空")

        generation_type = (generationType or "TEXT_2_VIDEO").strip() or "TEXT_2_VIDEO"
        image_urls = self._collect_image_urls(
            generation_type=generation_type,
            insecure_ssl=insecure_ssl,
            image=image,
            image_url=image_url,
            image_2=image_2,
            image_url_2=image_url_2,
            image_3=image_3,
            image_url_3=image_url_3,
        )

        request_body = {
            "model": (model or "veo_3_1_fast_vip").strip() or "veo_3_1_fast_vip",
            "prompt": prompt_text,
            "aspect_ratio": (aspect_ratio or "16:9").strip() or "16:9",
            "enhance_prompt": bool(enhance_prompt),
            "enable_upsample": bool(enable_upsample),
        }
        if image_urls:
            request_body["images"] = image_urls
        if generation_type == "REFERENCE_2_VIDEO":
            request_body["veo_fl_close"] = True

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(10)

        submit_result = _http_json(
            "POST",
            self._api_url(base_url, "/v1/video/create"),
            headers=self._headers(resolved_api_key),
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(35)

        task_id = self._extract_task_id(submit_result)
        if not task_id:
            raise ValueError("kuai-veo3 提交成功但未返回任务ID: {}".format(submit_result))

        task_data = self._poll_result(
            base_url=base_url,
            api_key=resolved_api_key,
            task_id=task_id,
            poll_interval_s=5.0,
            max_wait_s=1800.0,
            insecure_ssl=bool(insecure_ssl),
        )
        video_url = self._extract_video_url(task_data)
        pbar.update_absolute(85)

        if not video_url:
            raise ValueError("kuai-veo3 任务未返回视频URL: {}".format(task_data))

        video_path = _download_video_to_temp(video_url, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "request": request_body,
                "submit_response": submit_result,
                "taskId": task_id,
                "task_data": task_data,
                "video_url": video_url,
                "video_path": video_path,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url, response_json, task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url, response_json, task_id)


class MWGeminiOmniVideoKie:
    MODELS = ["gemini-omni-video"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "model": (cls.MODELS, {"default": "gemini-omni-video", "display_name": "🧠 模型"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9", "display_name": "🖼️ 视频比例"}),
                "duration": (["4s", "6s", "8s", "10s"], {"default": "8s", "display_name": "⏱️ 秒数"}),
                "resolution": (["720p", "1080p", "4K"], {"default": "720p", "display_name": "📺 分辨率"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True, "display_name": "🎲 种子"}),
                "image_url": ("STRING", {"multiline": True, "default": "", "display_name": "🔗 图片URL"}),
                "audio_id_1": ("STRING", {"default": "", "display_name": "🎧 audio_ids 1"}),
                "audio_id_2": ("STRING", {"default": "", "display_name": "🎧 audio_ids 2"}),
                "audio_id_3": ("STRING", {"default": "", "display_name": "🎧 audio_ids 3"}),
                "video_url": ("STRING", {"default": "", "display_name": "🔗 视频URL"}),
                "video_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.1, "display_name": "▶️ 视频开始秒"}),
                "video_ends": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 99999.0, "step": 0.1, "display_name": "⏹️ 视频结束秒"}),
                "character_id_1": ("STRING", {"default": "", "display_name": "🆔 character_ids 1"}),
                "character_id_2": ("STRING", {"default": "", "display_name": "🆔 character_ids 2"}),
                "character_id_3": ("STRING", {"default": "", "display_name": "🆔 character_ids 3"}),
                "call_back_url": ("STRING", {"default": "", "display_name": "🔔 回调地址"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "📄 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-gemini-omni"

    def _duration_value(self, duration):
        value = str(duration or "8s").strip().lower().replace("秒", "").replace("s", "")
        if value not in ("4", "6", "8", "10"):
            raise ValueError("Gemini Omni Video 秒数只支持 4s、6s、8s、10s")
        return value

    def _aspect_ratio_value(self, aspect_ratio):
        value = (aspect_ratio or "16:9").strip() or "16:9"
        if value not in ("16:9", "9:16"):
            raise ValueError("Gemini Omni Video 视频比例只支持 16:9、9:16")
        return value

    def _resolution_value(self, resolution):
        value = (resolution or "720p").strip() or "720p"
        if value not in ("720p", "1080p", "4K"):
            raise ValueError("Gemini Omni Video 分辨率只支持 720p、1080p、4K")
        return value

    def _seed_value(self, seed):
        try:
            value = int(seed)
        except Exception:
            value = 0
        if value < 0:
            value = 0
        if value > 2147483647:
            value = 2147483647
        return value

    def _clean_number(self, value):
        n = float(value)
        if n.is_integer():
            return int(n)
        return n

    def _collect_image_urls(
        self,
        resolved_api_key,
        insecure_ssl,
        image_url="",
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
    ):
        urls = _split_text_items(image_url)
        for image_tensor in (image, image_2, image_3, image_4, image_5, image_6, image_7):
            if image_tensor is not None:
                urls.append(_kie_file_base64_upload(image_tensor, resolved_api_key, insecure_ssl=bool(insecure_ssl)))
        return urls

    def _build_video_list(self, video_url, video_start, video_ends):
        clean_url = (video_url or "").strip()
        if not clean_url:
            return []

        start_value = self._clean_number(video_start)
        ends_value = self._clean_number(video_ends)
        if ends_value <= start_value:
            raise ValueError("Gemini Omni Video 的 video_ends 必须大于 video_start")
        return [{"url": clean_url, "start": start_value, "ends": ends_value}]

    def _check_quota(self, image_urls, video_list, character_ids):
        if len(video_list) > 1:
            raise ValueError("Gemini Omni Video 每次最多只能传 1 个视频")
        if len(character_ids) > 3:
            raise ValueError("Gemini Omni Video 每次最多只能传 3 个 character_ids")

        units = len(image_urls) + len(video_list) * 2 + len(character_ids)
        if units > 7:
            raise ValueError("Gemini Omni Video 输入资源超过 7 单位限制：图片数 + 视频数×2 + character_ids 数 <= 7")

    def _extract_task_id(self, submit_result):
        data = (submit_result or {}).get("data") or {}
        task_id = data.get("taskId") if isinstance(data, dict) else ""
        if _is_nonempty_string(task_id):
            return task_id.strip()
        return ""

    def _poll_record(self, api_key, task_id, insecure_ssl=False, poll_interval_s=5.0, max_wait_s=1800.0):
        started = time.time()
        while True:
            record_result = _http_json(
                "GET",
                "https://api.kie.ai/api/v1/jobs/recordInfo",
                headers={"Authorization": "Bearer {}".format(api_key)},
                params={"taskId": task_id},
                timeout=60,
                insecure_ssl=bool(insecure_ssl),
            )
            code = (record_result or {}).get("code")
            if code is not None and int(code) != 200:
                raise ValueError("Gemini Omni Video 查询失败: {}".format((record_result or {}).get("msg") or record_result))

            data = (record_result or {}).get("data") or {}
            response = data.get("response") if isinstance(data, dict) else {}
            if not isinstance(response, dict):
                response = {}

            state = "{}".format(data.get("state") or data.get("status") or "").strip().lower() if isinstance(data, dict) else ""
            success_flag = response.get("successFlag")
            if success_flag is None and isinstance(data, dict):
                success_flag = data.get("successFlag")

            if state in ("success", "succeeded", "completed") or success_flag in (1, "1"):
                return record_result

            if state in ("fail", "failed", "error", "canceled", "cancelled") or success_flag in (2, "2", 3, "3"):
                fail_msg = response.get("errorMessage") or data.get("failMsg") or data.get("errorMessage") or (record_result or {}).get("msg") or ""
                raise ValueError("Gemini Omni Video 任务失败: {}".format(fail_msg or data))

            if time.time() - started >= max_wait_s:
                raise TimeoutError("Gemini Omni Video 任务超时未完成: {}".format(task_id))

            time.sleep(poll_interval_s)

    def _extract_video_url_from_record(self, record_result):
        candidates = []

        def add_url(value):
            if _is_nonempty_string(value):
                candidates.append(value.strip())

        def collect(value):
            if isinstance(value, dict):
                result_json = value.get("resultJson")
                if _is_nonempty_string(result_json):
                    try:
                        collect(json.loads(result_json))
                    except Exception:
                        pass

                for key in ("resultUrls", "fullResultUrls", "originUrls", "videoUrls"):
                    urls = value.get(key)
                    if isinstance(urls, list):
                        for item in urls:
                            collect(item)
                    else:
                        add_url(urls)

                for key in ("resultUrl", "videoUrl", "video_url", "url"):
                    add_url(value.get(key))

                for key in ("data", "response", "result", "output"):
                    child = value.get(key)
                    if child is not None:
                        collect(child)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            else:
                add_url(value)

        collect(record_result)
        for url in candidates:
            if url.startswith("http"):
                return url
        return ""

    def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        duration,
        resolution,
        seed,
        image_url,
        audio_id_1,
        audio_id_2,
        audio_id_3,
        video_url,
        video_start,
        video_ends,
        character_id_1,
        character_id_2,
        character_id_3,
        call_back_url,
        insecure_ssl,
        api_key,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议使用环境变量 KIE_API_KEY）")

        prompt_text = (prompt or "").strip()
        if not prompt_text:
            raise ValueError("Gemini Omni Video 提示词不能为空")

        model_value = (model or "gemini-omni-video").strip() or "gemini-omni-video"
        if model_value not in self.MODELS:
            raise ValueError("Gemini Omni Video 当前只支持模型 gemini-omni-video")

        aspect_ratio_value = self._aspect_ratio_value(aspect_ratio)
        duration_value = self._duration_value(duration)
        resolution_value = self._resolution_value(resolution)
        seed_value = self._seed_value(seed)
        collected_image_urls = self._collect_image_urls(
            resolved_api_key=resolved_api_key,
            insecure_ssl=insecure_ssl,
            image_url=image_url,
            image=image,
            image_2=image_2,
            image_3=image_3,
            image_4=image_4,
            image_5=image_5,
            image_6=image_6,
            image_7=image_7,
        )
        collected_audio_ids = [v.strip() for v in (audio_id_1, audio_id_2, audio_id_3) if _is_nonempty_string(v)]
        collected_video_list = self._build_video_list(video_url, video_start, video_ends)
        collected_character_ids = [v.strip() for v in (character_id_1, character_id_2, character_id_3) if _is_nonempty_string(v)]
        self._check_quota(collected_image_urls, collected_video_list, collected_character_ids)

        input_payload = {
            "prompt": prompt_text,
            "aspect_ratio": aspect_ratio_value,
            "duration": duration_value,
            "resolution": resolution_value,
        }
        if collected_image_urls:
            input_payload["image_urls"] = collected_image_urls
        if collected_audio_ids:
            input_payload["audio_ids"] = collected_audio_ids
        if collected_video_list:
            input_payload["video_list"] = collected_video_list
        if collected_character_ids:
            input_payload["character_ids"] = collected_character_ids

        request_body = {
            "model": model_value,
            "input": input_payload,
        }
        if _is_nonempty_string(call_back_url):
            request_body["callBackUrl"] = call_back_url.strip()

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(10)

        submit_result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/jobs/createTask",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(35)

        result_code = (submit_result or {}).get("code")
        if result_code is not None and int(result_code) != 200:
            raise ValueError("Gemini Omni Video 提交失败: {}".format((submit_result or {}).get("msg") or submit_result))

        task_id = self._extract_task_id(submit_result)
        if not task_id:
            raise ValueError("Gemini Omni Video 提交成功但未返回 taskId: {}".format(submit_result))

        record_result = self._poll_record(
            api_key=resolved_api_key,
            task_id=task_id,
            poll_interval_s=5.0,
            max_wait_s=1800.0,
            insecure_ssl=bool(insecure_ssl),
        )
        video_url_result = self._extract_video_url_from_record(record_result)
        pbar.update_absolute(85)

        if not video_url_result:
            raise ValueError("Gemini Omni Video 任务未返回视频地址: {}".format(record_result))

        video_path = _download_video_to_temp(video_url_result, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "request": request_body,
                "submit_response": submit_result,
                "taskId": task_id,
                "task_data": record_result,
                "video_url": video_url_result,
                "video_path": video_path,
                "local_seed": seed_value,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url_result, response_json, task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url_result, response_json, task_id)


class MWGrokImagineVideoKie:
    MODELS = ["grok-imagine-video-1-5-preview"]
    ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
    RESOLUTIONS = ["480p", "720p"]
    DURATIONS = ["{}s".format(i) for i in range(1, 16)]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "display_name": "📝 提示词"}),
                "model": (cls.MODELS, {"default": "grok-imagine-video-1-5-preview", "display_name": "🧠 模型"}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "auto", "display_name": "🖼️ 比例"}),
                "resolution": (cls.RESOLUTIONS, {"default": "480p", "display_name": "📺 分辨率"}),
                "duration": (cls.DURATIONS, {"default": "8s", "display_name": "⏱️ 秒数"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "display_name": "🎲 种子"}),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL"}),
                "call_back_url": ("STRING", {"default": "", "display_name": "🔔 回调地址"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎞️ video", "🔗 video_url", "🧾 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-grok-1.5"

    def _duration_value(self, duration):
        value = str(duration or "8s").strip().lower().replace("秒", "").replace("s", "")
        try:
            seconds = int(value)
        except Exception:
            raise ValueError("Grok Imagine Video 秒数必须是 1s 到 15s")
        if seconds < 1 or seconds > 15:
            raise ValueError("Grok Imagine Video 秒数必须是 1s 到 15s")
        return seconds

    def _aspect_ratio_value(self, aspect_ratio):
        value = (aspect_ratio or "auto").strip() or "auto"
        if value not in self.ASPECT_RATIOS:
            raise ValueError("Grok Imagine Video 比例只支持: {}".format(", ".join(self.ASPECT_RATIOS)))
        return value

    def _resolution_value(self, resolution):
        value = (resolution or "480p").strip() or "480p"
        if value not in self.RESOLUTIONS:
            raise ValueError("Grok Imagine Video 分辨率只支持: {}".format(", ".join(self.RESOLUTIONS)))
        return value

    def _collect_image_urls(self, resolved_api_key, insecure_ssl, image_url="", image=None):
        urls = _split_text_items(image_url)
        if len(urls) > 1:
            raise ValueError("Grok Imagine Video 每次最多只能传 1 张图片")
        if urls:
            return urls
        if image is not None:
            return [_kie_file_base64_upload(image, resolved_api_key, insecure_ssl=bool(insecure_ssl))]
        return []

    def _extract_task_id(self, submit_result):
        data = (submit_result or {}).get("data") or {}
        task_id = data.get("taskId") if isinstance(data, dict) else ""
        if _is_nonempty_string(task_id):
            return task_id.strip()
        return ""

    def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        resolution,
        duration,
        seed,
        image_url,
        call_back_url,
        insecure_ssl,
        api_key,
        image=None,
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议使用环境变量 KIE_API_KEY）")

        prompt_text = (prompt or "").strip()
        if not prompt_text:
            raise ValueError("Grok Imagine Video 提示词不能为空")

        model_value = (model or "grok-imagine-video-1-5-preview").strip() or "grok-imagine-video-1-5-preview"
        if model_value not in self.MODELS:
            raise ValueError("Grok Imagine Video 当前只支持模型 grok-imagine-video-1-5-preview")

        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        input_payload = {
            "prompt": prompt_text,
            "aspect_ratio": self._aspect_ratio_value(aspect_ratio),
            "resolution": self._resolution_value(resolution),
            "duration": self._duration_value(duration),
        }

        collected_image_urls = self._collect_image_urls(
            resolved_api_key=resolved_api_key,
            insecure_ssl=insecure_ssl,
            image_url=image_url,
            image=image,
        )
        if collected_image_urls:
            input_payload["image_urls"] = collected_image_urls

        request_body = {
            "model": model_value,
            "input": input_payload,
        }
        if _is_nonempty_string(call_back_url):
            request_body["callBackUrl"] = call_back_url.strip()

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyPbar:
                def update_absolute(self, _v: int):
                    return None

            pbar = _DummyPbar()
        pbar.update_absolute(10)

        submit_result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/jobs/createTask",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        pbar.update_absolute(35)

        result_code = (submit_result or {}).get("code")
        if result_code is not None and "{}".format(result_code) != "200":
            raise ValueError("Grok Imagine Video 提交失败: {}".format((submit_result or {}).get("msg") or submit_result))

        task_id = self._extract_task_id(submit_result)
        if not task_id:
            raise ValueError("Grok Imagine Video 提交成功但未返回 taskId: {}".format(submit_result))

        task_data = _kie_poll_result(
            api_key=resolved_api_key,
            task_id=task_id,
            poll_interval_s=5.0,
            max_wait_s=1800.0,
            insecure_ssl=bool(insecure_ssl),
        )
        video_url_result = _extract_video_url(task_data)
        pbar.update_absolute(85)

        if not video_url_result:
            raise ValueError("Grok Imagine Video 任务未返回视频地址: {}".format(task_data))

        video_path = _download_video_to_temp(video_url_result, insecure_ssl=bool(insecure_ssl))
        pbar.update_absolute(100)

        response_json = json.dumps(
            {
                "request": request_body,
                "submit_response": submit_result,
                "taskId": task_id,
                "task_data": task_data,
                "video_url": video_url_result,
                "video_path": video_path,
                "local_seed": seed_value,
            },
            ensure_ascii=False,
        )

        try:
            from comfy_api.input_impl import VideoFromFile

            return (VideoFromFile(video_path), video_url_result, response_json, task_id)
        except Exception:
            return (_LocalOrUrlVideo(video_path), video_url_result, response_json, task_id)


class MWGeminiOmniVideoToUrlKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO,),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("🔗 video_url", "📄 response_json")
    FUNCTION = "upload"
    CATEGORY = "🤖MINGWEI-API/MW-gemini-omni"

    def _save_video_to_temp(self, video):
        if video is None:
            raise ValueError("请连接 VIDEO 类型输入")

        if isinstance(video, _string_types):
            path = video.strip().replace('"', "").replace("'", "")
            if path.startswith(("http://", "https://")):
                return "", path, False
            if os.path.exists(path):
                return path, "", False

        out_dir = os.path.join(folder_paths.get_temp_directory(), "kie_gemini_omni_upload")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        temp_path = os.path.join(out_dir, "gemini_omni_video_{}.mp4".format(uuid.uuid4().hex[:10]))

        if hasattr(video, "save_to"):
            video.save_to(temp_path)
            if os.path.exists(temp_path):
                return temp_path, "", True

        raise ValueError("不支持的视频输入格式，请连接 VIDEO 类型节点")

    def upload(self, video, insecure_ssl, api_key):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议使用环境变量 KIE_API_KEY）")

        temp_path = ""
        cleanup_temp = False
        try:
            temp_path, direct_url, cleanup_temp = self._save_video_to_temp(video)
            if direct_url:
                response_json = json.dumps({"video_url": direct_url, "source": "direct_url"}, ensure_ascii=False)
                return (direct_url, response_json)

            file_name = "gemini_omni_video_{}.mp4".format(uuid.uuid4().hex[:10])
            video_url = _kie_local_file_upload(
                temp_path,
                resolved_api_key,
                upload_path="videos/user-uploads",
                file_name=file_name,
                insecure_ssl=bool(insecure_ssl),
            )
        finally:
            if cleanup_temp and temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        response_json = json.dumps(
            {
                "video_url": video_url,
                "upload_path": "videos/user-uploads",
                "file_name": file_name,
            },
            ensure_ascii=False,
        )
        return (video_url, response_json)


class MWGeminiOmniAudioKie:
    AUDIO_ID_OPTIONS = [
        "achernar - female, soft, high pitch",
        "achird - male, friendly, mid pitch",
        "algenib - male, raspy, low pitch",
        "algieba - male, easygoing, mid-low pitch",
        "alnilam - male, steady, mid-low pitch",
        "aoede - female, brisk, mid pitch",
        "autonoe - female, bright, mid pitch",
        "callirrhoe - female, easygoing, mid pitch",
        "charon - male, intellectual, low pitch",
        "despina - female, smooth, mid pitch",
        "enceladus - male, breathy, low pitch",
        "erinome - female, clear, mid pitch",
        "fenrir - male, lively, younger pitch",
        "gacrux - female, mature, mid pitch",
        "iapetus - male, clear, mid-low pitch",
        "kore - female, capable, mid pitch",
        "laomedeia - female, cheerful, mid-high pitch",
        "leda - female, young, mid-high pitch",
        "orus - male, steady, mid-low pitch",
        "puck - male, cheerful, mid pitch",
        "pulcherrima - genderless, forward, mid-high pitch",
        "rasalgethi - male, intellectual, mid pitch",
        "sadachbia - male, vivid, low pitch",
        "sadaltager - male, knowledgeable, mid pitch",
        "schedar - male, smooth, mid-low pitch",
        "sulafat - female, warm, mid pitch",
        "umbriel - male, smooth, low pitch",
        "vindemiatrix - female, gentle, mid pitch",
        "zephyr - female, bright, mid-high pitch",
        "zubenelgenubi - male, casual, mid-low pitch",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_id": (cls.AUDIO_ID_OPTIONS, {"default": cls.AUDIO_ID_OPTIONS[0], "display_name": "🎧 audio_id"}),
                "name": ("STRING", {"default": "", "display_name": "🏷️ 名称"}),
                "voice_description": ("STRING", {"multiline": True, "default": "", "display_name": "📝 声音描述"}),
                "example_dialogue": ("STRING", {"multiline": True, "default": "", "display_name": "💬 示例对白"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🎧 kie_audio_id", "🏷️ name", "📄 response_json")
    FUNCTION = "create_audio"
    CATEGORY = "🤖MINGWEI-API/MW-gemini-omni"

    def _audio_id_value(self, audio_id):
        return (audio_id or "").split(" - ", 1)[0].strip()

    def create_audio(self, audio_id, name, voice_description, example_dialogue, insecure_ssl, api_key):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议使用环境变量 KIE_API_KEY）")

        clean_audio_id = self._audio_id_value(audio_id)
        clean_name = (name or "").strip()
        clean_voice_description = (voice_description or "").strip()
        clean_example_dialogue = (example_dialogue or "").strip()
        if not clean_audio_id:
            raise ValueError("Gemini Omni Audio 需要填写 audio_id")
        if not clean_name:
            raise ValueError("Gemini Omni Audio 需要填写名称")
        if not clean_voice_description:
            raise ValueError("Gemini Omni Audio 需要填写声音描述")
        if not clean_example_dialogue:
            raise ValueError("Gemini Omni Audio 需要填写示例对白")

        request_body = {
            "audio_id": clean_audio_id,
            "name": clean_name,
            "voice_description": clean_voice_description,
            "example_dialogue": clean_example_dialogue,
        }

        result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/omni/audio/create",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        result_code = (result or {}).get("code")
        if result_code is not None and int(result_code) not in (0, 200):
            raise ValueError("Gemini Omni Audio 提交失败: {}".format((result or {}).get("msg") or result))

        data = (result or {}).get("data") or {}
        audio_id_value = ""
        if isinstance(data, dict):
            audio_id_value = data.get("audioId") or data.get("kieAudioId") or ""
        if not _is_nonempty_string(audio_id_value):
            raise ValueError("Gemini Omni Audio 未返回 audioId: {}".format(result))

        response_json = json.dumps(
            {
                "request": request_body,
                "response": result,
                "audioId": audio_id_value.strip(),
            },
            ensure_ascii=False,
        )
        return (audio_id_value.strip(), clean_name, response_json)


class MWGeminiOmniCharacterKie:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {"default": "", "display_name": "🏷️ 角色名称"}),
                "description": ("STRING", {"multiline": True, "default": "", "display_name": "📝 角色描述"}),
                "image_url": ("STRING", {"default": "", "display_name": "🔗 图片URL"}),
                "insecure_ssl": ("BOOLEAN", {"default": False, "display_name": "🔒 insecure_ssl"}),
                "api_key": ("STRING", {"default": "", "display_name": "🔑 API Key"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio_id_1": ("STRING", {"default": "", "display_name": "🎧 audio_id 1"}),
                "audio_id_2": ("STRING", {"default": "", "display_name": "🎧 audio_id 2"}),
                "audio_id_3": ("STRING", {"default": "", "display_name": "🎧 audio_id 3"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 character_id", "🏷️ character_name", "🔗 image_url", "📄 response_json")
    FUNCTION = "create_character"
    CATEGORY = "🤖MINGWEI-API/MW-gemini-omni"

    def _image_url_value(self, resolved_api_key, insecure_ssl, image_url="", image=None):
        clean_image_url = (image_url or "").strip()
        if clean_image_url:
            return clean_image_url
        if image is not None:
            return _kie_file_base64_upload(image, resolved_api_key, insecure_ssl=bool(insecure_ssl))
        return ""

    def create_character(
        self,
        character_name,
        description,
        image_url,
        insecure_ssl,
        api_key,
        image=None,
        audio_id_1="",
        audio_id_2="",
        audio_id_3="",
    ):
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API_KEY（建议使用环境变量 KIE_API_KEY）")

        clean_character_name = (character_name or "").strip()
        clean_description = (description or "").strip()
        if not clean_character_name:
            raise ValueError("Gemini Omni Character 需要填写角色名称")
        if not clean_description:
            raise ValueError("Gemini Omni Character 需要填写角色描述")

        clean_image_url = self._image_url_value(
            resolved_api_key=resolved_api_key,
            insecure_ssl=insecure_ssl,
            image_url=image_url,
            image=image,
        )
        if not clean_image_url:
            raise ValueError("Gemini Omni Character 需要填写图片URL或连接 image 输入")

        audio_ids = [v.strip() for v in (audio_id_1, audio_id_2, audio_id_3) if _is_nonempty_string(v)]

        request_body = {
            "description": clean_description,
            "image_urls": [clean_image_url],
            "character_name": clean_character_name,
        }
        if audio_ids:
            request_body["audio_ids"] = audio_ids

        result = _http_json(
            "POST",
            "https://api.kie.ai/api/v1/omni/character/create",
            headers={"Authorization": "Bearer {}".format(resolved_api_key), "Content-Type": "application/json"},
            json_body=request_body,
            timeout=300,
            insecure_ssl=bool(insecure_ssl),
        )
        result_code = (result or {}).get("code")
        if result_code is not None and int(result_code) not in (0, 200):
            raise ValueError("Gemini Omni Character 提交失败: {}".format((result or {}).get("msg") or result))

        data = (result or {}).get("data") or {}
        character_id = data.get("characterId") if isinstance(data, dict) else ""
        if not _is_nonempty_string(character_id):
            raise ValueError("Gemini Omni Character 未返回 characterId: {}".format(result))

        response_json = json.dumps(
            {
                "request": request_body,
                "response": result,
                "characterId": character_id.strip(),
            },
            ensure_ascii=False,
        )
        return (character_id.strip(), clean_character_name, clean_image_url, response_json)


NODE_CLASS_MAPPINGS = {
    "Veo31Kie": Veo31Kie,
    "Veo31ExtendKie": Veo31ExtendKie,
    "KuaiVeo3VideoKie": KuaiVeo3VideoKie,
    "MWGeminiOmniVideoKie": MWGeminiOmniVideoKie,
    "MWGrokImagineVideoKie": MWGrokImagineVideoKie,
    "MWGeminiOmniVideoToUrlKie": MWGeminiOmniVideoToUrlKie,
    "MWGeminiOmniAudioKie": MWGeminiOmniAudioKie,
    "MWGeminiOmniCharacterKie": MWGeminiOmniCharacterKie,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo31Kie": "veo3.1-kie",
    "Veo31ExtendKie": "veo3.1-extend-kie",
    "KuaiVeo3VideoKie": "kuai-veo3视频生成",
    "MWGeminiOmniVideoKie": "MW-gemini-omni-video-kie",
    "MWGrokImagineVideoKie": "MW-grok-imagine-video-kie",
    "MWGeminiOmniVideoToUrlKie": "MW-gemini-omni-video-to-url-kie",
    "MWGeminiOmniAudioKie": "MW-gemini-omni-audio-kie",
    "MWGeminiOmniCharacterKie": "MW-gemini-omni-character-kie",
}


def _unregister_node_keys(node_keys):
    try:
        import nodes
    except Exception:
        nodes = None

    if nodes is not None:
        try:
            class_map = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
            if isinstance(class_map, dict):
                for k in node_keys:
                    class_map.pop(k, None)
        except Exception:
            pass
        try:
            name_map = getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", None)
            if isinstance(name_map, dict):
                for k in node_keys:
                    name_map.pop(k, None)
        except Exception:
            pass

    try:
        import sys

        for mod_name in (
            "custom_nodes.ComfyUI-nkxx.banana_nodes",
            "ComfyUI-nkxx.banana_nodes",
            "banana_nodes",
        ):
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            class_map = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(class_map, dict):
                for k in node_keys:
                    class_map.pop(k, None)
            name_map = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None)
            if isinstance(name_map, dict):
                for k in node_keys:
                    name_map.pop(k, None)
    except Exception:
        pass


def _schedule_unregister_node_keys(node_keys):
    try:
        import threading
        import time
    except Exception:
        return

    def _runner():
        for _ in range(6):
            _unregister_node_keys(node_keys)
            time.sleep(1.0)

    t = threading.Thread(target=_runner)
    t.daemon = True
    t.start()


_schedule_unregister_node_keys(
    (
        "GrsaiNanoBananaBatch",
        "GrsaiNanoBananaBatchKie",
        "GrsaiNanoBananaBatch_kie",
    )
)

