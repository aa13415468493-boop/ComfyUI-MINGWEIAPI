import base64
import json
import os
import time
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4
from typing import List, Optional, Tuple

import torch
from PIL import Image

import comfy.utils


def pil2tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    try:
        import numpy as np  # type: ignore

        img_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]
    except Exception:
        width, height = image.size
        buf = image.tobytes()
        storage = torch.ByteStorage.from_buffer(buf)
        out = torch.ByteTensor(storage).view(height, width, 3).to(torch.float32) / 255.0
        return out.unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out: List[Image.Image] = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    try:
        import numpy as np  # type: ignore

        numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return [Image.fromarray(numpy_image)]
    except Exception:
        t = (image.squeeze() * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        if len(t.shape) != 3 or t.shape[-1] != 3:
            raise ValueError("IMAGE tensor 必须是 [H, W, 3] 或 [1, H, W, 3]")
        height, width = int(t.shape[0]), int(t.shape[1])
        raw = bytes(t.contiguous().view(-1).tolist())
        return [Image.frombytes("RGB", (width, height), raw)]


def _image_to_base64_png(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _download_image(url: str, timeout: int) -> Image.Image:
    try:
        import requests  # type: ignore

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        req = Request(url, headers={"User-Agent": "ComfyUI"})
        with urlopen(req, timeout=timeout) as resp:
            content = resp.read()
        return Image.open(BytesIO(content))


def _http_json(
    method: str,
    url: str,
    headers: dict,
    payload: Optional[dict],
    timeout: int,
    params: Optional[dict] = None,
) -> dict:
    try:
        import requests  # type: ignore

        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        else:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        import ssl

        final_url = url
        if params:
            final_url = f"{final_url}?{urlencode(params)}"
        data: Optional[bytes] = None
        req_headers = dict(headers)
        req_headers.setdefault("User-Agent", "ComfyUI")
        req_headers.setdefault("Connection", "close")
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        req = Request(final_url, method=method.upper(), headers=req_headers, data=data)
        ctx = ssl.create_default_context()
        try:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        except Exception:
            pass

        for i in range(2):
            try:
                with urlopen(req, timeout=timeout, context=ctx) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except Exception as e:
                if i == 0:
                    time.sleep(0.5)
                    continue
                raise


class MWKieBananaProBasic:
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["auto", "1:1", "4:5", "3:2", "16:9", "21:9", "9:16", "4:3", "2:3", "3:4", "5:4"]
        resolutions = ["1K", "2K", "4K"]
        output_formats = ["png", "jpg", "webp"]
        response_formats = ["url", "b64_json"]

        return {
            "required": {
                "🧩 生成模式": (["文生图", "图片编辑"], {"default": "文生图"}),
                "🤖 模型版本": (["nano-banana-pro"], {"default": "nano-banana-pro"}),
                "📝 提示词": ("STRING", {"multiline": True}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
                "🖼️ 分辨率": (resolutions, {"default": "2K"}),
                "🧾 输出格式": (output_formats, {"default": "png"}),
                "📦 返回格式": (response_formats, {"default": "url"}),
                "🖼️ 出图数量": ("INT", {"default": 1, "min": 1, "max": 8}),
                "🎲 种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
                "🖼️ 图像5": ("IMAGE",),
                "🖼️ 图像6": ("IMAGE",),
                "🖼️ 图像7": ("IMAGE",),
                "🖼️ 图像8": ("IMAGE",),
                "🖼️ 图像9": ("IMAGE",),
                "🖼️ 图像10": ("IMAGE",),
                "🖼️ 图像11": ("IMAGE",),
                "🖼️ 图像12": ("IMAGE",),
                "🖼️ 图像13": ("IMAGE",),
                "⏱️ 超时(秒)": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "🔁 轮询间隔(秒)": ("INT", {"default": 3, "min": 1, "max": 30}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "📄 response", "🔗 image_url")
    FUNCTION = "run"
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _upload_image_data_url(self, api_key: str, image_data_url: str, timeout_sec: int) -> str:
        file_name = f"mw_kie_{uuid4().hex}.png"
        upload_path = "comfyui/mw-kie-banana-pro"

        try:
            import requests  # type: ignore

            data_url = (image_data_url or "").strip()
            mime = "image/png"
            b64 = data_url
            if data_url.startswith("data:") and "base64," in data_url:
                head, tail = data_url.split("base64,", 1)
                b64 = tail
                try:
                    mime = head[5:].split(";", 1)[0] or mime
                except Exception:
                    mime = "image/png"

            file_bytes = base64.b64decode(b64)
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{self.upload_base_url}/api/file-stream-upload"

            for i in range(3):
                try:
                    resp = requests.post(
                        url=url,
                        headers=headers,
                        files={"file": (file_name, BytesIO(file_bytes), mime)},
                        data={"uploadPath": upload_path, "fileName": file_name},
                        timeout=min(120, timeout_sec),
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    if result.get("success") is not True or int(result.get("code", 0)) != 200:
                        raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))
                    data = result.get("data") or {}
                    file_url = data.get("fileUrl") or data.get("downloadUrl")
                    if not file_url:
                        raise ValueError(json.dumps(result, ensure_ascii=False))
                    return str(file_url)
                except Exception:
                    if i < 2:
                        time.sleep(0.5 * (i + 1))
                        continue
                    raise
        except Exception:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "base64Data": image_data_url,
                "uploadPath": upload_path,
                "fileName": file_name,
            }
            result = _http_json(
                method="POST",
                url=f"{self.upload_base_url}/api/file-base64-upload",
                headers=headers,
                payload=payload,
                timeout=min(60, timeout_sec),
            )
            if result.get("success") is not True or int(result.get("code", 0)) != 200:
                raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))
            data = result.get("data") or {}
            file_url = data.get("fileUrl") or data.get("downloadUrl")
            if not file_url:
                raise ValueError(json.dumps(result, ensure_ascii=False))
            return str(file_url)

    def _create_task(
        self,
        api_key: str,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_format: str,
        response_format: str,
        seed: int,
        images: List[torch.Tensor],
        timeout_sec: int,
        model: str = "nano-banana-pro",
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        image_input: List[str] = []
        for img in images:
            for pil_image in tensor2pil(img):
                data_url = f"data:image/png;base64,{_image_to_base64_png(pil_image)}"
                image_input.append(self._upload_image_data_url(api_key=api_key, image_data_url=data_url, timeout_sec=timeout_sec))

        input_payload = {
            "prompt": prompt,
            "image_input": image_input,
            "resolution": resolution,
            "output_format": output_format,
        }
        if aspect_ratio != "auto":
            input_payload["aspect_ratio"] = aspect_ratio
        if response_format:
            input_payload["response_format"] = response_format
        if seed > 0:
            input_payload["seed"] = seed

        payload = {
            "model": model,
            "input": input_payload,
        }

        result = _http_json(
            method="POST",
            url=f"{self.base_url}/api/v1/jobs/createTask",
            headers=headers,
            payload=payload,
            timeout=timeout_sec,
        )
        if int(result.get("code", 0)) != 200:
            raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))
        task_id = (result.get("data") or {}).get("taskId")
        if not task_id:
            raise ValueError(json.dumps(result, ensure_ascii=False))
        return task_id

    def _poll_task(
        self,
        api_key: str,
        task_id: str,
        timeout_sec: int,
        poll_interval_sec: int,
    ) -> Tuple[List[str], dict]:
        headers = {"Authorization": f"Bearer {api_key}"}
        start = time.time()
        last_data: dict = {}

        while True:
            elapsed = time.time() - start
            if elapsed >= timeout_sec:
                raise TimeoutError(f"任务超时：{task_id}")

            result = _http_json(
                method="GET",
                url=f"{self.base_url}/api/v1/jobs/recordInfo",
                headers=headers,
                payload=None,
                timeout=min(30, timeout_sec),
                params={"taskId": task_id},
            )
            data = result.get("data") or {}
            last_data = data
            state = (data.get("state") or "").lower()

            if state == "success":
                result_json = data.get("resultJson") or ""
                parsed = json.loads(result_json) if result_json else {}
                urls = parsed.get("resultUrls") or []
                if not isinstance(urls, list):
                    urls = []
                return urls, data

            if state == "fail":
                fail_msg = data.get("failMsg") or result.get("msg") or "任务失败"
                raise ValueError(f"{fail_msg}（taskId={task_id}）")

            time.sleep(poll_interval_sec)

    def _get_credits(self, api_key: str) -> Optional[int]:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            result = _http_json(
                method="GET",
                url=f"{self.base_url}/api/v1/chat/credit",
                headers=headers,
                payload=None,
                timeout=10,
            )
            if int(result.get("code", 0)) != 200:
                return None
            data = result.get("data")
            if data is None:
                return None
            return int(data)
        except Exception:
            return None

    def run(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        生成模式 = _pick("🧩 生成模式", "生成模式", default="文生图")
        模型版本 = _pick("🤖 模型版本", "模型版本", default="nano-banana-pro")
        prompt = _pick("📝 提示词", "prompt", default="")
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        resolution = _pick("🖼️ 分辨率", "resolution", default="2K")
        output_format = _pick("🧾 输出格式", "output_format", default="png")
        response_format = _pick("📦 返回格式", "response_format", default="url")
        image_count = int(_pick("🖼️ 出图数量", "image_count", default=1) or 1)
        seed = int(_pick("🎲 种子", "seed", default=0) or 0)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        timeout_sec = int(_pick("⏱️ 超时(秒)", "timeout_sec", default=600) or 600)
        poll_interval_sec = int(_pick("🔁 轮询间隔(秒)", "poll_interval_sec", default=3) or 3)

        images = []
        for i in range(1, 14):
            images.append(_pick(f"🖼️ 图像{i}", f"image{i}", default=None))
        images = [img for img in images if img is not None]

        key = self._get_api_key(str(api_key))
        if not key:
            raise ValueError("未提供 API Key：请设置环境变量 KIE_API_KEY 或填写节点 api_key")

        if 生成模式 == "文生图":
            images = []
        else:
            if not images:
                raise ValueError("图片编辑模式至少需要输入 1 张图片（image1~image13）")

        used_seed = int(seed) if seed else 0

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)
        task_count = max(1, min(8, image_count))
        task_ids: List[str] = []
        for idx in range(task_count):
            task_id = self._create_task(
                api_key=key,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                response_format=response_format,
                seed=used_seed,
                images=images,
                timeout_sec=timeout_sec,
                model=模型版本,
            )
            task_ids.append(task_id)
            pbar.update_absolute(5 + int(10 * (idx + 1) / task_count))

        outputs: List[torch.Tensor] = []
        task_results: List[dict] = []
        all_result_urls: List[str] = []
        primary_urls: List[str] = []
        base_size: Optional[Tuple[int, int]] = None
        for idx, task_id in enumerate(task_ids):
            urls, task_data = self._poll_task(
                api_key=key,
                task_id=task_id,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
            if not urls:
                raise ValueError(f"任务成功但没有返回图片 URL（taskId={task_id}）")
            main_url = str(urls[0])
            img = _download_image(main_url, timeout=min(60, timeout_sec))
            if base_size is None:
                base_size = img.size
            elif img.size != base_size:
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
                img = img.resize(base_size, resample=resample)
            outputs.append(pil2tensor(img))
            primary_urls.append(main_url)
            all_result_urls.extend([str(u) for u in urls])
            task_results.append(
                {
                    "taskId": task_id,
                    "state": task_data.get("state"),
                    "resultUrls": urls,
                }
            )
            pbar.update_absolute(15 + int(70 * (idx + 1) / task_count))

        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        pbar.update_absolute(100)

        credits = self._get_credits(key)
        response = {
            "taskId": task_ids[0],
            "state": task_results[0].get("state"),
            "taskCount": task_count,
            "taskIds": task_ids,
            "tasks": task_results,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "response_format": response_format,
            "seed": used_seed,
            "resultUrls": all_result_urls,
            "credits": credits,
        }
        image_url_output = primary_urls[0] if len(primary_urls) == 1 else "\n".join(primary_urls)
        return (out, json.dumps(response, ensure_ascii=False, indent=2), image_url_output)


class MWKieBanana2(MWKieBananaProBasic):
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = [
            "auto",
            "1:1",
            "1:4",
            "1:8",
            "2:3",
            "3:2",
            "3:4",
            "4:1",
            "4:3",
            "4:5",
            "5:4",
            "8:1",
            "9:16",
            "16:9",
            "21:9",
        ]
        resolutions = ["512px", "1K", "2K", "4K"]
        output_formats = ["png", "jpg", "webp"]
        response_formats = ["url", "b64_json"]

        return {
            "required": {
                "🧩 生成模式": (["文生图", "图片编辑"], {"default": "文生图"}),
                "🤖 模型版本": (["nano-banana-2"], {"default": "nano-banana-2"}),
                "📝 提示词": ("STRING", {"multiline": True}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
                "🖼️ 分辨率": (resolutions, {"default": "1K"}),
                "🧾 输出格式": (output_formats, {"default": "png"}),
                "📦 返回格式": (response_formats, {"default": "url"}),
                "🔎 谷歌搜索": ("BOOLEAN", {"default": False}),
                "🖼️ 出图数量": ("INT", {"default": 1, "min": 1, "max": 8}),
                "🎲 种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
                "🖼️ 图像5": ("IMAGE",),
                "🖼️ 图像6": ("IMAGE",),
                "🖼️ 图像7": ("IMAGE",),
                "🖼️ 图像8": ("IMAGE",),
                "🖼️ 图像9": ("IMAGE",),
                "🖼️ 图像10": ("IMAGE",),
                "🖼️ 图像11": ("IMAGE",),
                "🖼️ 图像12": ("IMAGE",),
                "🖼️ 图像13": ("IMAGE",),
                "🖼️ 图像14": ("IMAGE",),
                "⏱️ 超时(秒)": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "🔁 轮询间隔(秒)": ("INT", {"default": 3, "min": 1, "max": 30}),
            },
        }

    def _create_task(
        self,
        api_key: str,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_format: str,
        response_format: str,
        google_search: bool,
        seed: int,
        images: List[torch.Tensor],
        timeout_sec: int,
        model: str = "nano-banana-2",
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        image_input: List[str] = []
        for img in images:
            for pil_image in tensor2pil(img):
                data_url = f"data:image/png;base64,{_image_to_base64_png(pil_image)}"
                image_input.append(self._upload_image_data_url(api_key=api_key, image_data_url=data_url, timeout_sec=timeout_sec))

        input_payload = {
            "prompt": prompt,
            "image_input": image_input,
            "resolution": resolution,
            "output_format": output_format,
        }
        if aspect_ratio != "auto":
            input_payload["aspect_ratio"] = aspect_ratio
        if response_format:
            input_payload["response_format"] = response_format
        if bool(google_search):
            input_payload["google_search"] = True
        if seed > 0:
            input_payload["seed"] = seed

        payload = {
            "model": model,
            "input": input_payload,
        }

        result = _http_json(
            method="POST",
            url=f"{self.base_url}/api/v1/jobs/createTask",
            headers=headers,
            payload=payload,
            timeout=timeout_sec,
        )
        if int(result.get("code", 0)) != 200:
            raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))
        task_id = (result.get("data") or {}).get("taskId")
        if not task_id:
            raise ValueError(json.dumps(result, ensure_ascii=False))
        return task_id

    def run(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        生成模式 = _pick("🧩 生成模式", "生成模式", default="文生图")
        模型版本 = _pick("🤖 模型版本", "模型版本", default="nano-banana-2")
        prompt = _pick("📝 提示词", "� 提示词", "prompt", default="")
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        resolution = _pick("🖼️ 分辨率", "resolution", default="1K")
        output_format = _pick("🧾 输出格式", "output_format", default="png")
        response_format = _pick("📦 返回格式", "response_format", default="url")
        google_search = bool(_pick("🔎 谷歌搜索", "google_search", default=False))
        image_count = int(_pick("🖼️ 出图数量", "image_count", default=1) or 1)
        seed = int(_pick("🎲 种子", "seed", default=0) or 0)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        timeout_sec = int(_pick("⏱️ 超时(秒)", "timeout_sec", default=600) or 600)
        poll_interval_sec = int(_pick("🔁 轮询间隔(秒)", "poll_interval_sec", default=3) or 3)

        images = []
        for i in range(1, 15):
            images.append(_pick(f"🖼️ 图像{i}", f"image{i}", default=None))
        images = [img for img in images if img is not None]

        key = self._get_api_key(str(api_key))
        if not key:
            raise ValueError("未提供 API Key：请设置环境变量 KIE_API_KEY 或填写节点 api_key")

        if 生成模式 == "文生图":
            images = []
        else:
            if not images:
                raise ValueError("图片编辑模式至少需要输入 1 张图片（image1~image14）")
        if not str(prompt or "").strip():
            raise ValueError("提示词不能为空")

        used_seed = int(seed) if seed else 0

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)
        task_count = max(1, min(8, image_count))
        task_ids: List[str] = []
        for idx in range(task_count):
            task_id = self._create_task(
                api_key=key,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                response_format=response_format,
                google_search=google_search,
                seed=used_seed,
                images=images,
                timeout_sec=timeout_sec,
                model=模型版本,
            )
            task_ids.append(task_id)
            pbar.update_absolute(5 + int(10 * (idx + 1) / task_count))

        outputs: List[torch.Tensor] = []
        task_results: List[dict] = []
        all_result_urls: List[str] = []
        primary_urls: List[str] = []
        base_size: Optional[Tuple[int, int]] = None
        for idx, task_id in enumerate(task_ids):
            urls, task_data = self._poll_task(
                api_key=key,
                task_id=task_id,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
            if not urls:
                raise ValueError(f"任务成功但没有返回图片 URL（taskId={task_id}）")
            main_url = str(urls[0])
            img = _download_image(main_url, timeout=min(60, timeout_sec))
            if base_size is None:
                base_size = img.size
            elif img.size != base_size:
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
                img = img.resize(base_size, resample=resample)
            outputs.append(pil2tensor(img))
            primary_urls.append(main_url)
            all_result_urls.extend([str(u) for u in urls])
            task_results.append(
                {
                    "taskId": task_id,
                    "state": task_data.get("state"),
                    "resultUrls": urls,
                }
            )
            pbar.update_absolute(15 + int(70 * (idx + 1) / task_count))

        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        pbar.update_absolute(100)

        credits = self._get_credits(key)
        response = {
            "taskId": task_ids[0],
            "state": task_results[0].get("state"),
            "taskCount": task_count,
            "taskIds": task_ids,
            "tasks": task_results,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "response_format": response_format,
            "google_search": bool(google_search),
            "seed": used_seed,
            "resultUrls": all_result_urls,
            "credits": credits,
        }
        image_url_output = primary_urls[0] if len(primary_urls) == 1 else "\n".join(primary_urls)
        return (out, json.dumps(response, ensure_ascii=False, indent=2), image_url_output)


class Gemini3ProImagePreviewZhenzhen:
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["Auto", "1:1", "4:5", "3:2", "16:9", "21:9", "9:16", "4:3", "2:3", "3:4", "5:4"]
        response_modes = ["TEXT_AND_IMAGE", "IMAGE_ONLY", "TEXT_ONLY"]
        output_resolutions = ["Auto (Model Default)", "1K", "2K", "4K"]
        image_presets = ["hd", "standard"]
        style_presets = ["natural", "vivid"]
        zoom_presets = ["1x (不放大)", "2x", "4x"]
        upscale_models = ["High Fidelity", "Fast"]

        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": "请根据这些图片进行专业的图像编辑"}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "填入 t8star API Key（Header: Authorization: Bearer xxx）"}),
                "🌐 API 地址": ("STRING", {"default": "https://ai.t8star.org/v1/chat/completions"}),
                "🤖 模型名称": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "📡 请求方法": (["POST"], {"default": "POST"}),
                "📍 密钥位置": (["Header"], {"default": "Header"}),
                "🏷️ 密钥字段名": ("STRING", {"default": "Authorization"}),
                "🪪 认证方式": (["Bearer", "Raw"], {"default": "Bearer"}),
                "📐 宽高比": (aspect_ratios, {"default": "Auto"}),
                "🧩 响应模式": (response_modes, {"default": "TEXT_AND_IMAGE"}),
                "🖼️ 输出分辨率": (output_resolutions, {"default": "Auto (Model Default)"}),
                "🎨 画质预设": (image_presets, {"default": "hd"}),
                "🖌️ 风格预设": (style_presets, {"default": "natural"}),
                "🔍 放大倍数": (zoom_presets, {"default": "1x (不放大)"}),
                "🧠 放大模型": (upscale_models, {"default": "High Fidelity"}),
                "🧭 响应提取路径": ("STRING", {"default": "", "placeholder": "可选：例如 choices.0.message.content.0.image_url.url"}),
                "⏱️ 超时(秒)": ("INT", {"default": 180, "min": 10, "max": 3600}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "📄 response", "🔗 image_url", "🧾 raw_json")
    FUNCTION = "run"
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("T8STAR_API_KEY", "") or api_key).strip()

    def _build_messages(self, prompt: str, images: List[torch.Tensor]) -> List[dict]:
        content: List[dict] = [{"type": "text", "text": (prompt or "").strip()}]
        for img_t in images:
            pil_list = tensor2pil(img_t)
            if not pil_list:
                continue
            data_url = f"data:image/png;base64,{_image_to_base64_png(pil_list[0])}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        return [{"role": "user", "content": content}]

    def _download_image_authed(self, url: str, timeout: int, headers: dict) -> Image.Image:
        try:
            import requests  # type: ignore

            req_headers = dict(headers)
            req_headers.setdefault("User-Agent", "ComfyUI")
            resp = requests.get(url, headers=req_headers, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except Exception:
            req_headers = dict(headers)
            req_headers.setdefault("User-Agent", "ComfyUI")
            req = Request(url, headers=req_headers)
            with urlopen(req, timeout=timeout) as resp:
                content = resp.read()
            return Image.open(BytesIO(content))

    def _download_image_with_fallbacks(self, url: str, timeout_sec: int, download_headers: dict, api_url: str) -> Optional[Image.Image]:
        timeout = min(60, int(timeout_sec))
        tries: List[dict] = []

        tries.append(dict(download_headers))

        h2 = dict(download_headers)
        h2.pop("Authorization", None)
        tries.append(h2)

        h3 = dict(download_headers)
        h3.setdefault("Referer", (api_url or "https://ai.t8star.org/").strip() or "https://ai.t8star.org/")
        h3.setdefault("Origin", "https://ai.t8star.org")
        tries.append(h3)

        h4 = dict(h3)
        h4.pop("Authorization", None)
        tries.append(h4)

        for h in tries:
            try:
                return self._download_image_authed(url, timeout=timeout, headers=h)
            except Exception:
                continue
        try:
            return _download_image(url, timeout=timeout)
        except Exception:
            return None

    def _get_dimensions(self, aspect_ratio: str, resolution: str) -> Tuple[int, int]:
        if aspect_ratio == "Auto" or not aspect_ratio:
            return 0, 0
        
        # Base size for 1K (approx)
        base = 1024
        if resolution == "2K":
            base = 2048
        elif resolution == "4K":
            base = 4096
        
        # Parse aspect ratio
        try:
            if ":" in aspect_ratio:
                w_r, h_r = map(int, aspect_ratio.split(":"))
            else:
                return 0, 0
        except ValueError:
            return 0, 0

        # Calculate dimensions keeping area roughly base*base
        # w * h = base * base
        # w / h = w_r / h_r  => w = h * (w_r / h_r)
        # h * h * (w_r / h_r) = base * base
        # h = sqrt(base * base * h_r / w_r)
        
        import math
        area = base * base
        h = int(math.sqrt(area * h_r / w_r))
        w = int(h * w_r / h_r)
        
        # Align to 64 (common constraint)
        w = (w // 64) * 64
        h = (h // 64) * 64
        
        return w, h

    def _extract_image_url_like_dapao(self, data: dict) -> str:
        try:
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                url = (data["data"][0] or {}).get("url", "")
                return str(url or "").strip()

            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                content = (data["choices"][0] or {}).get("message", {})
                if isinstance(content, dict):
                    content = content.get("content", "")
                if isinstance(content, str) and content:
                    import re

                    m = re.search(r"!\[.*?\]\((.*?)\)", content)
                    if m:
                        return (m.group(1) or "").strip()
                    m2 = re.search(r"(https?://[^\s\)\]\"\'\<\>]+)", content)
                    if m2:
                        return (m2.group(1) or "").rstrip(').,;:!?\"\']`').strip()

            url = data.get("url", "") or data.get("image_url", "")
            return str(url or "").strip()
        except Exception:
            return ""

    def _download_image_from_url_like_dapao(self, url: str, timeout_sec: int) -> Optional[Image.Image]:
        if not url:
            return None
        timeout = min(60, int(timeout_sec))
        try:
            import requests  # type: ignore

            resp = requests.get(url, timeout=timeout, verify=False, headers={"User-Agent": "ComfyUI"})
            if int(getattr(resp, "status_code", 0) or 0) != 200:
                return None
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            try:
                req = Request(url, headers={"User-Agent": "ComfyUI"})
                with urlopen(req, timeout=timeout) as r:
                    content = r.read()
                return Image.open(BytesIO(content)).convert("RGB")
            except Exception:
                return None

    def _get_by_path(self, obj: object, path: str) -> object:
        cur: object = obj
        for raw in [p for p in (path or "").split(".") if p.strip()]:
            part = raw.strip()
            if isinstance(cur, list):
                idx = int(part)
                cur = cur[idx]
            elif isinstance(cur, dict):
                cur = cur[part]
            else:
                raise KeyError(part)
        return cur

    def _extract_text(self, resp: dict) -> str:
        try:
            choices = resp.get("choices") or []
            if not choices:
                return ""
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                        t = item.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                return "\n".join(parts).strip()
            return ""
        except Exception:
            return ""

    def _extract_image(
        self,
        resp: dict,
        timeout_sec: int,
        download_headers: dict,
        response_extract_path: str,
    ) -> Tuple[Optional[Image.Image], str]:
        import re

        def _try_open_from_url(url: str) -> Optional[Image.Image]:
            try:
                return self._download_image_authed(url, timeout=min(60, timeout_sec), headers=download_headers)
            except Exception:
                try:
                    return _download_image(url, timeout=min(60, timeout_sec))
                except Exception:
                    return None

        def _try_open_from_b64(b64_str: str) -> Optional[Image.Image]:
            try:
                return Image.open(BytesIO(base64.b64decode(b64_str.strip())))
            except Exception:
                return None

        def _try_open_from_data_url(data_url: str) -> Optional[Image.Image]:
            try:
                if "base64," not in data_url:
                    return None
                b64_part = data_url.split("base64,", 1)[1].strip()
                return _try_open_from_b64(b64_part)
            except Exception:
                return None

        def _try_open_from_obj(v: object) -> Tuple[Optional[Image.Image], str]:
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("data:image/"):
                    img = _try_open_from_data_url(s)
                    return img, ""
                if s.startswith("http"):
                    img = _try_open_from_url(s)
                    return img, s
                if len(s) > 64:
                    img = _try_open_from_b64(s)
                    return img, ""
                img, url = _first_image_from_text(s)
                return img, url
            if isinstance(v, dict):
                if "url" in v and isinstance(v.get("url"), str):
                    return _try_open_from_obj(v.get("url"))
                if "b64_json" in v and isinstance(v.get("b64_json"), str):
                    return _try_open_from_obj(v.get("b64_json"))
                if "image_url" in v:
                    return _try_open_from_obj(v.get("image_url"))
            if isinstance(v, list) and v:
                for it in v:
                    img, url = _try_open_from_obj(it)
                    if img is not None:
                        return img, url
            return None, ""

        def _first_image_from_text(text: str) -> Tuple[Optional[Image.Image], str]:
            if not text:
                return None, ""
            m_data = re.search(r"(data:image/[^;]+;base64,[A-Za-z0-9+/=\\s]+)", text, flags=re.IGNORECASE)
            if m_data:
                img = _try_open_from_data_url(m_data.group(1))
                if img is not None:
                    return img, ""
            m_url = re.search(r"https?://\\S+", text, flags=re.IGNORECASE)
            if m_url:
                url = m_url.group(0).rstrip(")]>.,\"'`")
                img = _try_open_from_url(url)
                if img is not None:
                    return img, url
                return None, url
            return None, ""

        def _walk_find(obj: object) -> Tuple[Optional[Image.Image], str]:
            if isinstance(obj, dict):
                first_url: str = ""
                for k, v in obj.items():
                    key = str(k).lower()
                    if isinstance(v, str):
                        s = v.strip()
                        if s.startswith("data:image/"):
                            img = _try_open_from_data_url(s)
                            if img is not None:
                                return img, ""
                        if ("b64" in key or "base64" in key) and len(s) > 64:
                            img = _try_open_from_b64(s)
                            if img is not None:
                                return img, ""
                        if ("url" in key or "image" in key) and s.startswith("http"):
                            img = _try_open_from_url(s)
                            if img is not None:
                                return img, s
                            if not first_url:
                                first_url = s
                        img, url = _first_image_from_text(s)
                        if img is not None:
                            return img, url
                        if url and not first_url:
                            first_url = url
                    img, url = _walk_find(v)
                    if img is not None:
                        return img, url
                    if url and not first_url:
                        first_url = url
                return None, first_url
            if isinstance(obj, list):
                first_url = ""
                for item in obj:
                    img, url = _walk_find(item)
                    if img is not None:
                        return img, url
                    if url and not first_url:
                        first_url = url
                return None, first_url
            if isinstance(obj, str):
                return _first_image_from_text(obj)
            return None, ""

        if (response_extract_path or "").strip():
            try:
                extracted = self._get_by_path(resp, (response_extract_path or "").strip())
                img, url = _try_open_from_obj(extracted)
                if img is not None:
                    return img, url
            except Exception:
                pass

        img, url = _walk_find(resp)
        return img, url

    def run(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        prompt = _pick("📝 提示词", "prompt", default="")
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        api_url = _pick("🌐 API 地址", "api_url", default="https://ai.t8star.org/v1/chat/completions")
        model = _pick("🤖 模型名称", "model", default="gemini-3-pro-image-preview")
        request_method = _pick("📡 请求方法", "request_method", default="POST")
        secret_location = _pick("📍 密钥位置", "secret_location", default="Header")
        secret_field_name = _pick("🏷️ 密钥字段名", "secret_field_name", default="Authorization")
        auth_scheme = _pick("🪪 认证方式", "auth_scheme", default="Bearer")
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="Auto")
        response_mode = _pick("🧩 响应模式", "response_mode", default="TEXT_AND_IMAGE")
        output_resolution = _pick("🖼️ 输出分辨率", "output_resolution", default="Auto (Model Default)")
        image_preset = _pick("🎨 画质预设", "image_preset", default="hd")
        style_preset = _pick("🖌️ 风格预设", "style_preset", default="natural")
        zoom = _pick("🔍 放大倍数", "zoom", default="1x (不放大)")
        upscale_model = _pick("🧠 放大模型", "upscale_model", default="High Fidelity")
        response_extract_path = _pick("🧭 响应提取路径", "response_extract_path", default="")
        timeout_sec = int(_pick("⏱️ 超时(秒)", "timeout_sec", default=180) or 180)

        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 图像4", "image_4", default=None)

        key = self._get_api_key(str(api_key))
        if not key:
            raise ValueError("未提供 API Key：请设置环境变量 T8STAR_API_KEY 或填写节点 api_key")

        images = [img for img in [image_1, image_2, image_3, image_4] if img is not None]
        converted_prompt = (prompt or "")
        if len(images) >= 1:
            converted_prompt = converted_prompt.replace("图1", "第一张图片")
        if len(images) >= 2:
            converted_prompt = converted_prompt.replace("图2", "第二张图片")
        if len(images) >= 3:
            converted_prompt = converted_prompt.replace("图3", "第三张图片")
        if len(images) >= 4:
            converted_prompt = converted_prompt.replace("图4", "第四张图片")

        # 构建附加要求字符串，将宽高比和分辨率写入 Prompt 以增强效果
        requirements = [f"风格: {style_preset}", f"画质: {image_preset}"]
        if aspect_ratio and aspect_ratio != "Auto":
            requirements.append(f"宽高比: {aspect_ratio}")
        if output_resolution and output_resolution != "Auto (Model Default)":
            requirements.append(f"分辨率: {output_resolution}")
        
        req_str = "，".join(requirements)

        if len(images) > 1:
            image_list = "、".join([f"第{i + 1}张图片" for i in range(len(images))])
            full_prompt = (
                "【多图编辑任务】\n"
                f"我上传了 {len(images)} 张图片（{image_list}），请务必综合参考所有图片进行编辑。\n\n"
                "用户指令：\n"
                f"{converted_prompt}\n\n"
                "重要要求：\n"
                f"1. 必须同时参考所有 {len(images)} 张图片的内容\n"
                f"2. {req_str}\n"
                "3. 保持自然真实的视觉效果\n"
            )
        elif len(images) == 1:
            full_prompt = (
                "请根据以下要求编辑图片：\n"
                f"{converted_prompt}\n\n"
                f"要求：{req_str}\n"
            )
        else:
            full_prompt = converted_prompt

        messages = self._build_messages(prompt=full_prompt, images=images)

        final_secret_field = (secret_field_name or "Authorization").strip() or "Authorization"
        final_auth_scheme = (auth_scheme or "Bearer").strip()
        final_value = key
        if final_auth_scheme.lower() == "bearer":
            final_value = f"Bearer {key}"

        headers = {final_secret_field: final_value, "Content-Type": "application/json"}

        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}

        generation_config: Dict[str, Any] = {}
        if response_mode == "IMAGE_ONLY":
            generation_config["responseModalities"] = ["Image"]
        elif response_mode == "TEXT_ONLY":
            generation_config["responseModalities"] = ["Text"]
        else:
            generation_config["responseModalities"] = ["Text", "Image"]

        image_config: Dict[str, Any] = {}
        if (aspect_ratio or "").strip() and aspect_ratio != "Auto":
            image_config["aspectRatio"] = aspect_ratio
        
        # 注意：Gemini API 的 generationConfig 通常不支持直接传 imageSize/resolution 参数
        # 传递错误的参数会导致整个 generationConfig 被忽略
        # 因此这里不再将 output_resolution 放入 generationConfig，而是完全依赖 Prompt 控制分辨率
        # if output_resolution and output_resolution != "Auto (Model Default)":
        #    image_config["imageSize"] = output_resolution

        if image_config:
            generation_config["imageConfig"] = image_config

        # Try passing generationConfig at root (Standard Gemini)
        if generation_config:
            payload["generationConfig"] = generation_config
            # Also pass in extra_body (Common pattern for OpenAI-compatible proxies wrapping Gemini)
            payload["extra_body"] = {"generationConfig": generation_config}

        # Try OpenAI compatible 'size' and 'quality' parameters
        # This helps bypass potential adapter limitations that strip generationConfig
        if aspect_ratio and aspect_ratio != "Auto":
            w, h = self._get_dimensions(aspect_ratio, output_resolution)
            if w > 0 and h > 0:
                payload["size"] = f"{w}x{h}"
        
        if image_preset:
            payload["quality"] = image_preset # "hd" or "standard"

        resp = _http_json(
            method="POST",
            url=(api_url or "").strip(),
            headers=headers,
            payload=payload,
            timeout=int(timeout_sec),
        )

        raw_json = json.dumps(resp, ensure_ascii=False, indent=2)
        text = self._extract_text(resp)
        image_url = ""
        pil_img: Optional[Image.Image] = None

        if (response_extract_path or "").strip():
            try:
                extracted = self._get_by_path(resp, (response_extract_path or "").strip())
                if isinstance(extracted, str):
                    image_url = extracted.strip()
                elif isinstance(extracted, dict):
                    image_url = str(extracted.get("url", "") or extracted.get("image_url", "") or "").strip()
            except Exception:
                image_url = ""

        if not image_url:
            image_url = self._extract_image_url_like_dapao(resp)

        if image_url:
            pil_img = self._download_image_from_url_like_dapao(image_url, timeout_sec=int(timeout_sec))
            if pil_img is None:
                pil_img = self._download_image_with_fallbacks(
                    url=image_url,
                    timeout_sec=int(timeout_sec),
                    download_headers={final_secret_field: final_value},
                    api_url=(api_url or "").strip(),
                )

        if pil_img is None:
            out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (out, text, image_url, raw_json)

        out = pil2tensor(pil_img)
        return (out, text, image_url, raw_json)
