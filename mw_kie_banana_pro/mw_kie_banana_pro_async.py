import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import folder_paths
import torch

from .mw_kie_banana_pro_basic import pil2tensor, tensor2pil, _download_image, _http_json, _image_to_base64_png


KIE_TASK_FILE = os.path.join(folder_paths.get_temp_directory(), "mw_kie_banana_pro_tasks.json")
KIE_TASK_LOCK = threading.Lock()


def _upload_kie_data_url(upload_base_url: str, api_key: str, media_data_url: str, timeout_sec: int = 180) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "base64Data": media_data_url,
        "uploadPath": "comfyui/mw-kie-banana-pro",
        "fileName": f"mw_kie_media_{int(time.time() * 1000)}.bin",
    }
    result = _http_json(
        method="POST",
        url=f"{upload_base_url}/api/file-base64-upload",
        headers=headers,
        payload=payload,
        timeout=min(60, int(timeout_sec)),
    )
    if result.get("success") is not True or int(result.get("code", 0)) != 200:
        raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))
    data = result.get("data") or {}
    file_url = data.get("fileUrl") or data.get("downloadUrl")
    if not file_url:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return str(file_url)


def _read_tasks() -> Dict[str, Any]:
    if not os.path.exists(KIE_TASK_FILE):
        return {}
    try:
        with open(KIE_TASK_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_tasks(tasks: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(KIE_TASK_FILE), exist_ok=True)
    with open(KIE_TASK_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def _get_next_task_number(tasks: Dict[str, Any]) -> int:
    max_num = 0
    for k in tasks.keys():
        if k.startswith("任务"):
            try:
                n = int(k.replace("任务", "").strip())
                max_num = max(max_num, n)
            except Exception:
                pass
    return max_num + 1


def _get_next_batch_task_number(tasks: Dict[str, Any]) -> int:
    max_num = 0
    for k in tasks.keys():
        if k.startswith("批量任务"):
            try:
                n = int(k.replace("批量任务", "").strip())
                max_num = max(max_num, n)
            except Exception:
                pass
    return max_num + 1


def _status_map(state: str) -> str:
    s = (state or "").lower()
    if s in {"waiting", "queuing", "generating"}:
        return "RUNNING"
    if s == "success":
        return "SUCCEEDED"
    if s == "fail":
        return "FAILED"
    if s == "downloaded":
        return "DOWNLOADED"
    return s.upper() or "UNKNOWN"


class NanoBananaProAsyncSubmit:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
        image_sizes = ["默认", "1K", "2K", "4K"]
        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                "🤖 模型版本": (["nano-banana-pro", "nano-banana-2"], {"default": "nano-banana-pro"}),
                "🖼️ 输出分辨率": (image_sizes, {"default": "默认"}),
                "⚙️ 并发数": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("✅ 状态",)
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _upload_image_data_url(self, api_key: str, image_data_url: str, timeout_sec: int) -> str:
        file_name = f"mw_kie_{int(time.time() * 1000)}.png"
        upload_path = "comfyui/mw-kie-banana-pro"

        try:
            import base64
            from io import BytesIO

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
            payload = {"base64Data": image_data_url, "uploadPath": upload_path, "fileName": file_name}
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

    def _get_credits(self, api_key: str) -> Optional[int]:
        headers = {"Authorization": f"Bearer {api_key}"}
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
        try:
            return int(data)
        except Exception:
            return None

    def _submit_one(
        self,
        api_key: str,
        prompt: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        seed: int,
        image_urls: List[str],
        timeout_sec: int,
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "image_input": image_urls,
            "resolution": "2K" if image_size == "默认" else image_size,
            "output_format": "png",
            "aspect_ratio": aspect_ratio,
            "response_format": "url",
        }
        if aspect_ratio == "auto":
            input_payload.pop("aspect_ratio", None)
        if seed and int(seed) > 0:
            input_payload["seed"] = int(seed)

        payload = {"model": model, "input": input_payload}
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
        return str(task_id)

    def submit(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        prompt = _pick("📝 提示词", "prompt", default="")
        model = _pick("🤖 模型版本", "model", default="nano-banana-pro")
        image_size = _pick("🖼️ 输出分辨率", "image_size", default="默认")
        concurrency = int(_pick("⚙️ 并发数", "concurrency", default=1) or 1)
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        seed = int(_pick("🎲 种子", "seed", default=0) or 0)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            return {"ui": {"string": ["API Key 不能为空。"]}, "result": ("API Key 不能为空。",)}

        images_in = []
        for i in range(1, 15):
            images_in.append(_pick(f"🖼️ 图像{i}", f"image_{i}", default=None))
        pil_images = []
        for t in images_in:
            if t is not None:
                pil_images.extend(tensor2pil(t))

        image_urls: List[str] = []
        if pil_images:
            for p in pil_images:
                data_url = f"data:image/png;base64,{_image_to_base64_png(p)}"
                image_urls.append(self._upload_image_data_url(final_key, data_url, timeout_sec=600))

        task_ids: List[str] = []
        errors: List[str] = []

        for i in range(int(concurrency)):
            try:
                task_ids.append(
                    self._submit_one(
                        api_key=final_key,
                        prompt=prompt,
                        model=model,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio,
                        seed=seed,
                        image_urls=image_urls,
                        timeout_sec=600,
                    )
                )
            except Exception as e:
                errors.append(str(e))

        with KIE_TASK_LOCK:
            tasks = _read_tasks()
            task_name = f"任务{_get_next_task_number(tasks)}"
            tasks[task_name] = {
                "prompt": prompt,
                "model": model,
                "image_size": image_size,
                "aspect_ratio": aspect_ratio,
                "seed": int(seed) if seed else 0,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "subtasks": [{"taskId": tid, "state": "waiting", "resultUrls": [], "downloaded": False} for tid in task_ids],
            }
            _write_tasks(tasks)

        ok = len(task_ids)
        status = f"任务提交成功 | {task_name} | 模型: {model} | 子任务数: {ok}/{concurrency}"
        if seed and int(seed) > 0:
            status += f" | seed: {int(seed)}"
        credits = self._get_credits(final_key)
        status += f" | 剩余积分: {credits if credits is not None else 'N/A'}"
        if errors:
            status += f" | 失败: {len(errors)}"
        return {"ui": {"string": [status]}, "result": (status,)}


class NanoBananaProAsyncQuery:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📊 状态", "📦 ready_json")
    FUNCTION = "query"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _query_one(self, api_key: str, task_id: str, timeout_sec: int = 30) -> Tuple[str, List[str], dict]:
        headers = {"Authorization": f"Bearer {api_key}"}
        result = _http_json(
            method="GET",
            url=f"{self.base_url}/api/v1/jobs/recordInfo",
            headers=headers,
            payload=None,
            timeout=timeout_sec,
            params={"taskId": task_id},
        )
        data = result.get("data") or {}
        state = str(data.get("state") or "")
        urls: List[str] = []
        if state.lower() == "success":
            result_json = data.get("resultJson") or ""
            parsed = json.loads(result_json) if result_json else {}
            urls = parsed.get("resultUrls") or []
            if not isinstance(urls, list):
                urls = []
        if state.lower() == "fail":
            fail_msg = data.get("failMsg") or result.get("msg") or ""
            data["failMsg"] = fail_msg
        return state, [str(u) for u in urls], data

    def _render_status(self, tasks: Dict[str, Any]) -> str:
        lines: List[str] = []
        for name, info in sorted(tasks.items(), key=lambda x: x[1].get("submitted_at", ""), reverse=True):
            subtasks = info.get("subtasks") or []
            total = len(subtasks) if subtasks else 0
            success_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "success")
            fail_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "fail")
            downloaded_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "downloaded")
            running_count = total - success_count - fail_count - downloaded_count
            display = _status_map(info.get("status") or "running")
            prompt = str(info.get("prompt") or "")
            snippet = (prompt[:15] + "...") if len(prompt) > 15 else prompt
            lines.append(
                f"[{display}] {name} ({snippet}) - success:{success_count} fail:{fail_count} downloaded:{downloaded_count} running:{running_count} total:{total}"
            )
        return "\n".join(lines) if lines else "当前无任务记录"

    def query(self, **kwargs):
        api_key = kwargs.get("🔑 API 密钥") or kwargs.get("api_key") or ""
        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (msg, "[]")}

        with KIE_TASK_LOCK:
            tasks = _read_tasks()

        if not tasks:
            msg = "当前没有任务记录。"
            return {"ui": {"string": [msg]}, "result": (msg, "[]")}

        tasks_changed = False
        ready_items: List[Dict[str, Any]] = []

        for name, info in tasks.items():
            subtasks = info.get("subtasks") or []
            total = len(subtasks)
            any_running = False

            for sub in subtasks:
                task_id = sub.get("taskId")
                if not task_id:
                    continue

                state = (sub.get("state") or "").lower()
                if state not in {"success", "fail", "downloaded"}:
                    any_running = True
                    new_state, urls, data = self._query_one(final_key, str(task_id))
                    sub["state"] = new_state
                    if urls:
                        sub["resultUrls"] = urls
                    if data.get("failMsg"):
                        sub["failMsg"] = data.get("failMsg")
                    tasks_changed = True
                    state = str(new_state).lower()

                if state == "success" and not bool(sub.get("downloaded")):
                    urls = sub.get("resultUrls") or []
                    if urls:
                        ready_items.append({"taskId": str(task_id), "urls": [str(u) for u in urls]})

            success_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "success")
            fail_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "fail")
            downloaded_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "downloaded")
            running_count = total - success_count - fail_count - downloaded_count
            if total == 0:
                info["status"] = "running"
            elif running_count > 0:
                info["status"] = "running"
            elif success_count == total:
                info["status"] = "success"
            elif fail_count == total:
                info["status"] = "fail"
            else:
                info["status"] = "partial"

        if tasks_changed:
            with KIE_TASK_LOCK:
                _write_tasks(tasks)

        status_text = self._render_status(tasks)
        ready_json = json.dumps(ready_items, ensure_ascii=False, indent=2)
        return {"ui": {"string": [status_text]}, "result": (status_text, ready_json)}


class NanoBananaProAsyncDownload:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
                "⬇️ 下载上限": ("INT", {"default": 4, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "📊 状态")
    FUNCTION = "download"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _query_one(self, api_key: str, task_id: str, timeout_sec: int = 30) -> Tuple[str, List[str], dict]:
        headers = {"Authorization": f"Bearer {api_key}"}
        result = _http_json(
            method="GET",
            url=f"{self.base_url}/api/v1/jobs/recordInfo",
            headers=headers,
            payload=None,
            timeout=timeout_sec,
            params={"taskId": task_id},
        )
        data = result.get("data") or {}
        state = str(data.get("state") or "")
        urls: List[str] = []
        if state.lower() == "success":
            result_json = data.get("resultJson") or ""
            parsed = json.loads(result_json) if result_json else {}
            urls = parsed.get("resultUrls") or []
            if not isinstance(urls, list):
                urls = []
        if state.lower() == "fail":
            fail_msg = data.get("failMsg") or result.get("msg") or ""
            data["failMsg"] = fail_msg
        return state, [str(u) for u in urls], data

    def _render_status(self, tasks: Dict[str, Any], downloaded_images: int) -> str:
        lines: List[str] = [f"本次下载: {downloaded_images} 张"]
        for name, info in sorted(tasks.items(), key=lambda x: x[1].get("submitted_at", ""), reverse=True):
            subtasks = info.get("subtasks") or []
            total = len(subtasks) if subtasks else 0
            success_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "success")
            fail_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "fail")
            downloaded_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "downloaded")
            running_count = total - success_count - fail_count - downloaded_count
            display = _status_map(info.get("status") or "running")
            prompt = str(info.get("prompt") or "")
            snippet = (prompt[:15] + "...") if len(prompt) > 15 else prompt
            lines.append(
                f"[{display}] {name} ({snippet}) - success:{success_count} fail:{fail_count} downloaded:{downloaded_count} running:{running_count} total:{total}"
            )
        return "\n".join(lines)

    def download(self, **kwargs):
        api_key = kwargs.get("🔑 API 密钥") or kwargs.get("api_key") or ""
        download_limit = kwargs.get("⬇️ 下载上限") if "⬇️ 下载上限" in kwargs else kwargs.get("download_limit", 4)
        final_key = self._get_api_key(str(api_key))
        if not final_key:
            img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (img, msg)}

        with KIE_TASK_LOCK:
            tasks = _read_tasks()

        if not tasks:
            img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            msg = "当前没有任务记录。"
            return {"ui": {"string": [msg]}, "result": (img, msg)}

        downloaded_images: List[torch.Tensor] = []
        downloaded_count = 0
        tasks_changed = False
        base_size: Optional[Tuple[int, int]] = None

        for _name, info in tasks.items():
            if downloaded_count >= int(download_limit):
                break
            subtasks = info.get("subtasks") or []
            for sub in subtasks:
                if downloaded_count >= int(download_limit):
                    break
                if bool(sub.get("downloaded")) or (sub.get("state") or "").lower() == "downloaded":
                    continue

                task_id = str(sub.get("taskId") or "")
                if not task_id:
                    continue

                state = (sub.get("state") or "").lower()
                urls: List[str] = [str(u) for u in (sub.get("resultUrls") or [])] if isinstance(sub.get("resultUrls"), list) else []

                if state not in {"success", "fail"} or not urls:
                    try:
                        new_state, new_urls, data = self._query_one(final_key, task_id)
                        sub["state"] = new_state
                        if new_urls:
                            sub["resultUrls"] = new_urls
                            urls = new_urls
                        if data.get("failMsg"):
                            sub["failMsg"] = data.get("failMsg")
                        tasks_changed = True
                        state = str(new_state).lower()
                    except Exception:
                        continue

                if state != "success" or not urls:
                    continue

                for u in urls:
                    if downloaded_count >= int(download_limit):
                        break
                    try:
                        img_pil = _download_image(str(u), timeout=60)
                        img_pil = img_pil.convert("RGB")
                        if base_size is None:
                            base_size = img_pil.size
                        elif img_pil.size != base_size:
                            from PIL import Image

                            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                            img_pil = img_pil.resize(base_size, resample=resample)
                        downloaded_images.append(pil2tensor(img_pil))
                        downloaded_count += 1
                    except Exception:
                        pass

                sub["state"] = "downloaded"
                sub["downloaded"] = True
                sub["downloaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tasks_changed = True

        for _name, info in tasks.items():
            subtasks = info.get("subtasks") or []
            total = len(subtasks)
            success_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "success")
            fail_count = sum(1 for s in subtasks if (s.get("state") or "").lower() == "fail")
            downloaded_sub = sum(1 for s in subtasks if (s.get("state") or "").lower() == "downloaded")
            running_count = total - success_count - fail_count - downloaded_sub
            if total == 0:
                info["status"] = "running"
            elif running_count > 0:
                info["status"] = "running"
            elif downloaded_sub == total:
                info["status"] = "downloaded"
            elif success_count == total:
                info["status"] = "success"
            elif fail_count == total:
                info["status"] = "fail"
            else:
                info["status"] = "partial"

        if tasks_changed:
            with KIE_TASK_LOCK:
                _write_tasks(tasks)

        if downloaded_images:
            out = torch.cat(downloaded_images, dim=0)
        else:
            out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        status_text = self._render_status(tasks, downloaded_count)
        return {"ui": {"string": [status_text]}, "result": (out, status_text)}


class KieLLMVLMWriter:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🤖 模型": (
                    ["gemini-3-flash-openai", "gemini-3-pro-openai", "gemini-3.1-pro-openai"],
                    {"default": "gemini-3-flash-openai"},
                ),
                "📝 主提示词": ("STRING", {"multiline": True, "default": "请详细分析这个内容，包括所有细节，无废话无解释。"}),
                "🧠 系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "📄 输出文件名": ("STRING", {"default": "generated_prompts.csv"}),
                "🧾 列名": ("STRING", {"default": "prompt"}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📄 文件路径", "📊 状态")
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _resolve_chat_endpoint(self, model: str) -> str:
        endpoint_map = {
            "gemini-3-flash-openai": "gemini-3-flash",
            "gemini-3-pro-openai": "gemini-3-pro",
            "gemini-3.1-pro-openai": "gemini-3.1-pro",
        }
        route_model = endpoint_map.get(str(model), str(model))
        return f"{self.base_url}/{route_model}/v1/chat/completions"

    def _resolve_route_model(self, model: str) -> str:
        endpoint_map = {
            "gemini-3-flash-openai": "gemini-3-flash",
            "gemini-3-pro-openai": "gemini-3-pro",
            "gemini-3.1-pro-openai": "gemini-3.1-pro",
        }
        return endpoint_map.get(str(model), str(model))

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]
            normalized.append({"role": role, "content": content})
        return normalized

    def _chat_completion(self, api_key: str, model: str, messages: List[Dict[str, Any]], timeout_sec: int = 180) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self._resolve_route_model(model),
            "messages": self._normalize_messages(messages),
            "stream": False,
        }
        result = _http_json(
            method="POST",
            url=self._resolve_chat_endpoint(model),
            headers=headers,
            payload=payload,
            timeout=timeout_sec,
        )
        root = result
        if isinstance(result, dict) and isinstance(result.get("data"), dict):
            root = result["data"]
        if isinstance(result, dict) and "code" in result and int(result.get("code") or 0) != 200 and root is result:
            raise ValueError(result.get("msg") or json.dumps(result, ensure_ascii=False))

        choices = (root.get("choices") if isinstance(root, dict) else None) or []
        if not choices:
            raise ValueError(json.dumps(result, ensure_ascii=False))
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).lower() == "text":
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            content = "\n".join(parts).strip()
        if not isinstance(content, str) or not content.strip():
            raise ValueError(json.dumps(result, ensure_ascii=False))
        return content.strip()

    def run(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        model = _pick("🤖 模型", "model", default="gemini-3-flash-openai")
        main_prompt = _pick("📝 主提示词", "main_prompt", default="")
        system_prompt = _pick("🧠 系统提示词", "system_prompt", default="You are a helpful assistant.")
        output_filename = _pick("📄 输出文件名", "output_filename", default="generated_prompts.csv")
        column_name = _pick("🧾 列名", "column_name", default="prompt")
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        import re

        try:
            import pandas as pd  # type: ignore
        except Exception:
            msg = "缺少依赖 pandas：请先安装 pandas（以及导出 Excel 需要 openpyxl）。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        import base64
        from io import BytesIO

        def _to_final_media_url(url_or_data: str) -> str:
            s = str(url_or_data or "").strip()
            if s.startswith("data:"):
                return _upload_kie_data_url(self.upload_base_url, final_key, s, timeout_sec=180)
            return s

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": (main_prompt or "").strip()}]
        for img_t in [image_1, image_2, image_3]:
            if img_t is None:
                continue
            pil_list = tensor2pil(img_t)
            if not pil_list:
                continue
            pil_img = pil_list[0]
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_b64}"
            media_url = _to_final_media_url(data_url)
            user_content.append({"type": "image_url", "image_url": {"url": media_url}})

        messages.append({"role": "user", "content": user_content})

        try:
            llm_response = self._chat_completion(final_key, model, messages, timeout_sec=180)
        except Exception as e:
            msg = f"LLM API 调用失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        try:
            json_match = re.search(r"\[.*\]", llm_response, re.DOTALL)
            parsed_list = json.loads(json_match.group()) if json_match else [l.strip() for l in llm_response.split("\n") if l.strip()]
        except Exception:
            parsed_list = [l.strip() for l in llm_response.split("\n") if l.strip()]

        parsed_list = [item for item in parsed_list if item]
        if not isinstance(parsed_list, list) or not parsed_list:
            msg = f"模型未返回有效列表。收到: {llm_response}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        df = pd.DataFrame(parsed_list, columns=[(column_name or "prompt").strip() or "prompt"])
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, (output_filename or "").strip())

        try:
            if full_path.lower().endswith(".csv"):
                df.to_csv(full_path, index=False, encoding="utf-8-sig")
            elif full_path.lower().endswith((".xls", ".xlsx")):
                df.to_excel(full_path, index=False)
            else:
                msg = "文件名必须以 .csv, .xls, 或 .xlsx 结尾。"
                return {"ui": {"string": [msg]}, "result": ("", msg)}
        except Exception as e:
            msg = f"写入文件失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        guide_map = {
            "gemini-3-flash-openai": "https://docs.kie.ai/market/gemini/gemini-3-flash",
            "gemini-3-pro-openai": "https://docs.kie.ai/market/gemini/gemini-3-pro",
            "gemini-3.1-pro-openai": "https://docs.kie.ai/market/gemini/gemini-3-1-pro",
        }
        guide_url = guide_map.get(model, "")
        status = f"成功生成 {len(parsed_list)} 条记录并写入: {os.path.basename(full_path)}"
        if guide_url:
            status = f"{status} | 模型指南: {guide_url}"
        return {"ui": {"string": [status]}, "result": (full_path, status)}


class ZhenzhenLLMVLMWriter:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🤖 模型": (
                    ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-pro-preview"],
                    {"default": "gemini-3-flash-preview"},
                ),
                "📝 主提示词": ("STRING", {"multiline": True, "default": "请详细分析这个内容，包括所有细节，无废话无解释。"}),
                "🧠 系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "📄 输出文件名": ("STRING", {"default": "generated_prompts.csv"}),
                "🧾 列名": ("STRING", {"default": "prompt"}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 ZHENZHEN_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📄 文件路径", "📊 状态")
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://ai.t8star.org"

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("ZHENZHEN_API_KEY", "") or api_key).strip()

    def _chat_completion(self, api_key: str, model: str, messages: List[Dict[str, Any]], timeout_sec: int = 180) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        result = _http_json(
            method="POST",
            url=f"{self.base_url}/v1/chat/completions",
            headers=headers,
            payload=payload,
            timeout=timeout_sec,
        )

        root = result
        if isinstance(result, dict) and isinstance(result.get("error"), dict):
            err = result.get("error") or {}
            raise ValueError(err.get("message") or json.dumps(result, ensure_ascii=False))

        choices = (root.get("choices") if isinstance(root, dict) else None) or []
        if not choices:
            raise ValueError(json.dumps(result, ensure_ascii=False))
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).lower() == "text":
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            content = "\n".join(parts).strip()
        if not isinstance(content, str) or not content.strip():
            raise ValueError(json.dumps(result, ensure_ascii=False))
        return content.strip()

    def run(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        model = _pick("🤖 模型", "model", default="gemini-3-flash-preview")
        main_prompt = _pick("📝 主提示词", "main_prompt", default="")
        system_prompt = _pick("🧠 系统提示词", "system_prompt", default="You are a helpful assistant.")
        output_filename = _pick("📄 输出文件名", "output_filename", default="generated_prompts.csv")
        column_name = _pick("🧾 列名", "column_name", default="prompt")
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        import re

        try:
            import pandas as pd  # type: ignore
        except Exception:
            msg = "缺少依赖 pandas：请先安装 pandas（以及导出 Excel 需要 openpyxl）。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        import base64
        from io import BytesIO

        def _to_final_media_url(url_or_data: str) -> str:
            s = str(url_or_data or "").strip()
            if s.startswith("data:"):
                return _upload_kie_data_url(self.upload_base_url, final_key, s, timeout_sec=int(timeout_sec))
            return s

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": (main_prompt or "").strip()}]
        for img_t in [image_1, image_2, image_3]:
            if img_t is None:
                continue
            pil_list = tensor2pil(img_t)
            if not pil_list:
                continue
            pil_img = pil_list[0]
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_b64}"
            media_url = _to_final_media_url(data_url)
            user_content.append({"type": "image_url", "image_url": {"url": media_url}})

        messages.append({"role": "user", "content": user_content})

        try:
            llm_response = self._chat_completion(final_key, str(model), messages, timeout_sec=180)
        except Exception as e:
            msg = f"LLM API 调用失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        try:
            json_match = re.search(r"\[.*\]", llm_response, re.DOTALL)
            parsed_list = json.loads(json_match.group()) if json_match else [l.strip() for l in llm_response.split("\n") if l.strip()]
        except Exception:
            parsed_list = [l.strip() for l in llm_response.split("\n") if l.strip()]

        parsed_list = [item for item in parsed_list if item]
        if not isinstance(parsed_list, list) or not parsed_list:
            msg = f"模型未返回有效列表。收到: {llm_response}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        df = pd.DataFrame(parsed_list, columns=[(column_name or "prompt").strip() or "prompt"])
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, (output_filename or "").strip())

        try:
            if full_path.lower().endswith(".csv"):
                df.to_csv(full_path, index=False, encoding="utf-8-sig")
            elif full_path.lower().endswith((".xls", ".xlsx")):
                df.to_excel(full_path, index=False)
            else:
                msg = "文件名必须以 .csv, .xls, 或 .xlsx 结尾。"
                return {"ui": {"string": [msg]}, "result": ("", msg)}
        except Exception as e:
            msg = f"写入文件失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        status = f"成功生成 {len(parsed_list)} 条记录并写入: {os.path.basename(full_path)}"
        return {"ui": {"string": [status]}, "result": (full_path, status)}


class Gemini3MultimodalChatKie:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🤖 模型": (
                    ["gemini-3-flash-openai", "gemini-3-pro-openai", "gemini-3.1-pro-openai"],
                    {"default": "gemini-3-flash-openai"},
                ),
                "🧠 系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "💬 用户提示词": ("STRING", {"multiline": True, "default": "请详细分析这个内容，包括所有细节，无废话无解释。"}),
                "🕘 历史JSON": ("STRING", {"multiline": True, "default": ""}),
                "🌡️ temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "🎯 top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🧮 max_tokens": ("INT", {"default": 2048, "min": 1, "max": 200000}),
                "⏱️ 超时(秒)": ("INT", {"default": 180, "min": 10, "max": 3600}),
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
                "🎞️ 视频/帧": ("*",),
                "🎧 音频": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("💎 回复",)
    FUNCTION = "chat"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _resolve_chat_endpoint(self, model: str) -> str:
        endpoint_map = {
            "gemini-3-flash-openai": "gemini-3-flash",
            "gemini-3-pro-openai": "gemini-3-pro",
            "gemini-3.1-pro-openai": "gemini-3.1-pro",
        }
        route_model = endpoint_map.get(str(model), str(model))
        return f"{self.base_url}/{route_model}/v1/chat/completions"

    def _resolve_route_model(self, model: str) -> str:
        endpoint_map = {
            "gemini-3-flash-openai": "gemini-3-flash",
            "gemini-3-pro-openai": "gemini-3-pro",
            "gemini-3.1-pro-openai": "gemini-3.1-pro",
        }
        return endpoint_map.get(str(model), str(model))

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]
            normalized.append({"role": role, "content": content})
        return normalized

    def _extract_text(self, message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, list):
            parts: List[str] = []
            for item in message_content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).lower() == "text":
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            return "\n".join(parts).strip()
        return ""

    def _chat_completion(
        self,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_sec: int,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self._resolve_route_model(model),
            "messages": self._normalize_messages(messages),
            "stream": False,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        return _http_json(
            method="POST",
            url=self._resolve_chat_endpoint(model),
            headers=headers,
            payload=payload,
            timeout=int(timeout_sec),
        )

    def chat(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        model = _pick("🤖 模型", "model", default="gemini-3-flash-openai")
        system_prompt = _pick("🧠 系统提示词", "system_prompt", default="You are a helpful assistant.")
        user_prompt = _pick("💬 用户提示词", "user_prompt", default="")
        history_json = _pick("🕘 历史JSON", "history_json", default="")
        temperature = float(_pick("🌡️ temperature", "temperature", default=0.7) or 0.7)
        top_p = float(_pick("🎯 top_p", "top_p", default=0.95) or 0.95)
        max_tokens = int(_pick("🧮 max_tokens", "max_tokens", default=2048) or 2048)
        timeout_sec = int(_pick("⏱️ 超时(秒)", "timeout_sec", default=180) or 180)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 图像4", "image_4", default=None)
        image_5 = _pick("🖼️ 图像5", "image_5", default=None)
        image_6 = _pick("🖼️ 图像6", "image_6", default=None)
        image_7 = _pick("🖼️ 图像7", "image_7", default=None)
        image_8 = _pick("🖼️ 图像8", "image_8", default=None)
        视频 = _pick("🎞️ 视频/帧", "视频", default=None)
        音频 = _pick("🎧 音频", "音频", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        import base64
        from io import BytesIO

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if history_json and str(history_json).strip():
            try:
                parsed = json.loads(str(history_json))
                if isinstance(parsed, list):
                    for m in parsed:
                        if isinstance(m, dict) and isinstance(m.get("role"), str) and "content" in m:
                            messages.append(m)
            except Exception:
                pass

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": (user_prompt or "").strip()}]
        for img_t in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]:
            if img_t is None:
                continue
            pil_list = tensor2pil(img_t)
            if not pil_list:
                continue
            pil_img = pil_list[0]
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        def _read_file_base64(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        def _video_mime(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".mp4":
                return "video/mp4"
            if ext == ".webm":
                return "video/webm"
            if ext == ".mov":
                return "video/quicktime"
            if ext == ".mkv":
                return "video/x-matroska"
            return "video/mp4"

        def _encode_waveform_to_wav_base64(waveform: torch.Tensor, sample_rate: int) -> str:
            import wave

            wf = waveform
            if isinstance(wf, torch.Tensor):
                if wf.dim() == 3:
                    wf = wf[0]
                elif wf.dim() == 1:
                    wf = wf.unsqueeze(0)
            if not isinstance(wf, torch.Tensor) or wf.dim() != 2:
                raise ValueError("不支持的音频输入格式（waveform）")

            sr = int(sample_rate) if sample_rate else 44100
            wf = wf.detach().to(dtype=torch.float32).clamp(-1.0, 1.0)
            wf_i16 = (wf * 32767.0).round().to(torch.int16)
            interleaved = wf_i16.transpose(0, 1).contiguous()

            wav_buf = BytesIO()
            with wave.open(wav_buf, "wb") as wavf:
                wavf.setnchannels(int(wf_i16.shape[0]))
                wavf.setsampwidth(2)
                wavf.setframerate(sr)
                wavf.writeframes(interleaved.cpu().numpy().tobytes())
            return base64.b64encode(wav_buf.getvalue()).decode("utf-8")

        def _sample_indices(n: int, max_n: int) -> List[int]:
            if n <= 0:
                return []
            if n <= max_n:
                return list(range(n))
            picks = [0, n // 3, (2 * n) // 3, n - 1]
            seen = set()
            out: List[int] = []
            for i in picks:
                i2 = max(0, min(n - 1, int(i)))
                if i2 in seen:
                    continue
                seen.add(i2)
                out.append(i2)
                if len(out) >= max_n:
                    break
            if not out:
                out = [0]
            return out

        def _video_to_frames(video_obj: Any, max_frames: int = 4) -> List[Any]:
            if video_obj is None:
                return []
            if isinstance(video_obj, torch.Tensor):
                if video_obj.dim() == 4:
                    return [video_obj[i : i + 1] for i in _sample_indices(int(video_obj.shape[0]), max_frames)]
                return []
            if isinstance(video_obj, (list, tuple)):
                frames: List[Any] = []
                for item in video_obj:
                    if item is None:
                        continue
                    if isinstance(item, torch.Tensor):
                        frames.append(item)
                    elif isinstance(item, (list, tuple, dict)):
                        frames.extend(_video_to_frames(item, max_frames=max_frames))
                    if len(frames) >= max_frames:
                        break
                return frames[:max_frames]
            if isinstance(video_obj, dict):
                for k in ["frames", "frame_list", "images", "imgs", "data", "video_frames"]:
                    v = video_obj.get(k)
                    if v is None:
                        continue
                    extracted = _video_to_frames(v, max_frames=max_frames)
                    if extracted:
                        return extracted[:max_frames]
                for v in video_obj.values():
                    extracted = _video_to_frames(v, max_frames=max_frames)
                    if extracted:
                        return extracted[:max_frames]
                return []
            for k in ["frames", "images", "frame_list"]:
                try:
                    v = getattr(video_obj, k)
                except Exception:
                    v = None
                extracted = _video_to_frames(v, max_frames=max_frames)
                if extracted:
                    return extracted[:max_frames]
            return []

        def _audio_to_media_url(audio_obj: Any) -> Optional[str]:
            if audio_obj is None:
                return None
            if isinstance(audio_obj, dict):
                if isinstance(audio_obj.get("data"), str) and isinstance(audio_obj.get("format"), str):
                    fmt = str(audio_obj.get("format") or "wav").lower()
                    if fmt not in {"wav", "mp3", "m4a"}:
                        fmt = "wav"
                    return f"data:audio/{fmt};base64,{audio_obj['data']}"
                path = audio_obj.get("path") or audio_obj.get("file_path") or audio_obj.get("filename")
                if isinstance(path, str) and path.strip() and os.path.exists(path.strip()):
                    ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
                    fmt = ext if ext in {"wav", "mp3", "m4a"} else "wav"
                    return f"data:audio/{fmt};base64,{_read_file_base64(path.strip())}"
                wf = audio_obj.get("waveform")
                sr = audio_obj.get("sample_rate") or audio_obj.get("sr")
                if isinstance(wf, torch.Tensor):
                    return f"data:audio/wav;base64,{_encode_waveform_to_wav_base64(wf, int(sr or 44100))}"
            if isinstance(audio_obj, str) and audio_obj.strip() and os.path.exists(audio_obj.strip()):
                path = audio_obj.strip()
                ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
                fmt = ext if ext in {"wav", "mp3", "m4a"} else "wav"
                return f"data:audio/{fmt};base64,{_read_file_base64(path)}"
            raise ValueError("不支持的音频输入格式")

        def _video_to_media_url(video_obj: Any) -> Optional[str]:
            if video_obj is None:
                return None
            if isinstance(video_obj, str) and video_obj.strip():
                s = video_obj.strip()
                if s.startswith(("http://", "https://", "data:")):
                    return s
                if os.path.exists(s):
                    vb64 = _read_file_base64(s)
                    return f"data:{_video_mime(s)};base64,{vb64}"

            if isinstance(video_obj, dict):
                url_val = video_obj.get("url") or video_obj.get("video_url")
                if isinstance(url_val, str) and url_val.strip().startswith(("http://", "https://", "data:")):
                    return url_val.strip()

                for k in [
                    "path",
                    "file_path",
                    "filepath",
                    "filename",
                    "fullpath",
                    "full_path",
                    "absolute_path",
                    "video_path",
                ]:
                    v = video_obj.get(k)
                    if isinstance(v, str) and v.strip() and os.path.exists(v.strip()):
                        path = v.strip()
                        vb64 = _read_file_base64(path)
                        return f"data:{_video_mime(path)};base64,{vb64}"

                for k in ["paths", "files", "file_list"]:
                    v = video_obj.get(k)
                    if isinstance(v, list) and v:
                        for item in v:
                            if isinstance(item, str) and item.strip() and os.path.exists(item.strip()):
                                path = item.strip()
                                vb64 = _read_file_base64(path)
                                return f"data:{_video_mime(path)};base64,{vb64}"

                for v in video_obj.values():
                    if isinstance(v, (dict, list, tuple, str)):
                        try:
                            res = _video_to_media_url(v)
                            if res is not None:
                                return res
                        except Exception:
                            pass

            if isinstance(video_obj, (list, tuple)):
                for item in video_obj:
                    if isinstance(item, (dict, list, tuple, str)):
                        try:
                            res = _video_to_media_url(item)
                            if res is not None:
                                return res
                        except Exception:
                            pass

            for k in [
                "path",
                "file_path",
                "filepath",
                "filename",
                "fullpath",
                "full_path",
                "absolute_path",
                "video_path",
            ]:
                try:
                    v = getattr(video_obj, k)
                except Exception:
                    v = None
                if isinstance(v, str) and v.strip():
                    s = v.strip()
                    if s.startswith(("http://", "https://", "data:")):
                        return s
                    if os.path.exists(s):
                        vb64 = _read_file_base64(s)
                        return f"data:{_video_mime(s)};base64,{vb64}"

            raise ValueError("不支持的视频输入格式（请连接 VIDEO 类型节点）")

        audio_media_url = _audio_to_media_url(音频)
        if audio_media_url is not None:
            user_content.append({"type": "image_url", "image_url": {"url": _to_final_media_url(audio_media_url)}})

        video_frames = _video_to_frames(视频, max_frames=4)
        if video_frames:
            for f in video_frames[:4]:
                try:
                    if isinstance(f, torch.Tensor):
                        pil_list = tensor2pil(f)
                        if not pil_list:
                            continue
                        pil_img = pil_list[0]
                    else:
                        continue
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    data_url = f"data:image/png;base64,{img_b64}"
                    user_content.append({"type": "image_url", "image_url": {"url": _to_final_media_url(data_url)}})
                except Exception:
                    continue
        else:
            video_media_url = _video_to_media_url(视频)
            if video_media_url is not None:
                user_content.append({"type": "image_url", "image_url": {"url": _to_final_media_url(video_media_url)}})

        messages.append({"role": "user", "content": user_content})

        try:
            resp = self._chat_completion(
                api_key=final_key,
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            msg = f"LLM API 调用失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        raw_json = json.dumps(resp, ensure_ascii=False, indent=2)
        root = resp
        if isinstance(resp, dict) and isinstance(resp.get("data"), dict):
            root = resp["data"]
        if isinstance(resp, dict) and "code" in resp and int(resp.get("code") or 0) != 200 and root is resp:
            msg = str(resp.get("msg") or "LLM API 返回错误。")
            return {"ui": {"string": [msg]}, "result": (msg,)}

        choices = (root.get("choices") if isinstance(root, dict) else None) or []
        if not choices:
            msg = "模型未返回 choices。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        message = (choices[0] or {}).get("message") or {}
        reply_text = self._extract_text(message.get("content"))
        if not reply_text:
            reply_text = json.dumps(message, ensure_ascii=False)

        return {
            "ui": {"string": [reply_text]},
            "result": (reply_text,),
        }


class Gemini3MultimodalChatZhenzhen:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🤖 模型": (
                    ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-pro-preview"],
                    {"default": "gemini-3-flash-preview"},
                ),
                "🧠 系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "💬 用户提示词": ("STRING", {"multiline": True, "default": "请详细分析这个内容，包括所有细节，无废话无解释。"}),
                "🕘 历史JSON": ("STRING", {"multiline": True, "default": ""}),
                "🌡️ temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "🎯 top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🧮 max_tokens": ("INT", {"default": 2048, "min": 1, "max": 200000}),
                "⏱️ 超时(秒)": ("INT", {"default": 180, "min": 10, "max": 3600}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 ZHENZHEN_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
                "🎞️ 视频/帧": ("*",),
                "🎧 音频": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("💎 回复",)
    FUNCTION = "chat"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://ai.t8star.org"

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("ZHENZHEN_API_KEY", "") or api_key).strip()

    def _extract_text(self, message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, list):
            parts: List[str] = []
            for item in message_content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).lower() == "text":
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            return "\n".join(parts).strip()
        return ""

    def _chat_completion(
        self,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_sec: int,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        return _http_json(
            method="POST",
            url=f"{self.base_url}/v1/chat/completions",
            headers=headers,
            payload=payload,
            timeout=int(timeout_sec),
        )

    def chat(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        model = _pick("🤖 模型", "model", default="gemini-3-flash-preview")
        system_prompt = _pick("🧠 系统提示词", "system_prompt", default="You are a helpful assistant.")
        user_prompt = _pick("💬 用户提示词", "user_prompt", default="")
        history_json = _pick("🕘 历史JSON", "history_json", default="")
        temperature = float(_pick("🌡️ temperature", "temperature", default=0.7) or 0.7)
        top_p = float(_pick("🎯 top_p", "top_p", default=0.95) or 0.95)
        max_tokens = int(_pick("🧮 max_tokens", "max_tokens", default=2048) or 2048)
        timeout_sec = int(_pick("⏱️ 超时(秒)", "timeout_sec", default=180) or 180)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 图像4", "image_4", default=None)
        视频 = _pick("🎞️ 视频/帧", "视频", default=None)
        音频 = _pick("🎧 音频", "音频", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        import base64
        from io import BytesIO

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if history_json and str(history_json).strip():
            try:
                parsed = json.loads(str(history_json))
                if isinstance(parsed, list):
                    for m in parsed:
                        if isinstance(m, dict) and isinstance(m.get("role"), str) and "content" in m:
                            messages.append(m)
            except Exception:
                pass

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": (user_prompt or "").strip()}]
        for img_t in [image_1, image_2, image_3, image_4]:
            if img_t is None:
                continue
            pil_list = tensor2pil(img_t)
            if not pil_list:
                continue
            pil_img = pil_list[0]
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        def _read_file_base64(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        def _video_mime(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".mp4":
                return "video/mp4"
            if ext == ".webm":
                return "video/webm"
            if ext == ".mov":
                return "video/quicktime"
            if ext == ".mkv":
                return "video/x-matroska"
            return "video/mp4"

        def _encode_waveform_to_wav_base64(waveform: torch.Tensor, sample_rate: int) -> str:
            import wave

            wf = waveform
            if isinstance(wf, torch.Tensor):
                if wf.dim() == 3:
                    wf = wf[0]
                elif wf.dim() == 1:
                    wf = wf.unsqueeze(0)
            if not isinstance(wf, torch.Tensor) or wf.dim() != 2:
                raise ValueError("不支持的音频输入格式（waveform）")

            sr = int(sample_rate) if sample_rate else 44100
            wf = wf.detach().to(dtype=torch.float32).clamp(-1.0, 1.0)
            wf_i16 = (wf * 32767.0).round().to(torch.int16)
            interleaved = wf_i16.transpose(0, 1).contiguous()

            wav_buf = BytesIO()
            with wave.open(wav_buf, "wb") as wavf:
                wavf.setnchannels(int(wf_i16.shape[0]))
                wavf.setsampwidth(2)
                wavf.setframerate(sr)
                wavf.writeframes(interleaved.cpu().numpy().tobytes())
            return base64.b64encode(wav_buf.getvalue()).decode("utf-8")

        def _sample_indices(n: int, max_n: int) -> List[int]:
            if n <= 0:
                return []
            if n <= max_n:
                return list(range(n))
            picks = [0, n // 3, (2 * n) // 3, n - 1]
            seen = set()
            out: List[int] = []
            for i in picks:
                i2 = max(0, min(n - 1, int(i)))
                if i2 in seen:
                    continue
                seen.add(i2)
                out.append(i2)
                if len(out) >= max_n:
                    break
            if not out:
                out = [0]
            return out

        def _video_to_frames(video_obj: Any, max_frames: int = 4) -> List[Any]:
            if video_obj is None:
                return []
            if isinstance(video_obj, torch.Tensor):
                if video_obj.dim() == 4:
                    return [video_obj[i : i + 1] for i in _sample_indices(int(video_obj.shape[0]), max_frames)]
                return []
            if isinstance(video_obj, (list, tuple)):
                frames: List[Any] = []
                for item in video_obj:
                    if item is None:
                        continue
                    if isinstance(item, torch.Tensor):
                        frames.append(item)
                    elif isinstance(item, (list, tuple, dict)):
                        frames.extend(_video_to_frames(item, max_frames=max_frames))
                    if len(frames) >= max_frames:
                        break
                return frames[:max_frames]
            if isinstance(video_obj, dict):
                for k in ["frames", "frame_list", "images", "imgs", "data", "video_frames"]:
                    v = video_obj.get(k)
                    if v is None:
                        continue
                    extracted = _video_to_frames(v, max_frames=max_frames)
                    if extracted:
                        return extracted[:max_frames]
                for v in video_obj.values():
                    extracted = _video_to_frames(v, max_frames=max_frames)
                    if extracted:
                        return extracted[:max_frames]
                return []
            for k in ["frames", "images", "frame_list"]:
                try:
                    v = getattr(video_obj, k)
                except Exception:
                    v = None
                extracted = _video_to_frames(v, max_frames=max_frames)
                if extracted:
                    return extracted[:max_frames]
            return []

        def _audio_to_input_audio(audio_obj: Any) -> Optional[Dict[str, Any]]:
            if audio_obj is None:
                return None
            if isinstance(audio_obj, dict):
                if isinstance(audio_obj.get("data"), str) and isinstance(audio_obj.get("format"), str):
                    return {"data": audio_obj["data"], "format": audio_obj["format"]}
                path = audio_obj.get("path") or audio_obj.get("file_path") or audio_obj.get("filename")
                if isinstance(path, str) and path.strip() and os.path.exists(path.strip()):
                    ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
                    fmt = ext if ext in {"wav", "mp3", "m4a"} else "wav"
                    return {"data": _read_file_base64(path.strip()), "format": fmt}
                wf = audio_obj.get("waveform")
                sr = audio_obj.get("sample_rate") or audio_obj.get("sr")
                if isinstance(wf, torch.Tensor):
                    return {"data": _encode_waveform_to_wav_base64(wf, int(sr or 44100)), "format": "wav"}
            if isinstance(audio_obj, str) and audio_obj.strip() and os.path.exists(audio_obj.strip()):
                path = audio_obj.strip()
                ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
                fmt = ext if ext in {"wav", "mp3", "m4a"} else "wav"
                return {"data": _read_file_base64(path), "format": fmt}
            raise ValueError("不支持的音频输入格式")

        def _video_to_video_url(video_obj: Any) -> Optional[Dict[str, Any]]:
            if video_obj is None:
                return None
            if isinstance(video_obj, str) and video_obj.strip():
                s = video_obj.strip()
                if s.startswith(("http://", "https://", "data:")):
                    return {"url": s}
                if os.path.exists(s):
                    vb64 = _read_file_base64(s)
                    return {"url": f"data:{_video_mime(s)};base64,{vb64}"}

            if isinstance(video_obj, dict):
                url_val = video_obj.get("url") or video_obj.get("video_url")
                if isinstance(url_val, str) and url_val.strip().startswith(("http://", "https://", "data:")):
                    return {"url": url_val.strip()}

                for k in [
                    "path",
                    "file_path",
                    "filepath",
                    "filename",
                    "fullpath",
                    "full_path",
                    "absolute_path",
                    "video_path",
                ]:
                    v = video_obj.get(k)
                    if isinstance(v, str) and v.strip() and os.path.exists(v.strip()):
                        path = v.strip()
                        vb64 = _read_file_base64(path)
                        return {"url": f"data:{_video_mime(path)};base64,{vb64}"}

                for k in ["paths", "files", "file_list"]:
                    v = video_obj.get(k)
                    if isinstance(v, list) and v:
                        for item in v:
                            if isinstance(item, str) and item.strip() and os.path.exists(item.strip()):
                                path = item.strip()
                                vb64 = _read_file_base64(path)
                                return {"url": f"data:{_video_mime(path)};base64,{vb64}"}

                for v in video_obj.values():
                    if isinstance(v, (dict, list, tuple, str)):
                        try:
                            res = _video_to_video_url(v)
                            if res is not None:
                                return res
                        except Exception:
                            pass

            if isinstance(video_obj, (list, tuple)):
                for item in video_obj:
                    if isinstance(item, (dict, list, tuple, str)):
                        try:
                            res = _video_to_video_url(item)
                            if res is not None:
                                return res
                        except Exception:
                            pass

            for k in [
                "path",
                "file_path",
                "filepath",
                "filename",
                "fullpath",
                "full_path",
                "absolute_path",
                "video_path",
            ]:
                try:
                    v = getattr(video_obj, k)
                except Exception:
                    v = None
                if isinstance(v, str) and v.strip():
                    s = v.strip()
                    if s.startswith(("http://", "https://", "data:")):
                        return {"url": s}
                    if os.path.exists(s):
                        vb64 = _read_file_base64(s)
                        return {"url": f"data:{_video_mime(s)};base64,{vb64}"}

            raise ValueError("不支持的视频输入格式（请连接 VIDEO 类型节点）")

        audio_payload = _audio_to_input_audio(音频)
        if audio_payload is not None:
            user_content.append({"type": "input_audio", "input_audio": audio_payload})

        video_frames = _video_to_frames(视频, max_frames=4)
        if video_frames:
            for f in video_frames[:4]:
                try:
                    if isinstance(f, torch.Tensor):
                        pil_list = tensor2pil(f)
                        if not pil_list:
                            continue
                        pil_img = pil_list[0]
                    else:
                        continue
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
                except Exception:
                    continue
        else:
            video_payload = _video_to_video_url(视频)
            if video_payload is not None:
                user_content.append({"type": "video_url", "video_url": video_payload})

        messages.append({"role": "user", "content": user_content})

        try:
            resp = self._chat_completion(
                api_key=final_key,
                model=str(model),
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            msg = f"LLM API 调用失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        root = resp
        if isinstance(resp, dict) and isinstance(resp.get("error"), dict):
            err = resp.get("error") or {}
            msg = str(err.get("message") or "LLM API 返回错误。")
            return {"ui": {"string": [msg]}, "result": (msg,)}
        if isinstance(resp, dict) and isinstance(resp.get("data"), dict):
            root = resp["data"]
        if isinstance(resp, dict) and "code" in resp and int(resp.get("code") or 0) != 200 and root is resp:
            msg = str(resp.get("msg") or "LLM API 返回错误。")
            return {"ui": {"string": [msg]}, "result": (msg,)}

        choices = (root.get("choices") if isinstance(root, dict) else None) or []
        if not choices:
            msg = "模型未返回 choices。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        message = (choices[0] or {}).get("message") or {}
        reply_text = self._extract_text(message.get("content"))
        if not reply_text:
            reply_text = json.dumps(message, ensure_ascii=False)

        return {
            "ui": {"string": [reply_text]},
            "result": (reply_text,),
        }


class NanoBananaProAsyncBatchSubmit:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
        image_sizes = ["默认", "1K", "2K", "4K"]
        return {
            "required": {
                "📄 文件路径": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "🧾 列名": ("STRING", {"default": "prompt"}),
                "🧩 提示词前缀": ("STRING", {"multiline": True, "default": ""}),
                "🤖 模型版本": (["nano-banana-pro", "nano-banana-2"], {"default": "nano-banana-pro"}),
                "🖼️ 输出分辨率": (image_sizes, {"default": "默认"}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
                "🔁 执行次数/条": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("✅ 状态",)
    FUNCTION = "submit_batch"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _upload_image_data_url(self, api_key: str, image_data_url: str, timeout_sec: int) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "base64Data": image_data_url,
            "uploadPath": "comfyui/mw-kie-banana-pro",
            "fileName": f"mw_kie_{int(time.time() * 1000)}.png",
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

    def _submit_one(
        self,
        api_key: str,
        prompt: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        image_urls: List[str],
        timeout_sec: int,
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "image_input": image_urls,
            "resolution": "2K" if image_size == "默认" else image_size,
            "output_format": "png",
            "aspect_ratio": aspect_ratio,
            "response_format": "url",
        }
        if aspect_ratio == "auto":
            input_payload.pop("aspect_ratio", None)
        payload = {"model": model, "input": input_payload}
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
        return str(task_id)

    def _get_credits(self, api_key: str) -> Optional[int]:
        try:
            result = _http_json(
                method="GET",
                url=f"{self.base_url}/api/v1/chat/credit",
                headers={"Authorization": f"Bearer {api_key}"},
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

    def submit_batch(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        file_path = _pick("📄 文件路径", "file_path", default="")
        column_name = _pick("🧾 列名", "column_name", default="prompt")
        prompt_prefix = _pick("🧩 提示词前缀", "prompt_prefix", default="")
        model = _pick("🤖 模型版本", "model", default="nano-banana-pro")
        image_size = _pick("🖼️ 输出分辨率", "image_size", default="默认")
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        executions_per_prompt = int(_pick("🔁 执行次数/条", "executions_per_prompt", default=1) or 1)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 图像4", "image_4", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            return {"ui": {"string": ["API Key 不能为空。"]}, "result": ("API Key 不能为空。",)}

        if not file_path or not os.path.exists(file_path):
            return {"ui": {"string": ["文件路径为空或文件不存在。"]}, "result": ("文件路径为空或文件不存在。",)}

        try:
            import pandas as pd  # type: ignore
        except Exception:
            return {"ui": {"string": ["缺少依赖 pandas：请先安装 pandas（以及读取 Excel 需要 openpyxl）。"]}, "result": ("缺少依赖 pandas。",)}

        try:
            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
            elif file_path.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                return {"ui": {"string": ["仅支持 .csv, .xls, .xlsx 文件。"]}, "result": ("仅支持 .csv, .xls, .xlsx 文件。",)}
        except Exception as e:
            return {"ui": {"string": [f"读取文件失败: {str(e)}"]}, "result": (f"读取文件失败: {str(e)}",)}

        if column_name not in df.columns:
            return {"ui": {"string": [f"列 '{column_name}' 不存在。"]}, "result": (f"列 '{column_name}' 不存在。",)}

        base_prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()]
        if not base_prompts:
            return {"ui": {"string": [f"列 '{column_name}' 中未找到有效 prompt。"]}, "result": (f"列 '{column_name}' 中未找到有效 prompt。",)}

        prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]

        images_in = [image_1, image_2, image_3, image_4]
        pil_images = []
        for t in images_in:
            if t is not None:
                pil_images.extend(tensor2pil(t))

        image_urls: List[str] = []
        if pil_images:
            try:
                for p in pil_images:
                    data_url = f"data:image/png;base64,{_image_to_base64_png(p)}"
                    image_urls.append(self._upload_image_data_url(final_key, data_url, timeout_sec=600))
            except Exception as e:
                msg = f"图片上传失败: {str(e)}"
                return {"ui": {"string": [msg]}, "result": (msg,)}

        import concurrent.futures

        task_ids: List[str] = []
        errors: List[str] = []

        def _submit(i: int, p: str) -> Optional[str]:
            try:
                return self._submit_one(
                    api_key=final_key,
                    prompt=p,
                    model=model,
                    image_size=image_size,
                    aspect_ratio=aspect_ratio,
                    image_urls=image_urls,
                    timeout_sec=600,
                )
            except Exception as e:
                errors.append(str(e))
                return None

        max_workers = min(10, max(1, len(prompts)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_submit, i, p) for i, p in enumerate(prompts)]
            for f in concurrent.futures.as_completed(futures):
                tid = f.result()
                if tid:
                    task_ids.append(tid)

        if not task_ids:
            msg = f"所有任务提交均失败。错误示例: {errors[0] if errors else '未知'}"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        with KIE_TASK_LOCK:
            tasks = _read_tasks()
            task_name = f"批量任务{_get_next_batch_task_number(tasks)}"
            tasks[task_name] = {
                "prompt": f"批量文件: {os.path.basename(file_path)}",
                "model": model,
                "image_size": image_size,
                "aspect_ratio": aspect_ratio,
                "seed": 0,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "subtasks": [{"taskId": tid, "state": "waiting", "resultUrls": [], "downloaded": False} for tid in task_ids],
            }
            _write_tasks(tasks)

        credits = self._get_credits(final_key)
        status = f"批量提交完成 | {task_name} | 成功提交: {len(task_ids)}/{len(prompts)} | 剩余积分: {credits if credits is not None else 'N/A'}"
        if errors:
            status += f" | 提交失败数: {len(errors)}"
        return {"ui": {"string": [status]}, "result": (status,)}


class KieFolderBatchProcessCSV:
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
        image_sizes = ["默认", "1K", "2K", "4K"]
        return {
            "required": {
                "📁 图片文件夹": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "📄 CSV/Excel 路径": ("STRING", {"default": "", "placeholder": "填写 CSV/Excel 文件路径至此"}),
                "🧾 CSV列名": ("STRING", {"default": "prompt"}),
                "📤 输出文件夹": ("STRING", {"default": "", "placeholder": "输出文件夹路径（留空则自动创建）"}),
                "🤖 模型版本": (["nano-banana-pro", "nano-banana-2"], {"default": "nano-banana-pro"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
                "🖼️ 输出分辨率": (image_sizes, {"default": "默认"}),
                "🔁 单条Prompt执行次数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 备用图像2": ("IMAGE",),
                "🖼️ 备用图像3": ("IMAGE",),
                "🖼️ 备用图像4": ("IMAGE",),
                "📝 固定提示词(可选)": ("STRING", {"multiline": True, "default": "", "placeholder": "备用：如果不用CSV，则在此处填提示词跑所有图"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ 预览(最多15)", "📊 状态报告")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def __init__(self):
        self.base_url = "https://api.kie.ai"
        self.upload_base_url = (os.environ.get("KIE_UPLOAD_BASE_URL") or "https://kieai.redpandaai.co").strip()

    def _get_api_key(self, api_key: str) -> str:
        return (os.environ.get("KIE_API_KEY", "") or api_key).strip()

    def _file_to_data_url(self, file_path: str) -> str:
        import base64

        ext = os.path.splitext(file_path)[1].lower()
        mime = "image/png"
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".webp":
            mime = "image/webp"
        elif ext == ".bmp":
            mime = "image/bmp"

        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _upload_data_url(self, api_key: str, data_url: str, timeout_sec: int) -> str:
        file_name = f"mw_kie_{int(time.time() * 1000)}.png"
        upload_path = "comfyui/mw-kie-banana-pro"

        try:
            import base64
            from io import BytesIO

            import requests  # type: ignore

            d = (data_url or "").strip()
            mime = "image/png"
            b64 = d
            if d.startswith("data:") and "base64," in d:
                head, tail = d.split("base64,", 1)
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
            payload = {"base64Data": data_url, "uploadPath": upload_path, "fileName": file_name}
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

    def _get_credits(self, api_key: str) -> Optional[int]:
        try:
            result = _http_json(
                method="GET",
                url=f"{self.base_url}/api/v1/chat/credit",
                headers={"Authorization": f"Bearer {api_key}"},
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

    def _submit_one(
        self,
        api_key: str,
        prompt: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        image_urls: List[str],
        timeout_sec: int,
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "image_input": image_urls,
            "resolution": "2K" if image_size == "默认" else image_size,
            "output_format": "png",
            "aspect_ratio": aspect_ratio,
            "response_format": "url",
        }
        if aspect_ratio == "auto":
            input_payload.pop("aspect_ratio", None)
        payload = {"model": model, "input": input_payload}
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
        return str(task_id)

    def _poll_until_done(
        self, api_key: str, task_id: str, timeout_sec: int, poll_interval_sec: int
    ) -> Tuple[str, List[str]]:
        headers = {"Authorization": f"Bearer {api_key}"}
        start = time.time()
        while True:
            if time.time() - start > timeout_sec:
                return "timeout", []
            result = _http_json(
                method="GET",
                url=f"{self.base_url}/api/v1/jobs/recordInfo",
                headers=headers,
                payload=None,
                timeout=min(30, timeout_sec),
                params={"taskId": task_id},
            )
            data = result.get("data") or {}
            state = str(data.get("state") or "")
            s = state.lower()
            if s == "success":
                result_json = data.get("resultJson") or ""
                try:
                    parsed = json.loads(result_json) if result_json else {}
                except Exception:
                    parsed = {}
                urls = parsed.get("resultUrls") or []
                if not isinstance(urls, list):
                    urls = []
                return "success", [str(u) for u in urls]
            if s == "fail":
                return "fail", []
            time.sleep(max(1, int(poll_interval_sec)))

    def _pil_list_to_preview_tensor(self, pil_images: List["Image.Image"]) -> torch.Tensor:
        if not pil_images:
            return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        base_w, base_h = pil_images[0].size
        out_tensors: List[torch.Tensor] = []
        for p in pil_images[:15]:
            if p.size != (base_w, base_h):
                p = p.resize((base_w, base_h))
            out_tensors.append(pil2tensor(p))
        return torch.cat(out_tensors, dim=0) if out_tensors else torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        directory_path = _pick("📁 图片文件夹", "directory_path", default="")
        csv_file_path = _pick("📄 CSV/Excel 路径", "csv_file_path", default="")
        column_name = _pick("🧾 CSV列名", "column_name", default="prompt")
        output_dir = _pick("📤 输出文件夹", "output_dir", default="")
        model = _pick("🤖 模型版本", "model", default="nano-banana-pro")
        max_concurrent_files = int(_pick("⚙️ 同时处理文件数", "max_concurrent_files", default=3) or 3)
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        image_size = _pick("🖼️ 输出分辨率", "image_size", default="默认")
        executions_per_prompt = int(_pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1) or 1)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_2 = _pick("🖼️ 备用图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 备用图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 备用图像4", "image_4", default=None)
        fixed_prompt = _pick("📝 固定提示词(可选)", "fixed_prompt", default="") or ""

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            msg = "API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), msg)}

        timeout_sec = 600
        poll_interval_sec = 2

        if not directory_path or not os.path.exists(directory_path):
            msg = "文件夹路径不存在"
            return {"ui": {"string": [msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), msg)}

        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        image_files.sort()
        if not image_files:
            msg = "文件夹内无有效图片"
            return {"ui": {"string": [msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), msg)}

        base_prompts: List[str] = []
        source_mode = "Fixed"
        if csv_file_path and os.path.exists(csv_file_path):
            try:
                import pandas as pd  # type: ignore

                if csv_file_path.lower().endswith(".csv"):
                    df = pd.read_csv(csv_file_path, encoding="utf-8")
                elif csv_file_path.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(csv_file_path)
                else:
                    df = pd.DataFrame()
                if column_name in df.columns:
                    base_prompts = df[column_name].dropna().astype(str).tolist()
                    source_mode = "CSV"
            except Exception:
                base_prompts = []

        if not base_prompts:
            if not (fixed_prompt or "").strip():
                msg = "必须提供有效 CSV 或 固定提示词。"
                return {"ui": {"string": [msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), msg)}
            base_prompts = [fixed_prompt]
            source_mode = "Fixed Prompt"

        final_prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = folder_paths.get_output_directory()
        out_dir = (output_dir or "").strip()
        if out_dir:
            batch_output_dir = os.path.normpath(out_dir if os.path.isabs(out_dir) else os.path.join(base_output, out_dir))
        else:
            batch_output_dir = os.path.join(base_output, f"Kie_Batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)

        from PIL import Image
        import concurrent.futures

        extra_image_urls: List[str] = []
        try:
            for t in [image_2, image_3, image_4]:
                if t is None:
                    continue
                pil_list = tensor2pil(t)
                if not pil_list:
                    continue
                data_url = f"data:image/png;base64,{_image_to_base64_png(pil_list[0])}"
                extra_image_urls.append(self._upload_data_url(final_key, data_url, timeout_sec=int(timeout_sec)))
        except Exception as e:
            msg = f"参考图上传失败: {str(e)}"
            return {"ui": {"string": [msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), msg)}

        lock = threading.Lock()
        preview_pils: List[Image.Image] = []
        total_success = 0
        failed_list: List[str] = []

        def _process_one_file(file_idx: int, file_path: str) -> Tuple[int, List[Image.Image], Optional[str]]:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            try:
                data_url = self._file_to_data_url(file_path)
                uploaded_url = self._upload_data_url(final_key, data_url, timeout_sec=int(timeout_sec))
            except Exception as e:
                return 0, [], f"{base_filename} 上传失败: {str(e)}"

            file_success = 0
            file_preview: List[Image.Image] = []
            for p_idx, prompt_text in enumerate(final_prompts):
                try:
                    image_urls = [uploaded_url] + extra_image_urls
                    task_id = self._submit_one(
                        api_key=final_key,
                        prompt=prompt_text,
                        model=model,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio,
                        image_urls=image_urls,
                        timeout_sec=int(timeout_sec),
                    )
                    state, urls = self._poll_until_done(
                        api_key=final_key,
                        task_id=task_id,
                        timeout_sec=int(timeout_sec),
                        poll_interval_sec=int(poll_interval_sec),
                    )
                    if state != "success" or not urls:
                        continue
                    for j, u in enumerate(urls):
                        pil_img = _download_image(u, timeout=min(60, int(timeout_sec)))
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        prefix = f"Img{file_idx+1:03d}_{base_filename}"
                        suffix = f"P{p_idx+1:03d}" if len(urls) == 1 else f"P{p_idx+1:03d}_{j+1}"
                        save_name = f"{prefix}_{suffix}.png"
                        save_path = os.path.join(batch_output_dir, save_name)
                        counter = 1
                        while os.path.exists(save_path):
                            save_path = os.path.join(batch_output_dir, f"{prefix}_{suffix}_{counter}.png")
                            counter += 1
                        pil_img.save(save_path, "PNG", compress_level=4)
                        file_success += 1
                        if len(file_preview) < 15:
                            file_preview.append(pil_img)
                except Exception:
                    continue

            if file_success == 0:
                return 0, [], f"{base_filename} 未生成结果"
            return file_success, file_preview, None

        max_workers = min(max(1, int(max_concurrent_files)), 50)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_one_file, idx, fp): fp for idx, fp in enumerate(image_files)}
            for f in concurrent.futures.as_completed(futures):
                file_path = futures[f]
                base_name = os.path.basename(file_path)
                try:
                    cnt, pils, err = f.result()
                except Exception as e:
                    cnt, pils, err = 0, [], str(e)
                with lock:
                    if cnt > 0:
                        total_success += cnt
                        if pils and len(preview_pils) < 15:
                            remain = 15 - len(preview_pils)
                            preview_pils.extend(pils[:remain])
                    else:
                        failed_list.append(f"{base_name} -> {err or '失败'}")

        credits = self._get_credits(final_key)
        status_lines = [
            "✅ 文件夹批量处理完成",
            f"📁 输入: {os.path.basename(directory_path)}",
            f"📄 Prompts: {len(base_prompts)} ({source_mode})",
            f"🖼️ 图片数: {len(image_files)}",
            f"🧩 参考图: {len(extra_image_urls)}",
            f"✅ 生成成功: {total_success}",
            f"❌ 失败文件: {len(failed_list)}",
            f"💰 积分: {credits if credits is not None else 'N/A'}",
            f"💾 路径: {batch_output_dir}",
            "⚠️ 预览仅显示前 15 张，全量图请查看文件夹。",
        ]
        if failed_list:
            status_lines.append("\n--- 失败记录 (前5个) ---")
            for s in failed_list[:5]:
                status_lines.append(f"• {s}")
            if len(failed_list) > 5:
                status_lines.append(f"...以及其他 {len(failed_list) - 5} 个文件")

        status_report = "\n".join(status_lines)
        preview_tensor = self._pil_list_to_preview_tensor(preview_pils)
        return {"ui": {"string": [status_report]}, "result": (preview_tensor, status_report)}


class GrsaiNanoBananaBatchCSVExcelKie(KieFolderBatchProcessCSV):
    CATEGORY = "🤖MINGWEI-API/MW-nano banana"

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
        image_sizes = ["默认", "1K", "2K", "4K"]
        return {
            "required": {
                "📄 文件路径": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此（留空则使用下方提示词）"}),
                "📝 提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "留空则从文件读取；文件路径为空则使用这里"}),
                "🧾 列名": ("STRING", {"default": "prompt", "placeholder": "文件模式：CSV/Excel 的列名，如 prompt"}),
                "🧩 提示词前缀": ("STRING", {"multiline": True, "default": ""}),
                "🤖 模型版本": (["nano-banana-pro", "nano-banana-2"], {"default": "nano-banana-pro"}),
                "⚙️ 并发数": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "🔢 最大数量": ("INT", {"default": 50, "min": 1, "max": 100}),
                "📐 宽高比": (aspect_ratios, {"default": "auto"}),
                "🖼️ 输出分辨率": (image_sizes, {"default": "默认"}),
                "🔁 执行次数/条": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🔑 API 密钥": ("STRING", {"default": "", "placeholder": "优先使用环境变量 KIE_API_KEY；此处用于临时填写"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_batch", "status")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error_result(self, msg: str):
        empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"string": [msg]}, "result": (empty, msg)}

    def _pil_list_to_tensor_batch(self, pil_images: List["Image.Image"]) -> torch.Tensor:
        if not pil_images:
            return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        base = pil_images[0].convert("RGB")
        base_w, base_h = base.size
        out_tensors: List[torch.Tensor] = [pil2tensor(base)]
        for p in pil_images[1:]:
            p2 = p.convert("RGB")
            if p2.size != (base_w, base_h):
                p2 = p2.resize((base_w, base_h))
            out_tensors.append(pil2tensor(p2))
        return torch.cat(out_tensors, dim=0)

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.get(k)
            return default

        file_path = _pick("📄 文件路径", "file_path", default="")
        direct_prompt = _pick("📝 提示词", "direct_prompt", "prompt", default="")
        column_name = _pick("🧾 列名", "column_name", default="prompt")
        prompt_prefix = _pick("🧩 提示词前缀", "prompt_prefix", default="")
        model = _pick("🤖 模型版本", "model", default="nano-banana-pro")
        concurrency = int(_pick("⚙️ 并发数", "concurrency", default=10) or 10)
        max_count = int(_pick("🔢 最大数量", "max_count", default=50) or 50)
        aspect_ratio = _pick("📐 宽高比", "aspect_ratio", default="auto")
        image_size = _pick("🖼️ 输出分辨率", "image_size", default="默认")
        executions_per_prompt = int(_pick("🔁 执行次数/条", "executions_per_prompt", default=1) or 1)
        api_key = _pick("🔑 API 密钥", "api_key", default="") or ""
        image_1 = _pick("🖼️ 图像1", "image_1", default=None)
        image_2 = _pick("🖼️ 图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 图像4", "image_4", default=None)

        final_key = self._get_api_key(str(api_key))
        if not final_key:
            return self._error_result("API Key 不能为空。")

        file_path = (file_path or "").strip().strip('"').strip("'")
        direct_prompt = (direct_prompt or "").strip()

        if file_path and os.path.exists(file_path):
            try:
                import pandas as pd  # type: ignore

                if file_path.lower().endswith(".csv"):
                    df = pd.read_csv(file_path, encoding="utf-8")
                elif file_path.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path)
                else:
                    return self._error_result("仅支持 .csv, .xls, .xlsx 文件。")
            except Exception as e:
                return self._error_result(f"读取文件失败: {str(e)}")

            if column_name not in df.columns:
                return self._error_result(f"列 '{column_name}' 不存在。")

            base_prompts = [
                f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[: int(max_count)]
            ]
            if not base_prompts:
                return self._error_result(f"列 '{column_name}' 中未找到有效 prompt。")
        else:
            if not direct_prompt:
                return self._error_result("文件路径为空时，提示词不能为空。")
            base_prompts = [f"{prompt_prefix}{direct_prompt}"]

        prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]

        pil_images = []
        for t in [image_1, image_2, image_3, image_4]:
            if t is not None:
                pil_images.extend(tensor2pil(t))

        image_urls: List[str] = []
        if pil_images:
            try:
                for p in pil_images:
                    data_url = f"data:image/png;base64,{_image_to_base64_png(p)}"
                    image_urls.append(self._upload_data_url(final_key, data_url, timeout_sec=600))
            except Exception as e:
                return self._error_result(f"图片上传失败: {str(e)}")

        import concurrent.futures

        all_pils: List["Image.Image"] = []
        errors: List[str] = []

        def _run_one(p: str):
            task_id = self._submit_one(
                api_key=final_key,
                prompt=p,
                model=model,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
                image_urls=image_urls,
                timeout_sec=600,
            )
            state, urls = self._poll_until_done(final_key, task_id, timeout_sec=900, poll_interval_sec=2)
            if state != "success" or not urls:
                raise ValueError(f"{state or 'fail'}: {task_id}")
            return _download_image(urls[0], timeout=60)

        max_workers = min(max(1, int(concurrency)), 50)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one, p) for p in prompts]
            for f in concurrent.futures.as_completed(futures):
                try:
                    pil = f.result()
                    if pil is not None:
                        all_pils.append(pil)
                except Exception as e:
                    errors.append(str(e))

        if not all_pils:
            msg = f"所有图像生成均失败: {'; '.join(errors[:5])}"
            return self._error_result(msg)

        credits = self._get_credits(final_key)
        if int(executions_per_prompt) == 1:
            task_info = f"{len(prompts)}个总任务"
        else:
            task_info = f"{len(base_prompts)}个Prompt x {int(executions_per_prompt)}次 = {len(prompts)}个总任务"
        status = (
            f"批量完成 | {task_info} | 成功: {len(all_pils)} | 失败: {len(errors)} | 积分: {credits if credits is not None else 'N/A'}"
        )

        out_tensor = self._pil_list_to_tensor_batch(all_pils)
        return {"ui": {"string": [status]}, "result": (out_tensor, status)}


NODE_CLASS_MAPPINGS = {
    "NanoBananaProAsyncSubmit": NanoBananaProAsyncSubmit,
    "NanoBananaProAsyncQuery": NanoBananaProAsyncQuery,
    "NanoBananaProAsyncDownload": NanoBananaProAsyncDownload,
    "KieLLMVLMWriter": KieLLMVLMWriter,
    "ZhenzhenLLMVLMWriter": ZhenzhenLLMVLMWriter,
    "NanoBananaProAsyncBatchSubmit": NanoBananaProAsyncBatchSubmit,
    "KieFolderBatchProcessCSV": KieFolderBatchProcessCSV,
    "Gemini3MultimodalChatKie": Gemini3MultimodalChatKie,
    "Gemini3MultimodalChatZhenzhen": Gemini3MultimodalChatZhenzhen,
    "GrsaiNanoBananaBatchCSVExcelKie": GrsaiNanoBananaBatchCSVExcelKie,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProAsyncSubmit": "🍌 Nano Banana 异步提交-kie",
    "NanoBananaProAsyncQuery": "🍌 Nano Banana 异步查询-kie",
    "NanoBananaProAsyncDownload": "🍌 Nano Banana 异步下载-kie",
    "KieLLMVLMWriter": "✍️ Kie LLM/VLM Writer-kie",
    "ZhenzhenLLMVLMWriter": "✍️ zhenzhen LLM/VLM Writer",
    "NanoBananaProAsyncBatchSubmit": "🍌 Nano Banana 异步批量提交-kie (CSV/Excel)",
    "KieFolderBatchProcessCSV": "🍌📂 Kie 文件夹批量处理(CSV)",
    "Gemini3MultimodalChatKie": "💎 Gemini 3 多模态对话 (kie)",
    "Gemini3MultimodalChatZhenzhen": "💎 Gemini 3 多模态对话 (zhenzhen)",
    "GrsaiNanoBananaBatchCSVExcelKie": "🍌 Nano Banana Batch (CSV/Excel)-kie",
}
