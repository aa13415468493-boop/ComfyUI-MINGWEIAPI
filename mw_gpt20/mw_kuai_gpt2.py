import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product

from PIL import Image

from .mw_kie_gpt2 import (
    _empty_image_tensor,
    _get_requests_session,
    _http_download_bytes,
    _http_json,
    _images_to_batch_tensor,
    _is_nonempty_string,
    _pick_from_kwargs,
    _read_local_config_first_nonempty,
    _resolve_batch_output_dir,
    _save_result_image,
    _blt_decode_item_to_pil,
    _blt_first_tensor_to_png_bytes,
    _blt_local_file_to_png_bytes,
    _blt_make_png_tuple,
    _blt_request_file_from_png_bytes,
    _list_local_image_files,
    tensor2pil,
)


KUAI_BASE_URL = "https://api.kuai.host"
KUAI_IMAGE_GENERATIONS_PATH = "/v1/images/generations"
KUAI_IMAGE_EDITS_PATH = "/v1/images/edits"


def _resolve_kuai_api_key(widget_value):
    for env_key in ("KUAI_API_KEY", "KUAI_HOST_API_KEY", "KUAIHOST_API_KEY"):
        env_value = os.environ.get(env_key, "")
        if _is_nonempty_string(env_value):
            return env_value.strip()

    config_value = _read_local_config_first_nonempty("kuai_api_key", "kuai_host_api_key", "api_key")
    if config_value:
        return config_value

    if _is_nonempty_string(widget_value):
        return widget_value.strip()

    return ""


def _normalize_base_url(base_url):
    value = str(base_url or KUAI_BASE_URL).strip() or KUAI_BASE_URL
    return value.rstrip("/")


def _kuai_headers_json(api_key):
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _kuai_headers_bearer(api_key):
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Accept": "application/json",
    }


def _split_image_urls(text):
    if not _is_nonempty_string(text):
        return []
    values = re.split(r"[\s,，]+", text.strip())
    urls = []
    for value in values:
        clean_value = value.strip()
        if clean_value.startswith("http") and clean_value not in urls:
            urls.append(clean_value)
    return urls


def _collect_input_image_urls(image_url_text):
    return _split_image_urls(image_url_text)


def _collect_input_image_files(kwargs):
    files = []
    for index in range(1, 17):
        fallback_key = "image" if index == 1 else "image_{}".format(index)
        value = _pick_from_kwargs(
            kwargs,
            "🖼️ 图像{}".format(index),
            "图像{}".format(index),
            "image{}".format(index),
            fallback_key,
            default=None,
        )
        if value is None:
            continue
        for batch_index, pil_image in enumerate(tensor2pil(value), start=1):
            files.append(
                (
                    "image",
                    _blt_make_png_tuple(
                        pil_image.convert("RGB"),
                        "kuai_gpt_image2_{}_{}.png".format(index, batch_index),
                    ),
                )
            )
    return files


def _url_to_image_file(url, index):
    image_bytes = _http_download_bytes(url, timeout=300)
    with Image.open(BytesIO(image_bytes)) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        buffer.seek(0)
    return ("image", ("kuai_gpt_image2_url_{}.png".format(index), buffer, "image/png"))


def _urls_to_image_files(urls):
    return [_url_to_image_file(url, index) for index, url in enumerate(urls, start=1)]


def _rewind_files(files):
    for _field_name, file_tuple in files:
        try:
            file_tuple[1].seek(0)
        except Exception:
            pass


def _kuai_post_multipart(api_key, url, data, files, timeout=900):
    session = _get_requests_session()
    if session is None:
        raise ValueError("Kuai GPT2.0 节点依赖 requests 模块。")

    last_error = None
    for retry_index in range(2):
        try:
            _rewind_files(files)
            response = session.post(
                url,
                headers=_kuai_headers_bearer(api_key),
                data=data,
                files=files,
                timeout=int(timeout),
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            response_text = getattr(response, "text", "")
            if status_code < 200 or status_code >= 300:
                raise ValueError("HTTP {}: {}".format(status_code, response_text))
            try:
                return response.json()
            except Exception:
                return json.loads(response_text)
        except Exception as e:
            last_error = e
            if retry_index == 0:
                time.sleep(1.0)
                continue
    if last_error is not None:
        raise last_error
    raise ValueError("Kuai GPT2.0 multipart 请求失败")


def _collect_result_items(payload):
    if not isinstance(payload, dict):
        return []

    direct_candidates = []
    for key in ("url", "image_url", "imageUrl", "b64_json"):
        if _is_nonempty_string(payload.get(key)):
            direct_candidates.append(payload)
            break
    if direct_candidates:
        return direct_candidates

    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("url", "image_url", "imageUrl", "b64_json"):
            if _is_nonempty_string(data.get(key)):
                return [data]
        for key in ("data", "images", "output", "outputs"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]

    for key in ("images", "output", "outputs"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]

    return []


def _collect_result_urls(items):
    urls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("url") or item.get("image_url") or item.get("imageUrl")
        if _is_nonempty_string(value):
            clean_value = value.strip()
            if clean_value not in urls:
                urls.append(clean_value)
    return urls


def _items_to_pil_images(items):
    pil_images = []
    for item in items:
        pil_image = _blt_decode_item_to_pil(item, max_retries=3, initial_timeout=900)
        if pil_image is not None:
            pil_images.append(pil_image)
    return pil_images


def _extract_task_id(payload):
    if not isinstance(payload, dict):
        return ""
    for key in ("id", "task_id", "taskId"):
        value = payload.get(key)
        if _is_nonempty_string(value):
            return value.strip()
    created = payload.get("created")
    return str(created) if created is not None else ""


def _build_generation_payload(model, prompt, size, output_format, quality, n, image_urls=None):
    payload = {
        "model": model,
        "prompt": (prompt or "").strip(),
        "n": int(n),
        "size": str(size or "auto").strip() or "auto",
        "quality": str(quality or "auto").strip() or "auto",
        "format": str(output_format or "png").strip() or "png",
    }
    if image_urls:
        payload["image"] = image_urls
    return payload


def _build_edit_data(model, prompt, size, output_format, quality, n):
    data = {
        "model": model,
        "prompt": (prompt or "").strip(),
        "n": str(int(n)),
        "size": str(size or "auto").strip() or "auto",
    }
    if str(quality or "auto") != "auto":
        data["quality"] = str(quality).strip()
    if str(output_format or "png") != "png":
        data["format"] = str(output_format).strip()
    return data


def _normalize_resolution(resolution):
    value = str(resolution or "1K").strip().upper()
    if value not in ("1K", "2K", "4K"):
        raise ValueError("Kuai GPT2.0 不支持的分辨率: {}".format(resolution))
    return value.lower()


def _normalize_size_text(size):
    value = str(size or "").strip().lower().replace("×", "x")
    return value


def _validate_kuai_size(size):
    value = _normalize_size_text(size)
    if value == "auto":
        return value
    if "x" not in value:
        raise ValueError("Kuai GPT2.0 size 必须是 1024x1024 这种格式。")
    width_text, height_text = value.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except Exception:
        raise ValueError("Kuai GPT2.0 size 宽高必须是整数。")

    if width <= 0 or height <= 0:
        raise ValueError("Kuai GPT2.0 size 宽高必须大于 0。")
    if max(width, height) > 3840:
        raise ValueError("Kuai GPT2.0 size 最大边长不能超过 3840px。")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("Kuai GPT2.0 size 宽高都必须是 16px 的倍数。")
    if max(width, height) / float(min(width, height)) > 3.0:
        raise ValueError("Kuai GPT2.0 size 长边/短边比例不能超过 3:1。")

    total_pixels = width * height
    if total_pixels < 655360 or total_pixels > 8294400:
        raise ValueError("Kuai GPT2.0 size 总像素必须在 655360 到 8294400 之间。")
    return "{}x{}".format(width, height)


def _resolve_size_info_from_ratio(aspect_ratio, resolution="1K"):
    value = str(aspect_ratio or "1:1").strip() or "1:1"
    if value == "auto" or "x" in value or "×" in value:
        return _validate_kuai_size(value), "自定义 size 已通过 Kuai 限制规则校验。"
    normalized_resolution = _normalize_resolution(resolution)
    size_map = {
        ("1:1", "1k"): "1024x1024",
        ("1:1", "2k"): "2048x2048",
        ("1:1", "4k"): "2880x2880",
        ("3:2", "1k"): "1248x832",
        ("3:2", "2k"): "2496x1664",
        ("3:2", "4k"): "3504x2336",
        ("2:3", "1k"): "832x1248",
        ("2:3", "2k"): "1664x2496",
        ("2:3", "4k"): "2336x3504",
        ("4:3", "1k"): "1152x864",
        ("4:3", "2k"): "2304x1728",
        ("4:3", "4k"): "3264x2448",
        ("3:4", "1k"): "864x1152",
        ("3:4", "2k"): "1728x2304",
        ("3:4", "4k"): "2448x3264",
        ("5:4", "1k"): "1120x896",
        ("5:4", "2k"): "2240x1792",
        ("5:4", "4k"): "3200x2560",
        ("4:5", "1k"): "896x1120",
        ("4:5", "2k"): "1792x2240",
        ("4:5", "4k"): "2560x3200",
        ("16:9", "1k"): "1280x720",
        ("16:9", "2k"): "2560x1440",
        ("16:9", "4k"): "3840x2160",
        ("9:16", "1k"): "720x1280",
        ("9:16", "2k"): "1440x2560",
        ("9:16", "4k"): "2160x3840",
        ("2:1", "1k"): "2048x1024",
        ("2:1", "2k"): "2688x1344",
        ("2:1", "4k"): "3840x1920",
        ("1:2", "1k"): "1024x2048",
        ("1:2", "2k"): "1344x2688",
        ("1:2", "4k"): "1920x3840",
        ("21:9", "1k"): "1456x624",
        ("21:9", "2k"): "3024x1296",
        ("21:9", "4k"): "3696x1584",
        ("9:21", "1k"): "624x1456",
        ("9:21", "2k"): "1296x3024",
        ("9:21", "4k"): "1584x3696",
    }
    size = size_map.get((value, normalized_resolution))
    if size is None:
        raise ValueError("Kuai GPT2.0 不支持的图像比例: {}".format(value))
    return _validate_kuai_size(size), "已按图像比例与分辨率生成自定义 size，并通过 Kuai 限制规则校验。"


def _resolve_size_from_ratio(aspect_ratio, resolution="1K"):
    size, _note = _resolve_size_info_from_ratio(aspect_ratio, resolution)
    return size


class MWKuaiGPTImage2:
    _MODEL_CHOICES = ["gpt-image-2", "gpt-image-2-all"]
    _ASPECT_RATIO_CHOICES = [
        "1:1",
        "3:2",
        "2:3",
        "4:3",
        "3:4",
        "5:4",
        "4:5",
        "16:9",
        "9:16",
        "2:1",
        "1:2",
        "21:9",
        "9:21",
    ]
    _RESOLUTION_CHOICES = ["1K", "2K", "4K"]
    _FORMAT_CHOICES = ["png", "jpeg", "webp"]
    _QUALITY_CHOICES = ["auto", "low", "medium", "high"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🧩 生成模式": (["自动", "文生图", "图像编辑"], {"default": "自动"}),
                "🤖 模型": (cls._MODEL_CHOICES, {"default": "gpt-image-2"}),
                "📝 提示词": ("STRING", {"multiline": True, "default": ""}),
                "📐 图像比例": (cls._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "📺 分辨率": (cls._RESOLUTION_CHOICES, {"default": "1K"}),
                "🗂️ 输出格式": (cls._FORMAT_CHOICES, {"default": "png"}),
                "✨ 质量": (cls._QUALITY_CHOICES, {"default": "auto"}),
                "🖼️ 出图数量": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🔑 API密钥": ("STRING", {"default": ""}),
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
                "🖼️ 图像15": ("IMAGE",),
                "🖼️ 图像16": ("IMAGE",),
                "🔗 图片URL": ("STRING", {"multiline": True, "default": ""}),
                "🌐 接口地址": ("STRING", {"default": KUAI_BASE_URL}),
                "⏱️ 超时(秒)": ("INT", {"default": 900, "min": 30, "max": 3600, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "🔗 图片地址", "🧾 响应信息", "🆔 任务ID")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kuai"

    def generate(self, **kwargs):
        prompt = _pick_from_kwargs(kwargs, "📝 提示词", "prompt", default="")
        if not _is_nonempty_string(prompt):
            raise ValueError("Kuai GPT2.0 图像生成需要填写提示词。")
        if len(prompt.strip()) > 1000:
            raise ValueError("Kuai GPT2.0 提示词最大长度为 1000 个字符。")

        mode = _pick_from_kwargs(kwargs, "🧩 生成模式", "mode", default="自动")
        model = _pick_from_kwargs(kwargs, "🤖 模型", "model", default="gpt-image-2")
        aspect_ratio = _pick_from_kwargs(kwargs, "📐 图像比例", "aspect_ratio", "📐 图片尺寸", "size", default="1:1")
        resolution = _pick_from_kwargs(kwargs, "📺 分辨率", "resolution", default="1K")
        size, size_note = _resolve_size_info_from_ratio(aspect_ratio, resolution)
        output_format = _pick_from_kwargs(kwargs, "🗂️ 输出格式", "format", default="png")
        quality = _pick_from_kwargs(kwargs, "✨ 质量", "quality", default="auto")
        n = int(_pick_from_kwargs(kwargs, "🖼️ 出图数量", "n", default=1) or 1)
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="")
        image_url_text = _pick_from_kwargs(kwargs, "🔗 图片URL", "image_url", default="") or ""
        base_url = _normalize_base_url(_pick_from_kwargs(kwargs, "🌐 接口地址", "base_url", default=KUAI_BASE_URL))
        timeout = int(_pick_from_kwargs(kwargs, "⏱️ 超时(秒)", "timeout", default=900) or 900)

        if model not in self._MODEL_CHOICES:
            raise ValueError("Kuai GPT2.0 不支持的模型: {}".format(model))
        if str(output_format) not in self._FORMAT_CHOICES:
            raise ValueError("Kuai GPT2.0 不支持的输出格式: {}".format(output_format))
        if str(quality) not in self._QUALITY_CHOICES:
            raise ValueError("Kuai GPT2.0 不支持的质量: {}".format(quality))
        if n < 1 or n > 10:
            raise ValueError("Kuai GPT2.0 出图数量必须介于 1 到 10。")

        resolved_api_key = _resolve_kuai_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 Kuai API Key，请优先使用环境变量 KUAI_API_KEY，或在节点中填写。")

        image_urls = _collect_input_image_urls(image_url_text)
        image_files = _collect_input_image_files(kwargs)
        has_images = bool(image_urls or image_files)

        if mode == "图像编辑" and not has_images:
            raise ValueError("图像编辑模式需要提供图片输入或图片URL。")
        use_edit_mode = has_images if mode == "自动" else mode == "图像编辑"

        if not use_edit_mode:
            request_payload = _build_generation_payload(model, prompt, size, output_format, quality, n)
            request_info = {
                "endpoint": base_url + KUAI_IMAGE_GENERATIONS_PATH,
                "mode": "文生图",
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "submitted_size": size,
                "size_note": size_note,
                "payload": request_payload,
            }
            response = _http_json(
                "POST",
                base_url + KUAI_IMAGE_GENERATIONS_PATH,
                headers=_kuai_headers_json(resolved_api_key),
                json_body=request_payload,
                timeout=timeout,
            )
        elif model == "gpt-image-2-all" and image_urls and not image_files:
            request_payload = _build_generation_payload(model, prompt, size, output_format, quality, n, image_urls=image_urls)
            request_info = {
                "endpoint": base_url + KUAI_IMAGE_GENERATIONS_PATH,
                "mode": "带图生成",
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "submitted_size": size,
                "size_note": size_note,
                "image_url_count": len(image_urls),
                "payload": request_payload,
            }
            response = _http_json(
                "POST",
                base_url + KUAI_IMAGE_GENERATIONS_PATH,
                headers=_kuai_headers_json(resolved_api_key),
                json_body=request_payload,
                timeout=timeout,
            )
        else:
            request_files = list(image_files)
            if image_urls:
                request_files.extend(_urls_to_image_files(image_urls))
            if not request_files:
                raise ValueError("图像编辑模式没有解析到可上传的图片。")
            request_data = _build_edit_data(model, prompt, size, output_format, quality, n)
            request_info = {
                "endpoint": base_url + KUAI_IMAGE_EDITS_PATH,
                "mode": "图像编辑",
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "submitted_size": size,
                "size_note": size_note,
                "image_count": len(request_files),
                "form_data": request_data,
            }
            response = _kuai_post_multipart(
                resolved_api_key,
                base_url + KUAI_IMAGE_EDITS_PATH,
                request_data,
                request_files,
                timeout=timeout,
            )

        items = _collect_result_items(response)
        pil_images = _items_to_pil_images(items)
        if not pil_images:
            raise ValueError("Kuai GPT2.0 未返回可解析的图片结果: {}".format(response))

        image_url_result = "\n".join(_collect_result_urls(items))
        response_text = json.dumps({"request": request_info, "response": response}, ensure_ascii=False, indent=2)
        return (_images_to_batch_tensor(pil_images), image_url_result, response_text, _extract_task_id(response))


def _kuai_make_request_data(prompt, model, size):
    return _build_edit_data(model, prompt, size, "png", "auto", 1)


def _kuai_response_to_pil_images(response):
    items = _collect_result_items(response)
    return _items_to_pil_images(items)


def _kuai_tensor_reference(image_tensor, slot_label, slot_index):
    return [
        {
            "slot": slot_label,
            "slot_index": int(slot_index),
            "file_name": slot_label,
            "png_bytes": _blt_first_tensor_to_png_bytes(image_tensor),
        }
    ]


def _kuai_folder_references(folder_path, slot_label, slot_index):
    if not folder_path:
        return []
    if not os.path.isdir(folder_path):
        raise ValueError("{} 不存在".format(slot_label))
    files = _list_local_image_files(folder_path)
    if not files:
        raise ValueError("{} 内无有效图片".format(slot_label))
    references = []
    for file_path in files:
        references.append(
            {
                "slot": slot_label,
                "slot_index": int(slot_index),
                "file_name": os.path.splitext(os.path.basename(file_path))[0],
                "png_bytes": _blt_local_file_to_png_bytes(file_path),
            }
        )
    return references


def _kuai_save_pil_results(pil_images, output_dir, prefix, run_index):
    saved_paths = []
    for image_index, pil_image in enumerate(pil_images, start=1):
        if len(pil_images) == 1:
            suffix = "P{:03d}".format(run_index)
        else:
            suffix = "P{:03d}_{}".format(run_index, image_index)
        saved_paths.append(_save_result_image(pil_image, output_dir, prefix, suffix))
    return saved_paths


def _kuai_batch_status_preview(preview_pils):
    return _images_to_batch_tensor(preview_pils[:15]) if preview_pils else _empty_image_tensor()


class MWKuaiGPTImage2FolderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📁 图片文件夹": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "📤 输出文件夹": ("STRING", {"default": "", "placeholder": "输出文件夹路径（留空则自动创建）"}),
                "🤖 模型": (MWKuaiGPTImage2._MODEL_CHOICES, {"default": "gpt-image-2"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (MWKuaiGPTImage2._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "📺 分辨率": (MWKuaiGPTImage2._RESOLUTION_CHOICES, {"default": "1K"}),
                "🔁 单条Prompt执行次数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            },
            "optional": {
                "🖼️ 备用图像2": ("IMAGE",),
                "🖼️ 备用图像3": ("IMAGE",),
                "🖼️ 备用图像4": ("IMAGE",),
                "📝 固定提示词(必填)": ("STRING", {"multiline": True, "default": "", "placeholder": "填写统一提示词跑完整个文件夹"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ 预览(最多15)", "📊 状态报告")
    FUNCTION = "execute"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kuai"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            return _pick_from_kwargs(kwargs, *keys, default=default)

        directory_path = (_pick("📁 图片文件夹", "directory_path", default="") or "").strip()
        output_dir = (_pick("📤 输出文件夹", "output_dir", default="") or "").strip()
        model = _pick("🤖 模型", "model", default="gpt-image-2")
        max_concurrent_files = max(1, min(int(_pick("⚙️ 同时处理文件数", "max_concurrent_files", default=3) or 3), 32))
        aspect_ratio = _pick("📐 图像比例", "aspect_ratio", default="1:1")
        resolution = _pick("📺 分辨率", "resolution", default="1K")
        executions_per_prompt = max(1, min(int(_pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1) or 1), 10))
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        backup_image_2 = _pick("🖼️ 备用图像2", "backup_image_2", "image_2", default=None)
        backup_image_3 = _pick("🖼️ 备用图像3", "backup_image_3", "image_3", default=None)
        backup_image_4 = _pick("🖼️ 备用图像4", "backup_image_4", "image_4", default=None)

        resolved_api_key = _resolve_kuai_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少 Kuai API Key，请优先使用环境变量 KUAI_API_KEY，或在节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not directory_path or not os.path.isdir(directory_path):
            msg = "图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if model not in MWKuaiGPTImage2._MODEL_CHOICES:
            msg = "Kuai GPT2.0 不支持的模型: {}".format(model)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            size, size_note = _resolve_size_info_from_ratio(aspect_ratio, resolution)
        except Exception as e:
            msg = str(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        image_files = _list_local_image_files(directory_path)
        if not image_files:
            msg = "文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            backup_entries = []
            for slot_index, image_tensor in ((2, backup_image_2), (3, backup_image_3), (4, backup_image_4)):
                if image_tensor is None:
                    continue
                backup_entries.append(
                    {
                        "slot_index": int(slot_index),
                        "file_name": "backup_{}".format(slot_index),
                        "png_bytes": _blt_first_tensor_to_png_bytes(image_tensor),
                    }
                )
        except Exception as e:
            msg = "备用图处理失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "Kuai_GPTImage2_Batch")
        endpoint = KUAI_BASE_URL + KUAI_IMAGE_EDITS_PATH
        request_data = _kuai_make_request_data(fixed_prompt, model, size)
        preview_pils = []
        failed_list = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()

        def _process_one_file(file_index, file_path):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_png_bytes = _blt_local_file_to_png_bytes(file_path)
            except Exception as e:
                return 0, 0, [], "{} 读取失败: {}".format(base_name, e)

            file_success = 0
            file_preview = []
            task_count = 0
            for run_index in range(1, executions_per_prompt + 1):
                request_files = [
                    _blt_request_file_from_png_bytes(
                        "primary_{}_{}.png".format(file_index + 1, base_name),
                        primary_png_bytes,
                        "image",
                    )
                ]
                for backup_entry in backup_entries:
                    request_files.append(
                        _blt_request_file_from_png_bytes(
                            "backup{}_{}.png".format(backup_entry["slot_index"], backup_entry["file_name"]),
                            backup_entry["png_bytes"],
                            "image",
                        )
                    )

                task_count += 1
                try:
                    response = _kuai_post_multipart(
                        resolved_api_key,
                        endpoint,
                        request_data,
                        request_files,
                        timeout=900,
                    )
                    pil_images = _kuai_response_to_pil_images(response)
                    if not pil_images:
                        return file_success, task_count, file_preview, "{} 未解析到图片结果".format(base_name)
                    prefix = "Img{:03d}_{}".format(file_index + 1, base_name)
                    _kuai_save_pil_results(pil_images, batch_output_dir, prefix, run_index)
                    file_success += len(pil_images)
                    if len(file_preview) < 15:
                        file_preview.extend(pil_images[: 15 - len(file_preview)])
                except Exception as e:
                    return file_success, task_count, file_preview, "{} 生成失败: {}".format(base_name, e)

            if file_success == 0:
                return 0, task_count, [], "{} 未生成结果".format(base_name)
            return file_success, task_count, file_preview, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {
                executor.submit(_process_one_file, index, file_path): file_path
                for index, file_path in enumerate(image_files)
            }
            for future in as_completed(future_map):
                file_name = os.path.basename(future_map[future])
                try:
                    success_count, task_count, file_preview, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, error_msg = 0, 0, [], str(e)
                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    if file_preview and len(preview_pils) < 15:
                        preview_pils.extend(file_preview[: 15 - len(preview_pils)])
                    if error_msg:
                        failed_list.append("{} -> {}".format(file_name, error_msg))

        status_lines = [
            "✅ kuai GPT2.0 文件夹批量处理完成",
            "📁 输入文件夹: {}".format(os.path.basename(directory_path.rstrip("\\/")) or directory_path),
            "🖼️ 输入图片数: {}".format(len(image_files)),
            "🧩 备用图数量: {}".format(len(backup_entries)),
            "📐 图像比例: {}".format(aspect_ratio),
            "📺 分辨率: {}".format(resolution),
            "📏 实际size: {}".format(size),
            "📝 size说明: {}".format(size_note),
            "🔁 单条Prompt执行次数: {}".format(int(executions_per_prompt)),
            "⚙️ 同时处理文件数: {}".format(int(max_concurrent_files)),
            "📦 总任务数: {}".format(int(total_tasks)),
            "✅ 成功输出数: {}".format(int(total_success)),
            "❌ 失败文件数: {}".format(len(failed_list)),
            "💾 输出目录: {}".format(batch_output_dir),
            "⚠️ 预览仅显示前 15 张，全量图片请查看输出文件夹。",
        ]
        if failed_list:
            status_lines.append("")
            status_lines.append("--- 失败记录(前5个) ---")
            for item in failed_list[:5]:
                status_lines.append("• {}".format(item))
            if len(failed_list) > 5:
                status_lines.append("...以及其他 {} 个失败项".format(len(failed_list) - 5))

        status_report = "\n".join(status_lines)
        return {"ui": {"string": [status_report]}, "result": (_kuai_batch_status_preview(preview_pils), status_report)}


class MWKuaiGPTImage2MultiReferenceBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📁 主图片文件夹": ("STRING", {"default": "", "placeholder": "输入主图片文件夹路径"}),
                "📁 参考图文件夹1": ("STRING", {"default": "", "placeholder": "输入参考图文件夹1路径"}),
                "📁 参考图文件夹2": ("STRING", {"default": "", "placeholder": "可选：输入参考图文件夹2路径；若参考图像2有输入则此项失效"}),
                "📁 参考图文件夹3": ("STRING", {"default": "", "placeholder": "可选：输入参考图文件夹3路径；若参考图像3有输入则此项失效"}),
                "📁 参考图文件夹4": ("STRING", {"default": "", "placeholder": "可选：输入参考图文件夹4路径；若参考图像4有输入则此项失效"}),
                "📤 输出文件夹": ("STRING", {"default": "", "placeholder": "输出文件夹路径（留空则自动创建）"}),
                "🤖 模型": (MWKuaiGPTImage2._MODEL_CHOICES, {"default": "gpt-image-2"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (MWKuaiGPTImage2._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "📺 分辨率": (MWKuaiGPTImage2._RESOLUTION_CHOICES, {"default": "1K"}),
                "🔁 单条Prompt执行次数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            },
            "optional": {
                "📝 固定提示词(必填)": ("STRING", {"multiline": True, "default": "", "placeholder": "填写统一提示词跑主图文件夹和参考图的全部组合"}),
                "🖼️ 参考图像1": ("IMAGE",),
                "🖼️ 参考图像2": ("IMAGE",),
                "🖼️ 参考图像3": ("IMAGE",),
                "🖼️ 参考图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ 预览(最多15)", "📊 状态报告")
    FUNCTION = "execute"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kuai"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            return _pick_from_kwargs(kwargs, *keys, default=default)

        primary_directory = (_pick("📁 主图片文件夹", "primary_directory", default="") or "").strip()
        reference_directory_1 = (_pick("📁 参考图文件夹1", "reference_directory_1", default="") or "").strip()
        reference_directory_2 = (_pick("📁 参考图文件夹2", "reference_directory_2", default="") or "").strip()
        reference_directory_3 = (_pick("📁 参考图文件夹3", "reference_directory_3", default="") or "").strip()
        reference_directory_4 = (_pick("📁 参考图文件夹4", "reference_directory_4", default="") or "").strip()
        output_dir = (_pick("📤 输出文件夹", "output_dir", default="") or "").strip()
        model = _pick("🤖 模型", "model", default="gpt-image-2")
        max_concurrent_files = max(1, min(int(_pick("⚙️ 同时处理文件数", "max_concurrent_files", default=3) or 3), 32))
        aspect_ratio = _pick("📐 图像比例", "aspect_ratio", default="1:1")
        resolution = _pick("📺 分辨率", "resolution", default="1K")
        executions_per_prompt = max(1, min(int(_pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1) or 1), 10))
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        reference_image_1 = _pick("🖼️ 参考图像1", "reference_image_1", default=None)
        reference_image_2 = _pick("🖼️ 参考图像2", "reference_image_2", default=None)
        reference_image_3 = _pick("🖼️ 参考图像3", "reference_image_3", default=None)
        reference_image_4 = _pick("🖼️ 参考图像4", "reference_image_4", default=None)

        resolved_api_key = _resolve_kuai_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少 Kuai API Key，请优先使用环境变量 KUAI_API_KEY，或在节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not primary_directory or not os.path.isdir(primary_directory):
            msg = "主图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if model not in MWKuaiGPTImage2._MODEL_CHOICES:
            msg = "Kuai GPT2.0 不支持的模型: {}".format(model)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            size, size_note = _resolve_size_info_from_ratio(aspect_ratio, resolution)
        except Exception as e:
            msg = str(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        primary_files = _list_local_image_files(primary_directory)
        if not primary_files:
            msg = "主图片文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            reference_groups = []
            slot_labels = []
            if reference_image_1 is not None:
                reference_groups.append(_kuai_tensor_reference(reference_image_1, "参考槽位1", 1))
                slot_labels.append("参考槽位1(图片)")
            else:
                if not reference_directory_1:
                    raise ValueError("参考图文件夹1不存在。")
                reference_groups.append(_kuai_folder_references(reference_directory_1, "参考槽位1", 1))
                slot_labels.append("参考槽位1(文件夹)")

            for slot_index, folder_path, image_tensor in (
                (2, reference_directory_2, reference_image_2),
                (3, reference_directory_3, reference_image_3),
                (4, reference_directory_4, reference_image_4),
            ):
                if image_tensor is not None:
                    reference_groups.append(_kuai_tensor_reference(image_tensor, "参考槽位{}".format(slot_index), slot_index))
                    slot_labels.append("参考槽位{}(图片)".format(slot_index))
                elif folder_path:
                    reference_groups.append(_kuai_folder_references(folder_path, "参考槽位{}".format(slot_index), slot_index))
                    slot_labels.append("参考槽位{}(文件夹)".format(slot_index))
        except Exception as e:
            msg = "参考素材处理失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not reference_groups or not reference_groups[0]:
            msg = "参考槽位1未准备成功。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "Kuai_GPTImage2_MultiReferenceBatch")
        endpoint = KUAI_BASE_URL + KUAI_IMAGE_EDITS_PATH
        request_data = _kuai_make_request_data(fixed_prompt, model, size)
        reference_combos = list(product(*reference_groups))
        total_combinations = len(primary_files) * len(reference_combos)
        preview_pils = []
        failed_list = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()

        primary_entries = []
        for file_index, file_path in enumerate(primary_files):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_entries.append(
                    {
                        "file_index": int(file_index),
                        "base_name": base_name,
                        "png_bytes": _blt_local_file_to_png_bytes(file_path),
                    }
                )
            except Exception as e:
                failed_list.append("{} -> 读取失败: {}".format(base_name, e))

        def _process_one_combo(primary_entry, combo_index, reference_combo):
            combo_name = " + ".join(item["file_name"] for item in reference_combo)
            file_success = 0
            file_preview = []
            task_count = 0
            ordered_references = sorted(reference_combo, key=lambda item: int(item.get("slot_index", 999)))
            for run_index in range(1, executions_per_prompt + 1):
                request_files = [
                    _blt_request_file_from_png_bytes(
                        "primary_{}_{}.png".format(primary_entry["file_index"] + 1, primary_entry["base_name"]),
                        primary_entry["png_bytes"],
                        "image",
                    )
                ]
                for ref_index, reference_item in enumerate(ordered_references, start=2):
                    request_files.append(
                        _blt_request_file_from_png_bytes(
                            "ref{}_{}.png".format(ref_index, reference_item["file_name"]),
                            reference_item["png_bytes"],
                            "image",
                        )
                    )

                task_count += 1
                try:
                    response = _kuai_post_multipart(
                        resolved_api_key,
                        endpoint,
                        request_data,
                        request_files,
                        timeout=900,
                    )
                    pil_images = _kuai_response_to_pil_images(response)
                    if not pil_images:
                        return file_success, task_count, file_preview, "{} + {} 未解析到图片结果".format(primary_entry["base_name"], combo_name)
                    combo_parts = [
                        "S{}_{}".format(reference_item["slot_index"], reference_item["file_name"])
                        for reference_item in ordered_references
                    ]
                    prefix = "Img{:03d}_{}_C{:03d}_{}".format(
                        primary_entry["file_index"] + 1,
                        primary_entry["base_name"],
                        combo_index,
                        "_".join(combo_parts),
                    )
                    _kuai_save_pil_results(pil_images, batch_output_dir, prefix, run_index)
                    file_success += len(pil_images)
                    if len(file_preview) < 15:
                        file_preview.extend(pil_images[: 15 - len(file_preview)])
                except Exception as e:
                    return file_success, task_count, file_preview, "{} + {} 生成失败: {}".format(primary_entry["base_name"], combo_name, e)

            if file_success == 0:
                return 0, task_count, [], "{} + {} 未生成结果".format(primary_entry["base_name"], combo_name)
            return file_success, task_count, file_preview, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {}
            for primary_entry in primary_entries:
                for combo_index, reference_combo in enumerate(reference_combos, start=1):
                    future_map[executor.submit(_process_one_combo, primary_entry, combo_index, reference_combo)] = (
                        primary_entry["base_name"],
                        combo_index,
                    )

            for future in as_completed(future_map):
                try:
                    success_count, task_count, file_preview, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, error_msg = 0, 0, [], str(e)
                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    if file_preview and len(preview_pils) < 15:
                        preview_pils.extend(file_preview[: 15 - len(preview_pils)])
                    if error_msg:
                        failed_list.append(error_msg)

        status_lines = [
            "✅ kuai GPT2.0 多参考批量处理完成",
            "📁 主图片数: {}".format(len(primary_files)),
            "🧩 参考槽位数: {}".format(len(reference_groups)),
            "🔢 组合数: {}".format(int(total_combinations)),
            "📐 图像比例: {}".format(aspect_ratio),
            "📺 分辨率: {}".format(resolution),
            "📏 实际size: {}".format(size),
            "📝 size说明: {}".format(size_note),
            "🔁 单条Prompt执行次数: {}".format(int(executions_per_prompt)),
            "⚙️ 同时处理文件数: {}".format(int(max_concurrent_files)),
            "📦 总任务数: {}".format(int(total_tasks)),
            "✅ 成功输出数: {}".format(int(total_success)),
            "❌ 失败项数: {}".format(len(failed_list)),
            "💾 输出目录: {}".format(batch_output_dir),
            "📚 已启用参考槽位: {}".format("、".join(slot_labels)),
            "⚠️ 预览仅显示前 15 张，全量图片请查看输出文件夹。",
        ]
        if failed_list:
            status_lines.append("")
            status_lines.append("--- 失败记录(前5个) ---")
            for item in failed_list[:5]:
                status_lines.append("• {}".format(item))
            if len(failed_list) > 5:
                status_lines.append("...以及其他 {} 个失败项".format(len(failed_list) - 5))

        status_report = "\n".join(status_lines)
        return {"ui": {"string": [status_report]}, "result": (_kuai_batch_status_preview(preview_pils), status_report)}


NODE_CLASS_MAPPINGS = {
    "MWKuaiGPTImage2": MWKuaiGPTImage2,
    "MWKuaiGPTImage2FolderBatch": MWKuaiGPTImage2FolderBatch,
    "MWKuaiGPTImage2MultiReferenceBatch": MWKuaiGPTImage2MultiReferenceBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MWKuaiGPTImage2": "🎨 kuai-GPT2.0 图像生成",
    "MWKuaiGPTImage2FolderBatch": "📂 kuai-GPT2.0 文件夹批量处理",
    "MWKuaiGPTImage2MultiReferenceBatch": "📂🧩 kuai-GPT2.0 多参考批量处理",
}
