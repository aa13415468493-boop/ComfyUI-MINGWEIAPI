import json
import io
import os
import re
import time

import requests

import folder_paths
from comfy_api.latest import InputImpl

from .doubao_seedance20_zhenzhen_node import (
    block_video,
    empty_image,
    get_config,
    _asset_ref,
    _audio_input_to_data_url,
    _download_image_tensor,
    _download_video_preview,
    _find_first_url,
    _get_registered_tasks,
    _image_tensor_to_data_url,
    _normalize_asset_type,
    _register_task,
    _safe_json_loads,
    _status_is_failed,
    _status_is_success,
    tensor2pil,
    _update_task_record,
    _video_input_to_data_url,
)


DUOYUAN_DEFAULT_BASE_URL = "https://zx1.deepwl.net"
DUOYUAN_CREATE_ENDPOINT = "/v1/video/generations"
DUOYUAN_ASSET_GROUP_ENDPOINT = "/v1/seedance/asset/CreateAssetGroup"
DUOYUAN_ASSET_CREATE_ENDPOINT = "/v1/seedance/asset/CreateAsset"
DUOYUAN_ASSET_QUERY_ENDPOINT = "/v1/seedance/asset/GetAsset"
DUOYUAN_TEMP_IMAGE_UPLOAD_URL = "https://imageproxy.zhongzhuan.chat/api/upload"
DUOYUAN_NODE_CATEGORY = "🤖MINGWEI-API/MW-SD2/多元-SD2"
DUOYUAN_TASK_NAMESPACE = "duoyuan"


def get_duoyuan_api_key(widget_value):
    env_keys = [
        "DOUBAO_SEEDANCE_DUOYUAN_API_KEY",
        "DEEPWL_API_KEY",
        "ZX1_DEEPWL_API_KEY",
        "DOUBAO_SEEDANCE_API_KEY",
        "DOUBAO_SEEDANCE2_API_KEY",
        "MW_SD2_API_KEY",
    ]
    for env_key in env_keys:
        value = str(os.getenv(env_key, "") or "").strip()
        if value:
            return value

    config = get_config()
    for key in [
        "doubao_seedance_duoyuan_api_key",
        "deepwl_api_key",
        "zx1_deepwl_api_key",
        "doubao_seedance_api_key",
        "doubao_seedance2_api_key",
        "api_key",
    ]:
        value = str(config.get(key, "") or "").strip()
        if value:
            return value

    return str(widget_value or "").strip()


def _duoyuan_headers(api_key):
    auth_value = str(api_key or "").strip()
    if not auth_value.lower().startswith("bearer "):
        auth_value = f"Bearer {auth_value}"
    return {
        "Authorization": auth_value,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _duoyuan_is_privacy_image_error(text):
    lowered = str(text or "").lower()
    return (
        "inputimagesensitivecontentdetected.privacyinformation" in lowered
        or "input image may contain real person" in lowered
    )


def _duoyuan_response_error_text(response):
    text = response.text
    if _duoyuan_is_privacy_image_error(text):
        return (
            f"HTTP {response.status_code}: 检测到真人/隐私图片拦截。"
            "Seedance 2.0 不支持把真人图片作为 IMAGE/base64 直接上传到视频生成接口；"
            "请先通过多元资产节点创建/查询素材，得到 asset:// 或素材ID 后接入素材绑定。"
            f" 原始返回：{text}"
        )
    return f"HTTP {response.status_code}: {text}"


def _duoyuan_request(method, url, headers, **kwargs):
    request_headers = dict(headers or {})
    request_headers["Connection"] = "close"
    last_error = None
    for attempt in range(3):
        try:
            return requests.request(method, url, headers=request_headers, **kwargs)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as error:
            last_error = error
            if attempt >= 2:
                raise
            time.sleep(attempt + 1)
    raise last_error


def _duoyuan_image_to_upload_file(image, image_format, quality):
    image_format = str(image_format or "png").lower()
    if image_format not in {"jpeg", "png", "webp"}:
        image_format = "png"

    quality = max(1, min(100, int(quality)))
    image_pil = tensor2pil(image)
    save_kwargs = {}
    if image_format == "jpeg":
        image_pil = image_pil.convert("RGB")
        pil_format, ext, mime_type = "JPEG", "jpg", "image/jpeg"
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif image_format == "webp":
        pil_format, ext, mime_type = "WEBP", "webp", "image/webp"
        save_kwargs["quality"] = quality
    else:
        pil_format, ext, mime_type = "PNG", "png", "image/png"

    buffer = io.BytesIO()
    image_pil.save(buffer, format=pil_format, **save_kwargs)
    file_name = f"comfyui_duoyuan_{int(time.time())}.{ext}"
    return file_name, buffer.getvalue(), mime_type


def _duoyuan_find_any_http_url(value):
    if isinstance(value, dict):
        preferred_keys = ["url", "URL", "image_url", "imageUrl", "download_url", "downloadUrl", "link", "src"]
        for key in preferred_keys:
            if key in value:
                found = _duoyuan_find_any_http_url(value.get(key))
                if found:
                    return found
        for item in value.values():
            found = _duoyuan_find_any_http_url(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _duoyuan_find_any_http_url(item)
            if found:
                return found
    else:
        match = re.search(r"https?://[^\s\"'<>]+", str(value or ""))
        if match:
            return match.group(0).rstrip(",.;)]}")
    return ""


def _duoyuan_find_created_time(value):
    if isinstance(value, dict):
        for key in ["CreateTime", "create_time", "created_at", "createdAt", "time", "timestamp"]:
            item = value.get(key)
            if item:
                return str(item)
        for item in value.values():
            found = _duoyuan_find_created_time(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _duoyuan_find_created_time(item)
            if found:
                return found
    return ""


def _duoyuan_asset_query_task_id(response_json):
    if not isinstance(response_json, dict):
        return ""
    data = response_json.get("data", {})
    if not isinstance(data, dict):
        data = {}
    return str(
        data.get("task_id")
        or data.get("TaskId")
        or data.get("TaskID")
        or data.get("taskId")
        or response_json.get("task_id")
        or response_json.get("TaskId")
        or response_json.get("TaskID")
        or response_json.get("taskId")
        or ""
    ).strip()


def _duoyuan_asset_ref(asset_id):
    asset_id = str(asset_id or "").strip()
    if not asset_id:
        return ""
    if asset_id.startswith("asset://"):
        return asset_id
    return _asset_ref(asset_id)


def _normalize_duoyuan_asset_bundle(asset_bundle):
    if isinstance(asset_bundle, dict):
        return asset_bundle
    if isinstance(asset_bundle, str) and asset_bundle.strip():
        try:
            value = json.loads(asset_bundle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _split_url_values(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    if not text:
        return []
    normalized = text.replace("\r", "\n").replace(",", "\n")
    return [item.strip() for item in normalized.split("\n") if item.strip()]


def _content_item(media_type, url, role):
    key = f"{media_type}_url"
    item = {
        "type": key,
        key: {
            "url": url,
        },
    }
    if role:
        item["role"] = role
    return item


def _duoyuan_task_data(response_json):
    data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
    inner_data = data.get("data", {}) if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}, inner_data if isinstance(inner_data, dict) else {}


def _duoyuan_extract_task_info(response_json, fallback_task_id=""):
    data, inner_data = _duoyuan_task_data(response_json)
    top_task_id = response_json.get("task_id", "") if isinstance(response_json, dict) else ""
    task_id = str(data.get("task_id") or top_task_id or fallback_task_id)
    if not task_id:
        task_id = str(fallback_task_id or "")
    top_status = response_json.get("status", "") if isinstance(response_json, dict) else ""
    status = str(data.get("status") or inner_data.get("status") or top_status)
    video_url = _find_first_url(response_json, extensions=[".mp4", ".mov", ".webm", ".mkv"], preferred_keys=["video_url", "url", "download_url"])
    last_frame_url = _find_first_url(response_json, extensions=[".png", ".jpg", ".jpeg", ".webp"], preferred_keys=["last_frame_url", "last_frame", "tail_frame_url", "tail_frame", "cover_url"])
    fail_reason = str(data.get("fail_reason") or inner_data.get("fail_reason") or "")
    response_info = json.dumps(response_json, ensure_ascii=False) if isinstance(response_json, dict) else str(response_json or "")
    return task_id, status, video_url, last_frame_url, fail_reason, response_info


def _duoyuan_create_task(base_url, headers, payload):
    response = _duoyuan_request(
        "POST",
        f"{base_url.rstrip('/')}{DUOYUAN_CREATE_ENDPOINT}",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        return None, _duoyuan_response_error_text(response)
    return response.json(), ""


def _duoyuan_query_task_once(base_url, headers, task_id):
    response = _duoyuan_request(
        "GET",
        f"{base_url.rstrip('/')}{DUOYUAN_CREATE_ENDPOINT}/{task_id}",
        headers=headers,
        timeout=60,
    )
    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw_text": response.text}
    if response.status_code >= 400:
        if "fail_to_fetch_task" in response.text:
            return {
                "code": "fail_to_fetch_task",
                "data": {
                    "task_id": task_id,
                    "status": "IN_PROGRESS",
                    "data": {
                        "status": "processing",
                    },
                },
                "error": response_json,
            }
        response.raise_for_status()
    return response_json


def _duoyuan_poll_task(base_url, headers, task_id, max_wait_seconds, poll_interval_seconds):
    start_time = time.time()
    last_response = {}
    while True:
        try:
            last_response = _duoyuan_query_task_once(base_url, headers, task_id)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as error:
            last_response = {
                "code": "query_network_error",
                "data": {
                    "task_id": task_id,
                    "status": "IN_PROGRESS",
                    "data": {
                        "status": "processing",
                    },
                },
                "error": str(error),
            }
        _, status, _, _, _, _ = _duoyuan_extract_task_info(last_response, task_id)
        if _status_is_success(status) or _status_is_failed(status):
            return last_response
        if time.time() - start_time >= max_wait_seconds:
            return last_response
        time.sleep(poll_interval_seconds)


def _duoyuan_resolve_video_result(task_response, task_id, output_dir):
    resolved_task_id, status, video_url, last_frame_url, fail_reason, response_info = _duoyuan_extract_task_info(task_response, task_id)
    last_frame_tensor = _download_image_tensor(last_frame_url) if last_frame_url else empty_image()

    if video_url and _status_is_success(status):
        _, file_path = _download_video_preview(video_url, output_dir, resolved_task_id)
        video_output = InputImpl.VideoFromFile(file_path)
        return (video_output, file_path, resolved_task_id, response_info, last_frame_tensor)

    if _status_is_failed(status):
        message = fail_reason or "多元视频任务执行失败，当前没有可保存的视频输出。"
        return (block_video(message), "", resolved_task_id, response_info, last_frame_tensor)

    return (block_video("多元视频任务仍在排队或生成中，请稍后再次查询。"), "", resolved_task_id, response_info, last_frame_tensor)


def _duoyuan_format_task_report_line(status, task_id, prompt, model):
    prompt_text = str(prompt or "").replace("\n", " ").strip()
    if len(prompt_text) > 30:
        prompt_text = f"{prompt_text[:30]}..."
    model_text = f" [{model}]" if model else ""
    title_text = f" - {prompt_text}" if prompt_text else ""
    return f"[{status or 'unknown'}] {task_id}{model_text}{title_text}"


def _query_registered_duoyuan_tasks(api_key, base_url, limit=20):
    headers = _duoyuan_headers(api_key)
    records = _get_registered_tasks(api_key, namespace=DUOYUAN_TASK_NAMESPACE)
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0), reverse=True)
    report_lines = ["--- 多元任务队列总览 ---"]
    response_items = []

    for item in records[:limit]:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        prompt = item.get("prompt", "")
        model = item.get("model", "")
        try:
            response_json = _duoyuan_query_task_once(base_url, headers, task_id)
            resolved_task_id, status, video_url, _, _, response_info = _duoyuan_extract_task_info(response_json, task_id)
            _update_task_record(api_key, resolved_task_id, namespace=DUOYUAN_TASK_NAMESPACE, status=status, video_url=video_url, response_info=response_info)
            report_lines.append(_duoyuan_format_task_report_line(status, resolved_task_id, prompt, model))
            response_items.append({
                "task_id": resolved_task_id,
                "status": status,
                "video_url": video_url,
                "prompt": prompt,
                "model": model,
                "response": response_json,
            })
        except Exception as error:
            error_text = str(error)
            _update_task_record(api_key, task_id, namespace=DUOYUAN_TASK_NAMESPACE, status=f"error: {error_text}")
            report_lines.append(_duoyuan_format_task_report_line("error", task_id, prompt, model))
            response_items.append({
                "task_id": task_id,
                "status": "error",
                "video_url": "",
                "prompt": prompt,
                "model": model,
                "error": error_text,
            })

    if len(report_lines) == 1:
        report_lines.append("暂无本地提交任务记录")

    return "\n".join(report_lines), json.dumps(response_items, ensure_ascii=False)


def _select_downloadable_duoyuan_task(api_key, base_url):
    headers = _duoyuan_headers(api_key)
    records = _get_registered_tasks(api_key, namespace=DUOYUAN_TASK_NAMESPACE)
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0))
    fallback = None

    for item in records:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        try:
            response_json = _duoyuan_query_task_once(base_url, headers, task_id)
        except Exception:
            continue
        resolved_task_id, status, video_url, _, _, response_info = _duoyuan_extract_task_info(response_json, task_id)
        _update_task_record(api_key, resolved_task_id, namespace=DUOYUAN_TASK_NAMESPACE, status=status, video_url=video_url, response_info=response_info)
        if _status_is_success(status) and video_url:
            if not bool(item.get("downloaded", False)):
                return response_json, resolved_task_id
            fallback = (response_json, resolved_task_id)

    return fallback if fallback is not None else (None, "")


def _duoyuan_base_inputs():
    return {
        "📝 提示词": ("STRING", {
            "multiline": True,
            "default": "",
            "placeholder": "请输入文生视频、图生视频、视频延续或多模态合成提示词",
        }),
        "🤖 模型名称": ([
            "doubao-seedance-2-0-fast-260128",
            "doubao-seedance-2-0-260128",
        ], {
            "default": "doubao-seedance-2-0-fast-260128",
        }),
        "🎛️ 生成模式": ([
            "文生视频",
            "首帧图生视频",
            "首尾帧生视频",
            "参考图生视频",
            "视频延续",
            "视频编辑",
            "多模态合成",
        ], {
            "default": "文生视频",
        }),
        "⏱️ 时长(秒)": ("INT", {
            "default": 5,
            "min": -1,
            "max": 15,
        }),
        "🎞️ 总帧数(0=不用)": ("INT", {
            "default": 0,
            "min": 0,
            "max": 1000,
        }),
        "🖥️ 分辨率": (["480p", "720p", "1080p"], {
            "default": "720p",
        }),
        "📐 视频比例": (["16:9", "9:16", "1:1", "4:3", "adaptive"], {
            "default": "16:9",
        }),
        "📷 固定机位": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false",
        }),
        "💧 水印": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false",
        }),
        "🎵 生成音频": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false",
        }),
        "🎬 返回尾帧": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false",
        }),
        "🧪 草稿模式": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false",
        }),
        "🔑 API密钥": ("STRING", {
            "default": "",
            "placeholder": "环境变量 / config.json / 节点输入三选一",
        }),
        "🌐 BaseURL": ("STRING", {
            "default": DUOYUAN_DEFAULT_BASE_URL,
            "placeholder": "https://zx1.deepwl.net",
        }),
        "🌐 CallbackURL": ("STRING", {
            "default": "",
            "placeholder": "可选，不填则节点内部轮询",
        }),
        "🏷️ ServiceTier": ("STRING", {
            "default": "",
            "placeholder": "可选，留空则不传",
        }),
        "⏳ 执行过期(秒)": ("INT", {
            "default": 172800,
            "min": 3600,
            "max": 259200,
        }),
        "🎲 随机种子": ("INT", {
            "default": 0,
            "min": 0,
            "max": 2147483647,
            "control_after_generate": "randomize",
        }),
        "⏳ 最大等待(秒)": ("INT", {
            "default": 600,
            "min": 10,
            "max": 3600,
        }),
        "🔁 查询间隔(秒)": ("INT", {
            "default": 3,
            "min": 1,
            "max": 60,
        }),
        "➕ 额外参数": ("STRING", {
            "multiline": True,
            "default": "",
            "placeholder": "{\"metadata\": {\"watermark\": false}}",
        }),
    }


def _duoyuan_media_inputs():
    return {
        "🧩 素材绑定": ("DUOYUAN_SD2_ASSET_BUNDLE",),
        "🎬 首帧图片": ("IMAGE",),
        "🏁 尾帧图片": ("IMAGE",),
        "🖼️ 参考图片1": ("IMAGE",),
        "🖼️ 参考图片2": ("IMAGE",),
        "🖼️ 参考图片3": ("IMAGE",),
        "🖼️ 参考图片4": ("IMAGE",),
        "🖼️ 参考图片5": ("IMAGE",),
        "🖼️ 参考图片6": ("IMAGE",),
        "🖼️ 参考图片7": ("IMAGE",),
        "🖼️ 参考图片8": ("IMAGE",),
        "🎞️ 参考视频1": ("VIDEO",),
        "🎞️ 参考视频2": ("VIDEO",),
        "🎞️ 参考视频3": ("VIDEO",),
        "🎞️ 参考视频4": ("VIDEO",),
        "🎵 参考音频1": ("AUDIO",),
        "🎵 参考音频2": ("AUDIO",),
        "🎵 参考音频3": ("AUDIO",),
        "🎵 参考音频4": ("AUDIO",),
    }


class DoubaoSeedance20DuoyuanNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _duoyuan_base_inputs(),
            "optional": _duoyuan_media_inputs(),
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "generate_video"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def _collect_media_urls(self, kwargs, extra_params):
        media = {
            "first_frame": [],
            "last_frame": [],
            "reference_images": [],
            "reference_videos": [],
            "reference_audios": [],
        }

        bundle = _normalize_duoyuan_asset_bundle(kwargs.get("🧩 素材绑定"))

        first_frame_asset = _duoyuan_asset_ref(bundle.get("first_frame", ""))
        if first_frame_asset:
            media["first_frame"].append(first_frame_asset)
        else:
            first_frame = _image_tensor_to_data_url(kwargs.get("🎬 首帧图片"))
            if first_frame:
                media["first_frame"].append(first_frame)

        last_frame_asset = _duoyuan_asset_ref(bundle.get("last_frame", ""))
        if last_frame_asset:
            media["last_frame"].append(last_frame_asset)
        else:
            last_frame = _image_tensor_to_data_url(kwargs.get("🏁 尾帧图片"))
            if last_frame:
                media["last_frame"].append(last_frame)

        for index in range(1, 9):
            asset_url = _duoyuan_asset_ref(bundle.get(f"ref_image{index}", ""))
            if asset_url:
                media["reference_images"].append(asset_url)
            else:
                url = _image_tensor_to_data_url(kwargs.get(f"🖼️ 参考图片{index}"))
                if url:
                    media["reference_images"].append(url)

        for index in range(1, 5):
            asset_url = _duoyuan_asset_ref(bundle.get(f"video{index}", ""))
            if asset_url:
                media["reference_videos"].append(asset_url)
            else:
                url = _video_input_to_data_url(kwargs.get(f"🎞️ 参考视频{index}"))
                if url:
                    media["reference_videos"].append(url)

        for index in range(1, 5):
            asset_url = _duoyuan_asset_ref(bundle.get(f"audio{index}", ""))
            if asset_url:
                media["reference_audios"].append(asset_url)
            else:
                url = _audio_input_to_data_url(kwargs.get(f"🎵 参考音频{index}"))
                if url:
                    media["reference_audios"].append(url)

        for key in ["first_frame_url", "first_frame_urls"]:
            media["first_frame"].extend(_split_url_values(extra_params.pop(key, [])))
        for key in ["last_frame_url", "last_frame_urls"]:
            media["last_frame"].extend(_split_url_values(extra_params.pop(key, [])))
        for key in ["reference_image_urls", "reference_images"]:
            media["reference_images"].extend(_split_url_values(extra_params.pop(key, [])))
        for key in ["reference_video_urls", "reference_videos"]:
            media["reference_videos"].extend(_split_url_values(extra_params.pop(key, [])))
        for key in ["reference_audio_urls", "reference_audios"]:
            media["reference_audios"].extend(_split_url_values(extra_params.pop(key, [])))

        return media

    def _build_content(self, kwargs, media):
        generation_mode = str(kwargs.get("🎛️ 生成模式", "文生视频") or "").strip()
        content = [{"type": "text", "text": kwargs.get("📝 提示词", "")}]

        if generation_mode == "文生视频":
            return content

        if generation_mode == "首帧图生视频":
            if not media["first_frame"]:
                raise ValueError("当前生成模式为首帧图生视频，请提供首帧图片，或通过素材绑定节点接入首帧素材ID。")
            content.append(_content_item("image", media["first_frame"][0], "first_frame"))
            return content

        if generation_mode == "首尾帧生视频":
            if not media["first_frame"]:
                raise ValueError("当前生成模式为首尾帧生视频，请提供首帧图片，或通过素材绑定节点接入首帧素材ID。")
            if not media["last_frame"]:
                raise ValueError("当前生成模式为首尾帧生视频，请提供尾帧图片，或通过素材绑定节点接入尾帧素材ID。")
            content.append(_content_item("image", media["first_frame"][0], "first_frame"))
            content.append(_content_item("image", media["last_frame"][0], "last_frame"))
            return content

        if generation_mode == "参考图生视频":
            if not media["reference_images"]:
                raise ValueError("当前生成模式为参考图生视频，请至少提供一张参考图片，或通过素材绑定节点接入参考图片素材ID。")
            for url in media["reference_images"]:
                content.append(_content_item("image", url, "reference_image"))
            return content

        if generation_mode in {"视频延续", "视频编辑"}:
            if not media["reference_videos"]:
                raise ValueError(f"当前生成模式为{generation_mode}，请至少提供一个参考视频或参考视频素材ID。")
            for url in media["reference_videos"]:
                content.append(_content_item("video", url, "reference_video"))
            if generation_mode == "视频编辑":
                for url in media["reference_images"]:
                    content.append(_content_item("image", url, "reference_image"))
                for url in media["reference_audios"]:
                    content.append(_content_item("audio", url, "reference_audio"))
            return content

        if generation_mode == "多模态合成":
            has_reference = any([media["reference_images"], media["reference_videos"], media["reference_audios"]])
            if not has_reference:
                raise ValueError("当前生成模式为多模态合成，请至少提供参考图片、参考视频或参考音频。")
            for url in media["reference_images"]:
                content.append(_content_item("image", url, "reference_image"))
            for url in media["reference_videos"]:
                content.append(_content_item("video", url, "reference_video"))
            for url in media["reference_audios"]:
                content.append(_content_item("audio", url, "reference_audio"))
            return content

        return content

    def _build_payload(self, kwargs):
        extra_params = _safe_json_loads(kwargs.get("➕ 额外参数", ""))
        media = self._collect_media_urls(kwargs, extra_params)
        metadata = {
            "duration": int(kwargs.get("⏱️ 时长(秒)", 5)),
            "resolution": kwargs.get("🖥️ 分辨率", "720p"),
            "ratio": kwargs.get("📐 视频比例", "16:9"),
            "camera_fixed": bool(kwargs.get("📷 固定机位", False)),
            "watermark": bool(kwargs.get("💧 水印", False)),
            "generate_audio": bool(kwargs.get("🎵 生成音频", False)),
            "return_last_frame": bool(kwargs.get("🎬 返回尾帧", False)),
            "draft": bool(kwargs.get("🧪 草稿模式", False)),
            "execution_expires_after": int(kwargs.get("⏳ 执行过期(秒)", 172800)),
        }

        frames = int(kwargs.get("🎞️ 总帧数(0=不用)", 0))
        if frames > 0:
            metadata["frames"] = frames

        seed = int(kwargs.get("🎲 随机种子", 0))
        if seed > 0:
            metadata["seed"] = seed

        callback_url = str(kwargs.get("🌐 CallbackURL", "") or "").strip()
        if callback_url:
            metadata["callback_url"] = callback_url

        service_tier = str(kwargs.get("🏷️ ServiceTier", "") or "").strip()
        if service_tier:
            metadata["service_tier"] = service_tier

        extra_metadata = extra_params.pop("metadata", {})
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)

        payload = {
            "model": kwargs.get("🤖 模型名称", "doubao-seedance-2-0-fast-260128"),
            "content": self._build_content(kwargs, media),
            "metadata": metadata,
        }

        extra_content = extra_params.pop("content", None)
        if isinstance(extra_content, list):
            payload["content"].extend(extra_content)

        if extra_params:
            payload.update(extra_params)

        return payload

    def generate_video(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        headers = _duoyuan_headers(api_key)
        payload = self._build_payload(kwargs)
        create_response, error_text = _duoyuan_create_task(base_url, headers, payload)
        if create_response is None:
            return (block_video(f"多元视频任务创建失败：{error_text}"), "", "", error_text, empty_image())

        data = create_response.get("data", {}) if isinstance(create_response, dict) else {}
        task_id = str(create_response.get("task_id") or create_response.get("id") or data.get("task_id") or data.get("id") or "")
        if not task_id:
            return (block_video("多元接口未返回任务ID，当前没有可保存的视频输出。"), "", "", json.dumps(create_response, ensure_ascii=False), empty_image())

        _register_task(api_key, task_id, {
            "prompt": kwargs.get("📝 提示词", ""),
            "model": payload.get("model", ""),
        }, namespace=DUOYUAN_TASK_NAMESPACE)

        task_response = _duoyuan_poll_task(
            base_url,
            headers,
            task_id,
            int(kwargs.get("⏳ 最大等待(秒)", 600)),
            int(kwargs.get("🔁 查询间隔(秒)", 3)),
        )
        return _duoyuan_resolve_video_result(task_response, task_id, self.output_dir)


class DoubaoSeedance20DuoyuanSubmitNode(DoubaoSeedance20DuoyuanNode):
    @classmethod
    def INPUT_TYPES(cls):
        return DoubaoSeedance20DuoyuanNode.INPUT_TYPES()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("🆔 任务ID", "🧾 响应信息")
    FUNCTION = "submit_task"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def submit_task(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        headers = _duoyuan_headers(api_key)
        payload = self._build_payload(kwargs)
        create_response, error_text = _duoyuan_create_task(base_url, headers, payload)
        if create_response is None:
            return ("", error_text)

        data = create_response.get("data", {}) if isinstance(create_response, dict) else {}
        task_id = str(create_response.get("task_id") or create_response.get("id") or data.get("task_id") or data.get("id") or "")
        _register_task(api_key, task_id, {
            "prompt": kwargs.get("📝 提示词", ""),
            "model": payload.get("model", ""),
        }, namespace=DUOYUAN_TASK_NAMESPACE)
        return (task_id, json.dumps(create_response, ensure_ascii=False))


class DoubaoSeedance20DuoyuanQueryTaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一",
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DUOYUAN_DEFAULT_BASE_URL,
                    "placeholder": "https://zx1.deepwl.net",
                }),
                "🆔 任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "留空则查询本地已提交任务队列",
                }),
                "⏳ 最大等待(秒)": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3600,
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 60,
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("📋 任务报告", "🆔 任务ID", "📌 状态", "🔗 视频URL", "🧾 响应信息")
    FUNCTION = "query_task"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def query_task(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        task_id = str(kwargs.get("🆔 任务ID", "") or "").strip()
        if not task_id:
            report_text, response_info = _query_registered_duoyuan_tasks(api_key, base_url)
            return (report_text, "", "", "", response_info)

        headers = _duoyuan_headers(api_key)
        task_response = _duoyuan_poll_task(
            base_url,
            headers,
            task_id,
            int(kwargs.get("⏳ 最大等待(秒)", 1)),
            int(kwargs.get("🔁 查询间隔(秒)", 1)),
        )
        resolved_task_id, status, video_url, _, _, response_info = _duoyuan_extract_task_info(task_response, task_id)
        report = _duoyuan_format_task_report_line(status, resolved_task_id, "", "")
        _update_task_record(api_key, resolved_task_id, namespace=DUOYUAN_TASK_NAMESPACE, status=status, video_url=video_url, response_info=response_info)
        return (report, resolved_task_id, status, video_url, response_info)


class DoubaoSeedance20DuoyuanGetVideoNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一",
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DUOYUAN_DEFAULT_BASE_URL,
                    "placeholder": "https://zx1.deepwl.net",
                }),
                "🆔 任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "可选，留空则获取本地队列中可下载的视频",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "get_video"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def get_video(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        task_id = str(kwargs.get("🆔 任务ID", "") or "").strip()
        if task_id:
            task_response = _duoyuan_query_task_once(base_url, _duoyuan_headers(api_key), task_id)
        else:
            task_response, task_id = _select_downloadable_duoyuan_task(api_key, base_url)
            if not task_response:
                return (block_video("当前没有可下载的多元视频任务，请先提交任务或等待生成完成。"), "", "", "", empty_image())

        result = _duoyuan_resolve_video_result(task_response, task_id, self.output_dir)
        if result[1]:
            _update_task_record(api_key, result[2], namespace=DUOYUAN_TASK_NAMESPACE, downloaded=True, last_download_path=result[1])
        return result


class DoubaoSeedance20DuoyuanTempImageHostNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upload_url": ("STRING", {
                    "default": DUOYUAN_TEMP_IMAGE_UPLOAD_URL,
                    "placeholder": "https://imageproxy.zhongzhuan.chat/api/upload",
                }),
                "format": (["png", "jpeg", "webp"], {
                    "default": "png",
                }),
                "quality": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 100,
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 300,
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片URL", "创建时间", "响应信息")
    FUNCTION = "upload_image"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def upload_image(self, image, upload_url, format="png", quality=100, timeout=30):
        upload_url = str(upload_url or "").strip() or DUOYUAN_TEMP_IMAGE_UPLOAD_URL
        file_name, file_bytes, mime_type = _duoyuan_image_to_upload_file(image, format, quality)

        def post_file(field_name):
            files = {field_name: (file_name, file_bytes, mime_type)}
            return requests.post(upload_url, files=files, timeout=int(timeout))

        try:
            response = post_file("file")
            if response.status_code >= 400:
                response = post_file("image")
        except Exception as error:
            created_time = time.strftime("%Y-%m-%d %H:%M:%S")
            return ("", created_time, f"UploadError: {error}")

        try:
            response_value = response.json()
        except Exception:
            response_value = response.text

        created_time = _duoyuan_find_created_time(response_value) or time.strftime("%Y-%m-%d %H:%M:%S")
        response_info = (
            json.dumps(response_value, ensure_ascii=False)
            if isinstance(response_value, (dict, list))
            else str(response_value)
        )
        if response.status_code >= 400:
            return ("", created_time, f"HTTP {response.status_code}: {response_info}")

        image_url = _duoyuan_find_any_http_url(response_value)
        return (image_url, created_time, response_info)


class DoubaoSeedance20DuoyuanCreateAssetGroupNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一",
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DUOYUAN_DEFAULT_BASE_URL,
                    "placeholder": "https://zx1.deepwl.net",
                }),
                "🏷️ 素材组名称": ("STRING", {
                    "default": "seedance-assets",
                }),
                "📝 素材组描述": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("🧩 素材组ID", "🧾 响应信息")
    FUNCTION = "create_asset_group"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def create_asset_group(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        payload = {
            "Name": kwargs.get("🏷️ 素材组名称", "seedance-assets"),
            "Description": kwargs.get("📝 素材组描述", ""),
        }
        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        response = _duoyuan_request(
            "POST",
            f"{base_url.rstrip('/')}{DUOYUAN_ASSET_GROUP_ENDPOINT}",
            headers=_duoyuan_headers(api_key),
            json=payload,
            timeout=120,
        )
        response_text = response.text
        try:
            response_json = response.json()
        except Exception:
            return ("", response_text)
        if response.status_code >= 400:
            return ("", _duoyuan_response_error_text(response))

        data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
        group_id = str(data.get("Id") or data.get("GroupId") or response_json.get("Id") or "")
        return (group_id, json.dumps(response_json, ensure_ascii=False))


class DoubaoSeedance20DuoyuanCreateAssetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一",
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DUOYUAN_DEFAULT_BASE_URL,
                    "placeholder": "https://zx1.deepwl.net",
                }),
                "🧩 素材组ID": ("STRING", {
                    "default": "",
                    "placeholder": "创建素材组节点返回的 group-xxx",
                }),
                "📦 素材类型": (["image", "video", "audio"], {
                    "default": "image",
                }),
                "🔗 素材URL": ("STRING", {
                    "default": "",
                    "placeholder": "公网 http/https 图片/视频/音频 URL",
                }),
                "🏷️ 素材名称": ("STRING", {
                    "default": "asset",
                }),
                "➕ 额外参数": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "{\"custom_field\": true}",
                }),
            },
            "optional": {
                "🔗 素材URL输入": ("STRING", {"forceInput": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 素材ID", "🧾 素材任务ID", "📌 状态", "🧾 响应信息")
    FUNCTION = "create_asset"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def create_asset(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        group_id = str(kwargs.get("🧩 素材组ID", "") or "").strip()
        if not group_id:
            return ("", "", "", "请先填写素材组ID。")

        material_type = kwargs.get("📦 素材类型", "image")
        source_url = str(kwargs.get("🔗 素材URL输入", "") or kwargs.get("🔗 素材URL", "") or "").strip()
        if not source_url:
            message = "请填写公网 http/https 素材URL。CreateAsset 文档要求 URL 是图片/视频/音频地址，不是 ComfyUI 本地图像输入。"
            return ("缺少素材URL", "", "MissingURL", message)
        if not (source_url.startswith("http://") or source_url.startswith("https://")):
            message = f"素材URL必须是公网 http:// 或 https:// 地址，当前收到：{source_url}"
            return ("素材URL无效", "", "InvalidURL", message)

        payload = {
            "GroupId": group_id,
            "URL": source_url,
            "AssetType": _normalize_asset_type(material_type),
            "Name": kwargs.get("🏷️ 素材名称", "asset"),
        }
        extra_params = _safe_json_loads(kwargs.get("➕ 额外参数", ""))
        if extra_params:
            payload.update(extra_params)

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        response = _duoyuan_request(
            "POST",
            f"{base_url.rstrip('/')}{DUOYUAN_ASSET_CREATE_ENDPOINT}",
            headers=_duoyuan_headers(api_key),
            json=payload,
            timeout=120,
        )
        response_text = response.text
        try:
            response_json = response.json()
        except Exception:
            return ("接口返回非JSON", "", "ResponseError", response_text)
        if response.status_code >= 400:
            return ("创建素材失败", "", f"HTTP {response.status_code}", _duoyuan_response_error_text(response))

        data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
        asset_id = str(data.get("Id") or data.get("AssetId") or response_json.get("Id") or "")
        task_id = _duoyuan_asset_query_task_id(response_json)
        status = "Submitted" if task_id else ""
        return (asset_id, task_id, status, json.dumps(response_json, ensure_ascii=False))


class DoubaoSeedance20DuoyuanQueryAssetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一",
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DUOYUAN_DEFAULT_BASE_URL,
                    "placeholder": "https://zx1.deepwl.net",
                }),
                "🧾 素材任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "上传素材节点返回的 task_id，不是 asset-xxx",
                }),
                "⏳ 最大等待(秒)": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 3600,
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60,
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 素材ID", "📌 状态", "🔗 素材URL", "🧩 素材组ID", "🧾 响应信息")
    FUNCTION = "query_asset"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def _query_asset_once(self, base_url, headers, task_id):
        response = _duoyuan_request(
            "POST",
            f"{base_url.rstrip('/')}{DUOYUAN_ASSET_QUERY_ENDPOINT}",
            headers=headers,
            json={"task_id": task_id},
            timeout=60,
        )
        try:
            response_json = response.json()
        except Exception:
            response_json = {"raw_text": response.text}
        if response.status_code >= 400:
            return {
                "state": 0,
                "data": {
                    "Status": f"HTTP {response.status_code}",
                    "Error": _duoyuan_response_error_text(response),
                },
                "error": response_json,
            }
        return response_json

    def query_asset(self, **kwargs):
        api_key = get_duoyuan_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        task_id = str(kwargs.get("🧾 素材任务ID", "") or "").strip()
        if not task_id:
            return ("", "", "", "", "请填写素材任务ID后再查询。")
        if task_id.startswith("asset-") or task_id.startswith("asset://"):
            message = "查询素材接口需要上传素材节点返回的 task_id。当前传入的是素材ID，请把上传素材节点的“素材任务ID”输出连接到这里。"
            return ("", "WrongInput", "", "", message)

        base_url = str(kwargs.get("🌐 BaseURL", "") or "").strip() or DUOYUAN_DEFAULT_BASE_URL
        headers = _duoyuan_headers(api_key)
        max_wait_seconds = int(kwargs.get("⏳ 最大等待(秒)", 300))
        poll_interval_seconds = int(kwargs.get("🔁 查询间隔(秒)", 2))
        start_time = time.time()
        response_json = {}

        while True:
            response_json = self._query_asset_once(base_url, headers, task_id)
            data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
            if not isinstance(data, dict):
                data = {}
            status = str(data.get("Status") or data.get("status") or "")
            if isinstance(response_json, dict) and response_json.get("state") == 0 and response_json.get("error"):
                break
            if status in {"Active", "Failed"} or status.startswith("HTTP "):
                break
            if time.time() - start_time >= max_wait_seconds:
                break
            time.sleep(poll_interval_seconds)

        data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
        if not isinstance(data, dict):
            data = {}
        asset_id = str(data.get("Id") or data.get("AssetId") or "")
        status = str(data.get("Status") or data.get("status") or "Processing")
        error_message = str(data.get("Error") or data.get("error") or "")
        if status == "Failed" and error_message:
            status = f"Failed: {error_message[:120]}"
        if status == "Processing" and isinstance(response_json, dict) and response_json.get("error"):
            error_value = response_json.get("error")
            error_text = json.dumps(error_value, ensure_ascii=False) if isinstance(error_value, (dict, list)) else str(error_value)
            status = f"Failed: {error_text[:120]}"
        asset_url = str(data.get("URL") or data.get("url") or "")
        group_id = str(data.get("GroupId") or data.get("group_id") or "")
        return (asset_id, status, asset_url, group_id, json.dumps(response_json, ensure_ascii=False))


class DoubaoSeedance20DuoyuanAssetIdBundleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬 首帧素材ID": ("STRING", {"default": ""}),
                "🏁 尾帧素材ID": ("STRING", {"default": ""}),
                "🖼️ 参考图片1": ("STRING", {"default": ""}),
                "🖼️ 参考图片2": ("STRING", {"default": ""}),
                "🖼️ 参考图片3": ("STRING", {"default": ""}),
                "🖼️ 参考图片4": ("STRING", {"default": ""}),
                "🖼️ 参考图片5": ("STRING", {"default": ""}),
                "🖼️ 参考图片6": ("STRING", {"default": ""}),
                "🖼️ 参考图片7": ("STRING", {"default": ""}),
                "🖼️ 参考图片8": ("STRING", {"default": ""}),
                "🎞️ 参考视频1": ("STRING", {"default": ""}),
                "🎞️ 参考视频2": ("STRING", {"default": ""}),
                "🎞️ 参考视频3": ("STRING", {"default": ""}),
                "🎞️ 参考视频4": ("STRING", {"default": ""}),
                "🎵 参考音频1": ("STRING", {"default": ""}),
                "🎵 参考音频2": ("STRING", {"default": ""}),
                "🎵 参考音频3": ("STRING", {"default": ""}),
                "🎵 参考音频4": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("DUOYUAN_SD2_ASSET_BUNDLE",)
    RETURN_NAMES = ("🧩 素材绑定",)
    FUNCTION = "build_bundle"
    CATEGORY = DUOYUAN_NODE_CATEGORY

    def build_bundle(self, **kwargs):
        bundle = {
            "first_frame": kwargs.get("🎬 首帧素材ID", ""),
            "last_frame": kwargs.get("🏁 尾帧素材ID", ""),
            "ref_image1": kwargs.get("🖼️ 参考图片1", ""),
            "ref_image2": kwargs.get("🖼️ 参考图片2", ""),
            "ref_image3": kwargs.get("🖼️ 参考图片3", ""),
            "ref_image4": kwargs.get("🖼️ 参考图片4", ""),
            "ref_image5": kwargs.get("🖼️ 参考图片5", ""),
            "ref_image6": kwargs.get("🖼️ 参考图片6", ""),
            "ref_image7": kwargs.get("🖼️ 参考图片7", ""),
            "ref_image8": kwargs.get("🖼️ 参考图片8", ""),
            "video1": kwargs.get("🎞️ 参考视频1", ""),
            "video2": kwargs.get("🎞️ 参考视频2", ""),
            "video3": kwargs.get("🎞️ 参考视频3", ""),
            "video4": kwargs.get("🎞️ 参考视频4", ""),
            "audio1": kwargs.get("🎵 参考音频1", ""),
            "audio2": kwargs.get("🎵 参考音频2", ""),
            "audio3": kwargs.get("🎵 参考音频3", ""),
            "audio4": kwargs.get("🎵 参考音频4", ""),
        }
        return (bundle,)


NODE_CLASS_MAPPINGS = {
    "DoubaoSeedance20DuoyuanNode": DoubaoSeedance20DuoyuanNode,
    "DoubaoSeedance20DuoyuanSubmitNode": DoubaoSeedance20DuoyuanSubmitNode,
    "DoubaoSeedance20DuoyuanQueryTaskNode": DoubaoSeedance20DuoyuanQueryTaskNode,
    "DoubaoSeedance20DuoyuanGetVideoNode": DoubaoSeedance20DuoyuanGetVideoNode,
    "DoubaoSeedance20DuoyuanTempImageHostNode": DoubaoSeedance20DuoyuanTempImageHostNode,
    "DoubaoSeedance20DuoyuanCreateAssetGroupNode": DoubaoSeedance20DuoyuanCreateAssetGroupNode,
    "DoubaoSeedance20DuoyuanCreateAssetNode": DoubaoSeedance20DuoyuanCreateAssetNode,
    "DoubaoSeedance20DuoyuanQueryAssetNode": DoubaoSeedance20DuoyuanQueryAssetNode,
    "DoubaoSeedance20DuoyuanAssetIdBundleNode": DoubaoSeedance20DuoyuanAssetIdBundleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedance20DuoyuanNode": "doubao-seedance2.0-多元",
    "DoubaoSeedance20DuoyuanSubmitNode": "doubao-seedance2.0-多元-提交任务",
    "DoubaoSeedance20DuoyuanQueryTaskNode": "doubao-seedance2.0-多元-查询任务",
    "DoubaoSeedance20DuoyuanGetVideoNode": "doubao-seedance2.0-多元-获取视频",
    "DoubaoSeedance20DuoyuanTempImageHostNode": "doubao-seedance2.0-多元-传图到临时图床",
    "DoubaoSeedance20DuoyuanCreateAssetGroupNode": "doubao-seedance2.0-多元-创建素材组",
    "DoubaoSeedance20DuoyuanCreateAssetNode": "doubao-seedance2.0-多元-上传素材",
    "DoubaoSeedance20DuoyuanQueryAssetNode": "doubao-seedance2.0-多元-查询素材",
    "DoubaoSeedance20DuoyuanAssetIdBundleNode": "doubao-seedance2.0-多元-素材绑定",
}
