import base64
import hashlib
import io
import json
import os
import time
import wave

import numpy as np
import requests
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import InputImpl
from comfy_execution.graph_utils import ExecutionBlocker

try:
    import imageio
except ImportError:
    imageio = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, "config.json")
TASK_CACHE_FILE = os.path.join(CURRENT_DIR, "task_cache.json")
DEFAULT_BASE_URL = "https://ai.t8star.org"
CREATE_ENDPOINT = "/v2/videos/generations"
DEFAULT_ASSET_UPLOAD_ENDPOINT = "/seedance/v3/assets/create"
DEFAULT_ASSET_QUERY_ENDPOINT = "/seedance/v3/assets/query"
DEFAULT_FILE_UPLOAD_ENDPOINT = "/v1/files"
NODE_CATEGORY = "🤖MINGWEI-API/MW-SD2/zhenzhen-SD2"
KIE_DEFAULT_BASE_URL = "https://api.kie.ai"
KIE_CREATE_ENDPOINT = "/api/v1/jobs/createTask"
KIE_QUERY_ENDPOINT = "/api/v1/jobs/recordInfo"
KIE_ASSET_CREATE_ENDPOINT = "/api/v1/playground/createAsset"
KIE_ASSET_QUERY_ENDPOINT = "/api/v1/playground/getAsset"
KIE_UPLOAD_BASE_URL = "https://kieai.redpandaai.co"
KIE_FILE_STREAM_UPLOAD_ENDPOINT = "/api/file-stream-upload"
KIE_NODE_CATEGORY = "🤖MINGWEI-API/MW-SD2/kie-SD2"


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    array = tensor.cpu().numpy()
    array = np.clip(array, 0, 1)
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def empty_image(width=512, height=512):
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def block_video(message):
    return ExecutionBlocker(message)


def get_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_task_cache():
    if not os.path.exists(TASK_CACHE_FILE):
        return {}
    try:
        with open(TASK_CACHE_FILE, "r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _save_task_cache(data):
    try:
        with open(TASK_CACHE_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _hash_api_key(api_key):
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()


def _cache_namespace_key(api_key, namespace="zhenzhen"):
    return f"{namespace}:{_hash_api_key(api_key)}"


def _register_task(api_key, task_id, payload, namespace="zhenzhen"):
    task_id = str(task_id or "").strip()
    if not task_id:
        return
    cache = _load_task_cache()
    cache_key = _cache_namespace_key(api_key, namespace)
    task_list = cache.get(cache_key, [])
    if not isinstance(task_list, list):
        task_list = []

    prompt = str(payload.get("prompt", "") or "")
    model = str(payload.get("model", "") or "")
    for item in task_list:
        if str(item.get("task_id", "")) == task_id:
            item["prompt"] = prompt
            item["model"] = model
            item.setdefault("created_at", int(time.time()))
            item.setdefault("downloaded", False)
            cache[cache_key] = task_list
            _save_task_cache(cache)
            return

    task_list.append({
        "task_id": task_id,
        "prompt": prompt,
        "model": model,
        "created_at": int(time.time()),
        "downloaded": False,
    })
    cache[cache_key] = task_list
    _save_task_cache(cache)


def _get_registered_tasks(api_key, namespace="zhenzhen"):
    cache = _load_task_cache()
    task_list = cache.get(_cache_namespace_key(api_key, namespace), [])
    return task_list if isinstance(task_list, list) else []


def _update_task_record(api_key, task_id, namespace="zhenzhen", **updates):
    task_id = str(task_id or "").strip()
    if not task_id:
        return
    cache = _load_task_cache()
    cache_key = _cache_namespace_key(api_key, namespace)
    task_list = cache.get(cache_key, [])
    if not isinstance(task_list, list):
        return
    changed = False
    for item in task_list:
        if str(item.get("task_id", "")) == task_id:
            item.update(updates)
            changed = True
            break
    if changed:
        cache[cache_key] = task_list
        _save_task_cache(cache)


def get_api_key(widget_value):
    env_keys = [
        "DOUBAO_SEEDANCE_API_KEY",
        "DOUBAO_SEEDANCE2_API_KEY",
        "MW_SD2_API_KEY",
        "T8STAR_API_KEY",
    ]
    for env_key in env_keys:
        value = (os.getenv(env_key, "") or "").strip()
        if value:
            return value

    config = get_config()
    for key in ["doubao_seedance_api_key", "doubao_seedance2_api_key", "api_key", "t8star_api_key"]:
        value = str(config.get(key, "") or "").strip()
        if value:
            return value

    return (widget_value or "").strip()


def _safe_json_loads(text):
    if not text or not str(text).strip():
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _guess_mime_type(file_path):
    ext = os.path.splitext(file_path or "")[1].lower().lstrip(".")
    mapping = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
        "avi": "video/x-msvideo",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    return mapping.get(ext, "application/octet-stream")


def _to_data_url(raw_bytes, mime_type):
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"


def _is_empty_media_input(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _normalize_url_or_file(value):
    if not value:
        return ""
    value = value.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://") or value.startswith("data:"):
        return value
    if not os.path.exists(value):
        return ""
    try:
        with open(value, "rb") as file:
            return _to_data_url(file.read(), _guess_mime_type(value))
    except Exception:
        return ""


def _image_tensor_to_data_url(image_tensor, max_size=2048):
    if image_tensor is None:
        return ""
    try:
        image = tensor2pil(image_tensor)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return _to_data_url(buffer.getvalue(), "image/jpeg")
    except Exception:
        return ""


def _video_input_to_data_url(video_input):
    if _is_empty_media_input(video_input):
        return ""
    if isinstance(video_input, str):
        return _normalize_url_or_file(video_input)

    temp_dir = folder_paths.get_temp_directory()
    temp_path = os.path.join(temp_dir, f"doubao_seedance2_ref_{int(time.time() * 1000)}.mp4")

    try:
        if hasattr(video_input, "save_to"):
            video_input.save_to(temp_path)
            return _normalize_url_or_file(temp_path)
    except Exception:
        return ""
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return ""


def _audio_tensor_to_wav_bytes(waveform, sample_rate):
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()

    waveform = np.asarray(waveform)
    waveform = np.squeeze(waveform)

    if waveform.ndim == 1:
        waveform = waveform.reshape(-1, 1)
    elif waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
        waveform = waveform.T
    elif waveform.ndim > 2:
        waveform = waveform.reshape(-1, 1)

    if np.issubdtype(waveform.dtype, np.floating):
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767.0).astype(np.int16)
    else:
        pcm = waveform.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _audio_input_to_data_url(audio_input):
    if _is_empty_media_input(audio_input):
        return ""
    if isinstance(audio_input, str):
        return _normalize_url_or_file(audio_input)
    if isinstance(audio_input, dict):
        waveform = audio_input.get("waveform")
        sample_rate = audio_input.get("sample_rate", audio_input.get("sampler_rate", 44100))
        if waveform is None:
            return ""
        return _to_data_url(_audio_tensor_to_wav_bytes(waveform, int(sample_rate)), "audio/wav")
    return ""


def _image_tensor_to_upload_file(image_tensor, file_name="image.jpg"):
    if image_tensor is None:
        return None
    image = tensor2pil(image_tensor)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return (file_name, buffer.getvalue(), "image/jpeg")


def _video_input_to_upload_file(video_input, file_name="video.mp4"):
    if _is_empty_media_input(video_input):
        return None
    if isinstance(video_input, str):
        file_path = video_input.strip()
        if not file_path or not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as file:
            return (os.path.basename(file_path), file.read(), _guess_mime_type(file_path))

    temp_dir = folder_paths.get_temp_directory()
    temp_path = os.path.join(temp_dir, f"doubao_seedance2_upload_{int(time.time() * 1000)}.mp4")
    try:
        if hasattr(video_input, "save_to"):
            video_input.save_to(temp_path)
            with open(temp_path, "rb") as file:
                return (file_name, file.read(), "video/mp4")
    except Exception:
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return None


def _audio_input_to_upload_file(audio_input, file_name="audio.wav"):
    if _is_empty_media_input(audio_input):
        return None
    if isinstance(audio_input, str):
        file_path = audio_input.strip()
        if not file_path or not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as file:
            return (os.path.basename(file_path), file.read(), _guess_mime_type(file_path))
    if isinstance(audio_input, dict):
        waveform = audio_input.get("waveform")
        sample_rate = audio_input.get("sample_rate", audio_input.get("sampler_rate", 44100))
        if waveform is None:
            return None
        return (file_name, _audio_tensor_to_wav_bytes(waveform, int(sample_rate)), "audio/wav")
    return None


def _path_to_upload_file(file_path):
    file_path = (file_path or "").strip()
    if not file_path or not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as file:
        return (os.path.basename(file_path), file.read(), _guess_mime_type(file_path))


def _resolve_asset_upload_file(material_type, kwargs):
    material_type = (material_type or "").lower()
    if material_type == "image":
        return _image_tensor_to_upload_file(kwargs.get("🖼️ 上传图片"))
    if material_type == "video":
        return _video_input_to_upload_file(kwargs.get("🎞️ 上传视频"))
    if material_type == "audio":
        return _audio_input_to_upload_file(kwargs.get("🎵 上传音频"))
    if material_type == "file":
        return _path_to_upload_file(kwargs.get("📂 本地文件路径", ""))
    return None


def _resolve_asset_source_url(material_type, kwargs):
    source_url = (kwargs.get("🔗 素材URL", "") or "").strip()
    if source_url:
        return source_url

    material_type = (material_type or "").lower()
    if material_type == "image":
        return _image_tensor_to_data_url(kwargs.get("🖼️ 上传图片"))
    if material_type == "video":
        return _video_input_to_data_url(kwargs.get("🎞️ 上传视频"))
    if material_type == "audio":
        return _audio_input_to_data_url(kwargs.get("🎵 上传音频"))
    return ""


def _normalize_asset_type(material_type):
    mapping = {
        "image": "Image",
        "video": "Video",
        "audio": "Audio",
    }
    return mapping.get((material_type or "").lower(), "Image")


def _upload_file_bytes(base_url, api_key, file_name, file_bytes, mime_type):
    response = requests.post(
        f"{base_url.rstrip('/')}{DEFAULT_FILE_UPLOAD_ENDPOINT}",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (file_name, file_bytes, mime_type)},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict):
        return str(result.get("url") or result.get("data", {}).get("url") or "")
    return ""


def _download_and_reupload_asset(base_url, api_key, source_url, material_type):
    response = requests.get(source_url, timeout=120)
    response.raise_for_status()

    asset_type = _normalize_asset_type(material_type)
    fallback_mapping = {
        "Image": ("upload.png", "image/png"),
        "Video": ("upload.mp4", "video/mp4"),
        "Audio": ("upload.mp3", "audio/mpeg"),
    }
    default_name, default_mime = fallback_mapping.get(asset_type, ("upload.bin", "application/octet-stream"))
    content_type = response.headers.get("Content-Type", default_mime) or default_mime
    file_name = default_name
    return _upload_file_bytes(base_url, api_key, file_name, response.content, content_type)


def _kie_upload_file_bytes(api_key, file_name, file_bytes, mime_type, upload_path):
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                f"{KIE_UPLOAD_BASE_URL}{KIE_FILE_STREAM_UPLOAD_ENDPOINT}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Connection": "close",
                },
                files={"file": (file_name, file_bytes, mime_type)},
                data={"uploadPath": upload_path, "fileName": file_name},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                return ""
            data = result.get("data", {})
            return str(data.get("fileUrl") or data.get("downloadUrl") or "")
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as error:
            last_error = error
            if attempt >= 2:
                raise
            time.sleep(attempt + 1)
    if last_error is not None:
        raise last_error
    return ""


def _make_unique_kie_upload_name(file_name, file_bytes):
    base_name = os.path.basename(str(file_name or "").strip()) or "upload.bin"
    name_root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".bin"
    content_hash = hashlib.sha1(file_bytes).hexdigest()[:10] if file_bytes else "empty"
    timestamp = time.time_ns()
    safe_root = name_root or "upload"
    return f"{safe_root}_{timestamp}_{content_hash}{ext}"


def _kie_upload_media_input(api_key, media_input, media_type, upload_path):
    upload_file = None
    if media_type == "image":
        upload_file = _image_tensor_to_upload_file(media_input, "kie_seedance_image.jpg")
    elif media_type == "video":
        upload_file = _video_input_to_upload_file(media_input, "kie_seedance_video.mp4")
    elif media_type == "audio":
        upload_file = _audio_input_to_upload_file(media_input, "kie_seedance_audio.wav")

    if upload_file is None:
        return ""

    file_name, file_bytes, mime_type = upload_file
    unique_file_name = _make_unique_kie_upload_name(file_name, file_bytes)
    return _kie_upload_file_bytes(api_key, unique_file_name, file_bytes, mime_type, upload_path)


def _collect_kie_media_urls(api_key, media_type, values, upload_path):
    urls = []
    for value in values:
        if _is_empty_media_input(value):
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                urls.append(stripped)
            continue
        uploaded_url = _kie_upload_media_input(api_key, value, media_type, upload_path)
        if uploaded_url:
            urls.append(uploaded_url)
    return urls


def _validate_kie_input_payload(input_payload):
    has_first_frame = bool(str(input_payload.get("first_frame_url", "") or "").strip())
    has_last_frame = bool(str(input_payload.get("last_frame_url", "") or "").strip())
    has_reference_media = any([
        bool(input_payload.get("reference_image_urls")),
        bool(input_payload.get("reference_video_urls")),
        bool(input_payload.get("reference_audio_urls")),
    ])

    if has_reference_media and (has_first_frame or has_last_frame):
        raise ValueError(
            "KIE 官方限制：图生视频（首帧）、图生视频（首帧+尾帧）与多模态参考是互斥场景，不能同时使用。"
            "请只保留其中一种：1. 只用首帧；2. 只用首帧+尾帧；3. 只用参考图片/视频/音频。"
        )

    if has_last_frame and not has_first_frame:
        raise ValueError("KIE 图生视频模式下，使用尾帧图片时需要同时提供首帧图片。")


def _apply_kie_generation_mode(input_payload, generation_mode):
    generation_mode = str(generation_mode or "文生视频").strip()

    if generation_mode == "文生视频":
        input_payload.pop("first_frame_url", None)
        input_payload.pop("last_frame_url", None)
        input_payload.pop("reference_image_urls", None)
        input_payload.pop("reference_video_urls", None)
        input_payload.pop("reference_audio_urls", None)
        return input_payload

    if generation_mode == "首帧图生视频":
        input_payload.pop("last_frame_url", None)
        input_payload.pop("reference_image_urls", None)
        input_payload.pop("reference_video_urls", None)
        input_payload.pop("reference_audio_urls", None)
        if not str(input_payload.get("first_frame_url", "") or "").strip():
            raise ValueError("当前生成模式为首帧图生视频，请至少提供一张首帧图片。")
        return input_payload

    if generation_mode == "首尾帧图生视频":
        input_payload.pop("reference_image_urls", None)
        input_payload.pop("reference_video_urls", None)
        input_payload.pop("reference_audio_urls", None)
        if not str(input_payload.get("first_frame_url", "") or "").strip():
            raise ValueError("当前生成模式为首尾帧图生视频，请提供首帧图片。")
        if not str(input_payload.get("last_frame_url", "") or "").strip():
            raise ValueError("当前生成模式为首尾帧图生视频，请提供尾帧图片。")
        return input_payload

    if generation_mode == "多模态参考":
        input_payload.pop("first_frame_url", None)
        input_payload.pop("last_frame_url", None)
        has_reference_media = any([
            bool(input_payload.get("reference_image_urls")),
            bool(input_payload.get("reference_video_urls")),
            bool(input_payload.get("reference_audio_urls")),
        ])
        if not has_reference_media:
            raise ValueError("当前生成模式为多模态参考，请至少提供参考图片、参考视频、参考音频中的一种。")
        return input_payload

    return input_payload


def _parse_kie_record_response(response_json):
    data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
    result_json = data.get("resultJson", "")
    result_payload = {}
    if isinstance(result_json, str) and result_json.strip():
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                result_payload = parsed
        except Exception:
            result_payload = {}
    elif isinstance(result_json, dict):
        result_payload = result_json
    return data, result_payload


def _find_kie_video_url(result_payload):
    return _find_first_url(result_payload, extensions=[".mp4", ".mov", ".webm", ".mkv"], preferred_keys=["resultUrls", "videoUrl", "video_url", "url"])


def _find_kie_last_frame_url(result_payload):
    return _find_first_url(result_payload, extensions=[".png", ".jpg", ".jpeg", ".webp"], preferred_keys=["lastFrameUrl", "last_frame_url", "resultUrls", "coverUrl"])


def _kie_state_is_success(state):
    return str(state or "").strip().lower() == "success"


def _kie_state_is_failed(state):
    return str(state or "").strip().lower() == "fail"


def _kie_query_asset_response(api_key, asset_task_id):
    response = requests.get(
        f"{KIE_DEFAULT_BASE_URL}{KIE_ASSET_QUERY_ENDPOINT}",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"assetId": asset_task_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _kie_query_task_response(api_key, task_id):
    response = requests.get(
        f"{KIE_DEFAULT_BASE_URL}{KIE_QUERY_ENDPOINT}",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"taskId": task_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _extract_kie_task_info(response_json, fallback_task_id=""):
    record_data, result_payload = _parse_kie_record_response(response_json if isinstance(response_json, dict) else {})
    task_id = str(record_data.get("taskId") or fallback_task_id or "")
    state = str(record_data.get("state", "") or "")
    video_url = _find_kie_video_url(result_payload)
    response_info = json.dumps(response_json, ensure_ascii=False) if isinstance(response_json, dict) else str(response_json or "")
    return task_id, state, video_url, response_info, record_data, result_payload


def _format_kie_task_report_line(state, task_id, prompt, model):
    prompt_text = str(prompt or "").replace("\n", " ").strip()
    if len(prompt_text) > 30:
        prompt_text = f"{prompt_text[:30]}..."
    model_text = f" [{model}]" if model else ""
    title_text = f" - {prompt_text}" if prompt_text else ""
    return f"[{state or 'unknown'}] {task_id}{model_text}{title_text}"


def _query_registered_kie_tasks(api_key, limit=20):
    records = _get_registered_tasks(api_key, namespace="kie")
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0), reverse=True)
    report_lines = ["--- KIE 任务队列总览 ---"]
    response_items = []

    for item in records[:limit]:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        prompt = item.get("prompt", "")
        model = item.get("model", "")
        try:
            response_json = _kie_query_task_response(api_key, task_id)
            resolved_task_id, state, video_url, response_info, _, _ = _extract_kie_task_info(response_json, task_id)
            _update_task_record(api_key, resolved_task_id, namespace="kie", status=state, video_url=video_url, response_info=response_info)
            report_lines.append(_format_kie_task_report_line(state, resolved_task_id, prompt, model))
            response_items.append({
                "task_id": resolved_task_id,
                "status": state,
                "video_url": video_url,
                "prompt": prompt,
                "model": model,
                "response": response_json,
            })
        except Exception as error:
            error_text = str(error)
            _update_task_record(api_key, task_id, namespace="kie", status=f"error: {error_text}")
            report_lines.append(_format_kie_task_report_line("error", task_id, prompt, model))
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


def _select_downloadable_kie_task(api_key):
    records = _get_registered_tasks(api_key, namespace="kie")
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0))
    fallback = None

    for item in records:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        try:
            response_json = _kie_query_task_response(api_key, task_id)
        except Exception:
            continue
        resolved_task_id, state, video_url, response_info, _, _ = _extract_kie_task_info(response_json, task_id)
        _update_task_record(api_key, resolved_task_id, namespace="kie", status=state, video_url=video_url, response_info=response_info)
        if _kie_state_is_success(state) and video_url:
            if not bool(item.get("downloaded", False)):
                return response_json, resolved_task_id
            fallback = (response_json, resolved_task_id)

    return fallback if fallback is not None else (None, "")


def _kie_resolve_asset_task_url(api_key, asset_task_id):
    if not asset_task_id:
        return ""
    response_json = _kie_query_asset_response(api_key, str(asset_task_id).strip())
    data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
    status = str((data.get("status") if isinstance(data, dict) else "") or response_json.get("status") or "")
    asset_url = str((data.get("url") if isinstance(data, dict) else "") or response_json.get("url") or "")
    if status != "Active":
        error_msg = str((data.get("errorMsg") if isinstance(data, dict) else "") or response_json.get("errorMsg") or "")
        raise ValueError(f"火山素材 {asset_task_id} 当前状态为 {status or 'Unknown'}，未激活前不能用于生成。{error_msg}".strip())
    if not asset_url:
        raise ValueError(f"火山素材 {asset_task_id} 已激活但未返回可用URL。")
    return asset_url


def _kie_resolve_asset_task_url_list(api_key, asset_task_ids):
    urls = []
    for asset_task_id in asset_task_ids or []:
        asset_task_id = str(asset_task_id or "").strip()
        if asset_task_id:
            urls.append(_kie_resolve_asset_task_url(api_key, asset_task_id))
    return urls


def _sanitize_asset_name(value):
    text = str(value or "").strip()
    if not text:
        return ""
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        elif char in {" ", "."}:
            sanitized.append("_")
    return "".join(sanitized).strip("_")


def _normalize_kie_asset_bundle(asset_bundle):
    if isinstance(asset_bundle, dict):
        return asset_bundle
    if isinstance(asset_bundle, str) and asset_bundle.strip():
        try:
            value = json.loads(asset_bundle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _asset_ref(asset_id):
    asset_id = str(asset_id or "").strip()
    return f"asset://{asset_id}" if asset_id else ""


def _normalize_zhenzhen_asset_bundle(asset_bundle):
    if isinstance(asset_bundle, dict):
        return asset_bundle
    if isinstance(asset_bundle, str) and asset_bundle.strip():
        try:
            value = json.loads(asset_bundle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}

def _find_first_url(value, extensions=None, preferred_keys=None):
    if preferred_keys is None:
        preferred_keys = []

    def has_extension(text):
        if not extensions:
            return True
        lower_text = text.lower()
        return any(ext in lower_text for ext in extensions)

    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                result = _find_first_url(value[key], extensions, preferred_keys)
                if result:
                    return result
        for key, item in value.items():
            if isinstance(item, str) and (item.startswith("http://") or item.startswith("https://")) and has_extension(item):
                return item
            result = _find_first_url(item, extensions, preferred_keys)
            if result:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_first_url(item, extensions, preferred_keys)
            if result:
                return result
    elif isinstance(value, str):
        if (value.startswith("http://") or value.startswith("https://")) and has_extension(value):
            return value

    return ""


def _status_is_success(status):
    normalized = (status or "").strip().lower()
    return normalized in {"succeeded", "success", "completed", "done"} or "success" in normalized or "succeed" in normalized


def _status_is_failed(status):
    normalized = (status or "").strip().lower()
    return normalized in {"failed", "error", "cancelled", "canceled"} or "fail" in normalized or "error" in normalized


def _download_video_preview(video_url, output_dir, task_id=""):
    response = requests.get(video_url, stream=True, timeout=120)
    response.raise_for_status()

    file_name = f"doubao_seedance2_{task_id or int(time.time())}.mp4"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    preview_tensor = empty_image()
    if imageio is not None:
        try:
            reader = imageio.get_reader(file_path)
            frames = []
            for frame in reader:
                frames.append(frame)
                if len(frames) >= 8:
                    break
            if frames:
                preview_tensor = torch.from_numpy(np.array(frames).astype(np.float32) / 255.0)
        except Exception:
            preview_tensor = empty_image()

    return preview_tensor, file_path


def _download_image_tensor(image_url):
    if not image_url:
        return empty_image()
    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return pil2tensor(image)
    except Exception:
        return empty_image()


def _submit_required_inputs():
    return {
        "📝 提示词": ("STRING", {
            "multiline": True,
            "default": "",
            "placeholder": "请输入文生视频或多模态参考生视频提示词"
        }),
        "🤖 模型名称": ([
            "doubao-seedance-2-0-fast-260128",
            "doubao-seedance-2-0-260128"
        ], {
            "default": "doubao-seedance-2-0-fast-260128"
        }),
        "⏱️ 时长(秒)": ("INT", {
            "default": 5,
            "min": 4,
            "max": 15
        }),
        "🖥️ 分辨率": (["480p", "720p", "1080p", "native1080p"], {
            "default": "480p"
        }),
        "📐 视频比例": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"], {
            "default": "16:9"
        }),
        "💧 水印": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false"
        }),
        "🎬 返回尾帧": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false"
        }),
        "🎵 生成音频": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false"
        }),
        "🌐 启用联网搜索": ("BOOLEAN", {
            "default": False,
            "label_on": "true",
            "label_off": "false"
        }),
        "🔑 API密钥": ("STRING", {
            "default": "",
            "placeholder": "环境变量 / config.json / 节点输入三选一"
        }),
        "🎲 随机种子": ("INT", {
            "default": 0,
            "min": 0,
            "max": 2147483647,
            "control_after_generate": "randomize"
        }),
        "🌐 BaseURL": ("STRING", {
            "default": DEFAULT_BASE_URL,
            "placeholder": "https://ai.t8star.org"
        }),
        "⏳ 最大等待(秒)": ("INT", {
            "default": 600,
            "min": 10,
            "max": 3600
        }),
        "🔁 查询间隔(秒)": ("INT", {
            "default": 2,
            "min": 1,
            "max": 60
        }),
        "➕ 额外参数": ("STRING", {
            "multiline": True,
            "default": "",
            "placeholder": "{\"watermark\": false}"
        }),
    }


def _submit_optional_inputs(include_task_id=False):
    inputs = {
        "🧩 素材绑定": ("ZHENZHEN_SD2_ASSET_BUNDLE",),
        "🎬 首帧图片": ("IMAGE",),
        "🏁 尾帧图片": ("IMAGE",),
        "🎞️ 参考视频1": ("VIDEO",),
        "🎞️ 参考视频2": ("VIDEO",),
        "🎞️ 参考视频3": ("VIDEO",),
        "🎞️ 参考视频4": ("VIDEO",),
        "🖼️ 图像参考1": ("IMAGE",),
        "🖼️ 图像参考2": ("IMAGE",),
        "🖼️ 图像参考3": ("IMAGE",),
        "🖼️ 图像参考4": ("IMAGE",),
        "🖼️ 图像参考5": ("IMAGE",),
        "🖼️ 图像参考6": ("IMAGE",),
        "🖼️ 图像参考7": ("IMAGE",),
        "🖼️ 图像参考8": ("IMAGE",),
        "🎵 参考音频1": ("AUDIO",),
        "🎵 参考音频2": ("AUDIO",),
        "🎵 参考音频3": ("AUDIO",),
        "🎵 参考音频4": ("AUDIO",),
    }
    if include_task_id:
        return {
            "🆔 任务ID": ("STRING", {
                "default": "",
                "placeholder": "填写后直接查询任务，不发起新生成"
            }),
            **inputs
        }
    return inputs


def _query_required_inputs():
    return {
        "🔑 API密钥": ("STRING", {
            "default": "",
            "placeholder": "环境变量 / config.json / 节点输入三选一"
        }),
    }


def _build_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _create_video_task(base_url, headers, payload):
    response = requests.post(
        f"{base_url.rstrip('/')}{CREATE_ENDPOINT}",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        return None, response.text
    return response.json(), ""


def _resolve_video_task_result(task_response, task_id, output_dir):
    response_info = json.dumps(task_response, ensure_ascii=False)
    status = task_response.get("status", "")
    video_url = _find_first_url(task_response, extensions=[".mp4", ".mov", ".webm", ".mkv"], preferred_keys=["video_url", "url", "file_url", "download_url"])
    last_frame_url = _find_first_url(task_response, extensions=[".png", ".jpg", ".jpeg", ".webp"], preferred_keys=["last_frame_url", "last_frame", "tail_frame_url", "tail_frame", "cover_url"])
    last_frame_tensor = _download_image_tensor(last_frame_url) if last_frame_url else empty_image()

    if video_url and _status_is_success(status):
        _, file_path = _download_video_preview(video_url, output_dir, task_id)
        video_output = InputImpl.VideoFromFile(file_path)
        return (video_output, file_path, task_id, response_info, last_frame_tensor)

    if _status_is_failed(status):
        return (block_video("视频任务执行失败，当前没有可保存的视频输出。"), "", task_id, response_info, last_frame_tensor)

    return (block_video("视频任务仍在排队或生成中，请使用任务ID再次查询。"), "", task_id, response_info, last_frame_tensor)


def _extract_video_task_info(task_response, task_id=""):
    response_info = json.dumps(task_response, ensure_ascii=False)
    status = str(task_response.get("status", "") or "")
    video_url = _find_first_url(task_response, extensions=[".mp4", ".mov", ".webm", ".mkv"], preferred_keys=["video_url", "url", "file_url", "download_url"])
    task_id = str(task_id or task_response.get("task_id") or task_response.get("id") or "")
    return task_id, status, video_url, response_info


def _parse_response_info_text(response_info_text):
    if not response_info_text or not str(response_info_text).strip():
        return {}
    try:
        value = json.loads(response_info_text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _query_task_once(base_url, headers, task_id):
    task_url = f"{base_url.rstrip('/')}{CREATE_ENDPOINT}/{task_id}"
    response = requests.get(task_url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _format_task_report_line(status, task_id, prompt, model):
    prompt_text = str(prompt or "").replace("\n", " ").strip()
    if len(prompt_text) > 30:
        prompt_text = f"{prompt_text[:30]}..."
    model_text = f" [{model}]" if model else ""
    title_text = f" - {prompt_text}" if prompt_text else ""
    return f"[{status or 'unknown'}] {task_id}{model_text}{title_text}"


def _query_registered_tasks(api_key, base_url, limit=20):
    headers = _build_headers(api_key)
    records = _get_registered_tasks(api_key)
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0), reverse=True)
    report_lines = ["--- 任务队列总览 ---"]
    response_items = []

    for item in records[:limit]:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        prompt = item.get("prompt", "")
        model = item.get("model", "")
        try:
            task_response = _query_task_once(base_url, headers, task_id)
            task_id, status, video_url, response_info = _extract_video_task_info(task_response, task_id)
            _update_task_record(api_key, task_id, status=status, video_url=video_url, response_info=response_info)
            report_lines.append(_format_task_report_line(status, task_id, prompt, model))
            response_items.append({
                "task_id": task_id,
                "status": status,
                "video_url": video_url,
                "prompt": prompt,
                "model": model,
                "response": task_response,
            })
        except Exception as error:
            error_text = str(error)
            _update_task_record(api_key, task_id, status=f"error: {error_text}")
            report_lines.append(_format_task_report_line(f"error", task_id, prompt, model))
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


def _select_downloadable_task(api_key, base_url):
    headers = _build_headers(api_key)
    records = _get_registered_tasks(api_key)
    records = sorted(records, key=lambda item: int(item.get("created_at", 0) or 0))
    fallback = None

    for item in records:
        task_id = str(item.get("task_id", "") or "")
        if not task_id:
            continue
        try:
            task_response = _query_task_once(base_url, headers, task_id)
        except Exception:
            continue
        task_id, status, video_url, response_info = _extract_video_task_info(task_response, task_id)
        _update_task_record(api_key, task_id, status=status, video_url=video_url, response_info=response_info)
        if _status_is_success(status) and video_url:
            if not bool(item.get("downloaded", False)):
                return task_response, task_id
            fallback = (task_response, task_id)

    return fallback if fallback is not None else (None, "")


def _poll_until_done(query_func, base_url, headers, task_id, max_wait_seconds, poll_interval_seconds):
    start_time = time.time()
    last_response = None

    while True:
        last_response = query_func(base_url, headers, task_id)
        status = last_response.get("status", "")
        if _status_is_success(status) or _status_is_failed(status):
            return last_response
        if time.time() - start_time >= max_wait_seconds:
            return last_response
        time.sleep(poll_interval_seconds)


class DoubaoSeedance20ZhenzhenNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _submit_required_inputs(),
            "optional": _submit_optional_inputs()
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "generate_video"
    CATEGORY = NODE_CATEGORY

    def _merge_asset_bundle(self, payload, asset_bundle):
        asset_bundle = _normalize_zhenzhen_asset_bundle(asset_bundle)
        if not asset_bundle:
            return

        images = list(payload.get("images", []))
        videos = list(payload.get("videos", []))
        audios = list(payload.get("audios", []))

        first_frame_asset_id = str(asset_bundle.get("first_frame", "") or "").strip()
        if first_frame_asset_id:
            images.insert(0, _asset_ref(first_frame_asset_id))

        for index in range(1, 9):
            asset_id = str(asset_bundle.get(f"ref_image{index}", "") or "").strip()
            if asset_id:
                images.append(_asset_ref(asset_id))

        last_frame_asset_id = str(asset_bundle.get("last_frame", "") or "").strip()
        if last_frame_asset_id:
            images.append(_asset_ref(last_frame_asset_id))

        for index in range(1, 5):
            asset_id = str(asset_bundle.get(f"video{index}", "") or "").strip()
            if asset_id:
                videos.append(_asset_ref(asset_id))

        for index in range(1, 5):
            asset_id = str(asset_bundle.get(f"audio{index}", "") or "").strip()
            if asset_id:
                audios.append(_asset_ref(asset_id))

        if images:
            payload["images"] = images
        if videos:
            payload["videos"] = videos
        if audios:
            payload["audios"] = audios

    def _build_payload(self, kwargs):
        payload = {
            "prompt": kwargs.get("📝 提示词", ""),
            "model": kwargs.get("🤖 模型名称", "doubao-seedance-2-0-fast-260128"),
            "duration": int(kwargs.get("⏱️ 时长(秒)", 5)),
            "resolution": kwargs.get("🖥️ 分辨率", "480p"),
            "ratio": kwargs.get("📐 视频比例", "16:9"),
            "watermark": bool(kwargs.get("💧 水印", False)),
            "return_last_frame": bool(kwargs.get("🎬 返回尾帧", False)),
            "generate_audio": bool(kwargs.get("🎵 生成音频", False)),
        }

        seed = int(kwargs.get("🎲 随机种子", 0))
        if seed > 0:
            payload["seed"] = seed

        if bool(kwargs.get("🌐 启用联网搜索", False)):
            payload["tools"] = [{"type": "web_search"}]

        images = []

        first_frame_image = _image_tensor_to_data_url(kwargs.get("🎬 首帧图片"))
        if first_frame_image:
            images.append(first_frame_image)

        for index in range(1, 9):
            data_url = _image_tensor_to_data_url(kwargs.get(f"🖼️ 图像参考{index}"))
            if data_url:
                images.append(data_url)

        last_frame_image = _image_tensor_to_data_url(kwargs.get("🏁 尾帧图片"))
        if last_frame_image:
            images.append(last_frame_image)

        if images:
            payload["images"] = images

        videos = []
        for index in range(1, 5):
            data_url = _video_input_to_data_url(kwargs.get(f"🎞️ 参考视频{index}"))
            if data_url:
                videos.append(data_url)
        if videos:
            payload["videos"] = videos

        audios = []
        for index in range(1, 5):
            data_url = _audio_input_to_data_url(kwargs.get(f"🎵 参考音频{index}"))
            if data_url:
                audios.append(data_url)
        if audios:
            payload["audios"] = audios

        asset_bundle = kwargs.get("🧩 素材绑定")
        if asset_bundle:
            self._merge_asset_bundle(payload, asset_bundle)

        extra_params = _safe_json_loads(kwargs.get("➕ 额外参数", ""))
        if extra_params:
            first_frame_asset_id = extra_params.pop("first_frame_asset_id", extra_params.pop("first_frame_asset_task_id", ""))
            if first_frame_asset_id:
                payload.setdefault("images", []).insert(0, _asset_ref(first_frame_asset_id))

            reference_image_asset_ids = extra_params.pop("reference_image_asset_ids", extra_params.pop("reference_image_asset_task_ids", []))
            if isinstance(reference_image_asset_ids, str):
                reference_image_asset_ids = [reference_image_asset_ids]
            for asset_id in reference_image_asset_ids:
                asset_ref = _asset_ref(asset_id)
                if asset_ref:
                    payload.setdefault("images", []).append(asset_ref)

            last_frame_asset_id = extra_params.pop("last_frame_asset_id", extra_params.pop("last_frame_asset_task_id", ""))
            if last_frame_asset_id:
                payload.setdefault("images", []).append(_asset_ref(last_frame_asset_id))

            reference_video_asset_ids = extra_params.pop("reference_video_asset_ids", extra_params.pop("reference_video_asset_task_ids", []))
            if isinstance(reference_video_asset_ids, str):
                reference_video_asset_ids = [reference_video_asset_ids]
            for asset_id in reference_video_asset_ids:
                asset_ref = _asset_ref(asset_id)
                if asset_ref:
                    payload.setdefault("videos", []).append(asset_ref)

            reference_audio_asset_ids = extra_params.pop("reference_audio_asset_ids", extra_params.pop("reference_audio_asset_task_ids", []))
            if isinstance(reference_audio_asset_ids, str):
                reference_audio_asset_ids = [reference_audio_asset_ids]
            for asset_id in reference_audio_asset_ids:
                asset_ref = _asset_ref(asset_id)
                if asset_ref:
                    payload.setdefault("audios", []).append(asset_ref)

            payload.update(extra_params)

        return payload

    def _query_task(self, base_url, headers, task_id):
        task_url = f"{base_url.rstrip('/')}{CREATE_ENDPOINT}/{task_id}"
        response = requests.get(task_url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()

    def generate_video(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = (kwargs.get("🌐 BaseURL", "") or "").strip() or DEFAULT_BASE_URL
        max_wait_seconds = int(kwargs.get("⏳ 最大等待(秒)", 600))
        poll_interval_seconds = int(kwargs.get("🔁 查询间隔(秒)", 2))
        headers = _build_headers(api_key)
        payload = self._build_payload(kwargs)
        create_response, error_text = _create_video_task(base_url, headers, payload)

        if create_response is None:
            return (block_video("视频任务创建失败，当前没有可保存的视频输出。"), "", "", error_text, empty_image())

        task_id = str(create_response.get("task_id") or create_response.get("id") or "")
        if not task_id:
            return (block_video("接口未返回任务ID，当前没有可保存的视频输出。"), "", "", json.dumps(create_response, ensure_ascii=False), empty_image())

        _register_task(api_key, task_id, payload)
        task_response = _poll_until_done(self._query_task, base_url, headers, task_id, max_wait_seconds, poll_interval_seconds)
        return _resolve_video_task_result(task_response, task_id, self.output_dir)


class DoubaoSeedance20ZhenzhenQueryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _query_required_inputs()
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📋 任务报告", "🧾 响应信息")
    FUNCTION = "query_task"
    CATEGORY = NODE_CATEGORY

    def query_task(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        report_text, response_info = _query_registered_tasks(api_key, DEFAULT_BASE_URL)
        return (report_text, response_info)


class DoubaoSeedance20ZhenzhenSubmitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _submit_required_inputs(),
            "optional": _submit_optional_inputs()
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("🆔 任务ID", "🧾 响应信息")
    FUNCTION = "submit_task"
    CATEGORY = NODE_CATEGORY

    def _build_payload(self, kwargs):
        return DoubaoSeedance20ZhenzhenNode()._build_payload(kwargs)

    def submit_task(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        base_url = (kwargs.get("🌐 BaseURL", "") or "").strip() or DEFAULT_BASE_URL
        headers = _build_headers(api_key)
        payload = self._build_payload(kwargs)
        create_response, error_text = _create_video_task(base_url, headers, payload)

        if create_response is None:
            return ("", error_text)

        task_id = str(create_response.get("task_id") or create_response.get("id") or "")
        _register_task(api_key, task_id, payload)
        return (task_id, json.dumps(create_response, ensure_ascii=False))


class DoubaoSeedance20ZhenzhenGetVideoNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "get_video"
    CATEGORY = NODE_CATEGORY

    def get_video(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        task_response, task_id = _select_downloadable_task(api_key, DEFAULT_BASE_URL)
        if not task_response:
            return (block_video("当前没有可下载的视频任务，请先提交任务或等待生成完成。"), "", "", "", empty_image())

        _, _, video_url, response_info = _extract_video_task_info(task_response, task_id)
        if not video_url:
            return (block_video("当前没有可下载的视频URL，请稍后再试。"), "", task_id, response_info, empty_image())

        last_frame_url = _find_first_url(task_response, extensions=[".png", ".jpg", ".jpeg", ".webp"], preferred_keys=["last_frame_url", "last_frame", "tail_frame_url", "tail_frame", "cover_url"])
        last_frame_tensor = _download_image_tensor(last_frame_url) if last_frame_url else empty_image()
        _, file_path = _download_video_preview(video_url, self.output_dir, task_id)
        video_output = InputImpl.VideoFromFile(file_path)
        _update_task_record(api_key, task_id, downloaded=True, last_download_path=file_path)
        return (video_output, file_path, task_id, response_info, last_frame_tensor)


class DoubaoSeedance20AssetUploadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📦 素材类型": (["image", "video", "audio"], {
                    "default": "image"
                }),
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "placeholder": "https://ai.t8star.org"
                }),
                "🛣️ 上传接口路径": ("STRING", {
                    "default": DEFAULT_ASSET_UPLOAD_ENDPOINT,
                    "placeholder": "/seedance/v3/assets/create"
                }),
                "🔗 素材URL": ("STRING", {
                    "default": "",
                    "placeholder": "请输入公开可访问的 https:// 图片/视频/音频 URL"
                }),
                "➕ 额外表单参数": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "{\"scene\":\"seedance\"}"
                }),
            },
            "optional": {
                "🖼️ 上传图片": ("IMAGE",),
                "🎞️ 上传视频": ("VIDEO",),
                "🎵 上传音频": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 素材ID", "📌 状态", "🧾 响应信息")
    FUNCTION = "upload_asset"
    CATEGORY = NODE_CATEGORY

    def upload_asset(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        material_type = kwargs.get("📦 素材类型", "image")
        base_url = (kwargs.get("🌐 BaseURL", "") or "").strip() or DEFAULT_BASE_URL
        endpoint = (kwargs.get("🛣️ 上传接口路径", "") or "").strip() or DEFAULT_ASSET_UPLOAD_ENDPOINT
        extra_body = _safe_json_loads(kwargs.get("➕ 额外表单参数", ""))
        upload_file = _resolve_asset_upload_file(material_type, kwargs)
        source_url = (kwargs.get("🔗 素材URL", "") or "").strip()

        try:
            if upload_file is not None:
                file_name, file_bytes, mime_type = upload_file
                source_url = _upload_file_bytes(base_url, api_key, file_name, file_bytes, mime_type)
            elif source_url:
                if not (source_url.startswith("http://") or source_url.startswith("https://")):
                    return ("", "", "素材URL必须是公网可访问的 http:// 或 https:// 地址。")
                source_url = _download_and_reupload_asset(base_url, api_key, source_url, material_type)
            else:
                return ("", "", "请填写素材URL，或接入图像/视频/音频输入后再上传。")
        except Exception as error:
            return ("", "", json.dumps({"code": "error", "message": f"素材转存失败: {str(error)}"}, ensure_ascii=False))

        if not source_url:
            return ("", "", "素材转存失败，未获取到可用于创建素材的URL。")

        payload = {
            "url": source_url,
            "assetType": _normalize_asset_type(material_type),
        }
        if extra_body:
            payload.update(extra_body)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            headers=headers,
            json=payload,
            timeout=120,
        )

        response_text = response.text
        try:
            response_json = response.json()
        except Exception:
            return ("", "", response_text)

        if response.status_code >= 400:
            return ("", "", json.dumps(response_json, ensure_ascii=False))

        data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
        asset_id = str(data.get("assetId") or data.get("asset_id") or response_json.get("assetId") or "")
        status = str(data.get("status") or response_json.get("status") or "")
        return (asset_id, status, json.dumps(response_json, ensure_ascii=False))


class DoubaoSeedance20AssetQueryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🆔 素材ID": ("STRING", {
                    "default": "",
                    "placeholder": "填写上传素材节点返回的 assetId"
                }),
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
                "🌐 BaseURL": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "placeholder": "https://ai.t8star.org"
                }),
                "🛣️ 查询接口路径": ("STRING", {
                    "default": DEFAULT_ASSET_QUERY_ENDPOINT,
                    "placeholder": "/seedance/v3/assets/query"
                }),
                "⏳ 最大等待(秒)": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 3600
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 素材ID", "📌 状态", "🔗 预览URL", "🧾 响应信息")
    FUNCTION = "query_asset"
    CATEGORY = NODE_CATEGORY

    def query_asset(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        asset_id = (kwargs.get("🆔 素材ID", "") or "").strip()
        if not asset_id:
            return ("", "", "", "请填写素材ID后再查询。")

        base_url = (kwargs.get("🌐 BaseURL", "") or "").strip() or DEFAULT_BASE_URL
        endpoint = (kwargs.get("🛣️ 查询接口路径", "") or "").strip() or DEFAULT_ASSET_QUERY_ENDPOINT
        max_wait_seconds = int(kwargs.get("⏳ 最大等待(秒)", 300))
        poll_interval_seconds = int(kwargs.get("🔁 查询间隔(秒)", 2))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response_json = {}
        start_time = time.time()
        while True:
            response = requests.post(
                f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
                headers=headers,
                json={"assetId": asset_id},
                timeout=60,
            )
            response.raise_for_status()
            response_json = response.json()
            data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
            resolved_asset_id = str(data.get("assetId") or asset_id)
            status = str(data.get("status") or response_json.get("status") or "")
            preview_url = str(data.get("previewUrl") or data.get("sourceUrl") or "")

            if status == "Active" or status == "Failed":
                return (resolved_asset_id, status, preview_url, json.dumps(response_json, ensure_ascii=False))

            if time.time() - start_time >= max_wait_seconds:
                return (resolved_asset_id, status or "Processing", preview_url, json.dumps(response_json, ensure_ascii=False))

            time.sleep(poll_interval_seconds)


class DoubaoSeedance20AssetIdBundleNode:
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

    RETURN_TYPES = ("ZHENZHEN_SD2_ASSET_BUNDLE",)
    RETURN_NAMES = ("🧩 素材绑定",)
    FUNCTION = "build_bundle"
    CATEGORY = NODE_CATEGORY

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


class DoubaoSeedance20KieNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📝 提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入文生视频或图生视频提示词"
                }),
                "🤖 模型名称": ([
                    "bytedance/seedance-2",
                    "bytedance/seedance-2-fast"
                ], {
                    "default": "bytedance/seedance-2-fast"
                }),
                "🎛️ 生成模式": ([
                    "文生视频",
                    "首帧图生视频",
                    "首尾帧图生视频",
                    "多模态参考"
                ], {
                    "default": "文生视频"
                }),
                "⏱️ 时长(秒)": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15
                }),
                "🖥️ 分辨率": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "📐 视频比例": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {
                    "default": "16:9"
                }),
                "🎬 返回尾帧": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false"
                }),
                "🎵 生成音频": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false"
                }),
                "🌐 启用联网搜索": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false"
                }),
                "🔞 NSFW检查": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false"
                }),
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
                "🌐 CallbackURL": ("STRING", {
                    "default": "",
                    "placeholder": "可选，不填则节点内部轮询"
                }),
                "⏳ 最大等待(秒)": ("INT", {
                    "default": 600,
                    "min": 10,
                    "max": 3600
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 60
                }),
                "➕ 额外参数": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "{\"custom_field\": true}"
                }),
            },
            "optional": {
                "🧩 资产任务绑定": ("KIE_SD2_ASSET_BUNDLE",),
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
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "generate_video"
    CATEGORY = KIE_NODE_CATEGORY

    def _merge_asset_bundle(self, api_key, input_payload, asset_bundle):
        bundle = _normalize_kie_asset_bundle(asset_bundle)
        if not bundle:
            return

        first_frame_asset_task_id = str(bundle.get("first_frame", "") or "").strip()
        if first_frame_asset_task_id:
            input_payload["first_frame_url"] = _kie_resolve_asset_task_url(api_key, first_frame_asset_task_id)

        last_frame_asset_task_id = str(bundle.get("last_frame", "") or "").strip()
        if last_frame_asset_task_id:
            input_payload["last_frame_url"] = _kie_resolve_asset_task_url(api_key, last_frame_asset_task_id)

        image_ids = [bundle.get(f"ref_image{index}", "") for index in range(1, 9)]
        image_urls = _kie_resolve_asset_task_url_list(api_key, image_ids)
        if image_urls:
            existing = input_payload.get("reference_image_urls", [])
            input_payload["reference_image_urls"] = image_urls + list(existing)

        video_ids = [bundle.get(f"video{index}", "") for index in range(1, 5)]
        video_urls = _kie_resolve_asset_task_url_list(api_key, video_ids)
        if video_urls:
            existing = input_payload.get("reference_video_urls", [])
            input_payload["reference_video_urls"] = video_urls + list(existing)

        audio_ids = [bundle.get(f"audio{index}", "") for index in range(1, 5)]
        audio_urls = _kie_resolve_asset_task_url_list(api_key, audio_ids)
        if audio_urls:
            existing = input_payload.get("reference_audio_urls", [])
            input_payload["reference_audio_urls"] = audio_urls + list(existing)

    def _merge_asset_task_params(self, api_key, input_payload, extra_params):
        first_frame_asset_task_id = str(extra_params.pop("first_frame_asset_task_id", "") or extra_params.pop("first_frame_asset_id", "") or "").strip()
        if first_frame_asset_task_id:
            input_payload["first_frame_url"] = _kie_resolve_asset_task_url(api_key, first_frame_asset_task_id)

        last_frame_asset_task_id = str(extra_params.pop("last_frame_asset_task_id", "") or extra_params.pop("last_frame_asset_id", "") or "").strip()
        if last_frame_asset_task_id:
            input_payload["last_frame_url"] = _kie_resolve_asset_task_url(api_key, last_frame_asset_task_id)

        image_asset_task_ids = extra_params.pop("reference_image_asset_task_ids", extra_params.pop("reference_image_asset_ids", []))
        if isinstance(image_asset_task_ids, str):
            image_asset_task_ids = [image_asset_task_ids]
        image_asset_urls = _kie_resolve_asset_task_url_list(api_key, image_asset_task_ids)
        if image_asset_urls:
            existing = input_payload.get("reference_image_urls", [])
            input_payload["reference_image_urls"] = image_asset_urls + list(existing)

        video_asset_task_ids = extra_params.pop("reference_video_asset_task_ids", extra_params.pop("reference_video_asset_ids", []))
        if isinstance(video_asset_task_ids, str):
            video_asset_task_ids = [video_asset_task_ids]
        video_asset_urls = _kie_resolve_asset_task_url_list(api_key, video_asset_task_ids)
        if video_asset_urls:
            existing = input_payload.get("reference_video_urls", [])
            input_payload["reference_video_urls"] = video_asset_urls + list(existing)

        audio_asset_task_ids = extra_params.pop("reference_audio_asset_task_ids", extra_params.pop("reference_audio_asset_ids", []))
        if isinstance(audio_asset_task_ids, str):
            audio_asset_task_ids = [audio_asset_task_ids]
        audio_asset_urls = _kie_resolve_asset_task_url_list(api_key, audio_asset_task_ids)
        if audio_asset_urls:
            existing = input_payload.get("reference_audio_urls", [])
            input_payload["reference_audio_urls"] = audio_asset_urls + list(existing)

        return extra_params

    def _build_input_payload(self, api_key, kwargs):
        input_payload = {
            "prompt": kwargs.get("📝 提示词", ""),
            "return_last_frame": bool(kwargs.get("🎬 返回尾帧", False)),
            "generate_audio": bool(kwargs.get("🎵 生成音频", False)),
            "resolution": kwargs.get("🖥️ 分辨率", "720p"),
            "aspect_ratio": kwargs.get("📐 视频比例", "16:9"),
            "duration": int(kwargs.get("⏱️ 时长(秒)", 5)),
            "web_search": bool(kwargs.get("🌐 启用联网搜索", False)),
            "nsfw_checker": bool(kwargs.get("🔞 NSFW检查", False)),
        }

        first_frame_url = _kie_upload_media_input(api_key, kwargs.get("🎬 首帧图片"), "image", "seedance2/frames")
        if first_frame_url:
            input_payload["first_frame_url"] = first_frame_url

        last_frame_url = _kie_upload_media_input(api_key, kwargs.get("🏁 尾帧图片"), "image", "seedance2/frames")
        if last_frame_url:
            input_payload["last_frame_url"] = last_frame_url

        image_urls = _collect_kie_media_urls(api_key, "image", [
            kwargs.get("🖼️ 参考图片1"),
            kwargs.get("🖼️ 参考图片2"),
            kwargs.get("🖼️ 参考图片3"),
            kwargs.get("🖼️ 参考图片4"),
            kwargs.get("🖼️ 参考图片5"),
            kwargs.get("🖼️ 参考图片6"),
            kwargs.get("🖼️ 参考图片7"),
            kwargs.get("🖼️ 参考图片8"),
        ], "seedance2/images")
        if image_urls:
            input_payload["reference_image_urls"] = image_urls

        video_urls = _collect_kie_media_urls(api_key, "video", [
            kwargs.get("🎞️ 参考视频1"),
            kwargs.get("🎞️ 参考视频2"),
            kwargs.get("🎞️ 参考视频3"),
            kwargs.get("🎞️ 参考视频4"),
        ], "seedance2/videos")
        if video_urls:
            input_payload["reference_video_urls"] = video_urls

        audio_urls = _collect_kie_media_urls(api_key, "audio", [
            kwargs.get("🎵 参考音频1"),
            kwargs.get("🎵 参考音频2"),
            kwargs.get("🎵 参考音频3"),
            kwargs.get("🎵 参考音频4"),
        ], "seedance2/audios")
        if audio_urls:
            input_payload["reference_audio_urls"] = audio_urls

        asset_bundle = kwargs.get("🧩 资产任务绑定")
        if asset_bundle:
            self._merge_asset_bundle(api_key, input_payload, asset_bundle)

        extra_params = _safe_json_loads(kwargs.get("➕ 额外参数", ""))
        if extra_params:
            extra_params = self._merge_asset_task_params(api_key, input_payload, extra_params)
            input_payload.update(extra_params)

        _apply_kie_generation_mode(input_payload, kwargs.get("🎛️ 生成模式", "文生视频"))
        _validate_kie_input_payload(input_payload)
        return input_payload

    def _query_task(self, headers, task_id):
        response = requests.get(
            f"{KIE_DEFAULT_BASE_URL}{KIE_QUERY_ENDPOINT}",
            headers=headers,
            params={"taskId": task_id},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def generate_video(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        input_payload = self._build_input_payload(api_key, kwargs)
        request_payload = {
            "model": kwargs.get("🤖 模型名称", "bytedance/seedance-2-fast"),
            "input": input_payload,
        }

        callback_url = (kwargs.get("🌐 CallbackURL", "") or "").strip()
        if callback_url:
            request_payload["callBackUrl"] = callback_url

        response = requests.post(
            f"{KIE_DEFAULT_BASE_URL}{KIE_CREATE_ENDPOINT}",
            headers=headers,
            json=request_payload,
            timeout=120,
        )
        if response.status_code >= 400:
            return (block_video("Kie 任务创建失败，当前没有可保存的视频输出。"), "", "", response.text, empty_image())

        create_response = response.json()
        data = create_response.get("data", {}) if isinstance(create_response, dict) else {}
        task_id = str(data.get("taskId") or create_response.get("taskId") or "")
        if not task_id:
            return (block_video("Kie 接口未返回 taskId，当前没有可保存的视频输出。"), "", "", json.dumps(create_response, ensure_ascii=False), empty_image())

        max_wait_seconds = int(kwargs.get("⏳ 最大等待(秒)", 600))
        poll_interval_seconds = int(kwargs.get("🔁 查询间隔(秒)", 3))
        last_response = None
        start_time = time.time()

        while True:
            last_response = self._query_task(headers, task_id)
            record_data, result_payload = _parse_kie_record_response(last_response)
            state = str(record_data.get("state", "") or "")
            if _kie_state_is_success(state) or _kie_state_is_failed(state):
                break
            if time.time() - start_time >= max_wait_seconds:
                break
            time.sleep(poll_interval_seconds)

        response_info = json.dumps(last_response, ensure_ascii=False) if last_response else json.dumps(create_response, ensure_ascii=False)
        record_data, result_payload = _parse_kie_record_response(last_response or {})
        state = str(record_data.get("state", "") or "")
        video_url = _find_kie_video_url(result_payload)
        last_frame_url = _find_kie_last_frame_url(result_payload)
        last_frame_tensor = _download_image_tensor(last_frame_url) if last_frame_url else empty_image()

        if _kie_state_is_success(state) and video_url:
            _, file_path = _download_video_preview(video_url, self.output_dir, task_id)
            video_output = InputImpl.VideoFromFile(file_path)
            return (video_output, file_path, task_id, response_info, last_frame_tensor)

        if _kie_state_is_failed(state):
            fail_message = str(record_data.get("failMsg") or "Kie 视频任务执行失败")
            return (block_video(fail_message), "", task_id, response_info, last_frame_tensor)

        return (block_video("Kie 视频任务仍在处理中，请增大等待时间后再试。"), "", task_id, response_info, last_frame_tensor)


class DoubaoSeedance20KieSubmitNode(DoubaoSeedance20KieNode):
    @classmethod
    def INPUT_TYPES(cls):
        return DoubaoSeedance20KieNode.INPUT_TYPES()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("🆔 任务ID", "🧾 响应信息")
    FUNCTION = "submit_task"
    CATEGORY = KIE_NODE_CATEGORY

    def submit_task(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        input_payload = self._build_input_payload(api_key, kwargs)
        request_payload = {
            "model": kwargs.get("🤖 模型名称", "bytedance/seedance-2-fast"),
            "input": input_payload,
        }

        callback_url = (kwargs.get("🌐 CallbackURL", "") or "").strip()
        if callback_url:
            request_payload["callBackUrl"] = callback_url

        response = requests.post(
            f"{KIE_DEFAULT_BASE_URL}{KIE_CREATE_ENDPOINT}",
            headers=headers,
            json=request_payload,
            timeout=120,
        )
        if response.status_code >= 400:
            return ("", response.text)

        create_response = response.json()
        data = create_response.get("data", {}) if isinstance(create_response, dict) else {}
        task_id = str(data.get("taskId") or create_response.get("taskId") or "")
        _register_task(api_key, task_id, {
            "prompt": kwargs.get("📝 提示词", ""),
            "model": kwargs.get("🤖 模型名称", "bytedance/seedance-2-fast"),
        }, namespace="kie")
        return (task_id, json.dumps(create_response, ensure_ascii=False))


class DoubaoSeedance20KieQueryTaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📋 任务报告", "🧾 响应信息")
    FUNCTION = "query_task"
    CATEGORY = KIE_NODE_CATEGORY

    def query_task(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        try:
            report_text, response_info = _query_registered_kie_tasks(api_key)
            return (report_text, response_info)
        except Exception as error:
            return ("", json.dumps({"code": "error", "message": str(error)}, ensure_ascii=False))


class DoubaoSeedance20KieGetVideoNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("🎬 视频", "📂 视频URI", "🆔 任务ID", "🧾 响应信息", "🏁 尾帧图")
    FUNCTION = "get_video"
    CATEGORY = KIE_NODE_CATEGORY

    def get_video(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        response_json, task_id = _select_downloadable_kie_task(api_key)
        if not response_json:
            return (block_video("当前没有可下载的 KIE 视频任务，请先提交任务或等待生成完成。"), "", "", "", empty_image())

        resolved_task_id, _, video_url, response_info, _, result_payload = _extract_kie_task_info(response_json, task_id)
        if not video_url:
            return (block_video("当前没有可下载的视频URL，请稍后再试。"), "", resolved_task_id, response_info, empty_image())

        last_frame_url = _find_kie_last_frame_url(result_payload)
        last_frame_tensor = _download_image_tensor(last_frame_url) if last_frame_url else empty_image()
        _, file_path = _download_video_preview(video_url, self.output_dir, resolved_task_id)
        video_output = InputImpl.VideoFromFile(file_path)
        _update_task_record(api_key, resolved_task_id, namespace="kie", downloaded=True, last_download_path=file_path)
        return (video_output, file_path, resolved_task_id, response_info, last_frame_tensor)


class DoubaoSeedance20KieCreateAssetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📦 素材类型": (["image", "video", "audio"], {
                    "default": "image"
                }),
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
                "🔗 素材URL": ("STRING", {
                    "default": "",
                    "placeholder": "可选，填写公网素材URL；不填时可直接使用媒体输入"
                }),
                "🏷️ 名称": ("STRING", {
                    "default": "",
                    "placeholder": "可选，用于标识素材用途，如 first_frame / ref_image1"
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "➕ 额外参数": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "{\"custom_field\": true}"
                }),
            },
            "optional": {
                "🖼️ 上传图片": ("IMAGE",),
                "🎞️ 上传视频": ("VIDEO",),
                "🎵 上传音频": ("AUDIO",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 资产任务ID", "📌 状态", "🧾 响应信息")
    FUNCTION = "create_asset"
    CATEGORY = KIE_NODE_CATEGORY

    @staticmethod
    def _create_asset(api_key, material_type, kwargs):
        source_url = (kwargs.get("🔗 素材URL", "") or "").strip()
        asset_name = _sanitize_asset_name(kwargs.get("🏷️ 名称", ""))
        media_input = None
        if material_type == "image":
            media_input = kwargs.get("🖼️ 上传图片")
        elif material_type == "video":
            media_input = kwargs.get("🎞️ 上传视频")
        elif material_type == "audio":
            media_input = kwargs.get("🎵 上传音频")

        if media_input is not None:
            upload_path = f"kie-assets/{material_type}"
            if asset_name:
                upload_path = f"{upload_path}/{asset_name}"
            source_url = _kie_upload_media_input(api_key, media_input, material_type, upload_path)
        elif source_url:
            if not (source_url.startswith("http://") or source_url.startswith("https://")):
                raise ValueError("素材URL必须是公网可访问的 http:// 或 https:// 地址。")
        else:
            raise ValueError("请填写素材URL，或接入图像/视频/音频输入后再创建资产。")

        if not source_url:
            raise ValueError("素材转存失败，未获取到可用于创建资产的URL。")

        payload = {
            "url": source_url,
            "assetType": _normalize_asset_type(material_type),
        }
        extra_params = _safe_json_loads(kwargs.get("➕ 额外参数", ""))
        if extra_params:
            payload.update(extra_params)

        response = requests.post(
            f"{KIE_DEFAULT_BASE_URL}{KIE_ASSET_CREATE_ENDPOINT}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        response_json = response.json()
        data = response_json.get("data") if isinstance(response_json, dict) else None
        if isinstance(data, dict):
            asset_task_id = str(data.get("id") or data.get("assetId") or data.get("taskId") or "")
        elif isinstance(data, str):
            asset_task_id = data
        else:
            asset_task_id = str(response_json.get("id") or response_json.get("assetId") or response_json.get("taskId") or "")
        return asset_task_id, response_json

    def create_asset(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        material_type = kwargs.get("📦 素材类型", "image")
        try:
            asset_task_id, create_response_json = self._create_asset(api_key, material_type, kwargs)
            if not asset_task_id:
                return ("", "", json.dumps(create_response_json, ensure_ascii=False))
            return (asset_task_id, "Processing", json.dumps(create_response_json, ensure_ascii=False))
        except Exception as error:
            return ("", "", json.dumps({"code": "error", "message": str(error)}, ensure_ascii=False))


class DoubaoSeedance20KieQueryAssetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🆔 资产任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "填写 createAsset 返回的资产任务ID"
                }),
                "🔑 API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "环境变量 / config.json / 节点输入三选一"
                }),
                "⏳ 最大等待(秒)": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 3600
                }),
                "🔁 查询间隔(秒)": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🆔 素材ID", "📌 状态", "🧾 响应信息")
    FUNCTION = "query_asset"
    CATEGORY = KIE_NODE_CATEGORY

    @staticmethod
    def _query_asset(api_key, asset_task_id):
        response = requests.get(
            f"{KIE_DEFAULT_BASE_URL}{KIE_ASSET_QUERY_ENDPOINT}",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"assetId": asset_task_id},
            timeout=60,
        )
        response.raise_for_status()
        response_json = response.json()
        data = response_json.get("data", {}) if isinstance(response_json, dict) else {}
        status = str((data.get("status") if isinstance(data, dict) else "") or response_json.get("status") or "")
        return status, response_json

    def query_asset(self, **kwargs):
        api_key = get_api_key(kwargs.get("🔑 API密钥", ""))
        if not api_key:
            raise ValueError("未提供 API 密钥，请填写节点输入，或在环境变量 / config.json 中配置。")

        asset_task_id = (kwargs.get("🆔 资产任务ID", "") or "").strip()
        if not asset_task_id:
            return ("", "", "请填写资产任务ID后再查询。")

        max_wait_seconds = int(kwargs.get("⏳ 最大等待(秒)", 300))
        poll_interval_seconds = int(kwargs.get("🔁 查询间隔(秒)", 2))

        try:
            query_response_json = {}
            status = "Processing"
            start_time = time.time()

            while True:
                status, query_response_json = self._query_asset(api_key, asset_task_id)
                if status == "Active" or status == "Failed":
                    break
                if time.time() - start_time >= max_wait_seconds:
                    break
                time.sleep(poll_interval_seconds)

            return (asset_task_id, status or "Processing", json.dumps(query_response_json, ensure_ascii=False))
        except Exception as error:
            return ("", "", json.dumps({"code": "error", "message": str(error)}, ensure_ascii=False))


class DoubaoSeedance20KieAssetIdBundleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬 首帧资产任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "asset-xxx"
                }),
                "🏁 尾帧资产任务ID": ("STRING", {
                    "default": "",
                    "placeholder": "asset-xxx"
                }),
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

    RETURN_TYPES = ("KIE_SD2_ASSET_BUNDLE",)
    RETURN_NAMES = ("🧩 资产任务绑定",)
    FUNCTION = "build_bundle"
    CATEGORY = KIE_NODE_CATEGORY

    def build_bundle(self, **kwargs):
        bundle = {
            "first_frame": kwargs.get("🎬 首帧资产任务ID", ""),
            "last_frame": kwargs.get("🏁 尾帧资产任务ID", ""),
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
    "DoubaoSeedance20ZhenzhenNode": DoubaoSeedance20ZhenzhenNode,
    "DoubaoSeedance20ZhenzhenQueryNode": DoubaoSeedance20ZhenzhenQueryNode,
    "DoubaoSeedance20ZhenzhenSubmitNode": DoubaoSeedance20ZhenzhenSubmitNode,
    "DoubaoSeedance20ZhenzhenGetVideoNode": DoubaoSeedance20ZhenzhenGetVideoNode,
    "DoubaoSeedance20AssetUploadNode": DoubaoSeedance20AssetUploadNode,
    "DoubaoSeedance20AssetQueryNode": DoubaoSeedance20AssetQueryNode,
    "DoubaoSeedance20AssetIdBundleNode": DoubaoSeedance20AssetIdBundleNode,
    "DoubaoSeedance20KieNode": DoubaoSeedance20KieNode,
    "DoubaoSeedance20KieSubmitNode": DoubaoSeedance20KieSubmitNode,
    "DoubaoSeedance20KieQueryTaskNode": DoubaoSeedance20KieQueryTaskNode,
    "DoubaoSeedance20KieGetVideoNode": DoubaoSeedance20KieGetVideoNode,
    "DoubaoSeedance20KieCreateAssetNode": DoubaoSeedance20KieCreateAssetNode,
    "DoubaoSeedance20KieQueryAssetNode": DoubaoSeedance20KieQueryAssetNode,
    "DoubaoSeedance20KieAssetIdBundleNode": DoubaoSeedance20KieAssetIdBundleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedance20ZhenzhenNode": "doubao-seedance2.0-zhenzhen",
    "DoubaoSeedance20ZhenzhenQueryNode": "doubao-seedance2.0-zhenzhen-查询任务",
    "DoubaoSeedance20ZhenzhenSubmitNode": "doubao-seedance2.0-zhenzhen-提交任务",
    "DoubaoSeedance20ZhenzhenGetVideoNode": "doubao-seedance2.0-zhenzhen-获取视频",
    "DoubaoSeedance20AssetUploadNode": "doubao-seedance2.0-zhenzhen-上传素材",
    "DoubaoSeedance20AssetQueryNode": "doubao-seedance2.0-zhenzhen-查询素材状态",
    "DoubaoSeedance20AssetIdBundleNode": "doubao-seedance2.0-zhenzhen-素材绑定",
    "DoubaoSeedance20KieNode": "doubao-seedance2.0-kie",
    "DoubaoSeedance20KieSubmitNode": "doubao-seedance2.0-kie-提交任务",
    "DoubaoSeedance20KieQueryTaskNode": "doubao-seedance2.0-kie-查询任务",
    "DoubaoSeedance20KieGetVideoNode": "doubao-seedance2.0-kie-获取视频",
    "DoubaoSeedance20KieCreateAssetNode": "doubao-seedance2.0-kie-创建火山素材",
    "DoubaoSeedance20KieQueryAssetNode": "doubao-seedance2.0-kie-查询火山素材",
    "DoubaoSeedance20KieAssetIdBundleNode": "doubao-seedance2.0-kie-资产任务绑定",
}
