import base64
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import requests as _requests
except Exception:
    _requests = None

try:
    from requests.adapters import HTTPAdapter
except Exception:
    HTTPAdapter = None

try:
    from urllib.parse import urlencode
except Exception:
    from urllib import urlencode

try:
    from urllib.request import Request, urlopen
except Exception:
    from urllib2 import Request, urlopen


KIE_CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
KIE_FILE_UPLOAD_URL = "https://kieai.redpandaai.co/api/file-base64-upload"


def pil2tensor(image):
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor2pil(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    images = []
    for frame in image:
        np_image = frame.detach().cpu().numpy()
        np_image = np.clip(np_image * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(np_image))
    return images


def _is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _plugin_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _read_local_config_api_key():
    config_path = os.path.join(_plugin_dir(), "config.json")
    if not os.path.exists(config_path):
        return ""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""

    for key in ("kie_api_key", "api_key", "KIE_API_KEY"):
        value = data.get(key)
        if _is_nonempty_string(value):
            return value.strip()
    return ""


def _resolve_api_key(widget_value):
    for env_key in ("KIE_API_KEY", "KIEAI_API_KEY"):
        env_value = os.environ.get(env_key, "")
        if _is_nonempty_string(env_value):
            return env_value.strip()

    config_value = _read_local_config_api_key()
    if config_value:
        return config_value

    if _is_nonempty_string(widget_value):
        return widget_value.strip()

    return ""


def _pick_from_kwargs(kwargs, *keys, default=None):
    for key in keys:
        if key in kwargs:
            return kwargs.get(key)
    return default


_TASKS_LOCK = threading.Lock()
_REQUESTS_SESSION_LOCAL = threading.local()
_SSL_CONTEXT = None
_SSL_CONTEXT_LOCK = threading.Lock()


def _tasks_dir():
    base_dir = None
    if folder_paths is not None:
        try:
            base_dir = folder_paths.get_temp_directory()
        except Exception:
            base_dir = None
    if not base_dir:
        base_dir = os.path.join(_plugin_dir(), ".runtime")

    tasks_dir = os.path.join(base_dir, "mw_kie_gpt2_async")
    if not os.path.isdir(tasks_dir):
        os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir


def _tasks_file_path():
    return os.path.join(_tasks_dir(), "tasks.json")


def _read_tasks():
    path = _tasks_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _write_tasks(tasks):
    path = _tasks_file_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def _task_state_from_data(task_data):
    state = str((task_data or {}).get("state") or (task_data or {}).get("status") or "").strip().lower()
    if not state and _extract_result_urls(task_data):
        return "success"
    return state


def _update_task_entry(task_entry, task_data, record_response):
    state = _task_state_from_data(task_data)
    task_entry["state"] = state or task_entry.get("state") or "pending"
    task_entry["updated_at"] = time.time()
    task_entry["record_response"] = record_response
    task_entry["resultJson"] = (task_data or {}).get("resultJson")
    task_entry["failMsg"] = (task_data or {}).get("failMsg") or (task_data or {}).get("message") or ""
    result_urls = _extract_result_urls(task_data)
    if result_urls:
        task_entry["result_urls"] = result_urls
    return task_entry


def _format_task_line(task_id, task_entry):
    state = (task_entry.get("state") or "pending").lower()
    if task_entry.get("downloaded") is True:
        state = "downloaded"
    prompt = (task_entry.get("prompt") or "").replace("\n", " ").strip()
    prompt = prompt[:28] + "..." if len(prompt) > 28 else prompt
    batch_id = (task_entry.get("batch_id") or "")[:12]
    return "[{}] {}... 批次:{} {}".format(state, task_id[:8], batch_id or "-", prompt)


def _prepare_input_payload(model, prompt, aspect_ratio, resolution, image_url, kwargs, resolved_api_key):
    input_payload = {
        "prompt": (prompt or "").strip(),
        "aspect_ratio": str(aspect_ratio or "auto").strip() or "auto",
        "resolution": str(resolution or "1K").strip() or "1K",
    }

    if model == "gpt-image-2-image-to-image":
        images_in = []
        for i in range(1, 17):
            fallback_key = "image" if i == 1 else f"image_{i}"
            images_in.append(_pick_from_kwargs(kwargs, f"🖼️ 图像{i}", fallback_key, default=None))

        input_urls = _resolve_input_urls(image_url, images_in, resolved_api_key)
        if not input_urls:
            raise ValueError("Image To Image 模式需要提供 1-16 个 image 输入接口或 image_url。")
        input_payload["input_urls"] = input_urls

    return input_payload


def _validate_kie_image2_params(model, aspect_ratio, resolution):
    model_value = str(model or "").strip()
    aspect_ratio_value = str(aspect_ratio or "auto").strip() or "auto"
    resolution_value = str(resolution or "1K").strip().upper() or "1K"

    if model_value not in ("gpt-image-2-text-to-image", "gpt-image-2-image-to-image"):
        return

    if aspect_ratio_value == "auto" and resolution_value != "1K":
        raise ValueError("Kie GPT Image 2 要求：图像比例为 auto 时，分辨率只能使用 1K。")

    if aspect_ratio_value == "1:1" and resolution_value == "4K":
        raise ValueError("Kie GPT Image 2 要求：图像比例为 1:1 时，不支持 4K 分辨率。")


def _build_url(url, params):
    if not params:
        return url
    query = urlencode(params)
    if "?" in url:
        return url + "&" + query
    return url + "?" + query


def _get_requests_session():
    if _requests is None:
        return None

    session = getattr(_REQUESTS_SESSION_LOCAL, "session", None)
    if session is not None:
        return session

    session = _requests.Session()
    if HTTPAdapter is not None:
        try:
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except Exception:
            pass
    _REQUESTS_SESSION_LOCAL.session = session
    return session


def _get_ssl_context():
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT

    with _SSL_CONTEXT_LOCK:
        if _SSL_CONTEXT is not None:
            return _SSL_CONTEXT

        try:
            import ssl

            ssl_context = ssl.create_default_context()
            try:
                if hasattr(ssl, "TLSVersion") and hasattr(ssl_context, "minimum_version"):
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            except Exception:
                pass
            _SSL_CONTEXT = ssl_context
        except Exception:
            _SSL_CONTEXT = None
    return _SSL_CONTEXT


def _is_retryable_network_error(error):
    message = str(error or "").lower()
    retry_keywords = (
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "ssl",
        "connection reset",
        "remote end closed connection",
        "timed out",
        "timeout",
    )
    return any(keyword in message for keyword in retry_keywords)


def _http_json(method, url, headers=None, params=None, json_body=None, timeout=300):
    headers = headers or {}
    final_url = _build_url(url, params)
    body_bytes = None
    if json_body is not None:
        body_bytes = json.dumps(json_body).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    if _requests is not None:
        last_error = None
        session = _get_requests_session()
        for retry_index in range(2):
            try:
                response = session.request(
                    method=method,
                    url=final_url,
                    headers=headers,
                    data=body_bytes,
                    timeout=timeout,
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
        # requests 在部分环境下会出现 SSL EOF，这里自动降级到 urllib 继续请求
        if last_error is not None:
            pass

    last_error = None
    request_headers = dict(headers, **{"Connection": "close"})
    for retry_index in range(3):
        request = Request(final_url, data=body_bytes, headers=request_headers)
        try:
            request.get_method = lambda: method
        except Exception:
            pass

        try:
            ssl_context = _get_ssl_context()
            if ssl_context is not None:
                response = urlopen(request, timeout=timeout, context=ssl_context)
            else:
                response = urlopen(request, timeout=timeout)
        except Exception as ssl_error:
            try:
                response = urlopen(request, timeout=timeout)
            except Exception as fallback_error:
                last_error = fallback_error
                if retry_index < 2 and _is_retryable_network_error(fallback_error):
                    time.sleep(1.0 + retry_index)
                    continue
                if retry_index < 2 and _is_retryable_network_error(ssl_error):
                    last_error = ssl_error
                    time.sleep(1.0 + retry_index)
                    continue
                raise fallback_error

        status_code = int(getattr(response, "getcode", lambda: 200)() or 200)
        raw = response.read()
        if status_code < 200 or status_code >= 300:
            raise ValueError("HTTP {}: {}".format(status_code, raw[:500]))
        return json.loads(raw.decode("utf-8"))

    if last_error is not None:
        raise last_error
    raise ValueError("HTTP 请求失败: {}".format(final_url))


def _http_download_bytes(url, headers=None, timeout=300):
    headers = headers or {}

    if _requests is not None:
        last_error = None
        session = _get_requests_session()
        for retry_index in range(2):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                status_code = int(getattr(response, "status_code", 0) or 0)
                if status_code < 200 or status_code >= 300:
                    raise ValueError("HTTP {}: {}".format(status_code, getattr(response, "text", "")))
                return response.content
            except Exception as e:
                last_error = e
                if retry_index == 0:
                    time.sleep(1.0)
                    continue
        if last_error is not None:
            pass

    last_error = None
    request_headers = dict(headers, **{"Connection": "close"})
    for retry_index in range(3):
        request = Request(url, headers=request_headers)
        try:
            ssl_context = _get_ssl_context()
            if ssl_context is not None:
                response = urlopen(request, timeout=timeout, context=ssl_context)
            else:
                response = urlopen(request, timeout=timeout)
        except Exception as ssl_error:
            try:
                response = urlopen(request, timeout=timeout)
            except Exception as fallback_error:
                last_error = fallback_error
                if retry_index < 2 and _is_retryable_network_error(fallback_error):
                    time.sleep(1.0 + retry_index)
                    continue
                if retry_index < 2 and _is_retryable_network_error(ssl_error):
                    last_error = ssl_error
                    time.sleep(1.0 + retry_index)
                    continue
                raise fallback_error

        status_code = int(getattr(response, "getcode", lambda: 200)() or 200)
        if status_code < 200 or status_code >= 300:
            raise ValueError("HTTP {}".format(status_code))
        return response.read()

    if last_error is not None:
        raise last_error
    raise ValueError("下载失败: {}".format(url))


def _extract_uploaded_url(payload):
    data = (payload or {}).get("data") or {}
    for key in ("downloadUrl", "fileUrl", "url"):
        value = data.get(key)
        if _is_nonempty_string(value):
            return value.strip()
    raise ValueError("上传图片失败: {}".format(payload))


def _recommended_worker_count(item_count, max_workers=4):
    return max(1, min(int(item_count or 1), int(max_workers)))


def _normalize_output_count(value):
    return max(1, min(int(value or 1), 16))


def _upload_single_pil_image(pil_image, api_key):
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format="PNG")

    file_name = "gpt_image_2_{}.png".format(uuid.uuid4().hex[:10])
    data_url = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")

    payload = _http_json(
        "POST",
        KIE_FILE_UPLOAD_URL,
        headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
        json_body={
            "base64Data": data_url,
            "uploadPath": "images/user-uploads",
            "fileName": file_name,
        },
        timeout=300,
    )
    return _extract_uploaded_url(payload)


def _upload_image_tensor(image_tensor, api_key):
    pil_images = tensor2pil(image_tensor)
    if not pil_images:
        return []

    uploaded_urls = [None] * len(pil_images)
    max_workers = _recommended_worker_count(len(pil_images))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_upload_single_pil_image, pil_image, api_key): index
            for index, pil_image in enumerate(pil_images)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            uploaded_urls[index] = future.result()

    return uploaded_urls


def _split_urls(text):
    if not _is_nonempty_string(text):
        return []

    parts = re.split(r"[\r\n,]+", text)
    return [part.strip() for part in parts if part.strip()]


def _empty_image_tensor():
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _file_to_data_url(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".bmp":
        mime = "image/bmp"

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return "data:{};base64,{}".format(mime, encoded)


def _upload_local_file(file_path, api_key):
    file_name = os.path.basename(file_path) or "gpt_image_2_input.png"
    payload = _http_json(
        "POST",
        KIE_FILE_UPLOAD_URL,
        headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
        json_body={
            "base64Data": _file_to_data_url(file_path),
            "uploadPath": "images/user-uploads",
            "fileName": file_name,
        },
        timeout=300,
    )
    return _extract_uploaded_url(payload)


def _list_local_image_files(directory_path):
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if not os.path.isfile(file_path):
            continue
        if os.path.splitext(file_name)[1].lower() in valid_exts:
            image_files.append(file_path)
    image_files.sort()
    return image_files


def _resolve_batch_output_dir(output_dir, default_prefix):
    base_output = None
    if folder_paths is not None:
        try:
            base_output = folder_paths.get_output_directory()
        except Exception:
            base_output = None
    if not base_output:
        base_output = os.path.join(_plugin_dir(), "output")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_output_dir = (output_dir or "").strip()
    if raw_output_dir:
        if os.path.isabs(raw_output_dir):
            final_output_dir = os.path.normpath(raw_output_dir)
        else:
            final_output_dir = os.path.normpath(os.path.join(base_output, raw_output_dir))
    else:
        final_output_dir = os.path.join(base_output, "{}_{}".format(default_prefix, timestamp))
    os.makedirs(final_output_dir, exist_ok=True)
    return final_output_dir


def _save_result_image(pil_image, output_dir, prefix, suffix):
    save_path = os.path.join(output_dir, "{}_{}.png".format(prefix, suffix))
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, "{}_{}_{}.png".format(prefix, suffix, counter))
        counter += 1
    pil_image.save(save_path, "PNG", compress_level=4)
    return save_path


def _build_ordered_input_urls(primary_url, reference_combo):
    ordered_refs = sorted(reference_combo, key=lambda item: int(item.get("slot_index", 999)))
    return [primary_url] + [item["url"] for item in ordered_refs]


def _pad_images_to_max_size(pil_images):
    if not pil_images:
        return []

    max_width = max(image.size[0] for image in pil_images)
    max_height = max(image.size[1] for image in pil_images)
    normalized = []
    for image in pil_images:
        image = image.convert("RGB")
        if image.size == (max_width, max_height):
            normalized.append(image)
            continue
        canvas = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        offset_x = (max_width - image.size[0]) // 2
        offset_y = (max_height - image.size[1]) // 2
        canvas.paste(image, (offset_x, offset_y))
        normalized.append(canvas)
    return normalized


def _images_to_batch_tensor(pil_images):
    normalized_images = _pad_images_to_max_size(pil_images)
    tensors = [pil2tensor(image) for image in normalized_images]
    return torch.cat(tensors, dim=0)


def _resolve_input_urls(image_url_text, image_tensors, api_key):
    urls = _split_urls(image_url_text)
    valid_tensors = [image_tensor for image_tensor in image_tensors if image_tensor is not None]
    if valid_tensors:
        indexed_results = [None] * len(valid_tensors)
        max_workers = _recommended_worker_count(len(valid_tensors))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_upload_image_tensor, image_tensor, api_key): index
                for index, image_tensor in enumerate(valid_tensors)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                indexed_results[index] = future.result()

        for uploaded_group in indexed_results:
            urls.extend(uploaded_group or [])
    deduped_urls = []
    for url in urls:
        if url not in deduped_urls:
            deduped_urls.append(url)
    return deduped_urls[:16]


def _extract_http_urls_from_text(text):
    if not _is_nonempty_string(text):
        return []
    return re.findall(r"https?://[^\s\"'\\]+", text)


def _collect_urls_from_value(value, results):
    if isinstance(value, dict):
        for candidate_key in ("url", "imageUrl", "image_url", "resultUrl", "result_url", "downloadUrl"):
            candidate = value.get(candidate_key)
            if _is_nonempty_string(candidate):
                results.append(candidate.strip())
        for nested_value in value.values():
            _collect_urls_from_value(nested_value, results)
        return

    if isinstance(value, list):
        for item in value:
            _collect_urls_from_value(item, results)
        return

    if _is_nonempty_string(value):
        if value.strip().startswith("http"):
            results.append(value.strip())
        else:
            results.extend(_extract_http_urls_from_text(value))


def _extract_result_urls(task_data):
    results = []

    for key in ("resultUrls", "result_urls", "imageUrls", "image_urls", "images", "output", "outputs"):
        _collect_urls_from_value(task_data.get(key), results)

    response_data = task_data.get("response")
    if isinstance(response_data, (dict, list, str)):
        _collect_urls_from_value(response_data, results)

    result_json = task_data.get("resultJson")
    if _is_nonempty_string(result_json):
        try:
            parsed = json.loads(result_json)
        except Exception:
            parsed = result_json
        _collect_urls_from_value(parsed, results)

    deduped_urls = []
    for url in results:
        clean_url = url.strip()
        if clean_url.startswith("http") and clean_url not in deduped_urls:
            deduped_urls.append(clean_url)
    return deduped_urls


def _kie_create_task(api_key, model, input_payload, call_back_url=""):
    request_body = {
        "model": model,
        "input": input_payload,
    }
    if _is_nonempty_string(call_back_url):
        request_body["callBackUrl"] = call_back_url.strip()

    response = _http_json(
        "POST",
        KIE_CREATE_TASK_URL,
        headers={"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"},
        json_body=request_body,
        timeout=300,
    )

    code = response.get("code")
    if code is not None and int(code) != 200:
        raise ValueError("创建任务失败: {}".format(response.get("msg") or response))

    task_id = ((response.get("data") or {}).get("taskId")) or ""
    if not _is_nonempty_string(task_id):
        raise ValueError("createTask 未返回 taskId: {}".format(response))
    return task_id.strip(), response


def _create_tasks_concurrently(api_key, model, input_payload, call_back_url, output_count):
    total_task_count = _normalize_output_count(output_count)
    ordered_results = [None] * total_task_count
    max_workers = _recommended_worker_count(total_task_count, max_workers=8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_kie_create_task, api_key, model, input_payload, call_back_url): index
            for index in range(total_task_count)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            ordered_results[index] = future.result()

    task_ids = [item[0] for item in ordered_results]
    submit_responses = [item[1] for item in ordered_results]
    return task_ids, submit_responses


def _kie_get_record_info(api_key, task_id):
    response = _http_json(
        "GET",
        KIE_RECORD_INFO_URL,
        headers={"Authorization": "Bearer {}".format(api_key)},
        params={"taskId": task_id},
        timeout=120,
    )

    code = response.get("code")
    if code is not None and int(code) not in (200,):
        raise ValueError("查询任务失败: {}".format(response.get("msg") or response))

    data = response.get("data") or {}
    return data, response


def _kie_poll_result(api_key, task_id, poll_interval_seconds=3.0, max_wait_seconds=900.0, progress_callback=None):
    start_time = time.time()

    while True:
        try:
            data, response = _kie_get_record_info(api_key=api_key, task_id=task_id)
        except Exception as e:
            if _is_retryable_network_error(e):
                if time.time() - start_time >= max_wait_seconds:
                    raise TimeoutError("任务超时未完成: {}".format(task_id))
                time.sleep(max(1.0, float(poll_interval_seconds)))
                continue
            raise
        state = _task_state_from_data(data)
        result_urls = _extract_result_urls(data)

        if state in ("success", "succeeded", "completed", "finish", "finished"):
            return data, response

        if not state and result_urls:
            return data, response

        if state in ("fail", "failed", "error"):
            raise ValueError("任务失败: {}".format(data.get("failMsg") or data.get("message") or data))

        if time.time() - start_time >= max_wait_seconds:
            raise TimeoutError("任务超时未完成: {}".format(task_id))

        if progress_callback is not None:
            elapsed_seconds = time.time() - start_time
            if max_wait_seconds > 0:
                progress_ratio = min(1.0, elapsed_seconds / float(max_wait_seconds))
            else:
                progress_ratio = 0.0
            progress_callback(progress_ratio)

        elapsed_seconds = time.time() - start_time
        target_interval = max(0.2, float(poll_interval_seconds))
        if elapsed_seconds < 15.0:
            target_interval = min(target_interval, 0.8)
        time.sleep(target_interval)


def _poll_tasks_concurrently(api_key, task_ids, poll_interval_seconds=3.0, max_wait_seconds=900.0, progress_callback=None):
    if not task_ids:
        return []

    ordered_results = [None] * len(task_ids)
    max_workers = _recommended_worker_count(len(task_ids), max_workers=8)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _kie_poll_result,
                api_key,
                task_id,
                poll_interval_seconds,
                max_wait_seconds,
                None,
            ): index
            for index, task_id in enumerate(task_ids)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            ordered_results[index] = future.result()
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count / float(len(task_ids)))
    return ordered_results


def _download_result_images(image_urls):
    if not image_urls:
        raise ValueError("未下载到任何结果图片")

    pil_images = [None] * len(image_urls)
    max_workers = _recommended_worker_count(len(image_urls))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_http_download_bytes, image_url, None, 300): index
            for index, image_url in enumerate(image_urls)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            image_bytes = future.result()
            pil_images[index] = Image.open(BytesIO(image_bytes)).convert("RGB")
    if not pil_images:
        raise ValueError("未下载到任何结果图片")
    return pil_images


def _download_result_images_as_tensor(image_urls):
    pil_images = _download_result_images(image_urls)
    return _images_to_batch_tensor(pil_images)


class MWKieGPT20:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🤖 模型": (
                    [
                        "gpt-image-2-text-to-image",
                        "gpt-image-2-image-to-image",
                    ],
                    {"default": "gpt-image-2-text-to-image"},
                ),
                "📝 提示词": ("STRING", {"multiline": True, "default": ""}),
                "📐 图像比例": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "auto"}),
                "分辨率": (["1K", "2K", "4K"], {"default": "1K"}),
                "🖼️ 出图数量": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "⏱️ 轮询间隔(秒)": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 60.0, "step": 0.1}),
                "⌛ 最长等待(秒)": ("INT", {"default": 900, "min": 10, "max": 7200, "step": 10}),
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
                "🎲 随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True}),
                "🔔 回调地址": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ image", "🔗 image_urls", "🧾 response_json", "🆔 task_id")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    def generate(self, **kwargs):
        def _pick(*keys, default=None):
            return _pick_from_kwargs(kwargs, *keys, default=default)

        model = _pick("🤖 模型", "model", default="gpt-image-2-text-to-image")
        prompt = _pick("📝 提示词", "prompt", default="")
        aspect_ratio = _pick("📐 图像比例", "aspect_ratio", default="auto")
        resolution = _pick("分辨率", "resolution", default="1K")
        output_count = _normalize_output_count(_pick("🖼️ 出图数量", "output_count", default=1))
        poll_interval_seconds = float(_pick("⏱️ 轮询间隔(秒)", "poll_interval_seconds", default=1.0) or 1.0)
        max_wait_seconds = int(_pick("⌛ 最长等待(秒)", "max_wait_seconds", default=900) or 900)
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        image_url = _pick("🔗 图片URL", "image_url", default="") or ""
        local_seed = int(_pick("🎲 随机种子", "seed", default=0) or 0)
        call_back_url = _pick("🔔 回调地址", "call_back_url", default="") or ""

        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API Key，请优先使用环境变量 KIE_API_KEY，或在本地 config.json / 节点中填写。")

        _validate_kie_image2_params(model, aspect_ratio, resolution)

        input_payload = _prepare_input_payload(
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_url=image_url,
            kwargs=kwargs,
            resolved_api_key=resolved_api_key,
        )

        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(100)
        except Exception:
            class _DummyProgressBar:
                def update_absolute(self, _value):
                    return None

            progress_bar = _DummyProgressBar()

        task_ids = []
        submit_responses = []
        record_responses = []
        all_result_urls = []

        task_ids, submit_responses = _create_tasks_concurrently(
            api_key=resolved_api_key,
            model=model,
            input_payload=input_payload,
            call_back_url=call_back_url,
            output_count=output_count,
        )
        progress_bar.update_absolute(20)

        ordered_task_results = _poll_tasks_concurrently(
            api_key=resolved_api_key,
            task_ids=task_ids,
            poll_interval_seconds=float(poll_interval_seconds),
            max_wait_seconds=float(max_wait_seconds),
            progress_callback=lambda progress_ratio: progress_bar.update_absolute(20 + int(progress_ratio * 60)),
        )
        for task_data, record_response in ordered_task_results:
            record_responses.append(record_response)
            result_urls = _extract_result_urls(task_data)
            if not result_urls:
                raise ValueError("任务完成但未解析到图片地址: {}".format(task_data))
            all_result_urls.extend(result_urls)

        progress_bar.update_absolute(85)
        image_tensor = _download_result_images_as_tensor(all_result_urls)
        progress_bar.update_absolute(100)

        response_json = json.dumps(
            {
                "taskId": task_ids[0] if task_ids else "",
                "taskIds": task_ids,
                "model": model,
                "input": input_payload,
                "output_count": int(output_count),
                "local_seed": local_seed,
                "poll_interval_seconds": float(poll_interval_seconds),
                "max_wait_seconds": int(max_wait_seconds),
                "submit_response": submit_responses[0] if submit_responses else {},
                "submit_responses": submit_responses,
                "record_response": record_responses[0] if record_responses else {},
                "record_responses": record_responses,
                "result_urls": all_result_urls,
            },
            ensure_ascii=False,
        )

        return (image_tensor, "\n".join(all_result_urls), response_json, "\n".join(task_ids))


class MWKieGPT20SubmitTask:
    @classmethod
    def INPUT_TYPES(cls):
        return MWKieGPT20.INPUT_TYPES()

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🧾 response_json", "📋 report", "🆔 task_ids")
    FUNCTION = "submit"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, **kwargs):
        model = _pick_from_kwargs(kwargs, "🤖 模型", "model", default="gpt-image-2-text-to-image")
        prompt = _pick_from_kwargs(kwargs, "📝 提示词", "prompt", default="")
        aspect_ratio = _pick_from_kwargs(kwargs, "📐 图像比例", "aspect_ratio", default="auto")
        resolution = _pick_from_kwargs(kwargs, "分辨率", "resolution", default="1K")
        output_count = _normalize_output_count(_pick_from_kwargs(kwargs, "🖼️ 出图数量", "output_count", default=1))
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        image_url = _pick_from_kwargs(kwargs, "🔗 图片URL", "image_url", default="") or ""
        local_seed = int(_pick_from_kwargs(kwargs, "🎲 随机种子", "seed", default=0) or 0)
        call_back_url = _pick_from_kwargs(kwargs, "🔔 回调地址", "call_back_url", default="") or ""

        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API Key，请优先使用环境变量 KIE_API_KEY，或在本地 config.json / 节点中填写。")

        _validate_kie_image2_params(model, aspect_ratio, resolution)

        input_payload = _prepare_input_payload(
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_url=image_url,
            kwargs=kwargs,
            resolved_api_key=resolved_api_key,
        )

        batch_id = "batch_" + uuid.uuid4().hex[:12]
        task_ids, submit_responses = _create_tasks_concurrently(
            api_key=resolved_api_key,
            model=model,
            input_payload=input_payload,
            call_back_url=call_back_url,
            output_count=output_count,
        )

        with _TASKS_LOCK:
            tasks = _read_tasks()
            for index, (task_id, submit_response) in enumerate(zip(task_ids, submit_responses), start=1):
                created_at = time.time()
                tasks[task_id] = {
                    "taskId": task_id,
                    "batch_id": batch_id,
                    "sequence": index,
                    "model": model,
                    "prompt": (prompt or "").strip(),
                    "input": input_payload,
                    "state": "pending",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "downloaded": False,
                    "result_urls": [],
                    "call_back_url": (call_back_url or "").strip(),
                    "local_seed": local_seed,
                    "submit_response": submit_response,
                }
            _write_tasks(tasks)

        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "task_ids": task_ids,
                "model": model,
                "input": input_payload,
                "output_count": int(output_count),
                "local_seed": local_seed,
                "submit_responses": submit_responses,
            },
            ensure_ascii=False,
        )
        report = "已提交任务，批次ID: {}，任务数: {}".format(batch_id, len(task_ids))
        return (response_json, report, "\n".join(task_ids))


class MWKieGPT20QueryQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "📦 批次ID(可选)": ("STRING", {"default": ""}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📋 report", "🧾 response_json")
    FUNCTION = "query"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def query(self, **kwargs):
        batch_id = (_pick_from_kwargs(kwargs, "📦 批次ID(可选)", "batch_id", default="") or "").strip()
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        resolved_api_key = _resolve_api_key(api_key) if api_key else _resolve_api_key("")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        filtered_ids = []
        for task_id, task_entry in tasks.items():
            if batch_id and (task_entry.get("batch_id") or "").strip() != batch_id:
                continue
            filtered_ids.append(task_id)

        updated = False
        if resolved_api_key:
            for task_id in filtered_ids:
                task_entry = tasks.get(task_id) or {}
                state = (task_entry.get("state") or "").lower()
                if task_entry.get("downloaded") is True:
                    continue
                if state in ("success", "downloaded", "fail", "failed", "error") and task_entry.get("result_urls"):
                    continue
                try:
                    task_data, record_response = _kie_get_record_info(resolved_api_key, task_id)
                    tasks[task_id] = _update_task_entry(task_entry, task_data, record_response)
                    updated = True
                except Exception:
                    continue

        if updated:
            with _TASKS_LOCK:
                _write_tasks(tasks)

        filtered_entries = []
        for task_id in filtered_ids:
            filtered_entries.append((task_id, tasks.get(task_id) or {}))
        filtered_entries.sort(key=lambda item: item[1].get("created_at", 0.0), reverse=True)

        counts = {
            "total": len(filtered_entries),
            "pending": 0,
            "success": 0,
            "downloaded": 0,
            "failed": 0,
        }
        lines = ["--- GPT2.0 异步队列总览 ---"]
        for task_id, task_entry in filtered_entries[:100]:
            state = (task_entry.get("state") or "pending").lower()
            if task_entry.get("downloaded") is True:
                counts["downloaded"] += 1
            elif state in ("success", "succeeded", "completed", "finish", "finished"):
                counts["success"] += 1
            elif state in ("fail", "failed", "error"):
                counts["failed"] += 1
            else:
                counts["pending"] += 1
            lines.append(_format_task_line(task_id, task_entry))

        if len(lines) == 1:
            lines.append("当前没有任务记录。")

        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "counts": counts,
                "tasks": [task_entry for _, task_entry in filtered_entries],
            },
            ensure_ascii=False,
        )
        return ("\n".join(lines), response_json)


class MWKieGPT20DownloadReady:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "📦 批次ID(可选)": ("STRING", {"default": ""}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ image", "🔗 image_urls", "🧾 response_json", "🆔 task_ids")
    FUNCTION = "download"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def download(self, **kwargs):
        batch_id = (_pick_from_kwargs(kwargs, "📦 批次ID(可选)", "batch_id", default="") or "").strip()
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少 API Key，请优先使用环境变量 KIE_API_KEY，或在本地 config.json / 节点中填写。")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        selected_ids = []
        for task_id, task_entry in tasks.items():
            if batch_id and (task_entry.get("batch_id") or "").strip() != batch_id:
                continue
            selected_ids.append(task_id)
        selected_ids.sort(key=lambda task_id: (tasks.get(task_id) or {}).get("created_at", 0.0))

        updated = False
        for task_id in selected_ids:
            task_entry = tasks.get(task_id) or {}
            if task_entry.get("downloaded") is True:
                continue
            state = (task_entry.get("state") or "").lower()
            if state not in ("success", "succeeded", "completed", "finish", "finished"):
                try:
                    task_data, record_response = _kie_get_record_info(resolved_api_key, task_id)
                    tasks[task_id] = _update_task_entry(task_entry, task_data, record_response)
                    updated = True
                except Exception:
                    continue

        ready_task_ids = []
        all_result_urls = []
        ready_entries = []
        for task_id in selected_ids:
            task_entry = tasks.get(task_id) or {}
            if task_entry.get("downloaded") is True:
                continue
            state = (task_entry.get("state") or "").lower()
            result_urls = task_entry.get("result_urls") or []
            if state in ("success", "succeeded", "completed", "finish", "finished") and result_urls:
                ready_task_ids.append(task_id)
                ready_entries.append(task_entry)
                all_result_urls.extend(result_urls)
                task_entry["downloaded"] = True
                task_entry["downloaded_at"] = time.time()
                updated = True

        if updated:
            with _TASKS_LOCK:
                _write_tasks(tasks)

        if not ready_task_ids or not all_result_urls:
            raise ValueError("当前无已完成任务可下载。")

        image_tensor = _download_result_images_as_tensor(all_result_urls)
        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "task_ids": ready_task_ids,
                "tasks": ready_entries,
                "result_urls": all_result_urls,
            },
            ensure_ascii=False,
        )
        return (image_tensor, "\n".join(all_result_urls), response_json, "\n".join(ready_task_ids))


class MWKieGPT20FolderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📁 图片文件夹": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "📤 输出文件夹": ("STRING", {"default": "", "placeholder": "输出文件夹路径（留空则自动创建）"}),
                "🤖 模型": (["gpt-image-2-image-to-image"], {"default": "gpt-image-2-image-to-image"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "auto"}),
                "分辨率": (["1K", "2K", "4K"], {"default": "1K"}),
                "🔁 单条Prompt执行次数": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
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
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            return _pick_from_kwargs(kwargs, *keys, default=default)

        directory_path = (_pick("📁 图片文件夹", "directory_path", default="") or "").strip()
        output_dir = (_pick("📤 输出文件夹", "output_dir", default="") or "").strip()
        model = _pick("🤖 模型", "model", default="gpt-image-2-image-to-image")
        max_concurrent_files = max(1, min(int(_pick("⚙️ 同时处理文件数", "max_concurrent_files", default=3) or 3), 32))
        aspect_ratio = _pick("📐 图像比例", "aspect_ratio", default="auto")
        resolution = _pick("分辨率", "resolution", default="1K")
        executions_per_prompt = _normalize_output_count(
            _pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1)
        )
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        image_2 = _pick("🖼️ 备用图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 备用图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 备用图像4", "image_4", default=None)

        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少 API Key，请优先使用环境变量 KIE_API_KEY，或在本地 config.json / 节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not directory_path or not os.path.isdir(directory_path):
            msg = "图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            _validate_kie_image2_params(model, aspect_ratio, resolution)
        except Exception as e:
            msg = str(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        image_files = _list_local_image_files(directory_path)
        if not image_files:
            msg = "文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "GPTImage2_Batch")

        extra_image_urls = []
        try:
            for image_tensor in (image_2, image_3, image_4):
                if image_tensor is None:
                    continue
                pil_images = tensor2pil(image_tensor)
                if not pil_images:
                    continue
                extra_image_urls.append(_upload_single_pil_image(pil_images[0], resolved_api_key))
        except Exception as e:
            msg = "备用图上传失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        preview_pils = []
        failed_list = []
        saved_paths = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()

        def _process_one_file(file_index, file_path):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_url = _upload_local_file(file_path, resolved_api_key)
            except Exception as e:
                return 0, 0, [], [], "{} 上传失败: {}".format(base_name, e)

            input_payload = {
                "prompt": fixed_prompt,
                "aspect_ratio": str(aspect_ratio or "auto").strip() or "auto",
                "resolution": str(resolution or "1K").strip() or "1K",
                "input_urls": [primary_url] + list(extra_image_urls),
            }

            try:
                task_ids, _submit_responses = _create_tasks_concurrently(
                    api_key=resolved_api_key,
                    model=model,
                    input_payload=input_payload,
                    call_back_url="",
                    output_count=executions_per_prompt,
                )
                task_results = _poll_tasks_concurrently(
                    api_key=resolved_api_key,
                    task_ids=task_ids,
                    poll_interval_seconds=1.0,
                    max_wait_seconds=900.0,
                    progress_callback=None,
                )
            except Exception as e:
                return 0, 0, [], [], "{} 生成失败: {}".format(base_name, e)

            file_success = 0
            file_preview = []
            file_saved_paths = []
            for run_index, (task_data, _record_response) in enumerate(task_results, start=1):
                result_urls = _extract_result_urls(task_data)
                if not result_urls:
                    continue
                pil_images = _download_result_images(result_urls)
                for image_index, pil_image in enumerate(pil_images, start=1):
                    prefix = "Img{:03d}_{}".format(file_index + 1, base_name)
                    if len(pil_images) == 1:
                        suffix = "P{:03d}".format(run_index)
                    else:
                        suffix = "P{:03d}_{}".format(run_index, image_index)
                    save_path = _save_result_image(pil_image, batch_output_dir, prefix, suffix)
                    file_saved_paths.append(save_path)
                    file_success += 1
                    if len(file_preview) < 15:
                        file_preview.append(pil_image)

            if file_success == 0:
                return 0, len(task_ids), [], [], "{} 未生成结果".format(base_name)
            return file_success, len(task_ids), file_preview, file_saved_paths, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {
                executor.submit(_process_one_file, index, file_path): file_path
                for index, file_path in enumerate(image_files)
            }
            for future in as_completed(future_map):
                file_path = future_map[future]
                file_name = os.path.basename(file_path)
                try:
                    success_count, task_count, file_preview, file_saved_paths, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, file_saved_paths, error_msg = 0, 0, [], [], str(e)

                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    saved_paths.extend(file_saved_paths)
                    if file_preview and len(preview_pils) < 15:
                        remain = 15 - len(preview_pils)
                        preview_pils.extend(file_preview[:remain])
                    if error_msg:
                        failed_list.append("{} -> {}".format(file_name, error_msg))

        preview_tensor = _images_to_batch_tensor(preview_pils[:15]) if preview_pils else _empty_image_tensor()
        status_lines = [
            "✅ GPT_image_2 文件夹批量处理完成",
            "📁 输入文件夹: {}".format(os.path.basename(directory_path.rstrip("\\/")) or directory_path),
            "🖼️ 输入图片数: {}".format(len(image_files)),
            "🧩 备用图数量: {}".format(len(extra_image_urls)),
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
        return {"ui": {"string": [status_report]}, "result": (preview_tensor, status_report)}


class MWKieGPT20DualFolderBatch:
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
                "🤖 模型": (["gpt-image-2-image-to-image"], {"default": "gpt-image-2-image-to-image"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "auto"}),
                "分辨率": (["1K", "2K", "4K"], {"default": "1K"}),
                "🔁 单条Prompt执行次数": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            },
            "optional": {
                "📝 固定提示词(必填)": ("STRING", {"multiline": True, "default": "", "placeholder": "填写统一提示词跑主图文件夹和参考图文件夹的全部组合"}),
                "🖼️ 参考图像1": ("IMAGE",),
                "🖼️ 参考图像2": ("IMAGE",),
                "🖼️ 参考图像3": ("IMAGE",),
                "🖼️ 参考图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("🖼️ 预览(最多15)", "📊 状态报告")
    FUNCTION = "execute"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/kie"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, **kwargs):
        def _pick(*keys, default=None):
            return _pick_from_kwargs(kwargs, *keys, default=default)

        primary_directory = (_pick("📁 主图片文件夹", "primary_directory", default="") or "").strip()
        reference_directory_1 = (_pick("📁 参考图文件夹1", "reference_directory_1", default="") or "").strip()
        output_dir = (_pick("📤 输出文件夹", "output_dir", default="") or "").strip()
        model = _pick("🤖 模型", "model", default="gpt-image-2-image-to-image")
        max_concurrent_files = max(1, min(int(_pick("⚙️ 同时处理文件数", "max_concurrent_files", default=3) or 3), 32))
        aspect_ratio = _pick("📐 图像比例", "aspect_ratio", default="auto")
        resolution = _pick("分辨率", "resolution", default="1K")
        executions_per_prompt = _normalize_output_count(
            _pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1)
        )
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        reference_image_1 = _pick("🖼️ 参考图像1", "reference_image_1", default=None)
        reference_image_2 = _pick("🖼️ 参考图像2", "reference_image_2", default=None)
        reference_image_3 = _pick("🖼️ 参考图像3", "reference_image_3", default=None)
        reference_image_4 = _pick("🖼️ 参考图像4", "reference_image_4", default=None)
        reference_directory_2 = (_pick("📁 参考图文件夹2", "reference_directory_2", default="") or "").strip()
        reference_directory_3 = (_pick("📁 参考图文件夹3", "reference_directory_3", default="") or "").strip()
        reference_directory_4 = (_pick("📁 参考图文件夹4", "reference_directory_4", default="") or "").strip()

        resolved_api_key = _resolve_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少 API Key，请优先使用环境变量 KIE_API_KEY，或在本地 config.json / 节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not primary_directory or not os.path.isdir(primary_directory):
            msg = "主图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        try:
            _validate_kie_image2_params(model, aspect_ratio, resolution)
        except Exception as e:
            msg = str(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        primary_files = _list_local_image_files(primary_directory)
        if not primary_files:
            msg = "主图片文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if reference_image_1 is None:
            if not reference_directory_1 or not os.path.isdir(reference_directory_1):
                msg = "参考图文件夹1不存在。"
                return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}
            reference_files_1 = _list_local_image_files(reference_directory_1)
            if not reference_files_1:
                msg = "参考图文件夹1内无有效图片。"
                return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "GPTImage2_DualFolderBatch")

        def _upload_tensor_as_single_reference(image_tensor, slot_label, slot_index):
            pil_images = tensor2pil(image_tensor)
            if not pil_images:
                raise ValueError("{} 未解析到有效图片".format(slot_label))
            return [
                {
                    "slot": slot_label,
                    "slot_index": int(slot_index),
                    "file_name": slot_label,
                    "url": _upload_single_pil_image(pil_images[0], resolved_api_key),
                }
            ]

        def _upload_folder_as_references(folder_path, slot_label, slot_index):
            if not folder_path:
                return []
            if not os.path.isdir(folder_path):
                raise ValueError("{} 不存在".format(slot_label))
            files = _list_local_image_files(folder_path)
            if not files:
                raise ValueError("{} 内无有效图片".format(slot_label))
            uploaded = []
            for file_path in files:
                uploaded.append(
                    {
                        "slot": slot_label,
                        "slot_index": int(slot_index),
                        "file_name": os.path.splitext(os.path.basename(file_path))[0],
                        "url": _upload_local_file(file_path, resolved_api_key),
                    }
                )
            return uploaded

        reference_groups = []
        slot_labels = []
        try:
            if reference_image_1 is not None:
                reference_groups.append(_upload_tensor_as_single_reference(reference_image_1, "参考槽位1", 1))
                slot_labels.append("参考槽位1(图片)")
            else:
                group_1 = _upload_folder_as_references(reference_directory_1, "参考槽位1", 1)
                reference_groups.append(group_1)
                slot_labels.append("参考槽位1(文件夹)")

            if reference_image_2 is not None:
                reference_groups.append(_upload_tensor_as_single_reference(reference_image_2, "参考槽位2", 2))
                slot_labels.append("参考槽位2(图片)")
            elif reference_directory_2:
                reference_groups.append(_upload_folder_as_references(reference_directory_2, "参考槽位2", 2))
                slot_labels.append("参考槽位2(文件夹)")

            if reference_image_3 is not None:
                reference_groups.append(_upload_tensor_as_single_reference(reference_image_3, "参考槽位3", 3))
                slot_labels.append("参考槽位3(图片)")
            elif reference_directory_3:
                reference_groups.append(_upload_folder_as_references(reference_directory_3, "参考槽位3", 3))
                slot_labels.append("参考槽位3(文件夹)")

            if reference_image_4 is not None:
                reference_groups.append(_upload_tensor_as_single_reference(reference_image_4, "参考槽位4", 4))
                slot_labels.append("参考槽位4(图片)")
            elif reference_directory_4:
                reference_groups.append(_upload_folder_as_references(reference_directory_4, "参考槽位4", 4))
                slot_labels.append("参考槽位4(文件夹)")
        except Exception as e:
            msg = "参考素材上传失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not reference_groups or not reference_groups[0]:
            msg = "参考槽位1未准备成功。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        preview_pils = []
        failed_list = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()
        reference_combos = list(product(*reference_groups))
        total_combinations = len(primary_files) * len(reference_combos)

        primary_entries = []
        for file_index, file_path in enumerate(primary_files):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_url = _upload_local_file(file_path, resolved_api_key)
            except Exception as e:
                failed_list.append("{} -> 上传失败: {}".format(base_name, e))
                continue
            primary_entries.append(
                {
                    "file_index": int(file_index),
                    "file_path": file_path,
                    "base_name": base_name,
                    "primary_url": primary_url,
                }
            )

        def _process_one_generation_job(primary_entry, combo_index, reference_combo):
            file_success = 0
            file_task_count = 0
            file_preview = []
            combo_name = " + ".join(item["file_name"] for item in reference_combo)
            input_payload = {
                "prompt": fixed_prompt,
                "aspect_ratio": str(aspect_ratio or "auto").strip() or "auto",
                "resolution": str(resolution or "1K").strip() or "1K",
                "input_urls": _build_ordered_input_urls(primary_entry["primary_url"], reference_combo),
            }

            try:
                task_ids, _submit_responses = _create_tasks_concurrently(
                    api_key=resolved_api_key,
                    model=model,
                    input_payload=input_payload,
                    call_back_url="",
                    output_count=executions_per_prompt,
                )
                task_results = _poll_tasks_concurrently(
                    api_key=resolved_api_key,
                    task_ids=task_ids,
                    poll_interval_seconds=1.0,
                    max_wait_seconds=900.0,
                    progress_callback=None,
                )
            except Exception as e:
                return 0, 0, [], "{} + {} 生成失败: {}".format(primary_entry["base_name"], combo_name, e)

            file_task_count += len(task_ids)
            for run_index, (task_data, _record_response) in enumerate(task_results, start=1):
                result_urls = _extract_result_urls(task_data)
                if not result_urls:
                    continue
                pil_images = _download_result_images(result_urls)
                for image_index, pil_image in enumerate(pil_images, start=1):
                    combo_parts = [
                        "S{}_{:s}".format(slot_index + 1, reference_item["file_name"])
                        for slot_index, reference_item in enumerate(reference_combo)
                    ]
                    prefix = "Img{:03d}_{}_C{:03d}_{}".format(
                        primary_entry["file_index"] + 1,
                        primary_entry["base_name"],
                        combo_index,
                        "_".join(combo_parts),
                    )
                    if len(pil_images) == 1:
                        suffix = "P{:03d}".format(run_index)
                    else:
                        suffix = "P{:03d}_{}".format(run_index, image_index)
                    _save_result_image(pil_image, batch_output_dir, prefix, suffix)
                    file_success += 1
                    if len(file_preview) < 15:
                        file_preview.append(pil_image)

            if file_success == 0:
                return 0, file_task_count, [], "{} + {} 未生成结果".format(primary_entry["base_name"], combo_name)
            return file_success, file_task_count, file_preview, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {}
            for primary_entry in primary_entries:
                for combo_index, reference_combo in enumerate(reference_combos, start=1):
                    future = executor.submit(_process_one_generation_job, primary_entry, combo_index, reference_combo)
                    future_map[future] = (primary_entry["base_name"], combo_index)

            for future in as_completed(future_map):
                try:
                    success_count, task_count, file_preview, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, error_msg = 0, 0, [], str(e)

                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    if file_preview and len(preview_pils) < 15:
                        remain = 15 - len(preview_pils)
                        preview_pils.extend(file_preview[:remain])
                    if error_msg:
                        failed_list.append(error_msg)

        preview_tensor = _images_to_batch_tensor(preview_pils[:15]) if preview_pils else _empty_image_tensor()
        status_lines = [
            "✅ GPT_image_2 双文件夹批量处理完成",
            "📁 主图片数: {}".format(len(primary_files)),
            "🧩 参考槽位数: {}".format(len(reference_groups)),
            "组合数: {}".format(len(primary_files) * total_combinations),
            "🔁 单条Prompt执行次数: {}".format(int(executions_per_prompt)),
            "⚙️ 同时处理文件数: {}".format(int(max_concurrent_files)),
            "📦 总任务数: {}".format(int(total_tasks)),
            "✅ 成功输出数: {}".format(int(total_success)),
            "❌ 失败项数: {}".format(len(failed_list)),
            "💾 输出目录: {}".format(batch_output_dir),
            "⚠️ 预览仅显示前 15 张，全量图片请查看输出文件夹。",
        ]
        status_lines.append("📚 已启用参考槽位: {}".format("、".join(slot_labels)))
        if failed_list:
            status_lines.append("")
            status_lines.append("--- 失败记录(前5个) ---")
            for item in failed_list[:5]:
                status_lines.append("• {}".format(item))
            if len(failed_list) > 5:
                status_lines.append("...以及其他 {} 个失败项".format(len(failed_list) - 5))

        status_report = "\n".join(status_lines)
        return {"ui": {"string": [status_report]}, "result": (preview_tensor, status_report)}


BLT_BASE_URL = "https://ai.t8star.org"
BLT_IMAGE_GENERATIONS_URL = BLT_BASE_URL + "/v1/images/generations"
BLT_IMAGE_EDITS_URL = BLT_BASE_URL + "/v1/images/edits"


def _read_local_config_first_nonempty(*keys):
    config_path = os.path.join(_plugin_dir(), "config.json")
    if not os.path.exists(config_path):
        return ""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    for key in keys:
        value = data.get(key)
        if _is_nonempty_string(value):
            return value.strip()
    return ""


def _resolve_blt_api_key(widget_value):
    for env_key in ("ZHENZHEN_API_KEY", "BLT_API_KEY", "BLTCY_API_KEY", "PLATO_API_KEY"):
        env_value = os.environ.get(env_key, "")
        if _is_nonempty_string(env_value):
            return env_value.strip()

    config_value = _read_local_config_first_nonempty("zhenzhen_api_key", "blt_api_key", "plato_api_key", "api_key")
    if config_value:
        return config_value

    if _is_nonempty_string(widget_value):
        return widget_value.strip()

    return ""


def _blt_headers_json(api_key):
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }


def _blt_headers_bearer(api_key):
    return {"Authorization": "Bearer {}".format(api_key)}


def _blt_post_multipart(api_key, url, data, files, params=None, timeout=300, max_retries=3):
    if _requests is None:
        raise ValueError("zhenzhen节点依赖 requests 模块。")

    session = _get_requests_session()
    final_url = _build_url(url, params or {})
    last_error = None
    for retry_index in range(max(1, int(max_retries))):
        try:
            response = session.post(
                final_url,
                headers=_blt_headers_bearer(api_key),
                data=data,
                files=files,
                timeout=timeout,
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
            if retry_index + 1 >= max(1, int(max_retries)):
                break
            if _is_retryable_network_error(e):
                time.sleep(1.0 + retry_index)
                continue
            time.sleep(1.0)

    if last_error is not None:
        raise last_error
    raise ValueError("zhenzhen multipart 请求失败")


def _blt_extract_task_id(payload):
    if not isinstance(payload, dict):
        return ""

    direct_value = payload.get("task_id") or payload.get("taskId")
    if _is_nonempty_string(direct_value):
        return direct_value.strip()

    data = payload.get("data")
    if _is_nonempty_string(data):
        return data.strip()
    if isinstance(data, dict):
        nested_value = data.get("task_id") or data.get("taskId") or data.get("id")
        if _is_nonempty_string(nested_value):
            return nested_value.strip()

    return ""


def _blt_collect_result_items(payload):
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "images", "output", "outputs"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        nested_data = data.get("data")
        if isinstance(nested_data, dict):
            for key in ("data", "images", "output", "outputs"):
                value = nested_data.get(key)
                if isinstance(value, list):
                    return value

    return []


def _blt_collect_result_urls(items):
    urls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("image_url") or item.get("imageUrl")
        if _is_nonempty_string(url):
            clean_url = url.strip()
            if clean_url not in urls:
                urls.append(clean_url)
    return urls


def _blt_decode_item_to_pil(item, max_retries=5, initial_timeout=300):
    if not isinstance(item, dict):
        return None

    b64_json = item.get("b64_json") or ""
    if _is_nonempty_string(b64_json):
        b64_value = b64_json.strip()
        if b64_value.startswith("data:image"):
            b64_value = b64_value.split(",", 1)[-1]
        image_bytes = base64.b64decode(b64_value)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    image_url = item.get("url") or item.get("image_url") or item.get("imageUrl") or ""
    if _is_nonempty_string(image_url):
        last_error = None
        for retry_index in range(max(1, int(max_retries))):
            try:
                image_bytes = _http_download_bytes(image_url.strip(), timeout=min(int(initial_timeout), 900))
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                last_error = e
                if retry_index + 1 >= max(1, int(max_retries)):
                    break
                time.sleep(1.0 + retry_index)
        if last_error is not None:
            raise last_error

    return None


def _blt_items_to_pil_images(items, max_retries=5, initial_timeout=300):
    pil_images = []
    for item in items:
        pil_image = _blt_decode_item_to_pil(item, max_retries=max_retries, initial_timeout=initial_timeout)
        if pil_image is not None:
            pil_images.append(pil_image)
    return pil_images


def _blt_task_state_from_payload(payload):
    inner = (payload or {}).get("data") or {}
    status_value = inner.get("status") or (payload or {}).get("status") or ""
    return str(status_value).strip().lower()


def _blt_get_task_info(api_key, task_id):
    response = _http_json(
        "GET",
        BLT_BASE_URL + "/v1/images/tasks/{}".format(task_id),
        headers=_blt_headers_bearer(api_key),
        timeout=120,
    )
    return response


def _blt_update_task_entry(task_entry, record_response):
    state = _blt_task_state_from_payload(record_response)
    inner = (record_response or {}).get("data") or {}
    task_entry["state"] = state or task_entry.get("state") or "pending"
    task_entry["updated_at"] = time.time()
    task_entry["record_response"] = record_response
    task_entry["progress"] = inner.get("progress") or task_entry.get("progress") or ""
    task_entry["failMsg"] = (
        inner.get("fail_reason")
        or inner.get("message")
        or (record_response or {}).get("message")
        or ""
    )
    result_urls = _blt_collect_result_urls(_blt_collect_result_items(record_response))
    if result_urls:
        task_entry["result_urls"] = result_urls
    return task_entry


def _blt_make_png_tuple(pil_image, file_name):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return (file_name, buffer, "image/png")


def _blt_pil_to_png_bytes(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blt_request_file_from_png_bytes(file_name, png_bytes, field_name="image"):
    return (field_name, (file_name, BytesIO(png_bytes), "image/png"))


def _blt_local_file_to_png_bytes(file_path):
    with Image.open(file_path) as image:
        return _blt_pil_to_png_bytes(image.convert("RGB"))


def _blt_first_tensor_to_png_bytes(image_tensor):
    pil_images = tensor2pil(image_tensor)
    if not pil_images:
        raise ValueError("未解析到有效图片")
    return _blt_pil_to_png_bytes(pil_images[0].convert("RGB"))


def _blt_collect_input_images(kwargs):
    images = []
    for index in range(1, 17):
        value = _pick_from_kwargs(
            kwargs,
            "🖼️ 图像{}".format(index),
            "image{}".format(index),
            "图像{}".format(index),
            default=None,
        )
        if value is None:
            continue
        for batch_index, pil_image in enumerate(tensor2pil(value), start=1):
            images.append(
                (
                    "image",
                    _blt_make_png_tuple(
                        pil_image.convert("RGB"),
                        "plato_image_{}_{}.png".format(index, batch_index),
                    ),
                )
            )
    return images


def _blt_mask_to_request_file(mask_tensor, image_tensor):
    if mask_tensor is None:
        return None
    if image_tensor is None:
        raise ValueError("使用 mask 时必须提供 image1。")

    if image_tensor.ndim == 4:
        ref_image = image_tensor[0]
    else:
        ref_image = image_tensor

    height = int(ref_image.shape[0])
    width = int(ref_image.shape[1])

    mask = mask_tensor
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("mask 的张量形状必须为 [B, H, W] 或 [H, W]。")

    if int(mask.shape[1]) != height or int(mask.shape[2]) != width:
        raise ValueError("mask 和 image1 尺寸必须一致。")

    alpha = (1.0 - mask[0].detach().cpu().numpy())
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    mask_image = Image.fromarray(rgba, mode="RGBA")
    return ("mask", _blt_make_png_tuple(mask_image, "plato_mask.png"))


def _blt_create_progress_bar():
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(100)
    except Exception:
        class _DummyProgressBar:
            def update_absolute(self, _value):
                return None

        return _DummyProgressBar()


def _blt_submit_async_generation_once(api_key, payload, webhook, timeout):
    return _http_json(
        "POST",
        BLT_IMAGE_GENERATIONS_URL,
        headers=_blt_headers_json(api_key),
        params={"async": "true", "webhook": webhook.strip()} if _is_nonempty_string(webhook) else {"async": "true"},
        json_body=payload,
        timeout=int(timeout),
    )


def _blt_submit_async_edit_once(api_key, data, request_files, webhook, timeout, max_retries):
    return _blt_post_multipart(
        api_key,
        BLT_IMAGE_EDITS_URL,
        data,
        request_files,
        params={"async": "true", "webhook": webhook.strip()} if _is_nonempty_string(webhook) else {"async": "true"},
        timeout=int(timeout),
        max_retries=max_retries,
    )


def _blt_submit_async_tasks_concurrently(submit_callable, output_count):
    total_task_count = max(1, min(int(output_count or 1), 10))
    ordered_results = [None] * total_task_count
    max_workers = _recommended_worker_count(total_task_count, max_workers=8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(submit_callable): index for index in range(total_task_count)}
        for future in as_completed(future_map):
            index = future_map[future]
            submit_response = future.result()
            task_id = _blt_extract_task_id(submit_response)
            if not task_id:
                raise ValueError("zhenzhen异步提交成功，但未返回 task_id: {}".format(submit_response))
            ordered_results[index] = (task_id, submit_response)

    task_ids = [item[0] for item in ordered_results]
    submit_responses = [item[1] for item in ordered_results]
    return task_ids, submit_responses


def _blt_poll_tasks_concurrently(api_key, task_ids, max_poll_attempts, poll_interval, max_retries, initial_timeout, pbar=None):
    ordered_results = [None] * len(task_ids)
    max_workers = _recommended_worker_count(len(task_ids), max_workers=8)

    class _SilentProgressBar:
        def update_absolute(self, _value):
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                MWBolatuGPT20._poll_async_task,
                api_key,
                task_id,
                max_poll_attempts,
                poll_interval,
                max_retries,
                initial_timeout,
                _SilentProgressBar(),
            ): index
            for index, task_id in enumerate(task_ids)
        }
        completed_count = 0
        for future in as_completed(future_map):
            index = future_map[future]
            ordered_results[index] = future.result()
            completed_count += 1
            if pbar is not None and len(task_ids) > 0:
                progress_ratio = float(completed_count) / float(len(task_ids))
                pbar.update_absolute(20 + int(progress_ratio * 80))

    return ordered_results


class MWBolatuGPT20:
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

    _SIZE_MAP = {
        ("1:1", "1k"): "1024x1024",
        ("1:1", "2k"): "2048x2048",
        ("1:1", "4k"): "2880x2880",
        ("16:9", "1k"): "1280x720",
        ("16:9", "2k"): "2560x1440",
        ("16:9", "4k"): "3840x2160",
        ("9:16", "1k"): "720x1280",
        ("9:16", "2k"): "1440x2560",
        ("9:16", "4k"): "2160x3840",
        ("4:3", "1k"): "1152x864",
        ("4:3", "2k"): "2304x1728",
        ("4:3", "4k"): "3264x2448",
        ("3:4", "1k"): "864x1152",
        ("3:4", "2k"): "1728x2304",
        ("3:4", "4k"): "2448x3264",
        ("3:2", "1k"): "1248x832",
        ("3:2", "2k"): "2496x1664",
        ("3:2", "4k"): "3504x2336",
        ("2:3", "1k"): "832x1248",
        ("2:3", "2k"): "1664x2496",
        ("2:3", "4k"): "2336x3504",
        ("5:4", "1k"): "1120x896",
        ("5:4", "2k"): "2240x1792",
        ("5:4", "4k"): "3200x2560",
        ("4:5", "1k"): "896x1120",
        ("4:5", "2k"): "1792x2240",
        ("4:5", "4k"): "2560x3200",
        ("21:9", "1k"): "1456x624",
        ("21:9", "2k"): "3024x1296",
        ("21:9", "4k"): "3696x1584",
        ("9:21", "1k"): "624x1456",
        ("9:21", "2k"): "1296x3024",
        ("9:21", "4k"): "1584x3696",
        ("2:1", "1k"): "2048x1024",
        ("2:1", "2k"): "2688x1344",
        ("2:1", "4k"): "3840x1920",
        ("1:2", "1k"): "1024x2048",
        ("1:2", "2k"): "1344x2688",
        ("1:2", "4k"): "1920x3840",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": ""}),
                "📐 图像比例": (cls._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "🖼️ 分辨率": (cls._RESOLUTION_CHOICES, {"default": "1K"}),
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
                "🎭 遮罩": ("MASK",),
                "🔑 API密钥": ("STRING", {"default": ""}),
                "🤖 模型": (["gpt-image-2", "gpt-image-2-all"], {"default": "gpt-image-2"}),
                "🖼️ 出图数量": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "✨ 质量": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "🧱 背景": (["auto", "opaque"], {"default": "auto"}),
                "🗂️ 输出格式": (["png", "jpeg", "webp"], {"default": "png"}),
                "🗜️ 输出压缩": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "🛡️ 审核强度": (["auto", "low"], {"default": "auto"}),
                "📦 返回格式": (["url", "b64_json"], {"default": "url"}),
                "🔄 异步模式": ("BOOLEAN", {"default": True}),
                "🔔 回调地址": ("STRING", {"default": ""}),
                "🔁 最大轮询次数": ("INT", {"default": 300, "min": 10, "max": 1000, "step": 1}),
                "⏱️ 轮询间隔(秒)": ("INT", {"default": 5, "min": 2, "max": 60, "step": 1}),
                "♻️ 最大重试次数": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "⌛ 初始超时(秒)": ("INT", {"default": 900, "min": 60, "max": 1200, "step": 10}),
                "🎲 随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True}),
                "⛑️ 跳过报错": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "🔗 图片地址", "🧾 响应")
    FUNCTION = "generate"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

    @classmethod
    def _normalize_resolution(cls, resolution):
        value = str(resolution or "1K").strip().upper()
        if value not in ("1K", "2K", "4K"):
            value = "1K"
        return value.lower()

    @classmethod
    def _get_size_from_params(cls, aspect_ratio, resolution):
        normalized_resolution = cls._normalize_resolution(resolution)
        size = cls._SIZE_MAP.get((str(aspect_ratio or "1:1").strip(), normalized_resolution))
        if size is None:
            return None, "不支持的比例与分辨率组合: {} × {}".format(aspect_ratio, resolution)
        return size, None

    @staticmethod
    def _build_generation_payload(
        prompt,
        model,
        n,
        quality,
        size,
        background,
        output_format,
        output_compression,
        moderation,
        response_format,
        seed,
    ):
        payload = {
            "prompt": (prompt or "").strip(),
            "model": model,
            "n": int(n),
            "quality": quality,
            "size": size,
        }
        if background != "auto":
            payload["background"] = background
        if output_format != "png":
            payload["output_format"] = output_format
        if int(output_compression) != 100:
            payload["output_compression"] = int(output_compression)
        if moderation != "auto":
            payload["moderation"] = moderation
        if response_format != "url":
            payload["response_format"] = response_format
        if int(seed or 0) > 0:
            payload["seed"] = int(seed)
        return payload

    @staticmethod
    def _build_edits_payload(
        prompt,
        model,
        n,
        quality,
        size,
        background,
        output_format,
        output_compression,
        moderation,
        response_format,
        seed,
    ):
        payload = {
            "prompt": (prompt or "").strip(),
            "model": model,
            "n": str(int(n)),
            "quality": quality,
            "size": size,
        }
        if background != "auto":
            payload["background"] = background
        if output_format != "png":
            payload["output_format"] = output_format
        if int(output_compression) != 100:
            payload["output_compression"] = str(int(output_compression))
        if moderation != "auto":
            payload["moderation"] = moderation
        if response_format != "url":
            payload["response_format"] = response_format
        if int(seed or 0) > 0:
            payload["seed"] = str(int(seed))
        return payload

    @staticmethod
    def _poll_async_task(api_key, task_id, max_poll_attempts, poll_interval, max_retries, initial_timeout, pbar):
        query_url = BLT_BASE_URL + "/v1/images/tasks/{}".format(task_id)
        for attempt in range(1, int(max_poll_attempts) + 1):
            time.sleep(float(poll_interval))
            status_response = _http_json(
                "GET",
                query_url,
                headers=_blt_headers_bearer(api_key),
                timeout=min(int(initial_timeout), 300),
            )
            inner = status_response.get("data") or {}
            status_value = str(inner.get("status") or status_response.get("status") or "").strip().upper()
            progress_value = inner.get("progress") or status_response.get("progress") or ""

            if isinstance(progress_value, str) and progress_value.endswith("%"):
                try:
                    progress_number = int(progress_value[:-1])
                    pbar.update_absolute(min(95, 20 + int(progress_number * 0.75)))
                except Exception:
                    pass

            if status_value in ("SUCCESS", "SUCCEEDED", "COMPLETED", "FINISH", "FINISHED"):
                items = _blt_collect_result_items(status_response)
                pil_images = _blt_items_to_pil_images(
                    items,
                    max_retries=max_retries,
                    initial_timeout=initial_timeout,
                )
                if not pil_images:
                    raise ValueError("异步任务已完成，但未解析到图片结果。")
                return status_response, pil_images, _blt_collect_result_urls(items)

            if status_value in ("FAILURE", "FAILED", "ERROR"):
                fail_reason = inner.get("fail_reason") or inner.get("message") or status_response.get("message") or status_response
                raise ValueError("异步任务失败: {}".format(fail_reason))

        raise TimeoutError("异步轮询超时: {}".format(task_id))

    @staticmethod
    def _response_text(model, mode_name, prompt, aspect_ratio, resolution, size, n, task_id, image_urls, payload):
        lines = [
            "MW-zhenzhen-GPT2.0",
            "模式: {}".format(mode_name),
            "模型: {}".format(model),
            "提示词: {}".format(prompt),
            "图像比例: {}".format(aspect_ratio),
            "分辨率: {}".format(resolution),
            "实际尺寸: {}".format(size),
            "出图数量: {}".format(int(n)),
        ]
        if _is_nonempty_string(task_id):
            lines.append("任务ID: {}".format(task_id))
        if image_urls:
            lines.append("图片地址:")
            lines.extend(image_urls)
        lines.append("")
        lines.append("原始响应:")
        lines.append(json.dumps(payload, ensure_ascii=False))
        return "\n".join(lines)

    def generate(self, **kwargs):
        blank_tensor = _empty_image_tensor()
        try:
            prompt = _pick_from_kwargs(kwargs, "📝 提示词", "prompt", default="")
            aspect_ratio = _pick_from_kwargs(kwargs, "📐 图像比例", "aspect_ratio", default="1:1")
            resolution = _pick_from_kwargs(kwargs, "🖼️ 分辨率", "resolution", default="1K")
            mask = _pick_from_kwargs(kwargs, "🎭 遮罩", "mask", default=None)
            api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="")
            model = _pick_from_kwargs(kwargs, "🤖 模型", "model", default="gpt-image-2")
            n = _pick_from_kwargs(kwargs, "🖼️ 出图数量", "n", default=1)
            quality = _pick_from_kwargs(kwargs, "✨ 质量", "quality", default="auto")
            background = _pick_from_kwargs(kwargs, "🧱 背景", "background", default="auto")
            output_format = _pick_from_kwargs(kwargs, "🗂️ 输出格式", "output_format", default="png")
            output_compression = _pick_from_kwargs(kwargs, "🗜️ 输出压缩", "output_compression", default=100)
            moderation = _pick_from_kwargs(kwargs, "🛡️ 审核强度", "moderation", default="auto")
            response_format = _pick_from_kwargs(kwargs, "📦 返回格式", "response_format", default="url")
            async_mode = _pick_from_kwargs(kwargs, "🔄 异步模式", "async_mode", default=True)
            webhook = _pick_from_kwargs(kwargs, "🔔 回调地址", "webhook", default="")
            max_poll_attempts = _pick_from_kwargs(kwargs, "🔁 最大轮询次数", "max_poll_attempts", default=300)
            poll_interval = _pick_from_kwargs(kwargs, "⏱️ 轮询间隔(秒)", "poll_interval", default=5)
            max_retries = _pick_from_kwargs(kwargs, "♻️ 最大重试次数", "max_retries", default=5)
            initial_timeout = _pick_from_kwargs(kwargs, "⌛ 初始超时(秒)", "initial_timeout", default=900)
            seed = _pick_from_kwargs(kwargs, "🎲 随机种子", "seed", default=0)
            skip_error = _pick_from_kwargs(kwargs, "⛑️ 跳过报错", "skip_error", default=False)
            image1 = _pick_from_kwargs(kwargs, "🖼️ 图像1", "image1", default=None)

            resolved_api_key = _resolve_blt_api_key(api_key)
            if not resolved_api_key:
                raise ValueError("缺少zhenzhen API Key，请优先使用环境变量 ZHENZHEN_API_KEY，或在本地 config.json / 节点中填写。")

            if model == "gpt-image-2-all" and self._normalize_resolution(resolution) != "1k":
                raise ValueError("gpt-image-2-all 目前仅支持 1K 分辨率。")

            size, error_msg = self._get_size_from_params(aspect_ratio, resolution)
            if error_msg:
                raise ValueError(error_msg)

            request_files = _blt_collect_input_images(kwargs)
            mode_name = "Image To Image" if request_files else "Text to Image"

            if mask is not None and image1 is None:
                raise ValueError("使用 mask 时必须提供 image1。")

            pbar = _blt_create_progress_bar()
            pbar.update_absolute(5)

            if request_files:
                data = self._build_edits_payload(
                    prompt,
                    model,
                    n,
                    quality,
                    size,
                    background,
                    output_format,
                    output_compression,
                    moderation,
                    response_format,
                    seed,
                )
                mask_file = _blt_mask_to_request_file(mask, image1)
                if mask_file is not None:
                    request_files = list(request_files) + [mask_file]

                if async_mode:
                    single_task_data = dict(data)
                    single_task_data["n"] = "1"

                    def _submit_one_edit_task():
                        fresh_files = _blt_collect_input_images(kwargs)
                        fresh_mask_file = _blt_mask_to_request_file(mask, image1)
                        if fresh_mask_file is not None:
                            fresh_files = list(fresh_files) + [fresh_mask_file]
                        return _blt_submit_async_edit_once(
                            resolved_api_key,
                            single_task_data,
                            fresh_files,
                            webhook,
                            initial_timeout,
                            max_retries,
                        )

                    task_ids, submit_responses = _blt_submit_async_tasks_concurrently(
                        _submit_one_edit_task,
                        n,
                    )
                    pbar.update_absolute(20)
                    ordered_results = _blt_poll_tasks_concurrently(
                        resolved_api_key,
                        task_ids,
                        max_poll_attempts,
                        poll_interval,
                        max_retries,
                        initial_timeout,
                        pbar,
                    )
                    pil_images = []
                    image_urls = []
                    final_responses = []
                    for final_response, task_pil_images, task_image_urls in ordered_results:
                        final_responses.append(final_response)
                        pil_images.extend(task_pil_images)
                        image_urls.extend(task_image_urls)
                    return (
                        _images_to_batch_tensor(pil_images),
                        "\n".join(image_urls),
                        self._response_text(
                            model,
                            mode_name,
                            prompt,
                            aspect_ratio,
                            resolution,
                            size,
                            n,
                            "\n".join(task_ids),
                            image_urls,
                            {"task_ids": task_ids, "submit_responses": submit_responses, "record_responses": final_responses},
                        ),
                    )

                sync_response = _blt_post_multipart(
                    resolved_api_key,
                    BLT_IMAGE_EDITS_URL,
                    data,
                    request_files,
                    timeout=int(initial_timeout),
                    max_retries=max_retries,
                )
                pil_images = _blt_items_to_pil_images(
                    _blt_collect_result_items(sync_response),
                    max_retries=max_retries,
                    initial_timeout=initial_timeout,
                )
                if not pil_images:
                    raise ValueError("zhenzhen同步图生图未返回图片结果。")
                image_urls = _blt_collect_result_urls(_blt_collect_result_items(sync_response))
                pbar.update_absolute(100)
                return (
                    _images_to_batch_tensor(pil_images),
                    "\n".join(image_urls),
                    self._response_text(model, mode_name, prompt, aspect_ratio, resolution, size, n, "", image_urls, sync_response),
                )

            payload = self._build_generation_payload(
                prompt,
                model,
                n,
                quality,
                size,
                background,
                output_format,
                output_compression,
                moderation,
                response_format,
                seed,
            )

            if async_mode:
                single_task_payload = dict(payload)
                single_task_payload["n"] = 1

                def _submit_one_generation_task():
                    return _blt_submit_async_generation_once(
                        resolved_api_key,
                        single_task_payload,
                        webhook,
                        initial_timeout,
                    )

                task_ids, submit_responses = _blt_submit_async_tasks_concurrently(
                    _submit_one_generation_task,
                    n,
                )
                pbar.update_absolute(20)
                ordered_results = _blt_poll_tasks_concurrently(
                    resolved_api_key,
                    task_ids,
                    max_poll_attempts,
                    poll_interval,
                    max_retries,
                    initial_timeout,
                    pbar,
                )
                pil_images = []
                image_urls = []
                final_responses = []
                for final_response, task_pil_images, task_image_urls in ordered_results:
                    final_responses.append(final_response)
                    pil_images.extend(task_pil_images)
                    image_urls.extend(task_image_urls)
                return (
                    _images_to_batch_tensor(pil_images),
                    "\n".join(image_urls),
                    self._response_text(
                        model,
                        mode_name,
                        prompt,
                        aspect_ratio,
                        resolution,
                        size,
                        n,
                        "\n".join(task_ids),
                        image_urls,
                        {"task_ids": task_ids, "submit_responses": submit_responses, "record_responses": final_responses},
                    ),
                )

            sync_response = _http_json(
                "POST",
                BLT_IMAGE_GENERATIONS_URL,
                headers=_blt_headers_json(resolved_api_key),
                json_body=payload,
                timeout=int(initial_timeout),
            )
            pil_images = _blt_items_to_pil_images(
                _blt_collect_result_items(sync_response),
                max_retries=max_retries,
                initial_timeout=initial_timeout,
            )
            if not pil_images:
                raise ValueError("zhenzhen同步文生图未返回图片结果。")
            image_urls = _blt_collect_result_urls(_blt_collect_result_items(sync_response))
            pbar.update_absolute(100)
            return (
                _images_to_batch_tensor(pil_images),
                "\n".join(image_urls),
                self._response_text(model, mode_name, prompt, aspect_ratio, resolution, size, n, "", image_urls, sync_response),
            )

        except Exception as e:
            if not skip_error:
                raise
            return (blank_tensor, "", "MW-zhenzhen-GPT2.0 error: {}".format(e))


class MWBolatuGPT20SubmitTask:
    @classmethod
    def INPUT_TYPES(cls):
        return MWBolatuGPT20.INPUT_TYPES()

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("🧾 response_json", "📋 report", "🆔 task_ids")
    FUNCTION = "submit"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, **kwargs):
        prompt = _pick_from_kwargs(kwargs, "📝 提示词", "prompt", default="")
        aspect_ratio = _pick_from_kwargs(kwargs, "📐 图像比例", "aspect_ratio", default="1:1")
        resolution = _pick_from_kwargs(kwargs, "🖼️ 分辨率", "resolution", default="1K")
        mask = _pick_from_kwargs(kwargs, "🎭 遮罩", "mask", default=None)
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        model = _pick_from_kwargs(kwargs, "🤖 模型", "model", default="gpt-image-2")
        n = int(_pick_from_kwargs(kwargs, "🖼️ 出图数量", "n", default=1) or 1)
        quality = _pick_from_kwargs(kwargs, "✨ 质量", "quality", default="auto")
        background = _pick_from_kwargs(kwargs, "🧱 背景", "background", default="auto")
        output_format = _pick_from_kwargs(kwargs, "🗂️ 输出格式", "output_format", default="png")
        output_compression = int(
            _pick_from_kwargs(kwargs, "🗜️ 输出压缩", "output_compression", default=100) or 100
        )
        moderation = _pick_from_kwargs(kwargs, "🛡️ 审核强度", "moderation", default="auto")
        response_format = _pick_from_kwargs(kwargs, "📦 返回格式", "response_format", default="url")
        webhook = _pick_from_kwargs(kwargs, "🔔 回调地址", "webhook", default="") or ""
        seed = int(_pick_from_kwargs(kwargs, "🎲 随机种子", "seed", default=0) or 0)

        resolved_api_key = _resolve_blt_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少zhenzhen API Key，请优先使用环境变量 ZHENZHEN_API_KEY，或在本地 config.json / 节点中填写。")

        if model == "gpt-image-2-all" and MWBolatuGPT20._normalize_resolution(resolution) != "1k":
            raise ValueError("gpt-image-2-all 目前仅支持 1K 分辨率。")

        size, error_msg = MWBolatuGPT20._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            raise ValueError(error_msg)

        request_files = _blt_collect_input_images(kwargs)
        mode_name = "Image To Image" if request_files else "Text to Image"

        if mask is not None and _pick_from_kwargs(kwargs, "🖼️ 图像1", "image1", default=None) is None:
            raise ValueError("使用 mask 时必须提供 image1。")

        if request_files:
            data = MWBolatuGPT20._build_edits_payload(
                prompt,
                model,
                n,
                quality,
                size,
                background,
                output_format,
                output_compression,
                moderation,
                response_format,
                seed,
            )
            image1 = _pick_from_kwargs(kwargs, "🖼️ 图像1", "image1", default=None)
            mask_file = _blt_mask_to_request_file(mask, image1)
            if mask_file is not None:
                request_files = list(request_files) + [mask_file]

            single_task_data = dict(data)
            single_task_data["n"] = "1"

            def _submit_one_edit_task():
                fresh_files = _blt_collect_input_images(kwargs)
                fresh_mask_file = _blt_mask_to_request_file(mask, image1)
                if fresh_mask_file is not None:
                    fresh_files = list(fresh_files) + [fresh_mask_file]
                return _blt_submit_async_edit_once(
                    resolved_api_key,
                    single_task_data,
                    fresh_files,
                    webhook,
                    900,
                    5,
                )

            task_ids, submit_responses = _blt_submit_async_tasks_concurrently(
                _submit_one_edit_task,
                n,
            )
            input_payload = dict(single_task_data)
            input_payload["mode"] = "image_to_image"
        else:
            input_payload = MWBolatuGPT20._build_generation_payload(
                prompt,
                model,
                n,
                quality,
                size,
                background,
                output_format,
                output_compression,
                moderation,
                response_format,
                seed,
            )

            single_task_payload = dict(input_payload)
            single_task_payload["n"] = 1

            def _submit_one_generation_task():
                return _blt_submit_async_generation_once(
                    resolved_api_key,
                    single_task_payload,
                    webhook,
                    900,
                )

            task_ids, submit_responses = _blt_submit_async_tasks_concurrently(
                _submit_one_generation_task,
                n,
            )
            input_payload["mode"] = "text_to_image"

        batch_id = "blt_batch_" + uuid.uuid4().hex[:12]
        with _TASKS_LOCK:
            tasks = _read_tasks()
            for index, (task_id, submit_response) in enumerate(zip(task_ids, submit_responses), start=1):
                created_at = time.time()
                tasks[task_id] = {
                    "taskId": task_id,
                    "provider": "blt",
                    "batch_id": batch_id,
                    "sequence": index,
                    "model": model,
                    "prompt": (prompt or "").strip(),
                    "input": input_payload,
                    "state": "pending",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "downloaded": False,
                    "result_urls": [],
                    "call_back_url": webhook.strip(),
                    "submit_response": submit_response,
                    "mode_name": mode_name,
                    "size": size,
                }
            _write_tasks(tasks)

        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "task_ids": task_ids,
                "model": model,
                "input": input_payload,
                "submit_responses": submit_responses,
            },
            ensure_ascii=False,
        )
        report = "已提交zhenzhen任务，批次ID: {}，任务数: {}".format(batch_id, len(task_ids))
        return (response_json, report, "\n".join(task_ids))


class MWBolatuGPT20QueryQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "📦 批次ID(可选)": ("STRING", {"default": ""}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("📋 report", "🧾 response_json")
    FUNCTION = "query"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def query(self, **kwargs):
        batch_id = (_pick_from_kwargs(kwargs, "📦 批次ID(可选)", "batch_id", default="") or "").strip()
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        resolved_api_key = _resolve_blt_api_key(api_key) if api_key else _resolve_blt_api_key("")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        filtered_ids = []
        for task_id, task_entry in tasks.items():
            if (task_entry.get("provider") or "").strip().lower() != "blt":
                continue
            if batch_id and (task_entry.get("batch_id") or "").strip() != batch_id:
                continue
            filtered_ids.append(task_id)

        updated = False
        if resolved_api_key:
            for task_id in filtered_ids:
                task_entry = tasks.get(task_id) or {}
                state = (task_entry.get("state") or "").lower()
                if task_entry.get("downloaded") is True:
                    continue
                if state in ("success", "downloaded", "fail", "failed", "error") and task_entry.get("result_urls"):
                    continue
                try:
                    record_response = _blt_get_task_info(resolved_api_key, task_id)
                    tasks[task_id] = _blt_update_task_entry(task_entry, record_response)
                    updated = True
                except Exception:
                    continue

        if updated:
            with _TASKS_LOCK:
                _write_tasks(tasks)

        filtered_entries = []
        for task_id in filtered_ids:
            filtered_entries.append((task_id, tasks.get(task_id) or {}))
        filtered_entries.sort(key=lambda item: item[1].get("created_at", 0.0), reverse=True)

        counts = {
            "total": len(filtered_entries),
            "pending": 0,
            "success": 0,
            "downloaded": 0,
            "failed": 0,
        }
        lines = ["--- zhenzhen GPT2.0 异步队列总览 ---"]
        for task_id, task_entry in filtered_entries[:100]:
            state = (task_entry.get("state") or "pending").lower()
            if task_entry.get("downloaded") is True:
                counts["downloaded"] += 1
            elif state in ("success", "succeeded", "completed", "finish", "finished"):
                counts["success"] += 1
            elif state in ("fail", "failed", "error", "failure"):
                counts["failed"] += 1
            else:
                counts["pending"] += 1
            lines.append(_format_task_line(task_id, task_entry))

        if len(lines) == 1:
            lines.append("当前没有zhenzhen任务记录。")

        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "counts": counts,
                "tasks": [task_entry for _, task_entry in filtered_entries],
            },
            ensure_ascii=False,
        )
        return ("\n".join(lines), response_json)


class MWBolatuGPT20DownloadReady:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "📦 批次ID(可选)": ("STRING", {"default": ""}),
                "🔑 API密钥": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️ 图像", "🔗 图片地址", "🧾 response_json", "🆔 task_ids")
    FUNCTION = "download"
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def download(self, **kwargs):
        batch_id = (_pick_from_kwargs(kwargs, "📦 批次ID(可选)", "batch_id", default="") or "").strip()
        api_key = _pick_from_kwargs(kwargs, "🔑 API密钥", "api_key", default="") or ""
        resolved_api_key = _resolve_blt_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("缺少zhenzhen API Key，请优先使用环境变量 ZHENZHEN_API_KEY，或在本地 config.json / 节点中填写。")

        with _TASKS_LOCK:
            tasks = _read_tasks()

        selected_ids = []
        for task_id, task_entry in tasks.items():
            if (task_entry.get("provider") or "").strip().lower() != "blt":
                continue
            if batch_id and (task_entry.get("batch_id") or "").strip() != batch_id:
                continue
            selected_ids.append(task_id)
        selected_ids.sort(key=lambda task_id: (tasks.get(task_id) or {}).get("created_at", 0.0))

        updated = False
        for task_id in selected_ids:
            task_entry = tasks.get(task_id) or {}
            if task_entry.get("downloaded") is True:
                continue
            state = (task_entry.get("state") or "").lower()
            if state not in ("success", "succeeded", "completed", "finish", "finished"):
                try:
                    record_response = _blt_get_task_info(resolved_api_key, task_id)
                    tasks[task_id] = _blt_update_task_entry(task_entry, record_response)
                    updated = True
                except Exception:
                    continue

        ready_task_ids = []
        all_result_urls = []
        ready_entries = []
        for task_id in selected_ids:
            task_entry = tasks.get(task_id) or {}
            if task_entry.get("downloaded") is True:
                continue
            state = (task_entry.get("state") or "").lower()
            result_urls = task_entry.get("result_urls") or []
            if state in ("success", "succeeded", "completed", "finish", "finished") and result_urls:
                ready_task_ids.append(task_id)
                ready_entries.append(task_entry)
                all_result_urls.extend(result_urls)
                task_entry["downloaded"] = True
                task_entry["downloaded_at"] = time.time()
                updated = True

        if updated:
            with _TASKS_LOCK:
                _write_tasks(tasks)

        if not ready_task_ids or not all_result_urls:
            raise ValueError("当前无已完成的zhenzhen任务可下载。")

        image_tensor = _download_result_images_as_tensor(all_result_urls)
        response_json = json.dumps(
            {
                "batch_id": batch_id,
                "task_ids": ready_task_ids,
                "tasks": ready_entries,
                "result_urls": all_result_urls,
            },
            ensure_ascii=False,
        )
        return (image_tensor, "\n".join(all_result_urls), response_json, "\n".join(ready_task_ids))


class MWBolatuGPT20FolderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📁 图片文件夹": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "📤 输出文件夹": ("STRING", {"default": "", "placeholder": "输出文件夹路径（留空则自动创建）"}),
                "🤖 模型": (["gpt-image-2", "gpt-image-2-all"], {"default": "gpt-image-2"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (MWBolatuGPT20._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "🖼️ 分辨率": (MWBolatuGPT20._RESOLUTION_CHOICES, {"default": "1K"}),
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
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

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
        resolution = _pick("🖼️ 分辨率", "resolution", default="1K")
        executions_per_prompt = max(1, min(int(_pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1) or 1), 10))
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        image_2 = _pick("🖼️ 备用图像2", "image_2", default=None)
        image_3 = _pick("🖼️ 备用图像3", "image_3", default=None)
        image_4 = _pick("🖼️ 备用图像4", "image_4", default=None)

        resolved_api_key = _resolve_blt_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少zhenzhen API Key，请优先使用环境变量 ZHENZHEN_API_KEY，或在本地 config.json / 节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not directory_path or not os.path.isdir(directory_path):
            msg = "图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if model == "gpt-image-2-all" and MWBolatuGPT20._normalize_resolution(resolution) != "1k":
            msg = "gpt-image-2-all 目前仅支持 1K 分辨率。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        size, error_msg = MWBolatuGPT20._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            return {"ui": {"string": [error_msg]}, "result": (_empty_image_tensor(), error_msg)}

        image_files = _list_local_image_files(directory_path)
        if not image_files:
            msg = "文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "BLT_GPTImage2_Batch")

        extra_image_bytes = []
        try:
            for index, image_tensor in enumerate((image_2, image_3, image_4), start=2):
                if image_tensor is None:
                    continue
                extra_image_bytes.append(
                    {
                        "file_name": "extra_image_{}.png".format(index),
                        "png_bytes": _blt_first_tensor_to_png_bytes(image_tensor),
                    }
                )
        except Exception as e:
            msg = "备用图处理失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        preview_pils = []
        failed_list = []
        saved_paths = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()

        def _process_one_file(file_index, file_path):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_png_bytes = _blt_local_file_to_png_bytes(file_path)
            except Exception as e:
                return 0, 0, [], [], "{} 读取失败: {}".format(base_name, e)

            single_task_data = MWBolatuGPT20._build_edits_payload(
                fixed_prompt,
                model,
                1,
                "auto",
                size,
                "auto",
                "png",
                100,
                "auto",
                "url",
                0,
            )

            def _submit_one_edit_task():
                request_files = [
                    _blt_request_file_from_png_bytes("primary_{}.png".format(file_index + 1), primary_png_bytes, "image")
                ]
                for extra_index, extra_item in enumerate(extra_image_bytes, start=2):
                    request_files.append(
                        _blt_request_file_from_png_bytes(
                            extra_item["file_name"],
                            extra_item["png_bytes"],
                            "image",
                        )
                    )
                return _blt_submit_async_edit_once(
                    resolved_api_key,
                    single_task_data,
                    request_files,
                    "",
                    900,
                    5,
                )

            try:
                task_ids, _submit_responses = _blt_submit_async_tasks_concurrently(
                    _submit_one_edit_task,
                    executions_per_prompt,
                )
                task_results = _blt_poll_tasks_concurrently(
                    resolved_api_key,
                    task_ids,
                    300,
                    1,
                    5,
                    900,
                    None,
                )
            except Exception as e:
                return 0, 0, [], [], "{} 生成失败: {}".format(base_name, e)

            file_success = 0
            file_preview = []
            file_saved_paths = []
            for run_index, (_final_response, pil_images, _image_urls) in enumerate(task_results, start=1):
                if not pil_images:
                    continue
                for image_index, pil_image in enumerate(pil_images, start=1):
                    prefix = "Img{:03d}_{}".format(file_index + 1, base_name)
                    if len(pil_images) == 1:
                        suffix = "P{:03d}".format(run_index)
                    else:
                        suffix = "P{:03d}_{}".format(run_index, image_index)
                    save_path = _save_result_image(pil_image, batch_output_dir, prefix, suffix)
                    file_saved_paths.append(save_path)
                    file_success += 1
                    if len(file_preview) < 15:
                        file_preview.append(pil_image)

            if file_success == 0:
                return 0, len(task_ids), [], [], "{} 未生成结果".format(base_name)
            return file_success, len(task_ids), file_preview, file_saved_paths, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {
                executor.submit(_process_one_file, index, file_path): file_path
                for index, file_path in enumerate(image_files)
            }
            for future in as_completed(future_map):
                file_path = future_map[future]
                file_name = os.path.basename(file_path)
                try:
                    success_count, task_count, file_preview, file_saved_paths, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, file_saved_paths, error_msg = 0, 0, [], [], str(e)

                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    saved_paths.extend(file_saved_paths)
                    if file_preview and len(preview_pils) < 15:
                        remain = 15 - len(preview_pils)
                        preview_pils.extend(file_preview[:remain])
                    if error_msg:
                        failed_list.append("{} -> {}".format(file_name, error_msg))

        preview_tensor = _images_to_batch_tensor(preview_pils[:15]) if preview_pils else _empty_image_tensor()
        status_lines = [
            "✅ zhenzhen GPT2.0 文件夹批量处理完成",
            "📁 输入文件夹: {}".format(os.path.basename(directory_path.rstrip("\\/")) or directory_path),
            "🖼️ 输入图片数: {}".format(len(image_files)),
            "🧩 备用图数量: {}".format(len(extra_image_bytes)),
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
        return {"ui": {"string": [status_report]}, "result": (preview_tensor, status_report)}


class MWBolatuGPT20MultiReferenceBatch:
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
                "🤖 模型": (["gpt-image-2", "gpt-image-2-all"], {"default": "gpt-image-2"}),
                "⚙️ 同时处理文件数": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "📐 图像比例": (MWBolatuGPT20._ASPECT_RATIO_CHOICES, {"default": "1:1"}),
                "🖼️ 分辨率": (MWBolatuGPT20._RESOLUTION_CHOICES, {"default": "1K"}),
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
    CATEGORY = "🤖MINGWEI-API/MW-gpt2.0/🪐zhenzhen"

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
        resolution = _pick("🖼️ 分辨率", "resolution", default="1K")
        executions_per_prompt = max(1, min(int(_pick("🔁 单条Prompt执行次数", "executions_per_prompt", default=1) or 1), 10))
        api_key = _pick("🔑 API密钥", "api_key", default="") or ""
        fixed_prompt = (_pick("📝 固定提示词(必填)", "fixed_prompt", default="") or "").strip()
        reference_image_1 = _pick("🖼️ 参考图像1", "reference_image_1", default=None)
        reference_image_2 = _pick("🖼️ 参考图像2", "reference_image_2", default=None)
        reference_image_3 = _pick("🖼️ 参考图像3", "reference_image_3", default=None)
        reference_image_4 = _pick("🖼️ 参考图像4", "reference_image_4", default=None)

        resolved_api_key = _resolve_blt_api_key(api_key)
        if not resolved_api_key:
            msg = "缺少zhenzhen API Key，请优先使用环境变量 ZHENZHEN_API_KEY，或在本地 config.json / 节点中填写。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not primary_directory or not os.path.isdir(primary_directory):
            msg = "主图片文件夹不存在。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not fixed_prompt:
            msg = "固定提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if model == "gpt-image-2-all" and MWBolatuGPT20._normalize_resolution(resolution) != "1k":
            msg = "gpt-image-2-all 目前仅支持 1K 分辨率。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        size, error_msg = MWBolatuGPT20._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            return {"ui": {"string": [error_msg]}, "result": (_empty_image_tensor(), error_msg)}

        primary_files = _list_local_image_files(primary_directory)
        if not primary_files:
            msg = "主图片文件夹内无有效图片。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if reference_image_1 is None:
            if not reference_directory_1 or not os.path.isdir(reference_directory_1):
                msg = "参考图文件夹1不存在。"
                return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}
            if not _list_local_image_files(reference_directory_1):
                msg = "参考图文件夹1内无有效图片。"
                return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        batch_output_dir = _resolve_batch_output_dir(output_dir, "BLT_GPTImage2_MultiReferenceBatch")

        def _tensor_as_reference(image_tensor, slot_label, slot_index):
            return [
                {
                    "slot": slot_label,
                    "slot_index": int(slot_index),
                    "file_name": slot_label,
                    "png_bytes": _blt_first_tensor_to_png_bytes(image_tensor),
                }
            ]

        def _folder_as_references(folder_path, slot_label, slot_index):
            if not folder_path:
                return []
            if not os.path.isdir(folder_path):
                raise ValueError("{} 不存在".format(slot_label))
            files = _list_local_image_files(folder_path)
            if not files:
                raise ValueError("{} 内无有效图片".format(slot_label))
            uploaded = []
            for file_path in files:
                uploaded.append(
                    {
                        "slot": slot_label,
                        "slot_index": int(slot_index),
                        "file_name": os.path.splitext(os.path.basename(file_path))[0],
                        "png_bytes": _blt_local_file_to_png_bytes(file_path),
                    }
                )
            return uploaded

        reference_groups = []
        slot_labels = []
        try:
            if reference_image_1 is not None:
                reference_groups.append(_tensor_as_reference(reference_image_1, "参考槽位1", 1))
                slot_labels.append("参考槽位1(图片)")
            else:
                group_1 = _folder_as_references(reference_directory_1, "参考槽位1", 1)
                reference_groups.append(group_1)
                slot_labels.append("参考槽位1(文件夹)")

            if reference_image_2 is not None:
                reference_groups.append(_tensor_as_reference(reference_image_2, "参考槽位2", 2))
                slot_labels.append("参考槽位2(图片)")
            elif reference_directory_2:
                reference_groups.append(_folder_as_references(reference_directory_2, "参考槽位2", 2))
                slot_labels.append("参考槽位2(文件夹)")

            if reference_image_3 is not None:
                reference_groups.append(_tensor_as_reference(reference_image_3, "参考槽位3", 3))
                slot_labels.append("参考槽位3(图片)")
            elif reference_directory_3:
                reference_groups.append(_folder_as_references(reference_directory_3, "参考槽位3", 3))
                slot_labels.append("参考槽位3(文件夹)")

            if reference_image_4 is not None:
                reference_groups.append(_tensor_as_reference(reference_image_4, "参考槽位4", 4))
                slot_labels.append("参考槽位4(图片)")
            elif reference_directory_4:
                reference_groups.append(_folder_as_references(reference_directory_4, "参考槽位4", 4))
                slot_labels.append("参考槽位4(文件夹)")
        except Exception as e:
            msg = "参考素材处理失败: {}".format(e)
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        if not reference_groups or not reference_groups[0]:
            msg = "参考槽位1未准备成功。"
            return {"ui": {"string": [msg]}, "result": (_empty_image_tensor(), msg)}

        preview_pils = []
        failed_list = []
        total_success = 0
        total_tasks = 0
        lock = threading.Lock()
        reference_combos = list(product(*reference_groups))
        total_combinations = len(primary_files) * len(reference_combos)

        primary_entries = []
        for file_index, file_path in enumerate(primary_files):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                primary_png_bytes = _blt_local_file_to_png_bytes(file_path)
            except Exception as e:
                failed_list.append("{} -> 读取失败: {}".format(base_name, e))
                continue
            primary_entries.append(
                {
                    "file_index": int(file_index),
                    "file_path": file_path,
                    "base_name": base_name,
                    "png_bytes": primary_png_bytes,
                }
            )

        def _process_one_generation_job(primary_entry, combo_index, reference_combo):
            combo_name = " + ".join(item["file_name"] for item in reference_combo)
            single_task_data = MWBolatuGPT20._build_edits_payload(
                fixed_prompt,
                model,
                1,
                "auto",
                size,
                "auto",
                "png",
                100,
                "auto",
                "url",
                0,
            )

            ordered_references = sorted(reference_combo, key=lambda item: int(item.get("slot_index", 999)))

            def _submit_one_edit_task():
                request_files = [
                    _blt_request_file_from_png_bytes(
                        "primary_{}.png".format(primary_entry["file_index"] + 1),
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
                return _blt_submit_async_edit_once(
                    resolved_api_key,
                    single_task_data,
                    request_files,
                    "",
                    900,
                    5,
                )

            try:
                task_ids, _submit_responses = _blt_submit_async_tasks_concurrently(
                    _submit_one_edit_task,
                    executions_per_prompt,
                )
                task_results = _blt_poll_tasks_concurrently(
                    resolved_api_key,
                    task_ids,
                    300,
                    1,
                    5,
                    900,
                    None,
                )
            except Exception as e:
                return 0, 0, [], "{} + {} 生成失败: {}".format(primary_entry["base_name"], combo_name, e)

            file_success = 0
            file_preview = []
            for run_index, (_final_response, pil_images, _image_urls) in enumerate(task_results, start=1):
                if not pil_images:
                    continue
                for image_index, pil_image in enumerate(pil_images, start=1):
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
                    if len(pil_images) == 1:
                        suffix = "P{:03d}".format(run_index)
                    else:
                        suffix = "P{:03d}_{}".format(run_index, image_index)
                    _save_result_image(pil_image, batch_output_dir, prefix, suffix)
                    file_success += 1
                    if len(file_preview) < 15:
                        file_preview.append(pil_image)

            if file_success == 0:
                return 0, len(task_ids), [], "{} + {} 未生成结果".format(primary_entry["base_name"], combo_name)
            return file_success, len(task_ids), file_preview, None

        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            future_map = {}
            for primary_entry in primary_entries:
                for combo_index, reference_combo in enumerate(reference_combos, start=1):
                    future = executor.submit(_process_one_generation_job, primary_entry, combo_index, reference_combo)
                    future_map[future] = (primary_entry["base_name"], combo_index)

            for future in as_completed(future_map):
                try:
                    success_count, task_count, file_preview, error_msg = future.result()
                except Exception as e:
                    success_count, task_count, file_preview, error_msg = 0, 0, [], str(e)

                with lock:
                    total_tasks += int(task_count)
                    total_success += int(success_count)
                    if file_preview and len(preview_pils) < 15:
                        remain = 15 - len(preview_pils)
                        preview_pils.extend(file_preview[:remain])
                    if error_msg:
                        failed_list.append(error_msg)

        preview_tensor = _images_to_batch_tensor(preview_pils[:15]) if preview_pils else _empty_image_tensor()
        status_lines = [
            "✅ zhenzhen GPT2.0 多参考批量处理完成",
            "📁 主图片数: {}".format(len(primary_files)),
            "🧩 参考槽位数: {}".format(len(reference_groups)),
            "🔢 组合数: {}".format(int(total_combinations)),
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
        return {"ui": {"string": [status_report]}, "result": (preview_tensor, status_report)}


NODE_CLASS_MAPPINGS = {
    "MWKieGPT20": MWKieGPT20,
    "MWKieGPT20SubmitTask": MWKieGPT20SubmitTask,
    "MWKieGPT20QueryQueue": MWKieGPT20QueryQueue,
    "MWKieGPT20DownloadReady": MWKieGPT20DownloadReady,
    "MWKieGPT20FolderBatch": MWKieGPT20FolderBatch,
    "MWKieGPT20DualFolderBatch": MWKieGPT20DualFolderBatch,
    "MWBolatuGPT20": MWBolatuGPT20,
    "MWBolatuGPT20SubmitTask": MWBolatuGPT20SubmitTask,
    "MWBolatuGPT20QueryQueue": MWBolatuGPT20QueryQueue,
    "MWBolatuGPT20DownloadReady": MWBolatuGPT20DownloadReady,
    "MWBolatuGPT20FolderBatch": MWBolatuGPT20FolderBatch,
    "MWBolatuGPT20MultiReferenceBatch": MWBolatuGPT20MultiReferenceBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MWKieGPT20": "🎨 MW-kie-GPT2.0",
    "MWKieGPT20SubmitTask": "1. 📮 MW-kie-GPT2.0 提交任务",
    "MWKieGPT20QueryQueue": "2. 📋 MW-kie-GPT2.0 离线排队查询",
    "MWKieGPT20DownloadReady": "3. 📥 MW-kie-GPT2.0 自动查询并下载",
    "MWKieGPT20FolderBatch": "📂 MW-kie-GPT2.0 文件夹批量处理",
    "MWKieGPT20DualFolderBatch": "📂🧩 MW-kie-GPT2.0 多参考批量处理",
    "MWBolatuGPT20": "🪐 MW-zhenzhen-GPT2.0",
    "MWBolatuGPT20SubmitTask": "1. 🪐 MW-zhenzhen-GPT2.0 提交任务",
    "MWBolatuGPT20QueryQueue": "2. 🪐 MW-zhenzhen-GPT2.0 离线排队查询",
    "MWBolatuGPT20DownloadReady": "3. 🪐 MW-zhenzhen-GPT2.0 自动查询并下载",
    "MWBolatuGPT20FolderBatch": "📂 🪐 MW-zhenzhen-GPT2.0 文件夹批量处理",
    "MWBolatuGPT20MultiReferenceBatch": "📂🧩 🪐 MW-zhenzhen-GPT2.0 多参考批量处理",
}
