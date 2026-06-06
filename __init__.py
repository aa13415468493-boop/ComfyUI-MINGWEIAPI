WEB_DIRECTORY = "mingwei_web/js"

from .mingwei_kie import (
    NODE_CLASS_MAPPINGS as MINGWEI_KIE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MINGWEI_KIE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .mw_gpt20 import (
    NODE_CLASS_MAPPINGS as MW_GPT20_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MW_GPT20_NODE_DISPLAY_NAME_MAPPINGS,
)
from .mw_kie_banana_pro import (
    NODE_CLASS_MAPPINGS as MW_KIE_BANANA_PRO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MW_KIE_BANANA_PRO_NODE_DISPLAY_NAME_MAPPINGS,
)
from .mw_sd2 import (
    NODE_CLASS_MAPPINGS as MW_SD2_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MW_SD2_NODE_DISPLAY_NAME_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge_node_mappings(class_mappings, display_name_mappings):
    NODE_CLASS_MAPPINGS.update(class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)


_merge_node_mappings(MINGWEI_KIE_NODE_CLASS_MAPPINGS, MINGWEI_KIE_NODE_DISPLAY_NAME_MAPPINGS)
_merge_node_mappings(MW_GPT20_NODE_CLASS_MAPPINGS, MW_GPT20_NODE_DISPLAY_NAME_MAPPINGS)
_merge_node_mappings(MW_KIE_BANANA_PRO_NODE_CLASS_MAPPINGS, MW_KIE_BANANA_PRO_NODE_DISPLAY_NAME_MAPPINGS)
_merge_node_mappings(MW_SD2_NODE_CLASS_MAPPINGS, MW_SD2_NODE_DISPLAY_NAME_MAPPINGS)


_UI_TARGET_MENUS = (
    "MW-VEO",
    "MW-gemini-omni",
    "MW-grok-1.5",
    "MW-gpt2.0",
    "MW-nano banana",
    "MW-SD2",
)


_NODE_DISPLAY_NAME_OVERRIDES = {
    "Veo31Kie": "🎬 MW-VEO 视频生成",
    "Veo31ExtendKie": "🎞️ MW-VEO 扩展视频",
    "MWGeminiOmniVideoKie": "🎬 MW-gemini-omni 视频生成",
    "MWGeminiOmniVideoToUrlKie": "🔗 MW-gemini-omni 视频转URL",
    "MWGeminiOmniAudioKie": "🎧 MW-gemini-omni 创建音频",
    "MWGeminiOmniCharacterKie": "🧍 MW-gemini-omni 创建角色",
    "MWGrokImagineVideoKie": "🎬 MW-grok-1.5 视频生成",
    "MWKieGPT20": "🎨 MW-GPT2.0 图像生成",
    "MWKieGPT20SubmitTask": "1. 📮 MW-GPT2.0 提交任务",
    "MWKieGPT20QueryQueue": "2. 📋 MW-GPT2.0 队列查询",
    "MWKieGPT20DownloadReady": "3. 📥 MW-GPT2.0 查询并下载",
    "MWKieGPT20FolderBatch": "📂 MW-GPT2.0 文件夹批量处理",
    "MWKieGPT20DualFolderBatch": "📂🧩 MW-GPT2.0 多参考批量处理",
    "MWBolatuGPT20": "🎨 zhenzhen-GPT2.0 图像生成",
    "MWBolatuGPT20SubmitTask": "1. 📮 zhenzhen-GPT2.0 提交任务",
    "MWBolatuGPT20QueryQueue": "2. 📋 zhenzhen-GPT2.0 队列查询",
    "MWBolatuGPT20DownloadReady": "3. 📥 zhenzhen-GPT2.0 查询并下载",
    "MWBolatuGPT20FolderBatch": "📂 zhenzhen-GPT2.0 文件夹批量处理",
    "MWBolatuGPT20MultiReferenceBatch": "📂🧩 zhenzhen-GPT2.0 多参考批量处理",
    "MWKieBananaProBasic": "🍌 MW-nano banana 基础",
    "MWKieBanana2": "🍌 MW-nano banana 2",
    "Gemini3ProImagePreviewZhenzhen": "🎨 zhenzhen Gemini 3 Pro 图像预览",
    "NanoBananaProAsyncSubmit": "1. 📮 MW-nano banana 异步提交",
    "NanoBananaProAsyncQuery": "2. 📋 MW-nano banana 异步查询",
    "NanoBananaProAsyncDownload": "3. 📥 MW-nano banana 异步下载",
    "KieLLMVLMWriter": "✍️ KIE 图文写入",
    "ZhenzhenLLMVLMWriter": "✍️ zhenzhen 图文写入",
    "NanoBananaProAsyncBatchSubmit": "📄 MW-nano banana 表格批量提交",
    "KieFolderBatchProcessCSV": "📂 KIE 文件夹批量处理",
    "Gemini3MultimodalChatKie": "💬 KIE Gemini 3 多模态对话",
    "Gemini3MultimodalChatZhenzhen": "💬 zhenzhen Gemini 3 多模态对话",
    "GrsaiNanoBananaBatchCSVExcelKie": "📄 MW-nano banana 表格批量处理",
    "DoubaoSeedance20ZhenzhenNode": "🎬 zhenzhen-SD2 视频生成",
    "DoubaoSeedance20ZhenzhenQueryNode": "📋 zhenzhen-SD2 查询任务",
    "DoubaoSeedance20ZhenzhenSubmitNode": "📮 zhenzhen-SD2 提交任务",
    "DoubaoSeedance20ZhenzhenGetVideoNode": "📥 zhenzhen-SD2 获取视频",
    "DoubaoSeedance20AssetUploadNode": "📤 zhenzhen-SD2 上传素材",
    "DoubaoSeedance20AssetQueryNode": "📋 zhenzhen-SD2 查询素材",
    "DoubaoSeedance20AssetIdBundleNode": "🧩 zhenzhen-SD2 素材绑定",
    "DoubaoSeedance20KieNode": "🎬 KIE-SD2 视频生成",
    "DoubaoSeedance20KieSubmitNode": "📮 KIE-SD2 提交任务",
    "DoubaoSeedance20KieQueryTaskNode": "📋 KIE-SD2 查询任务",
    "DoubaoSeedance20KieGetVideoNode": "📥 KIE-SD2 获取视频",
    "DoubaoSeedance20KieCreateAssetNode": "📤 KIE-SD2 创建素材",
    "DoubaoSeedance20KieQueryAssetNode": "📋 KIE-SD2 查询素材",
    "DoubaoSeedance20KieAssetIdBundleNode": "🧩 KIE-SD2 资产绑定",
    "DoubaoSeedance20DuoyuanNode": "🎬 多元-SD2 视频生成",
    "DoubaoSeedance20DuoyuanSubmitNode": "📮 多元-SD2 提交任务",
    "DoubaoSeedance20DuoyuanQueryTaskNode": "📋 多元-SD2 查询任务",
    "DoubaoSeedance20DuoyuanGetVideoNode": "📥 多元-SD2 获取视频",
    "DoubaoSeedance20DuoyuanTempImageHostNode": "🖼️ 多元-SD2 图片转临时URL",
    "DoubaoSeedance20DuoyuanCreateAssetGroupNode": "📁 多元-SD2 创建素材组",
    "DoubaoSeedance20DuoyuanCreateAssetNode": "📤 多元-SD2 上传素材",
    "DoubaoSeedance20DuoyuanQueryAssetNode": "📋 多元-SD2 查询素材",
    "DoubaoSeedance20DuoyuanAssetIdBundleNode": "🧩 多元-SD2 素材绑定",
}


_INPUT_DISPLAY_NAME_OVERRIDES = {
    "generationType": "🎬 生成类型",
    "model": "🧠 模型",
    "prompt": "📝 提示词",
    "aspect_ratio": "🖼️ 比例",
    "resolution": "📺 分辨率",
    "duration": "⏱️ 秒数",
    "seconds": "⏱️ 秒数",
    "seed": "🎲 种子",
    "watermark": "🏷️ 水印",
    "call_back_url": "🔔 回调地址",
    "callback_url": "🔔 回调地址",
    "enable_fallback": "🛟 启用回退",
    "enable_translation": "🌍 启用翻译",
    "insecure_ssl": "🔒 跳过SSL验证",
    "api_key": "🔑 API密钥",
    "image_url": "🔗 图片URL",
    "video_url": "🔗 视频URL",
    "video_start": "▶️ 视频开始秒",
    "video_ends": "⏹️ 视频结束秒",
    "video": "🎬 视频",
    "audio": "🎵 音频",
    "audio_id": "🎧 音频ID",
    "name": "🏷️ 名称",
    "voice_description": "📝 声音描述",
    "example_dialogue": "💬 示例对白",
    "character_name": "🏷️ 角色名称",
    "description": "📝 描述",
    "task_id": "🆔 任务ID",
    "origin_task_id": "🆔 原视频任务ID",
    "start_time": "⏱️ 开始秒",
    "end_time": "⏱️ 结束秒",
    "video_path": "🎞️ 本地视频路径",
    "image": "🖼️ 图像",
    "upload_url": "🔗 上传URL",
    "format": "🧾 格式",
    "quality": "✨ 质量",
    "timeout": "⏱️ 超时(秒)",
}


_RETURN_DISPLAY_NAME_OVERRIDES = {
    "video": "🎬 视频",
    "video_url": "🔗 视频URL",
    "result_url": "🔗 结果URL",
    "response": "🧾 响应信息",
    "response_json": "🧾 响应信息",
    "raw_json": "🧾 原始响应",
    "task_id": "🆔 任务ID",
    "task_ids": "🆔 任务ID列表",
    "report": "📋 任务报告",
    "image": "🖼️ 图像",
    "images_batch": "🖼️ 图像批次",
    "image_url": "🔗 图片URL",
    "image_urls": "🔗 图片URL列表",
    "status": "📊 状态",
    "ready_json": "📦 就绪信息",
    "file_path": "📄 文件路径",
    "kie_audio_id": "🎧 KIE音频ID",
    "character_id": "🆔 角色ID",
    "name": "🏷️ 名称",
    "character_name": "🏷️ 角色名称",
}


def _is_target_ui_node(cls):
    category = str(getattr(cls, "CATEGORY", ""))
    return any(menu in category for menu in _UI_TARGET_MENUS)


def _numbered_label(key, prefix, text):
    tail = key.rsplit("_", 1)[-1]
    if tail.isdigit():
        return "{} {}{}".format(prefix, text, tail)
    return "{} {}".format(prefix, text)


def _input_display_name(key, current):
    key_text = str(key)
    if key_text in _INPUT_DISPLAY_NAME_OVERRIDES:
        return _INPUT_DISPLAY_NAME_OVERRIDES[key_text]
    if key_text.startswith("image_"):
        return _numbered_label(key_text, "🖼️", "图像")
    if key_text.startswith("image_url_"):
        return _numbered_label(key_text, "🔗", "图片URL")
    if key_text.startswith("audio_id_"):
        return _numbered_label(key_text, "🎧", "音频ID")
    if key_text.startswith("character_id_"):
        return _numbered_label(key_text, "🆔", "角色ID")

    label = str(current or key_text)
    replacements = (
        ("API Key", "API密钥"),
        ("API 密钥", "API密钥"),
        ("insecure_ssl", "跳过SSL验证"),
        ("response_json", "响应信息"),
        ("raw_json", "原始响应"),
        ("video_url", "视频URL"),
        ("image_url", "图片URL"),
        ("image_urls", "图片URL列表"),
        ("audio_ids", "音频ID"),
        ("character_ids", "角色ID"),
        ("BaseURL", "接口地址"),
        ("CallbackURL", "回调地址"),
        ("callBackUrl", "回调地址"),
        ("ServiceTier", "服务等级"),
        ("Prompt", "提示词"),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    return label or key_text


def _copy_input_spec_with_display_name(spec, display_name):
    if not isinstance(spec, tuple):
        return spec
    values = list(spec)
    if len(values) == 1:
        values.append({})
    if len(values) > 1 and isinstance(values[1], dict):
        options = dict(values[1])
        options["display_name"] = display_name
        values[1] = options
    return tuple(values)


def _unique_localized_key(label, used):
    if label not in used:
        used.add(label)
        return label
    index = 2
    while "{} {}".format(label, index) in used:
        index += 1
    value = "{} {}".format(label, index)
    used.add(value)
    return value


def _polished_input_types(original_input_types):
    def _input_types(cls):
        data = original_input_types()
        key_map = {}
        used = set()
        for section in ("required", "optional"):
            values = data.get(section)
            if not isinstance(values, dict):
                continue
            localized_values = {}
            for key, spec in list(values.items()):
                current = ""
                if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict):
                    current = spec[1].get("display_name", "")
                localized_key = _unique_localized_key(_input_display_name(key, current), used)
                localized_values[localized_key] = _copy_input_spec_with_display_name(spec, localized_key)
                key_map[localized_key] = key
            data[section] = localized_values
        cls._mw_ui_key_map = key_map
        return data

    return classmethod(_input_types)


def _wrap_node_function(cls):
    fn_name = getattr(cls, "FUNCTION", "")
    if not fn_name or getattr(cls, "_mw_function_wrapped", False):
        return
    original = getattr(cls, fn_name, None)
    if not callable(original):
        return

    def _wrapped(self, *args, **kwargs):
        key_map = getattr(cls, "_mw_ui_key_map", {})
        if kwargs and key_map:
            kwargs = {key_map.get(k, k): v for k, v in kwargs.items()}
        return original(self, *args, **kwargs)

    setattr(cls, fn_name, _wrapped)
    cls._mw_function_wrapped = True


def _return_display_name(name):
    text = str(name)
    lowered = text.lower().strip()
    for key, value in _RETURN_DISPLAY_NAME_OVERRIDES.items():
        if lowered == key or lowered.endswith(" " + key):
            return value
    replacements = (
        ("response_json", "响应信息"),
        ("raw_json", "原始响应"),
        ("video_url", "视频URL"),
        ("image_url", "图片URL"),
        ("image_urls", "图片URL列表"),
        ("task_id", "任务ID"),
        ("task_ids", "任务ID列表"),
        ("report", "任务报告"),
        ("status", "状态"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _polish_mingwei_api_ui():
    for key, cls in NODE_CLASS_MAPPINGS.items():
        if not _is_target_ui_node(cls):
            continue
        if key in _NODE_DISPLAY_NAME_OVERRIDES:
            NODE_DISPLAY_NAME_MAPPINGS[key] = _NODE_DISPLAY_NAME_OVERRIDES[key]

        input_types = getattr(cls, "INPUT_TYPES", None)
        if callable(input_types) and not getattr(cls, "_mw_ui_polished", False):
            cls.INPUT_TYPES = _polished_input_types(input_types)
            cls._mw_ui_polished = True
            _wrap_node_function(cls)

        return_names = getattr(cls, "RETURN_NAMES", None)
        if isinstance(return_names, tuple):
            cls.RETURN_NAMES = tuple(_return_display_name(name) for name in return_names)


_polish_mingwei_api_ui()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
