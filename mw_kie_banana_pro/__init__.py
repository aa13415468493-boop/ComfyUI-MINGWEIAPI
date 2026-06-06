from .mw_kie_banana_pro_basic import Gemini3ProImagePreviewZhenzhen, MWKieBanana2, MWKieBananaProBasic
from .mw_kie_banana_pro_async import (
    Gemini3MultimodalChatKie,
    Gemini3MultimodalChatZhenzhen,
    GrsaiNanoBananaBatchCSVExcelKie,
    KieFolderBatchProcessCSV,
    KieLLMVLMWriter,
    ZhenzhenLLMVLMWriter,
    NanoBananaProAsyncBatchSubmit,
    NanoBananaProAsyncDownload,
    NanoBananaProAsyncQuery,
    NanoBananaProAsyncSubmit,
)

NODE_CLASS_MAPPINGS = {
    "MWKieBananaProBasic": MWKieBananaProBasic,
    "MWKieBanana2": MWKieBanana2,
    "Gemini3ProImagePreviewZhenzhen": Gemini3ProImagePreviewZhenzhen,
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
    "MWKieBananaProBasic": "🍌 MW-kie-banana pro基础",
    "MWKieBanana2": "🍌 MW-kie-banana 2",
    "Gemini3ProImagePreviewZhenzhen": "🎨 gemini-3-pro-image-preview-zhenzhen",
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
