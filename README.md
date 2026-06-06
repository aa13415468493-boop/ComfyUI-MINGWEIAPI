# ComfyUI-MINGWEIAPI

ComfyUI-MINGWEIAPI 是一个面向 ComfyUI 的 MINGWEI API 聚合节点包。它把多个常用图像、视频、音频和批量工作流 API 整合到 `🤖MINGWEI-API` 菜单下，方便在 ComfyUI 里直接提交任务、查询任务、下载结果，并把图片或视频结果继续连接到后续节点。

## API 第三方供应商

本插件中的节点主要用于对接第三方 API 服务。当前涉及的第三方 API 供应商包括：

- [KIE.AI](https://kie.ai/zh-CN)
- [T8Star API](https://ai.t8star.org/)

具体模型能力、接口规则、计费方式和服务可用性，以对应第三方供应商页面和接口文档为准。

## 主要特点

- 中文图标 UI：节点名、输入项、输出项尽量使用中文和图标，适合直接在工作流里识别。
- 多模型聚合：集中管理 VEO、Gemini Omni、Grok、GPT Image、nano banana、Seedance 2.0 等节点。
- 常用参数下拉化：比例、分辨率、秒数、模型版本等常用参数尽量做成下拉选择。
- 异步任务支持：支持提交任务、队列查询、结果下载等长任务流程。
- 批量处理：支持文件夹批量、表格批量、多参考批量等高频工作流。
- ComfyUI 连接友好：图片、视频结果会尽量转换为 ComfyUI 可继续连接和保存的输出格式。
- API Key 安全：真实密钥通过环境变量、本地配置或节点输入提供，不应该提交到 GitHub。

## 安装方式

在 ComfyUI 的 `custom_nodes` 目录下克隆本仓库：

```bash
git clone https://github.com/aa13415468493-boop/ComfyUI-MINGWEIAPI.git
```

进入插件目录并安装依赖：

```bash
cd ComfyUI-MINGWEIAPI
pip install -r requirements.txt
```

重启 ComfyUI 后，在右键菜单中找到：

```text
🤖MINGWEI-API
```

## API Key

节点通常支持在节点里填写 API Key，也支持从环境变量或本地配置读取。建议优先使用环境变量，避免把真实密钥写进工作流或提交到仓库。

常用环境变量包括：

```text
KIE_API_KEY
KIEAI_API_KEY
ZHENZHEN_API_KEY
BLT_API_KEY
```

也可以参考 `mw_gpt20/config.json.example` 创建本地配置文件。真实 `config.json` 已被 `.gitignore` 排除，不要提交真实密钥。

## 节点分组

### MW-VEO

VEO 视频相关节点，主要用于：

- Veo 3.1 视频生成
- Veo 3.1 视频扩展
- 视频 URL 输出、响应信息输出和任务 ID 输出
- 支持常用比例、分辨率、秒数、图片输入等参数

### MW-gemini-omni

Gemini Omni 系列节点，主要用于：

- Gemini Omni 视频生成
- 本地视频转可提交的 URL
- 创建 Gemini Omni 音频 ID
- 创建 Gemini Omni 角色 ID
- 支持图片、音频 ID、视频片段、角色 ID 等组合输入

### MW-grok-1.5

Grok Imagine Video 1.5 节点，主要用于：

- 调用 `grok-imagine-video-1-5-preview` 生成视频
- 支持比例、分辨率、时长、图片输入
- 支持 ComfyUI 本地 seed 控制，seed 不传入 KIE 请求体
- 输出视频、视频 URL、响应信息和任务 ID

### MW-gpt2.0

GPT Image 2.0 相关节点，包含 KIE 和 zhenzhen 两套工作流，主要用于：

- 文生图、图生图
- 提交任务
- 队列查询
- 自动查询并下载
- 文件夹批量处理
- 多参考批量处理
- 支持本地 seed 控制、异步模式、回调地址、轮询参数等

### MW-nano banana

nano banana / nano banana pro 相关节点，主要用于：

- 图片生成和编辑
- 异步提交、查询、下载
- 表格批量处理
- 文件夹批量处理
- 图文写入
- 多模态对话
- nano banana 2 / Gemini 3 Pro Image Preview 等扩展工作流

### MW-SD2

Seedance 2.0 视频工作流节点，主要用于：

- zhenzhen-SD2 视频生成
- KIE-SD2 视频生成
- 多元-SD2 视频生成
- 提交任务、查询任务、获取视频
- 素材上传、素材查询、素材绑定
- 支持首帧、尾帧、图片、视频、音频等素材组合

## 文件夹结构

```text
ComfyUI-MINGWEIAPI/
├─ __init__.py                  # 插件入口，合并所有节点并统一 UI 显示
├─ requirements.txt             # 根目录依赖
├─ mingwei_kie/                 # VEO、Gemini Omni、Grok、部分 KIE 视频节点
├─ mw_gpt20/                    # MW-GPT2.0 与 zhenzhen-GPT2.0 图像节点
├─ mw_kie_banana_pro/           # MW-nano banana 系列节点
├─ mw_sd2/                      # MW-SD2 / Seedance 2.0 系列节点
├─ mj_kie/                      # MW-MJ 相关节点
└─ mingwei_web/                 # ComfyUI 前端帮助按钮和图标资源
```

## 使用建议

- 第一次安装后请重启 ComfyUI。
- 修改节点代码后，如果旧节点界面没有刷新，可以重新添加节点。
- 遇到 API 参数错误时，优先检查模型、比例、分辨率、秒数等下拉项是否符合对应 API 文档。
- 批量任务建议先小批量测试，再增加任务数量。

## 免责声明

本插件只是 ComfyUI 中的 API 调用节点集合。具体模型能力、计费、可用地区、任务速度和错误信息以对应 API 服务商为准。
