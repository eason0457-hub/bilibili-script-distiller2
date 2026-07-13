# bilibili-script-distiller2

这是 `bilibili-script-distiller` 的底层基建重写版。仓库只负责视频与字幕数据的获取、截图、OCR、整句重建和文件保存，不负责人物性格分析、关系推断、写作规则或 WebGAL 格式化。

下游 Grok 或其他分析器只读取本项目的输出文件，不应重新下载视频或重复 OCR。

## 处理流程

1. 解析 Bilibili URL、短链接、BV ID 或 AV ID。
2. 优先用 `yt-dlp` 获取可用字幕轨；成功时直接跳过视频下载和 OCR。
3. 没有字幕轨时，只下载 360P 优先、480P 回退的无音频低清视频。
4. FFmpeg 单次解码并截图，默认每 3 秒一张；代码硬性禁止小于 3 秒的间隔。
5. 用字幕区边缘签名复用近似静止帧的 OCR 结果。
6. RapidOCR 正常只识别一次；文字过短或置信度不足时才追加增强、二值化识别，最多三次。
7. 对连续 OCR 结果做整句重建并原子写入结果文件。
8. 成功结果带配置指纹；相同视频和配置再次运行时直接跳过下载、截图和 OCR。

## 整句算法

整句重建不依赖语言模型，也不会补写不存在的词：

- 同一字幕逐步出现时，从置信度接近的候选中优先选择更完整的一句；
- 相邻片段存在可靠的后缀/前缀重叠时去重拼接，例如 `abcdef` + `defghij` 得到 `abcdefghij`；
- 相邻文本没有重叠证据时保持分开，避免误拼两句独立台词；
- 低置信度单字噪声会被丢弃，高置信度的真实单字台词仍会保留；
- 所有 OCR 原始行、坐标、置信度、候选版本和重建方式都保存在结构化文件中，便于下游复核。

3 秒采样是效率与覆盖率的明确取舍：持续时间短于采样间隔的硬字幕仍可能漏掉。可用字幕轨会优先使用；硬字幕模式允许把间隔调到 4、5、6 或 10 秒，但不允许低于 3 秒。

## 输出契约

每个视频写入 `outputs/<video-id>/`：

| 文件 | 用途 |
|---|---|
| `manifest.json` | 成功/失败状态、配置指纹、版本、来源和统计 |
| `segments.jsonl` | Grok 首选输入；逐句时间、正文、置信度和重建方式 |
| `subtitle.srt` | 通用字幕文件 |
| `transcript.md` | 便于人工阅读的时间轴文本 |
| `source-card.md` | 视频来源与处理配置，不含人物分析 |
| `frames-index.jsonl` | 每张采样帧的 OCR 行、坐标、签名和复用状态 |
| `ocr-status.json` | OCR 调用数、增强次数、复用率和整句统计 |
| `frames/` | 变化帧或全部截图；默认只进入 GitHub Actions Artifact |

建议 Grok 只读取 `manifest.json`、`segments.jsonl`、`subtitle.srt` 和 `source-card.md`。需要核对 OCR 时再读取 `frames-index.jsonl` 与截图。

## GitHub Actions

进入仓库的 **Actions** 页面，选择 **Extract Bilibili subtitles (core only)**，点击 **Run workflow**：

- `video_urls` 可输入多个 URL/BV/AV，支持空格、逗号和换行；
- `sample_interval` 默认 3 秒，可选择更大的间隔；
- `start_time`、`end_time` 可限制处理范围；
- `keep_frames=changed` 只保存字幕变化或低置信度证据帧；
- `force=false` 会复用已成功且配置相同的结果。

文本结果会自动提交到仓库。截图不进入 Git，只保留为 7 天 Artifact；提交失败时，完整结果会作为 30 天 Artifact 上传。

## 本地运行

需要 Python 3.11+ 和 FFmpeg：

```bash
python -m pip install -e .
bilibili-distiller-core BV1uknVz9EeN --sample-interval 3
```

处理指定片段：

```bash
bilibili-distiller-core BV1uknVz9EeN \
  --start-time 01:00 \
  --end-time 03:30 \
  --sample-interval 3 \
  --keep-frames changed
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 责任边界

本仓库输出的是可追溯的媒体与 OCR 证据。人物是谁、人物性格、口癖、关系、情绪、写作规则和最终格式全部属于下游分析层，不应写进本仓库的核心流水线。
