# audio_note 技术方案

全天无感录音 → 本地 VAD 只存语音 → 本地说话人分离/识别 → MiMo ASR → LLM 纠错 → 带身份的文字记录 → 每日分析推送 → 记忆问答。

Windows 11，CPU only，Python 3.12（uv 管理，`.venv`），所有本地模型走 sherpa-onnx（onnxruntime，不需要 torch）。

## 1. 选型（已在本机验证）

| 组件 | 选型 | 备注 / 备选 |
|---|---|---|
| 采集 | `sounddevice` 16 kHz mono, 512 样本块 | 已验证 Realtek 麦克风 16k 直采；失败时按设备默认采样率采集后 `np.interp` 重采样 |
| VAD | Silero VAD v4 ONNX (`models/silero_vad.onnx`) 直接用 onnxruntime | 自己实现门控状态机（pre-roll 0.5s / hangover 2s / 最长 300s / 最短 0.3s）；272s 音频 0.9s 处理完 |
| 存储 | Opus in OGG via `soundfile`(libsndfile 1.2.2)，`compression_level≈0.93` ≈ 24 kbps | 1 小时语音 ≈ 11 MB；只存语音段；写入中先落 `_partial.wav` 防崩溃丢失 |
| 分离 | sherpa-onnx `OfflineSpeakerDiarization`：pyannote segmentation-3.0 ONNX + 3D-Speaker CAM++ zh-cn 192 维 + fast clustering(threshold 0.5) | 272s 音频 17s（约 16x 实时）；备选 pyannote community-1（需 HF token + torch） |
| 说话人识别 | 注册样本 → CAM++ embedding 存 SQLite；聚类簇 embedding 与注册库 cosine，阈值 0.45（可调，`speakers calibrate`） | ModelScope 卡片 pairwise 阈值 0.31，sherpa 示例 0.6；需按真实数据校准 |
| ASR | MiMo `mimo-v2.5-asr`，`POST /v1/chat/completions` + `input_audio`(wav base64 ≤10MB) | **只返回带标点文本，无时间戳、无说话人**；8K ctx / 2K 输出；100 RPM / 10K TPM；¥0.5/小时音频；单次 >2 min 质量下降 → 每次 ≤60s |
| 纠错 | `mimo-v2.6-flash`，thinking 关闭，JSON 模式，输出 **编辑对**（不是整篇重写），每次 ≤ 3000 字 | 文献：整篇重写会让中文错误变多；JSON 编辑对减少幻觉；硬约束（编辑距离上限、不改数字、同音近音）比 prompt 更重要 |
| 记忆/检索 | SQLite FTS5(unicode61) + jieba 预分词 + 本地 bge-small-zh 向量（sqlite-vec）RRF 混合 + 结构化过滤(日期/说话人) + 预检索 + LLM function-calling 迭代检索 | MiMo **没有 embedding 接口**，向量用本地 90 MB 模型；Letta 实验显示"文件 + grep 工具 + 多轮查询"在 LoCoMo 上胜过 Mem0；不引入 Mem0/Graphiti |
| 每日分析 | map（按 session 分块总结）→ reduce（日报：概览/话题/决定/待办/人物/情绪/未闭环）；同时维护 `memory_notes` | 可用 `mimo.analysis_model` 切到 pro |
| 推送 | 飞书自定义机器人 webhook（主）、Windows toast（兜底，PowerShell 零依赖）、ntfy、Server酱 Turbo、PushPlus、WxPusher、企微群机器人 | 个人微信没有官方通道，Server酱/WxPusher 走公众号 |
| 进程 | `audio-note run`：录音线程 + 调度线程（处理/清理/日报）；单实例锁；Task Scheduler 开机自启 | 可选 pystray 托盘 |

## 2. MiMo API（已核实，2026-10）

- `base_url = https://api.xiaomimimo.com/v1`，openai SDK（Bearer）或 header `api-key`。
- 聊天模型：`mimo-v2.6-flash`（¥1/¥2 每 M tokens，1M ctx）、`mimo-v2.6-pro`（¥3/¥6）。thinking 默认开 → `extra_body={"thinking":{"type":"disabled"}}`。`response_format={"type":"json_object"}` 可用。
- ASR：`model="mimo-v2.5-asr"`，`messages=[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,..."}}]}]`，`extra_body={"asr_options":{"language":"auto|zh|en"}}`。响应 `choices[0].message.content`，`usage.seconds` 为计费秒数。
- 错误码：429 退避重试；402 余额；421 内容过滤；503 过载。
- 未核实：单请求音频时长上限（只有 10MB base64 限制）。默认 60s 一段。

## 3. 目录结构

```
audio_note/                 Python 包
  config.py                 DEFAULTS + config.yaml 合并，Config.data_dir/models_dir/db_path/...   [已完成]
  db.py                     SQLite schema + DB 类                                               [已完成]
  audio/capture.py          MicCapture, list_input_devices, resolve_device                       [已完成]
  audio/vad.py              SileroVAD, SpeechGate, GateConfig, segment_audio, speech_seconds     [已完成]
  audio/storage.py          finalize_segment, write/read, PartialWriter, recover_partials, cleanup[已完成]
  audio/recorder.py         Recorder.run() 常驻循环；gate_config_from(cfg)                       [已完成]
  audio/importer.py         import_file(cfg, db, path, start_time=None), decode_any             [已完成]
  speech/speakers.py        SpeakerEngine: diarize/identify_turns/enroll/embed; Turn dataclass   [已完成]
  speech/asr.py             ASR(cfg, client).transcribe(audio) -> ASRResult(text, seconds, parts)[已完成]
  llm/client.py             MimoClient: chat/text/json/transcribe_bytes；LLMError；parse_json_loose[已完成]
  speech/correct.py         Corrector(cfg, client).correct(items, glossary) 编辑对纠错 + 守卫     [已完成]
  speech/pipeline.py        Processor: 分离→分段→ASR→纠错→utterances→transcripts md             [已完成]
  memory/index.py           jieba 分词、FTS 检索、上下文窗口                                     [已完成]
  memory/embed.py           句向量（fastembed bge-small-zh / 哈希向量）                          [已完成]
  memory/vectors.py         向量表 + sqlite-vec 余弦检索                                         [已完成]
  memory/retrieval.py       Retriever：fts / vector / hybrid(RRF) + 查询扩展                     [已完成]
  memory/expand.py          LLM 查询扩展                                                        [已完成]
  memory/temporal.py        中文相对日期解析                                                    [已完成]
  memory/tools.py           给 LLM 的工具定义 + 执行                                             [已完成]
  memory/chat.py            MemoryChat 迭代检索问答 + REPL                                        [已完成]
  analysis/daily.py         DailyAnalyzer.run(day) / push(day)                                   [已完成]
  push/base.py …            Message, Channel, Notifier + 各渠道                                   [已完成]
  cli.py service.py scheduler.py   命令行 + 常驻服务                                              [已完成]
tests/                      pytest（不联网，mock MimoClient）                                     [已完成]
config.example.yaml README.md glossary.example.txt scripts/*.ps1                                 [已完成]
data/   audio/YYYY-MM-DD/HHMMSS_mmm.opus  transcripts/YYYY-MM-DD.md  daily/YYYY-MM-DD.md  audio_note.sqlite3  speakers/
models/ silero_vad.onnx  3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx  sherpa-onnx-pyannote-segmentation-3-0/model.onnx
```

## 4. 数据模型（db.py，已建）

- `segments(id, path, started_at ISO, duration_s, sample_rate, source mic|import, origin, status new|processing|done|failed|no_speech|deleted, speech_s, error)`
- `transcripts(id, segment_id, asr_model, raw_json, text_raw, text_corrected, correction_json, language)`
- `utterances(id, segment_id, start_s, end_s, abs_start ISO, day YYYY-MM-DD, cluster_label, speaker_id, speaker_name, speaker_score, text_raw, text)`
- `utterance_vectors(utterance_id, model, dim, vec float32 blob)`：句向量（L2 归一化）
- `utterance_tokens(utterance_id, text_seg)` → FTS5 `utterances_fts(text_seg)`（外部内容表，触发器同步；`text_seg` 是 jieba 分词后空格连接）
- `speakers(id, name, notes)`、`speaker_embeddings(speaker_id, embedding float32 blob, dim, source_path, duration_s)`
- `daily_summaries(day PK, summary_md, summary_json, model, utterance_count, pushed_at)`
- `memory_notes(id, kind person|topic|todo|decision|fact, title, content, day_from, day_to, source_utterance_ids)`
- `kv(key, value json)`

## 5. 处理流水线（pipeline.py 规范）

对每个 `status='new'` 且 `created_at` 早于 `pipeline.min_segment_age_s` 的 segment：

1. `audio = read_segment(path)`；`status='processing'`。
2. `turns = engine.diarize(audio)`；`engine.identify_turns(audio, turns, db)` → 每个 turn 有 cluster/speaker_name/score。
   - 没有任何 turn → `status='no_speech'`，删除文件。
3. **构造 ASR chunk**：按时间顺序合并相邻、同 cluster、间隔 ≤ 1.0s 的 turn；单 chunk 时长 ≤ `mimo.asr_max_chunk_s`（超过则在此处切开）。每个 chunk = (start_s, end_s, cluster, speaker_id, speaker_name, score, audio[start:end] 前后各留 0.15s padding)。时长 < 0.6s 的 chunk 丢弃。
4. 每个 chunk → `asr.transcribe(chunk_audio)` → text_raw（空则跳过；421 `ContentFiltered` 只跳过该 chunk，不让整段失败）。
5. `corrector.correct(items, glossary)`：items 是整段 segment 的 chunk 列表 `[{"i":0,"speaker":"张三","text":"..."}]`，返回每条的 `text`（纠正后）和 `edits`。纠错失败（LLMError）→ text = text_raw，记日志，不阻塞。
6. 写 `transcripts` 一行（raw_json = chunks 列表含时间与 raw/corrected 文本）；写 `utterances`（每 chunk 一行；`abs_start = started_at + start_s`，`day` 取 abs_start 的本地日期；`text_seg = memory.index.tokenize(text)`）。
7. `status='done'`；重写当天的 `transcripts/YYYY-MM-DD.md`（从 DB 生成，格式见下）。
8. 任一步异常 → `status='failed'`, error 记录；下一轮 `process --retry-failed` 可重试。

单实例：Processor 与 Recorder 在同一进程不同线程；DB 每线程一个连接（db.py 已处理）。

transcript Markdown 格式：
```
# 2026-10-31 记录

## 08:19:10–08:24:03 (#123 081910_250.opus)（导入：xxx.m4a）  ← 段落按 segment：起止时间 + id + 文件名；导入标注可省略
- **08:19:10 张三**：今天我们先过一下……
- **08:19:32 未知**：好的。
```

## 6. 纠错（correct.py 规范）

输入：items（同一 segment 内按时间排序的 chunk 文本，带说话人）、glossary（`glossary.txt` 每行一个词，另加所有已注册说话人名字）。
Prompt 要点（中文）：这是中文/中英混合语音识别结果；只修正**同音/近音错别字、人名/术语（优先用词表）、明显的分词断句错误**；不改口语风格、不增删句子、不改数字、不翻译；不确定就不改。按 `{"edits":[{"i":0,"from":"...","to":"...","reason":"..."}]}` 返回，`from` 必须是原文中的连续子串。
守卫（代码强制）：
- `from` 不在原文 → 丢弃该编辑；同一条 `from` 多次出现 → 只替换第一个未替换的位置。
- 单条 item 累计字符编辑距离 > 30% 长度 → 丢弃该 item 的所有编辑。
- `from` 或 `to` 含数字且数字变化 → 丢弃。
- 批次控制：按 `correction.max_chars_per_call` 分批。
输出：`[{"i":0,"text":corrected,"edits":[...]}]`。

## 7. 记忆与问答（memory/ 规范）

三层存储：原始层 `utterances`（每句一条，绝对时间 + 说话人 + 原文/纠错文）→ 摘要层 `daily_summaries` + `memory_notes`（每日分析维护的决定/待办/人物/事实）→ 检索层（下述）。不引入 Mem0/Graphiti 等记忆库：MiMo 没有 embedding 接口，向量用本地小模型；Letta 的实验表明"原始记录 + 检索工具 + 多轮查询"的智能体在长期记忆基准上胜过专门记忆库。

### 7.1 索引

- `index.tokenize(text)`：jieba `lcut`（启动时 `jieba.add_word` 所有 speaker 名和 glossary 词），去空白，空格连接，存 `utterance_tokens.text_seg` → FTS5 `utterances_fts`（unicode61）。
- `index.search(db, query, day_from, day_to, speaker, limit, strict=False)`：先全部词 AND，结果 < 3 再对内容词 OR（`strict=True` 不做 OR），按 `bm25()` 排序；`search_terms(terms)` 对多个说法做 OR（查询扩展用）；`like_search` 子串兜底；`window` 上下文。
- `embed.py`：`Embedder` 用 fastembed（ONNX, CPU）加载 `memory.embedding_model`（默认 `BAAI/bge-small-zh-v1.5`，512 维，约 90 MB，首次用时下载到 `models/embeddings/`，之后离线加载），`HashEmbedder`（字符 n-gram 哈希，无模型，测试和 `embedding_model: hash` 用）。向量都做 L2 归一化。
- `vectors.py`：`utterance_vectors(utterance_id, model, dim, vec)`；`index_missing(db, embedder)` 补齐缺失向量（流水线每段处理完后调用；`audio-note reindex` 和问答启动时也会补）；`search()` 用 sqlite-vec 的 `vec_distance_cosine` 在 SQL 里排序（扩展加载失败则退到 numpy 扫描），支持日期/说话人过滤。
- `retrieval.Retriever(cfg, db, client, mode, expand)`：统一入口。`fts` 关键词；`vector` 向量；`hybrid`（默认）= 关键词精确命中列表 + 向量列表做 RRF 融合（`score = Σ w/(rrf_k + rank)`，`fts_weight`、`rrf_k` 可配）。全部为空时子串兜底；向量模型加载失败自动退回 `fts`。`memory.query_expansion: true` 时先让模型给出同义词/品牌名（`expand.QueryExpander`，JSON，进程内缓存），扩展词以 0.7 权重并入。
- `temporal.parse_range(text, now)`：中文相对日期（昨天/前天/昨晚/周三/上周三/这周/上周/最近三天/这几天/10月28号/上个月…）→ `(day_from, day_to)`。

### 7.2 工具与问答策略

- `tools.py`：OpenAI function-calling 工具：`search_transcripts(query, day_from?, day_to?, speaker?, limit?)`（走 Retriever，结果注明检索方式和扩展词）、`get_transcript(day, hour_from?, hour_to?)`、`get_daily_summary(day)`、`list_days()`、`list_speakers()`、`search_notes(query?, kind?)`、`get_context(utterance_id)`。每个工具返回字符串（≤ 4000 字），出错返回 `ERROR: …` 不抛异常。
- `chat.MemoryChat(cfg, db, client, retriever, strategy)`，`memory.strategy`：
  - `agentic`：只给工具，模型自己迭代检索（最多 `max_tool_rounds` 轮）。
  - `prefetch`（默认）：先用 Retriever 对问题预检索 `prefetch_k` 条（问题里有相对日期时按该日期范围优先，并附上命中的 `memory_notes`），放进 system prompt，再进入工具循环；大多数问题一轮即答，难题仍可追查。
  - `rag`：只用预检索（2×`prefetch_k`），不给工具，单次调用。
  - `ask(question, history) -> answer`；`last_stats` 记录 llm/tool 调用次数；`now` 可固定当前时间（评测用）；`repl()` 用 rich 交互。

### 7.3 评测（scripts/memory_bench.py）

问题集 `scripts/memory_questions.yaml`（24 题，针对 `scripts/fake_days.py` 生成的 3 天假数据：含同义改述、多跳、相对日期、否定题），每题标注证据句子和答案关键词。两级评测：

- `retrieval`：各检索方式的证据召回（q_ok@k = 前 k 条里包含足够证据的题数 / 23 道有证据的题）：

  | 方式 | q_ok@5 | q_ok@10 | q_ok@20 | 证据召回@10 | 耗时 |
  |---|---|---|---|---|---|
  | fts（关键词） | 6 | 10 | 11 | 0.49 | 0.01 s |
  | fts + LLM 查询扩展 | 8 | 12 | 13 | 0.61 | +5 s/题 |
  | vector（bge-small-zh） | 15 | 17 | 19 | 0.76 | 0.09 s |
  | hybrid（关键词松散 OR + 向量） | 12 | 16 | 18 | 0.73 | 0.12 s |
  | **hybrid（关键词精确 AND + 向量，默认）** | **15** | **17** | **19** | **0.76** | 0.12 s |

  结论：向量检索把改述类问题（"扫地机"→"科沃斯"、"回退"→"回滚"）的召回翻了近一倍；关键词的松散 OR 列表只会往融合里掺噪声，改成只取精确命中后混合检索与纯向量持平，同时保留对人名、数字、代号（"T30"、"QPS"）的精确匹配；LLM 查询扩展对关键词有帮助，对混合检索没有增益，且每题多 2–5 s，默认关闭。剩余漏检是"昨天晚上聊了什么"这类无内容词的时间型问题（prefetch 用 `temporal` 解析日期过滤解决）和"李强几点到"这类需要说话人过滤的问题（工具循环里模型会自己加 speaker 过滤）。

- `chat`：端到端答案按关键词组判分，同时统计 LLM/工具调用次数与耗时。结果见 README「记忆问答」。

## 8. 每日分析（analysis/daily.py 规范）

`DailyAnalyzer(cfg, db, client)`:
- `run(day) -> (summary_md, summary_json)`：取当天 utterances；按时间间隔 > 20 分钟切 session；每个 session 文本 ≤ 6000 字分块；map：每块 → JSON `{title, summary, topics[], decisions[], todos[], people[], mood, open_questions[]}`；reduce：合并为日报 JSON 并渲染 Markdown（概览 / 主要话题 / 决定 / 待办 / 提到的人 / 情绪 / 未闭环 / 时间线）；写 `daily_summaries` 与 `daily/YYYY-MM-DD.md`；把 todos/decisions/people 写入 `memory_notes`（按 title 去重更新）。
- `push(day)`：用 push.Notifier 发送（标题"每日记录 YYYY-MM-DD"，正文 Markdown，超长按渠道上限截断）；成功后写 `pushed_at`。
- 当天无 utterances → 返回一条简短说明，不调 LLM。

## 9. 推送（push/ 规范）

```python
@dataclass
class Message: title: str; body_md: str; url: str | None = None
class Channel(Protocol): name: str; def send(self, msg: Message) -> None  # raise on failure
class Notifier: def __init__(cfg); def send(msg) -> dict[str, str|None]   # channel -> error/None；按 cfg.push.channels 顺序全部发送
```
渠道实现（httpx，超时 10s，失败抛异常）：feishu（text/post/interactive，可选 sign：`base64(HMAC-SHA256(key=f"{ts}\n{secret}", msg=b""))`）、wecom（markdown ≤4096B）、serverchan（`https://sctapi.ftqq.com/{key}.send` title+desp）、pushplus（`https://www.pushplus.plus/send` template markdown）、wxpusher（SPT simple-push 或 appToken+uids）、ntfy（POST body，headers Title/Markdown）、toast（PowerShell `Windows.UI.Notifications` XML，零依赖；失败不抛仅记日志）。

## 10. CLI / 服务（cli.py / service.py / scheduler.py 规范）

`audio-note <cmd>`（`python -m audio_note` 同效）：
- `devices` 列输入设备；`record` 只录音；`run [--tray]` 录音 + 调度；`status`
- `import <files...> [--start "2026-10-31 09:00"]`
- `process [--retry-failed] [--limit N]`；`transcribe <file>` 一次性转写并打印（不入库）
- `speakers enroll <name> <wav...>` / `speakers list` / `speakers label <segment_id|path> <name> --from s --to s` / `speakers calibrate`
- `daily [--day YYYY-MM-DD] [--push] [--force]`；`push-test "文本"`
- `chat [question]`（无参进入 REPL）
- `cleanup`；`init`（生成 config.yaml、检查模型文件、打印缺失项）
调度：`process_interval_min` 处理新段；`cleanup_interval_min` 清理；每天 `analysis.daily_time` 跑前一天/当天日报并推送（`kv` 记录 last_daily_day 防重复）。日志：rich console + `data/audio_note.log`（RotatingFileHandler）。单实例：`data/service.lock` 写 pid + 进程创建时间，启动时检测进程存活且创建时间一致（防 pid 复用）。`process` 启动时把超过 10 min 的 `processing` 行放回 `new`；日报前先跑 `process`，次日复检前一天的 utterance 数有变化则重生成并推送。

## 11. 默认参数

16 kHz / 512 样本块 / VAD 阈值 0.5（迟滞 0.15）/ 起始 0.3s / 静音结束 2.0s / 前后 0.5s / 最长 300s / Opus 24 kbps / ASR 单段 ≤60s / 说话人阈值 0.45 / 聚类阈值 0.5 / 处理间隔 10 min / 清理 30 min / 日报 22:30 / 音频保留 30 天 / 磁盘上限 20 GB。

## 12. 风险与未验证

- MiMo ASR 单请求时长上限未公开；10K TPM 若按音频 token 计（≈6.25 tok/s）则吞吐约 26 分钟音频/分钟，足够。
- 说话人阈值需用真实样本校准；跨设备/远场会降低分数。
- 飞书个人版是否能建自定义机器人需在客户端确认；兜底 toast/ntfy。
- 纠错模型可能过度修改：守卫 + 保留 text_raw。
