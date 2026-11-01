# audio_note 技术设计（v1，2026-10-31）

> 目标机器：Windows 11 Home，无 GPU，uv 0.13；今天即可开工。所有外部事实均标注来源 URL；未经验证的项标为 **「未验证 / 需在 config 中可调」**。事实核查结论（verdict）优先于原始调研。

---

## 0. 一句话架构

```
麦克风 ──sounddevice(16k mono int16, 512帧)──► 内存队列 ──Silero VAD(sherpa-onnx)──► 语音片段 .opus (24kbps)
   │                                                                                   │
   └─ 导入音频文件(wav/mp3/opus/flac) ───────────────► 同一 VAD 切分 ───────────────────┘
                                                                                       ▼
   清理任务(复检VAD/删无语音/保留期) ◄──── SQLite(segments) ◄──────────────────────────┘
                                                                                       ▼
   说话人分离 (sherpa-onnx pyannote-seg-3.0 + CAM++ 嵌入) ► 聚类 ► 登记库匹配(SpeakerEmbeddingManager)
                                                                                       ▼
   按"单说话人 turn"切块(≤60s) ► MiMo mimo-v2.5-asr (chat/completions + input_audio, wav base64)
                                                                                       ▼
   MiMo mimo-v2.6-flash (thinking off) JSON-diff 纠错 + 本地拼音/比例校验 ► utterances(text_raw/text)
                                                                                       ▼
   transcripts/YYYY-MM-DD/<session>.md|json ► 记忆索引(jieba FTS5 + sqlite-vec + 本地嵌入)
                                                                                       ▼
   每日 21:35 map-reduce 分析 (flash) ► daily/YYYY-MM-DD.md ► 推送(飞书 webhook → Windows toast 兜底)
                                                                                       ▼
   本地聊天: MiMo function-calling 智能体 ← search_memory / read_session / read_day / read_entity / grep_raw
```

---

## 1. 最终选型表

| 组件 | 选型 | 版本 | 理由 | 备选 |
|---|---|---|---|---|
| 录音采集 | `sounddevice` | 0.5.6 | 纯 Python wheel 自带 PortAudio DLL，无 cp 标签，3.12–3.14 皆可装；本机已验证 16 kHz mono int16 blocksize=512 回调无 overflow。https://pypi.org/project/sounddevice/#files | 系统声音回环需 `PyAudioWPatch` 0.2.12.9（stock PortAudio 无 WASAPI loopback）https://pypi.org/project/PyAudioWPatch/ |
| VAD | Silero VAD（k2-fsa 导出 `silero_vad.onnx`），经 `sherpa-onnx` `VoiceActivityDetector` | sherpa-onnx 2.0.1 / silero_vad.onnx 0.64 MB | 与说话人分离共用一个运行时（无 torch）；本机实测 0.088 ms/512 样本窗；Silero 在 AliMeeting/AISHELL-4 上 ROC-AUC 0.96/0.94，优于 TEN/WebRTC https://github.com/snakers4/silero-vad/wiki/Quality-Metrics | `onnxruntime` 1.31.1 直接跑官方 v5/v6 `silero_vad.onnx`；`ten-vad` 1.0.6.9 作第二意见 |
| 存储/编码 | 热写 raw int16 `.pcm.part`（flush+fsync 每 5 s）→ 关闭时 `soundfile` 转 Ogg/Opus 24 kbps | soundfile 0.14.0（内置 libsndfile 1.2.2） | 无需 ffmpeg；Opus 24 kbps ≈ 10.8 MB/小时语音，WAV 的 1/10；`compression_level` 映射 bitrate=((1−q)·250000+6000) https://raw.githubusercontent.com/libsndfile/libsndfile/master/src/ogg_opus.c | MP3 32 kbps（soundfile 亦可写）；导入 m4a/aac 时加 `static-ffmpeg` |
| ASR | MiMo `mimo-v2.5-asr`（云端） | — | 用户指定；¥0.5/小时音频；中英混说、方言、自动标点 https://mimo.mi.com/docs/zh-CN/price/pay-as-you-go | 本地 FunASR `paraformer-zh`（有时间戳+热词，需 torch，Python ≤3.12） |
| 说话人分离 + 识别 | `sherpa-onnx` `OfflineSpeakerDiarization`（pyannote-segmentation-3.0 int8 + 3D-Speaker CAM++ zh_en advanced）+ `SpeakerEmbeddingExtractor`/`SpeakerEmbeddingManager` 做登记识别 | sherpa-onnx 2.0.1；seg 模型 1.5 MB；CAM++ 28.3 MB，192 维 | Apache-2.0、无 torch、无 HF token、Windows cp312–cp315 wheel 本机已验证导入；CPU RTF 约 0.1–0.3 https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html | `pyannote.audio` 4.0.7 + `speaker-diarization-community-1`（gated，需 HF token + CPU torch 2.15 + ffmpeg）；识别层保持 CAM++ 嵌入不变 |
| LLM 纠错 | MiMo `mimo-v2.6-flash`，`thinking: disabled`，`response_format json_object`，输出 edits 列表 | — | ¥1/¥2 每百万 token；分段 JSON diff 比整篇重写少幻觉（ChFT）；本地拼音相似度+编辑比例校验防过度修正（LIR-ASR）https://arxiv.org/html/2409.07790v2 | `mimo-v2.6-pro` 用于难样本 |
| 记忆/检索 | SQLite（stdlib）+ FTS5（jieba 预分词 + unicode61）+ `sqlite-vec` 0.1.9 + 本地嵌入 `fastembed` `BAAI/bge-small-zh-v1.5` | sqlite-vec 0.1.9；fastembed 0.9.1；jieba 0.42.1 | MiMo **无 embedding/rerank 接口**（已核实），必须本地嵌入；单文件、同事务、可按时间/说话人 JOIN 过滤；bge-small-zh 90 MB ONNX 零摩擦 https://qdrant.github.io/fastembed/examples/Supported_Models/ | 质量升级：fastembed `Qwen/Qwen3-Embedding-0.6B-Q`（int8，需 onnxruntime≥1.23）或 `BAAI/bge-m3` ONNX；重排 `BAAI/bge-reranker-v2-m3-int8` |
| 每日分析 | map（按会话块）→ reduce（日报+推送文案+遗留 open loops），`mimo-v2.6-flash` | — | 1M 上下文；一天 ≈ 36k 字 ≈ 22k token ≈ ¥0.02 | reduce 阶段可切 `mimo-v2.6-pro` |
| 推送 | 飞书自定义机器人 webhook（card，开签名）为主；Windows toast 兜底 | win11toast 0.36.3 / winotify 1.1.0 | 免费、群级、无需管理员审批，100 次/分钟 https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot | `ntfy`（250/天免费）、WxPusher（免费，SPT 无需注册）、Server酱 Turbo（免费 5 条/天）、Bark（iOS） |
| Windows 进程管理 | Task Scheduler ONLOGON + `pythonw.exe`；命名互斥量单实例；`pystray` 托盘（暂停/退出）；日志写文件 | pystray 0.19.5 | 无控制台、崩溃自动释放互斥量；需关闭任务的"运行超过 3 天停止" https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create | `filelock` 4.2.5 替代互斥量 |
| Python | **3.12**（uv pin） | — | 见 §7 | 3.13 同样安全；3.14 主栈可用但 FunASR 备选路径不可用 |

---

## 2. MiMo API 调用细节（仅列已核实项）

| 项 | 值 | 来源 |
|---|---|---|
| OpenAI 兼容 base URL | `https://api.xiaomimimo.com/v1` | https://mimo.mi.com/docs/en-US/quick-start/summary/first-api-call |
| Anthropic 兼容 base URL | `https://api.xiaomimimo.com/anthropic`（`POST /anthropic/v1/messages`） | https://mimo.mi.com/docs/en-US/api/chat/anthropic-api |
| 鉴权 | 请求头 `api-key: $MIMO_API_KEY` **或** `Authorization: Bearer $MIMO_API_KEY`；openai SDK 默认 Bearer 可用。按量付费 key 为 `sk-…`；Token Plan 的 `tp-`/`ttp-` key 走 `https://token-plan-cn.xiaomimimo.com/v1`，两者不可混用 | https://mimo.mi.com/docs/en-US/api/chat/openai-api ；https://mimo.mi.com/docs/zh-CN/quick-start/faq/api-integration |
| 聊天模型 | `mimo-v2.6-flash`（默认）、`mimo-v2.6-pro`；1M 上下文、128K 输出、100 RPM / 10M TPM（按账号、按模型计）。`mimo-v2.5-pro`/`mimo-v2.5` 已于 2026-10-21 下线 | https://mimo.mi.com/docs/zh-CN/quick-start/summary/model ；https://mimo.mi.com/docs/zh-CN/api/guidance/rate-limit |
| thinking | 非标字段 `thinking: {"type": "enabled"\|"disabled"}`，经 `extra_body` 传；V2.6 默认开启；开启时 temperature/top_p 被强制 1.0/0.95，`max_completion_tokens` 默认 131072 含推理；多轮工具调用须回传 `reasoning_content` 否则 400 | https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking |
| 结构化输出 | 仅 `response_format: {"type": "json_object"}`（json_schema 未文档化）；提示词中必须要求 JSON | https://mimo.mi.com/docs/en-US/quick-start/usage-guide/text-generation/structured-output |
| ASR 端点 | **无** `/v1/audio/transcriptions`。`POST https://api.xiaomimimo.com/v1/chat/completions`，`model: "mimo-v2.5-asr"`，单条 user 消息、单个 `{"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,...","format":"wav"?}}`；data URL 时 `format` 可省，纯 base64 时必填且须与 MIME 一致；顶层 `asr_options: {"language": "auto"\|"zh"\|"en"}`（SDK 走 `extra_body`）；`stream: true` 返回 SSE delta | https://mimo.mi.com/docs/en-US/api/audio/Speech-Recognition |
| ASR 格式/大小 | 仅 wav（audio/wav）、mp3（audio/mpeg 或 audio/mp3）；**base64 字符串 ≤ 10 MB**；未文档化时长上限 | https://mimo.mi.com/docs/en-US/quick-start/usage-guide/audio/Speech-Recognition |
| ASR 限额 | 8K 上下文 / 2K 最大输出；100 RPM / 10K TPM | https://mimo.mi.com/docs/zh-CN/quick-start/summary/model |
| ASR 响应 | 标准 chat.completion；文本在 `choices[0].message.content`（带标点）；`usage.seconds` = 计费音频秒数；`finish_reason` ∈ stop/length/content_filter；**无时间戳、无说话人标签、无热词参数** | https://mimo.mi.com/docs/en-US/api/audio/Speech-Recognition |
| 计费 | ASR ¥0.5/小时（海外 $0.074/h），精确到秒；flash 输入 ¥1 / 缓存 ¥0.02 / 输出 ¥2 每百万 token；pro ¥3 / ¥0.025 / ¥6 | https://mimo.mi.com/docs/zh-CN/price/pay-as-you-go |
| 错误码 | 400 格式错；401 鉴权；402 余额不足；403 风控/地区；404 模型不存在；421 内容过滤；429 限流（指数退避）；500/503 | https://mimo.mi.com/docs/en-US/api/guidance/error-codes |
| 隐私 | 政策只承诺"文本内容"不用于训练，对音频未明示；数据存于中国境内 | https://privacy.mi.com/XiaomiMiMoPlatform/en_GB/ |

**未验证 / 需在 config 中可调：**
- ASR 单次最长音频时长（仅有 10 MB base64 与 2K 输出上限；社区开源权重报告 >~2 分钟退化 https://github.com/KurtDu/mimo-asr-vllm ）→ `asr.max_chunk_seconds` 默认 60，首日用 30/60/120/240 s 片段实测。
- ASR 的 10K TPM 是否计入音频 token → 实现令牌桶限速并记录 `usage.prompt_tokens_details.audio_tokens`。
- ASR 响应是否带任何分段结构 → 首次调用 `completion.model_dump_json()` 落盘检查。
- 账号计费区域（CNY/USD）。

---

## 3. 目录结构与数据模型

### 3.1 目录

```
C:\path\to\audio_note\
├─ .python-version            # 3.12
├─ pyproject.toml
├─ config.example.yaml        # 提交；config.yaml 本地填 key，.gitignore
├─ config.yaml
├─ run.py                     # pythonw 入口：单实例 + 托盘 + recorder + worker
├─ audio_note/
│  ├─ __main__.py / cli.py    # record | worker | run | import <file> | enroll <name> <wav...> | calibrate | daily [date] | chat | label-unknown | install-task
│  ├─ config.py               # pydantic/dataclass 加载 config.yaml，env 覆盖 MIMO_API_KEY
│  ├─ single_instance.py      # CreateMutexW
│  ├─ tray.py                 # pystray
│  ├─ recorder/
│  │   ├─ capture.py          # sounddevice InputStream → queue
│  │   ├─ vad.py              # sherpa-onnx VoiceActivityDetector 封装
│  │   └─ segmenter.py        # pre-roll / hangover / max_len / .pcm.part → .opus / 启动恢复
│  ├─ storage/
│  │   ├─ db.py               # sqlite3 + sqlite_vec.load + schema 迁移
│  │   ├─ schema.sql
│  │   └─ paths.py
│  ├─ jobs/
│  │   ├─ scheduler.py        # 单线程循环：每 N 分钟跑队列；21:35 daily；每周日 weekly
│  │   ├─ cleanup.py          # 复检 VAD、删无语音、保留期、孤儿 .part
│  │   ├─ sessions.py         # 按间隔把 segments 归并为 session
│  │   ├─ diarize.py          # OfflineSpeakerDiarization → turns
│  │   ├─ speaker_id.py       # SpeakerEmbeddingManager 登记/匹配/校准
│  │   ├─ asr.py              # 按 turn 切块 → wav bytes → MiMo ASR
│  │   ├─ correct.py          # LLM JSON-diff 纠错 + 校验
│  │   ├─ export.py           # transcripts md/json
│  │   ├─ daily.py            # map-reduce 日报 + 推送
│  │   └─ weekly.py           # 实体/周笔记 + MEMORY.md
│  ├─ mimo/
│  │   ├─ client.py           # openai.OpenAI(base_url=...), 重试/限速
│  │   ├─ asr.py
│  │   └─ chat.py
│  ├─ memory/
│  │   ├─ embed.py            # fastembed 封装（名字+维度写 meta）
│  │   ├─ index.py            # chunks / notes → fts + vec
│  │   ├─ search.py           # hybrid RRF
│  │   ├─ tools.py            # LLM tool schemas + dispatch（相对时间解析、别名）
│  │   └─ agent.py            # 聊天循环
│  └─ notify/
│      ├─ base.py             # Message / Channel / Notifier
│      ├─ feishu.py  toast.py  ntfy.py  wxpusher.py  serverchan.py
├─ models/                    # 见 §7 下载清单
├─ data/
│  ├─ audio_note.db
│  ├─ audio/YYYY-MM-DD/HHMMSS_mmm.opus
│  ├─ imports/                # 用户放入的文件（untrusted）
│  ├─ transcripts/YYYY-MM-DD/<session_id>.md  + .json
│  ├─ daily/YYYY-MM-DD.md
│  ├─ memory/MEMORY.md  entities/<name>.md  weekly/YYYY-Www.md
│  ├─ speakers/<name>.json    # 每个登记片段的 192 维向量（全部保留，不只均值）
│  └─ logs/audio_note.log
└─ scripts/
   ├─ download_models.ps1
   └─ install_task.ps1
```

### 3.2 SQLite schema（`storage/schema.sql`）

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);   -- schema_version, embedder_name, embedder_dim, speaker_model, speaker_dim

CREATE TABLE IF NOT EXISTS speakers(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, aliases TEXT DEFAULT '[]',   -- JSON 列表
  is_owner INTEGER DEFAULT 0, enrolled_at TEXT, n_samples INTEGER DEFAULT 0, note TEXT);

CREATE TABLE IF NOT EXISTS embeddings(                                  -- 说话人登记向量（每片段一条）
  id INTEGER PRIMARY KEY, speaker_id INTEGER REFERENCES speakers(id) ON DELETE CASCADE,
  source TEXT,                 -- enroll_file | labeled_cluster
  source_path TEXT, duration_s REAL, model TEXT, dim INTEGER, vec BLOB,   -- float32 bytes
  created_at TEXT);

CREATE TABLE IF NOT EXISTS sessions(                                    -- 相邻 segment 间隔 < session_gap 归并
  id TEXT PRIMARY KEY,         -- 'S' + YYYYMMDD_HHMMSS
  day TEXT, start_ts REAL, end_ts REAL, source TEXT,                    -- mic | import
  title TEXT, n_segments INTEGER DEFAULT 0, speech_seconds REAL DEFAULT 0,
  status TEXT DEFAULT 'open',  -- open | closed | transcribed | corrected | exported | analyzed
  md_path TEXT, json_path TEXT);

CREATE TABLE IF NOT EXISTS segments(                                    -- 一个 .opus 文件
  id INTEGER PRIMARY KEY, session_id TEXT REFERENCES sessions(id),
  path TEXT UNIQUE, day TEXT, start_ts REAL, end_ts REAL, duration_s REAL,
  sample_rate INTEGER DEFAULT 16000, codec TEXT DEFAULT 'opus', bytes INTEGER,
  source TEXT,                 -- mic | import
  speech_s REAL,               -- cleanup 复检得到的语音秒数
  vad_checked INTEGER DEFAULT 0, diarized INTEGER DEFAULT 0, transcribed INTEGER DEFAULT 0,
  private INTEGER DEFAULT 0,   -- 命中 private_windows → 不上传
  deleted_at TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS seg_day ON segments(day, start_ts);
CREATE INDEX IF NOT EXISTS seg_todo ON segments(vad_checked, diarized, transcribed);

CREATE TABLE IF NOT EXISTS turns(                                       -- diarization 输出（局部簇）
  id INTEGER PRIMARY KEY, segment_id INTEGER REFERENCES segments(id) ON DELETE CASCADE,
  start_s REAL, end_s REAL,    -- 相对 segment 起点
  cluster INTEGER,             -- 该 segment 内的局部簇号
  speaker_id INTEGER REFERENCES speakers(id),   -- NULL = unknown
  speaker_score REAL, low_confidence INTEGER DEFAULT 0,
  cluster_vec BLOB);           -- 该簇嵌入，供 label-unknown 回写登记
CREATE INDEX IF NOT EXISTS turn_seg ON turns(segment_id, start_s);

CREATE TABLE IF NOT EXISTS transcripts(                                 -- 一次 ASR 请求
  id INTEGER PRIMARY KEY, segment_id INTEGER REFERENCES segments(id) ON DELETE CASCADE,
  turn_ids TEXT,               -- JSON：打包进本次请求的 turn id
  start_s REAL, end_s REAL, audio_seconds INTEGER, model TEXT, language TEXT,
  text_raw TEXT, finish_reason TEXT, usage_json TEXT, request_id TEXT,
  status TEXT DEFAULT 'ok',    -- ok | failed | skipped_private
  error TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS utterances(                                  -- 最终带身份的句子
  id INTEGER PRIMARY KEY, session_id TEXT REFERENCES sessions(id), segment_id INTEGER,
  transcript_id INTEGER REFERENCES transcripts(id),
  start_ts REAL, end_ts REAL,  -- 绝对时间戳 = segment.start_ts + turn.start_s（句级近似）
  speaker_id INTEGER REFERENCES speakers(id), speaker_label TEXT,       -- 名字或 'unknown_k'
  speaker_conf REAL, text_raw TEXT, text TEXT, edits_json TEXT, corrected INTEGER DEFAULT 0,
  seq INTEGER);
CREATE INDEX IF NOT EXISTS utt_session ON utterances(session_id, seq);
CREATE INDEX IF NOT EXISTS utt_time ON utterances(start_ts);

-- ===== 记忆层 =====
CREATE TABLE IF NOT EXISTS mem_chunks(                                  -- 检索单元：同 session、300–500 字、≤5 min
  id INTEGER PRIMARY KEY, session_id TEXT, day TEXT, month TEXT, start_ts REAL, end_ts REAL,
  speaker_ids TEXT,            -- ',1,3,'
  utt_from INTEGER, utt_to INTEGER, header TEXT, text TEXT, text_raw TEXT,
  text_tok TEXT, text_raw_tok TEXT, embedded INTEGER DEFAULT 0);
CREATE VIRTUAL TABLE IF NOT EXISTS mem_chunks_fts USING fts5(
  text_tok, text_raw_tok, content='mem_chunks', content_rowid='id', tokenize='unicode61');
-- 维度由 config.memory.embedding.dim 决定；建表时由 db.py 用 f-string 生成：
-- CREATE VIRTUAL TABLE mem_chunks_vec USING vec0(
--   month TEXT PARTITION KEY, day TEXT, speaker_id INTEGER,
--   emb float[{dim}] distance_metric=cosine, +chunk_id INTEGER);

CREATE TABLE IF NOT EXISTS daily_summaries(
  day TEXT PRIMARY KEY, md_path TEXT, push_message TEXT, report_md TEXT,
  blocks_json TEXT,            -- map 阶段各块 JSON
  open_loops_json TEXT, questions_json TEXT, model TEXT, usage_json TEXT,
  pushed_at TEXT, pushed_channels TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS memory_notes(                                -- observation | day_summary | week_summary | entity
  id INTEGER PRIMARY KEY, note_type TEXT, day TEXT, month TEXT,
  valid_from TEXT, valid_to TEXT, created_at TEXT, supersedes INTEGER,  -- 双时态：知识更新不删旧条
  entity_id INTEGER, speaker_id INTEGER, source_chunk_ids TEXT,
  title TEXT, text TEXT, text_tok TEXT, md_path TEXT, embedded INTEGER DEFAULT 0);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_notes_fts USING fts5(
  title_tok, text_tok, content='memory_notes', content_rowid='id', tokenize='unicode61');
-- CREATE VIRTUAL TABLE memory_notes_vec USING vec0(month TEXT PARTITION KEY, note_type TEXT, entity_id INTEGER,
--   emb float[{dim}] distance_metric=cosine, +note_id INTEGER);

CREATE TABLE IF NOT EXISTS entities(id INTEGER PRIMARY KEY, kind TEXT, name TEXT UNIQUE, aliases TEXT,
  note_id INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS entity_mentions(entity_id INTEGER, note_id INTEGER, chunk_id INTEGER, day TEXT);

CREATE TABLE IF NOT EXISTS job_log(id INTEGER PRIMARY KEY, job TEXT, ref TEXT, status TEXT, detail TEXT, ts TEXT);

-- external-content FTS 同步触发器（mem_chunks 示例；memory_notes 同理）
CREATE TRIGGER IF NOT EXISTS mem_chunks_ai AFTER INSERT ON mem_chunks BEGIN
  INSERT INTO mem_chunks_fts(rowid, text_tok, text_raw_tok) VALUES (new.id, new.text_tok, new.text_raw_tok); END;
CREATE TRIGGER IF NOT EXISTS mem_chunks_ad AFTER DELETE ON mem_chunks BEGIN
  INSERT INTO mem_chunks_fts(mem_chunks_fts, rowid, text_tok, text_raw_tok) VALUES('delete', old.id, old.text_tok, old.text_raw_tok); END;
CREATE TRIGGER IF NOT EXISTS mem_chunks_au AFTER UPDATE ON mem_chunks BEGIN
  INSERT INTO mem_chunks_fts(mem_chunks_fts, rowid, text_tok, text_raw_tok) VALUES('delete', old.id, old.text_tok, old.text_raw_tok);
  INSERT INTO mem_chunks_fts(rowid, text_tok, text_raw_tok) VALUES (new.id, new.text_tok, new.text_raw_tok); END;
```

来源：FTS5 external content / bm25 https://www.sqlite.org/fts5.html ；vec0 元数据列与分区键 https://alexgarcia.xyz/sqlite-vec/features/vec0.html ；Windows 官方 Python 自 3.11 起支持 `enable_load_extension` https://github.com/python/cpython/issues/95656 。

### 3.3 transcript 文件格式

`data/transcripts/2026-10-31/S20261031_093012.md`
```
# 2026-10-31 09:30–10:12 · <title>
参与者: 刘恩怀, 王总, unknown_1
---
[09:30:12] 刘恩怀: 我们下周把 MiMo 的 ASR 接到流水线里。
[09:30:19] 王总(0.61): 好，先跑一周看看成本。
[09:31:02] unknown_1: ……
```
`.json`：`{"session": {...}, "utterances": [{"id","start_ts","end_ts","speaker","speaker_conf","text","text_raw","edits":[...]}]}`。speaker 后括号为低置信分数（在阈值 ±0.05 内）。

---

## 4. Pipeline 流程

### 4.1 Recorder（常驻线程）
1. `sd.InputStream(samplerate=16000, blocksize=512, channels=1, dtype="int16", callback)`；回调只做 `queue.put_nowait(indata[:,0].copy())`，满则丢帧（不阻塞音频引擎）。WASAPI 设备需 `extra_settings=sd.WasapiSettings(auto_convert=True)`。
2. Worker 线程逐 512 样本块喂 `VoiceActivityDetector`（sherpa-onnx，min_silence 0.8 s、min_speech 0.5 s、max_speech 45 s）。sherpa-onnx 的 VAD 自带前后 roll 与分段队列；`vad.front.samples` 即一个完整语音片段。
3. 片段落盘：先写 `data/audio/<day>/<ts>.pcm.part`（raw int16，flush+fsync 每 5 s 用于崩溃恢复），结束后 `sf.write(..., format="OGG", subtype="OPUS", compression_level=0.928)` 转 `.opus`，删除 `.part`，`INSERT segments`。
4. 片段起始绝对时间 = `time.time() − 队列积压秒数 − 片段长度`，并在 `segments.start_ts` 保存。
5. 托盘「暂停」= 不喂 VAD（仍保持流打开）；`private_windows` 时段的片段标 `private=1`（本地保留、不上传）。
6. 启动时：`recover_parts()` 把孤儿 `.part` 转 `.opus`；长度 < 0.5 s 直接删。

### 4.2 导入
`audio_note import <file>`：soundfile 读（wav/flac/ogg/opus/mp3），不支持的扩展名提示安装 `static-ffmpeg`；重采样到 16k mono（`numpy` 线性插值或 `soxr`，可选依赖），喂同一个 VAD 流程（离线 `get_speech_timestamps` 风格：一次性过完），生成 segments（`source='import'`，`start_ts` = 文件 mtime 或 `--date` 参数）。

### 4.3 Cleanup job（每小时 + 启动时）
- 对 `vad_checked=0` 的 `.opus`：解码，批量 VAD 统计 `speech_s`；`speech_s < 1.0` → 删除文件、`deleted_at` 置位。
- 保留策略：`retention.audio_days`（默认 30）过期删音频文件（DB 行与转写保留）；`retention.max_audio_gb` 超额时按最旧删。
- 把 `status='open'` 且距最后片段 > `session_gap`（600 s）的 session 关闭。

### 4.4 Sessions
新 segment 若与同源上一个 segment 的 `end_ts` 间隔 < 600 s 则归入同一 session，否则新建 `S<YYYYMMDD_HHMMSS>`。

### 4.5 Diarization + Speaker-ID job（每 5 分钟，处理 `diarized=0` 且 session 已关闭的片段）
1. 解码 `.opus` → float32 16k；`OfflineSpeakerDiarization.process()` → `turns(start,end,cluster)`。
2. 每个局部簇：拼接该簇所有窗口音频 → `SpeakerEmbeddingExtractor.compute()` → `manager.search(vec, threshold)`；命中则 `speaker_id`，否则 `unknown_k`。`score` 在阈值 ±0.05 内标 `low_confidence=1`。簇向量存 `turns.cluster_vec`。
3. 单说话人片段（< 1.5 s 或 diarization 只出一个簇）直接取整段嵌入匹配。
4. `audio_note label-unknown <session> <k> <name>`：把该簇向量追加到 `embeddings`（source=labeled_cluster）并重建 manager —— 半监督扩充登记。

### 4.6 ASR job（每 5 分钟，diarized=1 且 transcribed=0 且 private=0）
**Diarization-first，不重叠切块：**
1. 同一 segment 内，把 **连续同簇** 的 turns 合并（间隙 < 0.6 s），短于 0.8 s 的 turn 并入相邻 turn。
2. 每个合并块 ≤ `asr.max_chunk_seconds`（60 s），超长在块内最长静音处切（Silero 批处理 `max_speech_duration_s` 逻辑）。
3. 块音频两侧 pad 0.15 s（不越过 segment 边界）→ 内存 WAV（16k/16bit mono，60 s ≈ 1.9 MB，base64 ≈ 2.6 MB < 10 MB）→ `mimo-v2.5-asr`，`asr_options.language` 取 config（默认 `auto`）。
4. 写 `transcripts`，再按块生成 `utterances`（`speaker_label` = 簇对应名字/unknown_k；时间戳 = 块起止，句级近似；一块内多句按标点拆成多条 utterance，时间按字符比例线性插值）。
5. 失败（429/5xx）指数退避、最多 5 次；`finish_reason=='length'` 记录告警并把该块二分重试。
6. 令牌桶：≤ 90 请求/分钟；按 `usage.seconds` 累计当日成本写 `job_log`。

### 4.7 LLM 纠错 job（session 关闭且全部 transcribed 后，一次性处理整个 session）
1. 把 utterances 按 ≤ 2000 字、每条 20–100 字分批，附 `pypinyin.lazy_pinyin` 行、`glossary`（config + speakers 名字/别名）与 `context`。
2. 调 `mimo-v2.6-flash`（thinking disabled，`response_format json_object`，temperature 0.3）→ `{"edits":[{id,old,new,reason}]}`。
3. 本地校验 `accept_edit()`：old 必须子串存在；编辑比例 ≤ 30%；数字保护；拼音相似度 ≥ 0.6（术语表命中放宽）；通过的才 `replace(old,new,1)`，写 `text` 与 `edits_json`，`text_raw` 永远保留。
4. 上云前用 `Pseudonymizer` 把登记姓名替换为 `P1..Pn`，返回后反向替换（可在 config 关闭）。

### 4.8 导出
写 `transcripts/<day>/<session>.md|json`，`sessions.status='exported'`，并触发记忆索引：
- 把 utterances 按同 session、300–500 字、≤ 5 分钟切 `mem_chunks`；`header = "日期 2026-10-31 09:30-09:34 | 参与者 刘恩怀,王总 | 会话 <title>"`；`text_tok = ' '.join(jieba.cut_for_search(header + text))`（jieba 已 `load_userdict` 登记的人名/术语）。
- 嵌入 `header + text`（fastembed `embed()`），写入 `mem_chunks_vec`（partition=month）。

### 4.9 每日分析（21:35，或 `audio_note daily 2026-10-31`）
1. **Map**：对当天每个 session（> 1500 字再按 ≤ 8000 字拆块）用 §9 的 map 提示词 → JSON（is_meaningful/title/summary/topics/decisions/todos/people/open_loops/mood/key_quotes）。`is_meaningful=false` 丢弃。
2. **Reduce**：全部块 + 昨日 `open_loops_carry_over` → `{push_message, report_markdown, open_loops_carry_over, questions_for_user}`。
3. 写 `daily/YYYY-MM-DD.md`、`daily_summaries`；每个 block 的 observations 写 `memory_notes(note_type='observation')`，day_summary 也写一条，均索引。
4. 推送：`Notifier.send(Message(title="每日摘要 10-31", body_md=push_message + 链接))` → 飞书 card；失败则 Windows toast。超过 20 KB（飞书）自动分条。
5. **Weekly（周日 22:00）**：按 speaker/entity 聚合本周 observations，LLM 重写 `entities/<name>.md`（上周笔记 + 新观察 → 新版本，旧版本 `supersedes`），生成 `weekly/`，重建 `memory/MEMORY.md`（≤ 200 行索引：实体/话题一行一条）。

### 4.10 聊天 / 检索循环（`audio_note chat`）
- system = 今天日期 + `MEMORY.md` + 最近 2 天 day_summary（≈2–3k token）。
- tools：`search_memory(query, time_from?, time_to?, speakers?, tiers, k)`、`read_session(session_id, from_ts?, to_ts?)`、`read_day(date)`、`read_entity(name)`、`list_sessions(date_from, date_to, speaker?)`、`grep_raw(regex, time_from?, time_to?)`（同时搜 `text_raw`，防纠错把人名改坏）。
- `search_memory`：相对时间（昨天/上周三/上个月）本地解析成范围；说话人别名解析；并行 FTS5 bm25（top 50，JOIN 时间/说话人过滤）+ sqlite-vec KNN（top 50，`month` 分区 + `day`/`speaker_id` 元数据过滤）→ RRF(k=60) → notes 命中排前、chunks 排后 → 返回 top 15 + 相邻 1 块。可选重排（`BAAI/bge-reranker-v2-m3-int8`，top 40）。
- 最多 6 轮工具调用（`mimo-v2.6-pro`，thinking 可开；多轮须回传 `reasoning_content`），答案附引用 `(10-28 14:32 王总)`，无证据时明确说"记录里没有找到"。

---

## 5. 关键参数默认值

| 参数 | 默认 | 依据 |
|---|---|---|
| `audio.sample_rate` | 16000 | Silero 仅 8k/16k；CAM++/pyannote 16k |
| `audio.channels / dtype / blocksize` | 1 / int16 / 512 | Silero 16k 窗口必须 512 样本（32 ms）https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py |
| `vad.threshold / neg_threshold` | 0.5 / 0.35 | Silero 默认 0.5，neg = max(thr−0.15, 0.01) |
| `vad.pre_roll_s` | 1.0 | 环形缓冲，防掐头 |
| `vad.min_silence_s`（hangover） | 0.8 | 句间停顿不切段；Silero 默认 100 ms 过短 |
| `vad.min_speech_s` | 0.5 | 短于此丢弃 |
| `vad.max_segment_s` | 45 | 保证 ASR 块 ≤ 60 s 且 WAV base64 远低于 10 MB |
| `storage.opus_bitrate_kbps` | 24（`compression_level` = 1 − (24000−6000)/250000 = 0.928） | RFC 7587 宽带语音甜点 16–20 kbps，留余量 https://www.rfc-editor.org/rfc/rfc7587.html |
| `storage.fsync_interval_s` | 5 | 崩溃最多丢 5 s |
| `cleanup.min_speech_s` | 1.0 | 低于此删文件 |
| `cleanup.interval_min / retention.audio_days / retention.max_audio_gb` | 60 / 30 / 5 | — |
| `sessions.gap_s` | 600 | — |
| `diarization.cluster_threshold` | 0.5 | sherpa-onnx FastClustering 默认；越大簇越少 |
| `diarization.min_duration_on / off` | 0.3 / 0.5 | 官方示例 |
| `diarization.seg_num_threads / emb_num_threads` | 2 / 2 | — |
| `speaker_id.threshold` | 0.55（**须用 `calibrate` 校准**） | sherpa 示例 0.6（wespeaker zh），ModelScope CAM++ 0.31 不可直接比 |
| `speaker_id.low_conf_margin` | 0.05 | — |
| `speaker_id.min_enroll_clips / clip_len_s` | 3 / 5–20 | — |
| `asr.max_chunk_seconds` | 60（**未验证，可调**） | 10 MB base64 + 社区 >2 min 退化报告 |
| `asr.min_turn_s / merge_gap_s / pad_s` | 0.8 / 0.6 / 0.15 | — |
| `asr.language` | auto | 官方建议手动设更稳；中文为主可设 zh |
| `asr.rpm_limit / max_retries` | 90 / 5 | 官方 100 RPM |
| `correction.batch_chars / seg_min_chars / seg_max_chars` | 2000 / 20 / 100 | ChFT 段落 20–100 字 |
| `correction.max_edit_ratio / pinyin_sim_min` | 0.3 / 0.6 | LIR-ASR 规则约束 |
| `memory.chunk_chars / chunk_max_minutes` | 300–500 / 5 | — |
| `memory.embedding.model / dim` | `BAAI/bge-small-zh-v1.5` / 512 | fastembed 内置 |
| `memory.search.k_lex / k_vec / rrf_k / return_k` | 50 / 50 / 60 / 15 | Anthropic：20 块优于 10 块 |
| `daily.time / weekly.day_time` | 21:35 / Sun 22:00 | 避开整点/半点（飞书 11232 限流）|
| `daily.map_model / reduce_model / chat_model` | flash / flash / pro | — |

---

## 6. config.yaml 字段草案

```yaml
# config.yaml  —— 复制自 config.example.yaml，只在本机保存；MIMO_API_KEY 环境变量可覆盖
mimo:
  api_key: "sk-REPLACE_ME"                       # 按量付费 key；Token Plan 请同时改 base_url
  base_url: "https://api.xiaomimimo.com/v1"
  auth_style: "bearer"                            # bearer | api-key（requests 直连时用）
  chat_model: "mimo-v2.6-flash"
  chat_model_strong: "mimo-v2.6-pro"
  asr_model: "mimo-v2.5-asr"
  timeout_s: 180
  thinking_default: "disabled"

audio:
  device: null                                    # null=默认输入；或 sd.query_devices() 的索引/名称子串
  sample_rate: 16000
  channels: 1
  blocksize: 512
  wasapi_auto_convert: true
  loopback: false                                 # true 需额外安装 PyAudioWPatch（未实现，占位）

vad:
  model: "models/silero_vad.onnx"
  threshold: 0.5
  pre_roll_s: 1.0
  min_silence_s: 0.8
  min_speech_s: 0.5
  max_segment_s: 45
  num_threads: 1

storage:
  data_dir: "data"
  opus_bitrate_kbps: 24
  fsync_interval_s: 5
  private_windows: []                             # [["22:30","07:00"]] 时段只录不传
  pseudonymize_names: true                        # 上云前把登记姓名换成 P1..Pn

cleanup:
  interval_min: 60
  min_speech_s: 1.0
  retention_audio_days: 30
  max_audio_gb: 5

sessions:
  gap_s: 600

diarization:
  segmentation_model: "models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
  embedding_model: "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  cluster_threshold: 0.5
  num_clusters: -1
  min_duration_on: 0.3
  min_duration_off: 0.5
  num_threads: 2

speaker_id:
  threshold: 0.55                                 # 跑 `audio_note calibrate` 后回填
  low_conf_margin: 0.05
  min_enroll_clips: 3
  speakers_dir: "data/speakers"
  owner_name: "刘恩怀"                             # 日报中的"本人"

asr:
  max_chunk_seconds: 60                           # 未验证上限；首日实测后调整
  min_turn_s: 0.8
  merge_gap_s: 0.6
  pad_s: 0.15
  language: "auto"                                # auto | zh | en
  rpm_limit: 90
  max_retries: 5
  upload_format: "wav"                            # wav | mp3（mp3 可容纳更长块）

correction:
  enabled: true
  model: "mimo-v2.6-flash"
  batch_chars: 2000
  seg_min_chars: 20
  seg_max_chars: 100
  max_edit_ratio: 0.3
  pinyin_sim_min: 0.6
  glossary: []                                    # ["MiMo | 米莫,咪摩 | 小米大模型", ...]
  context: ""                                     # 场景/常聊话题

memory:
  embedding:
    backend: "fastembed"
    model: "BAAI/bge-small-zh-v1.5"               # 升级: Qwen/Qwen3-Embedding-0.6B-Q
    dim: 512
    query_prefix: ""                              # Qwen3 需 'Instruct: ...\nQuery:'
  rerank:
    enabled: false
    model: "BAAI/bge-reranker-v2-m3-int8"
    top_n: 40
  chunk_chars: 400
  chunk_max_minutes: 5
  search: {k_lex: 50, k_vec: 50, rrf_k: 60, return_k: 15, neighbours: 1}
  jieba_userdict: "data/memory/userdict.txt"
  memory_md: "data/memory/MEMORY.md"

daily:
  time: "21:35"
  map_model: "mimo-v2.6-flash"
  reduce_model: "mimo-v2.6-flash"
  map_max_chars: 8000
  push: true
weekly:
  day: "sun"
  time: "22:00"

chat:
  model: "mimo-v2.6-pro"
  thinking: "enabled"
  max_tool_rounds: 6

notify:
  channels:
    - type: feishu
      webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE_ME"
      secret: ""                                  # 签名校验密钥，建议开启
      format: card                                # text | post | card
    - type: windows_toast
      fallback: true
    # - type: ntfy
    #   server: https://ntfy.sh
    #   topic: audio-note-REPLACE_WITH_RANDOM
    #   token: ""
    # - type: wxpusher
    #   spt: "SPT_xxx"
    # - type: serverchan
    #   sendkey: "SCTxxx"
  max_bytes_per_message: 18000

process:
  single_instance_name: "Local\\audio_note_recorder"
  tray: true
  log_file: "data/logs/audio_note.log"
  log_level: "INFO"
```

---

## 7. 依赖清单与 Python 版本

### 7.1 Python 版本：**3.12**（`uv python pin 3.12`）

| 包 | cp312 | cp313 | cp314 | 来源 |
|---|---|---|---|---|
| sherpa-onnx 2.0.1 | ✔（本机验证） | ✔ | ✔（本机验证） | https://pypi.org/pypi/sherpa-onnx/json |
| sounddevice 0.5.6 / soundfile 0.14.0 | 纯 wheel | ✔ | ✔ | https://pypi.org/project/sounddevice/#files |
| onnxruntime 1.31.1 | ✔ | ✔ | ✔ | https://pypi.org/project/onnxruntime/ |
| torch 2.15.0+cpu（仅 pyannote 备选） | ✔ | ✔ | ✔ | https://download.pytorch.org/whl/cpu/torch/ |
| winrt-runtime（win11toast） | ✔ | ✔ | ✔ | https://pypi.org/simple/winrt-runtime/ |
| PyAudio 0.2.14（仅 loopback 备选需 PyAudioWPatch，后者 cp314 有） | ✔ | ✔ | ✘ | https://pypi.org/project/PyAudio/#files |
| funasr 1.4.17 / modelscope 1.40.2（备选 ASR/分离） | ✔（classifier 上限 3.12） | 未声明 | 未声明 | https://pypi.org/pypi/funasr/json |
| fastembed 0.9.1 | ✔ | 未核查 | 未核查 | https://github.com/qdrant/fastembed/releases |
| webrtcvad-wheels 2.0.14.post1（可选） | ✔ | ✔ | ✔（无 cp315） | https://pypi.org/project/webrtcvad-wheels/ |

结论：主栈（sherpa-onnx/sounddevice/soundfile/onnxruntime/torch/winrt）在 3.12–3.14 都有 Windows wheel，3.14 **不是**障碍；但 3.12 是唯一让 **所有备选路径**（FunASR、modelscope、PyAudio）也有已声明支持的版本，且 fastembed 在 3.13/3.14 的元数据未核查。因此 pin 3.12；若将来不需要 FunASR 备选，可升 3.13。

### 7.2 安装

```powershell
cd C:\path\to\audio_note
uv python pin 3.12
uv venv .venv
uv pip install -p .venv\Scripts\python.exe `
  sounddevice==0.5.6 soundfile==0.14.0 numpy `
  sherpa-onnx==2.0.1 `
  openai requests pyyaml `
  jieba==0.42.1 pypinyin sqlite-vec==0.1.9 fastembed==0.9.1 `
  pystray==0.19.5 pillow win11toast==0.36.3 winotify==1.1.0 filelock
# 可选：ten-vad==1.0.6.9  static-ffmpeg  soxr
# 备选说话人分离（不装也行）：
# uv pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu ; uv pip install pyannote.audio==4.0.7
```
`openai` SDK 版本未在调研中核实，用最新（官方文档 `pip install -U openai`）。

### 7.3 模型下载（`scripts/download_models.ps1`，放入 `models/`）

| 文件 | 大小 | URL |
|---|---|---|
| silero_vad.onnx | 0.64 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx |
| sherpa-onnx-pyannote-segmentation-3-0.tar.bz2（用 model.int8.onnx） | 7.0 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 |
| 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx | 28.3 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx |
| 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx（纯中文备选） | 28.3 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx |
| bge-small-zh-v1.5 | 90 MB | fastembed 首次运行自动下载（HF） |

注意 release tag 拼写就是 `speaker-recongition-models`。不要用 `reverb-diarization-v1`（非商业许可）。

### 7.4 Windows 任务注册（`scripts/install_task.ps1`）

```powershell
schtasks /create /sc ONLOGON /tn "AudioNote Recorder" /rl LIMITED /delay 0000:30 /f `
  /tr "\"C:\path\to\audio_note\.venv\Scripts\pythonw.exe\" \"C:\path\to\audio_note\run.py\""
# taskschd.msc：设置 → 取消"如果任务运行时间超过 3 天则停止"；条件 → 取消"只有在使用交流电源时才启动"
# Windows 设置 → 隐私和安全性 → 麦克风 → 允许桌面应用访问（未从文档验证，需本机确认）
```

---

## 8. 风险与未验证点

1. **ASR 单次时长上限未文档化**：仅 10 MB base64 / 2K 输出 token；首日用 30/60/120/240 s 实测 `finish_reason` 与质量，回填 `asr.max_chunk_seconds`。
2. **ASR 10K TPM 是否按音频 token 计**：若计入（约 6.25 tok/s 的话 ≈ 27 分钟音频/分钟），需限速；通过 `usage.prompt_tokens_details.audio_tokens` 验证。
3. **ASR 无时间戳/无说话人/无热词**：设计已用 diarization-first + 自有切块时间戳 + LLM 术语表补偿；句内多句的时间为字符比例插值（近似）。
4. **说话人阈值未校准**：CAM++ zh_en advanced 的同/异人分数分布未在本机测量；`calibrate` 命令必须在登记后运行；低置信簇交给用户 `label-unknown`。
5. **sherpa-onnx CAM++ 在 Windows CPU 的 RTF 未测**（文档数据为 macOS、ERes2Net/TitaNet）；按 10 分钟片段测一次，必要时换 `nemo_en_titanet_small`（英文）或减少线程争用。
6. **Opus 二次有损**：存储 Opus 24 kbps → 解码 → WAV 上传；对 ASR 的影响未测。若 CER 明显上升，改为 32 kbps 或直接存 MP3 32 kbps 原样上传。
7. **隐私**：MiMo 政策只承诺文本不训练、音频未明示、数据存境内；已加伪名化与 private_windows，但音频本身仍上传。
8. **飞书账号类型**：个人版是否能看到"群机器人 → 自定义机器人"入口，官方文档未说明；个人版注册疑似已关闭 → 建议建免费团队。卡片 JSON 2.0 header/element 结构已核实。
9. **fastembed 在 Python 3.13/3.14 未核查**；bge-m3 自定义 ONNX（外置 .onnx_data）需 onnxruntime ≥ 1.24 与 huggingface_hub ≥ 1.32（已满足但未实测）。
10. **sqlite-vec 为 pre-v1、暴力 KNN**：~36k 向量/年按月分区足够；0.1.10 预发布 API 可能变化，锁 0.1.9。
11. **FTS5 中文**：主表用 jieba 预分词 + unicode61；trigram 不能匹配 1–2 字查询，故不做主表；`grep_raw` 用 LIKE 兜底。LLM 纠错可能改坏人名 → 同时索引 `text_raw`。
12. **纠错提示词为论文结论的合成**，非公开模板；需用一周真实录音做 CER before/after 对比，调 `max_edit_ratio` / 拼音阈值。
13. **sounddevice 无 WASAPI loopback**：系统声音（线上会议对方声音）不会被录到；需要时加 PyAudioWPatch 第二路。
14. **Task Scheduler + pythonw 的麦克风权限**：未从官方来源验证，需本机测试。
15. **单实例互斥量可被恶意预占**（微软文档提示）；个人机器可接受，或改 filelock。
16. **中文 token 比例 0.6/字**为 DeepSeek 数据，MiMo 需用 `usage` 校准预算。

---

## 9. 关键代码片段（原样摘录，附来源）

### 9.1 MiMo ASR（官方 Python 示例模式）
来源：https://mimo.mi.com/docs/en-US/quick-start/usage-guide/audio/Speech-Recognition
```python
import os
import base64
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("MIMO_API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

with open("audio_file.wav", "rb") as f:
    audio_bytes = f.read()
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

completion = client.chat.completions.create(
    model="mimo-v2.5-asr",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:audio/wav;base64,{audio_base64}"
                    }
                }
            ]
        }
    ],
    extra_body={
        "asr_options": {
            "language": "auto"   # 'auto' | 'zh' | 'en'
        }
    },
    # stream=True  -> iterate chunks, text in chunk.choices[0].delta.content
)

text = completion.choices[0].message.content      # punctuated transcript
seconds = completion.usage.seconds               # billed audio duration (s)
print(text, seconds)
```

### 9.2 MiMo ASR（requests 直连，镜像官方 curl）
来源：https://mimo.mi.com/docs/en-US/api/audio/Speech-Recognition
```python
import os, base64, requests

url = "https://api.xiaomimimo.com/v1/chat/completions"
headers = {"api-key": os.environ["MIMO_API_KEY"], "Content-Type": "application/json"}
b64 = base64.b64encode(open("clip.mp3", "rb").read()).decode()
assert len(b64) <= 10 * 1024 * 1024, "base64 payload must be <= 10 MB"
body = {
    "model": "mimo-v2.5-asr",
    "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": f"data:audio/mpeg;base64,{b64}"}}
    ]}],
    "asr_options": {"language": "zh"},
    "stream": False,
}
r = requests.post(url, headers=headers, json=body, timeout=180)
r.raise_for_status()
j = r.json()
print(j["choices"][0]["message"]["content"], j["usage"]["seconds"])
```

### 9.3 MiMo Chat（thinking 关闭，纠错/日报）
来源：https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MIMO_API_KEY"],          # sk-... pay-as-you-go key
    base_url="https://api.xiaomimimo.com/v1",   # OpenAI-compatible
)

resp = client.chat.completions.create(
    model="mimo-v2.6-flash",
    messages=[
        {"role": "system", "content": "你是语音识别文本的校对助手，只输出修正后的文本。"},
        {"role": "user", "content": "下面是ASR转写，请修正同音错别字并保持原意：..."},
    ],
    max_completion_tokens=2048,
    temperature=0.3,
    extra_body={"thinking": {"type": "disabled"}},   # non-standard field -> extra_body
    # response_format={"type": "json_object"},        # optional JSON mode
)
print(resp.choices[0].message.content)
# with thinking enabled, reasoning is in resp.choices[0].message.reasoning_content
```

### 9.4 sherpa-onnx 说话人分离（官方示例精简）
来源：https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py
```python
import sherpa_onnx, soundfile as sf

config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model="models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx", window_shift_ratio=0.1),
        num_threads=2, provider="cpu"),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model="models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", num_threads=2),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
    min_duration_on=0.3, min_duration_off=0.5)
assert config.validate()
sd = sherpa_onnx.OfflineSpeakerDiarization(config)
audio, sr = sf.read("chunk.wav", dtype="float32", always_2d=True); audio = audio[:, 0]
assert sr == sd.sample_rate  # 16000
segs = sd.process(audio).sort_by_start_time()
for r in segs:
    print(f"{r.start:.3f} {r.end:.3f} speaker_{r.speaker:02}")
```

### 9.5 sherpa-onnx 登记 + 识别
来源：https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification-with-vad.py （方法签名见 https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/sherpa-onnx/python/csrc/speaker-embedding-manager.cc ）
```python
import json, numpy as np, sherpa_onnx, soundfile as sf

cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model="models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", num_threads=2, provider="cpu")
assert cfg.validate()
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

def embed(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    s = extractor.create_stream()
    s.accept_waveform(sample_rate=sr, waveform=samples)
    s.input_finished()
    assert extractor.is_ready(s)
    return np.asarray(extractor.compute(s), dtype=np.float32)

# enroll: store every clip embedding per person in speakers/<name>.json
# load: manager = SpeakerEmbeddingManager(extractor.dim); manager.add(name, [emb1, emb2, ...])  # averaged
manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
for name, embs in json.load(open("speakers.json")).items():
    manager.add(name, [np.asarray(e, dtype=np.float32) for e in embs])

# identify a diarized cluster
THRESH = 0.55  # calibrate; sherpa examples default 0.6, modelscope pairwise 0.31
name = manager.search(embed(cluster_audio), threshold=THRESH) or "unknown"
score = manager.score(name, embed(cluster_audio)) if name != "unknown" else None
```

### 9.6 Silero VAD 录音循环（sherpa-onnx VoiceActivityDetector，镜像 vad-microphone.py）
来源：https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/python-api-examples/vad-microphone.py
```python
import numpy as np, sounddevice as sd, soundfile as sf, sherpa_onnx, time, datetime as dt
SR = 16000
cfg = sherpa_onnx.VadModelConfig()
cfg.silero_vad.model = r"models\silero_vad.onnx"   # https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
cfg.silero_vad.threshold = 0.5
cfg.silero_vad.min_silence_duration = 0.8      # hangover seconds
cfg.silero_vad.min_speech_duration = 0.25
cfg.silero_vad.max_speech_duration = 45
cfg.silero_vad.window_size = 512
cfg.sample_rate = SR; cfg.num_threads = 1
vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
win = cfg.silero_vad.window_size
buf = np.zeros(0, np.float32)
with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=win) as s:
    while True:
        samples, _ = s.read(int(0.1 * SR))        # blocking 100 ms reads
        buf = np.concatenate([buf, samples[:, 0]])
        while len(buf) >= win:
            vad.accept_waveform(buf[:win]); buf = buf[win:]
        while not vad.empty():
            seg = vad.front; vad.pop()
            start = time.time() - len(buf) / SR - len(seg.samples) / SR   # approx wall-clock of segment start
            name = dt.datetime.fromtimestamp(start).strftime("%Y%m%d_%H%M%S") + ".opus"
            sf.write(name, np.asarray(seg.samples, np.float32), SR, format="OGG", subtype="OPUS", compression_level=0.928)
```

### 9.7 Silero VAD 录音循环（torch-free onnxruntime 版，带 pre-roll/hangover/fsync，备选）
来源：https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py
```python
import os, queue, threading, time, collections, datetime as dt
import numpy as np, sounddevice as sd, soundfile as sf, onnxruntime as ort

SR, CHUNK = 16000, 512                     # Silero needs exactly 512 samples @16k (32 ms)
CTX = 64                                    # v5/v6 context samples carried between chunks
THRESH, NEG_THRESH = 0.5, 0.35              # hysteresis like VADIterator (neg = thr-0.15)
PRE_ROLL_S, HANGOVER_S, MAX_SEG_S = 1.0, 0.8, 45.0
OPUS_LEVEL = 1.0 - (24000 - 6000) / 250000  # libsndfile: bitrate=(1-q)*250000+6000 -> 24 kbps
OUT = r"C:\path\to\audio_note\data\segments"
os.makedirs(OUT, exist_ok=True)

class SileroOnnx:
    def __init__(self, path):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.reset()
    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32); self.ctx = np.zeros((1, CTX), np.float32)
    def __call__(self, pcm_i16: np.ndarray) -> float:
        x = (pcm_i16.astype(np.float32) / 32768.0)[None, :]
        inp = np.concatenate([self.ctx, x], axis=1)
        prob, self.state = self.s.run(None, {"input": inp, "state": self.state, "sr": np.array(SR, np.int64)})
        self.ctx = x[:, -CTX:]
        return float(prob.squeeze())

q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

def audio_cb(indata, frames, t, status):      # real-time thread: copy only, never block
    if status: print("audio status:", status, flush=True)
    try: q.put_nowait(indata[:, 0].copy())
    except queue.Full: pass                      # drop rather than stall the audio engine

class SegmentWriter:
    """Raw int16 PCM .part file, fsync'd periodically; transcoded to Opus on close."""
    def __init__(self, start_ts: float):
        self.start_ts = start_ts
        stem = dt.datetime.fromtimestamp(start_ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.part = os.path.join(OUT, stem + ".pcm.part"); self.final = os.path.join(OUT, stem + ".opus")
        self.f = open(self.part, "wb"); self.n = 0; self.last_sync = time.monotonic()
    def write(self, pcm_i16):
        self.f.write(pcm_i16.tobytes()); self.n += len(pcm_i16)
        if time.monotonic() - self.last_sync > 5:
            self.f.flush(); os.fsync(self.f.fileno()); self.last_sync = time.monotonic()
    def close(self):
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close()
        pcm = np.fromfile(self.part, dtype=np.int16)
        sf.write(self.final, pcm, SR, format="OGG", subtype="OPUS", compression_level=OPUS_LEVEL)
        os.remove(self.part)
        return self.final, self.n / SR

def worker(stop: threading.Event):
    vad = SileroOnnx(r"models\silero_vad.onnx")
    pre = collections.deque(maxlen=int(PRE_ROLL_S * SR / CHUNK))
    seg, silence_s, seg_s = None, 0.0, 0.0
    chunk_s = CHUNK / SR
    while not stop.is_set():
        try: block = q.get(timeout=0.5)
        except queue.Empty: continue
        p = vad(block); now = time.time()
        if seg is None:
            pre.append(block)
            if p >= THRESH:
                seg = SegmentWriter(now - len(pre) * chunk_s)
                for b in pre: seg.write(b)
                pre.clear(); silence_s = seg_s = 0.0
        else:
            seg.write(block); seg_s += chunk_s
            silence_s = 0.0 if p >= NEG_THRESH else silence_s + chunk_s
            if silence_s >= HANGOVER_S or seg_s >= MAX_SEG_S:
                path, dur = seg.close(); seg = None
                print(f"segment {path} {dur:.1f}s", flush=True)   # -> enqueue for ASR
    if seg: seg.close()

if __name__ == "__main__":
    stop = threading.Event(); th = threading.Thread(target=worker, args=(stop,), daemon=True); th.start()
    dev = None   # or an index / substring from sd.query_devices(); WASAPI devices may need
                 # extra_settings=sd.WasapiSettings(auto_convert=True) to open at 16 kHz
    with sd.InputStream(samplerate=SR, blocksize=CHUNK, channels=1, dtype="int16", device=dev, callback=audio_cb):
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: pass
    stop.set(); th.join()
```
注意：此版用官方 v5/v6 `silero_vad.onnx`（输入 `input/state/sr`，https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx ），与 k2-fsa 导出的 v4 布局（`x/h/c`）不同，不可混用。

### 9.8 清理任务 + 孤儿 .part 恢复
来源：https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
```python
import os, glob, time, numpy as np, soundfile as sf
SR, CHUNK = 16000, 512
MIN_SPEECH_S, RETENTION_DAYS = 1.0, 30

def speech_seconds(vad, pcm_f32):
    vad.reset(); n = 0
    for i in range(0, len(pcm_f32) - CHUNK + 1, CHUNK):
        if vad((pcm_f32[i:i+CHUNK] * 32767).astype(np.int16)) >= 0.5: n += 1
    return n * CHUNK / SR

def cleanup(seg_dir, vad):
    now = time.time()
    for p in glob.glob(os.path.join(seg_dir, "*.opus")):
        if now - os.path.getmtime(p) > RETENTION_DAYS * 86400:
            os.remove(p); continue
        if os.path.exists(p + ".checked"): continue
        audio, sr = sf.read(p, dtype="float32")
        if sr != SR or speech_seconds(vad, audio) < MIN_SPEECH_S:
            os.remove(p)
        else:
            open(p + ".checked", "w").close()   # or record in a sqlite index

def recover_parts(seg_dir, opus_level=0.928):
    for part in glob.glob(os.path.join(seg_dir, "*.pcm.part")):
        pcm = np.fromfile(part, dtype=np.int16)          # raw PCM has no header to repair
        if len(pcm) >= SR // 2:
            sf.write(part.replace(".pcm.part", ".opus"), pcm, SR, format="OGG", subtype="OPUS", compression_level=opus_level)
        os.remove(part)
```

### 9.9 单实例互斥量
来源：https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createmutexw
```python
import ctypes, sys
from ctypes import wintypes
ERROR_ALREADY_EXISTS = 183
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateMutexW.restype = wintypes.HANDLE
k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]

def acquire_single_instance(name="Local\\audio_note_recorder"):
    h = k32.CreateMutexW(None, False, name)          # keep h alive for the process lifetime
    if not h or ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        print("audio_note is already running", file=sys.stderr); sys.exit(1)
    return h

_MUTEX = acquire_single_instance()
# pure-Python alternative: from filelock import FileLock, Timeout
#   lock = FileLock(os.path.expanduser('~/.audio_note/recorder.lock'))
#   try: lock.acquire(timeout=0)
#   except Timeout: sys.exit(1)
```

### 9.10 飞书自定义机器人（text / post / card，可选签名）
来源：https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot
```python
import base64, hashlib, hmac, time, requests

WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx"
SECRET = ""  # optional, from the bot's 签名校验 setting


def _sign(secret: str, ts: int) -> str:
    key = f"{ts}\n{secret}".encode("utf-8")
    return base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode()


def feishu_send(payload: dict) -> dict:
    if SECRET:
        ts = int(time.time())
        payload = {"timestamp": str(ts), "sign": _sign(SECRET, ts), **payload}
    r = requests.post(WEBHOOK, json=payload, timeout=10)
    data = r.json()
    if data.get("code") != 0:  # 11232 rate limit, 19021 sign, 19024 keyword
        raise RuntimeError(f"feishu error {data}")
    return data

# text
feishu_send({"msg_type": "text", "content": {"text": "今日摘要已生成"}})

# post (rich text)
feishu_send({"msg_type": "post", "content": {"post": {"zh_cn": {
    "title": "每日摘要 2026-10-31",
    "content": [
        [{"tag": "text", "text": "共 12 段对话，3 位说话人。"}],
        [{"tag": "a", "text": "打开本地报告", "href": "http://127.0.0.1:8000/daily/2026-10-31"}],
    ]}}}})

# interactive card (markdown body)
feishu_send({"msg_type": "interactive", "card": {
    "schema": "2.0",
    "header": {"title": {"tag": "plain_text", "content": "每日摘要 2026-10-31"}},
    "body": {"elements": [{"tag": "markdown", "content": "**要点**\n- 上午会议：...\n- 待办：..."}]},
}})
```
成功响应实际为 `{"StatusCode":0,"StatusMessage":"success","code":0,"data":{},"msg":"success"}`，只判 `code == 0`。

### 9.11 推送通道抽象
来源：https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot
```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("audio_note.notify")

@dataclass
class Message:
    title: str
    body_md: str                # markdown source; channels downgrade as needed
    url: str | None = None      # optional click-through (local web UI)
    tags: list[str] = field(default_factory=list)
    priority: int = 0           # -1 low, 0 normal, 1 high

class Channel(Protocol):
    name: str
    def send(self, msg: Message) -> None: ...   # raise on failure

class Notifier:
    """Fan-out: try primary channels; if all fail, use fallbacks (Windows toast)."""
    def __init__(self, primary: list[Channel], fallback: list[Channel]):
        self.primary, self.fallback = primary, fallback

    def send(self, msg: Message) -> list[str]:
        delivered = []
        for ch in self.primary:
            try:
                ch.send(msg); delivered.append(ch.name)
            except Exception as e:  # noqa: BLE001
                log.warning("channel %s failed: %s", ch.name, e)
        if not delivered:
            for ch in self.fallback:
                try:
                    ch.send(msg); delivered.append(ch.name); break
                except Exception as e:  # noqa: BLE001
                    log.warning("fallback %s failed: %s", ch.name, e)
        return delivered
```

### 9.12 Windows toast（win11toast → winotify 兜底）
来源：https://github.com/GitHub30/win11toast
```python
def windows_toast(title: str, body: str, click_url: str | None = None) -> None:
    try:
        from win11toast import notify  # non-blocking; needs winrt-* wheels (Python >= 3.11)
        notify(title, body, on_click=click_url, app_id="audio_note", duration="long")
        return
    except Exception:
        pass
    from winotify import Notification, audio  # pure Python, shells out to PowerShell
    t = Notification(app_id="audio_note", title=title, msg=body, duration="long")
    if click_url:
        t.add_actions(label="打开", launch=click_url)
    t.set_audio(audio.Default, loop=False)
    t.show()
```

### 9.13 ASR 纠错提示词（分段 JSON diff）
来源（合成依据）：https://arxiv.org/html/2409.07790v2 ；https://arxiv.org/html/2407.01909 ；https://arxiv.org/html/2509.15095v1 ；https://developers.openai.com/cookbook/examples/whisper_correct_misspelling
```text
[system]
你是中文语音转写(ASR)的校对员。输入是按句切分的转写片段列表，每个片段有 id、speaker、start、end、text、pinyin 字段。
你的任务：只找出“听错”的地方并给出修正，不做润色、不改语气、不删口语词、不合并或拆分片段、不改动 speaker/start/end。

规则：
1. 常见错误几乎都是同音字/近音字、人名、地名、产品名、数字、专业术语。修正后的词读音必须与原文相同或非常接近（参考 pinyin 字段）。读音差异很大的改动一律不要做。
2. 优先依据【术语表】。术语表给出“正确写法 | 常见误写 | 说明”，命中时直接替换。术语表之外的专有名词如无把握，保持原样。
3. 数字、金额、时间、单位：只在明显听错时修正，绝不把数字替换成术语表中的词。
4. 看不懂或存疑的句子：不要猜，保持原样；可在 note 中写“存疑”。
5. 每个片段的修改不得超过其字数的 30%；超过说明你在重写，请放弃该修改。
6. 只输出 JSON，不要解释，不要 markdown 代码块。无需修改时输出 {"edits": []}。

输出格式：
{"edits": [{"id": "片段id", "old": "原文中要替换的最短子串", "new": "替换后的子串", "reason": "同音|术语表|人名|数字|存疑"}]}

【术语表】
{glossary_lines}

【对话背景】（可选，用户填写：场景、参与人、常聊话题）
{context}

[user]
{"segments": [
  {"id": "s0012", "speaker": "刘恩怀", "start": 3661.2, "end": 3665.8, "text": "我们下周把mimo的ASR接到流水线里，然后让小张看一下。", "pinyin": "wo men xia zhou ba mimo de ASR jie dao liu shui xian li ..."},
  ...
]}
```

### 9.14 纠错结果本地校验
来源：https://arxiv.org/html/2509.15095v1
```python
from pypinyin import lazy_pinyin
import difflib, re

def accept_edit(seg_text: str, old: str, new: str, max_ratio=0.3) -> bool:
    if old not in seg_text or old == new or not new:
        return False
    # 1) 编辑比例上限
    if len(new) > len(old) * 2 or (len(old) - len(new)) > len(old) * 0.5:
        return False
    if len(old) / max(len(seg_text), 1) > max_ratio:
        return False
    # 2) 数字/金额保护：不允许把数字改没
    if re.search(r'\d', old) and not re.search(r'\d', new):
        return False
    # 3) 拼音相似度（忽略声调）
    po, pn = ' '.join(lazy_pinyin(old)), ' '.join(lazy_pinyin(new))
    sim = difflib.SequenceMatcher(None, po, pn).ratio()
    return sim >= 0.6  # 术语表命中可放宽到 0.0

def apply_edits(segments, edits, glossary_terms):
    by_id = {s['id']: s for s in segments}
    for e in edits:
        s = by_id.get(e['id'])
        if not s:
            continue
        ok = e['new'] in glossary_terms or accept_edit(s['text'], e['old'], e['new'])
        if ok:
            s['text'] = s['text'].replace(e['old'], e['new'], 1)
            s.setdefault('edits', []).append(e)
    return segments
```

### 9.15 每日分析 Map 提示词
来源：https://raw.githubusercontent.com/BasedHardware/omi/main/backend/utils/llm/conversation_notes_prompts.py
```text
[system]
你是用户的私人生活日志助理。用户名：{owner_name}。下面是 {date} {start_time}-{end_time} 之间的一段对话转写，格式为 `[时间] 说话人: 文本`。说话人若为 SPK_x 表示未识别身份，不要猜测其姓名。

请仅依据转写内容输出 JSON（不要解释、不要 markdown）：
{
  "is_meaningful": true/false,          // 闲聊寒暄、碎片噪声、对语音助手的简短指令 -> false
  "title": "≤15字的小标题",
  "summary": "3-5句客观概述，谁和谁谈了什么、结论是什么",
  "topics": ["话题1", "话题2"],
  "decisions": [{"what": "做出的决定", "why": "理由（原文有才写）", "time": "HH:MM"}],
  "todos": [{"task": "只记录 {owner_name} 自己承诺做或被明确要求做的事", "owner": "{owner_name}", "due": "", "source_time": "HH:MM"}],
  "people": [{"name": "仅用转写中出现的已识别姓名", "role_or_relation": "", "mentioned_facts": ["原文提到的事实"]}],
  "open_loops": ["悬而未决的问题 / 等对方回复的事 / 说了要做但没定时间的事"],
  "mood": {"score": 1-10, "evidence": "支持该判断的原话片段"},
  "key_quotes": [{"time": "HH:MM", "speaker": "", "text": "值得原样保留的一句话"}]
}

约束：
- 不要编造名字、日期、金额、单位；原文没有的信息留空字符串或空数组。
- due 默认留空，只有原文明确说了具体日期才填。
- 未识别说话人的话用“有人说”表述，不要归到具体的人。
- 可理解的中英混说内容不是噪声。

[user]
{transcript_block}
```

### 9.16 每日分析 Reduce 提示词
来源：https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/how_to/summarize_map_reduce.ipynb
```text
[system]
你是 {owner_name} 的私人生活日志助理。下面是 {date} 当天按时间顺序排列的多段对话分析结果（JSON 数组），以及昨日遗留的 open_loops。
请生成当日日报，输出 JSON：
{
  "push_message": "≤200字的推送文案：今天最重要的3件事、1-3条待办、1条提醒（昨日未闭环的事）。口语化、无标题、无列表符号。",
  "report_markdown": "## {date} 日报\n### 今日概览\n（按时间顺序，每段对话一行：时间 · 标题 · 一句话）\n### 决定\n### 待办（合并去重，标注来源时间）\n### 人物\n（今天接触的人及新获知的事实）\n### 未闭环 / 需要跟进\n（合并昨日遗留；已在今天解决的标记为已闭环）\n### 情绪与状态\n（整日 1-10 分及依据，不要过度解读）\n### 值得记住的话",
  "open_loops_carry_over": ["仍未闭环的事项，供明天使用"],
  "questions_for_user": ["最多3个，用于澄清转写中存疑的人名/术语/决定"]
}
规则：只合并、去重、按时间排序，不新增转写中没有的信息；不同块中指向同一件事的 todo/open_loop 合并为一条；去掉 is_meaningful=false 的块。

[user]
{"date": "{date}", "carry_over_open_loops": [...], "blocks": [ ...map 阶段输出... ]}
```

### 9.17 记忆问答系统提示词
来源：https://raw.githubusercontent.com/BasedHardware/omi/main/backend/utils/llm/conversation_processing.py
```text
[system]
你是 {owner_name} 的记忆助手。只能依据下面检索到的资料回答（资料包含日报摘要和原始转写片段，带日期、时间、说话人）。
- 回答时引用来源，如“(10-28 14:32 王总)”。
- 资料中没有的内容，直接说“记录里没有找到”，不要猜。
- 转写可能有同音字错误；当两处记录冲突时，指出冲突并给出你认为更可能的解释。
今天是 {today}。

[资料]
{retrieved_chunks}

[user]
{question}
```

### 9.18 混合检索核心（FTS5 bm25 + sqlite-vec KNN + RRF）
来源：https://www.sqlite.org/fts5.html ；https://alexgarcia.xyz/sqlite-vec/features/knn.html
```python
import jieba, sqlite_vec, numpy as np

def search_memory(db, embed, q, t_from=None, t_to=None, speakers=(), tier='chunks', k=20, fuse_k=60):
    tok = ' '.join(jieba.cut_for_search(q))
    where, args = [], []
    if t_from: where.append('c.start_ts >= ?'); args.append(t_from)
    if t_to:   where.append('c.start_ts <  ?'); args.append(t_to)
    if speakers:
        where.append('(' + ' OR '.join(['c.speaker_ids LIKE ?']*len(speakers)) + ')')
        args += [f'%,{s},%' for s in speakers]
    w = (' AND ' + ' AND '.join(where)) if where else ''
    lex = db.execute(f"""SELECT c.id, bm25(chunks_fts, 1.0, 0.5) AS s FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ? {w} ORDER BY s LIMIT 50""", [tok, *args]).fetchall()
    qv = embed([q], is_query=True)[0].astype(np.float32)
    months = month_range(t_from, t_to)            # partition-key pruning; None -> all months
    vec = db.execute(f"""SELECT v.chunk_id, v.distance FROM chunks_vec v
        WHERE v.emb MATCH ? AND k = 50 {'AND v.month BETWEEN ? AND ?' if months else ''}
        {'AND v.day >= ?' if t_from else ''} {'AND v.day < ?' if t_to else ''}""",
        [qv.tobytes(), *(months or []), *([t_from[:10]] if t_from else []), *([t_to[:10]] if t_to else [])]).fetchall()
    scores = {}
    for rank, (cid, _) in enumerate(lex): scores[cid] = scores.get(cid, 0) + 1/(fuse_k+rank+1)
    for rank, (cid, _) in enumerate(vec): scores[cid] = scores.get(cid, 0) + 1/(fuse_k+rank+1)
    top = sorted(scores, key=scores.get, reverse=True)[:40]
    if RERANK: top = rerank(q, top)[:k]            # fastembed TextCrossEncoder('BAAI/bge-reranker-v2-m3-int8')
    return hydrate(db, top[:k])                     # id, day, start_ts, speakers, header, text, neighbours
```
（本项目表名为 `mem_chunks` / `mem_chunks_fts` / `mem_chunks_vec`，实现时替换。）

### 9.19 聊天智能体循环（MiMo function calling）
来源：https://mimo.mi.com/docs/en-US/quick-start/summary/first-api-call
```python
TOOLS = [search_memory, read_session, read_day, read_entity, list_sessions, grep_raw]
system = f"今天是 {today}。你是用户的个人记忆助手。\n" + open('memory/MEMORY.md').read() + "\n" + last_two_day_summaries()
msgs = [{'role':'system','content':system}, {'role':'user','content':question}]
for _ in range(6):
    r = client.chat.completions.create(model=cfg.chat_model, messages=msgs, tools=TOOL_SCHEMAS)
    m = r.choices[0].message; msgs.append(m)
    if not m.tool_calls: break                     # final answer with [session@ts, speaker] citations or 明确说明没有记录
    for tc in m.tool_calls:
        out = dispatch(tc)                         # relative-time parsing + speaker alias resolution happen inside dispatch
        msgs.append({'role':'tool','tool_call_id':tc.id,'content':json.dumps(out, ensure_ascii=False)})
```
注意：thinking 开启时 `msgs.append(m)` 必须保留 `reasoning_content`，否则 MiMo 返回 400。

### 9.20 伪名化
来源：https://privacy.mi.com/XiaomiMiMoPlatform/en_GB/
```python
import re

class Pseudonymizer:
    """name -> 稳定别名（P1, P2 ...），映射只存本地；LLM 返回后反向替换。"""
    def __init__(self, names, extra_terms=()):
        self.fwd = {}
        for i, n in enumerate(sorted(set(names) | set(extra_terms), key=len, reverse=True), 1):
            self.fwd[n] = f'P{i}'
        self.rev = {v: k for k, v in self.fwd.items()}
        self.pat = re.compile('|'.join(map(re.escape, self.fwd))) if self.fwd else None

    def forward(self, text):
        return self.pat.sub(lambda m: self.fwd[m.group(0)], text) if self.pat else text

    def backward(self, text):
        return re.sub(r'P\d+', lambda m: self.rev.get(m.group(0), m.group(0)), text)
```

### 9.21 Task Scheduler + 托盘
来源：https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create ；https://pystray.readthedocs.io/en/latest/usage.html
```powershell
# Task Scheduler, runs at this user's logon, 30 s delay, least privilege; pythonw => no console
schtasks /create /sc ONLOGON /tn "AudioNote Recorder" /rl LIMITED /delay 0000:30 /f `
  /tr "\"C:\path\to\audio_note\.venv\Scripts\pythonw.exe\" \"C:\path\to\audio_note\run.py\""
# then in taskschd.msc: Settings tab -> untick 'Stop the task if it runs longer than: 3 days';
# Conditions tab -> untick 'Start only if on AC power'. Start it now: schtasks /run /tn "AudioNote Recorder"

# --- tray icon (python, pystray 0.19.5 + Pillow) ---
# from PIL import Image, ImageDraw
# from pystray import Icon, Menu, MenuItem
# img = Image.new('RGB', (64, 64), 'red'); ImageDraw.Draw(img).ellipse((16,16,48,48), fill='white')
# icon = Icon('audio_note', icon=img, menu=Menu(
#     MenuItem('Pause recording', lambda i, m: paused.set()),
#     MenuItem('Resume', lambda i, m: paused.clear()),
#     MenuItem('Quit', lambda i, m: (stop.set(), i.stop()))))
# icon.run()   # blocking; on Windows it is documented as safe to call from a non-main thread
```

---

## 10. 今日实施顺序（建议）

1. `uv venv` + 依赖 + 模型下载；`audio_note record` 跑 10 分钟，确认 `.opus` 生成、cleanup 删静音。
2. `audio_note enroll 刘恩怀 clip1.wav clip2.wav clip3.wav` ×（每人 ≥3 段）→ `audio_note calibrate` 回填阈值。
3. 用一段 2 分钟多人录音跑 `worker` 一轮：检查 turns → ASR（落盘原始响应 JSON 验证无时间戳）→ 纠错 edits → `transcripts/*.md`。
4. 实测 ASR 30/60/120/240 s 片段，定 `asr.max_chunk_seconds`。
5. 飞书建群 → 自定义机器人 → 填 webhook/secret → `audio_note daily --date today --dry-run` 再真推。
6. 注册 Task Scheduler；次日早上检查 `data/logs`、`job_log`、`daily/`。