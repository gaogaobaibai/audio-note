# audio-note 1.0：总体升级设计（SPEC v1）

本文是 1.0 升级的**总契约**：数据结构、模块边界、函数签名、接口形状、事件名、文件归属都以这里为准。
各工作包（§11）并行实施，只改自己名下的文件；需要别人名下的东西，按这里的契约调用，不要自己去改。

约定同前几轮：中文界面文案与文档、英文代码注释、UTF-8 + LF；不新增第三方依赖（本轮唯一例外：`pypinyin`，已装进 `.venv`，
要写进 `pyproject.toml`）；测试不联网、不用麦克风、不碰真实数据。

## 0. 目标与原则

用户的要求（原话概括）：系统级重构；记忆怎么管理；更准地分辨声音且本机开销不能大；界面更漂亮；用大模型提高转写准确率——
不只是猜上下文，要**基于记忆校准**，不确定的词**问用户**，问过以后能自动猜、自动标；并且放开想象力做一次大升级。

原则：

1. **自动决定宁缺毋滥，拿不准就问。** 自动改字、自动认人只在证据充分时做；灰区一律进「待确认」，一次回答要能推广到历史和以后。
2. **一切可撤销、可追溯。** 文本每次改动留修订记录（谁改的、依据什么）；记忆条目带出处；删除留墓碑，不会被下一次自动生成复活。
3. **本机只做便宜的事。** 声纹向量算一次就存下来，之后的重判、聚类、校准全是 numpy；分离模型只在需要时跑；线程数默认压低。
4. **大模型在后台做细活。** 有上下文、有词库、有记忆事实、有 JSON schema、有代码侧的护栏；结果分「改」和「问」两条通道。
5. **每个可选方案都有配置开关和基准数字**（`scripts/bench/`），默认值由实测决定。
6. **老数据平滑升级。** 版本化迁移 + 迁移前自动备份；老接口、老命令、老测试尽量不动。

## 1. 现状证据（2026-11-11，对用户真实数据的只读副本和 MiMo 接口的实测）

数据规模：4 个录音日，1173 个片段（786 完成 / 387 无语音），1882 句（中位 2.5 s / 13 字），ASR 计费 9894 s。

**声音**
- 只登记了本人，3 条同一次录的近讲样本（互相 0.95）；真实环境里本人的声音对这些样本只有 0.45–0.65，
  用已确认片段做的"现场中心"对同一批片段是 0.89（对样本 0.64）。27.5% 的判定落在阈值 ±0.10 内；
  约 226 个未识别簇（约 400 句）很可能就是离麦克风远一点的本人。
- 删样本 / 改阈值后不会重判：41.6% 的自动归属按现在的样本已经不成立。
- 41% 的句子不到 2 s；1 s 的声纹对自己长样本的相似度均值只有 0.62。
- 本机开销：分离（pyannote）占绝大部分，4 线程时每秒音频 0.28 CPU·秒，1 线程 0.106（4 线程只快 1.6 倍，却多花 2.6 倍 CPU）；
  10 s 以下的片段几乎不花钱，95–98% 的分离开销在 10 s 以上的片段；36% 的分离开销花在后来被判"太轻"丢掉的片段上；
  CAM++ 被加载了两份；长片段后内存涨到约 600 MB 不释放。典型一天 1500 CPU·秒（4 线程）。

**转写与校对**
- 每个片段单独校对一次（784 次调用，上下文中位 21 字）；8.7% 的句子被改，人工复核这 175 处修改：
  26% 明确改对、20% 无关紧要、42% 无法验证的猜测、12% 改坏。词表是空的，唯一的"词"是说话人名字，9 处改成名字的有 7 处是错的。
- 同一个人的昵称有 3 种写法，于是记忆里出现 3 个人；一个英文术语在同一场会里被写成 3 种样子。
- 没有任何置信度信号：logprobs 为空、不支持 n-best、重复请求几乎完全一致。
- 一句 9 s 的录音被转出 5109 个字的重复标签（占全库字符的 9.7%），照样入库、建索引、进了日报。
- MiMo 接口实测：
  - ASR 模型接受 **system 消息**（或音频前的一条纯文本 user 消息）作为软热词 / 上下文：6/6 次把同音错词纠正成热词表里的写法，
    错误同音词不会被采纳，16 次带提示的调用没有一处把对的改坏；**但**对非语音音频会把提示原样吐出来当转写。
    `asr_options.hotwords/prompt` 会被静默忽略；text part 和音频放在同一条消息里会 400。
  - `mimo-v2.6-flash / pro` 能直接听音频（omni），可以当"第二个听写员"；但只要把现有转写或候选给它看，它就会被锚定，
    并且永远自称 0.85–0.95 的把握——只能**盲听 + 投票**，不能让它"二选一"。它做声纹开集判断会错，不能替代本机声纹。
  - **不同条件下的两次听写取差异**（带热词 vs 不带、换切分、换听写员）能暴露每一处错词；对的地方约 40 次听写里基本不动。
  - `response_format: json_schema` 可用且会被强制；`json_object` 键名会漂。
  - flash 开 thinking 做校对质量明显好（约 40 s 一批），不开约 5–10 s。
  - ASR 可能返回 HTTP 200 + `finish_reason=content_filter` 和一句英文拒绝语——现在会被当成转写存下来（bug）。

**记忆**
- 记忆 = 原始句子 + FTS + 向量 + 每日日报 + `memory_notes`（只有 todo / decision / person 三种在写）。
  120 条笔记里 36 条是重新生成日报留下的孤儿，约 25% 是换了说法的重复；待办没有状态，决定没有出处，人物每天被覆盖。
- "混合检索"实际等于纯向量（严格 AND 的关键词列表对 23 个问题全部为空）。多视角向量（原文 + "说话人：原文" + 前后各一句）
  融合后 q_ok@10 17→19、@20 19→22；把已有的分块摘要当"情节"加入预取，答案覆盖 42→45 / 46。
- 会话结构：句间间隔中位 2.3 s，97% 小于 5 分钟；按 5 分钟切，每天 9–22 个会话，有内容的会话约 12 分钟 / 26 句 / 400 字。
- 聊天记录不落库；跨层删除不存在（删了句子，日报、笔记、转写原文里都还在）。

**工程**
- `ui/app.py` 1635 行（一个 1270 行的闭包，54 个接口）、`app.js` 1635 行（一个 IIFE）、`cli.py` 1643 行、`pipeline.py` 1184 行。
- 没有 schema 版本、没有事件通道（页面 5 s 轮询，每次遍历音频目录）、没有指标、没有文本编辑路径。
- 已确认的 bug：`reset_stale_processing` 因为时间格式混用（`T` vs 空格）当天永远收不回卡住的片段；停服务时正在转写的片段被标成不可重试的
  failed；30% 编辑比例护栏在短句上一刀切；拆句丢修改记录；三套词表解析器规则不一。

## 2. 架构总览

```
录音 ─► 片段(new)
          │  事件 segment.closed 立即唤醒（不再等 10 分钟）
          ▼
  ┌─ 声音阶段 speech/stage_voice.py ────────────────────────────────────────────┐
  │ 音量预筛 → [短/单人: 快速路径 | 长/多人: 分离] → 声纹向量(存) → 评分层判定       │
  │ scoring.py：锚点样本 + 条件中心(近/远/系统) + 队列归一化 + 时长感知 → 确定 / 可能 / 未知 │
  └────────────────────────────────────────────────────────────────────────────┘
          ▼
  ┌─ 文字阶段 speech/stage_text.py ─────────────────────────────────────────────┐
  │ ASR(带词库热词) → 输出护栏 → 第一遍：词库确定性替换 → 入库(句子 + 修订 + 声纹)     │
  └────────────────────────────────────────────────────────────────────────────┘
          ▼  chain.py（后处理链，后台）
  会话切分 sessions.sync ─► 会话校对 refine（LLM：上下文 + 词库 + 记忆 → 改 / 问）
          ─► 第二次听写（只对被标记的片段，盲听，取差异）─► 待确认 inbox
          ─► 情节摘要 + 记忆提取 episodes ─►（每晚）记忆整理 consolidate ─► 日报 ─►（早上）晨报
          ▲
  用户回答 / 手改 ──► 词库别名规则 ──► 回溯历史 + 以后自动改并标记 + 进热词
```

新模块一览（括号里是负责的工作包，见 §11）：

| 路径 | 作用 |
|---|---|
| `audio_note/__init__.py` | `__version__ = "1.0.0"` (A1) |
| `audio_note/db.py` | 版本化迁移 + 新表 + 新 helper (A1；之后各包只做**追加式**修改) |
| `audio_note/events.py` | 进程内事件总线 (A1) |
| `audio_note/metrics.py` | 分阶段耗时 / CPU / LLM 用量记账 (A1) |
| `audio_note/governor.py` | 线程数、后台优先级、电池状态 (A1) |
| `audio_note/chain.py` | 后处理链调度 (A1) |
| `audio_note/llm/client.py` | json_schema、听音频、ASR 提示、用途标记、流式 (A1) |
| `audio_note/speech/{chunking,channels,markdown,processor,relabel,split,stage_voice,stage_text}.py` | `pipeline.py` 拆分，`pipeline.py` 留作门面 (A4) |
| `audio_note/speech/scoring.py`、`tracks.py` | 纯 numpy 评分层、会话内声音轨迹与持久匿名声音 (B1) |
| `audio_note/speech/{pinyin,lexicon,guard,revisions,refine,hearing}.py`、`audio_note/inbox.py` | 词库、输出护栏、文本修订、会话校对、第二次听写、待确认 (B2) |
| `audio_note/memory/{sessions,episodes,items,consolidate,profile,forget,brief}.py` | 会话 / 情节 / 记忆条目 / 整理 / 关于我 / 遗忘 / 晨报 (B3) |
| `audio_note/ui/{deps,errors,serializers,settings_schema,static_site}.py`、`ui/routers/*.py` | `app.py` 拆分 (A2)，新路由文件归各功能包 |
| `audio_note/ui/static/js/**`、`css/**` | 前端 ES 模块 (F1，之后 FA/FB/FC) |
| `audio_note/cli/` | `cli.py` 拆成包 (A3) |
| `scripts/bench/` | 评测工具：固定剧本、合成、CER、声纹、成本 (E) |

## 3. 数据模型

### 3.1 迁移机制（A1）

- 保留 `SCHEMA`（`CREATE TABLE IF NOT EXISTS`）和 `MIGRATION_COLUMNS`（加列）两套机制，新表、新列照旧加进去。
- 新增 `SCHEMA_VERSION = 2` 和 `MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]]`（版本号、说明、数据步骤），
  用 `PRAGMA user_version` 记录。打开数据库时：`executescript(SCHEMA)` → `_migrate`（加列）→ 依次执行版本号大于当前值的数据步骤
  （每步 `BEGIN IMMEDIATE` 内重新检查版本，防并发）→ 写 `user_version`。
- **数据步骤执行前**，如果库里已有数据（`segments` 非空），先用 `Connection.backup()` 备份到
  `<data_dir>/backups/audio_note-v<旧版本>-<YYYYmmdd-HHMMSS>.sqlite3`，只留最近 3 份。备份失败则不迁移并抛出清楚的错误。
- 库的 `user_version` 比代码新：拒绝打开，报"数据库来自更新的版本"。
- `DB.schema_version() -> int`。`SCHEMA` 里不能给迁移加的列建索引——这类索引放进数据步骤。
- v2 的数据步骤：
  1. `segments.updated_at` 统一成 `now_iso()` 的 `T` 格式，并修掉 `reset_stale_processing` / `unfinished_segments_for_day` 的格式混用。
  2. `memory_notes` → `memory_items`（kind 原样；`source='legacy'`；`fingerprint` 见 §7.2；`evidence` 取 `source_utterance_ids`）。
     `memory_notes` 表保留但不再写入。
  3. 词表文件 + 说话人名字 → `lexicon`（`source='glossary'` / `'speaker'`；说话人条目 `kind='person'` 并填 `speaker_id`）。
     这一步由 `lexicon.sync_sources(db, cfg)` 在运行时幂等完成（迁移里拿不到 cfg），不是迁移步骤。
  4. 给新列建索引（`utterances.session_id`、`cluster_embeddings.voice_id`）。

### 3.2 新表（DDL，写进 `SCHEMA`）

```sql
-- 词库：术语 / 人名 / 项目……（规范写法）及其别名（误识别写法、别称）
CREATE TABLE IF NOT EXISTS lexicon (
    id INTEGER PRIMARY KEY,
    term TEXT NOT NULL UNIQUE,            -- 规范写法
    kind TEXT NOT NULL DEFAULT 'term',    -- person | term | project | org | place | product | other
    note TEXT,                            -- 一句话说明（给模型和人看）
    pinyin TEXT,                          -- 无声调拼音键，音节用空格分隔；拉丁词为小写原文
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    source TEXT NOT NULL DEFAULT 'user',  -- user | answer | edit | glossary | speaker | auto
    status TEXT NOT NULL DEFAULT 'active',-- active | candidate（自动发现、未确认） | disabled
    hotword INTEGER NOT NULL DEFAULT 1,   -- 是否进入 ASR 热词
    hits INTEGER NOT NULL DEFAULT 0,
    last_hit_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS lexicon_aliases (
    id INTEGER PRIMARY KEY,
    lexicon_id INTEGER NOT NULL REFERENCES lexicon(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'mishear', -- mishear（误识别，可改写成 term） | aka（别称：不改写，只用于检索和实体归并）
    scope TEXT NOT NULL DEFAULT 'auto',   -- auto（出现即改） | hint（只作为提示交给模型判断）
    pinyin TEXT,
    source TEXT NOT NULL DEFAULT 'user',  -- user | answer | edit | llm
    hits INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(alias, lexicon_id)
);
CREATE INDEX IF NOT EXISTS idx_lex_alias ON lexicon_aliases(alias);

-- 文本修订：utterances.text 的每一次改动
CREATE TABLE IF NOT EXISTS utterance_revisions (
    id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    old_text TEXT NOT NULL,
    new_text TEXT NOT NULL,
    source TEXT NOT NULL,                 -- lexicon | llm | refine | answer | user | guard | undo
    detail TEXT,                          -- JSON：{"edits":[{"from","to","reason","alias_id"?,"lexicon_id"?}],"inbox_id"?,"model"?}
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    undone_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rev_utt ON utterance_revisions(utterance_id);

-- 待确认：系统想问用户的事
CREATE TABLE IF NOT EXISTS inbox_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                   -- term | voice | hint | conflict | rule
    status TEXT NOT NULL DEFAULT 'open',  -- open | answered | dismissed | snoozed | expired
    dedupe_key TEXT,                      -- 同一 kind 下，open 状态唯一
    priority REAL NOT NULL DEFAULT 0,
    title TEXT NOT NULL,                  -- 一句话问题
    payload TEXT NOT NULL DEFAULT '{}',   -- JSON，形状见 §6.6
    utterance_id INTEGER,                 -- 主要证据（不设外键：句子可能被拆 / 删）
    session_id INTEGER,
    speaker_id INTEGER,
    lexicon_id INTEGER,
    occurrences INTEGER NOT NULL DEFAULT 1,
    answer TEXT,                          -- JSON
    snooze_until TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    answered_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_open_key ON inbox_items(kind, dedupe_key)
    WHERE status='open' AND dedupe_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_inbox_status ON inbox_items(status, priority);

-- 每句话的声纹向量：算一次，之后重判全靠它
CREATE TABLE IF NOT EXISTS utterance_voice (
    utterance_id INTEGER PRIMARY KEY REFERENCES utterances(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,              -- float32，L2 归一化
    speech_s REAL
);

-- 每个说话人按录音条件的自适应中心
CREATE TABLE IF NOT EXISTS speaker_centroids (
    id INTEGER PRIMARY KEY,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,              -- near | far | system | all
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT,
    n INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(speaker_id, condition)
);

-- 持久的匿名声音（跨片段的同一个人 / 同一个节目），可命名、可忽略
CREATE TABLE IF NOT EXISTS voices (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'unknown', -- unknown | named | ignored
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    label TEXT,                             -- 用户备注（"电视"、"楼下"）
    centroid BLOB,
    dim INTEGER,
    model TEXT,
    n_clusters INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    first_seen TEXT,
    last_seen TEXT,
    days INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- 会话（情节）：按时间间隔切出来的一段连续对话
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,                    -- 开始那天
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    utterance_count INTEGER NOT NULL DEFAULT 0,
    speech_s REAL NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',  -- open | closed
    kind TEXT,                            -- conversation | meeting | monologue | media | noise
    title TEXT,
    summary TEXT,
    topics TEXT,                          -- JSON list
    participants TEXT,                    -- JSON list
    raw_hash TEXT,                        -- 句子 id + text_raw 的指纹（组成变了才重新校对）
    refined_hash TEXT,
    refined_at TEXT,
    text_hash TEXT,                       -- 句子 id + text 的指纹（文字变了才重新摘要）
    digested_hash TEXT,
    digested_at TEXT,
    meta TEXT,                            -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_day ON sessions(day);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- 记忆条目：有出处、有状态、可编辑（取代 memory_notes）
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                   -- fact | todo | decision | person | preference | project | event | topic | note
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    subject TEXT,                         -- 关于谁 / 什么（人名或词库写法）
    lexicon_id INTEGER REFERENCES lexicon(id) ON DELETE SET NULL,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'active',-- active | done | superseded | archived | rejected（墓碑）
    source TEXT NOT NULL DEFAULT 'auto',  -- auto | digest | user | chat | voice | legacy
    confidence REAL,
    importance REAL NOT NULL DEFAULT 0.5,
    pinned INTEGER NOT NULL DEFAULT 0,
    day TEXT,
    valid_from TEXT,
    valid_to TEXT,
    due TEXT,                             -- 待办截止日 YYYY-MM-DD
    owner TEXT,                           -- 待办负责人
    session_id INTEGER,
    evidence TEXT,                        -- JSON list of utterance ids
    superseded_by INTEGER,
    fingerprint TEXT,                     -- kind + 规范化标题的指纹：去重 / 墓碑
    hits INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_items_kind ON memory_items(kind, status);
CREATE INDEX IF NOT EXISTS idx_items_subject ON memory_items(subject);
CREATE INDEX IF NOT EXISTS idx_items_fp ON memory_items(fingerprint);
CREATE TABLE IF NOT EXISTS memory_item_tokens (
    item_id INTEGER PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    text_seg TEXT NOT NULL
);
-- memory_items_fts：外部内容 FTS5，建法和触发器同 utterances_fts（utt_tok_* 三个触发器照抄，改名 item_tok_*）

-- 通用向量表：记忆条目、会话摘要、句子的额外视角
CREATE TABLE IF NOT EXISTS memory_vectors (
    kind TEXT NOT NULL,                   -- item | session | utt
    ref_id INTEGER NOT NULL,
    view TEXT NOT NULL DEFAULT '',        -- utt 的视角：speaker / window；其余为空串
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,                    -- float32，L2 归一化
    PRIMARY KEY (kind, ref_id, view)
);

-- 聊天记录
CREATE TABLE IF NOT EXISTS chat_conversations (
    id INTEGER PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                   -- user | assistant
    content TEXT NOT NULL,
    meta TEXT,                            -- JSON：citations / stats / tools
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- 用量记账（按天聚合，表很小）
CREATE TABLE IF NOT EXISTS usage_local (
    day TEXT NOT NULL,
    stage TEXT NOT NULL,                  -- vad | gate | diarize | embed | identify | rescore | text_embed | ...
    n INTEGER NOT NULL DEFAULT 0,
    audio_s REAL NOT NULL DEFAULT 0,
    wall_s REAL NOT NULL DEFAULT 0,
    cpu_s REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, stage)
);
CREATE TABLE IF NOT EXISTS usage_llm (
    day TEXT NOT NULL,
    purpose TEXT NOT NULL,                -- asr | listen | correct | refine | episode | consolidate | daily | chat | relations | ...
    model TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    audio_s REAL NOT NULL DEFAULT 0,
    wall_s REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, purpose, model)
);
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'ui',    -- ui | scheduler | cli | chain
    status TEXT NOT NULL,                 -- running | done | failed
    started_at TEXT NOT NULL,
    finished_at TEXT,
    result TEXT,
    error TEXT
);
```

### 3.3 新列（`MIGRATION_COLUMNS`）

| 表.列 | 类型 | 含义 |
|---|---|---|
| `utterances.session_id` | INTEGER | 所属会话 |
| `utterances.text_source` | TEXT | 当前文字最后由谁决定：NULL=ASR 原样 / `lexicon` / `llm` / `refine` / `answer` / `user` / `guard` |
| `utterances.edited_at` | TEXT | 最后一次修订时间 |
| `utterances.flags` | TEXT | JSON：`{"guard":"loop"}`、`{"media":true}` 等 |
| `utterances.speaker_guess_id` | INTEGER | "可能是 X"：证据不够自动认，但值得问 |
| `utterances.speaker_guess_score` | REAL | 猜测的（归一化后）分数 |
| `utterances.speaker_meta` | TEXT | JSON：判定依据（方法、原始分、归一化分、条件） |
| `speaker_embeddings.model` / `speaker_negatives.model` / `cluster_embeddings.model` | TEXT | 声纹模型标识，NULL = 默认 CAM++ zh-cn |
| `cluster_embeddings.voice_id` | INTEGER | 所属持久匿名声音 |
| `cluster_embeddings.loudness_db` / `system_share` | REAL | 这个簇的录音条件（自适应中心分条件用） |
| `segments.realtime` | INTEGER | 1 = 由"录完即处理"通道处理（统计用） |

## 4. 核心契约（A1）

### 4.1 配置（`config.py` DEFAULTS + `config.example.yaml`，两边必须一致，有测试卡着）

A1 先加下面这些；之后各功能包可以在**自己的分区**追加键（同时改 `config.example.yaml` 和 `ui/settings_schema.py` 的白名单）。

```yaml
performance:
  mode: balanced              # eco | balanced | fast：本机模型用多少 CPU
  threads: 0                  # 本机模型线程数；0 = 跟随 mode（eco 1 / balanced 2 / fast 4）
  background_priority: true   # 把进程设为"低于正常"优先级，录音线程提高，不抢前台
  defer_on_battery: false     # 用电池时推迟本机重活（分离、重判），转写等到插电再做
pipeline:
  realtime: true              # 片段一录完就处理（process_interval_min 只做兜底）
mimo:
  listen_model: mimo-v2.6-flash   # 能听音频的对话模型（第二次听写）
  asr_prompt: true                # 把词库热词作为 system 消息交给 ASR
  asr_prompt_max_terms: 50
ui:
  events: true                # 服务端事件推送（SSE）；关掉则页面轮询
```
`speakers.num_threads` 保留，默认改成 `0`（= 跟随 `performance`）。

### 4.2 `MimoClient`（`llm/client.py`）

保持现有方法签名可用（测试里的假客户端只实现了老参数，调用方传新参数前要用签名探测或捕获 `TypeError` 回退）。新增：

```python
def chat(self, messages, *, model=None, temperature=None, max_tokens=4096, json_mode=False, json_schema=None,
         thinking=False, tools=None, tool_choice=None, purpose=None): ...
    # json_schema: {"name": str, "schema": dict} -> response_format={"type":"json_schema","json_schema":{..., "strict": True}}
    # 服务端 400 拒绝 schema 时自动退回 json_object 再试一次
def json(self, system, user, *, schema=None, schema_name="result", **kw) -> Any
def chat_stream(self, messages, *, model=None, temperature=None, max_tokens=4096, thinking=False, purpose=None) -> Iterator[str]
def transcribe_bytes(self, wav_or_mp3, mime="audio/wav", language="auto", model=None, *, prompt=None, context=None) -> tuple[str, float]
    # prompt -> system 消息；context -> 音频之前的一条纯文本 user 消息；都为 None 时请求与现在逐字节相同
    # finish_reason == "content_filter" -> raise ContentFiltered（修 bug）
def listen(self, clips: list[bytes], text: str, *, model=None, json_schema=None, thinking=False, max_tokens=600,
           purpose="listen") -> str
    # 一条 user 消息：若干 input_audio part + 一个 text part，发给 mimo.listen_model
def scoped(self, purpose: str) -> "MimoClient"     # 同一个底层客户端 / 限速器，只是给用量打上用途标签
on_usage: Callable[[dict], None] | None           # 每次调用结束回调 {purpose, model, ok, prompt_tokens, completion_tokens, audio_s, wall_s}
```
限速器按 `(base_url, api_key)` 进程内共享（现在每个实例各有一个）。`metrics.install(client, db)` 把 `on_usage` 接到 `usage_llm`。

### 4.3 事件总线（`events.py`）

```python
class EventBus:
    def publish(self, type: str, **data) -> int           # 线程安全；返回事件 id；保留最近 500 条
    def subscribe(self, since: int | None = None) -> "Subscription"   # .get(timeout) -> dict | None；.close()
    def close(self) -> None                               # 唤醒并结束所有订阅（UIServer.stop 先调它）
```
`RuntimeHolder.events`（切换档案时沿用同一个总线）；`Runtime.events`；没有总线的地方用 `events.NULL`（空实现）。

事件类型（`{"id", "type", "ts", ...}`）：`recording {on, speaking?}`、`segment {id, status}`、`utterances {day, segment_id, ids}`、
`utterance {id, change: "text"|"speaker"|"deleted"}`、`job {name, state: "queued"|"running"|"done"|"failed", result?, error?}`、
`session {id, change}`、`inbox {open, kind?}`、`memory {change}`、`daily {day}`、`speakers {change}`、`lexicon {change}`、`config {}`、`profile {name}`。

`GET /api/events`（`ui/routers/events.py`，A1 写）：`text/event-stream`，先发一条 `hello`（含 `status` 快照的精简版和 `inbox.open`），
15 s 一个注释心跳，`?once=1` 发完 hello 就结束（测试用）。`UIServer.stop()` 先 `events.close()`，并给 uvicorn 配 `timeout_graceful_shutdown`。
`Runtime.status()` 里遍历音频目录的结果缓存 60 s，并带上 `schema_version`、`version`、`inbox_open`、`min_speech_dbfs`、`unknown_label`、
`system_turn_share`。

### 4.4 指标（`metrics.py`）

```python
@contextmanager
def stage(db, name: str, audio_s: float = 0.0, n: int = 1): ...   # 记 wall（perf_counter）与进程 CPU（process_time）到 usage_local（按天累加）
def add_llm(db, rec: dict) -> None                                 # 累加到 usage_llm
def install(client, db) -> None
def summary(db, days: int = 7) -> dict   # {"local": [{day, stage, n, audio_s, wall_s, cpu_s}], "llm": [...], "totals": {...}}
```
`GET /api/metrics?days=7`（A1，放 `ui/routers/events.py` 或 `system.py`）。写库失败只记日志，绝不影响主流程。

### 4.5 资源管家（`governor.py`）

```python
def threads(cfg) -> int                 # performance.threads，0 则按 mode；speakers.num_threads 非 0 时优先
def apply_process_priority(cfg) -> bool # Windows：进程 BELOW_NORMAL；其他平台 no-op
def boost_current_thread() -> None      # 录音线程调用：线程优先级 +2（在低优先级进程里保持正常水平）
def on_battery() -> bool | None         # Windows GetSystemPowerStatus；未知返回 None
def defer_reason(cfg) -> str | None     # 现在不该做本机重活的原因（中文短语）或 None
```
全部 ctypes，失败静默。调度器在跑本机重活前问 `defer_reason`；延后时 `status()` 里说明原因。

### 4.6 后处理链（`chain.py`）与任务

```python
STEPS = (
    ("sessions",   "audio_note.memory.sessions:sync"),
    ("refine",     "audio_note.speech.refine:run_pending"),
    ("episodes",   "audio_note.memory.episodes:run_pending"),
)
def run_chain(cfg, db, client_factory, *, stop_event=None, events=None, only=None, limit=None) -> dict
```
每个步骤函数签名固定为 `fn(cfg, db, client, *, stop_event=None, events=None, limit=None) -> dict`
（`client` 可能为 None：没配 key 时步骤自己决定跳过哪些部分）。模块还不存在（ImportError）时该步骤静默跳过；
单个步骤抛异常只记日志，不影响后面的步骤。调度器在每次 `process` 之后、以及每个 tick 发现有"已结束未处理"的会话时跑链；
`daily` 之前先跑一遍链；`daily` 之后跑 `audio_note.memory.consolidate:run`（同样的签名）；
每天 `analysis.brief_time`（B3 加的键）跑 `audio_note.memory.brief:run`。

`runtime.JOBS` 扩充（中文标签）：`chain`（整理新内容）、`refine`（校对会话）、`episodes`（会话摘要）、`consolidate`（整理记忆）、
`recorrect`（按词库重校历史 → `audio_note.speech.lexicon:job_recorrect(rt, **kw)`）、`rescore`（重判说话人 →
`audio_note.speech.scoring:job_rescore(rt, **kw)`）、`brief`（晨报）、`maintenance`（数据库维护）。
任务注册改成表：`JOB_TARGETS: dict[str, str]`（`"模块:函数"`，函数签名 `fn(rt, **kwargs) -> dict`），老的 `_job_<name>` 方法优先。
任务状态变化发 `job` 事件并写 `job_runs`。**录完即处理**：录音线程在片段落盘后 `events.publish("segment", ...)` 并 set 调度器的唤醒 Event，
调度器立即跑一次 `process`（`pipeline.realtime` 为真时；`min_segment_age_s` 对实时通道不生效）。

顺手修：`reset_stale_processing` 的时间格式；停服务时 `RuntimeError('stopped')` 的片段应回到 `new` 而不是 `failed`；
`job_running()` 读不存在的 `JobState.label`；`pipeline.max_failed_retries` 进 DEFAULTS。

## 5. 声音 v5（B1）

### 5.1 评分层 `speech/scoring.py`（纯 numpy，不加载模型）

```python
@dataclass
class Decision:
    speaker_id: int | None        # 确定是谁（自动归属）
    name: str | None
    score: float | None           # 用于展示的原始余弦（对胜出者）
    norm: float | None            # 归一化后的分数
    guess_id: int | None          # "可能是谁"（进待确认 / 核对），与 speaker_id 互斥
    guess_score: float | None
    method: str                   # anchor | centroid:<condition> | track | smooth | veto | none
    runner_up: float | None

class Gallery:
    @classmethod
    def load(cls, db, cfg) -> "Gallery"        # 样本（含 origin）、条件中心、负样本、队列；带版本号（样本集变了就变）
    def score(self, emb, *, speech_s: float | None = None, loudness_db: float | None = None,
              system_share: float | None = None, exclude: frozenset[int] = frozenset()) -> Decision
def condition_of(loudness_db, system_share, cfg) -> str       # near | far | system
def adapt(db, cfg, *, dry_run=False) -> dict                  # 用可信证据重建 speaker_centroids
def rescore(db, cfg, *, day=None, segment_ids=None, dry_run=False) -> dict   # 用存下来的向量重判历史（不读音频、不跑模型）
def job_rescore(rt, **kw) -> dict
```

规则（默认值由 `scripts/bench/bench_speakers.py` 的数字决定，下面是起点）：

1. **打分**：候选人 S 的分数 = max(对"核心样本"的加权分, 对相同条件中心的余弦)。核心样本 = 通过一致性检查的样本
   （可信来源 mic/file 优先）；**不一致的离群样本不再参与取最大**（堵住"一条样本单独认出几百句"的侧门，
   它只作为"远条件"的候选种子，等用户确认）。
2. **条件中心**：`adapt()` 从三类证据里按条件（近 / 远 / 系统）攒中心：用户确认过或手标的句子；
   高置信的自动判定（归一化分高、时长 ≥ 3 s）；用户命名的声音组。防漂移：新中心与可信锚点的余弦必须 ≥ `speakers.adapt_min_anchor_cos`，
   且每个条件至少 `speakers.adapt_min_clusters` 个簇；负样本和否决过的句子永远不进中心。
3. **队列归一化（AS-norm）**：冒名者队列取自已存的簇向量——系统声音为主的簇、已命名为别人的声音、被否决的簇、
   离所有人都远且 ≥ 3 s 的未知簇；与目标自适应中心接近的排除在外。队列不足 `speakers.cohort_min`（30）时退回原始余弦 + 老阈值。
4. **时长感知**：≥ 2 s 的句子自己做硬判定；1–2 s 作为软证据，跟随所在簇 / 轨迹；< 1 s 只跟随。首次判定和重判用**同一套规则**。
5. **三态**：确定（自动归属）/ 可能（`speaker_guess_*`，不写 `speaker_id`，进核对队列，界面显示"可能是 X"）/ 未知。
   否决、负样本、系统声音排除照旧优先。
6. **时间平滑**（`tracks.py`）：同一会话内相邻片段的簇向量先连成轨迹再整体判定并回传；孤立的 A-B-A 短句翻转抹平。
7. **自动重判**：样本 / 负样本 / 阈值 / 中心一变，`rescore` 用存下来的向量秒级重判，`speaker_source='user'` 的永远不动。

`SpeakerEngine.match(...)` 保留现签名和行为（薄封装，154 个现有测试钉着）。

### 5.2 开销

- `speakers.num_threads` 默认跟随 `performance`（balanced = 2，eco = 1）。
- **音量预筛**：分离之前先按 VAD + 响度看整段是不是都低于 `audio.min_speech_dbfs`，是就直接 `no_speech`，不跑分离。
- **快速路径**：短片段（< `speakers.fast_path_max_s`，默认 10 s，实测几乎不花钱的区间保持走分离也可以——由基准决定）
  或经滑窗声纹一致性检查判定为单人的长片段：不跑 pyannote，VAD 分句 + 每句一个 CAM++ 向量。
- CAM++ 只加载一份；长片段后释放 ONNX arena（或限制 arena）；分离可被 `stop_event` 中断。
- 每句的声纹向量写进 `utterance_voice`（首次处理、拆句时写；历史由 `voices` 任务回填，回填只对还有音频的片段）。
- 记账：`metrics.stage(db, "diarize"|"embed"|"identify"|"gate", audio_s=...)`。

### 5.3 持久匿名声音

簇向量增量归入 `voices`（最近中心 ≥ 阈值则并入，否则新建；夜间合并一遍）。`voices.status='ignored'`（电视、路人）之后不再出现在
"未识别的声音"里，也不再进待确认。反复出现的未知声音（≥ 2 天且 ≥ `speakers.voice_prompt_min_s` 秒）生成一条 `inbox_items(kind='voice')`
（"这个声音出现了 3 天，是谁？"）。现有 `group_unknown_voices` / `name_voice` 的接口与测试保持可用。

### 5.4 评测 `scripts/bench/bench_speakers.py`

用存下来的向量回放各种评分方案（老规则 / 只换中心 / 加归一化 / 加时长规则 / 加平滑 / 第二模型级联），
在合成集（有真值，含远场 / 噪声 / 重叠变体，见 §10）上给出 FAR / FRR / 准确率（按时长和响度分桶）与每秒音频的 CPU 开销，
在真实数据副本上给出代理指标（手标 / 否决 / 系统声音这几类有把握的证据上的错误数、"确定 / 可能 / 未知"的比例变化）。默认值按结果定。

## 6. 转写 v2（B2）

### 6.1 拼音与词库（`speech/pinyin.py`、`speech/lexicon.py`）

```python
# pinyin.py（pypinyin 懒加载；缺库时退化为逐字比较，功能降级不报错）
def key(text: str) -> str                      # "灰度发布" -> "hui du fa bu"；拉丁 / 数字原样小写
def similarity(a: str, b: str) -> float        # 0..1，音节级编辑距离，声母韵母近音（n/l、en/eng、z/zh…）算半步
def close(a: str, b: str, min_sim: float = 0.75) -> bool

# lexicon.py
@dataclass
class Term: id, term, kind, note, aliases: list[Alias], hotword, status, source, hits, speaker_id
class Lexicon:
    @classmethod
    def load(cls, db) -> "Lexicon"             # 进程内按 db.path 缓存；写操作后失效
    def find_in(self, text: str) -> list[Hit]  # 文本里出现的规范写法 / 别名（位置、term、alias）
    def near(self, text: str, limit=20) -> list[Term]   # 拼音上接近文本中某一段的词条（给校对提示用）
    def apply(self, text: str) -> tuple[str, list[dict]]   # 确定性替换：scope='auto' 的 mishear 别名 -> 规范写法；返回 edits
    def hotwords(self, limit: int, context: str = "") -> list[str]   # 热词：说话人 / 用户确认的优先，其次最近命中
    def prompt_rows(self, terms: list[Term]) -> str   # "写法｜说明｜常见误识别" 表格行
def upsert_term(db, term, *, kind="term", note=None, source="user", status="active", speaker_id=None) -> int
def add_alias(db, lexicon_id, alias, *, kind="mishear", scope=None, source="user") -> int
    # scope=None 时自动判断：别名是常用词（jieba 词频高）或长度为 1、或同一别名已指向别的词条 -> 'hint'，否则 'auto'
def sync_sources(db, cfg) -> dict              # 幂等：glossary.txt 的词与说话人名字 -> lexicon（source glossary / speaker）
def apply_alias_to_history(db, cfg, alias_id, *, dry_run=False, limit=None) -> dict   # {matches:[{utterance_id, before, after}], applied}
def job_recorrect(rt, *, day=None, lexicon_id=None, alias_id=None) -> dict
def export_glossary(db) -> str                 # 一行一个规范写法（兼容老的 GET /api/glossary）
```
词库是**唯一**词表来源：jieba 注册、校对提示、ASR 热词、检索时的别名展开都从它来。老的三套 `load_glossary` 合并成一个
（保留各自的函数名作薄封装，行内 ` #` 注释统一剥掉），`glossary.txt` 只作导入 / 导出。

### 6.2 ASR 热词与输出护栏（`speech/asr.py`、`speech/guard.py`）

- `ASR.transcribe(audio, *, prompt: str | None = None) -> ASRResult`；`ASRResult` 增加 `flags: list[str]`（默认空）。
  `transcribe_chunks(asr, audio, chunks, concurrency, stop_event=None, prompt_for=None)`：`prompt_for(chunk) -> str | None`。
  假 ASR 只有 `transcribe(audio)` 时要回退（签名探测）。
- 提示词：`"可能出现的专有名词：" + "、".join(lexicon.hotwords(cfg.mimo.asr_prompt_max_terms, context))`，词库为空则不带提示。
- `guard.check(text, audio_s, *, prompt=None) -> GuardResult(ok, reason, cleaned)`：
  - `echo`：输出以提示开头 / 被提示覆盖 ≥ 60% / 连续出现 ≥ 3 个热词表里的词且顺序一致 → 不带提示重试一次；
  - `loop`：同一 n-gram 连续重复 ≥ 5 次或字符 / 秒超过 `correction.max_chars_per_s`（默认 15）→ 压缩重复、仍超标则丢弃并记 flag；
  - `refusal`：英文拒绝语 / `content_filter` → 丢弃；
  - `filler`：语音不足 1 s 且输出只有单个语气词 → 丢弃。
  被丢弃 / 清洗的记到 `utterances.flags`（清洗的同时留一条 `source='guard'` 的修订）。提供 `guard.scan_existing(db, cfg, dry_run)`
  清理历史里的同类问题（走修订，可撤销）。

### 6.3 文本修订（`speech/revisions.py`）

```python
def update_text(db, cfg, utterance_id, new_text, *, source, detail=None, events=None) -> dict | None
    # 事务内：写修订 -> 更新 utterances.text / text_source / edited_at -> 更新 utterance_tokens（FTS 触发器跟着走）
    # -> 删掉该句的 utterance_vectors / memory_vectors(kind='utt')（下次索引时重建）
    # -> 标记当天 Markdown 需要重写、会话 text_hash 变化 -> 发 utterance 事件。文本没变返回 None。
def undo(db, cfg, utterance_id, revision_id=None) -> dict     # 撤回最后一次（或指定的）修订：写一条 source='undo' 的反向修订
def history(db, utterance_id) -> list[dict]
def marks(db, rows) -> dict[int, list[dict]]
    # 给一批句子算界面标记：[{start, end, kind: "fixed"|"uncertain", from?, source?, inbox_id?, candidates?}]
    # fixed：最近修订里 edits 的 to 在当前文本中的位置（找不到就按 text_raw 与 text 的字符 diff）；uncertain：open 的 term 待确认
def learn_from_edit(db, cfg, utterance_id, old_text, new_text) -> dict | None
    # 用户手改：字符 diff 出 (错写 -> 正写) 的最小片段；像术语 / 人名（或用户勾了"记住"）就建词条 + 别名，
    # 返回 {"lexicon_id", "alias_id", "term", "alias", "scope", "other_matches": n} 供界面问"其他 N 处也改吗"
```

### 6.4 第一遍（片段级，随处理立即做）

`correction.mode: session | segment | off`（默认 `session`）。`Processor` 仍然调用 `corrector.correct(items, glossary)` 这个缝
（测试注入的假校对器照常生效）。默认校对器：
- `session` 模式：`LexiconCorrector`——只做 `Lexicon.apply` 的确定性替换，不调用 LLM；
- `segment` 模式：老的 `Corrector`（LLM，逐片段），并接上新的护栏；
- 入库时每处应用的修改写修订（`source='lexicon'` 或 `'llm'`），`text_source` 相应设置。

### 6.5 会话校对（`speech/refine.py`，后台）

`run_pending(cfg, db, client, *, stop_event=None, events=None, limit=None) -> dict`：对 `status='closed'` 且
`refined_hash != raw_hash` 的会话逐个 `refine_session(...)`。没有 client（没配 key）时直接跳过。

一次校对的输入：
- 会话全文（`[编号] 时间 说话人：文本`；超过 `correction.refine_max_chars`（默认 3000）切块，块间带前 3 句作只读上下文）；
- 词库表格行：出现在文本里的词条 + 拼音上接近的词条（`Lexicon.near`）+ 说话人；
- 记忆事实：`memory.profile.core_block(db, cfg)`（"关于我"，没有该模块时为空）+ 与文中实体相关的若干条记忆条目
  （`memory.items.related_facts(db, cfg, text, limit)`，没有时为空）；
- 输出 schema（`json_schema` 强制）：
  ```json
  {"edits":[{"i":0,"from":"","to":"","reason":"","basis":"lexicon|memory|context|homophone"}],
   "uncertain":[{"i":0,"span":"","candidates":[""],"why":"","kind":"name|term|number|other"}],
   "speaker_hints":[{"i":0,"name":"","why":""}]}
  ```
- 模型 / thinking：`correction.refine_model`（空 = `mimo.chat_model`）、`correction.refine_thinking`（默认 true）。

代码侧护栏（`apply_edits` 升级，老行为在 `segment` 模式下保持兼容）：
1. `from` 必须是原文子串；数字串不许变（现有规则）；**字母数字型号**（`[A-Za-z]+\d+` 之类）不许改——这类修改转成一条 `uncertain`；
2. 非词库依据的修改必须**拼音接近**（`pinyin.close`），否则转成 `uncertain`（挡住"头凤→排期"这类自信的错改）；
   依据是词库 / 记忆的修改不受 30% 比例限制；其余按**每处修改**计预算，不再整句一刀切；
3. 不许把无关的词改成说话人名字（改成人名必须拼音接近）；
4. 中英文之间的空格归一化（`星河 App` → `星河App`，按原文风格）；
5. `uncertain.span` 必须是原文里 ≤ 12 个字的子串、不与已应用的修改重叠；每个会话最多 `correction.max_questions_per_session`（默认 6）。
应用的修改走 `revisions.update_text(source='refine')`。被护栏拦下的修改不丢——转成 `uncertain` 候选。

### 6.6 第二次听写与待确认（`speech/hearing.py`、`audio_note/inbox.py`）

**差异即不确定**：对每条 `uncertain`（以及被护栏拦下的修改），在音频还在的前提下取该句音频做一次"条件不同"的听写：
`correction.second_hearing: listen | asr | off`（默认 `listen`）——`listen` 用 `client.listen([clip], "请逐字转写这段录音，只输出转写文本。")`
（**绝不**把现有转写或候选给它看）；`asr` 用不带热词、前后各多带 0.5 s 的普通 ASR。把第二次听写与现有文本做字符对齐：
- 该片段两次一致 → 证据增强：若 LLM 给的候选只有一个且拼音接近，可降级为不问（保持原文）；
- 不一致 → 差异片段加入候选（去重，最多 4 个），`payload.heard` 记录两种听法。
每天第二次听写的次数上限 `correction.second_hearing_daily_max`（默认 60）。

**inbox.py**（待确认的存取与回答，B2 负责；其他包通过它建条目）：

```python
def add(db, kind, *, title, payload, dedupe_key=None, priority=0.0, utterance_id=None, session_id=None,
        speaker_id=None, lexicon_id=None, events=None) -> int
    # open 状态下同 (kind, dedupe_key) 已存在：occurrences += 1，把新例子并进 payload.examples（最多 5 个），抬高 priority，返回原 id
    # 该 key 曾被"以后别问"（dismissed + payload.never）：不再创建，返回 0
def list_items(db, *, status="open", kind=None, limit=50, offset=0) -> list[dict]
def counts(db) -> dict                           # {"open": n, "by_kind": {...}}
def answer(db, cfg, item_id, *, choice=None, text=None, as_correct=False, apply_history=True, events=None) -> dict
def dismiss(db, item_id, *, never=False, events=None) -> dict
def snooze(db, item_id, days=1) -> dict
def register_handler(kind, fn)                   # 各包注册自己 kind 的回答处理：fn(db, cfg, item, answer) -> dict
```
`kind='term'` 的 payload：
`{"span","sentence","utterance_id","start","end","candidates":[...],"why","source":"llm|diff|guard|conflict","heard":{"asr":"","second":""},"examples":[{"utterance_id","sentence"}],"term_kind":"name|term|number|other"}`；
`dedupe_key = pinyin.key(span)`（同音不同字的写法合并成一个问题）。优先级 = 出现次数 + 是否出现在待办 / 决定 / 人物里 + 新近度。
每天新建的 term 问题上限 `inbox.daily_max`（默认 12）；超出的留给下一天（按优先级）。

回答 `term`：
- 选候选 / 自己输入 Y（≠ span X）：`upsert_term(Y)` + `add_alias(Y, X)`；`apply_history` 时对历史里的 X 做替换
  （`scope='auto'` 才批量替换，`hint` 只改这个问题的例句），全部走修订（`source='answer'`，`detail.inbox_id`）；Y 进热词。
  返回 `{"lexicon_id","alias_id","scope","changed":[utterance_id...]}`。
- `as_correct`（原文没错）：`upsert_term(X, status='active')`，以后不再问这个写法。
- 其余 kind：`voice`（B1 注册：命名 / 忽略这个声音）、`hint`（B1 注册：按内容猜的说话人，确认 / 否定）、
  `conflict`（B3 注册：记忆冲突，选哪个为准）、`rule`（B2：手改之后"其他 N 处也这样改吗"）。

### 6.7 评测 `scripts/bench/bench_cer.py`

固定剧本（真值）上测：ASR 原样 / +热词 / +词库第一遍 / +会话校对（thinking 开 / 关）/ 老的逐片段校对，
报告 CER、专有名词错误率、改对 / 改坏的处数、问题数量与命中率（问的地方是不是真错了）、调用次数与耗时。
再模拟"回答问题 → 第二天同样的词"的闭环：回答第一天的问题后，第二天的专有名词错误率应明显下降。默认值按结果定。

## 7. 记忆 v2（B3）

### 7.1 会话与情节（`memory/sessions.py`、`memory/episodes.py`）

```python
# sessions.py
def sync(cfg, db, client=None, *, stop_event=None, events=None, limit=None) -> dict
    # 幂等：给 session_id 为空的句子归会话（与某会话首尾相距 <= memory.session_gap_min（默认 5 分钟）则并入，
    # 跨接两个会话则合并），更新 started_at / ended_at / 计数 / raw_hash / text_hash；
    # 最后一句结束后超过 gap 的会话置 closed；closed 的会话又来了新句子则回到 open。会话可以跨午夜（day = 开始那天）。
def for_day(db, day) -> list[dict]
def get(db, session_id, *, with_utterances=False) -> dict | None
def lines(db, session_id) -> list[dict]          # 给 LLM 的行：{"i", "id", "time", "speaker", "text"}
# episodes.py
def run_pending(cfg, db, client, *, stop_event=None, events=None, limit=None) -> dict
    # 对 closed 且 digested_hash != text_hash、且已校对（或校对被关 / 没 key）的会话生成情节：
    # 一次 LLM 调用（json_schema）返回 {kind, title, summary, topics[], participants[], items:[{kind,title,content,subject,evidence:[i...],
    #  importance(1-5), due?, owner?, confidence}]}；行前带 [编号]，evidence 在代码里校验并换成句子 id。
    # 句子太少（< memory.episode_min_chars，默认 60 字）只填 kind/title 不调 LLM。media / noise 的会话不提取记忆。
    # 本人说"记一下 / 提醒我 / 备忘"之类 -> 高重要度条目，source='voice'。
```
`DailyAnalyzer` 改为优先复用情节（已有摘要的会话不再做 map 调用；没有的照旧），日报的 reduce 输入里带上当天新增 / 完成的待办。
`analysis.session_gap_min`（日报用，20 分钟）保留；`memory.session_gap_min` 是记忆层的会话粒度。

### 7.2 记忆条目（`memory/items.py`）

```python
def fingerprint(kind, title) -> str             # 去空白 / 标点 / 大小写后取 sha1 前 16 位
def upsert(db, cfg, item: dict, *, source="auto", embedder=None) -> tuple[int, str]   # -> (id, "added"|"updated"|"skipped")
    # 同指纹已存在：status='rejected' -> skipped（墓碑）；用户编辑过（source='user' 或 pinned）-> 只补 evidence / last_seen；否则合并
def search(db, cfg, query, *, kinds=None, status=("active",), subject=None, limit=20, embedder=None) -> list[dict]   # FTS + 向量 RRF + 新近度 / 重要度
def related_facts(db, cfg, text, limit=8) -> list[str]     # 给校对 / 日报用的短事实行
def update(db, item_id, **fields) -> dict       # 编辑 / 置顶 / 改状态（done / archived）；标记 source='user'
def reject(db, item_id) -> None                 # 墓碑：status='rejected'，以后同指纹不再自动生成
def merge(db, keep_id, drop_id) -> dict
def todos(db, *, status="active", due_before=None) -> list[dict]
def people(db, cfg) -> list[dict]               # 人物档案：登记的说话人 + 词库里的人名 + 记忆里的 subject，按别名归并
def person(db, cfg, name) -> dict               # {name, aliases, relation, speaker_id, facts[], todos[], recent_sessions[], speech_s...}
```
老的 `memory_notes` 读写全部改走 `memory_items`：`DailyAnalyzer._upsert_notes`、`relations.person_notes`、聊天工具 `search_notes`、
`GET /api/notes` / `DELETE /api/notes/{id}`（兼容保留，内部读新表，删除 = 墓碑）。受影响的测试同步更新。

### 7.3 整理、关于我、遗忘、晨报

- `memory/consolidate.py`：`run(cfg, db, client, *, stop_event=None, events=None, limit=None) -> dict`。每晚日报之后：
  ① 同类条目按向量相似度（≥ 0.85）成组；② 每组一次 LLM 调用（json_schema）决定 `KEEP / MERGE / SUPERSEDE / DONE`
  （新的事实取代旧的、待办被后面的话证明已完成 → `done`）；③ 冲突拿不准 → `inbox.add(kind='conflict')`；
  ④ 重要度衰减（长期没被提到、没被引用的非置顶条目降权，低于阈值归档——不删除）；⑤ 刷新"关于我"。
  幂等（kv 记录已处理的指纹组）；用户编辑过 / 置顶的条目只读。
- `memory/profile.py`："关于我"（kv `core_profile`：`{"text": "...", "auto": true, "updated_at": ...}`，≤ 1500 字）：
  本人是谁、常提到的人及关系与别称、在做的项目、偏好、未完成的要紧待办。`core_block(db, cfg) -> str`、`refresh(cfg, db, client)`、
  `set_text(db, text)`（用户手改后 `auto=false`，自动刷新只在旁边给"建议更新"，不覆盖）。注入：聊天 system prompt、日报 map/reduce、会话校对。
- `memory/forget.py`：`forget(db, cfg, *, day=None, time_from=None, time_to=None, utterance_ids=None, session_id=None, dry_run=True) -> dict`。
  跨层删除并给出报告：句子（FTS / 向量 / 声纹 / 修订级联）、片段的转写原文、音频文件、簇向量、会话、由这些句子支撑的记忆条目
  （证据全没了的置墓碑，部分没了的去掉证据）、日报与 Markdown 标记为需重建、待确认条目。`dry_run` 只报告数量。
- `memory/brief.py`：`run(cfg, db, client, ...)`：每天 `analysis.brief_time`（默认 `"08:30"`，空串关闭）推送晨报——
  今天到期 / 已过期的待办、昨天没闭环的事、待确认里有几个问题。没有内容就不推。`push.detail: full | brief`（brief 时日报推送只发概览）。

### 7.4 检索与问答

- 让"混合"名副其实：关键词查询只用内容词（去停用词，保留实体 / 数字 / 英文），并用词库别名做查询扩展；
  时间表达支持到"上午 / 晚上 / 中午"。
- 多视角向量：`memory.vector_views`（默认 `["plain","window"]`，可选 `speaker`），额外视角存 `memory_vectors(kind='utt')`，RRF 融合。
  默认值按 `scripts/memory_bench.py` 的数字定（并把存储增量写进报告）。
- 预取上下文 = 关于我 + 相关记忆条目 + 相关情节摘要 + 句子命中（带前后各一句）。
- 新工具：`search_memory(query, kind?)`、`list_todos(status?)`、`get_session(session_id)`、`get_person(name)`；
  用户**明确要求**时的写工具：`remember(text, kind?)`、`complete_todo(id)`、`correct_term(wrong, right)`（→ 词库别名）。
- 回答里用 `[#句子id]` 引用，服务端解析成 `citations`；被引用的条目 / 句子 `hits += 1`。
- 聊天落库（`chat_conversations` / `chat_messages`）；流式输出（`chat_stream`，假客户端没有该方法时退回整段）。

### 7.5 评测

扩展 `scripts/memory_bench.py`：无 LLM 的层级（预取覆盖、情节、条目去重、待办闭环、遗忘是否干净）+ 原有 retrieval / chat 层级，
按 tag 汇总；修掉 `requests` / `calls` 键名不一致。默认值按结果定。

## 8. 接口（新增 / 变化；除注明外都返回 JSON，错误统一 `{"error": "中文说明"}`）

路由文件及归属见 §11。已有接口的路径和返回形状**不许变**（只许加字段）。

| 接口 | 说明 |
|---|---|
| `GET /api/events` | SSE（A1） |
| `GET /api/metrics?days=7` | 本机 / LLM 用量（A1） |
| `GET /api/transcript/{day}` | 每句新增 `session_id`、`text_source`、`marks`、`speaker_guess: {id,name,score}?`、`flags`；顶层新增 `sessions: [...]`（B2 + B3） |
| `PATCH /api/utterances/{id}` `{text, remember?: bool}` | 手改文字 → `{utterance, learned?}`（B2） |
| `GET /api/utterances/{id}/revisions`、`POST /api/utterances/{id}/undo {revision_id?}` | 修订历史 / 撤销（B2） |
| `GET /api/inbox?status=open&kind=&limit=&offset=` | `{items, counts}`（B2） |
| `POST /api/inbox/{id}/answer` `{choice?, text?, as_correct?, apply_history?}`、`/dismiss {never?}`、`/snooze {days?}` | （B2） |
| `GET /api/lexicon?q=&kind=&status=`、`POST /api/lexicon`、`PATCH /api/lexicon/{id}`、`DELETE /api/lexicon/{id}` | 词库 CRUD（B2） |
| `POST /api/lexicon/{id}/aliases`、`DELETE /api/lexicon/aliases/{id}`、`POST /api/lexicon/aliases/{id}/apply {dry_run?}` | 别名与回溯（B2） |
| `GET/PUT /api/glossary` | 兼容：导出 / 导入词库（B2） |
| `GET /api/sessions?day=`、`GET /api/sessions/{id}` | 会话 / 情节（B3） |
| `GET /api/memory/items?kind=&status=&q=&subject=&limit=&offset=`、`POST`、`PATCH /{id}`、`DELETE /{id}`、`POST /{id}/merge {into}` | 记忆条目（B3） |
| `GET /api/memory/people`、`GET /api/memory/people/{name}` | 人物档案（B3） |
| `GET/PUT /api/memory/profile`、`POST /api/memory/profile/refresh` | 关于我（B3） |
| `POST /api/memory/forget` `{day?, time_from?, time_to?, utterance_ids?, session_id?, dry_run?}` | 遗忘（B3） |
| `POST /api/chat` | 新增入参 `conversation_id?`；事件流新增 `delta {text}`、`sources {citations}`；`answer` 带 `conversation_id`（B3） |
| `GET /api/chat/conversations`、`GET/DELETE /api/chat/conversations/{id}` | 聊天记录（B3） |
| `GET /api/voices`（已有）新增持久 `voice_id`、`status`；`POST /api/voices/{id}/ignore {ignored: bool, label?}` | （B1） |
| `GET /api/speakers/{sid}/review`（已有）| 队列包含"可能是 TA"的句子（两侧都问），按影响排序（B1） |
| `POST /api/jobs/{name}` | 支持 §4.6 的新任务名（A1 / A2） |

## 9. 界面（F1 打底，FA / FB / FC 扩展）

### 9.1 设计语言："安静的日记本"

不是后台管理面板，而是一本自己会整理的日记：留白多、层级清楚、正文好读、状态一眼可见、专业词汇不出现在主界面。

- **外壳**：左侧导航栏（232 px，可收成 64 px 图标栏；< 860 px 变底部标签栏）；顶栏 56 px：页面标题、全局搜索 / 命令面板入口（Ctrl+K）、
  录音控制胶囊（状态点 + 已录时长 + 开始 / 停止）、待确认铃铛（带数字）、主题切换（自动 / 浅 / 深）、档案切换；
  内容区最大 1120 px 居中；播放时底部出现全局播放条。导航：总览 · 记录 · 待确认 · 记忆 · 日报 · 问答 · 说话人 · 设置（诊断 / 日志 / 开销并入设置）。
- **字体**：`"Segoe UI Variable Text","Segoe UI","PingFang SC","Microsoft YaHei UI","Microsoft YaHei",system-ui,sans-serif`；
  字号阶梯 12 / 13 / 14 / 15（正文阅读，行高 1.7）/ 17 / 20 / 28；数字用 `tabular-nums`。
- **颜色**（语义 token，`:root` 浅色、`:root[data-theme=dark]` 与 `prefers-color-scheme` 深色；设置 `color-scheme`；
  所有文字 token 对其背景对比度 ≥ 4.5）：暖纸色背景 `--bg #F6F5F1`、`--surface #FFFFFF`、`--surface-2 #F0EEE8`、`--line #E4E1D9`、
  `--text #1F1E1B`、`--text-2 #5E5B54`、`--text-3 #77736A`；主色 `--accent #3A56D4`（靛蓝）；录音 `--rec #D9482B`；
  `--ok #2E7D5B`、`--warn #B7791F`、`--danger #C93A3A`；深色背景 `#151513 / #1E1D1B / #262522`，文字 `#ECEAE4 / #B3AFA6 / #8F8B82`，主色 `#8DA2FF`。
  说话人 12 色（墨色 + 浅底成对），**按说话人 id 稳定分配**；本人用主色；未知声音灰色。
- **形状与层次**：圆角 6 / 10 / 14；卡片靠浅阴影与底色分层，尽量不用边框；间距阶梯 4 / 8 / 12 / 16 / 24 / 32 / 48；
  `:focus-visible` 统一焦点环；动效 120–180 ms，尊重 `prefers-reduced-motion`。
- **组件**（`js/ui/`）：按钮（主 / 次 / 幽灵 / 危险）、标签、卡片、分段控件、标签页、气泡、右侧抽屉、可叠放的对话框（Esc / 点遮罩必定 resolve）、
  带"撤销"的 toast、骨架屏、空状态（一句话 + 一个动作）、内联 SVG 图标（不引外部资源）、无依赖 SVG 图表。
- **安全**：只允许通过会自动转义的 `html`` ` 模板写 `innerHTML`；用户数据一律转义。
- **记录页**：按会话分节（标题、时间段、时长、参与人头像、类型徽标、整段播放）；同一个人连续的几句合成一个"发言块"（头像 + 名字 + 时间，
  正文逐句成行）；悬停 / 聚焦出现操作条（播放、是谁、不是 TA、拆分、编辑、上下文）；被自动改过的词细点线下划线（悬停看原文和依据）；
  不确定的词琥珀色波浪线（点开候选气泡，直接回答）；"可能是 X"用虚线标签 + 一键确认 / 否定；顶部一条 24 小时"谁在什么时候说话"色带。

### 9.2 模块结构（无构建、原生 ES 模块）

`index.html` → `<script type="module" src="/static/app.js?v=__V__">`；`app.js` 只 `import "./js/main.js"`。
`js/core/{util,format,md,html,dom,api,bus,store,router,status,theme,shortcuts}.js`、`js/ui/{icons,components,modal,toast,charts,player,palette,speaker,utterance}.js`、
`js/pages/{dashboard,timeline,inbox,memory,daily,chat,people,settings}.js`（动态 `import()`）、`css/{tokens,base,components,pages}.css`。
`core/{util,format,md,html}.js` 导入时不碰 DOM（node 能直接跑单测）。静态资源戳按 `static/` 下所有文件的最新修改时间算；
`.js/.mjs` 的 MIME 固定为 `text/javascript`。状态走 `/api/events`（SSE）+ 断线退避轮询。

### 9.3 分工

- **F1**：外壳、设计 token、组件库、路由、SSE 客户端、全局播放条、命令面板骨架；把现有 7 个页面搬到新结构并按 §9.1 重做视觉
  （只用现有接口）；「待确认」「记忆」先放空状态占位页。
- **FA**：记录页的标记 / 内联编辑 / 会话分节 / 猜测确认；待确认页（逐条卡片、数字键回答、撤销）；总览上的"有 N 件事想问你"。
- **FB**：记忆页（条目 / 待办 / 人物档案 / 词库 / 关于我 / 遗忘）；问答页（流式、停止、引用可点、历史对话）。
- **FC**：总览图表（日历热力、7 天趋势、今天谁说了多久）、开销面板、说话人页升级（持久声音 / 忽略 / 两侧核对 / 校准图）、
  由 schema 驱动的设置页、命令面板命令与搜索。

## 10. 评测工具（E，`scripts/bench/`）

- `story.py`：**手写的固定剧本**（3 天、每天 4 场、5 个角色 + 1 个只在系统声音里出现的"视频旁白"），台词里刻意反复出现一批专有名词
  （人名昵称、项目名、英文缩写、产品型号、数字金额），数字一律按 ASR 的写法约定；同时给出：每场的类型、应提取的事实 / 待办 / 决定、
  第二天会被"证明已完成"的待办、相互矛盾的事实（考冲突检测）。
- `build.py`：用 MiMo TTS 合成（固定音色；自定义音色先设计一条再克隆，保证同一角色音色一致），缓存到 git 忽略的目录；
  拼成会话音频并生成声学变体：干净近讲 / 远讲（-20 dB + 合成混响 + 底噪）/ 两人重叠 / 系统声音通道（带 `channel_rms`）；
  输出 `truth.json`（每句：说话人、文字、起止时间、条件）；能把结果导入一个独立的 bench 档案目录。
- `bench_cer.py`、`bench_speakers.py`、`bench_cost.py`（各阶段 CPU 秒 / 音频秒，1/2/4 线程）、`report.py`（汇总成 Markdown 表）。
- 先在**升级前的代码**（`git worktree` 检出 `a1f36ac` 到临时目录）上跑出基线，升级后各包用同一套数据对比。
- 评测脚本里的纯函数（对齐、CER、归一化）有单元测试。

## 11. 工作包与文件归属

所有人都在同一个工作目录里干活，**只改自己名下的文件**；别人名下的文件只读。共享文件（`db.py`、`config.py`、`config.example.yaml`、
`ui/settings_schema.py`、`runtime.py` 的任务表、`ui/routers/__init__.py` 的列表、`tests/conftest.py`、`pyproject.toml`）在第一波之后
只允许**追加式的小修改**：改之前重新读文件，用精确的小块替换，绝不整文件重写。

**第一波（结构与地基，行为不变，800 个现有测试必须全过）**

| 包 | 归属 |
|---|---|
| A1 核心 | `db.py`、`config.py`、`config.example.yaml`、`events.py`、`metrics.py`、`governor.py`、`chain.py`、`llm/client.py`、`runtime.py`、`scheduler.py`、`service.py`、`ui/server.py`、`ui/routers/events.py`、`audio/recorder.py`（只加事件 / 唤醒 / 线程优先级）、`__init__.py`、`pyproject.toml` |
| A2 接口拆分 | `ui/app.py`、`ui/deps.py`、`ui/errors.py`、`ui/serializers.py`、`ui/settings_schema.py`、`ui/static_site.py`、`ui/routers/`（除 `events.py`）|
| A3 命令行拆分 | `cli.py` → `cli/` 包、`logging_setup.py` |
| A4 流水线拆分 | `speech/pipeline.py` → §2 列的模块 |
| F1 前端打底 | `ui/static/**` |
| E 评测 | `scripts/bench/**`、`tests/test_bench_*.py` |

**第二波（功能，后端）**：B1 = `speech/{scoring,tracks,speakers,voices,review,stage_voice}.py`、`ui/routers/{speakers,voices,enroll}.py`、`cli/` 里说话人相关命令；
B2 = `speech/{pinyin,lexicon,guard,revisions,refine,hearing,asr,correct,stage_text}.py`、`inbox.py`、`ui/routers/{inbox,lexicon,utterances,transcripts}.py`、`cli/` 里新增的 `lexicon` / `inbox` 命令；
B3 = `memory/**`、`analysis/**`、`push/base.py`、`ui/routers/{memory,sessions,chat,daily}.py`、`scripts/memory_bench.py`、`cli/` 里新增的 `memory` 命令。

**第三波（功能，前端）**：FA / FB / FC 见 §9.3，各自的页面模块；`js/ui/` 下的公共组件只追加。

**第四波**：集成验证、对抗式评审、修复、文档（README / DESIGN / CHANGELOG / 版本号 1.0.0）。

## 12. 测试与安全约定

- 运行：`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/<文件>.py`（在本工作目录下）。
  其他人同时在改别的文件：与你无关的文件里偶发的导入错误，等一分钟重跑；不要去修别人的文件，写进汇报。
- 新功能的测试放新文件 `tests/test_v1_<主题>.py`；现有测试只有在行为**有意**改变时才改，并在汇报里逐条说明。
  假 LLM / 假 ASR / 假引擎沿用鸭子类型；新参数必须可选，调用假对象前做签名探测。
- 绝不：读写 `C:\Users\enhua\Documents\docs\audio_note\profiles\`（真实数据）和主目录的任何文件；打印 / 记录 / 写入 API key；
  启动 8765 端口的服务；用麦克风；`git commit` / `git push` / 切分支；按名字杀进程。
- 仓库是公开的：代码、测试、文档里不出现真实人名、真实对话原文；统计数字可以写。
