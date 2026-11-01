# 说话人模块 v2：本人、关系、从记录里直接标记

实现规格（给实现者看的；实现完成后 README / DESIGN 另写用户文档）。项目根 `C:\Users\enhua\Documents\docs\audio_note`，
Python 在 `.venv/Scripts/python.exe`，测试命令：

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/<file>.py

测试约定见 `tests/conftest.py`（`cfg_factory` 走 AUDIO_NOTE_CONFIG，`speaker_engine` 会话级真实 CAM++ 模型，
`SAMPLE_WAV` 真实录音：前 29 s 静音，29–37 s 有语音；`tests/test_ui_api.py` 里的 `_seg/_utt/_speaker_with_samples`
是建假片段 / 假语句的 helper，可复制到新测试文件）。所有新测试不得碰根目录 config.yaml、不得联网、不得用真实麦克风。

## 0. 用户提出的三点

1. 要有唯一的「本人」：后续日报、推送、问答都是站在本人视角；记录里本人的话就是"我"说的。
2. 其他说话人除了名字，可以标注和本人的关系（朋友 / 好朋友 / 领导 / 导师…），不强制；后续可以从记录里自动分析补上。
3. 登记说话人太繁琐，不可能请每个人单独来录样本。应当**直接从已有录音里标记**：把未识别的声音列出来，听一下、填名字就行；
   不准再补素材。

## 1. 数据库（`audio_note/db.py`）

SCHEMA 追加（`CREATE TABLE IF NOT EXISTS`）：

```sql
CREATE TABLE IF NOT EXISTS cluster_embeddings (
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    cluster_label INTEGER NOT NULL,
    embedding BLOB NOT NULL,             -- float32, L2-normalised
    dim INTEGER NOT NULL,
    speech_s REAL NOT NULL,              -- total duration of this cluster's turns in the segment
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (segment_id, cluster_label)
);
```

`DB.__init__` 在 `executescript(SCHEMA)` 之后调用 `self._migrate(c)`：用 `PRAGMA table_info` 检查并 `ALTER TABLE … ADD COLUMN` 缺的列
（老库升级；新库也走同一条路，SCHEMA 里的 CREATE TABLE 不改）：

- `speakers.is_self INTEGER NOT NULL DEFAULT 0`（全表最多一行为 1）
- `speakers.relation TEXT`、`speakers.relation_source TEXT`（'user' | 'auto' | NULL）、`speakers.relation_note TEXT`（推断理由 / 用户备注）、`speakers.relation_updated_at TEXT`
- `utterances.speaker_source TEXT`（NULL = 自动识别；'user' = 用户亲手标的，relabel 不许改）

新增方法：

- `self_speaker() -> sqlite3.Row | None`
- `set_self_speaker(sid: int | None) -> None`：事务里先把所有 is_self 清 0，再把 sid 置 1（None 只清）。
- `set_relation(sid, relation: str | None, source: str | None, note: str | None = None) -> None`：relation 为 None/"" 时三列都清空；否则写入并更新 relation_updated_at。
- `people_table() -> list[dict]`：`[{id, name, is_self(bool), relation, relation_source, relation_note, utterances(int)}]`，本人排第一，其余按语句数降序。供 prompt / 工具 / API 用。
- `upsert_cluster_embedding(segment_id, cluster_label, emb, speech_s)`；`cluster_embeddings(segment_ids: list[int] | None = None) -> list[tuple[int, int, np.ndarray, float]]`；`delete_cluster_embeddings(segment_id)`。
- `speaker_embeddings()` 不变。

测试（`tests/test_db.py` 追加或新文件 `tests/test_speakers_v2_db.py`）：老库升级（先用旧 SCHEMA 建表再开 DB → 列齐全且幂等）、set_self 唯一性、set_relation 清空、people_table 顺序、cluster_embeddings 往返与级联删除。

## 2. 声纹引擎与流水线

`audio_note/speech/speakers.py`：

- `identify_turns(audio, turns, db, segment_id: int | None = None)`：行为不变，但当 `segment_id` 给了，就把每个 cluster 的 embedding（已算出）
  用 `db.upsert_cluster_embedding(segment_id, cid, emb, speech_s)` 存起来（speech_s = 该 cluster 所有 turn 时长之和；不足 0.5 s 的 cluster 不存）。
- 新增 `cluster_embedding(audio, turns_of_one_cluster) -> np.ndarray | None`（抽出上面的公共逻辑：拼接、最多 60 s、不足 0.5 s 返回 None）。

`audio_note/speech/pipeline.py`：

- `process_segment` 里 `identify_turns(audio, turns, self.db, segment_id=seg_id)`。
- `relabel_segment`：同样传 `segment_id`（顺便刷新 cluster_embeddings）；**跳过 `speaker_source='user'` 的语句**（不更新它们的 speaker_*），
  `changed` 只统计真正改了的。
- `write_day_markdown` 不变。

## 3. 未识别的声音：`audio_note/speech/voices.py`（新）

```python
RELATION_CHOICES = ("家人", "伴侣", "朋友", "好朋友", "同事", "领导", "下属", "导师", "学生", "客户", "合作伙伴", "同学", "其他")   # 放在 analysis/relations.py，这里 import

def group_unknown_voices(db, cfg, threshold: float | None = None, max_members: int = 2000) -> dict
def backfill_cluster_embeddings(db, cfg, engine, limit: int | None = None, stop_event=None) -> dict
def name_voice(db, cfg, engine, members: list[tuple[int, int]], name: str, *, is_self=False, relation=None, max_samples=3) -> dict
```

- `group_unknown_voices`：成员 = `cluster_embeddings` 里那些 (segment_id, cluster_label) 中**至少有一条 speaker_id IS NULL 的语句**的 cluster
  （用 utterances 聚合：n 句、总时长、day、最长的几句）。按 started_at 取最近 `max_members` 个。
  贪心平均连接聚类：每个成员初始一组；反复合并"质心 cosine 最大且 ≥ threshold"的两组（质心 = 按 speech_s 加权的均值再归一化），直到没有可合并的。
  threshold 默认 `cfg.speakers.voice_group_threshold`（DEFAULTS 新增，0.55）。
  返回：

  ```python
  {"groups": [{"key": "<sha1(sorted members)[:10]>", "members": [[seg, cluster], ...],
               "utterances": n, "speech_s": float, "segments": k, "days": ["2026-10-27", "2026-10-29"], "n_days": d,
               "samples": [{"utterance_id", "day", "time": "HH:MM", "text", "seconds", "audio": bool}],   # 最多 5 条，优先有音频且最长
               "closest": {"id", "name", "score"} | None}],                                            # 与已登记声纹的最高相似度（提示“可能是 X”）
   "missing_segments": m,      # status='done'、有未识别语句、音频文件还在、但没有 cluster_embeddings 的片段数（需要扫描）
   "threshold": t, "unknown_utterances": total}
  ```
  groups 按 speech_s 降序。纯 numpy，不加载模型（成员数 ≤ 2000 时 O(n²) 可接受）。
- `backfill_cluster_embeddings`：对上面的 "missing" 片段：读音频（`audio.storage.read_segment`），用该片段的 utterances 当 turns（同 relabel_segment 的做法），
  每个 cluster 算 embedding 存起来，不做识别。返回 `{"segments": n, "clusters": c, "skipped": s}`；尊重 stop_event。
- `name_voice`：
  1. `name` strip 后非空；`sid = db.upsert_speaker(name)`（新旧皆可）。
  2. 成员按 speech_s 降序，最多 `max_samples` 个有音频的：读片段音频、拼接该 cluster 的所有语句区间（按 start_s 排序），
     `engine.enroll_from_audio(db, name, audio, source=f"{path}#{first_start:.1f}-{last_end:.1f}")`；ValueError（语音太短）→ 跳过。
     若一个样本都没加成且这个 speaker 是刚建的（之前没 embedding），删掉 speaker 行并 `raise ValueError("这组声音里的语音太短，无法提取声纹")`。
  3. 直接指派：对每个成员 `UPDATE utterances SET speaker_id=?, speaker_name=?, speaker_score=NULL, speaker_source='user' WHERE segment_id=? AND cluster_label=?`
     （包括没音频的成员——用户亲耳确认过同组）。
  4. `is_self` → `db.set_self_speaker(sid)`；`relation` 非空 → `db.set_relation(sid, relation, 'user')`。
  5. 受影响的 day 重写 markdown（`pipeline.write_day_markdown`）。
  6. 返回 `{"id", "name", "samples_added", "utterances_assigned", "segments": [...], "days": [...]}`。
  不在这里做 FTS 重建（API 层调 `_after_enroll()`）。

测试 `tests/test_voices.py`：合成 embedding 的分组（两簇明显不同 + 一簇跨多片段应合并；阈值边界；空库；没有未识别语句的 cluster 不参与）；
`closest` 提示；真实引擎：用 SAMPLE_WAV 29–37 s 建片段 + 语句（cluster 0）→ backfill 产生 cluster_embeddings → group 得到 1 组 → name_voice
加样本、指派语句、is_self/relation 生效、markdown 重写；语音太短的组报错且不留空 speaker；relabel_segment 不改 user 标的语句。

## 4. 关系推断：`audio_note/analysis/relations.py`（新）

```python
RELATION_CHOICES = (...)            # 见上
def infer_relations(cfg, db, client, speaker_ids: list[int] | None = None, force: bool = False) -> list[dict]
def maybe_infer_after_daily(cfg, db, client) -> list[dict]     # 日报之后调用；配置关了或没 key 就返回 []
```

- 候选：`is_self=0`，`relation_source != 'user'`（force 时也包括 user 标的？**不**，用户标的永远不动；force 只是忽略"已推断过"的缓存），
  语句数 ≥ `cfg.speakers.relation_min_utterances`（DEFAULTS 15）。缓存：kv `relation_inferred_n:<sid>` 存上次推断时的语句数，
  语句数没涨到 1.5 倍且不 force 就跳过。
- 证据：该说话人最多 60 条语句（跨天均匀抽样），每条带同片段里前一条和后一条语句（说话人名；本人标「本人」），再加 memory_notes 里 kind='person'、title=该名字的 content。
- 提示词（JSON 模式，`client.json(system, user, max_tokens=512)`）：系统：你是用户的私人记录助手；记录里「S」是用户本人（没登记本人就说"用户本人没有登记，
  对话里的第一人称可能是本人"）；判断「X」和用户本人的关系，只能从给出的对话判断，不确定就说不确定；输出
  `{"relation": 候选之一或"不确定", "confidence": 0-1, "reason": "≤60字，引用称呼/事件", "address": "X 对本人的称呼，可空"}`。候选列表写进提示。
- `confidence ≥ cfg.speakers.relation_min_confidence`（DEFAULTS 0.6）且 relation 在候选里 → `db.set_relation(sid, relation, 'auto', note=reason)`，并写 kv。
  否则不写 relation 但仍写 kv（避免每天重复问）。LLMError 单人失败不影响其他人（记日志）。
- 返回 `[{"id", "name", "relation", "confidence", "reason", "applied": bool, "skipped": str|None}]`。
- 配置 DEFAULTS（`config.py` speakers 段）：`infer_relations: True`、`relation_min_utterances: 15`、`relation_min_confidence: 0.6`、`voice_group_threshold: 0.55`；
  `config.example.yaml` 同步加注释。

接入：

- `runtime.py`：JOBS 增加 `"voices"`（标签「扫描未识别的声音」→ `voices.backfill_cluster_embeddings(db, cfg, self.engine(), stop_event=...)`）和
  `"relations"`（标签「推断关系」→ `relations.infer_relations(cfg, db, self.client(), speaker_ids=[speaker_id] if speaker_id else None, force=force)`）。
  `_job_daily` 在 `analyzer.run` 之后调用 `relations.maybe_infer_after_daily(cfg, db, self.client())`，结果放进 `out["relations"]`（异常只记日志）。
  `scheduler.py` 的每日任务同样在 run 之后调用 `maybe_infer_after_daily`（异常只记日志）。
  `status()` 增加 `"self_name": <本人名字或 None>` 和 `"unknown_utterances": <speaker_id IS NULL 的语句数>`。
- 提示词：`analysis/prompts.py` 增加 `people_line(people: list[dict], unknown_label) -> str`，形如
  `人物：我=「张伟」（用户本人）；王敏（同事）；李强（朋友，自动推断）；小雨（关系未标）`；没登记本人时写 `用户本人：未登记（记录里没有标出哪位是本人）`。
  `MAP_USER` / `REDUCE_USER` 加一行 `{people}`；`MAP_SYSTEM` / `REDUCE_SYSTEM` 加一句说明：「人物」一行给出说话人与用户本人的关系；用户本人的话按第一人称理解，
  待办里负责人是本人的就是用户自己的待办；`people` 字段不要包含用户本人。`map_prompt(..., people: str = "")`、`reduce_prompt(..., people: str = "")` 可选参数，
  旧调用方式仍可用。`daily.py` 的 `run()` 计算 `people = prompts.people_line(db.people_table(), unknown_label)` 传进去；`_upsert_notes` 跳过本人名字的 person 笔记；
  `js["self_name"]`。
- `memory/chat.py` `system_prompt`：head 之后加 `people_line` 一行（本人 + 关系），并说明"提问里的“我”指用户本人「S」"。
- `memory/tools.py` `list_speakers`：每行显示 `（本人）` / `关系：同事` / `关系：朋友（自动推断）`。

测试 `tests/test_relations.py`：假 client（`json()` 返回脚本化结果）→ 候选筛选（本人 / user 标的 / 语句太少跳过）、缓存（kv）与 force、
confidence 门槛、非法 relation 不写、单人 LLMError 不影响其他人、`maybe_infer_after_daily` 配置关闭返回 []；
prompts 的 `people_line` 各种情况；daily.run 把 people 传进 prompt（现有 test_daily 的假 client 可捕获 user prompt）；chat system_prompt 含本人行；
tools.list_speakers 输出。

## 5. API（`audio_note/ui/app.py`）与 CLI（`audio_note/cli.py`）

API：

- `GET /api/speakers`：每个 speaker 增加 `is_self, relation, relation_source, relation_note`；顶层增加 `self_id`, `relation_choices`（列表）。
- `PATCH /api/speakers/{sid}`：新增可选字段 `is_self`（true 设为本人；false 且当前是本人则清除）、`relation`（字符串；"" 清空；写入 source='user'）、
  `relation_note`。原 name/notes 逻辑不变。
- `GET /api/voices` → `group_unknown_voices(...)` 结果 + `"scanning": <voices 任务是否在跑>`。
- `POST /api/voices/name` body `{members: [[seg, cluster], ...], name, is_self?, relation?, auto_relabel?: true}`：
  线程里 `name_voice`，然后 `_after_enroll()`（jieba + FTS 重建）；`auto_relabel` 为真且库里还有其他未识别语句时 `rt.submit("relabel")`（JobError/ReadOnly 忽略，
  响应里 `"relabel_started": bool`）。ValueError → 400；members 为空 / 格式错 → 400。
- `POST /api/speakers/label`：body 新增 `whole_cluster`（默认 true）、`is_self`、`relation`。流程：enroll（现有）→ 直接指派这句（whole_cluster 时连同
  同片段同 cluster_label 的语句）`speaker_source='user'` → `_after_enroll(seg_id)`（relabel 会跳过 user 标的）→ is_self / relation 写入。
  响应增加 `"assigned": n`。
- `POST /api/jobs/{name}`：`voices` 无参数；`relations` 接受 `{speaker_id?, force?}`。
- 设置允许列表 `EDITABLE["speakers"]` 增加 `voice_group_threshold, infer_relations, relation_min_utterances, relation_min_confidence`，
  `_RANGES` 相应范围（0–1 / 1–1000 / 0–1）。`/api/status` 的新字段来自 runtime。

CLI（`audio-note speakers …`）：

- `set-self <name>`；`relation <name> <关系|->`（`-` 清空，`--note`）；`relations [--force] [--speaker NAME]`（LLM 推断，打印表格）；
- `voices`（列出未识别声音组：序号、句数、时长、日期范围、可能是谁、3 条示例文本）；`voices --assign <序号> <name> [--self] [--relation R]`（命名；先自动 backfill 缺的片段）；
  `scan`（只做 backfill）。
- `speakers list` 表格增加「本人」「关系」两列。`cmd_speakers_label` 也接受 `--self` / `--relation`。

测试：`tests/test_ui_api.py` 追加或新文件 `tests/test_speakers_v2_api.py`（GET/PATCH 新字段、本人唯一、relation 清空、/api/voices 空库与有数据、
/api/voices/name 成功 / 400、label 的 whole_cluster 与 is_self、jobs voices/relations 提交、status 新字段）；`tests/test_cli_commands.py` 追加新子命令（用 fake 引擎 / 假 client）。

## 6. 前端（`audio_note/ui/static/`）

说话人页重做（`pageSpeakers`）：

1. 页头副标题改成：「不用单独录样本：下面列出记录里还没认出来的声音，听一下是谁、填上名字就行；登记过的人会自动识别。」
   按钮：`自动推断关系`（job relations）、`重新识别全部记录`、`校准阈值`、幽灵按钮 `用麦克风 / 文件补充样本`（打开原 enrollPanel）。
2. **「未识别的声音」**区（`GET /api/voices`）：
   - `missing_segments > 0` 且没在扫描 → 自动 `POST /api/jobs/voices` 一次并显示「正在分析 n 段录音…」，任务完成（jobs 轮询 `state.refresh`）后重新拉取。
   - 每组一张卡：标题「声音 A」（A、B、C…按顺序）+ `12 句 · 48 秒 · 3 天（10-27 ~ 10-29）`；若 `closest` 的 score ≥ 0.3 显示「可能是 张三（0.41）」并把名字预填。
     示例句列表（时间 + 文本 + ▶ 播放 `/api/audio/utterance/{id}?pad=0.3`，没音频的灰显）。
     底部一行：姓名输入（datalist = 已登记名字）、关系 `<select>`（空 / RELATION_CHOICES / 自定义…→ 变文本框）、勾选「这是我自己」（库里已有本人时文字为「这其实是我」）、按钮「就是 TA」。
     成功 → toast「已标记为 X：n 句，补充 k 个声纹样本；正在用新声纹重新识别其他记录」→ 重新渲染页面。
   - 没有未识别声音时显示一行「没有待确认的声音」。
3. **「已登记的人」**卡片：姓名旁本人显示徽章 `我`；关系 chip（没有时显示「＋ 关系」），点击弹窗：select（候选 + 自定义）+ 备注，保存 PATCH；
   `relation_source==='auto'` 的 chip 带「自动」小字和 title=relation_note。按钮行：`设为本人`（已有本人且不是它时文字「改为本人」，确认框）、`补充样本`、`样本列表`、`重命名`、`备注`、`删除`。
4. 空状态（没有登记的人且没有声音组）：「还没有记录，录一会儿或导入音频后，这里会列出听到的声音。」

记录页：

- `uttRow`：`state.status.self_name` 等于 u.speaker 时在名字后加 `<span class="badge me">我</span>`（样式小而淡）。
- `labelUtt` 弹窗：加勾选「这段录音里同一个声音的其他句子也一起标」（默认勾）；新名字（不在已登记列表里）时显示「这是我自己」勾选和关系 select；
  已登记的名字就不显示。请求体带 `whole_cluster / is_self / relation`。toast 用响应的 `assigned`。

总览页：`status.unknown_utterances > 0` 时在录音状态卡下方加一行提示「有 N 句话还没认出是谁说的 → 去标记」链接到 `#/speakers`。

设置页「说话人」小节：加 `voice_group_threshold`（数字，"未识别声音分组阈值，越高分得越细"）、`infer_relations`（开关「日报后自动推断关系」）、
`relation_min_utterances`、`relation_min_confidence`。

`style.css`：`.voice-card`、`.badge.me`、`.relation-chip`（含 `.auto` 变体）、示例句行 `.voice-sample`，深色模式都要看得清。无外部资源。

## 7. 文档（最后做）

README：第 7 条功能点、「说话人」章节改写（主流程 = 在说话人页命名未识别的声音 / 记录页 👤；本人；关系；自动推断）；命令表加新子命令；界面表格更新。
DESIGN.md：§说话人 增加本人 / 关系 / cluster_embeddings / voices 分组算法 / 关系推断；schema 列表更新。
`config.example.yaml`：新键。
