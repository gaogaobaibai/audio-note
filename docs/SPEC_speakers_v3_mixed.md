# 说话人 v3：混合句子的防与治

背景（2026-11-01，用户真实数据上的探针结果）：两人面对面聊天时，sherpa-onnx 的 fast clustering（阈值 0.5）常把两个人并成一个簇，
`build_chunks` 再把同簇相邻 turn 合并成 30–60 s 的 ASR 块，于是一句 utterance 里既有用户又有朋友的话，没法标记。
更糟的是用户给混合的声音组起名后，从混合簇提取的"声纹样本"混进了别人的声音（用户 7 个样本里 4 个和麦克风录的 3 个相似度只有 0.24–0.49），
之后谁都被认成用户。逐 turn（≥ 0.8 s）的声纹和干净样本比对是能分开的：用户 0.5–0.74，朋友 0.1–0.3。

实现规格。项目根 `C:\Users\enhua\Documents\docs\audio_note`，Python `.venv/Scripts/python.exe`，测试：
`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/<file>.py`。测试约定见 `tests/conftest.py`
（`cfg_factory`、会话级真实 `speaker_engine`、`SAMPLE_WAV` 29–37 s 有语音、`chirp/noise/sine` 合成信号的 CAM++ embedding 彼此近乎正交，
可以当"两个人"用）。不碰根目录 config.yaml / profiles/*，不联网，不用麦克风。

## 1. 声纹样本一致性（speech/speakers.py, db.py, ui/app.py, cli.py）

- 新配置 `speakers.sample_consistency_min`（DEFAULTS 0.5，config.example.yaml 注释：新样本和此人已有样本的最高相似度低于它就拒绝）。
- `SpeakerEngine.check_sample(db, speaker_id, emb) -> tuple[float | None, bool]`：返回 (与该人已有样本的最高 cosine 或 None（没有样本）, ok)。
- `enroll()` / `enroll_from_audio()` / `enroll_from_segment_turns()` 新增参数 `force: bool = False`：不 force 且 ok 为 False 时
  `raise SampleMismatch(name, score)`（新异常，`ValueError` 子类，中文消息：「这段声音和 X 已有的样本不像（相似度 0.31 < 0.5），没有加入；
  确定是同一个人可以用"仍然加入"」）。`enroll()` 对多文件逐个判断，不一致的跳过并在日志里说明，返回值不变。
- `voices.name_voice()`：成员之间先做一致性筛选——按 speech_s 取最大的成员作为锚，和锚的 cosine < sample_consistency_min 的成员不用来做样本
  （仍然指派语句，但结果里报 `members_inconsistent: n`）；提取出的样本经 `check_sample`（对已登记的人），不一致就跳过并计数 `samples_rejected`。
- 审计：`SpeakerEngine.audit_samples(db, speaker_id) -> list[dict]`：每个样本 `{id, source_path, duration_s, consistency, outlier}`，
  consistency = 该样本与其他样本的最高 cosine（只有一个样本时 None），outlier = consistency 不为 None 且 < sample_consistency_min。
  纯 numpy，不加载模型（embedding 已存库）。
- API：`GET /api/speakers/{sid}/samples` 每行增加 `consistency`、`outlier`；`GET /api/speakers` 每人增加 `outlier_samples`（数量）；
  `DELETE /api/speakers/{sid}/samples/outliers` 删除该人所有 outlier 样本，返回 `{deleted, remaining}`（删完若一个样本都不剩，speaker 行保留）；
  `POST /api/speakers/{name}/samples` 与 `POST /api/speakers/label`、`POST /api/voices/name` 接受 `force`（query / body），
  SampleMismatch → 400，消息里带 `"mismatch": true` 和分数，前端据此给「仍然加入」按钮。
- CLI：`speakers audit [--fix]`（表格：人 / 样本 / 时长 / 一致性 / 是否离群；`--fix` 删除离群样本）；`speakers enroll/label/voices --assign` 加 `--force`。

## 2. 逐 turn 识别（speech/speakers.py `identify_turns`）

在簇级识别之后加一层 turn 级修正（只在有登记声纹时）：

```
for each cluster C with turns T:
    cluster decision: (S_c, score_c) as today
    for t in T with duration >= speakers.turn_check_min_s (DEFAULTS 1.0):
        e_t = embed(audio[t]); (S_t, s_t) = match(e_t, enrolled, thr) ; sim_to_Sc = max cos to S_c's samples (if S_c)
        if S_c is not None:
            if S_t is not None and S_t != S_c and sim_to_Sc < thr:        -> t belongs to S_t: new cluster id, speaker S_t, score s_t
            elif sim_to_Sc < thr - speakers.turn_reject_margin (DEFAULTS 0.15) and S_t is None: -> t unknown: new cluster id, speaker None
        else:
            if S_t is not None: -> new cluster id, speaker S_t, score s_t
    turns shorter than turn_check_min_s keep the cluster decision
```

新簇 id 从 `max(existing)+1` 起，同一说话人在同一原簇里分出来的 turn 共用一个新簇 id（这样 `build_chunks` 仍能把相邻同人 turn 合并）。
返回的 turns 顺序不变；`cluster_embeddings` 按修正后的簇存。日志一行：`segment #N: turn-level check moved X turns (Y s)`。
配置键 `speakers.turn_check_min_s`、`speakers.turn_reject_margin` 进 DEFAULTS 与 config.example.yaml；`turn_check_min_s: 0` 关闭这层。

`relabel_segment`（pipeline.py）：改成**逐句**识别——每条 utterance 自己的音频（≥ turn_check_min_s 时）做 embedding 比对，
短句沿用同簇（cluster_label）的整簇结果；`speaker_score` 存该句自己的分数。手标（speaker_source='user' 且 speaker_id 非空）的句子仍不动。

## 3. 拆分已有的混合句子（speech/pipeline.py `Processor.split_utterance`, ui/app.py, ui/static）

```python
def split_utterance(self, utt_id: int, at: list[float] | None = None, auto: bool = False) -> dict
```

- 取这句的 segment 音频 `[start_s, end_s]`。`auto`：对这段音频跑 `engine.diarize`（片段 < 1 s 的 turn 忽略）+ §2 的 turn 级识别，
  按 (speaker 或 cluster) 把相邻同说话人的 turn 合并成 pieces；只得到 1 个 piece 时 `raise ValueError("没有分出第二个声音，请手动指定拆分点")`。
  `at`：相对这句开始的秒数列表（升序、去重、必须落在 (0.3, dur-0.3) 内，否则 ValueError），pieces = 按这些点切开。
- 每个 piece ≥ 0.6 s；短于它的并入前一个。对每个 piece：ASR（`self.asr.transcribe`，复用 `transcribe_chunks` 的 Chunk 结构，带 0.15 s padding），
  纠错（`_correct`），说话人：auto 模式用识别结果；`at` 模式每个 piece 单独比对登记声纹（≥ 1 s 才比，否则未知）。
- 事务里：删除原 utterance（tokens / vectors 级联），插入 N 条新 utterance（`segment_id` 同，`start_s/end_s` 为 piece 的语音边界，
  `abs_start` 重新算，`cluster_label` = 原 cluster_label 的 piece 0、其余用新的簇号 `max(cluster_label in segment)+k`，`speaker_source` NULL），
  `text_seg` 用 `tokenize`；之后 `_embed_new(ids)`、`cluster_embeddings` 更新（每个新簇一条）、重写当天 markdown。
- 返回 `{"removed": utt_id, "pieces": [{id, start_s, end_s, speaker, speaker_id, score, text}], "billed_s": x}`。ASR 失败（LLMError）→ 不改库，抛出。
- API：`POST /api/utterances/{id}/split` body `{"auto": true}` 或 `{"at": [1.8, 4.2]}`；需要 API key（没有 → 409 同 chat）；在 `rt.busy()` 下、线程里跑；
  ValueError → 400。`GET /api/utterances/{id}/preview-split?auto=1` 可选：只返回 auto 的 pieces 边界和每段的识别结果（不 ASR、不改库），
  前端先预览再确认。
- 前端：记录页每句的操作里加 ✂（title「拆分这句（两个人的话混在一起时）」）。弹窗：顶部 `<audio>` 播放这句；
  按钮「在当前位置拆分」把 `audio.currentTime` 加进拆分点列表（显示为 chips，可删）；按钮「自动按声音拆」调 preview-split 显示预览
  （每段：起止秒、识别到的人 / 未知、▶ 试听该段 `/api/audio/segment/{seg}?from=&to=`）；「确认拆分」→ POST split → toast「已拆成 N 句，重新转写了 X 秒」→ 刷新当天。
  拆完的句子在列表里高亮一下（已有 hl 样式）。

## 4. 其他

- `GET /api/speakers` 的 `speakers[].outlier_samples` 在说话人卡片上显示警告「有 N 个样本和其他样本不像（可能混了别人）」+ 按钮「移除不像的样本」（确认框）。
  样本列表每行显示一致性（两位小数，离群标红）。
- 声音组起名 / 标记 / 上传时遇到 SampleMismatch（400 且 `mismatch: true`）：toast 说明，并在原位置给「仍然加入」按钮（带 force 重发一次）。
- `speakers calibrate` 不变。
- 文档：README「认人」一节补一段"句子里混了两个人怎么办"（✂ 拆分）和"样本不像的提示"；DESIGN §10.3 补 turn 级识别、一致性检查、拆分；
  config.example.yaml 新键。
- 测试（新文件 `tests/test_mixed_speakers.py` + 各处追加）：check_sample / SampleMismatch / force；audit_samples 离群判定；name_voice 成员一致性筛选；
  identify_turns 的 turn 级修正（用 chirp/noise 两种"声音"：先登记 chirp 为 A，把 noise 的 turn 混进同一簇，修正后 noise turn 变成未知新簇；
  反向：未知簇里的 chirp turn 被认成 A）；relabel 逐句；split_utterance 的 at / auto（假 ASR 返回固定文本；真实引擎做 auto 的 diarize 可用
  chirp+noise 拼接的 6 s 音频）；API 端点 400/409/200；CLI audit --fix。
