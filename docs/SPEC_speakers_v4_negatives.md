# 说话人 v4：负样本、核对队列、样本影响力、系统声音分声道、页面顺序

承接 docs/SPEC_speakers_v3_mixed.md / v3b。约定同前：Python `.venv/Scripts/python.exe`，测试
`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/<file>.py`，tests/conftest.py 的 tmp 配置；
不碰根目录 config.yaml / profiles/*；不联网；不用麦克风；中文界面文案；不加第三方依赖。

## 0. 背景（用户真实数据，2026-11-08）

- 用户反馈："有时会把其他人的声音识别成自己"；要"标记负面样本"，总体提升声纹识别准确性。
- 档案里只登记了本人：3 条麦克风样本（彼此 0.95）+ 1 条从记录里标的样本（和麦克风样本只有 0.35–0.39，单一声音，用户当时强制加入）。
  识别规则是"像任一样本 ≥ 0.45 就算这个人"，于是这条样本成了侧门：自动认成本人的 961 句里有 413 句**只靠它**过线
  （音量中位 −32 dB，大多 1–2 s 短句；靠麦克风样本过线的 540 句中位 −24 dB）。
- 没有人工标注，无法离线测出"难的冒认"有多少——所以重点是：让用户低成本地给出标注（负样本、核对），并让识别马上用上这些标注。
- 用户还开会（系统声音和麦克风混成单声道录制）：对方的声音没有任何信号区分于本人。

数据库结构**已经加好**（db.py SCHEMA / MIGRATION_COLUMNS，勿重复添加）：
- 表 `speaker_negatives(id, speaker_id → speakers ON DELETE CASCADE, embedding, dim, duration_s, source_path, utterance_id, segment_id, created_at)`
- 表 `utterance_rejections(utterance_id → utterances CASCADE, speaker_id → speakers CASCADE, created_at, PK(utterance_id, speaker_id))`
- 列 `segments.channel_rms BLOB`（float16 交替的 (mic_rms, system_rms)，每个采集块一对）、`segments.channel_block_s REAL`、`utterances.system_share REAL`。

## A. 负样本与否决（db.py 辅助方法、speech/speakers.py、speech/voices.py、speech/pipeline.py、ui/app.py、cli.py）

1. DB 辅助：`add_speaker_negative(speaker_id, emb, source_path, duration_s, utterance_id=None, segment_id=None) -> int`；
   `speaker_negatives() -> list[tuple[int, np.ndarray]]`（speaker_id, 归一化 embedding）；`delete_speaker_negative(id)`；
   `reject_utterances(speaker_id, utterance_ids) -> int`（INSERT OR IGNORE 到 utterance_rejections，并把这些句子中当前 speaker_id = 该人的设为
   `speaker_id NULL, speaker_name = unknown_label 由调用方传, speaker_score NULL, speaker_source NULL`）；`rejected_speakers(utterance_ids) -> dict[int, set[int]]`。
   删除说话人时（db.delete_speaker）负样本和否决随 FK 级联删除，确认已有的 delete 路径里外键是开着的。
2. 判定规则（`SpeakerEngine.match` 及所有调用它的地方：identify_turns / identify_utterances / voices 的 closest 提示）：
   候选人 S 的正分 p = 现有规则（对 S 的正样本取最大）；负分 n = 对 S 的负样本取最大（没有负样本 = −1）。
   **只有 p ≥ 阈值 且 p − n ≥ `speakers.negative_margin`（DEFAULTS 0.0，config.example.yaml 注释）才认定为 S**；否则看下一个候选人（按 p 降序），都不行 = 未知。
   `match` 的签名保持向后兼容（新增可选参数 negatives，默认 None = 旧行为），返回的分数仍是 p。
   identify_turns / identify_utterances / relabel_segment 读一次 `db.speaker_negatives()` 传进去。
   **否决优先**：pipeline 写回识别结果时，若 (utterance, speaker) 在 utterance_rejections 里，则这句不认成该人（设为未知或下一个候选）。
   relabel_segment 的条件 UPDATE 已有 speaker_source 保护，这里再加否决检查。
3. 取负样本：`voices.reject_voice(db, cfg, engine, speaker_id, utterance_ids, whole_cluster=False) -> dict`：
   - whole_cluster 时把同片段同 cluster_label、当前也被认成该人（且不是 speaker_source='user' 手标给该人的）句子加入。
   - 把这些句子按片段拼音频（≥ 1.0 s 语音才取样；单次最多取 3 段、每段最长 30 s；音频不在了就跳过取样），embed → add_speaker_negative。
     负样本**不做一致性检查**（它本来就是"不像"的那一侧）；但若它和该人的核心参考（speakers.py 的 `_core_reference`）相似度 ≥ 0.75，
     说明用户可能点错了：仍然加入，但结果里返回 `warning: "这段声音和 X 的样本非常像（0.81），确定不是 TA 吗？"`，并且**不**把它当负样本（只做否决）。
   - 所有句子写 utterance_rejections + 清空归属（见 A1），受影响的日期重写 markdown，然后对受影响片段 relabel（新负样本可能也影响同片段其他句子）。
   - 返回 `{speaker_id, rejected: n, negatives_added: k, relabeled_changed: m, warning?}`。
4. API：
   - `POST /api/utterances/{id}/reject` body `{speaker_id?: int（默认这句当前的说话人；没有则 400）, whole_cluster?: bool=false}` → reject_voice。在 `rt.busy()` 下、线程里跑。
   - `GET /api/speakers/{sid}/negatives` → `[{id, duration_s, source_path, file, audio: bool, utterance_id, created_at, text?}]`；
     `DELETE /api/speakers/negatives/{neg_id}`；`GET /api/speakers/negatives/{neg_id}/audio`（同正样本试听的实现）。
   - `GET /api/speakers` 每人增加 `negatives: n`、`rejections: n`。
   - `POST /api/voices/reject` body `{members, speaker_id}`：声音组卡片上的「不是 X」→ 对组内所有未识别句子写否决 + 取负样本（同 A3，按成员取样）。
5. CLI：`speakers reject --at "2026-11-04 13:17:05" [--name X] [--cluster]`（按时间找句子，同 label --at 的解析）；
   `speakers negatives [NAME]` 列出；`speakers negatives --delete ID`。

## B. 核对队列（ui/app.py、speech/voices.py 或新模块 speech/review.py）

`review_queue(db, cfg, speaker_id, limit=20, offset=0) -> list[dict]`：该人**自动**认定的句子（speaker_source IS NULL），有音频，时长 ≥ 0.8 s，
按"不确定度"排序：主键 speaker_score 升序（NULL 视为最不确定），同分按时长升序；排除已被否决的。每项
`{utterance_id, day, time, text, seconds, score, loudness_db, system_share, audio}`。
- `GET /api/speakers/{sid}/review?limit=&offset=` → `{items, total}`（total = 符合条件的总数）。
- 确认：`POST /api/utterances/{id}/confirm` body `{speaker_id}` → 这句（以及可选 `whole_cluster`）设 `speaker_source='user'`（保持归属）。
  若这句语音 ≥ `speakers.min_label_sample_s` 且 check_sample 通过，再加一条正样本（origin 'label'）；不通过不报错，只在响应里 `sample_added: false`。
- 否定走 A4 的 reject。

## C. 样本影响力（speech/speakers.py audit_samples 扩展 + API）

`audit_samples` 每行新增 `influence: int`——**只靠这一条样本**才被认成该人的自动句子数：用 cluster_embeddings 近似（该人自动认定的句子按
(segment_id, cluster_label) 分组，簇 embedding 对该人所有正样本算分；去掉这一条后最大分 < 阈值的簇，其句子数累加）。没有簇 embedding 的句子不计。
纯 numpy，不加载模型。`GET /api/speakers/{sid}/samples` 行里带上 `influence`；`GET /api/speakers` 每人增加 `risky_samples`：
outlier 且 influence ≥ 20 的样本数。CLI `speakers audit` 表格加一列「单独认出」。

## D. 系统声音分声道（audio/capture.py、audio/recorder.py、audio/storage.py、speech/pipeline.py）

1. `MixedCapture.read()` 在混音时记录这一块的 `(mic_rms, system_rms)`（混音前各自的 RMS；没有系统声音的块 system_rms = 0），
   作为属性 `last_channels`；纯麦克风（MicCapture）没有这个属性。
2. Recorder：对写进片段的每一块同时记下 channel 对；片段 emit 时把对应范围的 channel 对存成 float16 数组，`finalize_segment(..., channel_rms=None, channel_block_s=None)`
   写进 segments.channel_rms / channel_block_s（读 Recorder / gate 的代码，弄清片段起止与块的对应；预滚动（pre-roll）部分也要对上）。
   导入的文件、纯麦克风录音：NULL。
3. 流水线：process_segment 对每个 turn 算 `system_share = Σsys² / (Σsys² + Σmic²)`（turn 覆盖的块）；`speakers.system_turn_share`（DEFAULTS 0.6）。
   - share ≥ 门槛的 turn **不能被认成本人**（is_self 的说话人）：识别时把本人从候选里去掉（可以被认成其他已登记的人）。
   - 每条 utterance 存 `system_share`（ASR chunk 覆盖范围的能量比）。relabel / identify_utterances 同样遵守（有 channel_rms 时）。
4. 前端：记录页句子行，system_share ≥ 0.6 时在说话人名旁加小图标 🔊（title「来自系统声音（会议 / 通话的对方）」）。

## E. 说话人页顺序与新入口（ui/static/app.js, style.css）

1. 「已登记的人」放在页面最上面（页头按钮下方），「未识别的声音」在它下面，提示卡片最后。
2. 每个说话人卡片：
   - 新按钮「核对」（有自动认定的句子时）：打开弹窗，列出 B 的队列（每行：时间、文本、分数、音量、🔊 标记、▶ 试听），每行 ✓「是 TA」/ ✗「不是 TA」；
     ✗ 时可勾「这段录音里同一个声音也一起」（whole_cluster）；处理后该行淡出，底部「再来 20 句」分页；关闭时刷新页面。
   - 负样本数显示在样本统计里（"· 3 个负样本"），样本列表弹窗/区域增加「负样本」分组：▶ 试听、删除。
   - 若 `risky_samples > 0`：卡片上醒目提示「有 N 个样本和其他样本不像，却单独认出了很多句——可能混进了别人的声音」+ 打开样本列表；
     样本列表每行显示「单独认出 N 句」，按 influence 降序时高亮 outlier。
3. 记录页句子行：当前有说话人的句子，操作区加 ✗（title「不是 TA（标为负样本）」）→ 小弹窗确认（显示这句 + ▶ + 勾「同一个声音的其他句子也一起」）→ POST reject。
   👤 标记弹窗里，当前说话人存在时增加按钮「不是 X」。
4. 声音组卡片：「可能是 X」chip 旁加「不是 X」小按钮 → POST /api/voices/reject。
5. 所有用户数据 esc()；深色模式可读；无外部资源。

## F. 文档与测试

- README「认人」一节：负样本 / 核对 / 单独认出 N 句的样本提示 / 开会时系统声音不会被认成本人；命令表新增 CLI；DESIGN §10.5 记录判定规则与本节背景数据。
  config.example.yaml 新键：negative_margin、system_turn_share。
- 测试（新文件 tests/test_negatives.py + 追加）：match 负样本规则（无负样本行为不变、p−n 边界、候选降级到第二人）；
  否决优先（relabel 不再认回）；reject_voice（whole_cluster 范围、手标保护、音频缺失只否决、相似度 ≥ 0.75 的 warning 且不入负样本、删除说话人级联）；
  review_queue 排序与过滤；confirm（加 / 不加样本）；audit influence（侧门样本 influence 大、正常样本 0）；
  MixedCapture.last_channels；recorder 片段的 channel_rms 长度与片段块数一致（假麦克风 + 假 loopback）；
  pipeline system_share 计算与"不能认成本人"；API 400/404/200；CLI reject / negatives。
