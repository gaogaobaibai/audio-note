# 说话人 v3b：审查修复 + 远处声音过滤（音量门槛）

承接 docs/SPEC_speakers_v3_mixed.md（已实现）。两部分：A. v3 审查确认的问题；B. 用户新需求——"离得远、不清晰的声音直接滤掉，
达到一定强度的才保留，这个 trade off 要能调"。项目约定同前（Bash、`.venv/Scripts/python.exe`、tests/conftest.py、不碰根 config.yaml / profiles/*、不联网、不用麦克风）。

## A. 审查修复

### A1. 样本一致性不能被"多数派"绑架（speakers.py `_core_reference` / `audit_samples` / `check_sample`）

现状：核心参考 = 时长加权中位样本 + 相似的样本。问题：若混进来的样本数量多 / 时长长，它们会成为核心，干净的麦克风样本反被判离群，`--fix` 删错。修改：

1. `speaker_embeddings` 增加列 `origin TEXT`（db.py MIGRATION_COLUMNS：'mic' = `speakers record`，'file' = enroll 上传 / 文件，'label' = 记录页标记 / `speakers label`，
   'voice' = 给声音组起名，'cluster' = 过期音频用存的簇声纹；老行 NULL）。`db.add_speaker_embedding(..., origin=None)` 新参数；各调用处填上。
2. 锚点优先级：核心中位样本只在 **origin ∈ {'mic','file'}** 的样本里选（这些不可能混入别人）；一个都没有时才在全部样本里选，且时长权重封顶 `min(duration, 10 s)`。
3. 歧义判定：非核心样本若彼此两两相似度 ≥ min_consistency 且 ≥ 2 条（自成一派），则 `audit_samples` 给这一派 `ambiguous: True, outlier: False`，核心一派 `ambiguous: True, outlier: False`；
   真正的离群（和两派都不像的单个样本）仍 `outlier: True`。`check_sample` 在歧义状态下退回 "对全部样本取最大 cosine"（不拒绝）。
   API `GET /api/speakers/{sid}/samples` 行里带 `ambiguous`、`origin`；`GET /api/speakers` 的 `outlier_samples` 不计歧义样本，另给 `ambiguous_samples: n`；
   `DELETE .../samples/outliers`、`speakers audit --fix` 只删 `outlier: True` 的。
   前端：卡片上歧义时提示「这个人的样本分成两组互不相像，请打开样本列表自己判断（试听后删掉不是 TA 的）」；样本列表显示 origin（麦克风 / 文件 / 标记 / 声音组 / 簇）和一致性。
4. 测试：reviewer 的场景（4 条相似的 'voice' 样本 + 3 条相似的 'mic' 样本，两派互不相像）→ mic 派为核心，voice 派 outlier（因为锚点优先 mic）；
   没有 mic/file 样本、两派都是 'voice' → 两派 ambiguous、不删；单个离群仍删；时长封顶生效（60 s voice vs 3 条 5 s file：file 派为核心）。
   `enroll()` 只在至少加入一条样本后才写 notes（`db.upsert_speaker(name, notes)` 移到循环后）；API / CLI 的 `force` 只接受 JSON true/false/0/1，其他 400「force 应为 true/false」。

### A2. 拆分（pipeline.split_utterance / app.py）

1. 每个 piece 都必须有文字：`transcribe_chunks` 之后 `if len(spoken) != len(chunks): raise ValueError("有一段没有转写出文字（或被内容过滤），没有修改这句")`；空文字的 piece 也算失败。
2. 事务内 `DELETE FROM utterances WHERE id=? AND start_s=? AND end_s=?`，`rowcount != 1` → `raise ValueError("这句已经被修改或拆分过了，请刷新")`（事务回滚）。
3. 合并短 piece 后 `len(pieces) < 2` → `raise ValueError("拆分点太靠近两端，只剩一段，没有拆分")`（preview 也要报）。
4. 新插入的 piece 用 `id > MAX(id)`（`INSERT ... (id, ...) VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM utterances), ...)`）避免 rowid 复用；
   `/api/audio/utterance/{id}` 响应改为 `Cache-Control: no-store`（segment 接口不变）。
5. `POST /api/utterances/{id}/split` 的工作在 `rt.sched.lock` 下执行（和任务互斥；preview 不需要）。
6. 前端：自动拆失败时保留手动点（只有自动预览成功才清空手动点）；多文件上传遇到 mismatch 不中断后面的文件（每个失败的文件各自带「仍然加入」）；
   声音卡片的 mismatch 提示进 `voiceDrafts`，页面因任务刷新后仍在；标记成功的 toast 带上 `sample_note`（`samples_added === 0` 的 voices/name 同理，用 warn 颜色）。

## B. 远处 / 不清晰声音的音量门槛

用户数据实测（50 ms 帧、取能量最高的一半帧的 RMS，dBFS）：本人近讲 132 句 中位 −26 dB、p10 −30 dB；两米外的朋友 7 句 中位 −53 dB；本人偶尔侧身的轮次 −35 ~ −45 dB。
两者差 20 dB 以上，门槛放在 **−40 dBFS** 既保住本人又滤掉远处的人。系统声音走 loopback 时有 AGC（峰值 0.5），永远在门槛之上。

1. `audio_note/audio/loudness.py`（新）：`speech_dbfs(x: np.ndarray, frame_s=0.05, top_frac=0.5) -> float`（上述定义；空数组 → −120.0），`DEFAULT_MIN_SPEECH_DBFS = -40.0`。
2. 配置：`audio.min_speech_dbfs: -40`（DEFAULTS + config.example.yaml 注释：低于此音量的说话轮次视为远处 / 不清晰，不转写、不入声纹；null 关闭；记录页每句显示音量，调门槛前先看看）。
   设置页「录音」小节加数字框「最低音量 (dBFS)」+ 说明；`EDITABLE["audio"]` / `_RANGES` (−90, 0)；允许 null（空字符串 = 关闭）。
3. 流水线 `process_segment`：diarize + identify 之后、`build_chunks` 之前，对每个 turn 计算 `speech_dbfs(audio[turn])`，低于门槛的 turn 丢掉（日志一行：
   `segment #N: dropped K faint turns (X s, < -40 dBFS)`；`segments` 表加列 `faint_s REAL`（migration）记被滤掉的秒数，status 不变）。
   全部 turn 都被滤掉 → 和"没有 turn"一样走 no_speech（文件删除）。
   `utterances` 表加列 `loudness_db REAL`（migration）：每条 utterance 存其音频的 `speech_dbfs`（新处理的句子必填；`relabel_segment` / `identify_utterances` 顺便给 NULL 的补上）。
   `split_utterance` 的 piece 也算并存 loudness。
4. 声纹：`identify_turns` 的簇级 embedding 和 turn 级核对都只用门槛之上的 turn（流水线已经丢掉了，这里只需在 relabel 的逐句路径里跳过 `loudness_db < 门槛` 的句子：
   它们保持原状并打上 `speaker_score=NULL`）；`voices.group_unknown_voices` 不把 `loudness_db < 门槛` 的句子算进未识别组（它们不进样本、不进计数），
   `name_voice` 取样时也跳过它们；`unknown_utterances` 统计排除它们。
5. 已有记录：`audio-note cleanup --faint`？不做。改为提供 `audio-note loudness [--day D] [--apply]`：补算 NULL 的 `loudness_db`（读音频），打印分布（每人中位 / p10 / p90，
   低于门槛的句数），`--apply` 把低于门槛且未手标的句子标记为 `speaker_name=未知, speaker_id=NULL, speaker_score=NULL`？不——用户要的是"滤掉"：
   `--apply` 删除这些句子（utterances 行，级联 tokens / vectors），并重写当天 markdown；删除前打印将删的句数并要求 `--yes`。UI 任务 `loudness`（JOBS）只做补算 + 统计（不删除），
   结果给 `{"computed": n, "below": k, "threshold": t}`；设置页音量字段旁边加「统计音量分布」按钮显示结果。
6. 记录页：`/api/transcript` 每句带 `loudness_db`；句子行的 title / 一个小灰字 `−32 dB`（低于门槛的句子显示为淡色 + 标记「远」）。
7. 测试（`tests/test_loudness.py` 新 + 追加）：speech_dbfs（静音、正弦 −20 dB、混合帧取最响一半）；process_segment 丢弃安静 turn（FakeEngine 给两个 turn，
   一个在安静音频上）、全部安静 → no_speech、`faint_s` 写入、`loudness_db` 写入；门槛 null 关闭；relabel 补算；voices 排除安静句；CLI loudness 统计与 --apply --yes；
   设置 PUT 校验（−200 → 400，"" → null）。

## 文档
README「认人」一节加「远处的声音」小段（默认 −40 dBFS、记录页看每句音量、设置里调、`audio-note loudness`）；DESIGN §10.4 补 A1 的锚点规则与 B 的门槛；config.example.yaml。
