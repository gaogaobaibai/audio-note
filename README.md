# audio_note — 全天无感录音 · 说话人识别 · 文字记录 · 每日分析 · 记忆问答

一个跑在 Windows 11 上的本地常驻程序：

1. **全天无感录音**：自动选择麦克风，本地 VAD（Silero）只保留有人说话的片段，Opus 24 kbps 存储（约 11 MB / 小时语音）。也支持导入已有录音文件。
2. **自动瘦身**：定时复检，删除没有语音的片段；已转写的音频到期自动删除（文字记录永久保留）；磁盘上限可配。
3. **带身份的文字记录**：本地说话人分离（pyannote + 3D-Speaker CAM++，CPU 可跑），认识的人会被自动识别出来——不用单独录样本，在界面里听一下「未识别的声音」填上名字就行；可以标出哪位是**本人**、每个人和你的**关系**（也可以让模型从对话里推断）；MiMo `mimo-v2.5-asr` 转文字；再用 `mimo-v2.6-flash` 做同音/人名/术语纠错（只输出编辑对，有守卫，不整篇重写），保存成 Markdown。
4. **每日分析 + 推送**：每天定时用大模型总结当天内容（话题 / 决定 / 待办 / 人物 / 情绪 / 未闭环），通过 飞书机器人 / Windows 通知 / ntfy / Server酱 / PushPlus / WxPusher / 企微机器人 推送。
5. **记忆问答**：本地关键词（FTS5 + jieba）与语义向量（bge-small-zh，CPU，90 MB）混合检索，先自动预检索再让模型按需调工具追查，直接问"上周三和张三聊了什么"、"扫地机买的什么牌子"。

6. **网页界面**：`audio-note ui` 打开本机页面（只监听 127.0.0.1）：一键开始/停止录音、按天看记录并试听某句话、给"未知"的人贴名字、看日报与待办、问答、按声音给说话人起名 / 标本人 / 标关系、改设置、看日志。命令行仍然全部可用。
7. **档案**：不同的人 / 用途各有独立的录音、记录、说话人和设置（自带一个 `demo` 示例档案），界面左上角一键切换。

除了 ASR 和 LLM 调用之外，所有东西都在本地。

## 架构一览

```
麦克风 / 导入文件
   │  sounddevice 16 kHz mono
   ▼
Silero VAD 门控 ──► 只存语音片段 data/audio/YYYY-MM-DD/HHMMSS_mmm.opus ──► SQLite segments
                                                                          │  每 process_interval_min
                                                                          ▼
                       说话人分离 (pyannote seg-3.0 + CAM++ 聚类) ──► 与注册声纹比对 ──► 身份
                                                                          │
                                                                          ▼
                       按说话人/时间切 ≤60s 块 ──► MiMo ASR ──► LLM 纠错(编辑对+守卫) ──► utterances + FTS5
                                                                          │
                                       ┌──────────────────────────────────┴──────────────────┐
                                       ▼                                                     ▼
                       transcripts/YYYY-MM-DD.md                            每日 daily_time: map→reduce 日报
                       记忆问答 chat (关键词+向量混合预检索 + 工具调用循环)    daily/YYYY-MM-DD.md → 推送渠道
```

完整技术方案（选型理由、API 细节、流水线规范、数据模型、风险）见 [docs/DESIGN.md](docs/DESIGN.md)。

主要组件：

| 环节 | 实现 | 位置 |
|---|---|---|
| 采集 | `sounddevice`，16 kHz；设备不支持时按默认采样率采集再重采样 | `audio_note/audio/capture.py` |
| VAD / 分段 | Silero VAD v4 ONNX + 门控状态机（前后 0.5 s、静音 2 s 结束、最长 300 s） | `audio_note/audio/vad.py` |
| 存储 / 清理 | Opus in OGG，写入中先落 `_partial.wav` 防崩溃丢失；复检、过期、磁盘上限 | `audio_note/audio/storage.py` |
| 导入 | libsndfile 直读，其他格式走内置 ffmpeg | `audio_note/audio/importer.py` |
| 说话人 | sherpa-onnx 分离 + CAM++ 声纹注册/识别 | `audio_note/speech/speakers.py` |
| ASR / 纠错 | MiMo `mimo-v2.5-asr`；`mimo-v2.6-flash` JSON 编辑对纠错 | `audio_note/speech/asr.py` `correct.py` `pipeline.py` |
| 记忆 | SQLite FTS5 + jieba 关键词、bge-small-zh 向量（sqlite-vec）RRF 混合、相对日期解析、预检索 + function-calling 检索循环 | `audio_note/memory/` |
| 日报 / 推送 | map-reduce 日报；多渠道推送 | `audio_note/analysis/` `audio_note/push/` |
| 服务 | 录音线程 + 调度线程，单实例锁，可选托盘 | `audio_note/cli.py` `service.py` `scheduler.py` `runtime.py` |
| 网页界面 | FastAPI 本地 API + 无构建的单页前端（原生 HTML/JS） | `audio_note/ui/` |

## 安装

环境要求：Windows 10/11，Python 3.12（用 [uv](https://docs.astral.sh/uv/) 管理），CPU 即可，不需要 GPU / torch。

一键脚本（创建 `.venv`、安装依赖、下载模型、生成 `config.yaml`）：

```powershell
# 没装 uv 的话先装：
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/gaogaobaibai/audio-note.git audio_note
cd audio_note
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

手动等价步骤：

```powershell
uv venv --python 3.12
uv pip install -e .
uv pip install pytest                                     # 可选，跑测试用
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
Copy-Item config.example.yaml config.yaml
Copy-Item glossary.example.txt glossary.txt
```

`scripts/download_models.ps1` 会把三个模型放到 `models/`（已存在则跳过）：

| 文件 | 用途 | 大小 |
|---|---|---|
| `silero_vad.onnx` | 语音活动检测 | ~0.6 MB |
| `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 声纹（说话人识别） | ~27 MB |
| `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | 说话人分割 | ~6 MB |

之后用 `.\.venv\Scripts\Activate.ps1` 激活环境，就可以直接敲 `audio-note`（等价于 `python -m audio_note`）。

## 首次运行

```powershell
audio-note init             # 生成 config.yaml（若不存在）、检查模型文件、列出缺失项
audio-note ui               # 打开网页界面：填 key、选麦克风、注册自己的声音、点「开始录音」都在页面上做
```

`ui` 会在浏览器打开 `http://127.0.0.1:8765`（见下文「网页界面」）。不想用界面的话，下面是命令行的等价步骤：

1. **填 API Key**：编辑 `config.yaml`，在 `mimo.api_key: ""` 的引号里填你的 MiMo key；或者不写进文件，设置环境变量 `MIMO_API_KEY`（优先级更高）。Key 在 [MiMo 开放平台](https://mimo.mi.com/) 申请。没填 key 时只录音不转写（`status` 会提示"未配置"），填好后待处理的片段会在下一轮自动转写。
2. **选麦克风**：

   ```powershell
   audio-note devices       # 列出所有输入设备（序号 / 名称 / 接口 / 默认采样率）
   ```

   默认用系统默认输入；想指定就在 `config.yaml` 里写 `audio.device: 1` 或 `audio.device: "Realtek"`（名称子串，不区分大小写）。
3. **认人**（可以先跳过，录一阵再做）：没登记的人都显示为"未知"。不用请每个人单独来录样本——录了一阵之后打开界面的「说话人」页，
   程序会把记录里还没认出来的声音按人分组列出来（每组几句示例，可以试听），听一下是谁、填上名字点「就是 TA」即可：这组的句子立刻归到这个人名下，
   程序顺手从这些录音里提取声纹，之后新录音里再出现这个人就会自动认出来，历史记录也会用新声纹重新识别一遍（不重复转写、不花钱）。
   认错的句子在「记录」页悬停 → 👤 改一下就行，手标过的句子不会再被自动识别改掉。识别不够准时再给这个人「补充样本」（麦克风 / 上传文件）。

   - **一句话里混了两个人**（面对面聊天时常见：分离模型把两个人并成一簇，句子就切不开）：悬停那句 → ✂ 拆分。弹窗里可以「自动按声音拆」
     （对这句重新做说话人分离并逐段和登记的声纹比对，先预览每段是谁、可试听），也可以边听边点「在当前位置拆分」手动加分割点；确认后各段重新转写
     （只重转这一句，几秒钟的 ASR 费用），之后就能分别标记了。流水线本身也加了一层防护：簇级识别之后，≥ 1 秒的每个说话轮次会再单独和登记声纹核对，
     和簇里的人不符的轮次会被切出来单独成句（`speakers.turn_check_min_s`，0 关闭）。
   - **样本一致性**：从记录里取的声纹样本必须有 ≥ 2.5 秒语音（`speakers.min_label_sample_s`，太短的远场片段不可靠；句子照样会标上名字，只是不入样本），
     并且要和这个人已有的样本像（`speakers.sample_consistency_min` 0.5，以时长加权的"中位样本"为基准，两条混进来的坏样本互相作证也没用）。
     不像的会被拒绝并提示「仍然加入」；说话人卡片会标出「有 N 个样本和其他样本不像」，一键「移除不像的样本」（命令行 `speakers audit [--fix]`）。
     最好给自己既录一段麦克风样本，也从日常对话里（远场）标几句，两种环境都有样本识别才稳。
     判断"哪些样本才是本人"时以你自己录 / 上传的样本（麦克风、文件）为基准：从记录里取的样本再多也抢不走基准。一个人一条麦克风 / 文件样本都没有、
     样本又分成互不相像的两组时，程序不替你猜（卡片上会提示「样本分成两组」），打开样本列表试听后把不是 TA 的删掉即可——列表里能看到每条样本的来源。
   - **认错了人：「不是 TA」（负样本）**：把别人认成了你（或别的谁）时，在「记录」页悬停那句 → ✗「不是 TA」（可勾「同一个声音的其他句子也一起」），
     或在 👤 标记弹窗里点「不是 X」；声音组卡片上「可能是 X」旁边也有「不是 X」。这句以后**再也不会被认成 TA**（重新识别也不会认回去），
     程序还会从这段录音里取一条**负样本**：之后只有比 TA 的负样本更像 TA 的正样本（多出 `speakers.negative_margin`，默认 0）才认成 TA，
     否则换下一个像的人或记为未知。负样本在样本列表的「负样本」分组里可以试听、删除。如果这段声音其实和 TA 的样本非常像（≥ 0.75），
     程序会提醒你是不是点错了，这时只记下"这句不是 TA"，不存负样本。命令行：`speakers reject --at "时间" [--cluster]`、`speakers negatives`。
   - **核对**：说话人卡片上的「核对」把自动认成这个人、最拿不准的句子（分数最低、最短的在前）一句句列出来，▶ 听一下点 ✓「是 TA」或 ✗「不是 TA」，
     每次 20 句、「再来 20 句」翻页。✓ 会把这句锁定为 TA（以后不会被改掉），够长又像的话顺便补一条声纹样本；✗ 同上面的「不是 TA」。
     花几分钟核对一遍，比调阈值管用。
   - **「单独认出 N 句」的样本**：样本列表每条样本旁显示有多少句**只靠这一条样本**才被认成这个人。一条和其他样本不像、却单独认出很多句（≥ 20）的样本，
     多半混进了别人的声音（实际遇到过：一条从记录里强行加入的样本单独认出了 413 句，大多是又轻又短的别人的话），
     卡片上会醒目提示「有 N 个样本和其他样本不像，却单独认出了很多句」，打开样本列表试听后删掉它，再「重新识别全部记录」。
   - **开会时对方不会被认成你**：同时录系统声音时，程序给每个采集块分别记下麦克风和系统声音的音量；一段话的能量 ≥ 60%（`speakers.system_turn_share`）
     来自系统声音（会议 / 通话里的对方），就**不会被认成本人**（仍可以被认成其他登记过的人）。记录页这样的句子在说话人名旁带 🔊。
     只对打开系统声音之后的新录音生效，导入的文件和只录麦克风的录音没有这个信息。
   - **远处的声音**：离麦克风远、听不清的人（两米外的朋友、隔壁的电视）声纹分不准、转出来也是半句，默认直接滤掉：说话音量低于
     `audio.min_speech_dbfs`（默认 **−40 dBFS**；本人近讲大约 −26，两米外的人大约 −53）的轮次不转写、不入声纹、不算进未识别的声音。
     记录页每句后面有一个小灰字的音量（低于门槛的句子显示为淡色并标「远」），调门槛前先看看自己和别人各在什么水平；
     门槛在「设置 → 录音 → 最低音量」改，留空即关闭，旁边的「统计音量分布」给出每人的中位 / p10 / p90 和低于门槛的句数。
     门槛出现之前已经转写的远处句子：`audio-note loudness` 补算音量并打印分布，`audio-note loudness --apply --yes` 删掉低于门槛的句子
     （手动标过说话人的不删），重写当天记录。系统声音（loopback）自带 AGC，永远在门槛之上。

   - **本人**：把自己标成本人（卡片上「设为本人」，或给声音起名时勾「这是我自己」）。日报、推送和问答都站在本人视角——记录里本人的话就是"我"说的，待办里负责人是本人的就是你自己的待办。
   - **关系**：其他人可以标上和你的关系（朋友 / 好朋友 / 家人 / 同事 / 领导 / 导师 / 客户 …，也可自定义），不强制；点「自动推断关系」或每次日报之后，模型会根据对话（称呼、谈的事）给没标过的人推断一个，
     带「自动」标记，手动标过的不会被覆盖。关系会进日报和问答的提示，模型就知道"李总"是领导、"小雨"是朋友。

   命令行也能做同样的事：

   ```powershell
   audio-note speakers voices                          # 列出未识别的声音组（句数 / 时长 / 示例）
   audio-note speakers voices --assign 1 张三 --relation 同事   # 给第 1 组起名（--self 表示是本人）
   audio-note speakers set-self 我                      # 标本人；speakers relation 张三 领导 标关系
   audio-note speakers relations                        # 让模型推断关系
   audio-note speakers record 我 --seconds 20           # 老办法：对着麦克风念 20 秒注册
   audio-note speakers label --at "2026-10-31 08:19:32" 张三    # 按记录里的时间标一句话
   audio-note speakers reject --at "2026-11-04 13:17:05" --cluster  # 这句（和同一个声音）不是当前的说话人
   audio-note speakers negatives 我                     # 看某人的负样本；--delete ID 删除
   ```

   识别阈值 `speakers.identify_threshold` 默认 0.45，`audio-note speakers calibrate` 会用声纹样本之间的相似度给出建议值；
   未识别声音的分组阈值 `speakers.voice_group_threshold` 默认 0.55（越高分得越细，同一个人被分成两组时都起同一个名字即可）。
4. **启动**：

   ```powershell
   audio-note run            # 录音 + 定时处理 / 清理 / 日报 + 网页界面（不自动开浏览器），Ctrl+C 退出
   audio-note run --tray     # 同上，带系统托盘图标（托盘菜单里有「打开界面」）
   ```

   第一次推荐先 `audio-note record` 录几分钟，然后 `audio-note process` 手动处理一次，看看 `data/transcripts/` 下的结果，再长期 `run`。

## 网页界面

```powershell
audio-note ui                 # 启动界面并打开浏览器；不自动录音，录音在页面上点「开始录音」
audio-note ui --record        # 启动时就开始录音（等于 run --open）
audio-note run                # 常驻服务也带界面：浏览器打开 http://127.0.0.1:8765
```

| 页面 | 能做什么 |
|---|---|
| 总览 | 录音状态（是否有人在说话、本次录了多少、系统声音开关）、今日语句 / 待转写 / 磁盘占用、还有多少句没认出是谁、后台任务一键运行（立即转写、生成日报、清理、重新识别说话人、重建索引）、最近的记录、失败片段重试 |
| 记录 | 按天看转写内容（按对话时段分组、说话人着色、本人带「我」标记、可按人筛选）；悬停一句话可以 ▶ 试听、👤 标记是谁说的（同一段录音里这个声音的句子一起标，补充声纹并重新识别那段录音；新名字可顺手标本人 / 关系）、✗ 不是 TA（否决并存负样本）、✂ 拆分（一句里混了两个人时：自动按声音拆或手动点分割点，预览试听后重新转写）、↕ 看前后文；来自系统声音的句子带 🔊；顶部搜索框跨天检索（和问答用同一套混合检索） |
| 日报 | 结构化查看每天的概览 / 话题 / 决定 / 待办 / 人物 / 时间线，重新生成、推送；「记忆笔记」页看日报抽出的待办 / 决定 / 人物并可删除 |
| 问答 | 对话式提问，显示检索过程（调了哪些工具）、模型调用次数和耗时；可临时切换策略 / 检索方式 |
| 说话人 | 最上面是已登记的人：本人标记、关系（手动或「自动推断关系」）、「核对」自动认定的句子（✓ / ✗）、补充样本（麦克风 / 上传）、试听样本（含每条「单独认出 N 句」和负样本）、重命名、删除；下面是「未识别的声音」按人分组列出（示例句试听、"可能是谁"提示和「不是 X」），填名字 / 勾本人 / 选关系一键起名；校准阈值并一键采用、重新识别全部记录 |
| 设置 | 改 API key、麦克风、阈值、保留天数、问答策略、日报时间、推送渠道等（保存即生效，改音频设置会自动重启录音；写回 `config.yaml` 时保留你的注释）；测试 API / 测试推送；编辑词表；导入录音文件 |
| 日志 | 实时看 `audio_note.log`，可过滤 |

### 开会 / 通话：把对方的声音也录下来

默认只录麦克风。线上会议、语音通话里对方的声音是从扬声器 / 耳机放出来的，麦克风最多只能隔着外放听到一点。打开**系统声音**后，程序用 Windows 的 WASAPI loopback 直接抓系统正在播放的声音，和麦克风混成一路录制，对方说的话会和你自己的话一起进记录、分离出说话人、可以注册和识别。

- 开关：总览页录音状态下面的「系统声音：开 / 关」按钮（立即生效，录音会自动重启），或 `config.yaml` 的 `audio.system_audio: true`；设置页可以选录哪个输出设备（默认当前的默认输出）和音量倍数。
- 系统音量调得很小也没关系：抓到的系统声音会自动增益到正常电平。
- 对方的声音不会被认成你：程序分别记下每一小段里麦克风和系统声音的音量，主要来自系统声音的话（≥ 60%，`speakers.system_turn_share`）不会认成本人，记录页标 🔊。
- **戴耳机效果最好**。外放时对方的声音会被麦克风和系统声音各录一次（略有延迟的叠加），转写一般仍可用但会差一些。
- 只在 Windows 可用，依赖 `PyAudioWPatch`（`uv pip install -e .` 会装上）；不可用时照常只录麦克风，总览页会提示原因。`audio-note devices` 会列出可抓取的输出设备。

### 档案：把不同的人 / 用途分开

每个**档案**有自己独立的录音、文字记录、日报、说话人声纹和设置，互不混淆。安装后自带一个 `demo` 档案（3 天合成的对话，随时可删），自己用就另建一个，例如「我」：

```powershell
audio-note profile list                  # 列出档案及各自的记录数
audio-note profile create 我 --use       # 新建 profiles/我/ 并切换过去
audio-note profile use demo              # 切换当前档案（界面左上角也能切）
audio-note --profile demo status         # 只这一条命令用某个档案
audio-note profile delete demo           # 删除档案及其全部数据（会要求输入档案名确认）
```

- 目录：`profiles/<名字>/config.yaml` + `profiles/<名字>/data/`；根目录的 `config.yaml` + `data/` 是「默认」档案。
- 设置分两层：根目录 `config.yaml` 是**所有档案共享**的（API key、模型名、推送渠道写一次就行），`profiles/<名字>/config.yaml` 只放这个档案不同的项（新建时只有数据目录和它自己的词表 `profiles/<名字>/glossary.txt`）。界面「设置」页保存到当前档案的文件，而且只写改动过的项，没改的继续继承。`--copy-from` / 界面的「复制当前档案的设置」只复制普通设置和词表，不复制密钥和路径。
- 界面左上角的档案按钮可以切换、新建、删除；切换是即时的（录音若在进行会在新档案上继续）。`profiles/current` 记住上次选的档案，之后的命令和服务都用它。
- 一个档案同一时间只能被一个进程录音 / 处理；用 `--config` 指定的自定义配置文件不属于任何档案。

界面只监听本机（`ui.host: 127.0.0.1`，没有登录），不要改成 `0.0.0.0` 暴露到局域网。同一数据目录只能有一个进程在录音 / 处理：`run` 已经在跑时再开 `ui`，页面是只读的（能看记录、问答，不能录音和跑任务），请在 `run` 那个进程的界面里操作（`--tray` 的托盘菜单或直接访问 8765 端口）。

## 命令一览

全局选项（放在子命令前面）：`--profile NAME` 用某个档案；`--config PATH` 指定任意一个 config.yaml（默认取环境变量 `AUDIO_NOTE_CONFIG`，再按档案规则：`AUDIO_NOTE_PROFILE` → `profiles/current` → 根目录 `config.yaml`），`-v` / `--verbose` 打 DEBUG 日志。例如 `audio-note --profile demo -v status`。

| 命令 | 作用 |
|---|---|
| `audio-note init` | 生成 `config.yaml`、检查模型文件、打印缺失项 |
| `audio-note devices` | 列出音频输入设备 |
| `audio-note record` | 只录音（VAD 分段落盘），不做处理 |
| `audio-note run [--tray] [--no-ui] [--open] [--port N]` | 常驻服务：录音 + 定时处理 / 清理 / 日报推送 + 网页界面；`--tray` 显示托盘图标，`--open` 启动时打开浏览器，`--no-ui` 不提供界面 |
| `audio-note ui [--record] [--port N] [--no-browser]` | 网页界面（后台处理照常跑，录音在页面上开始）；`--record` 启动即录音 |
| `audio-note profile list\|create <名字> [--use] [--copy-from X]\|use <名字>\|delete <名字> [-y]` | 档案管理（见「档案」一节）；全局 `--profile <名字>` 让单条命令用某个档案 |
| `audio-note status` | 片段 / 转写统计、磁盘占用、服务是否在跑 |
| `audio-note import <files...> [--start "2026-10-31 09:00"]` | 导入已有录音（wav/mp3/m4a/… 任意格式），按 VAD 切成片段入库；`--start` 指定录音开始时间，否则从文件名或修改时间推断 |
| `audio-note process [--retry-failed] [--limit N]` | 立即处理待转写片段（分离 → ASR → 纠错 → 写记录）；`--retry-failed` 重试失败的 |
| `audio-note transcribe <file>` | 一次性转写一个文件并打印，不入库 |
| `audio-note speakers record <name> [--seconds 20]` | 用麦克风现场录一段样本并注册（注册自己最方便；样本存到 `data/speakers/<name>/`） |
| `audio-note speakers enroll <name> <wav...> [--notes "备注"]` | 用样本音频注册说话人（可多次追加样本；接受的样本会复制一份到 `data/speakers/<name>/`） |
| `audio-note speakers list` | 列出已注册说话人及样本数 |
| `audio-note speakers label --at "2026-10-31 08:19:32" <name> [--dur s]` | 把记录里某句话（按时间找到对应片段与区间）标记为某人，追加为其声纹样本，并立即重新识别该片段 |
| `audio-note speakers label <segment_id\|path> <name> --from s --to s` | 同上，手动指定片段（记录文件的 `##` 标题里有 `#id` 和文件名）和片段内的秒数区间 |
| `audio-note speakers relabel [--day YYYY-MM-DD] [--segment ID]` | 用当前声纹库重新识别已转写片段的说话人（只做声纹比对，不重复 ASR、不花钱）；注册新人之后跑一次让历史记录也带上名字 |
| `audio-note speakers calibrate` | 用已注册样本计算相似度分布，建议识别阈值 |
| `audio-note speakers voices [--assign N\|编号 <name> [--self] [--relation R] [--force]]` | 列出记录里未识别的声音组（先自动分析没扫描过的片段）；`--assign` 给第 N 组（或表里的编号）起名：句子直接归到此人、从这些录音提取声纹、重新识别历史记录；`--force` 跳过样本一致性检查 |
| `audio-note speakers audit [--fix]` | 检查每个人的声纹样本是否彼此一致（以麦克风 / 文件样本为基准，混进别人声音的样本会标为离群；没有基准又分成两组时标「歧义」，不自动删；「单独认出」列 = 只靠这条样本才认成此人的句子数，离群又单独认出很多句的多半混进了别人）；`--fix` 删除离群样本 |
| `audio-note speakers reject --at "2026-11-04 13:17:05" [--name NAME] [--cluster]` | 某句话认错了人：标为「不是 TA」（默认否决这句当前的说话人，`--name` 指定别人）——这句以后不再认成 TA，从这段录音取一条负样本，并重新识别该片段；`--cluster` 把这段录音里同一个声音、也被认成 TA 的句子一起（手标过的不动） |
| `audio-note speakers negatives [NAME] [--delete ID]` | 列出负样本（全部或某人的：id / 时长 / 来源 / 那句话）；`--delete ID` 删除一个 |
| `audio-note loudness [--day D] [--apply --yes]` | 每句音量（dBFS）分布：补算缺的音量，按人打印中位 / p10 / p90 和低于 `audio.min_speech_dbfs` 的句数；`--apply --yes` 删除低于门槛、没手动标过的句子并重写当天记录 |
| `audio-note speakers scan` | 只做扫描：给已转写但还没分析过声音的片段提取每个声音的声纹（界面打开说话人页时会自动做） |
| `audio-note speakers set-self <name>` | 把某人标为本人（日报 / 推送 / 问答的视角） |
| `audio-note speakers relation <name> <关系\|-> [--note "备注"]` | 标注某人和本人的关系（`-` 清空）；手动标的不会被自动推断覆盖 |
| `audio-note speakers relations [--force] [--speaker NAME]` | 让模型根据对话推断每个人和本人的关系（语句数 ≥ `speakers.relation_min_utterances`，置信度 ≥ `relation_min_confidence` 才写入；日报之后也会自动做） |
| `audio-note daily [--day YYYY-MM-DD] [--push] [--force]` | 生成某天日报（默认今天）；`--push` 推送；`--force` 已存在也重新生成 |
| `audio-note push-test "文本"` | 向所有已配置渠道发一条测试消息 |
| `audio-note chat [question] [--strategy prefetch\|agentic\|rag] [--retrieval hybrid\|vector\|fts] [--expand]` | 记忆问答；不带参数进入交互式 REPL |
| `audio-note reindex [--no-vectors]` | 重建关键词索引（改了词表/人名后）并补齐向量索引 |
| `audio-note cleanup` | 立即执行一次清理（复检无语音片段 / 过期音频 / 磁盘上限） |
| `audio-note doctor` | 只读体检：配置、数据目录、数据库版本、模型、API key、ffmpeg、卡住的片段、最近的错误日志 |
| `audio-note speakers rescore` | 按现在的声纹库和评分规则重判历史记录的说话人（只用存下来的声纹，几秒钟，不读音频；手工标过的不动） |
| `audio-note speakers scoring` | 查看现在是怎么认人的：评分方式、每个人学到的声音中心、确定 / 可能 / 未知各多少句 |
| `audio-note lexicon list\|add\|alias\|import\|export\|apply` | 词库：人名 / 项目 / 术语的规范写法，以及它们常被听错成什么；`apply` 按词库修正以前的记录（不加 `--yes` 只预览） |
| `audio-note inbox list\|answer\|dismiss` | 待确认：系统拿不准、想问你的事 |
| `audio-note memory items\|todos\|people\|sessions\|forget` | 记忆条目、待办、人物、某天的会话；`forget "内容"` 从记录、音频、索引和记忆里一起删掉一段内容（默认只预览，加 `--yes` 才删） |

调度参数：`pipeline.process_interval_min`（默认 10 分钟处理一次新片段）、`storage.cleanup_interval_min`（30 分钟）、`analysis.daily_time`（22:30 跑日报并推送；跑之前会先处理一遍新片段）。22:30 之后还在说话的内容会在第二天补进前一天的日报并重新推送一次；某天的片段还没处理完（没填 key、API 故障）时不会把那天标成"已完成"，处理完后自动补报。

## 数据目录

默认档案在项目下的 `data/`，其他档案在 `profiles/<名字>/data/`（`paths.data_dir` 可改成绝对路径，例如放到 D 盘）：

```
data/
  audio/YYYY-MM-DD/HHMMSS_mmm.opus   语音片段（文件名 = 本地开始时间）；写入中为 *_partial.wav
  transcripts/YYYY-MM-DD.md          当天文字记录（按片段分节，每行 时间 + 说话人 + 内容）
  daily/YYYY-MM-DD.md                每日分析报告
  speakers/<name>/*.wav              说话人注册样本的 16 kHz 副本（enroll / label 接受的部分），原文件删了也能重新提取声纹
  audio_note.sqlite3                 片段 / 转写 / utterances(FTS5) / 说话人声纹 / 日报 / 记忆笔记
  audio_note.log                     滚动日志
  service.lock                       单实例锁（pid）
models/                              本地模型（只读）
config.yaml                          你的配置（含密钥，已在 .gitignore），同时是所有档案的共享设置
profiles/<名字>/config.yaml          某个档案与共享设置不同的项（通常只有 data_dir）
profiles/<名字>/data/                该档案的数据目录（结构同上）；profiles/current 记录当前档案
glossary.txt                         纠错词表，每行一个人名 / 术语
```

文字记录示例（`transcripts/2026-10-31.md`）：

```
# 2026-10-31 记录

## 08:19:10–08:24:03 (#123 081910_250.opus)
- **08:19:10 张三**：今天我们先过一下……
- **08:19:32 未知**：好的。
```

`##` 标题 = 片段的起止时间、数据库 id 和音频文件名，`speakers label` 可以直接引用。

### 存储开销

- 只存有人说话的部分：Opus 24 kbps ≈ **11 MB / 小时语音**。一般办公日 3–4 小时有效语音 ≈ 40 MB/天，一个月 ≈ 1.2 GB。
- `storage.keep_audio_days`（默认 30）天后删除**已转写**（以及一直失败）的音频，文字记录与数据库永久保留；`storage.max_disk_gb`（默认 20）超出后从最旧的删。
- 落盘时就过滤：总语音 < `min_speech_total_s` 的片段不存；≥ 10 s 的片段语音占比 < `min_speech_ratio` 也不存（电视 / 空调让 VAD 半开半合的那种）。每 `cleanup_interval_min` 再复检一次没量过或占比偏低的片段。
- 想更省：`audio.opus_bitrate: 16000`（约 7 MB/小时，ASR 仍可用）。

### 费用（MiMo，2026-10 价目）

ASR ¥0.5 / 小时音频；纠错 + 日报 + 问答走 `mimo-v2.6-flash`（¥1 / ¥2 每百万 tokens），每天几分钱。追求日报质量可把 `mimo.analysis_model` 改为 `mimo-v2.6-pro`。

## 隐私说明

- **发到云端的只有语音块**：说话人分离后切出的 ≤60 s 片段（wav base64）发给 MiMo ASR；转写后的文本发给 MiMo 做纠错 / 日报 / 问答。
- 其余全部本地：原始音频文件、VAD、说话人分离与声纹、声纹库、数据库、全文索引、日报文件。没有任何遥测。
- 推送渠道只会收到日报文本（或你用 `push-test` 发的内容）。
- `config.yaml` 含 API key 与 webhook 密钥，已在 `.gitignore` 中；把 key 放在环境变量 `MIMO_API_KEY` 更稳妥。
- 录别人说话请先征得同意。

## 推送渠道配置

在 `config.yaml` 的 `push.channels` 里列出要用的渠道（可多个），再填对应字段。配置完用 `audio-note push-test "你好"` 验证。

| 渠道 | 配置步骤 |
|---|---|
| **toast**（默认） | Windows 系统通知，零配置；只在本机登录时可见 |
| **feishu** 飞书自定义机器人 | 飞书客户端打开一个群（可以自己建一个只有自己的群）→ 右上角 **群设置** → **群机器人** → **添加机器人** → **自定义机器人** → 起个名字 → 复制 **Webhook 地址** 填到 `push.feishu.webhook`。安全设置若勾选"签名校验"，把密钥填到 `push.feishu.secret`；若用"自定义关键词"，关键词里加上"每日记录"。个人版飞书若没有该入口，改用 ntfy / toast |
| **ntfy** | 装手机 App（[ntfy.sh](https://ntfy.sh)，iOS/Android），订阅一个随机主题名（如 `audio-note-8f3k2`），把主题填到 `push.ntfy.topic`；自建服务器则改 `push.ntfy.server`。任何知道主题名的人都能订阅，主题名要随机 |
| **serverchan** Server酱 Turbo | 打开 [sct.ftqq.com](https://sct.ftqq.com)，微信扫码登录 → **SendKey** 页复制 key 填到 `push.serverchan.sendkey` → **消息通道** 页绑定微信服务号（"方糖"公众号）。免费额度每天 5 条 |
| **wxpusher** | 打开 [wxpusher.zjiecode.com](https://wxpusher.zjiecode.com)。简单方式：**极简推送** 页扫码获取 **SPT** 填 `push.wxpusher.spt`。或者创建应用拿 **appToken** 填 `push.wxpusher.app_token`，关注应用后在"用户管理"复制你的 **UID** 填到 `push.wxpusher.uids: [UID_xxx]` |
| **pushplus** | 打开 [pushplus.plus](https://www.pushplus.plus)，微信扫码登录 → **一对一推送** 页复制 **token** 填 `push.pushplus.token`；需关注 pushplus 公众号接收 |
| **wecom** 企业微信群机器人 | 企业微信群 → 右上角 **…** → **群机器人** → **添加** → 复制 **Webhook 地址** 填 `push.wecom.webhook`。消息按 markdown 发送，上限 4096 字节 |

个人微信没有官方推送通道；想在微信里收到，用 Server酱 / WxPusher / PushPlus（都是走公众号）。

## 开机自启（任务计划程序）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1          # 注册：登录时后台启动
powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1 -Tray    # 带托盘图标
powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1 -Remove  # 取消
```

脚本执行的是 `schtasks /create /sc onlogon /tn AudioNote /tr "<项目>\.venv\Scripts\pythonw.exe -m audio_note run"`：`pythonw.exe` 没有控制台窗口，日志看 `data/audio_note.log`，或者直接在浏览器打开 `http://127.0.0.1:8765`（后台服务自带网页界面）。常用操作：

```powershell
schtasks /run   /tn AudioNote     # 立即启动
schtasks /query /tn AudioNote     # 查看状态
schtasks /end   /tn AudioNote     # 停止
```

服务有单实例锁（`data/service.lock`，记录 pid 和进程创建时间，重启后 pid 被别的程序复用也不会误判），重复启动会直接退出。笔记本用电池时如果任务不启动，在"任务计划程序"里编辑该任务的"条件"页，取消"只有在计算机使用交流电源时才启动"。

## 常见问题

**找不到输入设备 / `no input device matching`**
`audio-note devices` 看实际名字和序号；`audio.device` 填序号最稳。设备列表里同名会有 MME / DirectSound / WASAPI 多条，程序会优先选 WASAPI。确认 Windows **设置 → 隐私和安全性 → 麦克风** 允许桌面应用访问。

**日志里出现 `16 kHz capture failed ... falling back to N Hz + resample`**
设备不支持 16 kHz 直采（常见于 44.1/48 kHz 的 USB 麦或蓝牙耳机），程序自动按设备默认采样率采集再重采样到 16 kHz，功能不受影响，只是多一点 CPU。

**HTTP 429 / `rate limit`**
MiMo 限制 100 RPM / 10K TPM，客户端默认 `mimo.rpm: 90` 并对 429 指数退避重试 `max_retries` 次。如果导入了大量历史录音，`process --limit 20` 分批跑，或把 `mimo.rpm` 调到 60。

**402** 账户余额不足；**421** 内容被过滤（只跳过那个 ≤60 s 的块，片段其余部分照常入库）；**503** 服务过载，会自动重试。429 的退避是 10 / 20 / 40 / 60 s（并遵守 `Retry-After`），三次重试能跨过一个按分钟计的配额窗口；400 / 413 / 422 之类的确定性错误不重试。

**从旧版本升级**
拉取新代码、`uv sync` 之后照常启动即可。新版第一次打开数据库时会自动升级结构，升级前先把数据库备份到 `data/backups/`（保留最近 3 份）；想退回旧版本，停掉服务后把备份文件复制回 `data/audio_note.sqlite3` 即可。`audio-note doctor` 可以看数据库版本和备份。

**服务被强杀 / 断电后片段卡在 `processing`**
下次处理时会把超过 10 分钟没动静的 `processing` 片段自动放回 `new` 重新处理，不用手动改库。

**笔记本睡眠 / 拔掉麦克风后不录了**
录音线程会按每个音频块的真实时间校准时间戳（睡醒后的片段时间是对的，不会落到前一天）；输入流 10 s 没数据或设备失效时会自动重开设备（间隔 30 s），拔掉再插回来也能接上。

**所有人都是"未知"**
还没有认人：打开界面「说话人」页给列出来的声音起名（或 `speakers voices --assign`）。起了名还是认不出：`speakers calibrate` 看建议阈值；远场 / 跨设备录音相似度会偏低，可把 `speakers.identify_threshold` 调到 0.35–0.4 试试；同一个人被分成好几组声音就把 `speakers.voice_group_threshold` 调低一点（或都起同一个名字）。

**别人被认成了我**
在记录页对那句点 ✗「不是 TA」（勾上同一个声音一起），或用说话人卡片的「核对」过一遍；看看样本列表里有没有「单独认出」很多句又标离群的样本，有就删掉再重新识别全部记录。开会时打开系统声音，对方的话不会被认成你。还不够就把 `speakers.negative_margin` 调到 0.05–0.1（要比负样本像得更多才认）。

**录到了但没有文字**
`audio-note status` 看片段状态：`new` 等待处理（`pipeline.min_segment_age_s` 之后、每 `process_interval_min` 一轮）；`failed` 用 `process --retry-failed` 重试并看日志；`no_speech` 是复检后被判无语音删除了。

**纠错改得太多**
守卫会丢弃改动超过 30% 或改了数字的编辑；原文 `text_raw` 始终保留在数据库。想完全关掉：`correction.enabled: false`。

**import 报 ffmpeg 错误**
非 wav/flac/ogg/mp3 的格式走 `imageio-ffmpeg` 自带的 ffmpeg；首次使用会解压二进制，杀毒软件拦截的话加白名单。

## 记忆问答：架构与评测

记录分三层：原始层 `utterances`（每句一条，时间 + 说话人 + 文本）→ 摘要层（日报 + `memory_notes` 决定/待办/人物）→ 检索层。检索层是可插拔的，`config.yaml` 的 `memory` 段切换：

| 配置 | 选项 | 说明 |
|---|---|---|
| `retrieval` | `fts` / `vector` / **`hybrid`** | 关键词（FTS5 + jieba）/ 本地句向量（bge-small-zh，CPU，首次下载 90 MB）/ 两者 RRF 融合 |
| `strategy` | `agentic` / **`prefetch`** / `rag` | 只靠工具迭代检索 / 先自动预检索再按需调工具 / 只用预检索、单次调用 |
| `query_expansion` | `true` / **`false`** | 检索前让模型补同义词、品牌名（每次多 1 次调用） |

默认值是用 `scripts/memory_bench.py` 在 3 天合成数据（`scripts/fake_days.py`）上的 24 道题（含改述、多跳、相对日期、否定题）选出来的：

**检索层**（前 10 条里找到足够证据的题数，共 23 题）：关键词 10 → 关键词 + 查询扩展 12 → 向量 17 → 混合 17。向量把"扫地机→科沃斯""回退→回滚"这类改述问题的召回翻倍；混合检索在此基础上保留人名、数字、代号的精确匹配；查询扩展收益小且每题多 2–5 s，默认关。

**问答层**（24 题，mimo-v2.6-flash，混合检索）：

| 策略 | 答对 | 总耗时 | LLM 调用 | 工具调用 | prompt token |
|---|---|---|---|---|---|
| rag（预检索 + 单次调用） | 22 | 190 s | 24 | 0 | 3.3 万 |
| **prefetch（默认）** | 24 | 325 s | 50 | 42 | 17.4 万 |
| agentic（只靠工具） | 23 | 579 s | 73 | 90 | 22.5 万 |

prefetch 的预检索会把问题里的相对日期（"昨天晚上""上周三"）解析成日期范围优先检索，并把命中的记忆笔记一起放进上下文；加上这两步之前它是 22/24（rag 20/24），"昨天晚上聊了什么"和"李强几点到公司"（答案在前一天的"明天我十点到"，要按说话人追查）靠工具补查答对。rag 最省，适合简单事实查询；agentic 最贵，单题可能跑十几次工具。模型回答有随机性，同一配置两次跑会差 1–2 题。换数据集复测：

```powershell
$env:AUDIO_NOTE_CONFIG="config.yaml"; .\.venv\Scripts\python.exe scripts\memory_bench.py retrieval
.\.venv\Scripts\python.exe scripts\memory_bench.py chat --strategies rag,prefetch,agentic
```

## 开发

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q      # 测试不联网，MimoClient 均为 mock；向量用无模型的哈希嵌入
```

没下载 `models/` 时，依赖模型的测试会跳过。依赖真实语音的测试需要一段录音（16 kHz 单声道 wav，前 29 秒静音、之后有人说话），放到 `tests/data/sample.wav` 或用环境变量 `AUDIO_NOTE_TEST_WAV` 指定；没有就跳过。录音文件不进仓库。

代码：Python 3.12，类型标注，`logging.getLogger(__name__)`，UTF-8 / LF（见 `.editorconfig`）。
