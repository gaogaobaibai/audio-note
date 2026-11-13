# 评测工具（scripts/bench）

一套**固定剧本 + 合成音频 + 真值**的评测工具，用来回答三类问题：转写准不准（CER、专有名词）、说话人认得准不准
（准确率 / FRR / FAR）、本机花多少 CPU。所有脚本都能单独运行，也都能对着升级前的代码跑出基线。

剧本和脚本进仓库；音频、缓存、评测档案放在 `--root` 指定的目录（默认 `$BENCH_ROOT`，否则 `<仓库>/tmp_bench`，已被 git 忽略）。
API key 只从环境变量 `MIMO_API_KEY` 或 `--key-config <某个 config.yaml>` 读进内存，不会写进任何文件，也不会打印。

## 文件

| 文件 | 作用 |
|---|---|
| `story.py` | 手写的固定剧本：3 天 × 4 场、5 个常驻角色（其中“老秦”故意不登记声纹）+ 1 个只出现在系统声音里的旁白；每场的类型、应提取的事实 / 待办 / 决定、后来被证明已完成的待办、两组相互矛盾的事实；标准词库（24 个专有名词）和 16 个数字金额 |
| `build.py` | `synth` 用 MiMo TTS 合成并缓存；`voices` 用本机声纹模型检查“同一角色同一音色”；`render` 拼成会话音频并生成声学条件（近讲 / 远讲 −20 dB + 混响 / 两人重叠 / 系统声音通道）和 `truth.json`；`import` 建一个独立的评测档案并跑完整流水线 |
| `audiofx.py` | 纯 numpy 的声学处理（重采样、响度、混响、底噪、通道电平） |
| `textnorm.py` `align.py` | CER 用的文本归一化（标点、全半角、大小写、中文 / 阿拉伯数字等价）、字符对齐、按词统计、按时间把句子对到台词 |
| `bench_cer.py` | CER（原始 vs 校对后）、专有名词错误率、校对改动的“改对 / 改坏 / 无关”，按条件 / 场次 / 天；1.0 的库还会统计提问命中率 |
| `bench_speakers.py` | `profile`：流水线实际判定的准确率 / FRR / FAR；`replay`：用存下来的向量回放任意评分函数（不跑模型）；`proxy`：对真实数据副本只出汇总 |
| `bench_cost.py` | 分离 / 声纹向量 / 判定在 1 / 2 / 4 线程下每秒音频的 CPU 秒与墙钟秒 |
| `report.py` | 把多份结果 JSON 合成一份 Markdown 表 |

## 常用流程

```powershell
$R = "D:\bench"                                   # 任意目录
$K = "C:\path\to\config.yaml"                     # 里面有 mimo.api_key 的配置文件

# 1. 一次性：合成（约 170 次 TTS 调用，其中约 45 次是较慢的克隆音色）、检查音色、拼音频
python scripts/bench/build.py synth  --root $R --key-config $K
python scripts/bench/build.py voices --root $R
python scripts/bench/build.py render --root $R                      # 变体 mixed（默认）；另有 clean / far

# 2. 导入一个评测档案并跑流水线（ASR 结果按请求内容缓存，重跑不花钱）
python scripts/bench/build.py import --root $R --name v1 --key-config $K
python scripts/bench/build.py import --root $R --name v1_refine --key-config $K --chain --set correction.mode=session
python scripts/bench/build.py import --root $R --name spk --asr stub          # 只看说话人 / 开销：不联网

# 3. 打分
python scripts/bench/bench_cer.py --profile v1 --root $R --out v1_cer.json --compare baseline/cer.json
python scripts/bench/bench_speakers.py profile --profile v1 --root $R --out v1_spk.json --compare baseline/speakers.json
python scripts/bench/bench_speakers.py replay  --profile v1 --root $R --scorers legacy,centroid,asnorm
python scripts/bench/bench_speakers.py replay  --source lines --root $R --scorer audio_note.speech.scoring:my_scorer
python scripts/bench/bench_cost.py --root $R --out v1_cost.json --compare baseline/cost.json
python scripts/bench/report.py baseline/*.json v1_*.json --out report.md
```

## 对着升级前的代码跑（基线）

脚本用哪份 `audio_note`：环境变量 `AUDIO_NOTE_SRC` > `PYTHONPATH` 里带 `audio_note` 包的目录 > 本仓库。
所以基线就是把旧提交检出到别处，再把工作目录和 `PYTHONPATH` 指过去：

```powershell
git worktree add --detach D:\old a1f36ac
cd D:\old
$env:PYTHONPATH = "D:\old"
python <本仓库>\scripts\bench\build.py import --root $R --name baseline --key-config $K
```

每份结果里都记着 `code`（代码目录、版本、提交），不会搞混。模型目录默认 `<本仓库>/models`，可用 `--models-dir` 或
`BENCH_MODELS_DIR` 指定。打分脚本一律以只读方式打开数据库，给旧档案打分不会触发迁移。

## 读数时要知道的

- **CER** 按整场对齐：一场的全部台词（按时间）对全部句子（按时间），错误记到所属台词上，所以“按条件”的数字不受切句方式影响。
  “只算转写到的台词”排除了被音量门（`audio.min_speech_dbfs`）丢掉的远讲台词。
- **专有名词错误率**：真值里每出现一次算一次，转出来不是一字不差就算错；数字先归一化（“十二万五” = “125000” = “12.5万”）。
- **改对 / 改坏**：把校对前后每一处不同，拿它在真值里对应的那段文字比较编辑距离，变近 = 改对，变远 = 改坏，不变 = 无关；
  只改标点 / 空格 / 数字写法的句子单列为“cosmetic”。
- **FRR / FAR**：FRR = 登记过的人没被认成自己（含认成别人）；FAR = 没登记的人（旁白、老秦）被认成某个登记的人。
  同时给出按句数和按时长加权两套数。
- **回放**的三种向量来源：`clusters`（库里的簇向量，带分离误差）、`utterances`（1.0 的 `utterance_voice`）、
  `lines`（按真值时间直接从音频切，每句再切 1 / 2 / 4 秒，考的是纯评分层）。评分函数的接口见 `bench_speakers.py` 顶部说明。
- **开销**受机器当时负载影响，比较时要在同样空闲的机器上各跑一遍。
- 合成语音比真实录音干净：绝对数字偏乐观，用来比较“改之前 / 改之后”，不要当成真实环境的承诺。
