// What the settings page shows: sections, fields, labels, help texts, which fields are "advanced".
// The server owns which keys are editable (GET /api/config -> values) and validates ranges on PUT; a key the
// server offers that is not listed here still appears, as a plain field under the section's 高级 (see extraFields),
// so new backend options are usable before this file knows about them. No DOM access.
//
// field: {path: "section.key[.sub]", label, type?: "text" | "number" | "select" | "bool" | "secret" | "time",
//         help?, placeholder?, min?, max?, step?, options?: [[value, label]], dyn?: "devices" | "sysDevices",
//         nullable?: true (empty number = null = off), advanced?: true, wide?: true}

export const PUSH_CHANNELS = [["toast", "Windows 通知"], ["feishu", "飞书机器人"], ["wecom", "企业微信机器人"], ["serverchan", "Server酱"], ["pushplus", "PushPlus"], ["wxpusher", "WxPusher"], ["ntfy", "ntfy"]];

export const SECTIONS = [
  {
    id: "mimo", title: "MiMo 接口", icon: "key", desc: "转写、纠错、问答和日报都通过它。", action: { id: "test-api", label: "测试连接" },
    fields: [
      { path: "mimo.api_key", type: "secret", label: "API key", help: "在 MiMo 开放平台申请；也可以用环境变量 MIMO_API_KEY。", placeholder: "sk-…", wide: true },
      { path: "mimo.chat_model", label: "纠错 / 问答模型", help: "mimo-v2.6-flash 便宜；mimo-v2.6-pro 更强。" },
      { path: "mimo.analysis_model", label: "日报模型", help: "留空 = 和上面一样。" },
      { path: "mimo.asr_language", type: "select", label: "转写语言", options: [["auto", "自动"], ["zh", "中文"], ["en", "英文"]] },
      { path: "mimo.base_url", label: "接口地址", advanced: true },
      { path: "mimo.asr_model", label: "语音转写模型", advanced: true },
      { path: "mimo.asr_concurrency", type: "number", label: "转写并发数", min: 1, max: 32, step: 1, advanced: true },
    ],
  },
  {
    id: "audio", title: "录音", icon: "mic", desc: "保存后会自动重启录音。",
    fields: [
      { path: "audio.device", type: "select", dyn: "devices", label: "麦克风", help: "拔插设备后重新打开这页会刷新列表。" },
      { path: "audio.gain", type: "number", label: "输入增益", help: "录得太小声就调大，例如 2。", min: 0.05, max: 50, step: 0.1 },
      { path: "audio.min_speech_dbfs", type: "number", nullable: true, label: "最低音量（dBFS）", min: -90, max: 0, step: 1, placeholder: "留空 = 关闭", wide: true, extra: "loudness",
        help: "低于这个音量的说话视为远处 / 不清晰的声音：不转写、不入声纹。本人近讲大约 −26 dB，两米外的人大约 −53 dB，默认 −40；留空 = 关闭。" },
      { path: "audio.system_audio", type: "bool", label: "同时录系统声音", help: "把扬声器 / 耳机里放出来的声音（会议、通话的对方）和麦克风一起录。用耳机效果最好，外放时对方的声音会被录两次。" },
      { path: "audio.system_device", type: "select", dyn: "sysDevices", label: "录哪个输出设备" },
      { path: "audio.system_gain", type: "number", label: "系统声音音量倍数", help: "对方声音太小就调大。", min: 0.05, max: 50, step: 0.1 },
      { path: "audio.vad_threshold", type: "number", label: "语音检测阈值", help: "0–1，越高越保守（环境嘈杂时调高）。", min: 0, max: 1, step: 0.05, advanced: true },
      { path: "audio.min_silence_s", type: "number", label: "结束片段的静音秒数", min: 0.2, max: 60, step: 0.1, advanced: true },
      { path: "audio.max_segment_s", type: "number", label: "单个片段最长秒数", min: 10, max: 3600, step: 10, advanced: true },
      { path: "audio.min_speech_s", type: "number", label: "最短语音秒数", help: "比这更短的声音不保存。", min: 0, max: 10, step: 0.1, advanced: true },
      { path: "audio.codec", type: "select", label: "音频保存格式", options: [["opus", "opus（最省空间）"], ["flac", "flac（无损）"], ["wav", "wav"]], advanced: true },
    ],
  },
  {
    id: "storage", title: "存储", icon: "disk", desc: "文字永远保留；这里只管音频文件。",
    fields: [
      { path: "storage.keep_audio_days", type: "number", label: "音频保留天数", help: "到期删除已转写的音频；0 = 永久。", min: 0, step: 1 },
      { path: "storage.max_disk_gb", type: "number", label: "音频占用上限（GB）", help: "超过后从最旧的已转写音频开始删；0 = 不限。", min: 0, step: 1 },
      { path: "storage.cleanup_enabled", type: "bool", label: "定时清理没有人说话的片段" },
    ],
  },
  {
    id: "speakers", title: "说话人识别", icon: "people", desc: "谁在说话。大多数时候不用动这里，在「说话人」页告诉它谁是谁就行。",
    fields: [
      { path: "speakers.identify_threshold", type: "number", label: "识别阈值", help: "相似度达到它才认成登记过的人；「说话人」页可以校准。", min: 0, max: 1, step: 0.01 },
      { path: "speakers.unknown_label", label: "认不出时显示" },
      { path: "speakers.infer_relations", type: "bool", label: "日报后自动推断关系", help: "让模型从对话里推断每个人和本人的关系；手动标的不会被改。" },
      { path: "speakers.cluster_threshold", type: "number", label: "聚类阈值", help: "越小分出的人越多。", min: 0, max: 1, step: 0.05, advanced: true },
      { path: "speakers.voice_group_threshold", type: "number", label: "未识别声音的分组阈值", help: "0–1，越高分得越细（同一个人可能被分成几组）。", min: 0, max: 1, step: 0.01, advanced: true },
      { path: "speakers.relation_min_utterances", type: "number", label: "推断关系至少需要的语句数", min: 1, max: 1000, step: 1, advanced: true },
      { path: "speakers.relation_min_confidence", type: "number", label: "推断关系的把握门槛", help: "0–1，低于它不写入。", min: 0, max: 1, step: 0.05, advanced: true },
      { path: "speakers.sample_consistency_min", type: "number", label: "样本一致性下限", help: "一个样本和这个人其他样本的相似度低于它，就标为「不像」。", min: 0, max: 1, step: 0.01, advanced: true },
      { path: "speakers.negative_margin", type: "number", label: "负样本的否决余量", help: "一段声音像负样本超过像本人这么多，就不认成这个人。", min: -1, max: 1, step: 0.01, advanced: true },
      { path: "speakers.system_turn_share", type: "number", label: "判为系统声音的占比", help: "一句话里来自系统声音的能量超过这个比例，就当作通话的对方（不会认成本人）。", min: 0, max: 1, step: 0.05, advanced: true },
      { path: "speakers.min_label_sample_s", type: "number", label: "标记时取样本的最短秒数", min: 1, max: 60, step: 0.5, advanced: true },
      { path: "speakers.turn_check_min_s", type: "number", label: "逐轮核对的最短秒数", min: 0, max: 60, step: 0.5, advanced: true },
      { path: "speakers.turn_reject_margin", type: "number", label: "逐轮核对的拒绝余量", min: 0, max: 1, step: 0.01, advanced: true },
      { path: "speakers.num_threads", type: "number", label: "本机模型线程数", help: "越多越快，也越占 CPU。", min: 0, max: 64, step: 1, advanced: true },
    ],
  },
  {
    id: "pipeline", title: "转写与纠错", icon: "doc", desc: "录下来的声音什么时候、怎样变成文字。",
    fields: [
      { path: "pipeline.process_interval_min", type: "number", label: "自动转写间隔（分钟）", min: 1, max: 1440, step: 1 },
      { path: "correction.enabled", type: "bool", label: "用大模型纠正转写错别字", help: "人名、术语以词表为准。" },
      { path: "pipeline.min_segment_age_s", type: "number", label: "片段至少放多少秒再转写", min: 0, max: 3600, step: 1, advanced: true },
    ],
  },
  {
    id: "memory", title: "记忆问答", icon: "chat", desc: "「问答」页默认怎么找相关记录。",
    fields: [
      { path: "memory.strategy", type: "select", label: "回答方式", options: [["prefetch", "先检索，再按需补查（推荐）"], ["rag", "只检索一次（最快）"], ["agentic", "让模型自己反复查（最细）"]] },
      { path: "memory.retrieval", type: "select", label: "怎么找相关记录", options: [["hybrid", "关键词 + 语义（推荐）"], ["vector", "只按语义"], ["fts", "只按关键词"]], help: "语义检索需要本地约 90 MB 的模型，首次使用自动下载。" },
      { path: "memory.query_expansion", type: "bool", label: "检索前让模型补同义词", help: "每次多一次调用，约 2–5 秒。" },
      { path: "memory.chat_model", label: "问答模型", help: "留空 = 用上面的纠错 / 问答模型。" },
      { path: "memory.prefetch_k", type: "number", label: "预先检索多少条", min: 1, max: 100, step: 1, advanced: true },
      { path: "memory.max_tool_rounds", type: "number", label: "最多补查几轮", min: 0, max: 50, step: 1, advanced: true },
      { path: "memory.embedding_model", label: "语义检索模型", advanced: true },
    ],
  },
  {
    id: "daily", title: "日报与推送", icon: "calendar", desc: "每天把当天的内容整理成一份日报，可以推送到手机。", action: { id: "test-push", label: "发一条测试消息" },
    fields: [
      { path: "analysis.daily_time", type: "time", label: "每天生成日报的时间", help: "本地时间。" },
      { path: "analysis.push_daily", type: "bool", label: "生成后自动推送" },
      { path: "push.channels", type: "channels", label: "推送渠道（可多选）", wide: true },
      { path: "push.feishu.webhook", type: "secret", label: "飞书 webhook" },
      { path: "push.feishu.secret", type: "secret", label: "飞书签名密钥", help: "没有开启签名校验就留空。" },
      { path: "push.wecom.webhook", type: "secret", label: "企业微信 webhook" },
      { path: "push.serverchan.sendkey", type: "secret", label: "Server酱 SendKey" },
      { path: "push.pushplus.token", type: "secret", label: "PushPlus token" },
      { path: "push.wxpusher.spt", type: "secret", label: "WxPusher SPT" },
      { path: "push.ntfy.topic", label: "ntfy 主题", help: "建议用一串随机字符。" },
      { path: "push.ntfy.server", label: "ntfy 服务器" },
      { path: "analysis.catchup_days", type: "number", label: "停机后补几天的日报", min: 1, max: 365, step: 1, advanced: true },
      { path: "analysis.session_gap_min", type: "number", label: "日报里隔多少分钟算新的时段", min: 1, max: 1440, step: 1, advanced: true },
      { path: "push.wxpusher.app_token", type: "secret", label: "WxPusher appToken", advanced: true },
    ],
  },
  { id: "glossary", title: "词表", icon: "book", desc: "人名、产品名、术语，每行一个。转写纠错时优先使用，检索时当作一个词。", custom: "glossary", fields: [] },
  { id: "import", title: "导入录音", icon: "upload", desc: "手机录音、会议录音等，会按语音切段，进入和麦克风录音一样的转写流程。", custom: "import", fields: [] },
  {
    id: "other", title: "其他", icon: "sliders", desc: "",
    fields: [
      { path: "ui.port", type: "number", label: "界面端口", help: "改后需要重启。", min: 1024, max: 65535, step: 1 },
      { path: "ui.open_browser", type: "bool", label: "启动时打开浏览器" },
      { path: "log.level", type: "select", label: "日志级别", options: [["DEBUG", "DEBUG"], ["INFO", "INFO"], ["WARNING", "WARNING"], ["ERROR", "ERROR"]], help: "改后需要重启。" },
    ],
  },
  { id: "paths", title: "文件位置", icon: "disk", desc: "", custom: "paths", fields: [] },
];

/** Config section name -> the page section its unlisted keys are shown under. */
const HOME = { mimo: "mimo", audio: "audio", storage: "storage", speakers: "speakers", correction: "pipeline", pipeline: "pipeline", memory: "memory", analysis: "daily", push: "daily", ui: "other", log: "other" };

/**
 * Editable keys the server offers that SECTIONS does not list: {sectionId: [field, …]} (always advanced).
 * Only scalars are offered (the type comes from the value, or from the default when the value is null).
 */
export function extraFields(values, defaults = {}, secrets = {}) {
  const known = new Set(SECTIONS.flatMap((s) => s.fields.map((f) => f.path)));
  const out = {};
  const add = (sec, field) => { const home = HOME[sec] || "other"; (out[home] = out[home] || []).push({ ...field, advanced: true, auto: true }); };
  for (const [sec, keys] of Object.entries(values || {})) {
    if (!keys || typeof keys !== "object") continue;
    for (const [key, value] of Object.entries(keys)) {
      const path = `${sec}.${key}`;
      if (known.has(path) || secrets[path]) continue;
      const sample = value ?? (defaults[sec] ? defaults[sec][key] : undefined);
      if (typeof sample === "boolean") add(sec, { path, type: "bool", label: path });
      else if (typeof sample === "number") add(sec, { path, type: "number", label: path, step: Number.isInteger(sample) ? 1 : "any", nullable: value === null });
      else if (typeof sample === "string") add(sec, { path, type: "text", label: path });
    }
  }
  for (const path of Object.keys(secrets || {})) {
    if (!known.has(path)) add(path.split(".")[0], { path, type: "secret", label: path });
  }
  return out;
}
