// Boot: theme -> shortcuts -> shell -> player -> live events -> status loop -> router.
// Pages are loaded lazily with dynamic import(); later packages add a page by one definePage() line here and one
// module under ./pages/.
import { html } from "./core/html.js";
import { $ } from "./core/dom.js";
import { api } from "./core/api.js";
import { initTheme, setTheme, themePref, THEME_LABEL, THEMES } from "./core/theme.js";
import { definePage, alias, start, pageList, reload } from "./core/router.js";
import { startStatusLoop, runJob } from "./core/status.js";
import { startLive } from "./core/events.js";
import { initShortcuts, bind, listShortcuts, comboLabel } from "./core/shortcuts.js";
import { status, isReadOnly, hasApiKey } from "./core/store.js";
import { today } from "./core/format.js";
import { initShell, startRecording, stopRecording, profilesModal } from "./ui/shell.js";
import { initPlayer } from "./ui/player.js";
import { registerCommand, registerProvider, openPalette } from "./ui/palette.js";
import { openModal, closeAll } from "./ui/modal.js";
import { on as busOn } from "./core/bus.js";

definePage("dashboard", { title: "总览", icon: "home", load: () => import("./pages/dashboard.js"), aliases: ["today"] });
definePage("transcript", { title: "记录", icon: "doc", load: () => import("./pages/timeline.js"), aliases: ["timeline"] });
// 0.9: 待确认 and 记忆 stay reachable by URL but out of the nav until their pages are built (wave 3 of the 1.0
// upgrade); the inbox shows up in the nav and the bell only while it really holds a question.
definePage("inbox", { title: "待确认", icon: "inbox", load: () => import("./pages/inbox.js"), nav: false, navWhen: "inbox" });
definePage("memory", { title: "记忆", icon: "book", load: () => import("./pages/memory.js"), nav: false });
definePage("daily", { title: "日报", icon: "calendar", load: () => import("./pages/daily.js") });
definePage("chat", { title: "问答", icon: "chat", load: () => import("./pages/chat.js") });
definePage("speakers", { title: "说话人", icon: "people", load: () => import("./pages/people.js"), aliases: ["people"] });
definePage("settings", { title: "设置", icon: "sliders", load: () => import("./pages/settings.js") });
alias("logs", "settings", ["logs"]);            // the old 日志 page is a tab of 设置 now

function shortcutSheet() {
  const groups = new Map();
  for (const s of listShortcuts()) { if (!groups.has(s.group)) groups.set(s.group, []); groups.get(s.group).push(s); }
  return openModal({
    title: "键盘快捷键", size: "sm", actions: [{ label: "关闭", kind: "primary" }],
    body: html`${[...groups.entries()].map(([g, list]) => html`<div class="keys-group"><h3>${g}</h3>
      <dl class="keys">${list.map((s) => html`<dt>${comboLabel(s.combo).split(" ").map((k) => html`<kbd>${k}</kbd>`)}</dt><dd>${s.desc}</dd>`)}</dl></div>`)}
      <p class="help">在记录页，选中一句话后：↑ ↓ 换句，Enter 播放，L 标记说话人，X 不是 TA，S 拆分，C 前后文。</p>`,
  });
}

function registerCommands() {
  for (const p of pageList().filter((x) => x.nav)) {
    registerCommand({ id: `go:${p.id}`, title: p.title, group: "跳转", icon: p.icon, keywords: `${p.id} ${(p.aliases || []).join(" ")} 打开 页面`, run: () => { location.hash = `#/${p.id}`; } });
  }
  registerCommand({ id: "go:logs", title: "日志 / 诊断", group: "跳转", icon: "terminal", keywords: "logs log 日志 诊断", run: () => { location.hash = "#/settings/logs"; } });
  const recording = () => { const st = status(); return !!(st && st.service && st.service.recording); };
  registerCommand({ id: "rec:start", title: "开始录音", group: "操作", icon: "mic", keywords: "record start 录音", when: () => !!status() && !isReadOnly() && !recording(), run: () => startRecording() });
  registerCommand({ id: "rec:stop", title: "停止录音", group: "操作", icon: "stop", keywords: "record stop 录音", when: () => !isReadOnly() && recording(), run: () => stopRecording() });
  registerCommand({ id: "job:process", title: "立即转写", group: "操作", icon: "refresh", keywords: "process transcribe 转写 处理", when: () => !isReadOnly() && hasApiKey(), run: () => runJob("process") });
  registerCommand({ id: "job:daily", title: "生成今天的日报", group: "操作", icon: "calendar", keywords: "daily digest 日报 总结", when: () => !isReadOnly() && hasApiKey(), run: () => runJob("daily", { day: (status() && status().today && status().today.day) || today() }) });
  for (const t of THEMES) {
    registerCommand({ id: `theme:${t}`, title: `主题：${THEME_LABEL[t]}`, group: "操作", icon: t === "dark" ? "moon" : t === "light" ? "sun" : "monitor", keywords: "theme 主题 外观 深色 浅色 dark light", when: () => themePref() !== t, run: () => setTheme(t) });
  }
  registerCommand({ id: "profiles", title: "切换档案…", group: "操作", icon: "user", keywords: "profile 档案 切换", run: () => profilesModal() });
  registerCommand({ id: "keys", title: "键盘快捷键", group: "操作", icon: "help", keywords: "shortcuts keys 快捷键 帮助", run: () => shortcutSheet() });
  // record search: GET /api/search (keyword + vector retrieval on the server)
  registerProvider(async (q, signal) => {
    const r = await api("/api/search", { query: { q, limit: 8 }, signal });
    return (r.hits || []).map((h) => ({
      title: h.text, sub: `${h.day} ${h.time} · ${h.speaker}`, group: "记录", icon: "doc",
      run: () => { location.hash = `#/transcript/${h.day}?u=${h.id}`; },
    }));
  });
}

function boot() {
  // pages render asynchronously: the router decides where the scroll position goes, not the browser
  try { history.scrollRestoration = "manual"; } catch (e) { /* ignore */ }
  initTheme();
  initShortcuts();
  initShell();
  initPlayer();
  registerCommands();
  bind("mod+k", () => openPalette(), { desc: "搜索记录 / 输入命令", inInput: true, global: true });
  bind("/", () => openPalette(), { desc: "搜索记录 / 输入命令" });
  bind("?", () => { shortcutSheet(); }, { desc: "显示这张快捷键表" });
  // another profile (switched here or in another tab): everything on screen belongs to the old one
  busOn("profile:switched", () => { closeAll(); reload(); });
  startLive();
  startStatusLoop();
  start($("#main"), { fallback: "dashboard" });
}

boot();
