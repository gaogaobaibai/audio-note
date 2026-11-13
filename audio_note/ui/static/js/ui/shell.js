// The frame around every page: navigation rail (bottom tab bar on phones), top bar with the global recording
// control, palette trigger, inbox bell, theme toggle and profile switcher, the offline banner and the tab title.
import { html, when } from "../core/html.js";
import { $, $$, render, patch, on } from "../core/dom.js";
import { api } from "../core/api.js";
import { on as busOn, emit } from "../core/bus.js";
import { store, prefs, status, inboxOpen, isReadOnly } from "../core/store.js";
import { pageList, pageDef } from "../core/router.js";
import { refreshStatus } from "../core/status.js";
import { cycleTheme, themePref, THEME_LABEL } from "../core/theme.js";
import { comboLabel } from "../core/shortcuts.js";
import { fmtClock, fmtMin, fmtTime, tsMs } from "../core/format.js";
import { sleep } from "../core/util.js";
import { icon } from "./icons.js";
import { badge, busy, notice } from "./components.js";
import { openModal, confirm, prompt, closeAll } from "./modal.js";
import { openMenu, openPopover } from "./popover.js";
import { openPalette } from "./palette.js";
import { toast, toastErr } from "./toast.js";

const TAB_PAGES = ["dashboard", "transcript", "daily", "chat"];      // the phone tab bar; everything else is under 更多
const THEME_ICON = { auto: "monitor", light: "sun", dark: "moon" };
let pageTitle = "";

function navBadge(id) {
  const st = status();
  if (!st) return "";
  if (id === "inbox") { const n = inboxOpen(); return n ? html`<span class="nav-badge" aria-label="${n} 个问题">${n > 99 ? "99+" : n}</span>` : ""; }
  if (id === "transcript" && st.failed) return html`<span class="nav-dot warn" title="${st.failed} 段转写失败"></span>`;
  if (id === "settings" && (!st.api_key || (st.models_missing || []).length)) return html`<span class="nav-dot warn" title="${!st.api_key ? "还没有配置 API key" : "模型文件缺失"}"></span>`;
  return "";
}

function renderNav() {
  const cur = (store.get("route") || {}).page;
  const pages = pageList().filter((p) => p.nav || (p.navWhen === "inbox" && inboxOpen() > 0));
  patch($("#nav"), html`${pages.map((p) => html`<a class="nav-item" href="#/${p.id}" data-page="${p.id}" title="${p.title}" ${p.id === cur ? html`aria-current="page"` : ""}>
    <span class="nav-ic">${icon(p.icon, 20)}</span><span class="nav-label">${p.title}</span>${navBadge(p.id)}</a>`)}`);
  const inMore = !TAB_PAGES.includes(cur);
  const moreBadge = pages.some((p) => !TAB_PAGES.includes(p.id) && navBadge(p.id) !== "");
  patch($("#tabbar"), html`${pages.filter((p) => TAB_PAGES.includes(p.id)).map((p) => html`<a class="tab-item" href="#/${p.id}" ${p.id === cur ? html`aria-current="page"` : ""}>
      <span class="nav-ic">${icon(p.icon, 22)}${navBadge(p.id)}</span><span>${p.title}</span></a>`)}
    <button type="button" class="tab-item" id="tab-more" ${inMore ? html`aria-current="page"` : ""} aria-haspopup="menu">
      <span class="nav-ic">${icon("menu", 22)}${when(moreBadge, () => html`<span class="nav-dot"></span>`)}</span><span>更多</span></button>`);
}

// ---- recording pill
function recView(st) {
  // while the server is unreachable the last known state is stale: say so instead of showing "录音中"
  if (!st || store.get("online") === false) return { cls: store.get("online") === false ? "off" : "", text: store.get("online") === false ? "连接断开" : "连接中…", time: "", toggle: null };
  const s = st.service || {};
  if (s.read_only) return { cls: "ro", text: "只读", time: "", toggle: null, lock: true };
  if (s.recording) return { cls: s.is_speaking ? "on speaking" : "on", text: s.is_speaking ? "有人说话" : "录音中", time: elapsed(st), toggle: "stop" };
  return { cls: "", text: "未录音", time: "", toggle: "start" };
}
function elapsed(st) {
  const t0 = tsMs(st && st.service && st.service.recorder_started_at);
  return Number.isFinite(t0) ? fmtClock(Math.max(0, (Date.now() - t0) / 1000)) : "";
}
function renderRec() {
  const st = status();
  const v = recView(st);
  const missing = !!(st && (st.models_missing || []).length);
  const label = `录音状态：${v.text}${v.time ? ` ${v.time}` : ""}，点开看详情`;
  patch($("#rec-pill"), html`<div class="rec-pill ${v.cls}">
    <button type="button" class="rec-main" data-rec="details" aria-label="${label}" title="录音详情">
      ${v.lock ? icon("lock", 14) : html`<span class="rec-dot" aria-hidden="true"></span>`}<span class="rec-text">${v.text}</span><span class="rec-time" data-rec-time>${v.time}</span>
    </button>
    ${v.toggle === "stop" ? html`<button type="button" class="rec-toggle stop" data-rec="stop" aria-label="停止录音" title="停止录音">${icon("stop", 14)}</button>` : ""}
    ${v.toggle === "start" ? html`<button type="button" class="rec-toggle start" data-rec="start" aria-label="开始录音" title="${missing ? "模型文件缺失，无法录音" : "开始录音"}" ${missing ? html`disabled` : ""}>${icon("mic", 15)}</button>` : ""}
  </div>`);
}
function tickRec() {
  const st = status();
  const el = $("[data-rec-time]");
  if (el && st && st.service && st.service.recording) { const t = elapsed(st); if (el.textContent !== t) el.textContent = t; }
}

export async function startRecording(btn) {
  const restore = btn ? busy(btn, "") : () => {};
  try {
    const r = await api("/api/recording/start", { method: "POST" });
    if (r && r.last_error) toast(`录音启动失败：${r.last_error}`, { kind: "err", ms: 8000 }); else toast("开始录音", { kind: "ok" });
  } catch (e) { toastErr(e); }
  restore();
  await refreshStatus();
}
export async function stopRecording() {
  if (!(await confirm({ title: "停止录音", text: "停止后不再保存新的语音片段；后台转写和日报仍会继续。", okLabel: "停止" }))) return;
  try { await api("/api/recording/stop", { method: "POST" }); toast("已停止录音"); } catch (e) { toastErr(e); }
  await refreshStatus();
}
/** Turn "record system audio too" on / off (restarts the recorder when it is running); offers an undo. */
export async function setSystemAudio(on, { undo = true } = {}) {
  try {
    const r = await api("/api/config", { method: "PUT", json: { audio: { system_audio: !!on } } });
    emit("config:saved", { saved: (r && r.saved) || [] });
    toast(`已${on ? "开启" : "关闭"}系统声音${r && r.restarted_recorder ? "，录音已重启" : ""}`, { kind: "ok", action: undo ? { label: "撤销", run: () => setSystemAudio(!on, { undo: false }) } : null });
    await sleep(500);                       // the recorder needs a moment to come back before the status is truthful
  } catch (e) { toastErr(e); }
  await refreshStatus();
}

function recDetails(anchor) {
  const body = () => {
    const st = status();
    if (!st) return html`<div class="pop-pad t2">正在连接 audio-note…</div>`;
    const s = st.service || {};
    const rows = [];
    if (s.read_only) rows.push(notice({ kind: "info", compact: true, body: `服务正在另一个进程运行（pid ${s.other_pid}）。这里可以看记录和问答；录音和处理请在那个进程的界面操作。` }));
    else if (s.recording) rows.push(html`<div class="pop-line"><strong>${s.is_speaking ? "正在录音 · 有人说话" : "正在录音 · 安静"}</strong></div>
      <div class="pop-line t2">${fmtTime(s.recorder_started_at)} 开始 · 已保存 ${s.segments_written} 段，${fmtMin((s.speech_total_s || 0) / 60)}语音</div>`);
    else rows.push(html`<div class="pop-line"><strong>没有在录音</strong></div><div class="pop-line t2">开始后只保存有人说话的片段，每 ${st.process_interval_min} 分钟自动转写一次。</div>`);
    if (s.last_error && !s.recording) rows.push(notice({ kind: "danger", compact: true, body: `录音线程出错：${s.last_error}` }));
    if (s.recording && s.system_audio && s.system_error) rows.push(notice({ kind: "warn", compact: true, body: `系统声音没有录上（只录麦克风）：${s.system_error}` }));
    rows.push(html`<div class="pop-line t2">麦克风：${s.device == null ? "系统默认" : String(s.device)}${s.system_audio && s.recording && s.system_name ? ` · 系统声音：${s.system_name}` : ""}</div>`);
    if (!s.read_only) rows.push(html`<label class="switch-row" title="开会 / 通话时把对方的声音（扬声器或耳机里放出来的）一起录下来；用耳机效果最好">
        <span><strong>同时录系统声音</strong><small>会议、通话里对方的声音</small></span>
        <input type="checkbox" class="switch" role="switch" data-rec-sys ${s.system_audio ? html`checked` : ""}></label>`);
    if (st.pending) rows.push(html`<div class="pop-line t2">${st.pending} 段录音等待转写</div>`);
    rows.push(html`<div class="pop-foot"><a class="btn ghost sm" href="#/dashboard" data-pop-close>总览</a><a class="btn ghost sm" href="#/settings/audio" data-pop-close>录音设置</a></div>`);
    return html`<div class="pop-pad rec-pop">${rows}</div>`;
  };
  const pop = openPopover(anchor, body(), {
    label: "录音详情", className: "pop-panel",
    onMount: (p) => {
      p.el.addEventListener("change", async (e) => {
        const sw = e.target.closest("[data-rec-sys]");
        if (!sw) return;
        sw.disabled = true;
        await setSystemAudio(sw.checked);
        if (document.contains(p.el)) render(p.el, body());
      });
      p.el.addEventListener("click", (e) => { if (e.target.closest("[data-pop-close]")) p.close(true); });
    },
  });
  const off = busOn("status", () => {
    if (!document.contains(pop.el) || pop.el.querySelector("[data-rec-sys]:disabled")) return;
    const onSwitch = !!(document.activeElement && pop.el.contains(document.activeElement) && document.activeElement.matches("[data-rec-sys]"));
    if (patch(pop.el, body()) && onSwitch) { const sw = pop.el.querySelector("[data-rec-sys]"); if (sw) sw.focus(); }
  });
  pop.closed.then(off);
}

// ---- top bar bits
function renderTopRight() {
  const st = status();
  const n = inboxOpen();
  const bell = $("#bell-btn");
  if (bell) {
    bell.hidden = !n;
    bell.setAttribute("aria-label", n ? `待确认：${n} 个问题` : "待确认");
    bell.title = n ? `有 ${n} 件事想问你` : "待确认";
    patch(bell, html`${icon("bell", 18)}${n ? html`<span class="count-badge" aria-hidden="true">${n > 99 ? "99+" : n}</span>` : ""}`);
  }
  const pref = themePref();
  const tb = $("#theme-btn");
  if (tb) {
    tb.setAttribute("aria-label", `主题：${THEME_LABEL[pref]}（点击切换）`);
    tb.title = `主题：${THEME_LABEL[pref]}`;
    patch(tb, icon(THEME_ICON[pref], 18));
  }
  const pb = $("#profile-btn");
  if (pb) {
    const label = st && st.profile ? st.profile.label : "…";
    pb.setAttribute("aria-label", `档案：${label}（点击切换）`);
    patch(pb, html`<span class="profile-mark" aria-hidden="true">${Array.from(String(label))[0] || "·"}</span><span class="profile-name clip">${label}</span>${icon("chevron-down", 14)}`);
  }
}
function renderTitle() {
  const st = status();
  const h = $("#page-title");
  if (h && h.textContent !== pageTitle) h.textContent = pageTitle;
  const n = inboxOpen();
  const rec = st && st.service && st.service.recording;
  document.title = `${rec ? "● " : ""}${n ? `(${n}) ` : ""}${pageTitle ? `${pageTitle} · ` : ""}audio-note`;
}
function renderBanner() {
  const offline = store.get("online") === false;
  patch($("#banner"), offline ? html`<div class="banner" role="status">${icon("alert", 16)}<span>和 audio-note 的连接断开了，正在重试…（程序可能已退出）</span></div>` : html``);
}

// ---- profiles
export function profilesModal() {
  let data = null;
  const body = () => {
    if (!data) return html`<div class="t2">加载中…</div>`;
    return html`<p class="help">每个档案有自己的录音、文字记录、日报和说话人声纹，互不混淆。API key 等共享设置放在根目录 config.yaml，各档案继承。${when(data.custom_config, () => html`<br>当前用的是自定义配置文件 <code>${data.custom_config}</code>，不属于任何档案。`)}</p>
      <div class="profile-list">${data.profiles.map((p) => {
        const s = p.stats;
        const stats = p.error ? badge(p.error, "danger") : s ? `${s.utterances} 条记录 · ${s.days} 天 · ${s.speakers} 个说话人${s.last_day ? ` · 最近 ${s.last_day}` : ""}` : "还没有数据";
        return html`<div class="profile-row ${p.current ? "cur" : ""}">
          <div class="grow"><div class="profile-pn">${p.label}${when(p.current, () => badge("当前", "info"))}${when(p.default, () => badge("config.yaml + data/", "plain"))}${when(p.running_pid, () => badge(`另一进程在用 (pid ${p.running_pid})`, "warn"))}</div>
            <div class="small t2">${stats}</div><div class="xs t3 mono">${p.data_dir || ""}</div></div>
          ${p.current ? "" : html`<button type="button" class="btn sm primary" data-switch="${p.name}">切换</button>`}
          ${p.current || p.default ? "" : html`<button type="button" class="icon-btn" data-delp="${p.name}" aria-label="删除档案 ${p.name} 及其全部数据" title="删除档案及其全部数据">${icon("trash", 16)}</button>`}</div>`;
      })}</div>
      <div class="profile-new"><input class="input grow" id="np-name" placeholder="新档案名，例如：家里 / 工作" aria-label="新档案名"><label class="check"><input type="checkbox" id="np-copy"> 复制当前档案的设置</label><button type="button" class="btn" id="np-create">创建并切换</button></div>`;
  };
  return openModal({
    title: "档案", size: "md", body: body(), actions: [{ label: "关闭" }],
    onMount: (h) => {
      const load = async () => { try { data = await api("/api/profiles"); if (!h.closed) render(h.body, body()); } catch (e) { toastErr(e); h.close(); } };
      const switchTo = async (name) => {
        for (const b of $$("button", h.el)) b.disabled = true;
        try {
          await api(`/api/profiles/${encodeURIComponent(name)}/switch`, { method: "POST" });
          h.close(true);
          closeAll();
          toast(`已切换到档案 ${name === "default" ? "默认" : name}`, { kind: "ok" });
          // the status loop notices the new profile and announces "profile:switched" (main.js re-mounts the page);
          // the same path handles a switch made in another tab
          await refreshStatus();
        } catch (e) { toastErr(e); for (const b of $$("button", h.el)) b.disabled = false; }
      };
      on(h.body, "click", "[data-switch]", (e, b) => switchTo(b.dataset.switch));
      on(h.body, "click", "[data-delp]", async (e, b) => {
        const name = b.dataset.delp;
        const typed = await prompt({ title: `删除档案 ${name}`, label: "会删除它的全部录音、记录、日报和声纹，不可恢复。输入档案名确认：", placeholder: name, okLabel: "删除" });
        if (typed === null) return;
        if (typed !== name) { toast("名字不一致，已取消", { kind: "err" }); return; }
        try { await api(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE", json: { confirm: name } }); toast(`已删除档案 ${name}`); } catch (err) { toastErr(err); }
        load();
      });
      on(h.body, "click", "#np-create", async () => {
        const name = h.body.querySelector("#np-name").value.trim();
        if (!name) { toast("先填档案名", { kind: "err" }); h.body.querySelector("#np-name").focus(); return; }
        try {
          await api("/api/profiles", { method: "POST", json: { name, copy_from: h.body.querySelector("#np-copy").checked && data && data.current ? data.current : null } });
          await switchTo(name);
        } catch (e) { toastErr(e); }
      });
      on(h.body, "keydown", "#np-name", (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); h.body.querySelector("#np-create").click(); } });
      load();
    },
  });
}

export function initShell() {
  const app = $("#app");
  app.classList.toggle("rail-collapsed", !!prefs.get("railCollapsed", false));
  const collapse = $("#rail-toggle");
  const syncCollapse = () => { const c = app.classList.contains("rail-collapsed"); collapse.setAttribute("aria-expanded", c ? "false" : "true"); collapse.setAttribute("aria-label", c ? "展开导航栏" : "收起导航栏"); collapse.title = c ? "展开导航栏" : "收起导航栏"; };
  collapse.addEventListener("click", () => { prefs.set("railCollapsed", app.classList.toggle("rail-collapsed")); syncCollapse(); });
  syncCollapse();

  const kbd = $("#palette-kbd");
  if (kbd) kbd.textContent = comboLabel("mod+k");
  $("#palette-btn").addEventListener("click", () => openPalette());
  $("#bell-btn").addEventListener("click", () => { location.hash = "#/inbox"; });
  $("#theme-btn").addEventListener("click", () => { const t = cycleTheme(); toast(`主题：${THEME_LABEL[t]}`); });
  $("#profile-btn").addEventListener("click", () => profilesModal());
  on($("#rec-pill"), "click", "[data-rec]", (e, b) => {
    const a = b.dataset.rec;
    if (a === "start") startRecording(b);
    else if (a === "stop") stopRecording();
    else recDetails(b);
  });
  on($("#tabbar"), "click", "#tab-more", async (e, b) => {
    const cur = (store.get("route") || {}).page;
    const rest = pageList().filter((p) => (p.nav || (p.navWhen === "inbox" && inboxOpen() > 0)) && !TAB_PAGES.includes(p.id));
    const pick = await openMenu(b, rest.map((p) => ({ id: p.id, label: p.title, icon: p.icon, checked: p.id === cur ? true : undefined })), { label: "更多页面" });
    if (pick) location.hash = `#/${pick}`;
  });

  const paintAll = () => { renderNav(); renderRec(); renderTopRight(); renderTitle(); renderBanner(); };
  busOn("status", paintAll);
  busOn("offline", paintAll);
  busOn("online", () => { paintAll(); toast("已重新连接", { kind: "ok" }); });
  busOn("theme", renderTopRight);
  busOn("route", (r) => { const def = pageDef(r.page); pageTitle = (def && def.title) || ""; renderNav(); renderTitle(); });
  busOn("route:title", (t) => { pageTitle = t || pageTitle; renderTitle(); });
  setInterval(tickRec, 1000);
  paintAll();
}

export { isReadOnly };
