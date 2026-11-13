// 日志 / 诊断 (a tab of the settings page): the tail of the service / CLI log with level and text filters, and a
// short "what is running" summary. GET /api/logs?lines=400&which=service|cli -> {path, lines[], size}.
// The log has no event channel, so this is the one place that polls on a timer (every 3 s while 自动刷新 is on and
// the tab is visible); filtering happens on the lines already fetched, never per keystroke on the server.
import { html, when } from "../../core/html.js";
import { $, render, patch, on } from "../../core/dom.js";
import { api, isAbort } from "../../core/api.js";
import { status, store } from "../../core/store.js";
import { fmtBytes, fmtDT } from "../../core/format.js";
import { debounce } from "../../core/util.js";
import { button, chip, emptyState, kv, notice, segmented, skeleton } from "../../ui/components.js";
import { toast } from "../../ui/toast.js";

const LEVEL_RE = /\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s/;
const LEVELS = [["", "全部"], ["WARNING", "警告及以上"], ["ERROR", "只看错误"]];
const RANK = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 };

export function mountLogs(ctx, el) {
  let which = "service", auto = true, text = "", level = "", lines = [], info = null, sig = "", alive = true, err = "";

  render(el, html`<div class="stack lg">
    <section class="card" id="lg-diag" aria-label="运行状态"></section>
    <section class="card logs" aria-label="日志">
      <div class="logs-bar">
        <div id="lg-which"></div>
        <div class="chips" id="lg-levels" role="group" aria-label="级别"></div>
        <input class="input logs-filter" id="lg-filter" type="search" placeholder="过滤文本" aria-label="过滤日志文本" autocomplete="off">
        <label class="check"><input type="checkbox" id="lg-auto" checked> 自动刷新</label>
        ${button({ label: "刷新", icon: "refresh", size: "sm", attrs: { id: "lg-refresh" } })}
        ${button({ label: "复制", icon: "copy", size: "sm", kind: "ghost", attrs: { id: "lg-copy" }, title: "复制当前显示的日志" })}
      </div>
      <div class="small t3" id="lg-path"></div>
      <div class="logbox" id="lg-box" tabindex="0" role="log" aria-label="日志内容">${skeleton("text", 8)}</div>
    </section>
  </div>`);
  const box = $("#lg-box", el);

  const shown = () => {
    const q = text.trim().toLowerCase(), min = level ? RANK[level] : -1;
    return lines.filter((l) => {
      if (q && !l.toLowerCase().includes(q)) return false;
      if (min < 0) return true;
      const m = LEVEL_RE.exec(l);
      return m ? RANK[m[1]] >= min : false;
    });
  };
  const paintControls = () => {
    patch($("#lg-which", el), segmented([{ id: "service", label: "服务 / 界面" }, { id: "cli", label: "命令行" }], which, { label: "哪一份日志" }));
    patch($("#lg-levels", el), html`${LEVELS.map(([k, l]) => chip({ label: l, on: level === k, attrs: { "data-level": k } }))}`);
  };
  const paintLines = ({ stick = false } = {}) => {
    const atBottom = stick || box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
    const list = shown();
    if (err) render(box, notice({ kind: "danger", compact: true, body: err }));
    else render(box, list.length ? html`${list.map((l) => { const m = LEVEL_RE.exec(l); return html`<div class="ln ${m ? m[1] : ""}">${l}</div>`; })}` : emptyState({ compact: true, icon: "terminal", title: lines.length ? "没有匹配的日志" : "日志是空的" }));
    if (atBottom) box.scrollTop = box.scrollHeight;
    $("#lg-path", el).textContent = info ? `${info.path} · ${fmtBytes(info.size)} · 显示最后 ${lines.length} 行中的 ${list.length} 行` : "";
  };
  const paintDiag = () => {
    const st = status();
    if (!st) { patch($("#lg-diag", el), html`<div class="t2">正在连接…</div>`); return; }
    const s = st.service || {};
    const runs = Object.entries(st.scheduler_runs || {}), errs = Object.entries(st.scheduler_errors || {}).filter(([, v]) => v);
    patch($("#lg-diag", el), html`<div class="sec-head"><div class="sec-title"><h2>运行状态</h2><div class="sec-sub">出问题时先看这里，再看下面的日志。</div></div></div>
      <div class="diag-grid">${kv([
        ["版本", `${st.version || "–"}${st.schema_version != null ? `（数据库 v${st.schema_version}）` : ""}`],
        ["进程", `pid ${st.pid}，${fmtDT(st.started_at)} 启动`],
        ["后台处理", s.read_only ? `只读：服务在另一个进程（pid ${s.other_pid}）` : s.scheduler ? "运行中" : "没有运行"],
        ["界面更新", store.get("live") === "sse" ? "实时推送" : "定时查询（每 5 秒）"],
        ["上次自动转写", st.last_process_at ? fmtDT(st.last_process_at) : "还没有"],
        st.deferred ? ["暂缓处理", String(st.deferred)] : null,
      ])}${kv([
        ["配置文件", html`<span class="mono">${st.config_path || ""}</span>`],
        ["数据目录", html`<span class="mono">${st.data_dir || ""}</span>`],
        ["磁盘剩余", st.disk && st.disk.free_bytes != null ? fmtBytes(st.disk.free_bytes) : "–"],
        runs.length ? ["定时任务已跑", runs.map(([k, v]) => `${k} ${v} 次`).join("，")] : null,
      ])}</div>
      ${when(errs.length, () => html`<div class="notices">${errs.map(([k, v]) => notice({ kind: "warn", compact: true, body: `定时任务 ${k} 上次出错：${v}` }))}</div>`)}`);
  };

  const load = async ({ stick = false } = {}) => {
    try {
      const r = await api("/api/logs", { query: { lines: 400, which }, signal: ctx.signal });
      if (!alive) return;
      err = "";
      const next = `${which}|${r.size}|${r.lines.length}|${r.lines[r.lines.length - 1] || ""}`;
      info = r;
      if (next === sig && !stick) return;          // nothing new: leave the DOM (and the user's scroll / selection) alone
      sig = next;
      lines = r.lines;
    } catch (e) { if (isAbort(e) || !alive) return; err = e.message || String(e); }
    paintLines({ stick });
  };

  on(el, "click", "[data-seg]", (e, b) => { which = b.dataset.seg; sig = ""; paintControls(); load({ stick: true }); }, { signal: ctx.signal });
  on(el, "click", "[data-level]", (e, b) => { level = b.dataset.level; paintControls(); paintLines({ stick: true }); }, { signal: ctx.signal });
  on(el, "input", "#lg-filter", debounce((e, inp) => { text = inp.value; paintLines({ stick: true }); }, 120), { signal: ctx.signal });
  on(el, "change", "#lg-auto", (e, c) => { auto = c.checked; }, { signal: ctx.signal });
  on(el, "click", "#lg-refresh", () => { sig = ""; load({ stick: true }); }, { signal: ctx.signal });
  on(el, "click", "#lg-copy", async () => {
    try { await navigator.clipboard.writeText(shown().join("\n")); toast("已复制到剪贴板", { kind: "ok" }); }
    catch (e) { toast("复制失败：浏览器不允许访问剪贴板", { kind: "err" }); }
  }, { signal: ctx.signal });

  const stopPoll = ctx.poll(() => { if (alive && auto) return load(); return null; }, 3000);
  const offStatus = ctx.on("status", paintDiag);
  const offLive = ctx.on("live:mode", paintDiag);
  paintControls();
  paintDiag();
  load({ stick: true });
  return { destroy() { alive = false; stopPoll(); offStatus(); offLive(); } };
}
