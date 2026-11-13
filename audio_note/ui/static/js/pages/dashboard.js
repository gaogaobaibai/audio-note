// 总览: today at a glance. Regions are patched from status / live events (no timer rebuilds the page), so hover,
// focus and text selection survive; the "最近的记录" list follows new sentences as they arrive.
import { html, when } from "../core/html.js";
import { $, render, patch, on } from "../core/dom.js";
import { api, isAbort } from "../core/api.js";
import { status, loadDays, ensureMinDbfs, isReadOnly } from "../core/store.js";
import { ensureStatus, runJob, jobSummary } from "../core/status.js";
import { addDays, dayLabel, dayTitle, fmtBytes, fmtClock, fmtDT, fmtMin, fmtTime, today, tsMs, WEEK } from "../core/format.js";
import { debounce } from "../core/util.js";
import { icon } from "../ui/icons.js";
import { button, emptyState, notice, skeleton, spinner, tile } from "../ui/components.js";
import { bars, hbars, ring, sparkline } from "../ui/charts.js";
import { colorClass, rememberSpeakers } from "../ui/speaker.js";
import { uttRow } from "../ui/utterance.js";
import { player } from "../ui/player.js";
import { setSystemAudio, startRecording, stopRecording } from "../ui/shell.js";

const RUNNABLE = ["process", "daily", "cleanup", "relabel", "reindex"];      // offered first; other jobs the server reports follow
const NO_MANUAL = new Set(["push"]);                                           // needs a day: started from the 日报 page

function heroView(st) {
  const s = st.service || {};
  const src = html`声音来源：麦克风${s.system_audio ? html`＋系统声音${s.recording && s.system_name ? `（${s.system_name}）` : ""}` : ""}`;
  if (s.read_only) return { cls: "ro", title: "只读模式", detail: html`服务正在另一个进程运行（pid ${s.other_pid}）。这里可以看记录和问答，录音和处理请在那个进程的界面操作。` };
  if (s.recording) {
    return {
      cls: s.is_speaking ? "on speaking" : "on", title: s.is_speaking ? "正在录音 · 有人说话" : "正在录音 · 安静",
      detail: html`${fmtTime(s.recorder_started_at)} 开始 · 本次已保存 ${s.segments_written} 段，${fmtMin((s.speech_total_s || 0) / 60)}语音 · 麦克风：${s.device == null ? "系统默认" : String(s.device)}<br>${src}`,
    };
  }
  return {
    cls: "", title: "没有在录音",
    detail: html`开始后只保存有人说话的片段；每 ${st.process_interval_min} 分钟自动转写一次，${st.daily_time} 生成日报${s.scheduler ? "" : "（后台处理未运行）"}。<br>${src}`,
  };
}

function heroHtml(st) {
  const s = st.service || {};
  const v = heroView(st);
  const missing = (st.models_missing || []).length > 0;
  const t0 = tsMs(s.recorder_started_at);
  return html`<div class="hero ${v.cls}">
    <div class="hero-state"><span class="hero-dot" aria-hidden="true"></span>
      <div><div class="hero-title">${v.title}${when(s.recording && Number.isFinite(t0), () => html`<span class="hero-time" data-hero-time>${fmtClock((Date.now() - t0) / 1000)}</span>`)}</div>
      <div class="hero-detail">${v.detail}</div></div></div>
    ${when(!s.read_only, () => html`<div class="hero-acts">
      ${s.recording ? button({ label: "停止录音", icon: "stop", attrs: { "data-act": "stop" } }) : button({ label: "开始录音", icon: "mic", kind: "rec", attrs: { "data-act": "start" }, disabled: missing, title: missing ? "模型文件缺失，无法录音" : "" })}
      <label class="switch-row hero-sys" title="开会 / 通话时把对方的声音（扬声器或耳机里放出来的）一起录下来；用耳机效果最好，外放时对方的声音会被麦克风和系统声音各录一次">
        <span class="small">同时录系统声音</span><input type="checkbox" class="switch" role="switch" data-act="sys" ${s.system_audio ? html`checked` : ""}></label>
    </div>`)}
  </div>`;
}

function noticesHtml(st) {
  const s = st.service || {};
  const out = [];
  const link = (href, label) => html`<a class="btn sm" href="${href}">${label}</a>`;
  if ((st.models_missing || []).length) out.push(notice({ kind: "danger", body: html`模型文件缺失：${st.models_missing.join("、")}。运行 <code>scripts/download_models.ps1</code> 下载。` }));
  if (!st.api_key) out.push(notice({ kind: "warn", body: "还没有配置 MiMo API key：录音会保存，但不会转写、也不能问答。", actions: link("#/settings/mimo", "去设置") }));
  if (s.last_error && !s.recording) out.push(notice({ kind: "danger", body: `录音线程出错：${s.last_error}`, actions: link("#/settings/audio", "检查麦克风设置") }));
  if (s.recording && s.system_audio && s.system_error) out.push(notice({ kind: "warn", body: `系统声音没有录上（只录麦克风）：${s.system_error}`, actions: link("#/settings/audio", "检查设置") }));
  if (st.failed) out.push(notice({ kind: "warn", body: `${st.failed} 段转写失败。`, actions: html`${isReadOnly() ? "" : button({ label: "重试失败的段", size: "sm", attrs: { "data-act": "retry" } })}${link("#/transcript", "查看")}` }));
  const proc = st.jobs && st.jobs.process;
  if (st.pending && st.api_key && !(proc && proc.running)) out.push(notice({ kind: "info", body: `${st.pending} 段录音等待转写${s.scheduler ? `（每 ${st.process_interval_min} 分钟自动处理一次）` : ""}。`, actions: isReadOnly() ? "" : button({ label: "立即转写", size: "sm", kind: "primary", attrs: { "data-act": "process" } }) }));
  if (proc && proc.running) out.push(notice({ kind: "info", body: spinner("正在转写录音…") }));
  if (st.deferred) out.push(notice({ kind: "info", body: `本机的声音分析暂缓：${st.deferred}。` }));
  if (st.unknown_utterances) out.push(notice({ kind: "info", body: `有 ${st.unknown_utterances} 句话还没认出是谁说的。`, actions: link("#/speakers", "去认一认") }));
  return html`${out}`;
}

function tilesHtml(st, days) {
  const cap = st.disk && st.disk.cap_gb ? st.disk.cap_gb * 1024 ** 3 : 0;
  const pct = cap ? Math.min(100, (st.disk.audio_bytes / cap) * 100) : 0;
  const series = (key) => { const byDay = new Map((days || []).map((d) => [d.day, d])); const out = []; for (let i = 13; i >= 0; i--) { const d = byDay.get(addDays(st.today.day, -i)); out.push(d ? Number(d[key]) || 0 : 0); } return out; };
  const hasTrend = (days || []).length > 1;
  return html`
    ${tile({ label: "今日语句", value: st.today.utterances, unit: "条", chart: hasTrend ? sparkline(series("utterances"), { label: "最近 14 天每天的语句数" }) : "" })}
    ${tile({ label: "今日语音", value: Number(st.today.recorded_speech_min || 0).toFixed(0), unit: "分钟", chart: hasTrend ? sparkline(series("speech_min"), { label: "最近 14 天每天的语音分钟数" }) : "" })}
    ${tile({ label: "待转写", value: st.pending, unit: "段", hint: st.last_process_at ? `上次自动转写 ${fmtTime(st.last_process_at)}` : "" })}
    ${tile({ label: "全部记录", value: st.total.utterances, unit: `条 / ${st.total.days} 天` })}
    ${tile({ label: "音频占用", value: fmtBytes(st.disk.audio_bytes), unit: cap ? `/ ${st.disk.cap_gb} GB` : "", chart: cap ? html`<span class="${pct > 90 ? "ring-danger" : pct > 70 ? "ring-warn" : ""}">${ring(pct, { label: `已用 ${pct.toFixed(0)}%` })}</span>` : "" })}
    ${tile({ label: "下次日报", value: st.daily_time, hint: st.last_daily ? `最近一份：${st.last_daily.day}` : "还没有日报" })}`;
}

function weekHtml(st, days) {
  const byDay = new Map((days || []).map((d) => [d.day, d]));
  const items = [];
  for (let i = 6; i >= 0; i--) {
    const day = addDays(st.today.day, -i), d = byDay.get(day);
    const min = d ? Number(d.speech_min) || 0 : 0;
    items.push({ label: i === 0 ? "今天" : `周${WEEK[new Date(`${day}T00:00:00`).getDay()]}`, value: min, on: i === 0, title: `${day}：${fmtMin(min)}${d ? `，${d.utterances} 句` : ""}` });
  }
  const total = items.reduce((a, x) => a + x.value, 0);
  return html`<div class="sec-head"><div class="sec-title"><h2>最近 7 天</h2><div class="sec-sub">${total ? `一共说了 ${fmtMin(total)}` : "这一周还没有记录"}</div></div></div>
    ${bars(items, { w: 320, h: 104, label: "最近 7 天每天的语音时长" })}`;
}

function peopleHtml(st, days) {
  const list = days || [];
  const d = list.find((x) => x.day === st.today.day && x.utterances) || list.find((x) => x.utterances);
  if (!d) return html`<div class="sec-head"><div class="sec-title"><h2>谁在说话</h2></div></div>${emptyState({ compact: true, text: "有了转写内容之后，这里会显示每个人说了多少。" })}`;
  const rows = (d.speakers || []).slice(0, 6).map((s) => ({ label: s.name, value: s.n, text: `${s.n} 句`, cls: colorClass(s.name) }));
  return html`<div class="sec-head"><div class="sec-title"><h2>谁在说话</h2><div class="sec-sub">${d.day === st.today.day ? "今天" : `${dayTitle(d.day)} ${dayLabel(d.day)}`} · ${d.utterances} 句</div></div>
      <a class="btn ghost sm" href="#/transcript/${d.day}">看记录${icon("arrow-right", 14)}</a></div>
    ${hbars(rows)}`;
}

function digestHtml(st, dailies) {
  const d = (dailies || [])[0];
  if (!d) return html`<div class="sec-head"><div class="sec-title"><h2>最近的日报</h2></div></div>${emptyState({ compact: true, text: `每天 ${st.daily_time} 会把当天的内容整理成一份日报。` })}`;
  return html`<div class="sec-head"><div class="sec-title"><h2>最近的日报</h2><div class="sec-sub">${dayTitle(d.day)} ${dayLabel(d.day)} · ${d.utterance_count} 条记录 · ${d.pushed_at ? "已推送" : "未推送"}</div></div>
      <a class="btn ghost sm" href="#/daily/${d.day}">读全文${icon("arrow-right", 14)}</a></div>
    <p class="digest-excerpt">${d.overview || "（这份日报没有概览）"}${d.overview && d.overview.length >= 160 ? "…" : ""}</p>`;
}

function jobsHtml(st) {
  const jobs = st.jobs || {};
  const names = [...RUNNABLE.filter((n) => jobs[n]), ...Object.keys(jobs).filter((n) => !RUNNABLE.includes(n) && !NO_MANUAL.has(n))];
  const ro = isReadOnly();
  return html`${names.map((n) => {
    const j = jobs[n];
    const state = j.running ? spinner(`运行中，${fmtTime(j.started_at)} 开始`)
      : j.error ? html`<span class="job-s err" title="${j.error}">上次失败：${j.error}</span>`
        : j.finished_at ? html`<span class="job-s">上次 ${fmtDT(j.finished_at)}${jobSummary(n, j.result)}</span>` : html`<span class="job-s">尚未运行</span>`;
    return html`<div class="job-row"><span class="job-n">${j.label || n}</span>${state}${button({ label: "运行", size: "sm", attrs: { "data-job": n }, disabled: j.running || ro })}</div>`;
  })}`;
}

function recentHtml(st, utts, error) {
  const s = st.service || {};
  const head = html`<div class="sec-head"><div class="sec-title"><h2>最近的记录</h2><div class="sec-sub">新的句子转写出来后会自动出现在这里</div></div>
    <a class="btn ghost sm" href="#/transcript/${st.today.day}">今天全部${icon("arrow-right", 14)}</a></div>`;
  if (error) return html`${head}${notice({ kind: "danger", compact: true, body: error })}`;
  if (utts === null) return html`${head}${skeleton("rows", 4)}`;
  if (!utts.length) {
    return html`${head}${emptyState({ compact: true, icon: "doc", title: "今天还没有转写内容", text: st.pending ? `${st.pending} 段录音等待转写` : s.recording ? "录到有人说话后，会在下一轮自动转写" : s.read_only ? "" : "点「开始录音」，剩下的交给它。" })}`;
  }
  return html`${head}<div class="utt-rows">${utts.map((u) => uttRow(u, { link: true }))}</div>`;
}

export default {
  async mount(ctx) {
    const st0 = await ensureStatus();
    if (!ctx.alive()) return;
    if (!st0) throw new Error("无法连接到 audio-note（程序是否已退出？）");
    await ensureMinDbfs();
    if (!ctx.alive()) return;
    render(ctx.el, html`<div class="page dash">
      <header class="page-head"><div><h1 id="dash-day"></h1><p class="lead" id="dash-lead"></p></div></header>
      <div class="notices" id="dash-notices"></div>
      <section class="card" id="dash-hero" aria-label="录音状态"></section>
      <section class="tiles" id="dash-tiles" aria-label="今天的数字"></section>
      <div class="dash-grid">
        <section class="card" id="dash-recent" aria-label="最近的记录"></section>
        <div class="stack">
          <section class="card" id="dash-week"></section>
          <section class="card" id="dash-people"></section>
          <section class="card" id="dash-digest"></section>
        </div>
      </div>
      <details class="disclosure" id="dash-jobs"><summary><span>后台任务</span><span class="t3 small" id="dash-jobs-hint"></span></summary><div class="job-list" id="dash-jobs-body"></div></details>
    </div>`);

    let days = null, dailies = null, recent = null, recentErr = "";
    const paintStatus = () => {
      const st = status();
      if (!st) return;
      patch($("#dash-day"), html`${dayTitle(st.today.day)} <span class="dash-week">${dayLabel(st.today.day)}</span>`);      // follows midnight
      patch($("#dash-hero"), heroHtml(st));
      patch($("#dash-notices"), noticesHtml(st));
      patch($("#dash-tiles"), tilesHtml(st, days));
      patch($("#dash-jobs-body"), jobsHtml(st));
      const lead = [st.today.utterances ? `今天记下了 ${st.today.utterances} 句话，约 ${fmtMin(st.today.speech_min || 0)}` : "今天还没有记下什么", st.profile ? `档案：${st.profile.label}` : ""].filter(Boolean).join(" · ");
      if ($("#dash-lead").textContent !== lead) $("#dash-lead").textContent = lead;
      const hint = st.last_process_at ? `上次自动转写 ${fmtDT(st.last_process_at)}` : "";
      if ($("#dash-jobs-hint").textContent !== hint) $("#dash-jobs-hint").textContent = hint;
    };
    const paintData = () => {
      const st = status();
      if (!st) return;
      patch($("#dash-week"), weekHtml(st, days));
      patch($("#dash-people"), peopleHtml(st, days));
      patch($("#dash-digest"), digestHtml(st, dailies));
      patch($("#dash-tiles"), tilesHtml(st, days));
    };
    const paintRecent = () => { const st = status(); if (st && patch($("#dash-recent"), recentHtml(st, recent, recentErr))) player.mark($("#dash-recent")); };

    const loadRecent = async () => {
      const st = status();
      if (!st) return;
      try {
        const r = await api(`/api/transcript/${st.today.day}`, { signal: ctx.signal });
        const all = (r.sessions || []).flatMap((x) => x.utterances);
        rememberSpeakers(all, { idKey: "speaker_id", nameKey: "speaker" });
        recent = all.slice(-8).reverse();
        recentErr = "";
      } catch (e) { if (isAbort(e)) return; recentErr = e.message || String(e); }
      if (ctx.alive()) paintRecent();
    };
    const loadData = async ({ force = false } = {}) => {
      try { [days, dailies] = await Promise.all([loadDays({ force }), api("/api/daily", { signal: ctx.signal })]); }
      catch (e) { if (isAbort(e)) return; days = days || []; dailies = dailies || []; }
      if (ctx.alive()) paintData();
    };
    const refreshSoon = debounce(() => { loadRecent(); loadData({ force: true }); }, 400);

    on(ctx.el, "click", "[data-act], [data-job]", async (e, b) => {
      if (b.dataset.job) {
        const st = status();
        const n = b.dataset.job;
        await runJob(n, n === "daily" ? { day: (st && st.today.day) || today() } : {});
        return;
      }
      const a = b.dataset.act;
      if (a === "start") await startRecording(b);
      else if (a === "stop") await stopRecording();
      else if (a === "process") await runJob("process");
      else if (a === "retry") await runJob("process", { retry_failed: true });
    }, { signal: ctx.signal });
    on(ctx.el, "change", '[data-act="sys"]', async (e, sw) => { sw.disabled = true; await setSystemAudio(sw.checked); paintStatus(); }, { signal: ctx.signal });

    ctx.on("status", paintStatus);
    ctx.on("live:utterances", refreshSoon);
    ctx.on("utterances:changed", refreshSoon);
    ctx.on("live:daily", refreshSoon);
    ctx.onJob(null, refreshSoon);
    ctx.on("online", refreshSoon);
    ctx.on("profile:switched", refreshSoon);
    // the elapsed time in the hero ticks without re-rendering anything else
    ctx.poll(() => {
      const st = status(), el = $("[data-hero-time]");
      const t0 = tsMs(st && st.service && st.service.recorder_started_at);
      if (el && Number.isFinite(t0)) el.textContent = fmtClock((Date.now() - t0) / 1000);
    }, 1000);

    paintStatus();
    paintRecent();
    paintData();
    await Promise.all([loadRecent(), loadData()]);
  },
};
