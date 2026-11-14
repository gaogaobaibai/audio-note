// Status loop: keeps store "status" fresh, detects finished jobs, tracks online / offline.
// With the event stream connected (core/events.js) the status is re-read when the server says something changed
// and only every 30 s otherwise; without it the loop polls every 5 s like the old UI did (10 s / 30 s back-off while
// the server is unreachable, nothing while the tab is hidden).
import { api, isAbort } from "./api.js";
import { emit, on as busOn } from "./bus.js";
import { store } from "./store.js";
import { debounce } from "./util.js";
import { fmtDb } from "./format.js";
import { toast, toastErr } from "../ui/toast.js";

const POLL_MS = 5000, LIVE_POLL_MS = 30000, BACKOFF = [5000, 10000, 30000];
let timer = null, failures = 0, inflight = null;
const jobsSeen = {};             // job name -> finished_at already reported
const startedHere = new Set();   // jobs submitted from this tab: their result is announced with a toast
let firstStatus = true;

/** One-line summary of a job result for toasts and the job list (starts with "：" or is empty). */
export function jobSummary(name, r) {
  if (!r || typeof r !== "object") return "";
  if (name === "process") return `：处理 ${r.processed ?? 0} 段，${r.utterances ?? 0} 条语句${r.failed ? `，${r.failed} 段失败` : ""}`;
  if (name === "daily") return `：${r.day ?? ""}${r.cached ? "（内容没变，沿用已有日报）" : ""}${r.partial ? "（部分时段失败，稍后会重试）" : ""}`;
  if (name === "cleanup") return `：复检 ${r.rechecked ?? 0}，删除无语音 ${r.no_speech ?? 0}，过期 ${r.expired ?? 0}`;
  if (name === "reindex") return `：关键词 ${r.keywords ?? 0} 条${r.vectors != null ? `，向量 ${r.vectors} 条` : ""}`;
  if (name === "relabel") return `：${r.segments ?? 0} 段，${r.changed ?? 0} 条语句变化`;
  if (name === "push") return `：${Object.entries(r.channels || {}).map(([k, v]) => `${k} ${v ? "失败" : "成功"}`).join("，")}`;
  if (name === "voices") return `：分析 ${r.segments ?? 0} 段录音，${r.clusters ?? 0} 个声音${r.skipped ? `，${r.skipped} 段跳过` : ""}`;
  if (name === "relations") return `：检查 ${r.speakers ?? 0} 人，写入 ${r.applied ?? 0} 条关系`;
  if (name === "loudness") return `：补算 ${r.computed ?? 0} 句音量，${r.threshold != null ? `${r.below ?? 0} 句低于门槛 ${fmtDb(r.threshold)}` : "门槛未设置（没有句子会被滤掉）"}`;
  return "";
}

function detectJobs(st) {
  const jobs = (st && st.jobs) || {};
  for (const [name, job] of Object.entries(jobs)) {
    if (firstStatus) { jobsSeen[name] = job.finished_at; continue; }
    if (!job.finished_at || job.finished_at === jobsSeen[name]) continue;
    jobsSeen[name] = job.finished_at;
    const mine = startedHere.delete(name);
    if (job.error) {
      if (mine) toast(`${job.label || name}失败：${job.error}`, { kind: "err", ms: 8000 });
      emit("job:error", { name, job });
    } else {
      if (mine) toast(`${job.label || name}完成${jobSummary(name, job.result)}`, { kind: "ok", ms: 5000 });
      emit("job:done", { name, job });
    }
  }
  firstStatus = false;
}

/** Without the event stream, new sentences are noticed by the counters moving. */
function detectActivity(prev, st) {
  if (!prev || store.get("live") === "sse") return;
  const a = prev.total || {}, b = st.total || {};
  const day = st.today && st.today.day;
  if (a.utterances !== b.utterances || (prev.today && st.today && prev.today.utterances !== st.today.utterances) || prev.last_process_at !== st.last_process_at) {
    emit("live:utterances", { day, synthetic: true });
  }
  if (prev.last_daily_at !== st.last_daily_at) emit("live:daily", { day: st.last_daily_day, synthetic: true });
}

/** Read /api/status once (concurrent callers share the request). Resolves with the status or null when offline. */
export function refreshStatus() {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const st = await api("/api/status", { timeoutMs: 15000 });
      const prev = store.get("status") || null;
      const wasOffline = store.get("online") === false;
      failures = 0;
      // a profile switch elsewhere (another tab): forget what we knew about the old profile's jobs
      const switched = !!(prev && prev.profile && st.profile && prev.profile.name !== st.profile.name);
      if (switched) firstStatus = true;
      store.set("status", st);
      if (store.get("online") !== true) store.set("online", true);
      detectJobs(st);
      if (switched) emit("profile:switched", { name: st.profile.name });
      emit("status", st, prev);
      if (!switched) detectActivity(prev, st);
      if (wasOffline) emit("online");
      return st;
    } catch (e) {
      if (isAbort(e)) return store.get("status") || null;
      failures++;
      if (store.get("online") !== false) { store.set("online", false); emit("offline", e); }
      return null;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

/** The current status, reading it first when this is the very beginning of the session. May be null when offline. */
export async function ensureStatus() {
  return store.get("status") || (await refreshStatus());
}

function nextDelay() {
  if (failures) return BACKOFF[Math.min(failures - 1, BACKOFF.length - 1)];
  return store.get("live") === "sse" ? LIVE_POLL_MS : POLL_MS;
}
function schedule() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(async () => {
    timer = null;
    if (!document.hidden) await refreshStatus();
    schedule();
  }, nextDelay());
}

/** Debounced refresh for bursts of server events. */
export const refreshSoon = debounce(() => { refreshStatus(); }, 250);

export function startStatusLoop() {
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { refreshStatus(); schedule(); } });
  window.addEventListener("online", () => { refreshStatus(); });
  store.on("live", () => schedule());
  // server events that change what /api/status reports
  for (const t of ["hello", "segment", "utterances", "job", "daily", "speakers", "config", "profile", "session"]) {
    busOn(`live:${t}`, (ev) => { if (!(ev && ev.synthetic)) refreshSoon(); });
  }
  busOn("live:recording", (ev) => {
    const st = store.get("status");
    if (st && st.service && ev && typeof ev.on === "boolean") {
      const next = { ...st, service: { ...st.service, recording: ev.on, is_speaking: ev.on ? !!(ev.speaking ?? st.service.is_speaking) : false,
        paused_for_playback: ev.on ? !!(ev.paused ?? st.service.paused_for_playback) : false } };
      store.set("status", next);
      emit("status", next, st);
    }
    refreshSoon();
  });
  busOn("live:inbox", (ev) => {
    const st = store.get("status");
    if (st && ev && ev.open !== undefined) {
      const next = { ...st, inbox_open: Number(ev.open) || 0 };
      store.set("status", next);
      emit("status", next, st);
    }
    emit("inbox:changed", { open: ev && ev.open });
  });
  const first = refreshStatus();
  schedule();
  return first;
}

/**
 * Submit a background job. Resolves with {ok, job} or undefined when it was refused (the error is shown as a toast).
 * The job's completion is announced by the status loop (toast + "job:done" / "job:error" bus events).
 */
export async function runJob(name, body = {}, { quiet = false } = {}) {
  try {
    startedHere.add(name);
    const r = await api(`/api/jobs/${encodeURIComponent(name)}`, { method: "POST", json: body });
    if (!quiet) toast(`${(r.job && r.job.label) || name}已开始`);
    await refreshStatus();
    return r;
  } catch (e) {
    startedHere.delete(name);
    toastErr(e);
    return undefined;
  }
}

export const jobOf = (name) => { const st = store.get("status"); return (st && st.jobs && st.jobs[name]) || null; };
export const jobRunning = (name) => { const j = jobOf(name); return !!(j && j.running); };
