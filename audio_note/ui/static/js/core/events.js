// Live updates: a client for GET /api/events (server-sent events, SPEC 4.3).
// Every server event {id, type, ts, …} is re-emitted on the bus as "live:<type>". The connection is made with
// fetch (not EventSource) so that "the endpoint does not exist" (an older server, or ui.events: false) can be told
// apart from "the connection dropped": the former falls back to status polling for good (re-probing rarely), the
// latter reconnects with exponential back-off. store "live" says which mode is active: "sse" | "poll".
import { readEventStream } from "./api.js";
import { emit } from "./bus.js";
import { store } from "./store.js";

const RETRY_MIN = 1000, RETRY_MAX = 30000, MISSING_RETRY = 5 * 60 * 1000, SILENCE_MS = 45000;
let ctl = null, attempts = 0, lastId = "", retryTimer = null, watchdog = null, started = false, stopped = false, hellos = 0;

function setMode(mode) {
  if (store.get("live") !== mode) { store.set("live", mode); emit("live:mode", mode); }
}

function retryIn(ms) {
  if (stopped) return;
  if (retryTimer) clearTimeout(retryTimer);
  retryTimer = setTimeout(() => { retryTimer = null; connect(); }, ms);
}
/** 1 s, 2 s, 4 s … capped at 30 s, with a little jitter so several tabs do not reconnect in step. */
export function backoffDelay(n, rand = Math.random()) {
  const base = Math.min(RETRY_MAX, RETRY_MIN * 2 ** Math.max(0, n));
  return Math.round(base * (0.85 + 0.3 * rand));
}

function arm() {
  if (watchdog) clearTimeout(watchdog);
  // the server sends a comment heartbeat every 15 s: three missed in a row means the connection is dead
  watchdog = setTimeout(() => { if (ctl) ctl.abort(); }, SILENCE_MS);
}

function dispatch({ event, data, id }) {
  let obj = null;
  if (data) { try { obj = JSON.parse(data); } catch (e) { obj = null; } }
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) obj = { value: obj };
  const type = String(obj.type || event || "message");
  // the frame id / "seq" is the bus sequence number (an event's own "id" field is e.g. a segment id)
  const seq = id || (obj.seq !== undefined && obj.seq !== null ? String(obj.seq) : "");
  if (seq) lastId = seq;
  if (!obj.type) obj.type = type;
  attempts = 0;
  emit(`live:${type}`, obj);
  emit("live", obj);
  if (type === "hello") {
    // every connection starts with a hello; after the first one it means "you may have missed things"
    if (hellos++ > 0 || obj.gap) emit("resync", obj);
  }
}

async function connect() {
  if (stopped || ctl) return;
  ctl = new AbortController();
  const my = ctl;
  let res = null;
  try {
    const headers = { Accept: "text/event-stream" };
    if (lastId) headers["Last-Event-ID"] = lastId;
    res = await fetch(`/api/events${lastId ? `?since=${encodeURIComponent(lastId)}` : ""}`, { headers, signal: my.signal, cache: "no-store" });
  } catch (e) {
    ctl = null;
    setMode("poll");
    retryIn(backoffDelay(attempts++));
    return;
  }
  const type = res.headers.get("content-type") || "";
  if (!res.ok || !type.includes("text/event-stream") || !res.body) {
    try { my.abort(); } catch (e) { /* ignore */ }
    ctl = null;
    setMode("poll");
    // 4xx / not an event stream: this server has no event channel; 5xx: it may be starting up
    retryIn(res.status >= 500 ? backoffDelay(attempts++) : MISSING_RETRY);
    return;
  }
  setMode("sse");
  arm();
  try {
    await readEventStream(res, { onEvent: dispatch, onActivity: arm });
  } catch (e) {
    /* dropped or aborted: reconnect below */
  }
  if (watchdog) { clearTimeout(watchdog); watchdog = null; }
  if (ctl === my) ctl = null;
  if (stopped) return;
  setMode("poll");
  retryIn(backoffDelay(attempts++));
}

/** Start (once). Safe to call before the first status read. */
export function startLive() {
  if (started) return;
  started = true;
  store.set("live", "poll");
  window.addEventListener("pagehide", () => { stopped = true; if (ctl) ctl.abort(); });
  window.addEventListener("pageshow", (e) => { if (e.persisted) { stopped = false; connect(); } });
  window.addEventListener("online", () => { if (!ctl) { attempts = 0; retryIn(200); } });
  connect();
}

export const liveMode = () => store.get("live") || "poll";
