// fetch helpers for the local JSON API (audio_note/ui): api(), ndjson() streams, sse() streams, audio URLs.
import { qs } from "./util.js";

export class ApiError extends Error {
  constructor(message, { status = 0, data = null, network = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;      // HTTP status; 0 = the request never got an answer
    this.data = data;          // parsed JSON error body ({error, mismatch?, score?, …}) or null
    this.network = network;    // true: could not reach the server at all
  }
}

const OFFLINE = "无法连接到 audio-note（程序是否已退出？）";
export const isAbort = (e) => !!e && (e.name === "AbortError" || e.aborted === true);

/** Merge an optional caller signal with an optional timeout into one AbortSignal (+ a cleanup function). */
function linkSignal(signal, timeoutMs) {
  if (!timeoutMs) return { signal, done: () => {}, timedOut: () => false };
  const ctl = new AbortController();
  let timed = false;
  const timer = setTimeout(() => { timed = true; ctl.abort(); }, timeoutMs);
  const onAbort = () => ctl.abort();
  if (signal) { if (signal.aborted) ctl.abort(); else signal.addEventListener("abort", onAbort, { once: true }); }
  return { signal: ctl.signal, timedOut: () => timed, done: () => { clearTimeout(timer); if (signal) signal.removeEventListener("abort", onAbort); } };
}

/**
 * api("/api/days") / api("/api/jobs/process", {method: "POST", json: {}}) / api(url, {body: blob}).
 * Resolves with the parsed JSON (or the Response for a non-JSON answer). Rejects with ApiError — except when the
 * caller's signal aborted, which rejects with the browser's AbortError (check with isAbort()).
 */
export async function api(path, { method = "GET", query, json, body, contentType, signal, timeoutMs } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(json); }
  if (body !== undefined) { init.body = body; init.headers["Content-Type"] = contentType || "application/octet-stream"; }
  const link = linkSignal(signal, timeoutMs);
  if (link.signal) init.signal = link.signal;
  let res;
  try {
    res = await fetch(path + (query ? qs(query) : ""), init);
  } catch (e) {
    link.done();
    if (signal && signal.aborted) throw e;
    if (link.timedOut()) throw new ApiError("请求超时", { network: true });
    throw new ApiError(OFFLINE, { network: true });
  }
  try {
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("json")) {
      if (!res.ok) throw new ApiError(`${res.status} ${res.statusText}`.trim(), { status: res.status });
      return res;
    }
    let data = null;
    try { data = await res.json(); } catch (e) { if (signal && signal.aborted) throw e; data = null; }
    if (!res.ok) throw new ApiError((data && data.error) || `${res.status} ${res.statusText}`.trim(), { status: res.status, data });
    return data;
  } finally {
    link.done();
  }
}

/** 400 {"error", "mismatch": true, "score", "threshold", "name"}: the voice does not resemble the person's samples. */
export const isMismatch = (e) => !!(e && e.data && e.data.mismatch);

/**
 * POST a JSON body and yield the newline-delimited JSON events of the answer (the chat stream). Malformed lines
 * are skipped; a non-2xx answer rejects with ApiError.
 */
export async function* ndjson(path, { json, signal } = {}) {
  let res;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(json ?? {}), signal });
  } catch (e) {
    if (signal && signal.aborted) throw e;
    throw new ApiError(OFFLINE, { network: true });
  }
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new ApiError((data && data.error) || `${res.status} ${res.statusText}`.trim(), { status: res.status, data });
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const parse = (line) => { try { const v = JSON.parse(line); return v && typeof v === "object" ? v : null; } catch (e) { return null; } };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = parse(line);
        if (ev) yield ev;
      }
    }
    const tail = (buf + dec.decode()).trim();
    if (tail) { const ev = parse(tail); if (ev) yield ev; }
  } finally {
    try { reader.cancel(); } catch (e) { /* ignore */ }
  }
}

/**
 * Parse a text/event-stream body. Calls onEvent({event, data, id}) per dispatched event and onActivity() for every
 * chunk (comments / heartbeats included). Resolves when the stream ends; rejects on network errors / abort.
 * Kept separate from EventSource so that a missing endpoint (404) can be told apart from a dropped connection.
 */
export async function readEventStream(res, { onEvent, onActivity } = {}) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", event = "", data = [], id = "";
  const dispatch = () => {
    if (data.length || event) { try { onEvent && onEvent({ event: event || "message", data: data.join("\n"), id }); } catch (e) { /* a listener must not kill the stream */ } }
    event = ""; data = [];
  };
  const line = (ln) => {
    if (ln === "") { dispatch(); return; }
    if (ln[0] === ":") return;                                   // comment (heartbeat)
    const i = ln.indexOf(":");
    const field = i < 0 ? ln : ln.slice(0, i);
    let value = i < 0 ? "" : ln.slice(i + 1);
    if (value[0] === " ") value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") id = value;
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (onActivity) onActivity();
    buf += dec.decode(value, { stream: true });
    let m;
    while ((m = /\r\n|\n|\r/.exec(buf))) {
      if (m[0] === "\r" && m.index === buf.length - 1) break;    // might be the first half of \r\n
      line(buf.slice(0, m.index));
      buf = buf.slice(m.index + m[0].length);
    }
  }
}

export const audioUrl = {
  utterance: (id, pad = 0.3) => `/api/audio/utterance/${encodeURIComponent(id)}?pad=${pad}`,
  segment: (id, from, to) => `/api/audio/segment/${encodeURIComponent(id)}?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`,
  sample: (id) => `/api/speakers/samples/${encodeURIComponent(id)}/audio`,
  negative: (id) => `/api/speakers/negatives/${encodeURIComponent(id)}/audio`,
};
