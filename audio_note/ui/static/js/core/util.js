// Small pure helpers. No DOM access: node can import this file directly (tests/test_v1_frontend.py).

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
/** HTML-escape the five special characters; null / undefined become "". */
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
/** Finite number or the fallback (null / "" / NaN / Infinity are not numbers here). */
export const num = (v, fallback = null) => (v == null || v === "" || !Number.isFinite(Number(v)) ? fallback : Number(v));

let uidSeq = 0;
export const uid = (prefix = "id") => `${prefix}-${++uidSeq}`;

/** Trailing-edge debounce; the returned function has .cancel() and .flush(). */
export function debounce(fn, ms) {
  let timer = null, lastArgs = null;
  const run = () => { timer = null; const a = lastArgs; lastArgs = null; fn(...a); };
  const wrapped = (...args) => { lastArgs = args; if (timer) clearTimeout(timer); timer = setTimeout(run, ms); };
  wrapped.cancel = () => { if (timer) clearTimeout(timer); timer = null; lastArgs = null; };
  wrapped.flush = () => { if (timer) { clearTimeout(timer); run(); } };
  return wrapped;
}

/** {q: "a b", limit: 8, x: null, f: false} -> "?q=a%20b&limit=8&f=false"; empty -> "". */
export function qs(obj) {
  const parts = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (v === null || v === undefined || v === "") continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

export function groupBy(list, keyFn) {
  const out = new Map();
  for (const item of list) {
    const k = keyFn(item);
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(item);
  }
  return out;
}

/** Structural equality for JSON-like values (settings dirty tracking). */
export function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null || typeof a !== "object") return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
}

/** Read "a.b.c" from a nested object; undefined when any step is missing. */
export function getPath(obj, path) {
  let cur = obj;
  for (const k of String(path).split(".")) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = cur[k];
  }
  return cur;
}
/** Write "a.b.c" into a nested object, creating the intermediate objects. */
export function setPath(obj, path, value) {
  const keys = String(path).split(".");
  let cur = obj;
  for (const k of keys.slice(0, -1)) {
    if (cur[k] == null || typeof cur[k] !== "object") cur[k] = {};
    cur = cur[k];
  }
  cur[keys[keys.length - 1]] = value;
  return obj;
}

/**
 * Character diff of two short strings: [{op: "eq" | "ins" | "del" | "rep", a, b}] where `a` is the piece of the
 * old text and `b` the piece of the new one. Adjacent delete + insert are merged into one "rep".
 * Returns null when the inputs are too long for the quadratic table (callers then show the whole text as changed).
 */
export function diffChars(oldText, newText, maxCells = 250000) {
  const a = Array.from(String(oldText ?? "")), b = Array.from(String(newText ?? ""));
  const n = a.length, m = b.length;
  if (n * m > maxCells) return null;
  // longest common subsequence lengths, filled from the end so the walk below runs forwards
  const w = m + 1;
  const t = new Uint16Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      t[i * w + j] = a[i] === b[j] ? t[(i + 1) * w + j + 1] + 1 : Math.max(t[(i + 1) * w + j], t[i * w + j + 1]);
    }
  }
  const raw = [];
  const push = (op, ch) => { const last = raw[raw.length - 1]; if (last && last.op === op) last.s += ch; else raw.push({ op, s: ch }); };
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { push("eq", a[i]); i++; j++; }
    else if (t[(i + 1) * w + j] >= t[i * w + j + 1]) { push("del", a[i]); i++; }
    else { push("ins", b[j]); j++; }
  }
  while (i < n) push("del", a[i++]);
  while (j < m) push("ins", b[j++]);
  const out = [];
  for (let k = 0; k < raw.length; k++) {
    const r = raw[k], next = raw[k + 1];
    if (r.op === "eq") out.push({ op: "eq", a: r.s, b: r.s });
    else if (r.op === "del" && next && next.op === "ins") { out.push({ op: "rep", a: r.s, b: next.s }); k++; }
    else if (r.op === "ins" && next && next.op === "del") { out.push({ op: "rep", a: next.s, b: r.s }); k++; }
    else if (r.op === "del") out.push({ op: "del", a: r.s, b: "" });
    else out.push({ op: "ins", a: "", b: r.s });
  }
  return out;
}

/** Stable small hash of a string (fallback colour for a speaker known only by name). */
export function hashStr(s) {
  let h = 0;
  for (const ch of String(s || "")) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  return h;
}

/** Run `fn` and swallow its rejection (fire-and-forget refreshes). */
export const quiet = (p) => { try { const r = typeof p === "function" ? p() : p; if (r && r.catch) r.catch(() => {}); } catch (e) { /* ignore */ } };
