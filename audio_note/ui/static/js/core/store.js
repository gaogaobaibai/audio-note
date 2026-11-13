// Shared client state: a small observable key/value store, persisted preferences, and cached loaders for the
// payloads several pages need (speakers, config, days).
import { api } from "./api.js";
import { on as busOn } from "./bus.js";
import { num } from "./util.js";

const values = new Map();
const subs = new Map();

export const store = {
  /** keys in use: "status", "online", "live" ("sse" | "poll"), "minDbfs", "route" */
  get(key) { return values.get(key); },
  set(key, value) {
    const prev = values.get(key);
    values.set(key, value);
    const set = subs.get(key);
    if (set) for (const fn of [...set]) { try { fn(value, prev); } catch (e) { console.error(`[store] ${key}`, e); } }
    return value;
  },
  update(key, fn) { return this.set(key, fn(values.get(key))); },
  on(key, fn) {
    if (!subs.has(key)) subs.set(key, new Set());
    subs.get(key).add(fn);
    return () => subs.get(key).delete(fn);
  },
};

// ---- preferences (per browser): theme, collapsed rail, playback rate, chat options…
const PREFS_KEY = "audio-note:prefs:v1";
let prefsCache = null;
function readPrefs() {
  if (prefsCache) return prefsCache;
  try { const v = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"); prefsCache = v && typeof v === "object" && !Array.isArray(v) ? v : {}; }
  catch (e) { prefsCache = {}; }
  return prefsCache;
}
export const prefs = {
  get(name, fallback = undefined) { const v = readPrefs()[name]; return v === undefined ? fallback : v; },
  set(name, value) {
    const all = readPrefs();
    if (value === undefined) delete all[name]; else all[name] = value;
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(all)); } catch (e) { /* private mode / quota: keep it in memory */ }
    return value;
  },
};

// ---- status accessors (the status payload itself is kept fresh by core/status.js)
export const status = () => store.get("status") || null;
export const isReadOnly = () => { const st = status(); return !!(st && st.service && st.service.read_only); };
export const hasApiKey = () => { const st = status(); return !!(st && st.api_key); };
export const selfName = () => { const st = status(); return (st && st.self_name) || null; };
export const unknownLabel = () => { const st = status(); return (st && st.unknown_label) || store.get("unknownLabel") || "未知"; };
/** speakers.system_turn_share: at or above this share of system-audio energy a sentence is "the other side of a call". */
export const systemTurnShare = () => { const st = status(); return num(st && st.system_turn_share, 0.6); };
/** audio.min_speech_dbfs: sentences quieter than this are shown as far away. null = off, undefined = not known yet. */
export function minDbfs() {
  const st = status();
  if (st && st.min_speech_dbfs !== undefined) return num(st.min_speech_dbfs, null);
  return store.get("minDbfs");
}
export const setMinDbfs = (v) => store.set("minDbfs", v === undefined ? undefined : num(v, null));
export const inboxOpen = () => {
  const st = status();
  if (!st) return undefined;
  if (st.inbox_open !== undefined && st.inbox_open !== null) return num(st.inbox_open, 0);
  if (st.inbox && st.inbox.open !== undefined) return num(st.inbox.open, 0);
  return undefined;                                   // the server has no inbox yet: the bell stays hidden
};

// ---- cached loaders
function cached(loader, { ttlMs = 0 } = {}) {
  let value = null, at = 0, pending = null, gen = 0;
  const fn = async ({ force = false } = {}) => {
    if (!force && value !== null && (!ttlMs || Date.now() - at < ttlMs)) return value;
    if (pending) return pending;
    const my = gen;
    pending = loader().then((v) => { if (my === gen) { value = v; at = Date.now(); } return v; }).finally(() => { pending = null; });
    return pending;
  };
  fn.invalidate = () => { value = null; gen++; };
  fn.peek = () => value;
  return fn;
}

/** GET /api/speakers shaped for pickers: {list, byId, byName, names, selfId, self, choices, threshold, unknownLabel, raw}. */
export const loadSpeakers = cached(async () => {
  const d = await api("/api/speakers");
  const list = Array.isArray(d.speakers) ? d.speakers : [];
  const byId = new Map(list.map((s) => [s.id, s]));
  const byName = new Map(list.map((s) => [s.name, s]));
  if (d.unknown_label) store.set("unknownLabel", d.unknown_label);
  return { list, byId, byName, names: list.map((s) => s.name), selfId: d.self_id ?? null, self: byId.get(d.self_id) || null,
    choices: d.relation_choices || [], threshold: d.identify_threshold, unknownLabel: d.unknown_label, raw: d };
});

/** GET /api/config (settings page, chat defaults). */
export const loadConfig = cached(async () => {
  const cfg = await api("/api/config");
  const a = cfg && cfg.values && cfg.values.audio;
  if (a && store.get("minDbfs") === undefined) setMinDbfs(a.min_speech_dbfs);
  return cfg;
});

/** GET /api/days, shared by the dashboard, the timeline and the palette for a few seconds. */
export const loadDays = cached(() => api("/api/days"), { ttlMs: 4000 });

busOn("speakers:changed", () => loadSpeakers.invalidate());
busOn("live:speakers", () => loadSpeakers.invalidate());
busOn("config:saved", () => { loadConfig.invalidate(); loadSpeakers.invalidate(); });
busOn("live:config", () => loadConfig.invalidate());
busOn("utterances:changed", () => loadDays.invalidate());
busOn("live:utterances", () => loadDays.invalidate());
busOn("job:done", () => { loadDays.invalidate(); loadSpeakers.invalidate(); });
busOn("profile:switched", () => { loadSpeakers.invalidate(); loadConfig.invalidate(); loadDays.invalidate(); store.set("minDbfs", undefined); });

/** Make sure minDbfs() is known (one /api/config read at most when the status payload does not carry it). */
export async function ensureMinDbfs() {
  if (minDbfs() !== undefined) return minDbfs();
  try { await loadConfig(); } catch (e) { /* fall through */ }
  if (store.get("minDbfs") === undefined) store.set("minDbfs", null);
  return minDbfs();
}
