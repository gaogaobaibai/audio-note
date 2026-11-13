// Hash router with a page lifecycle.
//   definePage("transcript", {title, icon, load: () => import("../pages/timeline.js"), aliases: ["timeline"]})
//   page module: export default { async mount(ctx) { …; return cleanup? }, async update?(ctx) {…} }
//   ctx = { el (this mount's own container), page, params: string[], query: URLSearchParams, signal, alive(), on(busType, fn), onJob(names, fn),
//           poll(fn, ms), setTitle(text), guard(fn), navigate(path, opts), setQuery(obj) }
// ctx.signal aborts when the page is left: pass it to api() and on() so nothing from an old page can touch the new
// one. update(ctx) runs instead of a re-mount when only the params / query changed (switching the day).
import { html } from "./html.js";
import { render } from "./dom.js";
import { on as busOn, emit } from "./bus.js";
import { store } from "./store.js";
import { isAbort } from "./api.js";
import { icon } from "../ui/icons.js";
import { skeleton } from "../ui/components.js";

const pages = new Map();        // id -> definition
const aliases = new Map();      // alias -> {id, params: [...]}
let mainEl = null;
let current = null;             // {def, ctx, ctl, cleanup, hash, guard, mod}
let defaultPage = "dashboard";
let seq = 0;

export function definePage(id, def) {
  pages.set(id, { id, nav: true, ...def });
  for (const a of def.aliases || []) aliases.set(a, { id, params: [] });
  return pages.get(id);
}
/** alias("logs", "settings", ["logs"]): #/logs opens the settings page with params ["logs"]. */
export function alias(name, id, params = []) { aliases.set(name, { id, params }); }
export const pageList = () => [...pages.values()];
export const pageDef = (id) => pages.get(id) || null;

const safeDecode = (s) => { try { return decodeURIComponent(s); } catch (e) { return s; } };

/** "#/transcript/2026-11-10?u=42" -> {page: "transcript", params: ["2026-11-10"], query: URLSearchParams, known} */
export function parseHash(hash) {
  const h = String(hash || "").replace(/^#\/?/, "");
  const qi = h.indexOf("?");
  const path = qi < 0 ? h : h.slice(0, qi);
  const query = new URLSearchParams(qi < 0 ? "" : h.slice(qi + 1));
  const segs = path.split("/").filter((s) => s !== "").map(safeDecode);
  let page = segs.shift() || defaultPage;
  let params = segs;
  let known = pages.has(page);
  if (!known && aliases.has(page)) { const a = aliases.get(page); page = a.id; params = [...a.params, ...params]; known = true; }
  if (!known) { page = defaultPage; params = []; }
  return { page, params, query, known };
}

/** hrefFor("transcript", "2026-11-10", {u: 42}) -> "#/transcript/2026-11-10?u=42" */
export function hrefFor(page, ...rest) {
  let query = null;
  if (rest.length && rest[rest.length - 1] && typeof rest[rest.length - 1] === "object") query = rest.pop();
  const path = [page, ...rest.filter((x) => x !== null && x !== undefined && x !== "")].map((s) => encodeURIComponent(String(s))).join("/");
  const q = query ? new URLSearchParams(Object.entries(query).filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => [k, String(v)])).toString() : "";
  return `#/${path}${q ? `?${q}` : ""}`;
}

export function navigate(path, { replace = false } = {}) {
  const hash = path.startsWith("#") ? path : `#/${path.replace(/^\//, "")}`;
  if (hash === location.hash) { handle(); return; }
  if (replace) { history.replaceState(null, "", hash); handle(); } else location.hash = hash;
}

export const currentRoute = () => (current ? { page: current.def.id, params: current.ctx.params, query: current.ctx.query } : null);
/** Re-mount the current page (after a profile switch). */
export function reload() { if (current) { current.forceRemount = true; current.guard = null; } return handle(); }

function teardown() {
  if (!current) return;
  const c = current;
  current = null;
  try { c.ctl.abort(); } catch (e) { /* ignore */ }
  for (const off of c.offs) { try { off(); } catch (e) { /* ignore */ } }
  for (const t of c.timers) clearInterval(t);
  if (typeof c.cleanup === "function") { try { c.cleanup(); } catch (e) { console.error("[router] cleanup", e); } }
}

function makeCtx(def, route, entry) {
  const ctx = {
    el: entry.box,
    page: def.id,
    params: route.params,
    query: route.query,
    signal: entry.ctl.signal,
    alive: () => !entry.ctl.signal.aborted,
    /** Bus subscription that ends with the page. */
    on(type, fn) { const off = busOn(type, (...a) => { if (!entry.ctl.signal.aborted) return fn(...a); }); entry.offs.push(off); return off; },
    /** fn(name, job) when one of the named background jobs finished (ok or failed); no names = any job. */
    onJob(names, fn) {
      const want = names && names.length ? new Set(names) : null;
      const h = ({ name, job }) => { if (!want || want.has(name)) return fn(name, job); };
      this.on("job:done", h);
      this.on("job:error", h);
    },
    /** Interval that ends with the page and pauses while the tab is hidden. */
    poll(fn, ms) {
      const t = setInterval(() => { if (!document.hidden && !entry.ctl.signal.aborted) { try { const r = fn(); if (r && r.catch) r.catch(() => {}); } catch (e) { /* ignore */ } } }, ms);
      entry.timers.push(t);
      return () => clearInterval(t);
    },
    setTitle(text) { if (!entry.ctl.signal.aborted) emit("route:title", text || def.title); },
    /** fn(target: {page, params, query}) -> boolean | Promise<boolean>: may the user go there? (unsaved settings) */
    guard(fn) { entry.guard = fn; },
    navigate,
    /** Change the query part of the current hash without leaving the page. */
    setQuery(obj) { navigate(hrefFor(def.id, ...ctx.params, obj || {}), { replace: true }); },
  };
  return ctx;
}

function errorState(e, retry) {
  const msg = e && e.message ? e.message : String(e);
  return html`<div class="page"><div class="empty" role="alert">
    <div class="empty-ic warn">${icon("alert", 28)}</div>
    <div class="empty-title">这一页没能加载</div>
    <div class="empty-text">${msg}</div>
    ${retry ? html`<div class="empty-acts"><button class="btn" type="button" data-router-retry>重试</button></div>` : ""}
  </div></div>`;
}

let handling = Promise.resolve();
/** Serialise route handling so a fast double navigation cannot interleave mounts. */
function handle() { handling = handling.then(doHandle, doHandle); return handling; }

async function doHandle() {
  const hash = location.hash || `#/${defaultPage}`;
  const route = parseHash(hash);
  const def = pages.get(route.page);
  // unsaved-changes guard of the page being left
  if (current && current.guard && current.hash !== hash) {
    let ok = true;
    try { ok = await current.guard({ page: route.page, params: route.params, query: route.query }); } catch (e) { ok = true; }
    if (!current) return;
    if (!ok) { if (location.hash !== current.hash) location.replace(current.hash); return; }
    if (location.hash !== hash && (location.hash || `#/${defaultPage}`) !== hash) return;      // the user navigated again while the question was open
  }
  if (current && current.def === def && !current.forceRemount && current.mounted && current.mod && typeof current.mod.update === "function") {
    if (current.hash === hash) return;
    const entry = current;
    entry.hash = hash;
    entry.ctx.params = route.params;
    entry.ctx.query = route.query;
    store.set("route", { page: def.id, params: route.params });
    emit("route", { page: def.id, params: route.params, query: route.query, title: def.title });
    // not awaited: a slow page must never hold up the next navigation
    Promise.resolve().then(() => entry.mod.update(entry.ctx)).catch((e) => {
      if (current !== entry || isAbort(e)) return;
      console.error("[router] update", e);
      render(entry.box, errorState(e, true));
    });
    return;
  }
  if (current && current.def === def && current.hash === hash && !current.forceRemount) return;
  const samePage = !!(current && current.def === def);
  teardown();
  const entry = { def, ctl: new AbortController(), offs: [], timers: [], cleanup: null, hash, guard: null, mod: null, mounted: false, id: ++seq };
  // every mount gets its own container: whatever a page that was already left still renders lands in a detached node
  entry.box = document.createElement("div");
  entry.box.className = "route";
  entry.ctx = makeCtx(def, route, entry);
  current = entry;
  store.set("route", { page: def.id, params: route.params });
  emit("route", { page: def.id, params: route.params, query: route.query, title: def.title });
  render(entry.box, html`<div class="page">${skeleton("page")}</div>`);
  mainEl.replaceChildren(entry.box);
  if (!samePage) window.scrollTo(0, 0);
  mount(entry, samePage);
}

/** Load and mount the page of `entry` (not awaited by the router: navigation stays responsive while a page loads). */
async function mount(entry, samePage) {
  try {
    const mod = (await entry.def.load()).default;
    if (current !== entry) return;
    entry.mod = mod;
    const cleanup = await mod.mount(entry.ctx);
    if (current !== entry) { if (typeof cleanup === "function") { try { cleanup(); } catch (e) { /* ignore */ } } return; }
    entry.cleanup = cleanup;
    entry.mounted = true;
    // after following a navigation link, keyboard focus continues in the page, not in the menu
    const a = document.activeElement;
    if (!samePage && a && a !== document.body && a.closest && a.closest("#nav, #tabbar, .brand")) { try { mainEl.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
    // the hash may have changed (same page) while this mount was still loading: apply it now
    if ((location.hash || `#/${defaultPage}`) !== entry.hash) handle();
  } catch (e) {
    if (current !== entry || isAbort(e)) return;
    console.error("[router] mount", e);
    render(entry.box, errorState(e, true));
  }
}

export function start(el, { fallback = "dashboard" } = {}) {
  mainEl = el;
  defaultPage = fallback;
  window.addEventListener("hashchange", () => { handle(); });
  el.addEventListener("click", (e) => { if (e.target.closest && e.target.closest("[data-router-retry]")) reload(); });
  return handle();
}
