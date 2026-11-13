// Command palette (Ctrl/⌘+K, or "/" when not typing).
//   registerCommand({id, title, group, keywords, icon, when: () => bool, run: () => void})
//   registerProvider(async (query, signal) => [{title, sub?, group, icon?, run}])      async results (record search)
//   openPalette("昨天")
// Built in here: jumping to a day ("昨天", "11-03", "2026-11-03"). Pages, actions and the record search are
// registered from main.js; later packages add their own commands with the same two functions.
import { html, when } from "../core/html.js";
import { $, make, render } from "../core/dom.js";
import { debounce } from "../core/util.js";
import { dayLabel, dayTitle, parseDayQuery } from "../core/format.js";
import { isAbort } from "../core/api.js";
import { icon } from "./icons.js";
import { spinner } from "./components.js";

const commands = [];
const providers = [];
let open = null;

export function registerCommand(cmd) {
  const i = commands.findIndex((c) => c.id === cmd.id);
  if (i >= 0) commands[i] = cmd; else commands.push(cmd);
  return () => { const k = commands.indexOf(cmd); if (k >= 0) commands.splice(k, 1); };
}
export function registerProvider(fn) {
  providers.push(fn);
  return () => { const k = providers.indexOf(fn); if (k >= 0) providers.splice(k, 1); };
}

/** Every whitespace-separated term of the query must appear in the title or the keywords. Exported for tests. */
export function matches(cmd, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return true;
  const hay = `${cmd.title || ""} ${cmd.keywords || ""}`.toLowerCase();
  return q.split(/\s+/).every((t) => hay.includes(t));
}

function dayItems(query) {
  const day = parseDayQuery(query);
  if (!day) return [];
  const label = `${dayTitle(day)} ${dayLabel(day)}`.trim();
  return [
    { title: `打开 ${label} 的记录`, sub: day, group: "跳转", icon: "doc", run: () => { location.hash = `#/transcript/${day}`; } },
    { title: `打开 ${label} 的日报`, sub: day, group: "跳转", icon: "calendar", run: () => { location.hash = `#/daily/${day}`; } },
  ];
}

export function openPalette(initial = "") {
  if (open) { open.input.focus(); return; }
  const root = $("#modal-root");
  const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const backdrop = make(html`<div class="palette-backdrop">
    <div class="palette" role="dialog" aria-modal="true" aria-label="搜索和命令">
      <div class="palette-top">${icon("search", 18)}
        <input class="palette-input" type="text" role="combobox" aria-expanded="true" aria-controls="palette-list" aria-autocomplete="list" placeholder="搜索记录，或输入命令、日期（昨天 / 11-03）" autocomplete="off" spellcheck="false">
        <kbd>Esc</kbd></div>
      <div class="palette-list" id="palette-list" role="listbox" aria-label="结果"></div>
      <div class="palette-foot"><span><kbd>↑</kbd><kbd>↓</kbd> 选择</span><span><kbd>Enter</kbd> 打开</span><span class="palette-status" aria-live="polite"></span></div>
    </div>
  </div>`);
  const input = backdrop.querySelector("input"), list = backdrop.querySelector(".palette-list"), statusEl = backdrop.querySelector(".palette-status");
  let items = [], active = 0, ctl = null, asyncItems = [], searching = false, closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    if (ctl) ctl.abort();
    search.cancel();
    backdrop.remove();
    open = null;
    if (returnFocus && document.contains(returnFocus)) { try { returnFocus.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
  };
  const run = (it) => { close(); try { const r = it.run(); if (r && r.catch) r.catch((e) => console.error("[palette]", e)); } catch (e) { console.error("[palette]", e); } };

  const paint = () => {
    const q = input.value;
    const local = [...dayItems(q), ...commands.filter((c) => (!c.when || c.when()) && matches(c, q))];
    items = [...local, ...asyncItems];
    if (active >= items.length) active = Math.max(0, items.length - 1);
    const groups = [];
    items.forEach((it, i) => { const g = it.group || "其他"; let e = groups.find((x) => x.name === g); if (!e) { e = { name: g, rows: [] }; groups.push(e); } e.rows.push([it, i]); });
    render(list, items.length
      ? html`${groups.map((g) => html`<div class="palette-group" role="group" aria-label="${g.name}"><div class="palette-gname">${g.name}</div>
          ${g.rows.map(([it, i]) => html`<div class="palette-item ${i === active ? "on" : ""}" role="option" id="pal-${i}" aria-selected="${i === active ? "true" : "false"}" data-i="${i}">
            <span class="palette-ic">${icon(it.icon || "arrow-right", 16)}</span>
            <span class="palette-text"><span class="palette-title">${it.title}</span>${when(it.sub, () => html`<span class="palette-sub">${it.sub}</span>`)}</span>
            ${when(it.hint, () => html`<kbd>${it.hint}</kbd>`)}</div>`)}</div>`)}`
      : html`<div class="palette-empty">${searching ? spinner("正在搜索记录…") : q.trim() ? "没有匹配的命令或记录" : "输入点什么"}</div>`);
    input.setAttribute("aria-activedescendant", items.length ? `pal-${active}` : "");
    statusEl.textContent = searching ? "正在搜索记录…" : "";
    const on = list.querySelector(".palette-item.on");
    if (on) on.scrollIntoView({ block: "nearest" });
  };

  const search = debounce(async () => {
    const q = input.value.trim();
    if (ctl) ctl.abort();
    asyncItems = [];
    if (q.length < 2 || !providers.length) { searching = false; paint(); return; }
    ctl = new AbortController();
    const my = ctl;
    searching = true;
    paint();
    try {
      const results = await Promise.all(providers.map((p) => Promise.resolve(p(q, my.signal)).catch((e) => { if (!isAbort(e)) console.warn("[palette] provider", e); return []; })));
      if (my.signal.aborted || closed) return;
      asyncItems = results.flat();
    } finally {
      if (ctl === my) { searching = false; if (!closed) paint(); }
    }
  }, 200);

  input.addEventListener("input", () => { active = 0; asyncItems = []; paint(); search(); });
  backdrop.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); if (items.length) { active = (active + 1) % items.length; paint(); } }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (items.length) { active = (active - 1 + items.length) % items.length; paint(); } }
    else if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); if (items[active]) run(items[active]); }
    else if (e.key === "Tab") { e.preventDefault(); input.focus(); }      // one focus stop: the input drives the list
  });
  let down = false;
  backdrop.addEventListener("mousedown", (e) => { down = e.target === backdrop; });
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop && down) { close(); return; }
    const row = e.target.closest(".palette-item");
    if (row && items[Number(row.dataset.i)]) run(items[Number(row.dataset.i)]);
  });
  list.addEventListener("mousemove", (e) => {
    const row = e.target.closest(".palette-item");
    if (row && Number(row.dataset.i) !== active) { active = Number(row.dataset.i); for (const el of list.querySelectorAll(".palette-item")) { const on = Number(el.dataset.i) === active; el.classList.toggle("on", on); el.setAttribute("aria-selected", on ? "true" : "false"); } input.setAttribute("aria-activedescendant", `pal-${active}`); }
  });
  root.appendChild(backdrop);
  open = { input, close };
  input.value = initial;
  paint();
  input.focus();
  if (initial) search();
}

export const paletteOpen = () => !!open;
export function closePalette() { if (open) open.close(); }
