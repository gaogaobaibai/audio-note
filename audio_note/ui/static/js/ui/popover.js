// Anchored popovers: overflow menus ("⋯") and small panels (recording details).
//   const id = await openMenu(button, [{id: "rename", label: "重命名", icon: "edit"}, "-", {id: "delete", label: "删除", danger: true}])
//   const pop = openPopover(button, html`…`, {onMount}); pop.close()
// Popovers live in #pop-root above dialogs, close on Esc / outside click / scroll / resize, return focus to their
// anchor, and menus are keyboard navigable (arrows, Home / End, Enter).
import { html, when } from "../core/html.js";
import { $, $$, make } from "../core/dom.js";
import { icon } from "./icons.js";

let openOne = null;      // only one popover at a time

function place(el, anchor, align) {
  const r = anchor.getBoundingClientRect();
  const w = el.offsetWidth, h = el.offsetHeight;
  const vw = document.documentElement.clientWidth, vh = window.innerHeight;
  let left = align === "start" ? r.left : r.right - w;
  left = Math.max(8, Math.min(left, vw - w - 8));
  let top = r.bottom + 6;
  if (top + h > vh - 8 && r.top - h - 6 > 8) top = r.top - h - 6;      // flip above when there is no room below
  top = Math.max(8, Math.min(top, vh - h - 8));
  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;
}

/** Generic popover. Returns {el, close(value), closed: Promise<value>}. */
export function openPopover(anchor, body, { align = "end", onMount, className = "", role = "dialog", label = "" } = {}) {
  if (openOne) openOne.close(undefined);
  const root = $("#pop-root");
  const el = make(html`<div class="pop ${className}" role="${role}" ${label ? html`aria-label="${label}"` : ""} tabindex="-1">${body}</div>`);
  let settle, closed = false;
  const done = new Promise((r) => { settle = r; });
  const close = (value) => {
    if (closed) return;
    closed = true;
    document.removeEventListener("mousedown", onDown, true);
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", onMove);
    window.removeEventListener("scroll", onScroll, true);
    el.remove();
    if (openOne === api) openOne = null;
    if (anchor) anchor.setAttribute("aria-expanded", "false");
    if (anchor && document.contains(anchor) && (value === undefined || value === null || api.refocus)) { try { anchor.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
    settle(value);
  };
  const onDown = (e) => { if (!el.contains(e.target) && !(anchor && anchor.contains(e.target))) close(undefined); };
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(undefined); }
    else if (e.key === "Tab" && !el.contains(e.target)) close(undefined);
  };
  const onMove = () => close(undefined);
  const onScroll = (e) => { if (!el.contains(e.target)) close(undefined); };
  const api = { el, close, closed: done, refocus: true };
  root.appendChild(el);
  place(el, anchor, align);
  if (anchor) { anchor.setAttribute("aria-haspopup", role === "menu" ? "menu" : "dialog"); anchor.setAttribute("aria-expanded", "true"); }
  openOne = api;
  // listeners go on after the current event finished, or the click that opened the popover would close it
  setTimeout(() => {
    if (closed) return;
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onMove);
    window.addEventListener("scroll", onScroll, true);
  }, 0);
  if (onMount) onMount(api);
  if (!closed) { const f = el.querySelector("[data-autofocus], button:not([disabled]), input, select, a[href]") || el; try { f.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
  return api;
}

/**
 * Menu. items: {id, label, icon?, danger?, disabled?, hint?, checked?} or "-" (separator); falsy entries are skipped.
 * Resolves with the picked id, or undefined when dismissed.
 */
export function openMenu(anchor, items, { align = "end", label = "更多操作" } = {}) {
  const list = items.filter(Boolean);
  const body = html`${list.map((it, i) => (it === "-"
    ? html`<div class="menu-sep" role="separator"></div>`
    : html`<button type="button" class="menu-item ${it.danger ? "danger" : ""}" role="${it.checked !== undefined ? "menuitemradio" : "menuitem"}" ${it.checked !== undefined ? html`aria-checked="${it.checked ? "true" : "false"}"` : ""} data-i="${i}" ${it.disabled ? html`disabled` : ""}>
        <span class="menu-ic">${it.checked ? icon("check", 16) : it.icon ? icon(it.icon, 16) : ""}</span>
        <span class="menu-label">${it.label}${when(it.hint, () => html`<small>${it.hint}</small>`)}</span>
      </button>`))}`;
  const pop = openPopover(anchor, body, {
    align, role: "menu", label, className: "menu",
    onMount: (api) => {
      const buttons = () => $$(".menu-item:not([disabled])", api.el);
      api.el.addEventListener("click", (e) => {
        const b = e.target.closest(".menu-item");
        if (b && !b.disabled) api.close(list[Number(b.dataset.i)].id);
      });
      api.el.addEventListener("keydown", (e) => {
        const bs = buttons();
        const i = bs.indexOf(document.activeElement);
        let next = -1;
        if (e.key === "ArrowDown") next = (i + 1) % bs.length;
        else if (e.key === "ArrowUp") next = (i - 1 + bs.length) % bs.length;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = bs.length - 1;
        if (next >= 0 && bs[next]) { e.preventDefault(); bs[next].focus(); }
      });
    },
  });
  return pop.closed;
}

export const popoverOpen = () => !!openOne;
export function closePopover() { if (openOne) openOne.close(undefined); }
