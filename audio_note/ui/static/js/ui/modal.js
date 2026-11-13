// Dialogs and the right-hand drawer, on one overlay STACK.
//   const value = await openModal({title, body: html`…`, actions: [{label: "取消"}, {label: "确定", kind: "primary", value: true}]})
//   if (await confirm({title, text, okLabel, danger})) …
//   const name = await prompt({title, label, value})          // string, or null when dismissed
//   const d = openDrawer({title, body, onMount}); d.close()
// Guarantees: role="dialog" + aria-modal + a labelled title; focus moves in, is trapped, and returns to where it
// was; Esc and a click on the backdrop close the TOP overlay only; the promise ALWAYS settles (undefined when the
// dialog was dismissed), so no caller can hang on an abandoned dialog.
import { html, raw, when } from "../core/html.js";
import { $, make, focusables } from "../core/dom.js";
import { uid } from "../core/util.js";
import { icon } from "./icons.js";
import { toastErr } from "./toast.js";

const stack = [];       // open overlays, bottom -> top
let keysBound = false;

function onKeydown(e) {
  const top = stack[stack.length - 1];
  if (!top) return;
  const pop = $("#pop-root");
  if (pop && pop.childElementCount) return;          // an open menu / popover handles its own keys first
  if (document.querySelector(".palette-backdrop")) return;   // so does the command palette
  if (e.key === "Escape") {
    e.preventDefault();
    e.stopPropagation();
    if (top.dismissible) top.close(undefined);
    return;
  }
  if (e.key !== "Tab") return;
  const items = focusables(top.panel);
  if (!items.length) { e.preventDefault(); top.panel.focus(); return; }
  const first = items[0], last = items[items.length - 1];
  const active = document.activeElement;
  if (!top.panel.contains(active)) { e.preventDefault(); first.focus(); }
  else if (e.shiftKey && active === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && active === last) { e.preventDefault(); first.focus(); }
}

function syncLock() {
  const root = document.documentElement;
  const lock = stack.length > 0;
  if (lock && !root.classList.contains("has-overlay")) {
    // hiding the page scrollbar must not shift the page: keep its width as padding while the overlay is open
    root.style.setProperty("--sbw", `${Math.max(0, window.innerWidth - root.clientWidth)}px`);
  }
  root.classList.toggle("has-overlay", lock);
  if (!lock) root.style.removeProperty("--sbw");
  const app = $("#app");
  // everything behind the top overlay is inert for assistive tech
  if (app) { if (stack.length) app.setAttribute("aria-hidden", "true"); else app.removeAttribute("aria-hidden"); }
  stack.forEach((o, i) => { if (i < stack.length - 1) o.backdrop.setAttribute("aria-hidden", "true"); else o.backdrop.removeAttribute("aria-hidden"); });
}

function openOverlay({ kind = "modal", title = "", body = "", size = "md", actions = [], dismissible = true, onMount, onClose, label = "" }) {
  if (!keysBound) { document.addEventListener("keydown", onKeydown, true); keysBound = true; }
  const root = $("#modal-root");
  const titleId = uid("dlg");
  const foot = actions.length ? html`<footer class="modal-foot">${actions.map((a, i) => html`<button type="button" class="btn ${a.kind || ""} ${a.align === "left" ? "left" : ""}" data-action="${i}" ${a.disabled ? raw("disabled") : ""} ${a.autofocus ? raw("data-autofocus") : ""}>${a.icon ? icon(a.icon, 16) : ""}${a.label}</button>`)}</footer>` : "";
  const backdrop = make(html`<div class="${kind === "drawer" ? "drawer-backdrop" : "modal-backdrop"}">
    <div class="${kind === "drawer" ? "drawer" : `modal ${size}`}" role="dialog" aria-modal="true" ${title ? raw(`aria-labelledby="${titleId}"`) : html`aria-label="${label || "对话框"}"`} tabindex="-1">
      <header class="modal-head">
        ${when(title, () => html`<h2 id="${titleId}">${title}</h2>`)}
        ${when(dismissible, () => html`<button type="button" class="icon-btn" data-overlay-close aria-label="关闭">${icon("x", 18)}</button>`)}
      </header>
      <div class="modal-body">${body}</div>
      ${foot}
    </div>
  </div>`);
  const panel = backdrop.firstElementChild;
  const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  let settle, closed = false;
  const promise = new Promise((r) => { settle = r; });
  const entry = { backdrop, panel, dismissible, close };
  const handle = {
    el: panel,
    body: panel.querySelector(".modal-body"),
    foot: panel.querySelector(".modal-foot"),
    close,
    get closed() { return closed; },
    /** Button of actions[i]. */
    actionButton: (i) => panel.querySelector(`[data-action="${i}"]`),
    setTitle: (t) => { const h = panel.querySelector(".modal-head h2"); if (h) h.textContent = t; },
  };

  function close(value) {
    if (closed) return;
    closed = true;
    const i = stack.indexOf(entry);
    if (i >= 0) stack.splice(i, 1);
    backdrop.remove();
    syncLock();
    if (onClose) { try { onClose(value); } catch (e) { console.error("[modal] onClose", e); } }
    if (returnFocus && document.contains(returnFocus)) { try { returnFocus.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
    settle(value);
  }

  // close on a real click on the backdrop (press and release both outside the panel: a text selection that ends
  // outside must not close the dialog)
  let downOnBackdrop = false;
  backdrop.addEventListener("mousedown", (e) => { downOnBackdrop = e.target === backdrop; });
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop && downOnBackdrop && dismissible) { close(undefined); return; }
    const x = e.target.closest("[data-overlay-close]");
    if (x && panel.contains(x)) { close(undefined); return; }
    const b = e.target.closest("[data-action]");
    if (b && b.parentElement === handle.foot) {
      const a = actions[Number(b.dataset.action)];
      if (!a || b.disabled) return;
      if (typeof a.run === "function") Promise.resolve().then(() => a.run(handle, b)).catch(toastErr);
      else close(a.value);
    }
  });

  root.appendChild(backdrop);
  stack.push(entry);
  syncLock();
  if (onMount) { try { onMount(handle); } catch (e) { console.error("[modal] onMount", e); } }
  if (!closed) {
    const target = panel.querySelector("[data-autofocus], [autofocus]") || focusables(handle.body)[0] || panel.querySelector("[data-action]") || panel;
    try { target.focus({ preventScroll: true }); if (target.select && target.matches("input[type=text], input:not([type])")) target.select(); } catch (e) { /* ignore */ }
  }
  promise.close = close;
  promise.handle = handle;
  return promise;
}

/**
 * actions: [{label, kind?: "primary" | "danger" | "ghost", value?, icon?, autofocus?, disabled?, align?: "left",
 *            run?: async (handle, button) => void}]
 * An action without `run` closes the dialog and resolves the promise with its `value`; an action with `run` stays
 * open until run() calls handle.close(value). onMount(handle): handle = {el, body, foot, close(value), actionButton(i)}.
 * The returned promise also carries .close(value) and .handle.
 */
export function openModal(opts) { return openOverlay({ ...opts, kind: "modal" }); }

/** Resolves true only when the confirm button was pressed (Esc / backdrop / 取消 -> false). */
export function confirm({ title, text = "", okLabel = "确定", cancelLabel = "取消", danger = false } = {}) {
  return openModal({
    title, size: "sm",
    body: html`<p class="t2">${text}</p>`,
    actions: [{ label: cancelLabel, value: false }, { label: okLabel, kind: danger ? "danger solid" : "primary", value: true, autofocus: true }],
  }).then((v) => v === true);
}

/** One text input. Resolves with the trimmed string, or null when dismissed. */
export function prompt({ title, label = "", value = "", placeholder = "", help = "", list = [], okLabel = "确定" } = {}) {
  const inputId = uid("prompt"), listId = uid("dl");
  return openModal({
    title, size: "sm",
    body: html`<div class="field"><label for="${inputId}">${label}</label>
      <input class="input" id="${inputId}" value="${value}" placeholder="${placeholder}" ${list.length ? html`list="${listId}"` : ""} autocomplete="off" data-autofocus>
      ${list.length ? html`<datalist id="${listId}">${list.map((x) => html`<option value="${x}"></option>`)}</datalist>` : ""}
      ${when(help, () => html`<div class="help">${help}</div>`)}</div>`,
    actions: [{ label: "取消", value: null }, { label: okLabel, kind: "primary", run: (h) => h.close(h.body.querySelector("input").value.trim()) }],
    onMount: (h) => {
      const inp = h.body.querySelector("input");
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); h.close(inp.value.trim()); } });
    },
  }).then((v) => (typeof v === "string" ? v : null));
}

/**
 * Right-hand panel for "details of the thing I clicked" (the list underneath stays where it was).
 * Returns {el, body, close(), closed: Promise<void>}.
 */
export function openDrawer({ title = "", body = "", onMount, onClose, actions = [] } = {}) {
  const p = openOverlay({ kind: "drawer", title, body, actions, onMount, onClose });
  return { el: p.handle.el, body: p.handle.body, close: () => p.close(undefined), closed: p.then(() => undefined) };
}

export const overlayOpen = () => stack.length > 0;
/** Close every open overlay (profile switch). */
export function closeAll() { for (const o of [...stack].reverse()) o.close(undefined); }
