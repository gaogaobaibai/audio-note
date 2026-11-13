// Toasts: short status messages in a polite live region, optionally with one action button ("撤销").
//   toast("已保存", {kind: "ok"})
//   toast("已开启系统声音", {kind: "ok", action: {label: "撤销", run: async () => …}})
// The message is set as text (never parsed as markup). The timer pauses while the pointer or focus is on the toast.
import { html } from "../core/html.js";
import { $, make } from "../core/dom.js";
import { icon } from "./icons.js";

const MAX_VISIBLE = 4;

/**
 * kinds: "" (neutral) | "ok" | "err" | "warn". `ms` = lifetime; `sticky` = stays until closed.
 * Also accepts the old positional form toast(msg, "ok", 5000). Returns {close()}.
 */
export function toast(msg, opts = {}, legacyMs) {
  if (typeof opts === "string") opts = { kind: opts, ms: legacyMs };
  const { kind = "", action = null, sticky = false } = opts;
  const ms = opts.ms ?? (action ? 7000 : 3500);
  const root = $("#toasts");
  if (!root) return { close() {} };
  const el = make(html`<div class="toast ${kind}" role="${kind === "err" ? "alert" : "status"}">
    <span class="toast-msg"></span>
    ${action ? html`<button type="button" class="toast-act">${action.label}</button>` : ""}
    <button type="button" class="toast-x" aria-label="关闭提示">${icon("x", 14)}</button>
  </div>`);
  el.querySelector(".toast-msg").textContent = String(msg ?? "");
  let timer = null, closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    if (timer) clearTimeout(timer);
    el.remove();
  };
  const arm = () => { if (!sticky && !closed) { if (timer) clearTimeout(timer); timer = setTimeout(close, ms); } };
  const hold = () => { if (timer) { clearTimeout(timer); timer = null; } };
  el.addEventListener("mouseenter", hold);
  el.addEventListener("mouseleave", arm);
  el.addEventListener("focusin", hold);
  el.addEventListener("focusout", arm);
  el.querySelector(".toast-x").addEventListener("click", close);
  if (action) {
    el.querySelector(".toast-act").addEventListener("click", async (e) => {
      e.currentTarget.disabled = true;
      close();
      try { await action.run(); } catch (err) { toastErr(err); }
    });
  }
  root.appendChild(el);
  while (root.children.length > MAX_VISIBLE) root.firstElementChild.remove();
  arm();
  return { close };
}

export const toastErr = (e) => {
  if (e && (e.name === "AbortError")) return;
  toast(e && e.message ? e.message : String(e), { kind: "err", ms: 6000 });
};
