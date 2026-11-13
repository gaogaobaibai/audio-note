// Keyboard shortcut registry.
//   bind("mod+k", openPalette, {desc: "搜索 / 命令", global: true})
//   bind("space", toggle, {desc: "播放 / 暂停", group: "播放"})
// Keys: single characters ("j", "/", "?"), "space", "escape", "enter", arrow names; modifiers "mod" (Ctrl or ⌘),
// "shift", "alt". Bindings do not fire while the user types (unless inInput) and, unless `global`, not while a
// dialog / palette is open. A handler that returns false lets the key through.
import { isTyping } from "./dom.js";

const bindings = [];

function normalize(combo) {
  const parts = String(combo).toLowerCase().split("+").map((s) => s.trim()).filter(Boolean);
  const key = parts.pop();
  return { key: key === "space" ? " " : key === "esc" ? "escape" : key, mod: parts.includes("mod") || parts.includes("ctrl"), shift: parts.includes("shift"), alt: parts.includes("alt") };
}

export function bind(combo, fn, { desc = "", group = "通用", inInput = false, global = false, overlay = false } = {}) {
  const b = { combo, ...normalize(combo), fn, desc, group, inInput, global, overlay };
  bindings.push(b);
  return () => { const i = bindings.indexOf(b); if (i >= 0) bindings.splice(i, 1); };
}

/** [{combo, desc, group}] for the cheat sheet. */
export const listShortcuts = () => bindings.filter((b) => b.desc).map(({ combo, desc, group }) => ({ combo, desc, group }));

/** "mod+k" -> "Ctrl K" / "⌘ K" */
export function comboLabel(combo) {
  const mac = /Mac|iPhone|iPad/.test(navigator.platform || "");
  return String(combo).split("+").map((p) => {
    const k = p.trim().toLowerCase();
    if (k === "mod") return mac ? "⌘" : "Ctrl";
    if (k === "shift") return "Shift";
    if (k === "alt") return mac ? "⌥" : "Alt";
    if (k === "space") return "空格";
    if (k === "escape" || k === "esc") return "Esc";
    if (k === "enter") return "Enter";
    return k.length === 1 ? k.toUpperCase() : k;
  }).join(" ");
}

const overlayOpen = () => !!document.querySelector(".modal-backdrop, .palette-backdrop, .drawer-backdrop");
/** Space / Enter on a focused control belongs to that control. */
function onControl(el) {
  if (!el || el === document.body) return false;
  return !!el.closest('button, a[href], summary, select, input, textarea, [role="button"], [role="tab"], [role="menuitem"], [role="option"], [tabindex]:not([tabindex="-1"])');
}

function onKeydown(e) {
  if (e.defaultPrevented || e.isComposing) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key.toLowerCase();
  const mod = e.ctrlKey || e.metaKey;
  const typing = isTyping(e.target);
  for (const b of bindings) {
    if (b.key !== key || b.mod !== mod || b.alt !== e.altKey) continue;
    // "?" needs shift on most layouts: only compare shift for letter keys and named keys
    if ((b.key.length > 1 || /[a-z ]/.test(b.key)) && b.shift !== e.shiftKey) continue;
    if (typing && !b.inInput) continue;
    if (!b.global && !b.overlay && overlayOpen()) continue;
    if ((b.key === " " || b.key === "enter") && !b.mod && onControl(e.target)) continue;
    if (b.fn(e) === false) continue;
    e.preventDefault();
    return;
  }
}

let started = false;
export function initShortcuts() {
  if (started) return;
  started = true;
  document.addEventListener("keydown", onKeydown);
}
