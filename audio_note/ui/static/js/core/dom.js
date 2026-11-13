// DOM helpers. This is the ONLY module that assigns markup to elements, and it accepts nothing but SafeHtml
// (see core/html.js); tests/test_v1_frontend.py greps the rest of the code base for stray markup sinks.
import { SafeHtml } from "./html.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function markup(safe) {
  if (safe instanceof SafeHtml) return safe.value;
  if (safe === null || safe === undefined || safe === "") return "";
  throw new TypeError("render() needs html`…` / raw(…), not a plain string");
}

const lastMarkup = new WeakMap();

/** Replace the children of `el`. */
export function render(el, safe) {
  if (!el) return el;
  const s = markup(safe);
  lastMarkup.set(el, s);
  el.innerHTML = s;
  return el;
}

/**
 * Like render(), but a no-op when the markup is what the previous render()/patch() of this element produced.
 * Regions that are refreshed from status / live events use it so hover, focus and text selection survive.
 * Returns true when the DOM was touched.
 */
export function patch(el, safe) {
  if (!el) return false;
  const s = markup(safe);
  if (lastMarkup.get(el) === s) return false;
  lastMarkup.set(el, s);
  el.innerHTML = s;
  return true;
}

/** Insert markup relative to `el`: "beforebegin" | "afterbegin" | "beforeend" | "afterend". */
export function insert(el, position, safe) {
  if (el) el.insertAdjacentHTML(position, markup(safe));
  return el;
}
export const append = (el, safe) => insert(el, "beforeend", safe);

/** Build detached nodes; returns the first element. */
export function make(safe) {
  const t = document.createElement("template");
  t.innerHTML = markup(safe);
  return t.content.firstElementChild;
}

/** Replace `el` itself with new markup; returns the first new element. */
export function replace(el, safe) {
  const t = document.createElement("template");
  t.innerHTML = markup(safe);
  const first = t.content.firstElementChild;
  el.replaceWith(t.content);
  return first;
}

/**
 * Delegated listener: handler(event, matchedElement) runs for events whose target is inside an element matching
 * `selector` under `root`. Pass an AbortSignal (ctx.signal of a page) to remove it; returns an "off" function too.
 */
export function on(root, type, selector, handler, { signal, capture = false } = {}) {
  if (!root) return () => {};
  const listener = (e) => {
    const start = e.target instanceof Element ? e.target : e.target && e.target.parentElement;
    const hit = selector ? (start ? start.closest(selector) : null) : root;
    if (hit && (hit === root || root.contains(hit))) handler(e, hit);
  };
  root.addEventListener(type, listener, { signal, capture });
  return () => root.removeEventListener(type, listener, { capture });
}

/** First row of `selector` that is at least partly below the sticky header, with its viewport position. */
function anchorOf(selector, topOffset) {
  for (const el of document.querySelectorAll(selector)) {
    const r = el.getBoundingClientRect();
    if (r.bottom > topOffset) return { id: el.id, top: r.top };
    if (r.top > window.innerHeight) break;
  }
  return null;
}

/**
 * Run `fn` (sync or async) that re-renders part of the page and keep the reader where they were: the first visible
 * element matching `selector` (elements need an id) stays at the same viewport position afterwards.
 */
export async function keepScroll(fn, { selector = ".utt[id]", topOffset = 120 } = {}) {
  const before = anchorOf(selector, topOffset);
  const y = window.scrollY;
  const out = await fn();
  if (before && before.id) {
    const el = document.getElementById(before.id);
    if (el) {
      const delta = el.getBoundingClientRect().top - before.top;
      if (Math.abs(delta) > 1) window.scrollBy(0, delta);
      return out;
    }
  }
  if (Math.abs(window.scrollY - y) > 1) window.scrollTo(0, y);
  return out;
}

/** Focusable descendants in tab order (dialog focus trap). */
export function focusables(root) {
  return $$('a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])', root)
    .filter((el) => el.offsetParent !== null || el === document.activeElement);
}

/** True when the user is typing somewhere (shortcuts must not fire). */
export function isTyping(target = document.activeElement) {
  if (!target || !(target instanceof Element)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag !== "INPUT") return false;
  const type = (target.getAttribute("type") || "text").toLowerCase();
  return !["checkbox", "radio", "button", "submit", "range", "file", "color"].includes(type);
}

export const prefersReducedMotion = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

/** Scroll an element into view (centre) without smooth motion when the user asked for less of it. */
export function reveal(el, { block = "center" } = {}) {
  if (el) el.scrollIntoView({ block, behavior: prefersReducedMotion() ? "auto" : "smooth" });
}
