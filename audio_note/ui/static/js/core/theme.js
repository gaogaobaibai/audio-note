// Theme preference: "auto" (follow the OS) | "light" | "dark", kept in localStorage and applied as
// <html data-theme="light|dark"> (no attribute = auto; css/tokens.css then follows prefers-color-scheme).
// index.html applies the stored value before first paint; this module keeps it in sync afterwards.
import { prefs } from "./store.js";
import { emit } from "./bus.js";

export const THEMES = ["auto", "light", "dark"];
export const THEME_LABEL = { auto: "跟随系统", light: "浅色", dark: "深色" };

const media = typeof window !== "undefined" && window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

export function themePref() {
  const t = prefs.get("theme", "auto");
  return THEMES.includes(t) ? t : "auto";
}
export function resolvedTheme() {
  const t = themePref();
  return t === "auto" ? (media && media.matches ? "dark" : "light") : t;
}

function apply() {
  const t = themePref();
  const root = document.documentElement;
  if (t === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", resolvedTheme() === "dark" ? "#151513" : "#F6F5F1");
  emit("theme", { pref: t, resolved: resolvedTheme() });
}

export function setTheme(t) {
  prefs.set("theme", THEMES.includes(t) ? t : "auto");
  apply();
}
/** auto -> light -> dark -> auto */
export function cycleTheme() {
  const next = THEMES[(THEMES.indexOf(themePref()) + 1) % THEMES.length];
  setTheme(next);
  return next;
}

export function initTheme() {
  apply();
  if (media) {
    const onChange = () => { if (themePref() === "auto") apply(); };
    if (media.addEventListener) media.addEventListener("change", onChange); else if (media.addListener) media.addListener(onChange);
  }
}
