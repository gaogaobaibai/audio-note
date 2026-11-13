// Small presentational components. Every function returns SafeHtml (see core/html.js); busy() is the one DOM helper.
import { html, raw, when, attrs as attrsOf, join } from "../core/html.js";
import { icon } from "./icons.js";

/** button({label, icon, kind: "primary" | "ghost" | "danger" | "rec", size: "sm", attrs: {"data-act": "x"}, disabled, title}) */
export function button({ label = "", icon: ic = "", kind = "", size = "", attrs = {}, disabled = false, title = "", type = "button" } = {}) {
  return html`<button type="${type}" class="btn ${kind} ${size}" ${attrsOf({ ...attrs, disabled, title: title || null })}>${ic ? icon(ic, size === "sm" ? 14 : 16) : ""}${label ? html`<span>${label}</span>` : ""}</button>`;
}

/** Icon-only button: the label becomes aria-label and a tooltip. */
export function iconBtn({ icon: ic, label, attrs = {}, kind = "", size = 16, disabled = false, tip = true } = {}) {
  return html`<button type="button" class="icon-btn ${kind}" aria-label="${label}" ${tip ? html`title="${label}"` : ""} ${attrsOf({ ...attrs, disabled })}>${icon(ic, size)}</button>`;
}

/** kinds: "" | ok | warn | danger | info | plain | me */
export const badge = (text, kind = "", title = "") => html`<span class="badge ${kind}" ${title ? html`title="${title}"` : ""}>${text}</span>`;

/** Toggleable filter chip (a real button, aria-pressed). */
export function chip({ label, on = false, attrs = {}, count = null, lead = "" } = {}) {
  return html`<button type="button" class="chip ${on ? "on" : ""}" aria-pressed="${on ? "true" : "false"}" ${attrsOf(attrs)}>${lead}${label}${count !== null && count !== undefined ? html`<span class="chip-n">${count}</span>` : ""}</button>`;
}

/** Stat tile: label, big value, unit, optional hint line and chart (SafeHtml). */
export function tile({ label, value, unit = "", hint = "", chart = "", attrs = {} } = {}) {
  return html`<div class="tile" ${attrsOf(attrs)}>
    <div class="tile-k">${label}</div>
    <div class="tile-v">${value}${when(unit, () => html`<small>${unit}</small>`)}</div>
    ${when(hint, () => html`<div class="tile-hint">${hint}</div>`)}
    ${when(chart, () => html`<div class="tile-chart">${chart}</div>`)}
  </div>`;
}

/** Section heading: title, optional one-line description, optional actions (SafeHtml) on the right. */
export function sectionHead({ title, sub = "", actions = "", level = 2 } = {}) {
  const h = level === 3 ? html`<h3>${title}</h3>` : html`<h2>${title}</h2>`;
  return html`<div class="sec-head"><div class="sec-title">${h}${when(sub, () => html`<div class="sec-sub">${sub}</div>`)}</div>${when(actions, () => html`<div class="sec-acts">${actions}</div>`)}</div>`;
}

/** Real tabs: items [{id, label, count?}]; buttons carry data-tab. */
export function tabs(items, activeId, { label = "" } = {}) {
  return html`<div class="tabs" role="tablist" ${label ? html`aria-label="${label}"` : ""}>${items.map((it) => html`<button type="button" role="tab" class="tab" aria-selected="${it.id === activeId ? "true" : "false"}" tabindex="${it.id === activeId ? "0" : "-1"}" data-tab="${it.id}">${it.label}${it.count ? html`<span class="tab-n">${it.count}</span>` : ""}</button>`)}</div>`;
}

/** Segmented control (pick one of a few): items [{id, label, icon?, title?}]; buttons carry data-seg. */
export function segmented(items, activeId, { label = "" } = {}) {
  return html`<div class="seg" role="group" ${label ? html`aria-label="${label}"` : ""}>${items.map((it) => html`<button type="button" class="seg-item" aria-pressed="${it.id === activeId ? "true" : "false"}" data-seg="${it.id}" ${it.title ? html`title="${it.title}"` : ""}>${it.icon ? icon(it.icon, 15) : ""}${it.label ? html`<span>${it.label}</span>` : ""}</button>`)}</div>`;
}

export function progress(pct, kind = "", label = "") {
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  return html`<div class="progress ${kind}" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${p.toFixed(0)}" ${label ? html`aria-label="${label}"` : ""}><i style="width:${p.toFixed(1)}%"></i></div>`;
}

/** Key / value list: pairs [[key, value], …]; values may be SafeHtml. */
export const kv = (pairs) => html`<dl class="kv">${pairs.filter(Boolean).map(([k, v]) => html`<dt>${k}</dt><dd>${v}</dd>`)}</dl>`;

/** One sentence + one action. `action` is SafeHtml (a button or a link styled as one). */
export function emptyState({ icon: ic = "", title = "", text = "", action = "", compact = false } = {}) {
  return html`<div class="empty ${compact ? "compact" : ""}">
    ${when(ic, () => html`<div class="empty-ic">${icon(ic, compact ? 22 : 28)}</div>`)}
    ${when(title, () => html`<div class="empty-title">${title}</div>`)}
    ${when(text, () => html`<div class="empty-text">${text}</div>`)}
    ${when(action, () => html`<div class="empty-acts">${action}</div>`)}
  </div>`;
}

/** Placeholder while loading: "page" | "rows" | "cards" | "text". */
export function skeleton(kind = "rows", n = 6) {
  const line = (w) => html`<div class="sk" style="width:${w}%"></div>`;
  if (kind === "page") return html`<div class="sk-page" aria-busy="true" aria-label="加载中"><div class="sk sk-title"></div><div class="sk-cards">${[0, 1, 2].map(() => html`<div class="sk sk-card"></div>`)}</div>${skeleton("rows", 6)}</div>`;
  if (kind === "cards") return html`<div class="sk-cards" aria-busy="true" aria-label="加载中">${Array.from({ length: n }, () => html`<div class="sk sk-card"></div>`)}</div>`;
  if (kind === "text") return html`<div class="sk-rows" aria-busy="true" aria-label="加载中">${Array.from({ length: n }, (_, i) => line([96, 88, 92, 70, 84, 60][i % 6]))}</div>`;
  return html`<div class="sk-rows" aria-busy="true" aria-label="加载中">${Array.from({ length: n }, (_, i) => html`<div class="sk-row"><div class="sk sk-dot"></div><div class="grow">${line([38, 30, 44][i % 3])}${line([92, 76, 84, 64][i % 4])}</div></div>`)}</div>`;
}

/** Form field: label + control (SafeHtml) + help / error text. */
export function field({ label = "", help = "", control = "", error = "", id = "", attrs = {} } = {}) {
  return html`<div class="field" ${attrsOf(attrs)}>
    ${when(label, () => (id ? html`<label for="${id}">${label}</label>` : html`<span class="label">${label}</span>`))}
    ${control}
    ${when(help, () => html`<div class="help">${help}</div>`)}
    ${when(error, () => html`<div class="help err">${error}</div>`)}
  </div>`;
}

const NOTICE_ICON = { info: "info", warn: "alert", danger: "alert", ok: "check" };
/** Status line with an accent bar instead of a coloured box. body / actions are SafeHtml or plain text. */
export function notice({ kind = "info", body = "", actions = "", attrs = {}, compact = false } = {}) {
  return html`<div class="notice ${kind} ${compact ? "compact" : ""}" role="${kind === "danger" ? "alert" : "note"}" ${attrsOf(attrs)}>
    <span class="notice-ic">${icon(NOTICE_ICON[kind] || "info", 16)}</span>
    <div class="notice-body">${body}</div>
    ${when(actions, () => html`<div class="notice-acts">${actions}</div>`)}
  </div>`;
}

export const spinner = (label = "") => html`<span class="thinking"><span class="spin" aria-hidden="true"></span>${label}</span>`;

/** Inline "A · B · C" from the truthy parts. */
export const dots = (...parts) => join(parts.filter((p) => p !== null && p !== undefined && p !== false && p !== ""), raw('<span class="sep" aria-hidden="true"> · </span>'));

/**
 * Put a button into its busy state (disabled + spinner + label). Returns restore(), which brings back the original
 * content and enabled state. Replaces the hand-written copies of this dance.
 */
export function busy(btn, label = "") {
  if (!btn) return () => {};
  const saved = Array.from(btn.childNodes);
  const wasDisabled = btn.disabled;
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  const spin = document.createElement("span");
  spin.className = "spin";
  spin.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = label || btn.textContent.trim();
  btn.replaceChildren(spin, text);
  let restored = false;
  return () => {
    if (restored) return;
    restored = true;
    btn.replaceChildren(...saved);
    btn.disabled = wasDisabled;
    btn.removeAttribute("aria-busy");
  };
}
