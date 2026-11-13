// html`` — the only way markup is built in this app. Every interpolated value is HTML-escaped unless it is a
// SafeHtml (the result of another html`` / raw()) or an array of such values. That turns "forgot to escape" from a
// silent XSS hole into visible escaped text. core/dom.js render() accepts nothing but SafeHtml.
// No DOM access: node can import this file directly.
import { esc } from "./util.js";

export class SafeHtml {
  constructor(value) { this.value = String(value); Object.freeze(this); }
  toString() { return this.value; }
}

const EMPTY = new SafeHtml("");

function piece(v) {
  if (v === null || v === undefined || v === false || v === true) return "";
  if (v instanceof SafeHtml) return v.value;
  if (Array.isArray(v)) { let s = ""; for (const x of v) s += piece(x); return s; }
  return esc(v);
}

/** Tagged template: html`<b title="${user}">${user}</b>` -> SafeHtml with both values escaped. */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += piece(values[i]) + strings[i + 1];
  return new SafeHtml(out);
}

/** Explicit opt-out for markup this app produced itself (md() output, SVG from ui/icons.js / ui/charts.js). */
export function raw(trusted) {
  if (trusted instanceof SafeHtml) return trusted;
  return trusted === null || trusted === undefined || trusted === "" ? EMPTY : new SafeHtml(trusted);
}

/** Join SafeHtml / plain values (escaped) with a separator (escaped unless SafeHtml). */
export function join(list, sep = "") {
  const s = piece(sep);
  return new SafeHtml((list || []).filter((x) => x !== null && x !== undefined && x !== false && x !== true).map(piece).join(s));
}

/** `cond ? value : ""` without the ternary noise: when(u.far, html`<span>远</span>`). */
export const when = (cond, value) => (cond ? (typeof value === "function" ? value() : value) : EMPTY);

/** {disabled: true, title: 'a"b', hidden: false, "data-id": 3} -> ` disabled title="a&quot;b" data-id="3"` (SafeHtml). */
export function attrs(obj) {
  let s = "";
  for (const [k, v] of Object.entries(obj || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (!/^[a-zA-Z_:][-a-zA-Z0-9_:.]*$/.test(k)) continue;
    s += v === true ? ` ${k}` : ` ${k}="${esc(v)}"`;
  }
  return new SafeHtml(s);
}

/** Class list from strings / {name: condition} objects: cls("utt", {far, hl}) -> "utt far" (plain string, escape as usual). */
export function cls(...parts) {
  const out = [];
  for (const p of parts) {
    if (!p) continue;
    if (typeof p === "string") out.push(p);
    else for (const [k, v] of Object.entries(p)) if (v) out.push(k);
  }
  return out.join(" ");
}

export const isSafe = (v) => v instanceof SafeHtml;
