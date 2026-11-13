// Speaker presentation: stable colours, avatars, name chips, the relation picker, and the shared wording for
// "不是 TA" answers and sample-mismatch notices.
//
// Colour rule (SPEC 9.1): 12 ink / tint pairs assigned by speaker ID, so a rename never changes the colour and it
// is the same on every page; the owner ("本人") always gets the accent; unidentified voices are neutral grey.
// CSS classes: .c0 … .c11, .c-me, .c-unk set --ink / --tint (css/components.css).
import { html, raw, when } from "../core/html.js";
import { $$ } from "../core/dom.js";
import { hashStr, num } from "../core/util.js";
import { selfName, systemTurnShare, unknownLabel } from "../core/store.js";
import { icon } from "./icons.js";
import { toast } from "./toast.js";

export const PALETTE_SIZE = 12;
const nameToId = new Map();      // speakers seen so far: lets name-only payloads (search hits, digests) use the id colour

/** Feed the name -> id map from any payload that has both (speaker lists, transcript rows). */
export function rememberSpeakers(list, { idKey = "id", nameKey = "name" } = {}) {
  for (const s of list || []) {
    const id = s && s[idKey], name = s && s[nameKey];
    if (id != null && name) nameToId.set(String(name), Number(id));
  }
}

/** Pure colour rule: the class for a speaker id / name / self flag. */
export function colorClassFor({ id = null, name = "", isSelf = false, known = true } = {}) {
  if (isSelf) return "c-me";
  if (!known) return "c-unk";
  if (id != null && Number.isFinite(Number(id))) return `c${((Number(id) - 1) % PALETTE_SIZE + PALETTE_SIZE) % PALETTE_SIZE}`;
  if (name && nameToId.has(String(name))) return colorClassFor({ id: nameToId.get(String(name)) });
  return name ? `c${hashStr(name) % PALETTE_SIZE}` : "c-unk";
}

/**
 * Colour class of anything speaker-shaped:
 *   an utterance / search hit {speaker_id?, speaker}   ·   a speaker row {id, name, is_self}   ·   a bare name
 */
export function colorClass(s) {
  if (s == null) return "c-unk";
  if (typeof s === "string") {
    if (!s || s === unknownLabel()) return "c-unk";
    return colorClassFor({ name: s, isSelf: s === selfName() });
  }
  if ("speaker" in s || "speaker_id" in s) {
    const name = s.speaker || "";
    if ("speaker_id" in s && s.speaker_id == null) return "c-unk";
    if (!("speaker_id" in s) && (!name || name === unknownLabel())) return "c-unk";
    return colorClassFor({ id: s.speaker_id ?? null, name, isSelf: !!name && name === selfName() });
  }
  return colorClassFor({ id: s.id ?? null, name: s.name || "", isSelf: !!s.is_self || (!!s.name && s.name === selfName()) });
}

const initial = (name) => { const ch = Array.from(String(name || "").trim())[0]; return ch ? ch.toUpperCase() : "?"; };

/** Round avatar with the first character of the name (ink on tint). Decorative: the name is always next to it. */
export function avatar(s, size = 28, { system = false } = {}) {
  const c = colorClass(s);
  const name = typeof s === "string" ? s : (s && (s.speaker ?? s.name)) || "";
  const unk = c === "c-unk";
  return html`<span class="avatar ${c}" style="--sz:${Number(size) || 28}px" aria-hidden="true">${unk ? "?" : initial(name)}${when(system, () => html`<i class="avatar-sys">${icon("volume", 10)}</i>`)}</span>`;
}

export const isSystemVoice = (share) => { const v = num(share); return v != null && v >= systemTurnShare(); };
/** Small "came from system audio" mark (the far side of a meeting / call). */
export function sysBadge(share) {
  if (!isSystemVoice(share)) return "";
  const pct = Math.round(Number(share) * 100);
  return html`<span class="sys-mark" tabindex="0" data-tip="来自系统声音（会议 / 通话的对方）· 占 ${pct}%">${icon("volume", 13, { label: "系统声音" })}</span>`;
}

/** Coloured name with optional 我 badge / system mark / score. */
export function speakerName(s, { score = null, me = true, system = null } = {}) {
  const name = typeof s === "string" ? s : (s && (s.speaker ?? s.name)) || unknownLabel();
  const c = colorClass(s);
  return html`<span class="spk-name ${c}">${name}</span>${when(me && c === "c-me", () => html`<span class="badge me" title="本人">我</span>`)}${system != null ? sysBadge(system) : ""}${when(score != null && c !== "c-unk", () => html`<span class="spk-score" title="和这个人的声纹的相似度">${Number(score).toFixed(2)}</span>`)}`;
}

export const ORIGIN_LABEL = { mic: "麦克风", file: "文件", label: "标记", voice: "声音组", cluster: "簇" };

/** Text for the answer of POST /api/utterances/{id}/reject and POST /api/voices/reject. */
export function rejectSummary(r, name) {
  const n = Number(r && r.rejected) || 0, k = Number(r && r.negatives_added) || 0, m = Number(r && r.relabeled_changed) || 0;
  return `已标为不是 ${name || "TA"}：${n} 句${k ? `，加入 ${k} 个负样本` : "（没有取到负样本，只是这些句子不再认成 TA）"}${m ? `，这段录音里另有 ${m} 句重新识别` : ""}`;
}
export function toastReject(r, name) {
  const text = rejectSummary(r, name);
  if (r && r.warning) toast(`${text}。${r.warning}`, { kind: "warn", ms: 10000 });
  else toast(text, { kind: "ok", ms: 6000 });
}

/**
 * Notice for a 400 {mismatch: true, score, threshold, name}: "this voice does not sound like X's samples".
 * `forceAttr` is the data attribute name put on the 「仍然加入」 button, e.g. "data-force".
 */
export function mismatchNotice(data, forceAttr = "data-force") {
  const d = data || {};
  const why = d.score != null && d.threshold != null ? `相似度 ${Number(d.score).toFixed(2)} < ${Number(d.threshold).toFixed(2)}，` : "";
  const safeAttr = /^data-[a-z-]+$/.test(forceAttr) ? forceAttr : "data-force";
  return html`<div class="notice warn mismatch-note" role="alert"><span class="notice-ic">${icon("alert", 16)}</span>
    <div class="notice-body">这段声音和 ${d.name || "TA"} 已有的样本不像，没有加入。${why}确定是同一个人再加：</div>
    <div class="notice-acts"><button type="button" class="btn sm" ${raw(` ${safeAttr}`)}>仍然加入</button></div></div>`;
}

// ---- relation picker: <select> of the backend's RELATION_CHOICES plus "自定义…" that reveals a text box
export const CUSTOM_REL = "__custom__";
export function relationSelect(choices, value = "", { empty = "关系（可选）", label = "和我的关系" } = {}) {
  const v = String(value || "");
  const custom = !!v && !(choices || []).includes(v);
  return html`<span class="rel-pick"><select class="input" data-rel-sel aria-label="${label}">
      <option value="">${empty}</option>
      ${(choices || []).map((c) => html`<option value="${c}" ${c === v ? html`selected` : ""}>${c}</option>`)}
      <option value="${CUSTOM_REL}" ${custom ? html`selected` : ""}>自定义…</option>
    </select><input class="input" data-rel-custom placeholder="填写关系" aria-label="自定义关系" value="${custom ? v : ""}" ${custom ? "" : html`hidden`}></span>`;
}
export function wireRelationSelects(root) {
  for (const sel of $$("[data-rel-sel]", root)) {
    const custom = sel.parentElement.querySelector("[data-rel-custom]");
    sel.addEventListener("change", () => { const c = sel.value === CUSTOM_REL; custom.hidden = !c; if (c) custom.focus(); });
  }
}
export function relationValue(root) {
  const sel = root.querySelector("[data-rel-sel]");
  if (!sel) return "";
  if (sel.value === CUSTOM_REL) { const c = root.querySelector("[data-rel-custom]"); return c ? c.value.trim() : ""; }
  return sel.value;
}
