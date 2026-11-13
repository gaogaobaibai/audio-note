// Sentences: how one utterance is drawn (inside a speaker "turn block" on the timeline, or as a compact row in
// lists and dialogs) and the dialogs that act on it: who said it, 不是 TA, split, context.
// Every mutation emits "utterances:changed" {day, ids, reason} on the bus; pages patch themselves from that —
// nothing here re-renders a page.
//
// Utterance payload (GET /api/transcript/{day}, /api/context/{id}): {id, segment_id, day, time "HH:MM:SS",
//   start_s, end_s, speaker, speaker_id | null, score | null, text, text_raw?, cluster, loudness_db | null,
//   system_share | null, audio?: bool}; later packages add marks / text_source / speaker_guess (ignored when absent).
import { html, when, cls } from "../core/html.js";
import { $$, render, insert } from "../core/dom.js";
import { api, isMismatch } from "../core/api.js";
import { emit } from "../core/bus.js";
import { diffChars, num } from "../core/util.js";
import { fmtDb } from "../core/format.js";
import { loadSpeakers, minDbfs } from "../core/store.js";
import { refreshStatus } from "../core/status.js";
import { icon } from "./icons.js";
import { badge, busy, emptyState, iconBtn, notice, skeleton, spinner } from "./components.js";
import { openModal } from "./modal.js";
import { toast, toastErr } from "./toast.js";
import { clips, mountInline, player } from "./player.js";
import { avatar, colorClass, isSystemVoice, mismatchNotice, relationSelect, relationValue, sysBadge, toastReject, wireRelationSelects, rememberSpeakers } from "./speaker.js";

const hhmm = (u) => String(u.time || "").slice(0, 5);
export const isFar = (u) => { const db = num(u.loudness_db), thr = num(minDbfs()); return db != null && thr != null && db < thr; };
const loudnessTitle = (u) => {
  const db = num(u.loudness_db), thr = num(minDbfs());
  if (db == null) return "";
  return `音量 ${fmtDb(db)}${thr != null ? `（门槛 ${fmtDb(thr)}${db < thr ? "，低于门槛：远处 / 不清晰的声音" : ""}）` : ""}`;
};

const SOURCE_LABEL = { lexicon: "词库", llm: "模型校对", refine: "会话校对", answer: "你的回答", user: "手动修改", guard: "自动清理" };

/**
 * The sentence text with its corrections made visible.
 *   - u.marks (later packages): [{start, end, kind: "fixed" | "uncertain", from?, source?}] over u.text
 *   - otherwise text_raw vs text: the characters that differ get a hairline underline; hovering / focusing one
 *     shows what the recogniser originally wrote.
 */
export function textWithMarks(u) {
  const text = String(u.text ?? "");
  if (Array.isArray(u.marks) && u.marks.length) {
    const chars = Array.from(text);
    const marks = u.marks.filter((m) => Number.isFinite(m.start) && Number.isFinite(m.end) && m.end > m.start).sort((a, b) => a.start - b.start);
    const out = [];
    let pos = 0;
    for (const m of marks) {
      if (m.start < pos || m.start >= chars.length) continue;
      out.push(html`${chars.slice(pos, m.start).join("")}`);
      const piece = chars.slice(m.start, Math.min(m.end, chars.length)).join("");
      if (m.kind === "uncertain") out.push(html`<span class="mark-ask" tabindex="0" data-mark="uncertain" data-start="${m.start}" data-end="${m.end}" ${m.inbox_id != null ? html`data-inbox="${m.inbox_id}"` : ""} data-tip="这个词不太确定">${piece}</span>`);
      else out.push(html`<span class="mark-fix" tabindex="0" data-mark="fixed" data-tip="${`原文：${m.from ?? ""}${m.source && SOURCE_LABEL[m.source] ? `（${SOURCE_LABEL[m.source]}）` : ""}`}">${piece}</span>`);
      pos = Math.min(m.end, chars.length);
    }
    out.push(html`${chars.slice(pos).join("")}`);
    return html`${out}`;
  }
  const before = u.text_raw;
  if (!before || before === text) return html`${text}`;
  const ops = diffChars(before, text);
  if (!ops) return html`<span class="mark-fix" tabindex="0" data-tip="${`原文：${before}`}">${text}</span>`;
  return html`${ops.map((op) => {
    if (op.op === "eq") return html`${op.b}`;
    if (op.op === "del") return html`<span class="mark-fix gone" tabindex="0" data-tip="${`原文这里还有：${op.a}`}" aria-label="${`删去了：${op.a}`}"></span>`;
    return html`<span class="mark-fix" tabindex="0" data-tip="${op.a ? `原文：${op.a}` : "原文里没有这几个字"}">${op.b}</span>`;
  })}`;
}

/** Icon buttons of one sentence. `acts` limits which ones are offered. */
export function actionBar(u, { acts } = {}) {
  if (!Array.isArray(acts)) acts = ["play", "label", "reject", "split", "ctx"];
  const known = u.speaker_id != null;
  const meta = [String(u.time || ""), known && u.score != null ? `相似度 ${Number(u.score).toFixed(2)}` : "", fmtDb(u.loudness_db)].filter(Boolean).join(" · ");
  const b = [];
  if (acts.includes("play") && u.audio) b.push(iconBtn({ icon: "play", label: "播放这句", attrs: { "data-u": "play" }, size: 15 }));
  if (acts.includes("label")) b.push(iconBtn({ icon: "user-check", label: known ? "改成别人说的" : "标记这句话是谁说的", attrs: { "data-u": "label" }, size: 15 }));
  if (acts.includes("reject") && known) b.push(iconBtn({ icon: "user-x", label: `不是 ${u.speaker}`, attrs: { "data-u": "reject" }, size: 15, kind: "warn" }));
  if (acts.includes("split") && u.audio) b.push(iconBtn({ icon: "scissors", label: "拆分这句（两个人的话混在一起时）", attrs: { "data-u": "split" }, size: 15 }));
  if (acts.includes("ctx")) b.push(iconBtn({ icon: "context", label: "看前后文", attrs: { "data-u": "ctx" }, size: 15 }));
  return html`<span class="utt-acts" role="toolbar" aria-label="这句话的操作"><span class="utt-when">${meta}</span>${b}</span>`;
}

/**
 * One sentence line inside a turn block. Keeps id="u{id}" (deep links) and data-utt / data-id.
 * `acts: false` leaves the action bar out: the timeline adds it when the line is hovered / focused / selected
 * (a day has hundreds of lines; five buttons each would multiply the markup for nothing).
 */
export function uttLine(u, { hl = false, acts, first = false, brk = false } = {}) {
  const far = isFar(u);
  return html`<div class="${cls("utt", { far, hl, brk })}" id="u${u.id}" data-utt="${u.id}" data-id="${u.id}" tabindex="${first ? "0" : "-1"}">
    <span class="utt-text">${textWithMarks(u)}</span>${when(far, () => html`<span class="badge far" tabindex="0" data-tip="${loudnessTitle(u)}">远</span>`)}
    ${acts === false ? "" : actionBar(u, { acts })}
  </div>`;
}

/** Which voice a sentence belongs to, as far as the payload can tell. */
const voiceKey = (u) => (u.speaker_id != null ? `s${u.speaker_id}` : u.voice_id != null ? `v${u.voice_id}` : `u${u.segment_id}:${u.cluster ?? "?"}`);

/**
 * Split a session's sentences into turn blocks: consecutive sentences of the same person less than `gapS` apart.
 * Unidentified sentences share one block ("未知") until a known speaker talks; a change of the anonymous voice
 * inside it is only marked with a little extra space (see uttLine `brk`). With persistent anonymous voices
 * (voice_id, a later package) each of those gets its own block.
 */
export function groupTurns(utts, { gapS = 60 } = {}) {
  const keyOf = (u) => (u.speaker_id != null ? `s${u.speaker_id}` : u.voice_id != null ? `v${u.voice_id}` : "unk");
  const startMs = (u) => { const t = Date.parse(String(u.abs_start || "").replace(" ", "T")); return Number.isFinite(t) ? t : NaN; };
  const turns = [];
  let cur = null;
  for (const u of utts) {
    const key = keyOf(u), t = startMs(u), sys = isSystemVoice(u.system_share);
    const lastEnd = cur ? cur.end : NaN;
    if (cur && cur.key === key && cur.sys === sys && !(Number.isFinite(t) && Number.isFinite(lastEnd) && t - lastEnd > gapS * 1000)) cur.utts.push(u);
    else { cur = { key, sys, utts: [u] }; turns.push(cur); }
    cur.end = Number.isFinite(t) ? t + Math.max(0, (Number(u.end_s) - Number(u.start_s)) || 0) * 1000 : NaN;
  }
  return turns.map((t) => t.utts);
}

/** Avatar + name + time once, then the sentences of the turn as lines. */
export function turnBlock(utts, { hlIds = null, firstFocusable = false, acts } = {}) {
  const u0 = utts[0];
  const c = colorClass(u0);
  const known = u0.speaker_id != null;
  const sys = isSystemVoice(u0.system_share);
  return html`<div class="turn ${c}" data-turn="${u0.id}">
    <div class="turn-side">${avatar(u0, 32, { system: sys })}</div>
    <div class="turn-main">
      <div class="turn-head"><span class="turn-name">${u0.speaker}</span>${when(c === "c-me", () => badge("我", "me", "本人"))}${when(sys, () => sysBadge(u0.system_share))}${when(!known, () => html`<button type="button" class="link-btn turn-who" data-u="label" data-for="${u0.id}">这是谁？</button>`)}<span class="turn-time">${hhmm(u0)}</span></div>
      ${utts.map((u, i) => uttLine(u, { hl: !!(hlIds && hlIds.has(u.id)), first: firstFocusable && i === 0, acts, brk: i > 0 && voiceKey(u) !== voiceKey(utts[i - 1]) }))}
    </div>
  </div>`;
}

/** Compact single-line row (recent sentences, context dialog, search): time · speaker · text. */
export function uttRow(u, { hl = false, link = false } = {}) {
  const c = colorClass(u);
  const far = isFar(u);
  const body = html`<span class="row-time">${hhmm(u)}</span><span class="spk-name ${c}">${u.speaker}</span>${when(c === "c-me", () => badge("我", "me", "本人"))}${sysBadge(u.system_share)}<span class="row-text">${textWithMarks(u)}</span>`;
  const klass = cls("utt-row", { far, hl });
  return link
    ? html`<a class="${klass}" href="#/transcript/${u.day}?u=${u.id}" data-utt="${u.id}">${body}</a>`
    : html`<div class="${klass}" data-utt="${u.id}">${body}</div>`;
}

const quote = (u) => html`<blockquote class="utt-quote ${colorClass(u)}"><span class="row-time">${hhmm(u)}</span><span class="spk-name ${colorClass(u)}">${u.speaker}</span>${sysBadge(u.system_share)}<span>${u.text}</span></blockquote>`;
const changed = (u, reason, ids) => emit("utterances:changed", { day: u.day, ids: ids || [u.id], reason });

/** Dialog: the sentences around one sentence. */
export async function showContext(id) {
  const m = openModal({ title: "前后文", size: "lg", body: skeleton("rows", 6), actions: [{ label: "关闭", kind: "primary" }] });
  try {
    const r = await api(`/api/context/${id}`, { query: { before: 8, after: 8 } });
    rememberSpeakers(r.utterances, { idKey: "speaker_id", nameKey: "speaker" });
    if (m.handle.closed) return;
    render(m.handle.body, r.utterances.length ? html`<div class="utt-rows">${r.utterances.map((u) => uttRow(u, { hl: u.id === id }))}</div>` : emptyState({ icon: "doc", title: "没有前后文", compact: true }));
    const hl = m.handle.body.querySelector(".hl");
    if (hl) hl.scrollIntoView({ block: "center" });
  } catch (e) {
    if (!m.handle.closed) render(m.handle.body, notice({ kind: "danger", body: e.message || String(e) }));
  }
  return m;
}

/**
 * Dialog: "这句话是谁说的？" -> POST /api/speakers/label. A name that is not registered yet is enrolled from this
 * sentence's voice (with 本人 / relation); the same voice in the same recording can be labelled along.
 */
export async function labelUtt(u) {
  const sp = await loadSpeakers().catch(() => ({ list: [], names: [], self: null, choices: [] }));
  const names = sp.names, me = sp.self, choices = sp.choices;
  const dur = Math.max(0, Number(u.end_s) - Number(u.start_s));
  let inline = null;
  const send = async (h, btn, force) => {
    const body = h.body;
    const inp = body.querySelector("#lb-name");
    const name = inp.value.trim();
    if (!name) { inp.focus(); return; }
    const isNew = !names.includes(name);
    const payload = { utterance_id: u.id, name, whole_cluster: body.querySelector("#lb-whole").checked };
    if (force) payload.force = true;
    if (isNew) {
      payload.is_self = body.querySelector("#lb-self").checked;
      const rel = relationValue(body.querySelector("#lb-new"));
      if (rel) payload.relation = rel;
    }
    const ok = h.actionButton(h.okIndex);
    const restore = busy(ok, "处理中…");
    for (const n of $$(".mismatch-note", body)) n.remove();
    try {
      const r = await api("/api/speakers/label", { method: "POST", json: payload });
      const note = r.sample_note ? `；${r.sample_note}` : "";
      toast(`已标记为 ${name}：${r.assigned != null ? `${r.assigned} 句` : "已更新"}${r.relabeled_changed ? `，这段录音里另有 ${r.relabeled_changed} 句自动更新` : ""}${note}`, { kind: note ? "warn" : "ok", ms: note ? 8000 : 6000 });
      emit("speakers:changed", { reason: "label" });
      if (payload.is_self) await refreshStatus();          // the 我 badge reads status.self_name
      h.close(true);
      changed(u, "label");
    } catch (e) {
      restore();
      if (isMismatch(e)) {
        toast(e.message, { kind: "err", ms: 8000 });
        insert(body, "beforeend", mismatchNotice(e.data, "data-force"));
        body.querySelector("[data-force]").addEventListener("click", () => send(h, btn, true));
      } else toastErr(e);
    }
  };
  const actions = [];
  if (u.speaker_id != null) actions.push({ label: `不是 ${u.speaker}`, kind: "danger", align: "left", run: (h) => { h.close(false); rejectUtt(u); } });
  actions.push({ label: "取消" });
  actions.push({ label: "标记", kind: "primary", disabled: !u.audio, run: (h, btn) => send(h, btn, false) });
  const m = openModal({
    title: "这句话是谁说的？", size: "md", actions,
    body: html`${quote(u)}
      ${u.audio ? html`<div class="pl-slot" id="lb-audio"></div>` : notice({ kind: "warn", body: "这段音频已被删除，无法提取声纹。" })}
      <div class="field"><label for="lb-name">姓名（已有的人就选，新名字会自动登记）</label>
        <input class="input" id="lb-name" list="lb-list" placeholder="例如：张三" value="${u.speaker_id != null ? u.speaker : ""}" autocomplete="off" data-autofocus>
        <datalist id="lb-list">${names.map((n) => html`<option value="${n}"></option>`)}</datalist></div>
      <div id="lb-new" class="row" hidden><label class="check"><input type="checkbox" id="lb-self"> ${me ? "这其实是我" : "这是我自己"}</label>${relationSelect(choices, "", { empty: "和我的关系（可选）" })}</div>
      <label class="check"><input type="checkbox" id="lb-whole" checked> 这段录音里同一个声音的其他句子也一起标</label>
      <p class="help">会用这句话的声音补充这个人的声纹，并重新识别同一段录音里的其他句子（这句 ${dur.toFixed(1)} 秒；太短的话效果一般，建议选 3 秒以上的句子）。</p>`,
    onMount: (h) => {
      h.okIndex = actions.length - 1;
      const inp = h.body.querySelector("#lb-name"), newBox = h.body.querySelector("#lb-new");
      wireRelationSelects(h.body);
      const sync = () => { const n = inp.value.trim(); newBox.hidden = !(n && !names.includes(n)); };   // 本人 / 关系 only for a new name
      inp.addEventListener("input", sync);
      inp.addEventListener("change", sync);
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing && u.audio) { e.preventDefault(); send(h, null, false); } });
      sync();
      const slot = h.body.querySelector("#lb-audio");
      if (slot) inline = mountInline(slot, clips.utterance(u, { pad: 0.3 }));
    },
    onClose: () => { if (inline) inline.destroy(); },
  });
  return m;
}

/**
 * Dialog: "这句不是 X？" -> POST /api/utterances/{id}/reject. The sentence becomes unidentified, is never assigned
 * back to X, and (while its audio exists) its voice is kept as a negative sample of X.
 */
export function rejectUtt(u) {
  if (!u || u.speaker_id == null) return null;
  const name = u.speaker;
  let inline = null;
  return openModal({
    title: `这句不是 ${name}？`, size: "md",
    body: html`${quote(u)}
      ${u.audio ? html`<div class="pl-slot" id="rj-audio"></div>` : notice({ kind: "warn", body: `这段音频已被删除：只会把这句标为不是 ${name}，取不到负样本。` })}
      <label class="check"><input type="checkbox" id="rj-whole"> 同一个声音的其他句子也一起（这段录音里）</label>
      <p class="help">这句会变成「未识别」，以后重新识别也不会再认成 ${name}；它的声音会作为 ${name} 的负样本，像这样的声音以后不容易再被认成 ${name}。负样本可以在说话人页的样本列表里删掉。</p>`,
    actions: [{ label: "取消" }, {
      label: `不是 ${name}`, kind: "danger solid", autofocus: true,
      run: async (h, btn) => {
        const restore = busy(btn, "处理中…");
        if (inline) inline.pause();
        try {
          const r = await api(`/api/utterances/${u.id}/reject`, { method: "POST", json: { speaker_id: u.speaker_id, whole_cluster: h.body.querySelector("#rj-whole").checked } });
          toastReject(r, name);
          emit("speakers:changed", { reason: "reject" });
          h.close(true);
          changed(u, "reject", Array.isArray(r.utterance_ids) && r.utterance_ids.length ? r.utterance_ids : [u.id]);
        } catch (e) { restore(); toastErr(e); }
      },
    }],
    onMount: (h) => { const slot = h.body.querySelector("#rj-audio"); if (slot) inline = mountInline(slot, clips.utterance(u, { pad: 0.3 })); },
    onClose: () => { if (inline) inline.destroy(); },
  });
}

/**
 * Dialog: split one sentence that mixes two people. Points come from the player position ("在当前位置拆分") or
 * from an automatic by-voice preview; GET …/preview-split says who speaks in each piece (no ASR); the POST
 * transcribes the pieces again and replaces the sentence. Emits utterances:changed with the new ids.
 */
export function splitUtt(u) {
  const dur = Math.max(0, Number(u.end_s) - Number(u.start_s));
  const lo = 0.3, hi = dur - 0.3;
  let pts = [];                 // manual split points, seconds from the start of the sentence, ascending
  let mode = "";                // "at" | "auto": what 确认拆分 sends
  let preview = null;           // last preview-split answer (piece bounds + who was recognised)
  let gen = 0;                  // drops a preview answer that arrives after the points changed again
  let autoErr = "";             // why the last automatic attempt failed (shown above the manual preview)
  let inline = null;
  return openModal({
    title: "拆分这句", size: "lg",
    body: html`${quote(u)}
      <div class="pl-slot" id="sp-audio"></div>
      <div class="row"><button type="button" class="btn sm" data-x="here" title="把播放到的位置加为一个拆分点">${icon("scissors", 14)}在当前位置拆分</button><button type="button" class="btn sm" data-x="auto" title="自动找出换人的位置，并识别每段是谁">${icon("sparkle", 14)}自动按声音拆</button></div>
      <p class="help">共 ${dur.toFixed(1)} 秒：听到换人的地方暂停，点「在当前位置拆分」，可以加多个点。</p>
      <div class="split-chips" id="sp-chips" aria-live="polite"></div>
      <div id="sp-preview"></div>
      <p class="help">确认后每一段会重新转写（按秒数计费）并各自识别说话人，原来这句会被替换。</p>`,
    actions: [{ label: "取消" }, {
      label: "确认拆分", kind: "primary", disabled: true,
      run: async (h, btn) => {
        if (!preview || !mode) return;
        const restore = busy(btn, "拆分并重新转写…");
        player.pause();
        try {
          const r = await api(`/api/utterances/${u.id}/split`, { method: "POST", json: mode === "auto" ? { auto: true } : { at: pts } });
          toast(`已拆成 ${r.pieces.length} 句，重新转写了 ${Number(r.billed_s || 0).toFixed(1)} 秒`, { kind: "ok", ms: 6000 });
          h.close(true);
          emit("utterances:changed", { day: u.day, ids: r.pieces.map((p) => p.id), removed: [u.id], reason: "split" });
        } catch (e) { restore(); toastErr(e); }
      },
    }],
    onMount: (h) => {
      const body = h.body, chipsBox = body.querySelector("#sp-chips"), prev = body.querySelector("#sp-preview"), okBtn = h.actionButton(1);
      inline = mountInline(body.querySelector("#sp-audio"), clips.utterance(u, { pad: 0 }));
      const renderChips = () => render(chipsBox, pts.length
        ? html`<span class="small t3">拆分点：</span>${pts.map((p, i) => html`<span class="chip static">${p.toFixed(1)}s<button type="button" class="chip-x" data-rm="${i}" aria-label="去掉拆分点 ${p.toFixed(1)} 秒">${icon("x", 12)}</button></span>`)}`
        : html`<span class="small t3">还没有拆分点</span>`);
      const errBox = () => (autoErr ? notice({ kind: "warn", compact: true, body: `自动拆分失败：${autoErr}${pts.length ? "（手动拆分点还在）" : ""}` }) : "");
      const renderPreview = () => {
        if (!preview) { render(prev, html`${errBox()}`); return; }
        render(prev, html`${errBox()}<div class="small t3">${mode === "auto" ? "按声音分出" : "按拆分点分出"} ${preview.pieces.length} 段，可以逐段试听：</div>
          <div class="split-pieces">${preview.pieces.map((p, i) => html`<div class="split-piece" data-clip="${clips.segment(preview.segment_id, p.start_s, p.end_s).key}">
            <span class="split-n">${i + 1}</span><span class="split-t">${Number(p.rel_start).toFixed(1)}–${Number(p.rel_end).toFixed(1)}s</span>
            <span class="spk-name ${p.speaker_id ? colorClass({ speaker_id: p.speaker_id, speaker: p.speaker }) : "c-unk"}">${p.speaker}</span>${when(p.score != null, () => html`<span class="spk-score">${Number(p.score).toFixed(2)}</span>`)}
            ${iconBtn({ icon: "play", label: `试听第 ${i + 1} 段`, attrs: { "data-pplay": i }, size: 15 })}</div>`)}</div>`);
        player.mark(prev);
      };
      const setPreview = (p, m) => { preview = p; mode = p ? m : ""; okBtn.disabled = !p; renderPreview(); };
      const loadPreview = async (m) => {
        const g = ++gen;
        okBtn.disabled = true;
        render(prev, html`${errBox()}${spinner(m === "auto" ? "正在按声音分析…" : "正在识别每段的说话人…")}`);
        try {
          const r = await api(`/api/utterances/${u.id}/preview-split`, { query: m === "auto" ? { auto: 1 } : { auto: 0, at: pts.map((p) => p.toFixed(2)).join(",") } });
          if (g !== gen || h.closed) return;
          autoErr = "";
          if (m === "auto" && pts.length) { pts = []; renderChips(); }     // manual points are dropped only once the automatic preview succeeded
          setPreview(r, m);
        } catch (e) {
          if (g !== gen || h.closed) return;
          if (m === "auto") {
            autoErr = e.message || String(e);
            if (pts.length) { await loadPreview("at"); return; }           // keep the manual points: fall back to their preview
            setPreview(null, "");
            return;
          }
          setPreview(null, "");
          render(prev, html`${errBox()}${notice({ kind: "warn", compact: true, body: e.message || String(e) })}`);
        }
      };
      renderChips();
      body.addEventListener("click", (ev) => {
        const x = ev.target.closest("[data-x]");
        if (x && x.dataset.x === "here") {
          if (dur < 1.2) { toast("这句太短，无法拆分", { kind: "err" }); return; }
          const t = Math.round(inline.time() * 10) / 10;
          if (t <= lo || t >= hi) { toast(`拆分点要在 ${lo.toFixed(1)}s 和 ${hi.toFixed(1)}s 之间（先播放到换人的位置再点）`, { kind: "err", ms: 5000 }); return; }
          if (pts.some((p) => Math.abs(p - t) < 0.05)) return;
          pts = [...pts, t].sort((a, b) => a - b);
          renderChips();
          loadPreview("at");
          return;
        }
        if (x && x.dataset.x === "auto") { autoErr = ""; loadPreview("auto"); return; }     // pts stay until the auto preview succeeds
        const rm = ev.target.closest("[data-rm]");
        if (rm) {
          pts = pts.filter((_, i) => i !== Number(rm.dataset.rm));
          renderChips();
          if (pts.length) loadPreview("at"); else { gen++; setPreview(null, ""); }
          return;
        }
        const pp = ev.target.closest("[data-pplay]");
        if (pp && preview) {
          const p = preview.pieces[Number(pp.dataset.pplay)];
          if (p) player.playOrToggle(clips.segment(preview.segment_id, p.start_s, p.end_s, `第 ${Number(pp.dataset.pplay) + 1} 段 · ${p.speaker}`));
        }
      });
    },
    onClose: () => { if (inline) inline.destroy(); player.stop(); },
  });
}

/** Clips of the sentences that still have audio (a session's play queue). */
export const clipsOf = (utts) => utts.filter((u) => u.audio).map((u) => clips.utterance(u, { cls: colorClass(u) }));
