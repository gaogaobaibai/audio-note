// 未识别的声音: groups of sentences that sound like the same unknown person (GET /api/voices). Listen, type a
// name -> POST /api/voices/name enrols the person from those recordings; 「不是 X」 next to a 「可能是 X」 hint ->
// POST /api/voices/reject. What was typed into a card survives re-renders (job finished, another card named).
import { html, when } from "../../core/html.js";
import { $$, render, on, insert } from "../../core/dom.js";
import { api, isAbort, isMismatch } from "../../core/api.js";
import { emit } from "../../core/bus.js";
import { status, isReadOnly } from "../../core/store.js";
import { refreshStatus, runJob, jobRunning } from "../../core/status.js";
import { button, busy, emptyState, iconBtn, notice, sectionHead, spinner } from "../../ui/components.js";
import { CUSTOM_REL, mismatchNotice, relationSelect, relationValue, toastReject, wireRelationSelects } from "../../ui/speaker.js";
import { clips, player } from "../../ui/player.js";
import { confirm } from "../../ui/modal.js";
import { toast, toastErr } from "../../ui/toast.js";

const autoScanned = {};       // profile name -> true: the "voices" scan job is submitted at most once per profile per session
const drafts = {};            // group key -> {name, sel, custom, self, mismatch?}: what the user typed into that card
const mismatches = {};        // group key -> the 400 mismatch payload whose 「仍然加入」 note is showing
const notHints = {};          // "groupKey|speakerId" -> true: the user said 「不是 X」; hide the 「可能是 X」 hint
const letter = (i) => (i < 26 ? String.fromCharCode(65 + i) : String(i + 1));
const profileName = () => { const st = status(); return (st && st.profile && st.profile.name) || ""; };

const PAGE = 12;             // voice cards shown at a time (a busy profile has dozens of unknown voices)

export function createVoices(ctx, box, { speakers }) {
  let voices = null, pendingPaint = false, shown = PAGE;

  const stash = () => {
    for (const c of $$(".voice", box)) {
      const f = c.querySelector(".voice-form");
      if (!f) continue;
      const d = { name: f.querySelector("[data-vname]").value, sel: f.querySelector("[data-rel-sel]").value, custom: f.querySelector("[data-rel-custom]").value, self: f.querySelector("[data-vself]").checked };
      if (c.querySelector(".mismatch-note") && mismatches[c.dataset.key]) d.mismatch = mismatches[c.dataset.key];
      if (d.name || d.sel || d.self || d.mismatch) drafts[c.dataset.key] = d; else delete drafts[c.dataset.key];
    }
  };
  const restore = () => {
    for (const c of $$(".voice", box)) {
      const d = drafts[c.dataset.key];
      if (!d) continue;
      const f = c.querySelector(".voice-form");
      if (!f) continue;
      f.querySelector("[data-vname]").value = d.name;
      f.querySelector("[data-rel-sel]").value = d.sel;
      const custom = f.querySelector("[data-rel-custom]");
      custom.hidden = d.sel !== CUSTOM_REL;
      custom.value = d.custom;
      f.querySelector("[data-vself]").checked = d.self;
      if (d.mismatch) { mismatches[c.dataset.key] = d.mismatch; insert(f, "beforebegin", mismatchNotice(d.mismatch, "data-vforce")); }
    }
  };

  const card = (g, i) => {
    const sp = speakers();
    const me = sp && sp.self;
    const days = g.days || [];
    const range = days.length > 1 ? `${days[0].slice(5)} ~ ${days[days.length - 1].slice(5)}` : (days[0] || "").slice(5);
    const close = g.closest && g.closest.score >= 0.3 && !notHints[`${g.key}|${g.closest.id}`] ? g.closest : null;
    const ro = isReadOnly();
    return html`<article class="card voice" data-key="${g.key}" aria-label="声音 ${letter(i)}">
      <header class="voice-head"><div><h3>声音 ${letter(i)}</h3>
        <div class="small t2">${g.utterances} 句 · ${Math.round(g.speech_s)} 秒 · ${g.n_days} 天${range ? `（${range}）` : ""}${g.segments > 1 ? ` · ${g.segments} 段录音` : ""}</div></div>
        ${when(close && !ro, () => html`<div class="voice-hint">
          <button type="button" class="chip" data-vfill="${close.name}" title="与已登记声纹的相似度 ${Number(close.score).toFixed(2)}，点一下填入名字（不一定对，先听一听）">可能是 ${close.name}<span class="chip-n">${Number(close.score).toFixed(2)}</span></button>
          ${when(close.id != null, () => html`<button type="button" class="chip" data-vnot="${close.id}" data-vnotname="${close.name}" title="听过了，不是 ${close.name}：这组句子以后不会被认成 TA，并用它们的声音给 ${close.name} 加负样本">不是 ${close.name}</button>`)}</div>`)}
      </header>
      <ul class="voice-samples">${(g.samples || []).length ? g.samples.map((s) => html`<li class="voice-sample ${s.audio ? "" : "noaudio"}" data-utt="${s.utterance_id}">
          <span class="row-time">${String(s.day || "").slice(5)} ${s.time || ""}</span><span class="voice-text clip" title="${s.text}">${s.text || "（没有文字）"}</span><span class="row-time">${Number(s.seconds || 0).toFixed(1)}s</span>
          ${s.audio ? iconBtn({ icon: "play", label: "播放这句", attrs: { "data-vplay": s.utterance_id }, size: 14 }) : html`<span class="icon-btn" title="音频已删除，无法播放" aria-hidden="true"></span>`}</li>`) : html`<li class="t3 small">（没有示例句）</li>`}</ul>
      ${when(!ro, () => html`<div class="voice-form">
        <input class="input" data-vname list="spk-names" placeholder="这是谁？" aria-label="声音 ${letter(i)} 是谁" autocomplete="off">
        ${relationSelect((sp && sp.choices) || [])}
        <label class="check"><input type="checkbox" data-vself> ${me ? "这其实是我" : "这是我自己"}</label>
        ${button({ label: "就是 TA", kind: "primary", size: "sm", attrs: { "data-vok": "" } })}</div>`)}
    </article>`;
  };

  const paint = () => {
    const v = voices;
    if (!v || !ctx.alive()) return;
    const sp = speakers();
    const scanning = !!(v.scanning || jobRunning("voices"));
    const groups = v.groups || [];
    const ro = isReadOnly();
    if (!groups.length && !(sp && sp.list.length) && !scanning && !v.missing_segments && !v.unknown_utterances) {
      render(box, html`<div class="card">${emptyState({ icon: "people", title: "还没有听到任何人", text: "录一会儿或导入音频后，这里会列出听到的声音，告诉它每个声音是谁就行。", action: html`<a class="btn" href="#/dashboard">去总览开始录音</a>` })}</div>`);
      return;
    }
    const notes = [];
    if (scanning) notes.push(notice({ kind: "info", body: spinner(`正在分析${v.missing_segments ? ` ${v.missing_segments} 段` : ""}录音里的声音…完成后这里会自动更新。`) }));
    else if (v.missing_segments) notes.push(notice({ kind: "warn", body: `还有 ${v.missing_segments} 段录音没有分析过声音${ro ? "（只读模式下无法扫描，请在运行服务的那个界面操作）" : ""}。`, actions: ro ? "" : button({ label: "扫描", size: "sm", attrs: { "data-vact": "scan" } }) }));
    if (!groups.length && !scanning && !v.missing_segments && v.unknown_utterances) notes.push(notice({ kind: "info", body: html`有 ${v.unknown_utterances} 句未识别，但它们的音频已不在，无法按声音分组；可以在<a href="#/transcript">记录</a>里逐句标记。` }));
    stash();
    render(box, html`${sectionHead({ title: "未识别的声音", sub: v.unknown_utterances ? `${v.unknown_utterances} 句还没认出是谁说的` : "" })}
      <datalist id="spk-names">${((sp && sp.names) || []).map((n) => html`<option value="${n}"></option>`)}</datalist>
      ${when(notes.length, () => html`<div class="notices">${notes}</div>`)}
      ${groups.length ? html`<div class="voice-grid">${groups.slice(0, shown).map(card)}</div>
          ${when(groups.length > shown, () => html`<div class="voice-more">${button({ label: `再显示 ${Math.min(PAGE, groups.length - shown)} 个（还有 ${groups.length - shown} 个声音）`, kind: "ghost", attrs: { "data-vact": "more" } })}</div>`)}` : when(!notes.length, () => html`<div class="card sunk">${emptyState({ compact: true, icon: "check", title: "没有待确认的声音", text: "新的陌生声音出现时会列在这里。" })}</div>`)}`);
    wireRelationSelects(box);
    restore();
    player.mark(box);
  };

  /** A re-render replaces the inputs: while the user types into a card, wait until focus leaves it. */
  const editing = () => { const a = document.activeElement; return !!(a && box.contains(a) && a.matches("input, select, textarea")); };
  const paintWhenIdle = () => { if (editing()) { pendingPaint = true; return; } pendingPaint = false; paint(); };
  box.addEventListener("focusout", () => { setTimeout(() => { if (ctx.alive() && pendingPaint && !editing()) paintWhenIdle(); }, 0); }, { signal: ctx.signal });

  const load = async () => {
    try { voices = await api("/api/voices", { signal: ctx.signal }); }
    catch (e) { if (isAbort(e)) return; if (ctx.alive() && !voices) render(box, notice({ kind: "danger", body: e.message || String(e) })); return; }
    if (!ctx.alive()) return;
    const prof = profileName();
    if (voices.missing_segments > 0 && !voices.scanning && !isReadOnly() && !autoScanned[prof] && status()) {
      autoScanned[prof] = true;
      try { await api("/api/jobs/voices", { method: "POST", json: {} }); voices.scanning = true; refreshStatus(); }
      catch (e) { toast(`无法自动分析录音：${e.message}`, { kind: "err", ms: 6000 }); }
    }
    paintWhenIdle();
  };

  const groupOf = (cardEl) => (voices && (voices.groups || []).find((x) => x.key === cardEl.dataset.key)) || null;

  const nameVoice = async (cardEl, force = false) => {
    const g = groupOf(cardEl);
    if (!g) return;
    const sp = speakers(), me = sp && sp.self;
    const f = cardEl.querySelector(".voice-form");
    for (const n of $$(".mismatch-note", cardEl)) n.remove();
    delete mismatches[g.key];
    const inp = f.querySelector("[data-vname]"), name = inp.value.trim();
    if (!name) { toast("先填上这是谁", { kind: "err" }); inp.focus(); return; }
    const isSelf = f.querySelector("[data-vself]").checked;
    if (isSelf && me && me.name !== name && !(await confirm({ title: "改为本人", text: `目前本人是「${me.name}」，改成「${name}」？日报和问答会改站在 ${name} 的视角。`, okLabel: "改为本人" }))) return;
    const rel = relationValue(f);
    const restoreBtn = busy(f.querySelector("[data-vok]"), "提取声纹…");
    try {
      const r = await api("/api/voices/name", { method: "POST", json: { members: g.members, name, is_self: isSelf, relation: rel || null, auto_relabel: true, force } });
      const rejected = r.samples_rejected ? `，${r.samples_rejected} 段声音和已有样本不像没有加入` : "";
      const note = r.sample_note ? `；${r.sample_note}` : "";
      const noSample = !r.samples_added;       // named, but no voiceprint came out of it (too quiet / all rejected)
      toast(`已标记为 ${name}：${r.utterances_assigned} 句，补充 ${r.samples_added} 个声纹样本${rejected}${noSample && !note && !rejected ? "（没有可用的声纹样本，之后不会自动认出 TA）" : ""}${note}${r.relabel_started ? "；正在用新声纹重新识别其他记录" : ""}`, { kind: noSample || note ? "warn" : "ok", ms: noSample || note ? 9000 : 7000 });
      inp.value = "";                          // this card is done: nothing of it may come back as a draft
      f.querySelector("[data-vself]").checked = false;
      f.querySelector("[data-rel-sel]").value = "";
      delete drafts[g.key];
      stash();                                 // keep what was typed into the OTHER cards
      if (isSelf) await refreshStatus();
      emit("speakers:changed", { reason: "voice-named" });
      emit("utterances:changed", { reason: "voice-named" });
    } catch (err) {
      restoreBtn();
      if (isMismatch(err)) { toast(err.message, { kind: "err", ms: 8000 }); mismatches[g.key] = err.data; insert(f, "beforebegin", mismatchNotice(err.data, "data-vforce")); }
      else toastErr(err);
    }
  };

  const rejectVoice = async (cardEl, btn) => {
    const g = groupOf(cardEl);
    if (!g) return;
    const sid = Number(btn.dataset.vnot), name = btn.dataset.vnotname || "TA";
    const title = cardEl.querySelector("h3") ? cardEl.querySelector("h3").textContent : "这组声音";
    if (!(await confirm({ title: `${title} 不是 ${name}`, text: `这组 ${g.utterances} 句以后不会被认成 ${name}，并用它们的声音给 ${name} 加负样本（之后像这样的声音不容易再被认成 ${name}）。负样本可以在 ${name} 的样本列表里删掉。`, okLabel: `不是 ${name}`, danger: true }))) return;
    btn.disabled = true;
    try {
      const r = await api("/api/voices/reject", { method: "POST", json: { members: g.members, speaker_id: sid } });
      toastReject(r, name);
      notHints[`${g.key}|${sid}`] = true;
      emit("speakers:changed", { reason: "voice-rejected" });
    } catch (err) { btn.disabled = false; toastErr(err); }
  };

  on(box, "click", "[data-vplay]", (e, b) => {
    const g = groupOf(b.closest(".voice"));
    if (!g) return;
    const list = (g.samples || []).filter((s) => s.audio);
    const i = list.findIndex((s) => String(s.utterance_id) === b.dataset.vplay);
    player.playOrToggle(list.map((s) => clips.utterance({ id: s.utterance_id, speaker: b.closest(".voice").querySelector("h3").textContent, time: s.time, text: s.text }, { pad: 0.3 })), { index: Math.max(0, i) });
  }, { signal: ctx.signal });
  on(box, "click", "[data-vact]", async (e, b) => {
    if (b.dataset.vact === "more") { shown += PAGE; paint(); return; }
    if (b.dataset.vact === "scan") { autoScanned[profileName()] = true; await runJob("voices"); await load(); }
  }, { signal: ctx.signal });
  on(box, "click", "[data-vok]", (e, b) => nameVoice(b.closest(".voice")), { signal: ctx.signal });
  on(box, "click", "[data-vforce]", (e, b) => nameVoice(b.closest(".voice"), true), { signal: ctx.signal });
  on(box, "click", "[data-vfill]", (e, b) => { const inp = b.closest(".voice").querySelector("[data-vname]"); inp.value = b.dataset.vfill; inp.focus(); }, { signal: ctx.signal });
  on(box, "click", "[data-vnot]", (e, b) => { if (!b.disabled) rejectVoice(b.closest(".voice"), b); }, { signal: ctx.signal });
  on(box, "keydown", "[data-vname]", (e, inp) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); nameVoice(inp.closest(".voice")); } }, { signal: ctx.signal });

  return { load, paint: paintWhenIdle, destroy: () => { stash(); } };
}
