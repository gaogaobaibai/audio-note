// 说话人: the people the recogniser knows, and the voices it does not know yet.
// Each person card has ONE primary action (核对, or 补充样本 when there is nothing to check) and an overflow menu;
// samples open in a drawer, the review queue and enrolment in dialogs. Mutations re-read /api/speakers and patch
// the two regions; nothing rebuilds the page (an open drawer / dialog and typed-in names survive).
import { html, when } from "../core/html.js";
import { $, render, patch, on } from "../core/dom.js";
import { api, isAbort } from "../core/api.js";
import { emit } from "../core/bus.js";
import { loadSpeakers, hasApiKey, isReadOnly, unknownLabel } from "../core/store.js";
import { ensureStatus, refreshStatus, runJob } from "../core/status.js";
import { debounce, num } from "../core/util.js";
import { icon } from "../ui/icons.js";
import { badge, button, iconBtn, notice, sectionHead, skeleton } from "../ui/components.js";
import { avatar, relationSelect, relationValue, rememberSpeakers, wireRelationSelects } from "../ui/speaker.js";
import { confirm, openModal, prompt } from "../ui/modal.js";
import { openMenu } from "../ui/popover.js";
import { toast, toastErr } from "../ui/toast.js";
import { createVoices } from "./people/voices.js";
import { openReview } from "./people/review.js";
import { openSamples, openCalibration } from "./people/samples.js";
import { openEnroll } from "./people/enroll.js";

const cnt = (v) => num(v, 0);

/** "X 和我的关系" dialog; resolves true when something was saved. */
function relationModal(spk, choices) {
  const auto = spk.relation_source === "auto";
  const initial = spk.relation || "";
  let noteTouched = false;                 // an untouched note box must not overwrite an auto-inferred reason
  const patchSpk = (body) => api(`/api/speakers/${spk.id}`, { method: "PATCH", json: body });
  const actions = [];
  if (spk.relation) actions.push({ label: "清除关系", kind: "ghost", align: "left", run: async (h) => { try { await patchSpk({ relation: "" }); h.close(true); } catch (e) { toastErr(e); } } });
  actions.push({ label: "取消", value: false });
  actions.push({
    label: "保存", kind: "primary",
    run: async (h) => {
      const rel = relationValue(h.body), note = h.body.querySelector("#rm-note").value.trim();
      if (rel === initial && !noteTouched) { h.close(false); return; }        // nothing changed: no PATCH (keeps source='auto' + its reason)
      try {
        if (rel) await patchSpk({ relation: rel, relation_note: note });
        else { await patchSpk({ relation: "" }); if (note) await patchSpk({ relation_note: note }); }     // "" clears relation + note; a note alone is kept
        h.close(true);
      } catch (e) { toastErr(e); }
    },
  });
  return openModal({
    title: `${spk.name} 和我的关系`, size: "sm", actions,
    body: html`<div class="field"><span class="label">关系</span>${relationSelect(choices, initial)}</div>
      <div class="field"><label for="rm-note">备注（可选）</label><input class="input" id="rm-note" value="${auto ? "" : (spk.relation_note || "")}" placeholder="例如：同一个项目组的 / 大学同学"></div>
      ${when(auto && spk.relation_note, () => html`<p class="help">自动推断的理由：${spk.relation_note}（不改动直接保存会保留这条推断）</p>`)}`,
    onMount: (h) => { wireRelationSelects(h.body); h.body.querySelector("#rm-note").addEventListener("input", () => { noteTouched = true; }); },
  }).then((v) => v === true);
}

function relationChip(s) {
  if (s.is_self) return "";
  if (isReadOnly()) return s.relation ? badge(s.relation, "plain") : "";
  if (!s.relation) return html`<button type="button" class="rel-chip empty" data-act="relation" title="标注 TA 和你的关系">${icon("plus", 12)}关系</button>`;
  const auto = s.relation_source === "auto";
  return html`<button type="button" class="rel-chip ${auto ? "auto" : ""}" data-act="relation" title="${s.relation_note || (auto ? "模型从记录里推断的，点击修改" : "点击修改")}">${s.relation}${when(auto, () => html`<small>自动</small>`)}</button>`;
}

function statLine(s) {
  const neg = cnt(s.negatives), rej = cnt(s.rejections);
  const parts = [`${s.samples} 个声纹样本`];
  if (neg) parts.push(html`<span tabindex="0" data-tip="标为「不是 TA」的声音；像它们的声音不容易再被认成 ${s.name}${rej ? `（共 ${rej} 句标为不是 TA）` : ""}">${neg} 个负样本</span>`);
  else if (rej) parts.push(html`<span tabindex="0" data-tip="这些句子不会再被认成 ${s.name}">${rej} 句标为不是 TA</span>`);
  parts.push(`${Number(s.duration_s || 0).toFixed(0)} 秒语音`, `${s.utterances} 条语句`);
  return html`${parts.map((p, i) => html`${i ? " · " : ""}${p}`)}`;
}

function sampleAlerts(s) {
  const risky = cnt(s.risky_samples), out = cnt(s.outlier_samples), amb = cnt(s.ambiguous_samples);
  const open = button({ label: "打开样本列表", size: "sm", attrs: { "data-act": "samples" } });
  const alerts = [];
  if (risky) alerts.push(notice({ kind: "danger", compact: true, body: `有 ${risky} 个样本和其他样本不像，却单独认出了很多句——可能混进了别人的声音。`, actions: open }));
  else if (out) alerts.push(notice({ kind: "warn", compact: true, body: `有 ${out} 个样本和其他样本不像（可能混了别人）。`, actions: open }));
  if (amb) alerts.push(notice({ kind: "warn", compact: true, body: `这个人的样本分成两组互不相像（${amb} 个有歧义），请试听后删掉不是 TA 的。`, actions: risky || out ? "" : open }));
  return html`${alerts}`;
}

function personCard(s, { hasSelf }) {
  const ro = isReadOnly();
  const primary = s.utterances
    ? button({ label: "核对", icon: "check", kind: "primary", size: "sm", attrs: { "data-act": "review" }, title: `逐句确认自动认成 ${s.name} 的句子（最没把握的在前）` })
    : button({ label: "补充样本", icon: "mic", kind: "primary", size: "sm", attrs: { "data-act": "add" } });
  return html`<article class="card person ${s.is_self ? "me" : ""}" data-sid="${s.id}" aria-label="${s.name}">
    <div class="person-top">${avatar(s, 44)}
      <div class="grow"><div class="person-name"><span class="clip">${s.name}</span>${when(s.is_self, () => badge("我", "me", "本人"))}${relationChip(s)}</div>
        <div class="person-stat">${statLine(s)}</div>${when(s.notes, () => html`<div class="person-note">${s.notes}</div>`)}</div>
      <div class="person-acts">${ro ? button({ label: "样本", size: "sm", attrs: { "data-act": "samples" } }) : html`${primary}${iconBtn({ icon: "more", label: `${s.name} 的更多操作`, attrs: { "data-act": "menu", "data-has-self": hasSelf ? "1" : "" } })}`}</div>
    </div>
    ${sampleAlerts(s)}
  </article>`;
}

export default {
  async mount(ctx) {
    await ensureStatus();
    if (!ctx.alive()) return;
    render(ctx.el, html`<div class="page people">
      <header class="page-head"><p class="lead">不用单独录样本：下面「未识别的声音」里听一下是谁、填上名字就行。登记过的人会自动识别，认错的可以「核对」，或在记录里标为「不是 TA」。</p>
        <div class="page-acts" id="pp-acts"></div></header>
      <section id="pp-people" aria-label="已登记的人">${skeleton("cards", 2)}</section>
      <section id="pp-voices" aria-label="未识别的声音">${skeleton("cards", 2)}</section>
      <details class="disclosure"><summary><span>怎么让它认得更准</span></summary><ul class="tips">
        <li>给一组声音起名后，会从这些录音里自动提取声纹样本，之后这个人在新录音里会被自动识别。</li>
        <li>认错了人：在「记录」里对那句话点「不是 TA」，或在这里点「核对」逐句确认；标为不是 TA 的声音会成为负样本，以后像它的声音不容易再被认成这个人。</li>
        <li>样本列表里「单独认出 N 句」是只靠这一个样本才认出的句数：一个和其他样本不像的样本却单独认出很多句，多半混进了别人的声音，试听后删掉。</li>
        <li>开会、通话时系统声音里的对方不会被认成本人。</li>
        <li>把自己标为「本人」后，日报和问答会站在你的视角；关系可以手动标，也可以让模型从对话里推断。</li>
        <li id="pp-thr"></li></ul></details>
    </div>`);

    let sp = null;
    const paintPeople = () => {
      if (!sp) return;
      const people = [...sp.list].sort((a, b) => (b.is_self ? 1 : 0) - (a.is_self ? 1 : 0));
      const me = sp.self;
      patch($("#pp-people"), people.length
        ? html`${sectionHead({ title: "已登记的人", sub: me ? `本人：${me.name}` : "还没有标出哪位是本人（卡片的「⋯」里可以设）" })}
            <div class="person-grid">${people.map((s) => personCard(s, { hasSelf: !!me }))}</div>`
        : html``);
      patch($("#pp-acts"), isReadOnly() ? html`` : html`${button({ label: "补充样本", icon: "mic", attrs: { "data-page-act": "enroll" }, title: "用麦克风或录音文件给某个人补充声纹样本" })}${iconBtn({ icon: "more", label: "更多操作", attrs: { "data-page-act": "menu" } })}`);
      $("#pp-thr").textContent = `当前识别阈值 ${sp.threshold}；样本多了之后，可以在右上角「⋯ → 校准阈值」里看建议值。`;
    };
    const voices = createVoices(ctx, $("#pp-voices"), { speakers: () => sp });

    const reload = async ({ withVoices = true } = {}) => {
      try { sp = await loadSpeakers({ force: true }); }
      catch (e) { if (isAbort(e)) return; if (!sp) render($("#pp-people"), notice({ kind: "danger", body: e.message || String(e) })); return; }
      if (!ctx.alive()) return;
      rememberSpeakers(sp.list);
      paintPeople();
      if (withVoices) await voices.load();
    };
    const reloadSoon = debounce(() => reload(), 300);
    const spkOf = (el) => { const card = el.closest("[data-sid]"); return card && sp ? sp.byId.get(Number(card.dataset.sid)) || null : null; };
    const patchSpk = async (spk, body, okMsg) => {
      try { await api(`/api/speakers/${spk.id}`, { method: "PATCH", json: body }); if (okMsg) toast(okMsg, { kind: "ok" }); if ("is_self" in body) await refreshStatus(); emit("speakers:changed", { reason: "edit" }); }
      catch (e) { toastErr(e); }
    };

    const personAct = async (a, spk, anchor) => {
      const me = sp.self;
      if (a === "menu") {
        const out = cnt(spk.outlier_samples);
        const pick = await openMenu(anchor, [
          { id: "samples", label: "样本列表", icon: "list", hint: "试听、删除样本和负样本" },
          spk.utterances ? { id: "add", label: "补充样本", icon: "mic" } : null,
          "-",
          spk.is_self ? { id: "unself", label: "取消本人", icon: "user" } : { id: "self", label: me ? "改为本人" : "设为本人", icon: "user-check", hint: "日报和问答会站在本人的视角" },
          spk.is_self ? null : { id: "relation", label: spk.relation ? "修改关系" : "标注关系", icon: "people" },
          { id: "rename", label: "重命名", icon: "edit" },
          { id: "notes", label: "备注", icon: "doc" },
          out ? { id: "outliers", label: `移除 ${out} 个不像的样本`, icon: "scissors" } : null,
          "-",
          { id: "delete", label: "删除这个人", icon: "trash", danger: true },
        ], { label: `${spk.name} 的操作` });
        if (pick) await personAct(pick, spk, anchor);
        return;
      }
      if (a === "review") { if (await openReview(spk)) { emit("speakers:changed", { reason: "review" }); emit("utterances:changed", { reason: "review" }); } }
      else if (a === "samples") openSamples(spk);
      else if (a === "add") openEnroll(spk.name, sp.names);
      else if (a === "relation") { if (await relationModal(spk, sp.choices)) { toast("已保存关系", { kind: "ok" }); emit("speakers:changed", { reason: "relation" }); } }
      else if (a === "self") {
        const text = me ? `把本人从「${me.name}」改为「${spk.name}」？日报、推送和问答都会站在本人的视角，记录里本人的话就是“我”说的。` : `把「${spk.name}」设为本人（你自己）？日报、推送和问答都会站在本人的视角，记录里本人的话就是“我”说的。`;
        if (await confirm({ title: me ? "改为本人" : "设为本人", text, okLabel: me ? "改为本人" : "设为本人" })) await patchSpk(spk, { is_self: true }, `本人：${spk.name}`);
      } else if (a === "unself") {
        if (await confirm({ title: "取消本人", text: `不再把「${spk.name}」当作本人；之后可以再设。` })) await patchSpk(spk, { is_self: false });
      } else if (a === "rename") {
        const n = await prompt({ title: "重命名", label: "新名字", value: spk.name });
        if (n && n !== spk.name) await patchSpk(spk, { name: n }, "已重命名");
      } else if (a === "notes") {
        const n = await prompt({ title: `${spk.name} 的备注`, label: "例如：项目组的同事、楼下邻居", value: spk.notes || "" });
        if (n !== null) await patchSpk(spk, { notes: n });
      } else if (a === "outliers") {
        if (await confirm({ title: "移除不像的样本", text: `删除 ${spk.name} 的 ${spk.outlier_samples} 个和其他样本不像的声纹样本（多半是混进来的别人的声音）。删掉后建议重新识别全部记录。`, okLabel: "移除", danger: true })) {
          try { const r = await api(`/api/speakers/${spk.id}/samples/outliers`, { method: "DELETE" }); toast(`已移除 ${r.deleted} 个样本，还剩 ${r.remaining} 个`, { kind: "ok" }); emit("speakers:changed", { reason: "samples" }); } catch (e) { toastErr(e); }
        }
      } else if (a === "delete") {
        if (await confirm({ title: `删除 ${spk.name}`, text: `删除声纹和 ${spk.utterances} 条语句上的名字（语句会变成「${sp.unknownLabel || unknownLabel()}」，文字不会丢）。`, okLabel: "删除", danger: true })) {
          try { await api(`/api/speakers/${spk.id}`, { method: "DELETE" }); toast("已删除"); emit("speakers:changed", { reason: "delete" }); emit("utterances:changed", { reason: "speaker-deleted" }); } catch (e) { toastErr(e); }
        }
      }
    };

    on($("#pp-people"), "click", "[data-act]", (e, b) => { const spk = spkOf(b); if (spk) personAct(b.dataset.act, spk, b).catch(toastErr); }, { signal: ctx.signal });
    on($("#pp-acts"), "click", "[data-page-act]", async (e, b) => {
      if (b.dataset.pageAct === "enroll") { openEnroll("", sp ? sp.names : []); return; }
      const pick = await openMenu(b, [
        { id: "relations", label: "自动推断关系", icon: "sparkle", hint: "让模型从对话里推断每个人和你的关系" },
        { id: "relabel", label: "重新识别全部记录", icon: "refresh", hint: "用现在的声纹重新认一遍，不重新转写" },
        { id: "calib", label: "校准阈值", icon: "target", hint: "看同一人 / 不同人的相似度分布" },
      ]);
      if (pick === "relations") {
        if (!hasApiKey()) { toast("没有配置 MiMo API key，无法推断关系", { kind: "err" }); return; }
        if (await confirm({ title: "自动推断关系", text: "让模型根据记录里的对话推断每个人和你的关系，只处理语句足够多、还没手动标过的人；结果会标成「自动」，可以随时改。", okLabel: "开始推断" })) runJob("relations");
      } else if (pick === "relabel") {
        if (await confirm({ title: "重新识别全部记录", text: "用当前登记的声纹重新识别所有已转写的语句，不重新转写、不花钱；记录多的话要几分钟。手动标过的句子不会改。", okLabel: "重新识别" })) runJob("relabel");
      } else if (pick === "calib") openCalibration();
    }, { signal: ctx.signal });

    ctx.on("speakers:changed", reloadSoon);
    ctx.on("live:speakers", reloadSoon);
    ctx.onJob(["voices", "relabel", "relations", "process", "rescore"], reloadSoon);
    ctx.on("resync", reloadSoon);
    ctx.on("status", (st, prev) => { if (prev && prev.service && st.service && prev.service.read_only !== st.service.read_only) { paintPeople(); voices.paint(); } });

    await reload();
    return () => voices.destroy();
  },
};
