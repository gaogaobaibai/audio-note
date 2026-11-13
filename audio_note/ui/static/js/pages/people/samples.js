// A person's voiceprint samples and negative samples (drawer), and the threshold calibration dialog.
//   GET /api/speakers/{id}/samples   -> [{id, file, origin, duration_s, consistency, outlier, ambiguous, influence, created_at, audio}]
//   GET /api/speakers/{id}/negatives -> [{id, duration_s, file, audio, utterance_id, created_at, text?}]   (absent on an older server)
import { html, when } from "../../core/html.js";
import { render, on } from "../../core/dom.js";
import { api } from "../../core/api.js";
import { emit } from "../../core/bus.js";
import { fmtDT } from "../../core/format.js";
import { num } from "../../core/util.js";
import { status, isReadOnly } from "../../core/store.js";
import { badge, emptyState, iconBtn, notice, skeleton } from "../../ui/components.js";
import { ORIGIN_LABEL } from "../../ui/speaker.js";
import { clips, player } from "../../ui/player.js";
import { confirm, openDrawer, openModal } from "../../ui/modal.js";
import { toast, toastErr } from "../../ui/toast.js";

/** Outlier samples that alone recognised at least this many sentences are flagged (the server's risky_samples cut). */
const riskyInfluence = () => { const st = status(); return num(st && st.risky_influence, 20); };

export function openSamples(spk) {
  const ro = isReadOnly();
  const drawer = openDrawer({
    title: `${spk.name} 的样本`, body: skeleton("rows", 5),
    onMount: (h) => {
      const load = async () => {
        let rows, negs;
        try {
          // negatives: an older server has no such endpoint -> null = leave the group out
          [rows, negs] = await Promise.all([api(`/api/speakers/${spk.id}/samples`), api(`/api/speakers/${spk.id}/negatives`).catch(() => null)]);
        } catch (e) { if (!h.closed) render(h.body, notice({ kind: "danger", body: e.message || String(e) })); return; }
        if (h.closed) return;
        const hasInf = rows.some((r) => r.influence != null);
        const inf = (r) => num(r.influence, 0);
        if (hasInf) rows = [...rows].sort((x, y) => inf(y) - inf(x) || x.id - y.id);
        const cut = riskyInfluence();
        const risky = (r) => r.outlier && inf(r) >= cut;
        const consText = (r) => (r.consistency == null ? "–" : `${Number(r.consistency).toFixed(2)}`);
        const consTip = (r) => (r.consistency == null ? "只有这一个样本，没法比" : r.outlier ? "和其他样本不像，可能混了别人" : r.ambiguous ? "歧义：样本分成两组互不相像，这个属于其中一组；试听后删掉不是 TA 的" : "和这个人其他样本的最高相似度");
        const sampleRows = rows.map((r) => html`<li class="smp ${risky(r) ? "risky" : r.outlier ? "outlier" : ""}" data-clip="sample:${r.id}">
          <div class="smp-main"><div class="smp-title"><span class="clip" title="${r.file}">${r.file}</span>${r.origin && ORIGIN_LABEL[r.origin] ? badge(ORIGIN_LABEL[r.origin], "plain", "样本是怎么来的") : ""}${when(risky(r), () => badge("可疑", "danger"))}${when(!risky(r) && r.outlier, () => badge("不像", "warn"))}${when(r.ambiguous && !r.outlier, () => badge("有歧义", "warn"))}</div>
            <div class="smp-meta">${Number(r.duration_s || 0).toFixed(1)} 秒 · <span tabindex="0" data-tip="${consTip(r)}">一致性 ${consText(r)}</span>${when(hasInf, () => html` · <span tabindex="0" data-tip="${inf(r) ? `只靠这个样本才认出的 ${inf(r)} 句（去掉它这些句子就认不出了）` : "去掉这个样本也不影响识别"}">${inf(r) ? `单独认出 ${inf(r)} 句` : "没有单独认出的句子"}</span>`)} · ${fmtDT(r.created_at)}</div></div>
          <div class="smp-acts">${r.audio ? iconBtn({ icon: "play", label: "试听这个样本", attrs: { "data-play": r.id }, size: 15 }) : ""}${ro ? "" : iconBtn({ icon: "trash", label: "删除这个样本", attrs: { "data-del": r.id }, kind: "warn", size: 15 })}</div></li>`);
        const negRows = (negs || []).map((n) => html`<li class="smp" data-clip="neg:${n.id}">
          <div class="smp-main"><div class="smp-title"><span class="clip" title="${n.text || n.file || ""}">${n.text || n.file || `#${n.id}`}</span></div>
            <div class="smp-meta">${Number(n.duration_s || 0).toFixed(1)} 秒 · ${fmtDT(n.created_at)}</div></div>
          <div class="smp-acts">${n.audio ? iconBtn({ icon: "play", label: "试听这个负样本", attrs: { "data-negplay": n.id }, size: 15 }) : ""}${ro ? "" : iconBtn({ icon: "trash", label: "删除这个负样本", attrs: { "data-delneg": n.id }, kind: "warn", size: 15 })}</div></li>`);
        render(h.body, html`<section><h3 class="smp-h">声纹样本<span class="t3">${rows.length}</span></h3>
            <p class="help">用来认出 ${spk.name} 的声音片段。麦克风 / 文件来的样本不可能混进别人；和其他样本不像、却单独认出很多句的样本多半混进了别人的声音：试听后删掉。</p>
            ${rows.length ? html`<ul class="smp-list">${sampleRows}</ul>` : emptyState({ compact: true, icon: "mic", title: "还没有样本" })}</section>
          ${when(negs, () => html`<section><h3 class="smp-h">负样本<span class="t3">${negs.length}</span></h3>
            <p class="help">标为「不是 TA」的声音：和它们更像的声音不会被认成这个人。</p>
            ${negs.length ? html`<ul class="smp-list">${negRows}</ul>` : html`<p class="t3 small">还没有负样本：在「核对」或「记录」里把认错的句子标为「不是 TA」就会加入。</p>`}</section>`)}`);
        player.mark(h.body);
      };
      on(h.body, "click", "[data-play]", (e, b) => player.playOrToggle(clips.sample(b.dataset.play, `${spk.name} 的样本`)));
      on(h.body, "click", "[data-negplay]", (e, b) => player.playOrToggle(clips.negative(b.dataset.negplay, `${spk.name} 的负样本`)));
      on(h.body, "click", "[data-del]", async (e, b) => {
        if (!(await confirm({ title: "删除样本", text: "删除这个声纹样本。删掉后建议重新识别记录。", okLabel: "删除", danger: true }))) return;
        try { await api(`/api/speakers/samples/${b.dataset.del}`, { method: "DELETE" }); toast("已删除样本", { kind: "ok" }); emit("speakers:changed", { reason: "samples" }); await load(); } catch (err) { toastErr(err); }
      });
      on(h.body, "click", "[data-delneg]", async (e, b) => {
        if (!(await confirm({ title: "删除负样本", text: `删除这个负样本：像它的声音又可以被认成 ${spk.name}（已经标为「不是 TA」的句子不受影响）。`, okLabel: "删除", danger: true }))) return;
        try { await api(`/api/speakers/negatives/${b.dataset.delneg}`, { method: "DELETE" }); toast("已删除负样本", { kind: "ok" }); emit("speakers:changed", { reason: "samples" }); await load(); } catch (err) { toastErr(err); }
      });
      load();
    },
    onClose: () => { const s = player.state(); if (s.clip && /^(sample|neg):/.test(s.clip.key)) player.stop(); },
  });
  return drawer;
}

/** POST /api/speakers/calibrate -> same-person / different-person similarity ranges and a suggested threshold. */
export async function openCalibration() {
  let r;
  try { r = await api("/api/speakers/calibrate", { method: "POST" }); } catch (e) { toastErr(e); return; }
  const row = (label, s) => html`<tr><td>${label}</td><td class="num">${s ? s.n : 0}</td><td class="num">${s ? s.min.toFixed(3) : "–"}</td><td class="num">${s ? s.mean.toFixed(3) : "–"}</td><td class="num">${s ? s.max.toFixed(3) : "–"}</td></tr>`;
  const canApply = !isReadOnly() && r.suggested != null && Math.abs(r.suggested - r.current) > 0.005;
  const actions = [{ label: "关闭" }];
  if (canApply) actions.push({
    label: `采用建议值 ${r.suggested.toFixed(2)}`, kind: "primary",
    run: async (h) => {
      try {
        const res = await api("/api/config", { method: "PUT", json: { speakers: { identify_threshold: Number(r.suggested.toFixed(2)) } } });
        toast("识别阈值已更新", { kind: "ok" });
        emit("config:saved", { saved: (res && res.saved) || [] });
        emit("speakers:changed", { reason: "threshold" });
        h.close(true);
      } catch (e) { toastErr(e); }
    },
  });
  return openModal({
    title: "识别阈值校准", size: "md", actions,
    body: html`<p class="help">两段声音的相似度达到阈值才认成同一个人。下面是现有样本之间的相似度：同一人的应该高，不同人的应该低，阈值取在两者之间。</p>
      <div class="tbl-wrap"><table class="tbl"><thead><tr><th>类型</th><th class="num">对数</th><th class="num">最小</th><th class="num">平均</th><th class="num">最大</th></tr></thead>
        <tbody>${row("同一人", r.same)}${row("不同人", r.cross)}</tbody></table></div>
      <p>${r.n_embeddings} 个样本 / ${r.n_speakers} 人 · 当前阈值 <strong>${r.current}</strong>${when(r.suggested != null, () => html` · 建议 <strong>${r.suggested.toFixed(2)}</strong>`)}</p>
      ${when(!r.same, () => notice({ kind: "warn", compact: true, body: "没有同一人的多个样本，无法估计同人分数；请每人至少 2 个样本。" }))}
      ${when(!r.cross, () => notice({ kind: "warn", compact: true, body: "只有一个人，无法估计异人分数。" }))}
      ${when(r.overlap, () => notice({ kind: "warn", compact: true, body: "同人与异人分数区间重叠，会有误判；建议增加样本或检查录音质量。" }))}`,
  });
}
