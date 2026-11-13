// 核对: the sentences automatically recognised as one person, least certain first
// (GET /api/speakers/{id}/review). Each is answered with 是 TA (POST /api/utterances/{id}/confirm) or 不是 TA
// (POST /api/utterances/{id}/reject). Answered rows fade (a 不是 TA also marks every other row the veto took out of
// the queue); 「再来 20 句」 re-reads the queue from the top and skips what this dialog already showed, so sentences
// that left the queue meanwhile cannot shift the window. Resolves true when anything was answered.
import { html, when } from "../../core/html.js";
import { $$, render, insert } from "../../core/dom.js";
import { api } from "../../core/api.js";
import { fmtDb } from "../../core/format.js";
import { num } from "../../core/util.js";
import { minDbfs } from "../../core/store.js";
import { emptyState, iconBtn, notice, spinner } from "../../ui/components.js";
import { sysBadge } from "../../ui/speaker.js";
import { clips, player } from "../../ui/player.js";
import { openModal } from "../../ui/modal.js";
import { toast, toastErr } from "../../ui/toast.js";

const PAGE = 20, MAX = 200;      // the server caps limit at 200

export function openReview(spk) {
  const name = spk.name;
  let total = 0, items = [], answered = 0, changed = false, loading = false, hasMore = false;
  const shown = new Set();       // utterance ids already shown in this dialog session

  const row = (it) => {
    const db = num(it.loudness_db), thr = num(minDbfs());
    const far = db != null && thr != null && db < thr;
    const score = num(it.score) != null ? Number(it.score).toFixed(2) : "–";
    const uid = Number(it.utterance_id);
    return html`<li class="rv-row ${far ? "far" : ""}" data-rid="${uid}" data-utt="${uid}">
      <span class="row-time">${String(it.day || "").slice(5)} ${String(it.time || "").slice(0, 5)}</span>
      <span class="rv-text">${it.text || "（没有文字）"}</span>
      <span class="rv-meta"><span title="和 ${name} 的样本的相似度">${score}</span>${when(db != null, () => html`<span title="音量">${fmtDb(db)}</span>`)}${sysBadge(it.system_share)}<span>${Number(it.seconds || 0).toFixed(1)}s</span></span>
      <span class="rv-btns">${it.audio ? iconBtn({ icon: "play", label: "试听", attrs: { "data-rplay": "" }, size: 15 }) : html`<span class="icon-btn" title="音频已删除，无法播放" aria-hidden="true"></span>`}
        ${iconBtn({ icon: "check", label: `是 ${name}`, attrs: { "data-ryes": "" }, kind: "yes", size: 16 })}${iconBtn({ icon: "x", label: `不是 ${name}（标为负样本）`, attrs: { "data-rno": "" }, kind: "warn", size: 16 })}</span>
    </li>`;
  };

  return openModal({
    title: `核对 ${name}`, size: "xl",
    body: html`<p class="help">自动认成 ${name} 的句子，最没把握的在前。听一下：是 TA 就点 ✓（以后不会再被自动改掉，够长、够清楚的句子还会补充声纹样本），不是就点 ✕（会成为 ${name} 的负样本，识别马上用上）。不确定的可以跳过。</p>
      <label class="check"><input type="checkbox" id="rv-whole"> 点 ✕ 时，这段录音里同一个声音的其他句子也一起</label>
      <ul class="rv-list" id="rv-list">${spinner("加载中…")}</ul>
      <div class="small t3" id="rv-count" aria-live="polite"></div>`,
    actions: [{ label: `再来 ${PAGE} 句`, disabled: true, run: () => {} }, { label: "完成", kind: "primary", run: (h) => h.close(changed) }],
    onMount: (h) => {
      const list = h.body.querySelector("#rv-list"), count = h.body.querySelector("#rv-count"), more = h.actionButton(0);
      const updateCount = () => {
        const left = Math.max(0, total - answered);
        count.textContent = total ? `还有约 ${left} 句待核对` : "";
        more.disabled = loading || !hasMore;
      };
      const load = async () => {
        loading = true;
        updateCount();
        const before = shown.size;
        render(list, spinner("加载中…"));
        try {
          // always from offset 0: ask for enough rows to get past the ones already shown, then drop those here
          const limit = Math.min(MAX, PAGE + shown.size);
          const r = await api(`/api/speakers/${spk.id}/review`, { query: { limit, offset: 0 } });
          if (h.closed) return;
          const got = Array.isArray(r && r.items) ? r.items : [];
          const fresh = got.filter((it) => !shown.has(Number(it.utterance_id)));
          items = fresh.slice(0, PAGE);
          items.forEach((it) => shown.add(Number(it.utterance_id)));
          total = Number(r && r.total) || 0;
          answered = 0;
          // more unseen rows: some were left over in this answer, or the answer was full and the queue is longer
          hasMore = fresh.length > items.length || (got.length >= limit && total > got.length);
          render(list, items.length ? html`${items.map(row)}` : emptyState({ compact: true, icon: "check", title: "没有需要核对的句子", text: before ? "已经看完了。" : `自动认成 ${name} 的句子都核对过了（或太短 / 音频已删除）。` }));
          player.mark(list);
        } catch (e) { items = []; hasMore = false; if (!h.closed) render(list, notice({ kind: "danger", compact: true, body: e.message || String(e) })); }
        loading = false;
        updateCount();
      };
      // n = how many sentences this answer took out of the queue (for the 「还有约 N 句」 count)
      const finish = (rowEl, klass, text, n = 1) => {
        if (rowEl.classList.contains("done")) return;      // already taken out by another ✕ (whole cluster)
        rowEl.classList.add("done", klass);
        render(rowEl.querySelector(".rv-btns"), html`<span class="rv-res ${klass}">${text}</span>`);
        answered += n;
        changed = true;
        updateCount();
      };
      list.addEventListener("click", async (ev) => {
        const rowEl = ev.target.closest(".rv-row");
        if (!rowEl || rowEl.classList.contains("done")) return;
        const uid = Number(rowEl.dataset.rid);
        if (ev.target.closest("[data-rplay]")) {
          const it = items.find((x) => Number(x.utterance_id) === uid) || {};
          player.playOrToggle(clips.utterance({ id: uid, speaker: name, time: it.time, text: it.text }, { pad: 0.3 }));
          return;
        }
        const yes = ev.target.closest("[data-ryes]"), no = ev.target.closest("[data-rno]");
        if (!yes && !no) return;
        const btns = $$("button", rowEl);
        btns.forEach((b) => { b.disabled = true; });
        changed = true;                                    // even if the dialog is closed before the answer arrives
        if (player.isCurrent(`utt:${uid}`)) player.stop();
        try {
          if (yes) {
            const r = await api(`/api/utterances/${uid}/confirm`, { method: "POST", json: { speaker_id: spk.id } });
            finish(rowEl, "yes", `是 ${name}${r && r.sample_added ? "，已加入声纹样本" : ""}`);
          } else {
            const whole = h.body.querySelector("#rv-whole").checked;
            const r = await api(`/api/utterances/${uid}/reject`, { method: "POST", json: { speaker_id: spk.id, whole_cluster: whole } });
            const k = Number(r && r.negatives_added) || 0, n = Number(r && r.rejected) || 0;
            // every sentence the veto took out (older servers do not send the list: just the clicked one)
            const ids = new Set(Array.isArray(r && r.utterance_ids) ? r.utterance_ids.map(Number).filter(Number.isFinite) : []);
            ids.add(uid);
            finish(rowEl, "no", `不是 ${name}${n > 1 ? `（连同 ${n - 1} 句）` : ""}${k ? `，${k} 个负样本` : ""}`, ids.size);
            for (const x of $$(".rv-row", list)) if (x !== rowEl && ids.has(Number(x.dataset.rid))) finish(x, "no", `不是 ${name}（同一段里的这个声音一起标了）`, 0);
            if (r && r.warning) { insert(rowEl, "beforeend", html`<div class="rv-warn">${notice({ kind: "warn", compact: true, body: r.warning })}</div>`); toast(r.warning, { kind: "warn", ms: 9000 }); }
            updateCount();
          }
        } catch (e) { btns.forEach((b) => { b.disabled = false; }); toastErr(e); }
      });
      more.addEventListener("click", () => { if (!more.disabled) load(); });
      load();
    },
    onClose: () => { player.stop(); },
  }).then(() => changed);
}
