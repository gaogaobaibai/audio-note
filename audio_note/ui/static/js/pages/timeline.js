// 记录: one day of conversation. Sentences are grouped into sessions (a gap of 5 minutes starts a new one — derived
// here until the server sends its own sessions) and, inside a session, into speaker turn blocks.
// Mutations and live events re-read the day and patch only the sessions whose markup changed, keeping the reader's
// scroll position; nothing here rebuilds the page on a timer.
//   #/transcript                      newest day
//   #/transcript/2026-11-10           that day
//   #/transcript/2026-11-10?u=1234    … scrolled to sentence 1234 (element id "u1234")
import { html, when } from "../core/html.js";
import { $, $$, render, patch, on, append, replace, keepScroll } from "../core/dom.js";
import { api, isAbort } from "../core/api.js";
import { loadDays, setMinDbfs, minDbfs, ensureMinDbfs, isReadOnly, status } from "../core/store.js";
import { ensureStatus, runJob } from "../core/status.js";
import { hrefFor } from "../core/router.js";
import { dayLabel, dayTitle, fmtDb, fmtDur, fmtMin, fmtTime, isDay, secOfDay, today, tsMs } from "../core/format.js";
import { debounce, num } from "../core/util.js";
import { icon } from "../ui/icons.js";
import { badge, button, chip, dots, emptyState, iconBtn, notice, skeleton, spinner } from "../ui/components.js";
import { dayStrip } from "../ui/charts.js";
import { avatar, colorClass, rememberSpeakers } from "../ui/speaker.js";
import { actionBar, clipsOf, groupTurns, isFar, labelUtt, rejectUtt, showContext, splitUtt, turnBlock } from "../ui/utterance.js";
import { player } from "../ui/player.js";
import { confirm, openDrawer } from "../ui/modal.js";
import { openMenu } from "../ui/popover.js";
import { toastErr } from "../ui/toast.js";

export const SESSION_GAP_S = 300;

/** Group a day's sentences into sessions: by the server's session_id when every sentence has one, else by gaps. */
export function sessionsOf(utts, gapS = SESSION_GAP_S) {
  const out = [];
  const bySid = utts.length > 0 && utts.every((u) => u.session_id != null);
  let cur = null, lastEnd = NaN, lastSid = null;
  for (const u of utts) {
    const t = tsMs(u.abs_start);
    const split = bySid ? u.session_id !== lastSid : !(Number.isFinite(t) && Number.isFinite(lastEnd)) ? cur === null : t - lastEnd > gapS * 1000;
    if (!cur || split) { cur = { utts: [] }; out.push(cur); }
    cur.utts.push(u);
    lastSid = u.session_id;
    const end = Number.isFinite(t) ? t + Math.max(0, (Number(u.end_s) - Number(u.start_s)) || 0) * 1000 : NaN;
    if (Number.isFinite(end)) lastEnd = Number.isFinite(lastEnd) ? Math.max(lastEnd, end) : end;
  }
  return out.map((s, i) => {
    const a = s.utts[0], b = s.utts[s.utts.length - 1];
    const t0 = tsMs(a.abs_start), t1 = tsMs(b.abs_start) + Math.max(0, (Number(b.end_s) - Number(b.start_s)) || 0) * 1000;
    return { index: i + 1, key: String(a.id), utts: s.utts, start: String(a.time || "").slice(0, 5), end: String(b.time || "").slice(0, 5), seconds: Number.isFinite(t1 - t0) ? Math.max(0, (t1 - t0) / 1000) : 0 };
  });
}

export default {
  async mount(ctx) {
    await ensureStatus();
    let days = await loadDays({ force: true });
    await ensureMinDbfs();
    if (!ctx.alive()) return;

    let day = "", data = null, utts = [], byId = new Map(), sessions = [];
    let filter = "";                         // speaker name, "" = everyone
    let hlIds = new Set();                   // sentences to highlight on the next render (deep link, pieces of a split)
    let loadGen = 0;
    const sessMarkup = new Map();            // session key -> markup of the last render (to patch only what changed)

    render(ctx.el, html`<div class="page wide tl">
      <div class="tl-layout">
        <aside class="tl-side" aria-label="日期和搜索">
          <form class="tl-search" id="tl-search" role="search"><input class="input" name="q" type="search" placeholder="搜索全部记录" aria-label="搜索全部记录" autocomplete="off">${iconBtn({ icon: "search", label: "搜索", attrs: { type: "submit" } })}</form>
          <nav class="tl-days" id="tl-days" aria-label="有记录的日期"></nav>
        </aside>
        <div class="tl-main">
          <header class="tl-head" id="tl-head"></header>
          <div id="tl-strip"></div>
          <div class="chips" id="tl-filters"></div>
          <div class="tl-body" id="tl-body">${skeleton("rows", 8)}</div>
          <div id="tl-failed"></div>
        </div>
      </div>
    </div>`);
    const body = $("#tl-body");

    const pickDay = () => { const p = ctx.params[0]; return isDay(p) ? p : (days[0] ? days[0].day : today()); };
    const dayInfo = () => days.find((d) => d.day === day) || { pending: 0, failed: 0, utterances: 0 };

    // ---- day list
    const paintDays = () => {
      const max = Math.max(1, ...days.map((d) => Number(d.speech_min) || 0));
      patch($("#tl-days"), days.length ? html`${days.map((d) => html`<a class="day-item" href="${hrefFor("transcript", d.day)}" ${d.day === day ? html`aria-current="true"` : ""}>
          <span class="day-top"><span class="day-date">${dayTitle(d.day)}</span><span class="day-week">${dayLabel(d.day)}</span></span>
          <span class="day-sub">${d.utterances ? `${d.utterances} 句 · ${d.first}–${d.last}` : "还没有转写"}</span>
          ${when(d.pending || d.failed, () => html`<span class="day-flags">${d.pending ? badge(`${d.pending} 段待转写`, "warn") : ""}${d.failed ? badge(`${d.failed} 段失败`, "danger") : ""}</span>`)}
          <span class="day-bar" aria-hidden="true"><i style="width:${Math.max(d.utterances ? 3 : 0, ((Number(d.speech_min) || 0) / max) * 100).toFixed(1)}%"></i></span>
        </a>`)}` : html`<div class="t3 small tl-nodays">还没有任何记录</div>`);
      const cur = $("#tl-days [aria-current]");
      if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
    };

    // ---- header, strip, filters
    const paintHead = () => {
      const info = dayInfo(), ro = isReadOnly();
      const segs = (data && data.segments) || [];
      const thr = num(minDbfs());
      const farN = thr == null ? 0 : utts.filter(isFar).length;
      const i = days.findIndex((d) => d.day === day);
      const newer = i > 0 ? days[i - 1] : null, older = i >= 0 && i < days.length - 1 ? days[i + 1] : null;
      const meta = dots(
        segs.length ? `${segs.length} 段录音 · ${fmtMin(segs.reduce((a, s) => a + (s.duration_s || 0), 0) / 60)}` : "",
        utts.length ? `${utts.length} 句` : "",
        sessions.length > 1 ? `${sessions.length} 段对话` : "",
        farN ? html`<span tabindex="0" data-tip="音量低于设置里的最低音量 ${fmtDb(thr)}，显示为淡色；新录音里这样的声音会直接滤掉">${farN} 句偏远</span>` : "",
      );
      patch($("#tl-head"), html`<div class="tl-title">
          <div class="tl-nav">${older ? html`<a class="icon-btn" href="${hrefFor("transcript", older.day)}" aria-label="前一天 ${older.day}" title="前一天">${icon("chevron-left", 18)}</a>` : html`<span class="icon-btn" aria-hidden="true"></span>`}
            <h2>${dayTitle(day)} <span class="tl-week">${dayLabel(day)}</span></h2>
            ${newer ? html`<a class="icon-btn" href="${hrefFor("transcript", newer.day)}" aria-label="后一天 ${newer.day}" title="后一天">${icon("chevron-right", 18)}</a>` : ""}</div>
          <div class="tl-meta">${meta}</div></div>
        <div class="tl-acts">
          ${when(info.pending && !ro, () => button({ label: "转写这天的录音", kind: "primary", size: "sm", attrs: { "data-act": "process" } }))}
          ${when(info.failed && !ro, () => button({ label: "重试失败的段", size: "sm", attrs: { "data-act": "retry" } }))}
          <a class="btn sm ghost" href="${hrefFor("daily", day)}">${icon("calendar", 14)}<span>日报</span></a>
          ${when(!ro, () => iconBtn({ icon: "more", label: "更多操作", attrs: { "data-act": "more" } }))}
        </div>`);
    };
    const paintStrip = () => {
      const spans = utts.map((u) => ({ start: secOfDay(u.abs_start), dur: Math.max(0.5, (Number(u.end_s) - Number(u.start_s)) || 0), cls: colorClass(u), name: u.speaker }));
      patch($("#tl-strip"), utts.length > 3 ? html`<div class="tl-strip" title="这一天谁在什么时候说话（点一下跳到那个时间）">${dayStrip(spans)}</div>` : html``);
    };
    const paintFilters = () => {
      const counts = Object.entries((data && data.speakers) || {}).sort((a, b) => b[1] - a[1]);
      patch($("#tl-filters"), counts.length > 1 ? html`${counts.map(([name, n]) => chip({ label: name, on: filter === name, count: n, attrs: { "data-spk": name }, lead: html`<span class="chip-dot ${colorClass(name)}"></span>` }))}
        ${when(filter, () => html`<button type="button" class="link-btn small" data-act="clear-spk">清除筛选</button>`)}` : html``);
    };

    // ---- sessions
    const sessionHtml = (s) => {
      const shown = filter ? s.utts.filter((u) => u.speaker === filter) : s.utts;
      const people = [];
      for (const u of s.utts) { const k = u.speaker_id != null ? `s${u.speaker_id}` : "unk"; if (!people.some((p) => p.k === k)) people.push({ k, u }); }
      const turns = groupTurns(shown);
      const hasAudio = shown.some((u) => u.audio);
      return html`<section class="sess" data-sess="${s.key}" aria-label="${s.start} 到 ${s.end} 的对话">
        <header class="sess-head">
          <span class="sess-time">${s.start}${s.end !== s.start ? ` – ${s.end}` : ""}</span>
          <span class="sess-meta">${s.seconds >= 60 ? `${fmtDur(s.seconds)} · ` : ""}${shown.length} 句</span>
          <span class="avatars" aria-hidden="true">${people.slice(0, 5).map((p) => avatar(p.u, 22))}</span>
          ${when(hasAudio, () => html`<button type="button" class="btn ghost sm sess-play" data-act="play-sess" title="从头连续播放这一段">${icon("play", 13)}<span>播放本段</span></button>`)}
        </header>
        ${turns.map((t, i) => turnBlock(t, { hlIds, firstFocusable: i === 0, acts: false }))}
      </section>`;
    };
    const visibleSessions = () => (filter ? sessions.filter((s) => s.utts.some((u) => u.speaker === filter)) : sessions);
    const paintBody = () => {
      const list = visibleSessions();
      if (!list.length) {
        sessMarkup.clear();
        const info = dayInfo();
        render(body, html`<div class="card">${emptyState({
          icon: "doc",
          title: data && data.segments.length ? "这天的录音还没有转写" : days.length ? "这天没有记录" : "还没有任何记录",
          text: info.pending ? `${info.pending} 段录音等待转写。` : days.length ? "" : "开始录音后，说过的话会按天出现在这里。",
          action: days.length ? "" : html`<a class="btn primary" href="#/dashboard">去总览开始录音</a>`,
        })}</div>`);
        return;
      }
      const next = new Map(list.map((s) => [s.key, sessionHtml(s)]));
      const existing = $$(":scope > .sess", body);
      const sameKeys = existing.length === list.length && existing.every((el, i) => el.dataset.sess === list[i].key);
      if (sameKeys) {
        existing.forEach((el, i) => { const k = list[i].key; if (sessMarkup.get(k) !== next.get(k).value) replace(el, next.get(k)); });
      } else {
        render(body, html`${list.map((s) => next.get(s.key))}`);
      }
      sessMarkup.clear();
      for (const [k, v] of next) sessMarkup.set(k, v.value);
      player.mark(body);
    };

    const index = () => {
      utts = ((data && data.sessions) || []).flatMap((s) => s.utterances);
      byId = new Map(utts.map((u) => [u.id, u]));
      rememberSpeakers(utts, { idKey: "speaker_id", nameKey: "speaker" });
      sessions = sessionsOf(utts);
      if (filter && !utts.some((u) => u.speaker === filter)) filter = "";
    };
    const paintFailed = () => {
      const failed = ((data && data.segments) || []).filter((s) => s.status === "failed");
      patch($("#tl-failed"), failed.length ? html`<details class="disclosure"><summary><span>${failed.length} 段录音转写失败</span></summary>
        <div class="tbl-wrap tl-failed"><table class="tbl"><thead><tr><th>编号</th><th>时间</th><th class="num">时长</th><th>原因</th></tr></thead><tbody>
        ${failed.map((s) => html`<tr><td>#${s.id}</td><td>${fmtTime(s.started_at)}</td><td class="num">${Math.round(s.duration_s || 0)} 秒</td><td class="t2">${s.error || ""}</td></tr>`)}</tbody></table></div></details>` : html``);
    };
    const paintAll = () => { paintHead(); paintStrip(); paintFilters(); paintBody(); paintFailed(); };

    const focusUtt = (id, { flash = true } = {}) => {
      const el = document.getElementById(`u${id}`);
      if (!el) return false;
      el.scrollIntoView({ block: "center" });
      if (flash) { el.classList.remove("hl"); void el.offsetWidth; el.classList.add("hl"); }
      return true;
    };

    /** (Re)load the current day. `keep`: patch in place and hold the scroll position (after a mutation / live event). */
    const loadDay = async ({ keep = false } = {}) => {
      const g = ++loadGen;
      if (!keep) { sessMarkup.clear(); render(body, skeleton("rows", 8)); }
      let fresh;
      try { fresh = await api(`/api/transcript/${day}`, { signal: ctx.signal }); }
      catch (e) { if (isAbort(e) || g !== loadGen) return; if (!keep) render(body, notice({ kind: "danger", body: e.message || String(e), actions: button({ label: "重试", size: "sm", attrs: { "data-act": "reload" } }) })); return; }
      if (g !== loadGen || !ctx.alive()) return;
      data = fresh;
      if (data.min_speech_dbfs !== undefined) setMinDbfs(data.min_speech_dbfs);
      index();
      if (keep) await keepScroll(() => paintAll()); else paintAll();
      hlIds = new Set();
    };
    const reloadDays = async () => { try { days = await loadDays({ force: true }); } catch (e) { /* keep the old list */ } if (ctx.alive()) { paintDays(); paintHead(); } };
    const refreshSoon = debounce(() => { loadDay({ keep: true }); reloadDays(); }, 350);

    const applyRoute = async () => {
      const want = pickDay();
      const u = Number(ctx.query.get("u")) || 0;
      if (want !== day) {
        day = want;
        filter = "";
        hlIds = new Set(u ? [u] : []);
        ctx.setTitle(`记录 · ${dayTitle(day)}`);
        paintDays();
        paintHead();
        window.scrollTo(0, 0);
        await loadDay();
        if (u) focusUtt(u, { flash: false });
      } else if (u) {
        if (filter && byId.has(u) && byId.get(u).speaker !== filter) { filter = ""; paintFilters(); paintBody(); }
        focusUtt(u);
      }
    };

    // ---- interactions
    const uttOf = (el) => { const row = el.closest(".utt"); const id = Number(row ? row.dataset.utt : el.dataset.for); return byId.get(id) || null; };
    const sessionOf = (u) => sessions.find((s) => s.utts.includes(u)) || null;
    const ensureActs = (row) => {
      if (!row || row.querySelector(".utt-acts")) return;
      const u = byId.get(Number(row.dataset.utt));
      if (u) append(row, actionBar(u, { acts: isReadOnly() ? ["play", "ctx"] : undefined }));
    };
    const select = (row, on = true) => { for (const el of $$(".utt.sel", body)) if (el !== row) el.classList.remove("sel"); if (row) { ensureActs(row); row.classList.toggle("sel", on); } };
    const playFrom = (u) => {
      const s = sessionOf(u);
      const list = (s ? (filter ? s.utts.filter((x) => x.speaker === filter) : s.utts) : [u]).filter((x) => x.audio);
      const i = list.indexOf(u);
      if (i < 0) return;
      player.playOrToggle(clipsOf(list), { index: i });
    };
    const act = (name, u) => {
      if (!u) return;
      if (name === "play") { if (u.audio) playFrom(u); }
      else if (name === "label") labelUtt(u).catch(toastErr);
      else if (name === "reject") rejectUtt(u);
      else if (name === "split") { if (u.audio) splitUtt(u); }
      else if (name === "ctx") showContext(u.id);
    };

    on(body, "mouseover", ".utt", (e, row) => ensureActs(row), { signal: ctx.signal });
    on(body, "focusin", ".utt", (e, row) => {
      ensureActs(row);
      const sess = row.closest(".sess");
      if (sess && e.target === row) { for (const el of $$('.utt[tabindex="0"]', sess)) if (el !== row) el.tabIndex = -1; row.tabIndex = 0; }
    }, { signal: ctx.signal });
    on(body, "click", ".utt, [data-u], [data-act]", (e, el) => {
      const ub = e.target.closest("[data-u]");
      if (ub) { e.preventDefault(); act(ub.dataset.u, uttOf(ub)); return; }
      const ab = e.target.closest("[data-act]");
      if (ab) {
        if (ab.dataset.act === "play-sess") {
          const s = sessions.find((x) => x.key === ab.closest(".sess").dataset.sess);
          if (s) player.play(clipsOf(filter ? s.utts.filter((x) => x.speaker === filter) : s.utts));
        }
        return;
      }
      if (e.target.closest("a, button, .mark-fix, .mark-ask, .badge")) return;
      if (window.getSelection && String(window.getSelection())) return;       // the user is selecting text, not picking the row
      const row = e.target.closest(".utt");
      if (row) select(row, !row.classList.contains("sel"));
    }, { signal: ctx.signal });
    on(body, "keydown", ".utt", (e, row) => {
      if (e.target !== row || e.ctrlKey || e.metaKey || e.altKey) return;
      const all = $$(".utt", body), i = all.indexOf(row);
      const k = e.key.toLowerCase();
      if (k === "arrowdown" || k === "arrowup") {
        const next = all[i + (k === "arrowdown" ? 1 : -1)];
        if (next) { e.preventDefault(); next.focus(); next.scrollIntoView({ block: "nearest" }); }
        return;
      }
      const u = byId.get(Number(row.dataset.utt));
      const map = { enter: "play", " ": "play", l: "label", x: "reject", s: "split", c: "ctx" };
      if (k === "escape") { row.classList.remove("sel"); row.blur(); return; }
      if (!map[k] || !u) return;
      if (isReadOnly() && ["label", "reject", "split"].includes(map[k])) return;
      if (map[k] === "reject" && u.speaker_id == null) return;
      e.preventDefault();
      act(map[k], u);
    }, { signal: ctx.signal });

    on($("#tl-head"), "click", "[data-act]", async (e, b) => {
      const a = b.dataset.act;
      if (a === "process") runJob("process");
      else if (a === "retry") runJob("process", { retry_failed: true });
      else if (a === "more") {
        const pick = await openMenu(b, [
          { id: "relabel", label: "重新识别这天的说话人", icon: "refresh", hint: "用现在的声纹重新认一遍，不重新转写" },
          { id: "top", label: "回到这天开头", icon: "chevron-up" },
        ]);
        if (pick === "relabel") { if (await confirm({ title: "重新识别说话人", text: `用当前登记的声纹重新识别 ${day} 的所有语句（不重新转写，不花钱）。手动标过的句子不会改。`, okLabel: "重新识别" })) runJob("relabel", { day }); }
        else if (pick === "top") window.scrollTo({ top: 0 });
      }
    }, { signal: ctx.signal });
    on(ctx.el, "click", '[data-act="reload"]', () => loadDay(), { signal: ctx.signal });
    on($("#tl-filters"), "click", "[data-spk], [data-act]", (e, b) => {
      filter = b.dataset.spk !== undefined && filter !== b.dataset.spk ? b.dataset.spk : "";
      paintFilters();
      keepScroll(() => paintBody());
    }, { signal: ctx.signal });
    on($("#tl-strip"), "click", ".strip-b", (e, g) => {
      const t = g.dataset.t;
      const u = (filter ? utts.filter((x) => x.speaker === filter) : utts).find((x) => String(x.time || "").slice(0, 5) >= t);
      if (u) focusUtt(u.id);
    }, { signal: ctx.signal });

    // ---- search (results in a drawer; the day underneath stays where it was)
    on($("#tl-search"), "submit", null, (e) => {
      e.preventDefault();
      const q = e.currentTarget.elements.q.value.trim();
      if (!q) return;
      const d = openDrawer({
        title: `搜索「${q}」`, body: spinner("检索中…"),
        onMount: async (h) => {
          try {
            const r = await api("/api/search", { query: { q, limit: 30 }, signal: ctx.signal });
            if (h.closed) return;
            render(h.body, html`<div class="small t3">${dots(r.how || "", `${r.hits.length} 条`, r.expansions && r.expansions.length ? `扩展词：${r.expansions.join("、")}` : "")}</div>
              ${r.hits.length ? html`<div class="hits">${r.hits.map((x) => html`<a class="hit" href="${hrefFor("transcript", x.day, { u: x.id })}" data-hit>
                <span class="hit-when">${x.day.slice(5)} ${x.time}</span><span class="spk-name ${colorClass({ speaker: x.speaker })}">${x.speaker}</span><span class="hit-text">${x.text}</span></a>`)}</div>`
                : emptyState({ compact: true, icon: "search", title: "没有找到", text: "换个说法试试，或者用一句话描述你想找的内容。" })}`);
            h.body.addEventListener("click", (ev) => { if (ev.target.closest("[data-hit]")) h.close(); });
          } catch (err) { if (!isAbort(err) && !h.closed) render(h.body, notice({ kind: "danger", body: err.message || String(err) })); }
        },
      });
      return d;
    }, { signal: ctx.signal });

    // ---- live
    ctx.on("utterances:changed", (ev) => {
      if (ev && ev.day && ev.day !== day) { reloadDays(); return; }
      if (ev && ev.reason === "split" && Array.isArray(ev.ids)) hlIds = new Set(ev.ids);
      loadDay({ keep: true });
      reloadDays();
    });
    ctx.on("live:utterances", (ev) => { if (!ev || !ev.day || ev.day === day) refreshSoon(); else reloadDays(); });
    ctx.on("live:utterance", () => refreshSoon());
    ctx.on("live:segment", debounce(() => reloadDays(), 1500));
    ctx.onJob(["process", "relabel", "voices", "rescore", "recorrect", "refine", "chain"], () => refreshSoon());
    ctx.on("resync", () => refreshSoon());
    ctx.on("online", () => refreshSoon());
    ctx.on("status", (st, prev) => { if (prev && (prev.self_name !== st.self_name || (prev.service && st.service && prev.service.read_only !== st.service.read_only))) { sessMarkup.clear(); paintAll(); } });

    ctx.apply = applyRoute;
    await applyRoute();
    if (!status()) paintHead();
  },

  async update(ctx) { if (ctx.apply) await ctx.apply(); },
};
