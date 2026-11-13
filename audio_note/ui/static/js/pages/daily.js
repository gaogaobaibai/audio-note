// 日报: the daily digest as a reading page, plus the (legacy) memory notes list as a second tab.
//   #/daily              newest digest        #/daily/2026-11-10   that day        #/daily/notes   记忆笔记
import { html, raw, when } from "../core/html.js";
import { $, render, patch, on } from "../core/dom.js";
import { api, isAbort } from "../core/api.js";
import { md } from "../core/md.js";
import { status, loadDays, isReadOnly, hasApiKey } from "../core/store.js";
import { ensureStatus, runJob, jobRunning } from "../core/status.js";
import { hrefFor } from "../core/router.js";
import { dayLabel, dayTitle, fmtDT, isDay, today } from "../core/format.js";
import { icon } from "../ui/icons.js";
import { badge, button, chip, dots, emptyState, iconBtn, notice, segmented, skeleton, spinner, tabs } from "../ui/components.js";
import { colorClass } from "../ui/speaker.js";
import { confirm } from "../ui/modal.js";
import { toastErr } from "../ui/toast.js";

const KINDS = [["", "全部"], ["todo", "待办"], ["decision", "决定"], ["person", "人物"], ["topic", "话题"], ["fact", "事实"]];
const KIND_LABEL = Object.fromEntries(KINDS.filter(([k]) => k));

const textOf = (x) => (typeof x === "string" ? x : (x && (x.text || x.name || x.title)) || JSON.stringify(x));
const listOf = (items) => (items && items.length ? html`<ul class="dg-list">${items.map((x) => html`<li>${textOf(x)}</li>`)}</ul>` : html`<p class="t3">（无）</p>`);
const section = (title, ic, inner) => html`<section class="dg-sec"><h3>${icon(ic, 16)}<span>${title}</span></h3>${inner}</section>`;

function digestBody(j) {
  const stats = dots(j.utterance_count != null ? `${j.utterance_count} 条记录` : "", j.session_count ? `${j.session_count} 个时段` : "", j.time_range || "", j.speakers && j.speakers.length ? `说话人：${j.speakers.join("、")}` : "");
  const todos = (j.todos || []).map((t) => html`<li><span class="dg-todo-box" aria-hidden="true"></span><span>${textOf(t)}${when(t && t.owner, () => html` <span class="dg-tag">负责人：${t.owner}</span>`)}${when(t && t.due, () => html` <span class="dg-tag">截止：${t.due}</span>`)}</span></li>`);
  const people = (j.people || []).map((p) => html`<li><span class="spk-name ${colorClass(String((p && p.name) || ""))}">${(p && p.name) || textOf(p)}</span><span class="t2">${(p && p.context) || ""}</span></li>`);
  const tl = (j.timeline || []).map((t) => html`<li><span class="dg-time">${(t && t.time) || ""}</span><span>${(t && t.event) || textOf(t)}</span></li>`);
  return html`<div class="digest">
    ${when(stats.value, () => html`<p class="dg-stats">${stats}</p>`)}
    ${section("概览", "doc", html`<p class="dg-overview">${raw(String(j.overview || "（无）").split("\n").map((ln) => html`${ln}`.value).join("<br>"))}</p>`)}
    <div class="dg-cols">${section("主要话题", "list", listOf(j.topics))}${section("决定", "check", listOf(j.decisions))}</div>
    <div class="dg-cols">${section("待办", "target", todos.length ? html`<ul class="dg-todos">${todos}</ul>` : html`<p class="t3">（无）</p>`)}${section("提到的人", "people", people.length ? html`<ul class="dg-people">${people}</ul>` : html`<p class="t3">（无）</p>`)}</div>
    <div class="dg-cols">${section("情绪", "activity", html`<p>${j.mood || "（无）"}</p>`)}${section("未闭环", "help", listOf(j.open_loops))}</div>
    ${when(tl.length, () => section("时间线", "clock", html`<ol class="dg-timeline">${tl}</ol>`))}
  </div>`;
}

export default {
  async mount(ctx) {
    await ensureStatus();
    if (!ctx.alive()) return;
    let list = [], days = [], all = [];
    let tab = "digest", day = "", mdView = false, kind = "", digest = null, gen = 0;

    render(ctx.el, html`<div class="page wide daily">
      <div id="dl-tabs"></div>
      <div id="dl-body">${skeleton("rows", 6)}</div>
    </div>`);
    const bodyEl = $("#dl-body");

    const loadLists = async () => {
      [list, days] = await Promise.all([api("/api/daily", { signal: ctx.signal }), loadDays()]);
      const known = new Set(list.map((d) => d.day));
      const extra = days.filter((d) => !known.has(d.day) && d.utterances).map((d) => ({ day: d.day, missing: true, now_count: d.utterances }));
      all = [...list, ...extra].sort((a, b) => (a.day < b.day ? 1 : -1));
    };
    const paintTabs = () => patch($("#dl-tabs"), tabs([{ id: "digest", label: "日报" }, { id: "notes", label: "记忆笔记" }], tab, { label: "日报和记忆笔记" }));

    // ---- digest
    const daysHtml = () => (all.length ? html`${all.map((d) => html`<a class="day-item" href="${hrefFor("daily", d.day)}" ${d.day === day ? html`aria-current="true"` : ""}>
        <span class="day-top"><span class="day-date">${dayTitle(d.day)}</span><span class="day-week">${dayLabel(d.day)}</span></span>
        <span class="day-flags">${d.missing ? badge("未生成", "plain") : html`${d.pushed_at ? badge("已推送", "ok") : badge("未推送", "plain")}${d.partial ? badge("不完整", "warn") : ""}${d.stale ? badge("记录有更新", "warn") : ""}`}</span>
        <span class="day-sub day-preview">${d.missing ? `${d.now_count} 条记录，还没有日报` : d.overview || `${d.utterance_count} 条记录`}</span>
      </a>`)}` : html`<div class="t3 small tl-nodays">还没有日报</div>`);

    const viewHtml = () => {
      const r = digest;
      if (!r) return skeleton("text", 8);
      const j = r.json || {};
      const running = jobRunning("daily");
      const ro = isReadOnly();
      const canGen = !ro && hasApiKey();
      const head = html`<header class="dg-head">
        <div><h2>${dayTitle(day)} <span class="tl-week">${dayLabel(day)}</span></h2>
          <p class="tl-meta">${r.exists ? dots(`生成于 ${fmtDT(r.created_at)}`, r.model || "", `${r.utterance_count} 条记录`, r.pushed_at ? `已推送 ${fmtDT(r.pushed_at)}` : "未推送") : `${r.now_count} 条记录，还没有生成日报`}</p></div>
        <div class="tl-acts">
          ${r.exists ? html`${segmented([{ id: "view", label: "阅读" }, { id: "md", label: "Markdown" }], mdView ? "md" : "view", { label: "显示方式" })}
              ${when(!ro, () => button({ label: "推送", icon: "send", size: "sm", attrs: { "data-act": "push" } }))}
              ${when(!ro, () => button({ label: r.stale ? "更新日报" : "重新生成", icon: "refresh", size: "sm", kind: r.stale ? "primary" : "", attrs: { "data-act": "regen" }, disabled: running || !canGen, title: canGen ? "" : "需要先配置 API key" }))}`
            : when(!ro, () => button({ label: "生成日报", icon: "sparkle", kind: "primary", size: "sm", attrs: { "data-act": "regen" }, disabled: running || !r.now_count || !canGen, title: canGen ? "" : "需要先配置 API key" }))}
          <a class="btn sm ghost" href="${hrefFor("transcript", day)}">${icon("doc", 14)}<span>记录</span></a>
        </div></header>`;
      const notes = html`${when(running, () => notice({ kind: "info", compact: true, body: spinner("正在生成日报，大约需要几十秒到几分钟…") }))}
        ${when(r.exists && r.stale, () => notice({ kind: "warn", compact: true, body: `生成之后记录又有更新（现在 ${r.now_count} 条，日报基于 ${r.utterance_count} 条）。` }))}
        ${when(j.partial, () => notice({ kind: "warn", compact: true, body: `这份日报不完整：${(j.failed_blocks || []).length} 个时段分析失败，服务会自动重试。` }))}`;
      let content;
      if (!r.exists) content = emptyState({ icon: "calendar", title: r.now_count ? "这一天还没有日报" : "这天没有记录", text: r.now_count ? (canGen ? "点「生成日报」，它会读一遍当天的记录再写出来。" : "配置 API key 之后才能生成日报。") : "" });
      else if (mdView) content = html`<div class="md">${raw(md(r.markdown))}</div>`;
      else content = digestBody(j);
      return html`${head}${notes}${content}`;
    };
    const paintDigest = () => {
      if (tab !== "digest") return;
      if (!$("#dl-view")) render(bodyEl, html`<div class="tl-layout"><aside class="tl-side"><nav class="tl-days" id="dl-days" aria-label="有日报或记录的日期"></nav></aside><article class="card dg-card" id="dl-view"></article></div>`);
      patch($("#dl-days"), daysHtml());
      patch($("#dl-view"), viewHtml());
    };
    const loadDigest = async () => {
      const g = ++gen;
      try {
        const r = await api(`/api/daily/${day}`, { signal: ctx.signal });
        if (g !== gen || !ctx.alive()) return;
        digest = r;
      } catch (e) { if (isAbort(e) || g !== gen) return; digest = null; if ($("#dl-view")) render($("#dl-view"), notice({ kind: "danger", body: e.message || String(e) })); return; }
      paintDigest();
    };

    // ---- notes
    const paintNotes = async () => {
      const g = ++gen;
      let notes;
      try { notes = await api("/api/notes", { query: { kind }, signal: ctx.signal }); }
      catch (e) { if (isAbort(e) || g !== gen) return; render(bodyEl, notice({ kind: "danger", body: e.message || String(e) })); return; }
      if (g !== gen || tab !== "notes") return;
      render(bodyEl, html`<div class="card">
        <div class="sec-head"><div class="chips" role="group" aria-label="笔记类型">${KINDS.map(([k, l]) => chip({ label: l, on: kind === k, attrs: { "data-kind": k } }))}</div>
          <span class="small t3">${notes.length} 条 · 由每日分析自动整理，问答时会一并参考</span></div>
        ${notes.length ? html`<ul class="notes">${notes.map((n) => html`<li class="note">
            <span class="badge ${n.kind === "todo" ? "warn" : n.kind === "decision" ? "ok" : n.kind === "person" ? "info" : ""}">${KIND_LABEL[n.kind] || n.kind}</span>
            <div class="grow"><div><strong>${n.title}</strong> <span class="t3 xs">${n.day_from || ""}${n.day_to && n.day_to !== n.day_from ? ` ~ ${n.day_to}` : ""}</span></div><div class="t2">${n.content}</div></div>
            ${when(!isReadOnly(), () => iconBtn({ icon: "trash", label: `删除笔记：${n.title}`, attrs: { "data-del": n.id } }))}</li>`)}</ul>`
          : emptyState({ icon: "book", compact: true, title: "还没有笔记", text: "生成日报后，待办、决定和提到的人会整理到这里。" })}
      </div>`);
    };

    const apply = async () => {
      const p = ctx.params[0] || "";
      const wantTab = p === "notes" ? "notes" : "digest";
      const tabChanged = wantTab !== tab || !bodyEl.dataset.ready;
      tab = wantTab;
      bodyEl.dataset.ready = "1";
      paintTabs();
      if (tab === "notes") { ctx.setTitle("日报 · 记忆笔记"); if (tabChanged) render(bodyEl, skeleton("rows", 5)); await paintNotes(); return; }
      const wantDay = isDay(p) ? p : (day || (list[0] ? list[0].day : all[0] ? all[0].day : today()));
      if (tabChanged) render(bodyEl, html``);
      if (wantDay !== day || tabChanged) { day = wantDay; digest = null; mdView = false; }
      ctx.setTitle(`日报 · ${dayTitle(day)}`);
      paintDigest();
      await loadDigest();
    };
    const refresh = async () => { try { await loadLists(); } catch (e) { if (isAbort(e)) return; } if (!ctx.alive()) return; if (tab === "digest") { paintDigest(); await loadDigest(); } };

    on(ctx.el, "click", "[data-tab]", (e, b) => { location.hash = b.dataset.tab === "notes" ? "#/daily/notes" : hrefFor("daily", day || null); }, { signal: ctx.signal });
    on(ctx.el, "click", "[data-seg]", (e, b) => { mdView = b.dataset.seg === "md"; paintDigest(); }, { signal: ctx.signal });
    on(ctx.el, "click", "[data-act]", async (e, b) => {
      const a = b.dataset.act;
      if (a === "regen") { await runJob("daily", { day, force: !!(digest && digest.exists) }); paintDigest(); }
      else if (a === "push") await runJob("push", { day });
    }, { signal: ctx.signal });
    on(ctx.el, "click", "[data-kind]", (e, b) => { kind = b.dataset.kind; paintNotes(); }, { signal: ctx.signal });
    on(ctx.el, "click", "[data-del]", async (e, b) => {
      if (!(await confirm({ title: "删除笔记", text: "删除后问答不再参考这条笔记，以后的日报也不会再生成同一条。", okLabel: "删除", danger: true }))) return;
      try { await api(`/api/notes/${b.dataset.del}`, { method: "DELETE" }); } catch (err) { toastErr(err); }
      paintNotes();
    }, { signal: ctx.signal });

    ctx.onJob(["daily", "push"], refresh);
    ctx.on("live:daily", refresh);
    ctx.on("resync", refresh);
    ctx.on("status", (st, prev) => { const a = prev && prev.jobs && prev.jobs.daily && prev.jobs.daily.running, b = st.jobs && st.jobs.daily && st.jobs.daily.running; if (a !== b) paintDigest(); });

    try { await loadLists(); } catch (e) { if (isAbort(e)) return; throw e; }
    if (!ctx.alive()) return;
    ctx.apply = apply;
    await apply();
    if (!status()) paintDigest();
  },

  async update(ctx) { if (ctx.apply) await ctx.apply(); },
};
