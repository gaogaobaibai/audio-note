// 设置: sectioned form with a section navigator and a search box, advanced options folded away, dirty tracking
// (only what changed is sent), an unsaved-changes guard, and 日志 / 诊断 as a second tab.
//   #/settings            the form        #/settings/<section id>   … scrolled to that section
//   #/settings/logs       日志 / 诊断      (#/logs is an alias)
// GET /api/config -> {path, values, secrets{path: {set, hint}}, defaults, profile_label, base_config_path,
//                     env_api_key, glossary_path, data_dir, models_dir};  PUT /api/config {section: {key: value}}
import { html, when, attrs } from "../core/html.js";
import { $, $$, render, patch, on } from "../core/dom.js";
import { api, isAbort } from "../core/api.js";
import { emit } from "../core/bus.js";
import { loadConfig, setMinDbfs, status } from "../core/store.js";
import { ensureStatus, refreshStatus, runJob, jobOf, jobSummary } from "../core/status.js";
import { hrefFor } from "../core/router.js";
import { deepEqual, getPath, setPath, debounce } from "../core/util.js";
import { fmtDT } from "../core/format.js";
import { icon } from "../ui/icons.js";
import { badge, busy, button, emptyState, kv, notice, skeleton, spinner, tabs } from "../ui/components.js";
import { confirm } from "../ui/modal.js";
import { toast, toastErr } from "../ui/toast.js";
import { SECTIONS, PUSH_CHANNELS, extraFields } from "./settings/schema.js";
import { mountLogs } from "./settings/logs.js";

const fid = (path) => `f-${path.replace(/[^a-zA-Z0-9]+/g, "-")}`;

export default {
  async mount(ctx) {
    await ensureStatus();
    if (!ctx.alive()) return;
    render(ctx.el, html`<div class="page wide settings"><div id="st-tabs"></div><div id="st-body">${skeleton("rows", 6)}</div></div>`);
    const bodyEl = $("#st-body");
    let tab = "", cfg = null, devices = [], sysDevices = { available: false, devices: [] };
    let fields = new Map();          // path -> field definition (everything rendered)
    let initial = {};                // path -> value as loaded
    let glossary0 = null;            // glossary text as loaded (null = not loaded yet)
    let logs = null;

    // ---- values
    const initialOf = (f) => {
      const v = getPath(cfg.values, f.path);
      if (f.type === "secret") return "";
      if (f.type === "bool") return !!v;
      if (f.type === "channels") return [...(v || [])].sort();
      if (f.type === "number") return v === null || v === undefined || v === "" ? null : Number(v);
      if (f.path === "audio.device" || f.path === "audio.system_device") return v === null || v === undefined ? "" : String(v);
      return v === null || v === undefined ? "" : String(v);
    };
    const read = (f) => {
      if (f.type === "channels") return $$(`[data-chan]`, bodyEl).filter((e) => e.checked).map((e) => e.dataset.chan).sort();
      const el = document.getElementById(fid(f.path));
      if (!el) return initial[f.path];
      if (f.type === "bool") return el.checked;
      if (f.type === "number") return el.value === "" ? null : Number(el.value);
      return el.value;
    };
    /** What PUT /api/config gets for a changed field (undefined = leave it out). */
    const wire = (f, v) => {
      if (f.type === "secret") return v ? v : undefined;
      if (f.type === "number") { if (v === null) return f.nullable ? null : undefined; return Number.isFinite(v) ? v : undefined; }
      if (f.path === "audio.device") return v === "" ? null : (/^\d+$/.test(v) ? Number(v) : v);     // device names, or an index
      if (f.path === "audio.system_device") return v === "" ? null : v;
      return v;
    };
    const changedPaths = () => [...fields.values()].filter((f) => !deepEqual(read(f), initial[f.path])).map((f) => f.path);
    const glossaryEl = () => document.getElementById("glossary");
    const glossaryDirty = () => { const el = glossaryEl(); return !!el && glossary0 !== null && el.value !== glossary0; };
    const dirtyCount = () => (tab === "form" && cfg ? changedPaths().length + (glossaryDirty() ? 1 : 0) : 0);

    // ---- field markup
    const control = (f) => {
      const id = fid(f.path), v = initial[f.path];
      if (f.type === "secret") {
        const s = cfg.secrets[f.path] || { set: false, hint: "" };
        return html`<input class="input" type="password" id="${id}" data-path="${f.path}" placeholder="${f.placeholder || (s.set ? "留空表示不改" : "")}" autocomplete="new-password">`;
      }
      if (f.type === "bool") return html`<label class="check"><input type="checkbox" id="${id}" data-path="${f.path}" ${v ? html`checked` : ""}> <span>${f.label}</span></label>`;
      if (f.type === "channels") return html`<div class="chan-row" role="group" aria-label="${f.label}">${PUSH_CHANNELS.map(([k, l]) => html`<label class="check"><input type="checkbox" data-chan="${k}" ${v.includes(k) ? html`checked` : ""}> ${l}</label>`)}</div>`;
      if (f.type === "select") {
        let opts = f.options || [];
        if (f.dyn === "devices") opts = [["", "系统默认输入"], ...devices.map((d) => [String(d.index), `${d.index}: ${d.name} (${d.hostapi})`])];
        if (f.dyn === "sysDevices") opts = [["", "默认输出设备"], ...sysDevices.devices.map((d) => [d.name, `${d.name}${d.is_default ? "（默认）" : ""}`])];
        if (v !== "" && !opts.some(([k]) => String(k) === String(v))) opts = [...opts, [String(v), `${v}（当前配置，未在列表中）`]];
        return html`<select class="input" id="${id}" data-path="${f.path}">${opts.map(([k, l]) => html`<option value="${k}" ${String(v) === String(k) ? html`selected` : ""}>${l}</option>`)}</select>`;
      }
      const type = f.type === "number" ? "number" : f.type === "time" ? "time" : "text";
      return html`<input class="input" type="${type}" id="${id}" data-path="${f.path}" value="${v ?? ""}" ${attrs({ min: f.min, max: f.max, step: f.type === "number" ? (f.step ?? "any") : null, placeholder: f.placeholder || null, inputmode: f.type === "number" ? "decimal" : null })}>`;
    };
    const fieldHtml = (f) => {
      fields.set(f.path, f);
      const s = f.type === "secret" ? (cfg.secrets[f.path] || { set: false, hint: "" }) : null;
      const help = f.dyn === "sysDevices" && !sysDevices.available ? "当前环境没有可用的系统声音源（需要 Windows 和 PyAudioWPatch）。" : f.help || "";
      const search = `${f.label} ${f.help || ""} ${f.path}`.toLowerCase();
      const label = f.type === "bool" ? "" : html`<label ${f.type === "channels" ? "" : html`for="${fid(f.path)}"`} class="${f.auto ? "mono" : ""}">${f.label}${s ? (s.set ? badge(`已设置 ${s.hint}`, "ok") : badge("未设置", "plain")) : ""}</label>`;
      return html`<div class="field ${f.wide ? "wide" : ""}" data-field="${f.path}" data-search="${search}">
        ${label}${control(f)}
        ${when(f.extra === "loudness", () => html`<div class="row"><button type="button" class="btn sm" id="loudness-stats" title="给没算过音量的句子补算（读音频，不花钱），然后统计有多少句低于门槛；不会删除任何记录">统计音量分布</button><span class="help" id="loudness-note"></span></div>`)}
        ${when(help, () => html`<div class="help">${help}</div>`)}
        <button type="button" class="field-revert link-btn" data-revert="${f.path}" hidden>${icon("undo", 12)} 撤销这项修改</button>
      </div>`;
    };
    const customHtml = (s) => {
      if (s.custom === "glossary") {
        return html`<div class="field wide" data-field="glossary" data-search="词表 glossary 人名 术语"><label for="glossary">词表内容 <code>${cfg.glossary_path}</code></label>
          <textarea class="input mono" id="glossary" rows="8" placeholder="张三&#10;星河计划&#10;灰度发布" spellcheck="false"></textarea>
          <div class="help">和设置一起保存；保存后会更新检索索引。</div>
          <button type="button" class="field-revert link-btn" data-revert="glossary" hidden>${icon("undo", 12)} 撤销这项修改</button></div>`;
      }
      if (s.custom === "import") {
        return html`<div class="field wide" data-field="import" data-search="导入 录音 文件 import"><div class="row">
            <input type="file" id="imp-file" class="file-input" accept="audio/*,.m4a,.amr,.mp4" aria-label="选择要导入的录音文件">
            <input class="input imp-start" id="imp-start" type="datetime-local" aria-label="录音开始时间（可选）" title="录音开始时间（可选）">
            ${button({ label: "导入", icon: "upload", attrs: { id: "imp-btn" } })}</div>
          <div class="help">文件名里有时间（如 20261031_0900）就自动用它，否则按文件修改时间，也可以手动填。</div><div id="imp-result" aria-live="polite"></div></div>`;
      }
      return html`<div class="field wide" data-field="paths" data-search="配置文件 数据目录 模型目录 路径 path">${kv([["配置文件", html`<span class="mono">${cfg.path}</span>`], ["数据目录", html`<span class="mono">${cfg.data_dir}</span>`], ["模型目录", html`<span class="mono">${cfg.models_dir}</span>`]])}</div>`;
    };
    const sectionHtml = (s, extra) => {
      const all = [...s.fields.filter((f) => f.type === "secret" ? f.path in cfg.secrets : getPath(cfg.values, f.path) !== undefined), ...(extra || [])];
      if (!all.length && !s.custom) return "";
      for (const f of all) initial[f.path] = initialOf(f);
      const main = all.filter((f) => !f.advanced), adv = all.filter((f) => f.advanced);
      return html`<section class="card st-sec" id="sec-${s.id}" data-sec="${s.id}" aria-labelledby="sec-${s.id}-h">
        <div class="sec-head"><div class="sec-title"><h2 id="sec-${s.id}-h">${s.title}</h2>${when(s.desc, () => html`<div class="sec-sub">${s.desc}</div>`)}</div>
          ${when(s.action, () => html`<div class="sec-acts">${button({ label: s.action.label, size: "sm", attrs: { "data-sec-act": s.action.id } })}</div>`)}</div>
        ${when(main.length || s.custom, () => html`<div class="form-grid">${main.map(fieldHtml)}${s.custom ? customHtml(s) : ""}</div>`)}
        ${when(adv.length, () => html`<details class="adv"><summary><span>高级（${adv.length} 项）</span></summary><div class="form-grid">${adv.map(fieldHtml)}</div></details>`)}
      </section>`;
    };

    const formHtml = () => {
      fields = new Map();
      initial = {};
      const extra = extraFields(cfg.values, cfg.defaults, cfg.secrets);
      const secs = SECTIONS.map((s) => ({ s, markup: sectionHtml(s, extra[s.id]) })).filter((x) => x.markup !== "");
      return html`<div class="st-layout">
        <aside class="st-side"><div class="st-search"><label class="sr-only" for="st-q">搜索设置</label><input class="input" type="search" id="st-q" placeholder="搜索设置" autocomplete="off"></div>
          <nav class="st-nav" id="st-nav" aria-label="设置的分区">${secs.map(({ s }) => html`<a href="${hrefFor("settings", s.id)}" data-nav="${s.id}">${icon(s.icon, 16)}<span>${s.title}</span></a>`)}</nav></aside>
        <form class="st-form" id="cfg-form" novalidate>
          <p class="lead st-lead">档案「${cfg.profile_label}」${cfg.base_config_path ? html`：这里保存的值写入 <code>${cfg.path}</code>，没改过的项继承根目录 <code>config.yaml</code>（API key 等共享设置建议放在那里）。` : html`：保存后写入 <code>${cfg.path}</code>。`}保存后立即生效。</p>
          ${when(cfg.env_api_key, () => notice({ kind: "info", body: "当前 API key 来自环境变量 MIMO_API_KEY，文件里的会被忽略。" }))}
          ${secs.map((x) => x.markup)}
          <div id="st-empty" hidden>${emptyState({ compact: true, icon: "search", title: "没有匹配的设置", text: "换个词试试，比如「阈值」「推送」「麦克风」。" })}</div>
        </form>
        <div class="save-bar" id="save-bar" hidden role="region" aria-label="未保存的修改"><span id="save-count"></span><span class="grow"></span>
          <button type="button" class="btn ghost sm" data-save="discard">放弃修改</button><button type="button" class="btn primary sm" data-save="save">保存</button></div>
      </div>`;
    };

    // ---- dirty state
    const paintDirty = () => {
      if (tab !== "form" || !cfg) return;
      const changed = new Set(changedPaths());
      if (glossaryDirty()) changed.add("glossary");
      for (const el of $$("[data-field]", bodyEl)) {
        const on = changed.has(el.dataset.field);
        el.classList.toggle("mod", on);
        const rv = el.querySelector(".field-revert");
        if (rv) rv.hidden = !on;
      }
      const bar = $("#save-bar");
      if (bar) { bar.hidden = changed.size === 0; $("#save-count").textContent = `${changed.size} 项修改还没保存`; }
      document.documentElement.classList.toggle("has-savebar", changed.size > 0);
    };
    const revert = (path) => {
      if (path === "glossary") { const el = glossaryEl(); if (el && glossary0 !== null) el.value = glossary0; return; }
      const f = fields.get(path);
      if (!f) return;
      const v = initial[path];
      if (f.type === "channels") { for (const e of $$("[data-chan]", bodyEl)) e.checked = v.includes(e.dataset.chan); return; }
      const el = document.getElementById(fid(path));
      if (!el) return;
      if (f.type === "bool") el.checked = !!v; else el.value = v === null ? "" : String(v);
    };

    const loudnessNote = () => {
      const el = $("#loudness-note"), b = $("#loudness-stats");
      if (!el) return;
      const j = jobOf("loudness");
      if (b) b.disabled = !!(j && j.running);
      if (!j) patch(el, html``);
      else if (j.running) patch(el, spinner("正在统计（读音频补算音量）…"));
      else if (j.error) patch(el, html`<span class="err">上次统计失败：${j.error}</span>`);
      else if (j.finished_at) patch(el, html`上次统计 ${fmtDT(j.finished_at)}${jobSummary("loudness", j.result)}`);
      else patch(el, html``);
    };

    const filter = () => {
      const q = ($("#st-q") ? $("#st-q").value : "").trim().toLowerCase();
      let any = false;
      for (const sec of $$(".st-sec", bodyEl)) {
        const title = sec.querySelector("h2").textContent.toLowerCase();
        let shown = 0;
        for (const f of $$("[data-field]", sec)) { const hit = !q || title.includes(q) || f.dataset.search.includes(q); f.hidden = !hit; if (hit) shown++; }
        for (const d of $$("details.adv", sec)) {
          const hit = $$("[data-field]:not([hidden])", d).length > 0;
          d.hidden = !hit;
          if (q) { if (hit && !d.open) { d.open = true; d.dataset.auto = "1"; } } else if (d.dataset.auto) { d.open = false; delete d.dataset.auto; }
        }
        sec.hidden = shown === 0;
        if (shown) any = true;
        const nav = $(`[data-nav="${sec.dataset.sec}"]`);
        if (nav) nav.hidden = shown === 0;
      }
      $("#st-empty").hidden = any;
    };

    const scrollToSection = (id) => {
      const el = document.getElementById(`sec-${id}`);
      if (!el) return;
      el.scrollIntoView({ block: "start" });
      for (const a of $$("#st-nav a")) a.toggleAttribute("aria-current", a.dataset.nav === id);
    };

    const loadForm = async ({ force = false } = {}) => {
      [cfg, devices, sysDevices] = await Promise.all([
        loadConfig({ force }), api("/api/devices", { signal: ctx.signal }).catch(() => []),
        api("/api/devices/system", { signal: ctx.signal }).catch(() => ({ available: false, devices: [] })),
      ]);
      if (!ctx.alive() || tab !== "form") return;
      const keepQ = $("#st-q") ? $("#st-q").value : "";
      const openAdv = $$("details.adv[open]:not([data-auto])", bodyEl).map((d) => d.closest(".st-sec").dataset.sec);
      const text = glossaryEl() && glossaryDirty() ? glossaryEl().value : null;
      render(bodyEl, formHtml());
      for (const id of openAdv) { const d = $(`#sec-${id} details.adv`); if (d) d.open = true; }
      if (keepQ) { $("#st-q").value = keepQ; filter(); }
      loudnessNote();
      paintDirty();
      observeSections();
      try { const g = await api("/api/glossary", { signal: ctx.signal }); glossary0 = g.text || ""; const el = glossaryEl(); if (el) el.value = text !== null ? text : glossary0; }
      catch (e) { if (!isAbort(e)) glossary0 = null; }
      paintDirty();
    };

    // highlight the section in view in the navigator
    let io = null;
    const observeSections = () => {
      if (io) io.disconnect();
      if (!("IntersectionObserver" in window)) return;
      io = new IntersectionObserver((entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (!vis) return;
        for (const a of $$("#st-nav a")) a.toggleAttribute("aria-current", a.dataset.nav === vis.target.dataset.sec);
      }, { rootMargin: "-80px 0px -60% 0px" });
      for (const s of $$(".st-sec", bodyEl)) io.observe(s);
    };

    const save = async (btn) => {
      const restore = busy(btn, "保存中…");
      try {
        const paths = changedPaths();
        const out = {};
        for (const p of paths) { const f = fields.get(p); const v = wire(f, read(f)); if (v !== undefined) setPath(out, p, v); }
        let r = { saved: [], notes: [] };
        if (Object.keys(out).length) r = await api("/api/config", { method: "PUT", json: out });
        let glossaryMsg = "";
        if (glossaryDirty()) {
          const g = await api("/api/glossary", { method: "PUT", json: { text: glossaryEl().value, reindex: true } });
          glossary0 = glossaryEl().value;
          glossaryMsg = `词表已保存（${g.terms} 个词）${g.job ? "，正在更新检索索引" : ""}`;
        }
        const saved = r.saved || [];
        toast([saved.length ? `已保存 ${saved.length} 项设置` : "", glossaryMsg, (r.notes || []).join("；")].filter(Boolean).join("；") || "没有需要保存的修改", { kind: "ok", ms: (r.notes || []).length ? 7000 : 3500 });
        if (paths.includes("audio.min_speech_dbfs")) setMinDbfs(read(fields.get("audio.min_speech_dbfs")));
        emit("config:saved", { saved });
        await refreshStatus();
        await loadForm({ force: true });
      } catch (e) { toastErr(e); }
      restore();
      paintDirty();
    };

    // ---- tabs
    const apply = async () => {
      const p = ctx.params[0] || "";
      const want = p === "logs" ? "logs" : "form";
      patch($("#st-tabs"), tabs([{ id: "form", label: "设置" }, { id: "logs", label: "日志 / 诊断" }], want, { label: "设置和诊断" }));
      if (want !== tab) {
        tab = want;
        if (logs) { logs.destroy(); logs = null; }
        if (io) { io.disconnect(); io = null; }
        document.documentElement.classList.remove("has-savebar");
        render(bodyEl, skeleton("rows", 6));
        if (tab === "logs") { ctx.setTitle("设置 · 日志 / 诊断"); logs = mountLogs(ctx, bodyEl); return; }
        ctx.setTitle("设置");
        await loadForm();
      }
      if (tab === "form" && p && p !== "logs") scrollToSection(p);
    };

    on(ctx.el, "click", "[data-tab]", (e, b) => { location.hash = b.dataset.tab === "logs" ? "#/settings/logs" : "#/settings"; }, { signal: ctx.signal });
    on(bodyEl, "input", "#cfg-form", debounce(() => paintDirty(), 80), { signal: ctx.signal });
    on(bodyEl, "change", "#cfg-form", () => paintDirty(), { signal: ctx.signal });
    on(bodyEl, "submit", "#cfg-form", (e) => e.preventDefault(), { signal: ctx.signal });
    on(bodyEl, "input", "#st-q", debounce(() => filter(), 80), { signal: ctx.signal });
    on(bodyEl, "click", "[data-nav]", (e, a) => { e.preventDefault(); if ($("#st-q").value) { $("#st-q").value = ""; filter(); } scrollToSection(a.dataset.nav); }, { signal: ctx.signal });
    on(bodyEl, "click", "[data-revert]", (e, b) => { revert(b.dataset.revert); paintDirty(); }, { signal: ctx.signal });
    on(bodyEl, "click", "[data-save]", async (e, b) => {
      if (b.dataset.save === "save") { await save(b); return; }
      for (const p of changedPaths()) revert(p);
      revert("glossary");
      paintDirty();
    }, { signal: ctx.signal });
    on(bodyEl, "click", "[data-sec-act]", async (e, b) => {
      const a = b.dataset.secAct;
      if (dirtyCount() && !(await confirm({ title: "先保存再测试？", text: "测试用的是已经保存的设置；刚才的修改还没有保存。", okLabel: "按已保存的测试", cancelLabel: "先不测" }))) return;
      const restore = busy(b, a === "test-api" ? "测试中…" : "发送中…");
      try {
        if (a === "test-api") { const r = await api("/api/config/test-api", { method: "POST" }); toast(`连接正常：${r.model} 在 ${r.seconds}s 内回复「${r.reply}」`, { kind: "ok", ms: 6000 }); }
        else if (a === "test-push") { const r = await api("/api/push/test", { method: "POST", json: {} }); toast(Object.entries(r.channels).map(([k, v]) => `${k}：${v ? `失败 ${v}` : "已发送"}`).join("；") || "没有选择推送渠道", { kind: r.ok ? "ok" : "err", ms: 7000 }); }
      } catch (err) { toastErr(err); }
      restore();
    }, { signal: ctx.signal });
    on(bodyEl, "click", "#loudness-stats", async (e, b) => { b.disabled = true; await runJob("loudness"); loudnessNote(); }, { signal: ctx.signal });
    on(bodyEl, "click", "#imp-btn", async (e, b) => {
      const f = $("#imp-file").files[0];
      if (!f) { toast("先选择文件", { kind: "err" }); return; }
      const box = $("#imp-result");
      const restore = busy(b, "导入中…");
      render(box, spinner(`正在导入 ${f.name}（切分语音段）…`));
      try {
        const start = $("#imp-start").value ? $("#imp-start").value.replace("T", " ") : null;
        const r = await api("/api/import", { method: "POST", query: { filename: f.name, start }, body: f });
        render(box, notice({ kind: "ok", compact: true, body: `已导入 ${r.segments} 段语音。`, actions: button({ label: "立即转写", size: "sm", kind: "primary", attrs: { id: "imp-proc" } }) }));
        emit("utterances:changed", { reason: "import" });
      } catch (err) { render(box, notice({ kind: "danger", compact: true, body: err.message || String(err) })); }
      restore();
    }, { signal: ctx.signal });
    on(bodyEl, "click", "#imp-proc", () => runJob("process"), { signal: ctx.signal });

    ctx.on("status", loudnessNote);
    ctx.guard(async (to) => {
      if (to && to.page === "settings" && (to.params[0] || "") !== "logs" && tab === "form") return true;     // jumping between sections keeps the form
      return !dirtyCount() || (await confirm({ title: "还有没保存的修改", text: "离开这里会丢掉刚才改的设置。", okLabel: "放弃修改", cancelLabel: "留在这里", danger: true }));
    });
    const beforeUnload = (e) => { if (dirtyCount()) { e.preventDefault(); e.returnValue = ""; } };
    window.addEventListener("beforeunload", beforeUnload);

    ctx.apply = apply;
    await apply();
    return () => { window.removeEventListener("beforeunload", beforeUnload); if (logs) logs.destroy(); if (io) io.disconnect(); document.documentElement.classList.remove("has-savebar"); };
  },

  async update(ctx) { if (ctx.apply) await ctx.apply(); },
};
