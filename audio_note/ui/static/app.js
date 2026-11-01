/* audio-note web UI — single page, no build step. Talks to /api (see audio_note/ui/app.py). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtMin = (m) => (m >= 60 ? `${(m / 60).toFixed(1)} 小时` : `${Math.round(m)} 分钟`);
  const fmtBytes = (n) => { if (n == null) return "-"; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
  const fmtDT = (s) => (s ? String(s).replace("T", " ").slice(0, 16) : "-");
  const fmtTime = (s) => (s ? String(s).replace("T", " ").slice(11, 16) : "-");
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  const WEEK = ["日", "一", "二", "三", "四", "五", "六"];
  const dayLabel = (day) => {
    if (!day) return "";
    const d = new Date(day + "T00:00:00");
    const t = new Date(today() + "T00:00:00");
    const diff = Math.round((t - d) / 86400000);
    const w = `周${WEEK[d.getDay()]}`;
    if (diff === 0) return `今天 ${w}`;
    if (diff === 1) return `昨天 ${w}`;
    if (diff === 2) return `前天 ${w}`;
    return w;
  };
  const spkColor = (name) => { let h = 0; for (const ch of String(name || "")) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return h % 8; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.json !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
    if (opts.body !== undefined) { init.body = opts.body; init.headers["Content-Type"] = opts.contentType || "application/octet-stream"; }
    let res;
    try { res = await fetch(path, init); } catch (e) { throw new Error("无法连接到 audio-note（程序是否已退出？）"); }
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("json")) {
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return res;
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
    return data;
  }

  function toast(msg, kind = "", ms = 3500) {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), ms);
  }
  const toastErr = (e) => toast(e && e.message ? e.message : String(e), "err", 6000);

  function modal(html, onMount) {
    const root = $("#modal-root");
    root.innerHTML = `<div class="modal">${html}</div>`;
    const close = () => { root.innerHTML = ""; document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    root.onclick = (e) => { if (e.target === root) close(); };
    if (onMount) onMount(root.firstElementChild, close);
    return close;
  }
  function confirmModal(title, text, okLabel = "确定", danger = false) {
    return new Promise((resolve) => {
      modal(`<h2>${esc(title)}</h2><div class="muted">${esc(text)}</div>
        <div class="actions"><button class="btn" data-x="cancel">取消</button><button class="btn ${danger ? "danger" : "primary"}" data-x="ok">${esc(okLabel)}</button></div>`,
        (el, close) => {
          el.querySelector('[data-x="cancel"]').onclick = () => { close(); resolve(false); };
          el.querySelector('[data-x="ok"]').onclick = () => { close(); resolve(true); };
        });
    });
  }
  function promptModal(title, { label = "", value = "", placeholder = "", help = "", list = [] } = {}) {
    return new Promise((resolve) => {
      const dl = list.length ? `<datalist id="dl-prompt">${list.map((x) => `<option value="${esc(x)}">`).join("")}</datalist>` : "";
      modal(`<h2>${esc(title)}</h2>
        <div class="field"><label>${esc(label)}</label><input class="input" id="pm-input" value="${esc(value)}" placeholder="${esc(placeholder)}" list="dl-prompt">${dl}
        ${help ? `<div class="help">${esc(help)}</div>` : ""}</div>
        <div class="actions"><button class="btn" data-x="cancel">取消</button><button class="btn primary" data-x="ok">确定</button></div>`,
        (el, close) => {
          const inp = el.querySelector("#pm-input");
          inp.focus(); inp.select();
          const ok = () => { close(); resolve(inp.value.trim()); };
          inp.onkeydown = (e) => { if (e.key === "Enter") ok(); };
          el.querySelector('[data-x="cancel"]').onclick = () => { close(); resolve(null); };
          el.querySelector('[data-x="ok"]').onclick = ok;
        });
    });
  }

  // minimal markdown -> html (escaped first; headings, lists, quotes, code, tables, bold/italic/links)
  function md(src) {
    const lines = String(src || "").replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    const inline = (s) => esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    while (i < lines.length) {
      let ln = lines[i];
      if (/^\s*```/.test(ln)) {
        const buf = []; i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
        continue;
      }
      let m;
      if ((m = /^(#{1,6})\s+(.*)$/.exec(ln))) { const lv = Math.min(m[1].length, 3); out.push(`<h${lv}>${inline(m[2])}</h${lv}>`); i++; continue; }
      if (/^\s*(-{3,}|\*{3,})\s*$/.test(ln)) { out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(ln)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${md(buf.join("\n"))}</blockquote>`);
        continue;
      }
      if (/^\s*\|.*\|\s*$/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const head = ln.trim().slice(1, -1).split("|").map((c) => c.trim());
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++].trim().slice(1, -1).split("|").map((c) => c.trim()));
        out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      if (/^\s*([-*+]|\d+[.)])\s+/.test(ln)) {
        const ordered = /^\s*\d+[.)]\s+/.test(ln);
        const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let t = lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, "");
          i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) t += " " + lines[i++].trim();
          let box = "";
          t = t.replace(/^\[([ xX])\]\s*/, (_, c) => { box = `<input type="checkbox" disabled ${c.trim() ? "checked" : ""}>`; return ""; });
          items.push(`<li>${box}${inline(t)}</li>`);
        }
        out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
        continue;
      }
      if (!ln.trim()) { i++; continue; }
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*```|\s*>|\s*([-*+]|\d+[.)])\s+|\s*\|)/.test(lines[i])) buf.push(lines[i++]);
      out.push(`<p>${inline(buf.join("\n")).replace(/\n/g, "<br>")}</p>`);
    }
    return out.join("\n");
  }

  // ------------------------------------------------------------------ state & router
  const state = { status: null, page: "", speakers: [], jobsSeen: {}, refresh: null };
  let pollTimer = null;
  const main = $("#main");
  let routeGen = 0;                       // bumped on every route(); pages compare to drop stale renders
  const stale = (g) => g !== routeGen;

  function route() {
    const hash = location.hash || "#/dashboard";
    const [, page, ...rest] = hash.split("/");
    const arg = rest.join("/");
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    state.page = page;
    state.refresh = null;
    main.onfocusout = null;                 // the speakers page installs one; drop it with the page
    const my = ++routeGen;
    stopPagePoll();
    stopRecording();
    const fn = PAGES[page] || PAGES.dashboard;
    main.innerHTML = `<div class="muted">加载中…</div>`;
    fn(decodeURIComponent(arg), my).catch((e) => { if (!stale(my)) main.innerHTML = `<div class="alert danger">${esc(e.message || e)}</div>`; });
  }
  window.addEventListener("hashchange", route);

  let pagePoll = null;
  function startPagePoll(fn, ms, my) {
    if (my !== undefined && stale(my)) return;      // the page that asked for this poll is already gone
    stopPagePoll();
    pagePoll = setInterval(() => { if (!document.hidden && !(my !== undefined && stale(my))) fn().catch(() => {}); }, ms);
  }
  function stopPagePoll() { if (pagePoll) clearInterval(pagePoll); pagePoll = null; }

  // sidebar status (always polling)
  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      const prev = state.status;
      state.status = st;
      renderPill(st);
      notifyJobs(prev, st);
    } catch (e) {
      const pill = $("#rec-pill");
      pill.className = "rec-pill";
      $("#rec-pill-text").textContent = "连接断开";
    }
  }
  function renderPill(st) {
    const pill = $("#rec-pill");
    const s = st.service;
    let cls = "rec-pill", txt = "未录音";
    if (s.read_only) { cls += " ro"; txt = `只读（服务在 pid ${s.other_pid}）`; }
    else if (s.recording) { cls += s.is_speaking ? " on speaking" : " on"; txt = s.is_speaking ? "录音中 · 有人说话" : "录音中 · 安静"; }
    pill.className = cls;
    $("#rec-pill-text").textContent = txt;
    if (st.profile) $("#profile-name").textContent = st.profile.label;
    const bits = [];
    if (s.recording) bits.push(`本次 ${s.segments_written} 段 / ${Math.round(s.speech_total_s / 60)} 分钟`);
    if (st.pending) bits.push(`${st.pending} 段待转写`);
    if (!st.api_key) bits.push("未配置 API key");
    $("#sidebar-meta").innerHTML = bits.map(esc).join("<br>");
  }
  function notifyJobs(prev, st) {
    if (!prev) { for (const [n, j] of Object.entries(st.jobs)) state.jobsSeen[n] = j.finished_at; return; }
    for (const [n, j] of Object.entries(st.jobs)) {
      if (j.finished_at && j.finished_at !== state.jobsSeen[n]) {
        state.jobsSeen[n] = j.finished_at;
        if (j.error) toast(`${j.label}失败：${j.error}`, "err", 8000);
        else toast(`${j.label}完成${jobSummary(n, j.result)}`, "ok", 5000);
        if (state.refresh) state.refresh(n).catch(() => {});      // pages may ignore the job name; the speakers page uses it
      }
    }
  }
  function jobSummary(name, r) {
    if (!r || typeof r !== "object") return "";
    if (name === "process") return `：处理 ${r.processed} 段，${r.utterances} 条语句${r.failed ? `，${r.failed} 段失败` : ""}`;
    if (name === "daily") return `：${r.day}${r.cached ? "（内容没变，沿用已有日报）" : ""}${r.partial ? "（部分时段失败，稍后会重试）" : ""}`;
    if (name === "cleanup") return `：复检 ${r.rechecked}，删除无语音 ${r.no_speech}，过期 ${r.expired}`;
    if (name === "reindex") return `：关键词 ${r.keywords} 条${r.vectors != null ? `，向量 ${r.vectors} 条` : ""}`;
    if (name === "relabel") return `：${r.segments} 段，${r.changed} 条语句变化`;
    if (name === "push") return `：${Object.entries(r.channels || {}).map(([k, v]) => `${k} ${v ? "失败" : "成功"}`).join("，")}`;
    if (name === "voices") return `：分析 ${r.segments} 段录音，${r.clusters} 个声音${r.skipped ? `，${r.skipped} 段跳过` : ""}`;
    if (name === "relations") return `：检查 ${r.speakers} 人，写入 ${r.applied} 条关系`;
    return "";
  }

  async function runJob(name, body = {}, label = "") {
    try {
      const r = await api(`/api/jobs/${name}`, { method: "POST", json: body });
      toast(`${r.job.label}已开始`, "");
      await refreshStatus();
      return r;
    } catch (e) { toastErr(e); }
  }

  // ------------------------------------------------------------------ profiles
  async function profilesModal() {
    let data;
    try { data = await api("/api/profiles"); } catch (e) { toastErr(e); return; }
    const rows = data.profiles.map((p) => {
      const st = p.stats;
      const stats = p.error ? `<span class="badge danger">${esc(p.error)}</span>` : st ? `${st.utterances} 条记录 · ${st.days} 天 · ${st.speakers} 个说话人${st.last_day ? ` · 最近 ${esc(st.last_day)}` : ""}` : "还没有数据";
      return `<div class="profile-row ${p.current ? "cur" : ""}">
        <div class="grow"><div class="pn">${esc(p.label)}${p.current ? `<span class="badge info">当前</span>` : ""}${p.default ? `<span class="badge plain">config.yaml + data/</span>` : ""}${p.running_pid ? `<span class="badge warn">另一进程在用 (pid ${p.running_pid})</span>` : ""}</div>
          <div class="ps">${stats}</div><div class="ps mono">${esc(p.data_dir || "")}</div></div>
        ${p.current ? "" : `<button class="btn sm primary" data-switch="${esc(p.name)}">切换</button>`}
        ${p.current || p.default ? "" : `<button class="icon-btn" data-delp="${esc(p.name)}" title="删除档案及其全部数据">✕</button>`}</div>`;
    }).join("");
    modal(`<h2>档案</h2><div class="muted small mb">每个档案有自己的录音、文字记录、日报和说话人声纹，互不混淆（比如「demo」是示例数据，自己用另建一个）。API key 等共享设置放在根目录 config.yaml，各档案继承。${data.custom_config ? `<br>当前用的是自定义配置文件 <code>${esc(data.custom_config)}</code>，不属于任何档案。` : ""}</div>
      <div>${rows}</div>
      <div class="mt" style="border-top:1px solid var(--border);padding-top:12px"><div class="row"><input class="input grow" id="np-name" placeholder="新档案名，例如：我 / 家里 / 工作"><label class="check small"><input type="checkbox" id="np-copy"> 复制当前档案的设置</label><button class="btn" id="np-create">创建并切换</button></div></div>
      <div class="actions"><button class="btn" data-x="close">关闭</button></div>`,
      (el, close) => {
        el.querySelector('[data-x="close"]').onclick = close;
        const switchTo = async (name) => {
          const btns = el.querySelectorAll("button"); btns.forEach((b) => { b.disabled = true; });
          try {
            await api(`/api/profiles/${encodeURIComponent(name)}/switch`, { method: "POST" });
            chatState.msgs = [];
            close(); toast(`已切换到档案 ${name === "default" ? "默认" : name}`, "ok");
            await refreshStatus(); route();
          } catch (e) { toastErr(e); btns.forEach((b) => { b.disabled = false; }); }
        };
        el.onclick = async (e) => {
          const sw = e.target.closest("[data-switch]"); if (sw) return switchTo(sw.dataset.switch);
          const del = e.target.closest("[data-delp]");
          if (del) {
            const name = del.dataset.delp;
            const typed = await promptModal(`删除档案 ${name}`, { label: `会删除它的全部录音、记录、日报和声纹，不可恢复。输入档案名确认：`, placeholder: name });
            if (typed === null) { profilesModal(); return; }
            if (typed !== name) { toast("名字不一致，已取消", "err"); profilesModal(); return; }
            try { await api(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE", json: { confirm: name } }); toast(`已删除档案 ${name}`); }
            catch (err) { toastErr(err); }
            profilesModal();
          }
        };
        el.querySelector("#np-create").onclick = async () => {
          const name = el.querySelector("#np-name").value.trim(); if (!name) { toast("先填档案名", "err"); return; }
          try {
            await api("/api/profiles", { method: "POST", json: { name, copy_from: el.querySelector("#np-copy").checked && data.current ? data.current : null } });
            await switchTo(name);
          } catch (e) { toastErr(e); }
        };
        el.querySelector("#np-name").onkeydown = (e) => { if (e.key === "Enter") el.querySelector("#np-create").click(); };
      });
  }
  $("#profile-btn").onclick = profilesModal;

  // ------------------------------------------------------------------ dashboard
  async function pageDashboard(arg, my) {
    await refreshStatus();
    if (stale(my)) return;
    const render = async () => {
      const st = state.status; if (!st) return;
      const s = st.service;
      const recent = await api(`/api/transcript/${st.today.day}`).catch(() => null);
      if (stale(my)) return;
      const utts = recent ? recent.sessions.flatMap((x) => x.utterances).slice(-8).reverse() : [];
      const stateCls = s.recording ? (s.is_speaking ? "state on speaking" : "state on") : "state";
      const stateTxt = s.read_only ? "只读模式" : s.recording ? (s.is_speaking ? "正在录音 · 有人说话" : "正在录音 · 安静") : "未在录音";
      const srcLine = `声音来源：麦克风${s.system_audio ? `＋系统声音${s.recording && s.system_name ? `（${esc(s.system_name)}）` : ""}` : ""}`
        + ` <button class="chip ${s.system_audio ? "on" : ""}" data-act="${s.system_audio ? "sys-off" : "sys-on"}" title="开会 / 通话时把对方的声音（扬声器或耳机里放出来的）一起录下来；用耳机效果最好，外放时对方的声音会被麦克风和系统声音各录一次">${s.system_audio ? "系统声音：开" : "系统声音：关"}</button>`;
      const detail = s.read_only
        ? `服务正在另一个进程运行（pid ${s.other_pid}）。这里可以看记录和问答，录音和处理请在那个进程的界面操作。`
        : s.recording
          ? `${fmtTime(s.recorder_started_at)} 开始 · 本次已保存 ${s.segments_written} 段，${fmtMin(s.speech_total_s / 60)} 语音 · 麦克风：${s.device == null ? "系统默认" : esc(String(s.device))}<br>${srcLine}`
          : `开始后只保存有人说话的片段；每 ${st.process_interval_min} 分钟自动转写一次，${esc(st.daily_time)} 生成日报${s.scheduler ? "" : "（后台处理未运行）"}。<br>${srcLine}`;
      const alerts = [];
      if (st.models_missing.length) alerts.push(`<div class="alert danger">模型文件缺失：${st.models_missing.join("、")}。运行 <code>scripts/download_models.ps1</code> 下载。</div>`);
      if (!st.api_key) alerts.push(`<div class="alert warn">还没有配置 MiMo API key：录音会保存，但不会转写、也不能问答。<span class="right"><a class="btn sm" href="#/settings">去设置</a></span></div>`);
      if (s.last_error && !s.recording) alerts.push(`<div class="alert danger">录音线程出错：${esc(s.last_error)}<span class="right"><a class="btn sm" href="#/settings">检查麦克风设置</a></span></div>`);
      if (s.recording && s.system_audio && s.system_error) alerts.push(`<div class="alert warn">系统声音没有录上（只录麦克风）：${esc(s.system_error)}<span class="right"><a class="btn sm" href="#/settings">检查设置</a></span></div>`);
      if (st.failed) alerts.push(`<div class="alert warn">${st.failed} 段转写失败。<span class="right"><button class="btn sm" data-act="retry">重试失败的段</button><a class="btn sm" href="#/transcript">查看</a></span></div>`);
      if (st.pending && st.api_key && !st.jobs.process.running) alerts.push(`<div class="alert info">${st.pending} 段录音等待转写${s.scheduler ? `（每 ${st.process_interval_min} 分钟自动处理一次）` : ""}。<span class="right"><button class="btn sm primary" data-act="process">立即转写</button></span></div>`);
      const cap = st.disk.cap_gb ? st.disk.cap_gb * 1024 ** 3 : 0;
      const pct = cap ? Math.min(100, (st.disk.audio_bytes / cap) * 100) : 0;
      const jobs = ["process", "daily", "cleanup", "relabel", "reindex"].map((n) => {
        const j = st.jobs[n];
        const status = j.running ? `<span class="btn sm ghost"><span class="spin"></span>运行中 ${fmtTime(j.started_at)} 开始</span>`
          : j.error ? `<span class="s err" title="${esc(j.error)}">上次失败：${esc(j.error)}</span>`
            : j.finished_at ? `<span class="s">上次 ${fmtDT(j.finished_at)}${esc(jobSummary(n, j.result))}</span>` : `<span class="s">尚未运行</span>`;
        return `<div class="job-row"><span class="n">${esc(j.label)}</span>${status}<button class="btn sm" data-job="${n}" ${j.running || s.read_only ? "disabled" : ""}>运行</button></div>`;
      }).join("");
      main.innerHTML = `
        <div class="page-head"><div><h1>总览</h1><div class="sub">${esc(st.today.day)} ${esc(dayLabel(st.today.day))} · 档案：${esc(st.profile ? st.profile.label : "")}</div></div>
          <div class="row">${s.read_only ? "" : s.recording
            ? `<button class="btn" data-act="stop">停止录音</button>`
            : `<button class="btn rec" data-act="start" ${st.models_missing.length ? "disabled" : ""}>● 开始录音</button>`}</div></div>
        ${alerts.join("")}
        <div class="card hero"><div><div class="${stateCls}"><span class="dot"></span>${stateTxt}</div><div class="detail">${detail}</div></div></div>
        ${st.unknown_utterances ? `<div class="hint-line">有 ${st.unknown_utterances} 句话还没认出是谁说的 → <a href="#/speakers">去标记</a></div>` : ""}
        <div class="tiles mt">
          <div class="tile"><div class="k">今日语句</div><div class="v">${st.today.utterances}<small>条</small></div></div>
          <div class="tile"><div class="k">今日语音</div><div class="v">${st.today.recorded_speech_min.toFixed(0)}<small>分钟</small></div></div>
          <div class="tile"><div class="k">待转写</div><div class="v">${st.pending}<small>段</small></div></div>
          <div class="tile"><div class="k">全部记录</div><div class="v">${st.total.utterances}<small>条 / ${st.total.days} 天</small></div></div>
          <div class="tile"><div class="k">音频占用</div><div class="v">${fmtBytes(st.disk.audio_bytes)}${cap ? `<small>/ ${st.disk.cap_gb} GB</small>` : ""}</div>${cap ? `<div class="progress mt" style="margin-top:6px"><i class="${pct > 90 ? "danger" : pct > 70 ? "warn" : ""}" style="width:${pct.toFixed(1)}%"></i></div>` : ""}</div>
          <div class="tile"><div class="k">下次日报</div><div class="v">${esc(st.daily_time)}</div><div class="small muted">${st.last_daily ? `最近：${esc(st.last_daily.day)}` : "还没有日报"}</div></div>
        </div>
        <div class="grid grid-2 mt">
          <div class="card"><div class="card-head"><h2>最近的记录</h2><a href="#/transcript/${esc(st.today.day)}" class="small">查看今天全部 →</a></div>
            ${utts.length ? utts.map((u) => uttRow(u, { noActs: true })).join("") : `<div class="empty-state"><div class="big">今天还没有转写内容</div>${st.pending ? `${st.pending} 段录音等待转写` : s.recording ? "录到有人说话后，会在下一轮自动转写" : "点右上角「开始录音」"}</div>`}
          </div>
          <div>
            <div class="card"><div class="card-head"><h2>后台任务</h2><span class="small muted">${st.last_process_at ? `上次自动转写 ${fmtDT(st.last_process_at)}` : ""}</span></div>${jobs}</div>
            ${st.last_daily ? `<div class="card"><div class="card-head"><h2>最近日报 · ${esc(st.last_daily.day)}</h2><a href="#/daily/${esc(st.last_daily.day)}" class="small">查看 →</a></div><div class="small muted">生成于 ${fmtDT(st.last_daily.created_at)} · ${st.last_daily.pushed_at ? `已推送 ${fmtDT(st.last_daily.pushed_at)}` : "未推送"} · ${st.last_daily.utterance_count} 条记录</div></div>` : ""}
          </div>
        </div>`;
      main.onclick = async (e) => {
        const b = e.target.closest("[data-act],[data-job]"); if (!b) return;
        if (b.dataset.job) { const n = b.dataset.job; if (n === "daily") return runJob("daily", { day: st.today.day }); return runJob(n); }
        const act = b.dataset.act;
        try {
          if (act === "start") { b.disabled = true; const r = await api("/api/recording/start", { method: "POST" }); if (r.last_error) toast(`录音启动失败：${r.last_error}`, "err", 8000); else toast("开始录音", "ok"); }
          else if (act === "stop") { if (await confirmModal("停止录音", "停止后不再保存新的语音片段；后台转写和日报仍会继续。", "停止")) { await api("/api/recording/stop", { method: "POST" }); toast("已停止录音"); } }
          else if (act === "process") return runJob("process");
          else if (act === "retry") return runJob("process", { retry_failed: true });
          else if (act === "sys-on" || act === "sys-off") {
            const on = act === "sys-on";
            const r = await api("/api/config", { method: "PUT", json: { audio: { system_audio: on } } });
            toast(on ? `已开启系统声音${r.restarted_recorder ? "，录音已重启" : ""}` : `已关闭系统声音${r.restarted_recorder ? "，录音已重启" : ""}`, "ok");
            await sleep(600);
          }
          await refreshStatus(); await render();
        } catch (err) { toastErr(err); await refreshStatus(); await render(); }
      };
    };
    await render();
    state.refresh = render;
    startPagePoll(async () => { await refreshStatus(); const st = state.status; if (st && !$("#modal-root").children.length) await render(); }, 4000, my);
  }

  // ------------------------------------------------------------------ transcript
  function uttRow(u, { noActs = false, hl = false } = {}) {
    const unk = (state.status && state.status.unknown_label) || "未知";
    const isUnk = !u.speaker_id;
    const c = isUnk ? "" : `spk-c${spkColor(u.speaker)}`;
    const selfName = state.status && state.status.self_name;
    const meBadge = !isUnk && selfName && u.speaker === selfName ? `<span class="badge me" title="本人">我</span>` : "";
    return `<div class="utt ${isUnk ? "unknown" : ""} ${hl ? "hl" : ""}" data-id="${u.id}" id="u${u.id}">
      <span class="time">${esc(u.time.slice(0, 5))}</span>
      <span class="spk ${c}" title="${esc(u.speaker)}${u.score != null ? ` 相似度 ${u.score}` : ""}">${esc(u.speaker)}${meBadge}${u.score != null && !isUnk ? `<span class="sc">${u.score.toFixed(2)}</span>` : ""}</span>
      <span class="text">${esc(u.text)}${u.text_raw ? `<div class="raw" title="纠错前的原始转写">原文：${esc(u.text_raw)}</div>` : ""}</span>
      <span class="acts">${noActs ? "" : `${u.audio ? `<button class="icon-btn" data-u="play" title="播放这句">▶</button>` : ""}<button class="icon-btn" data-u="label" title="标记这句话是谁说的">👤</button><button class="icon-btn" data-u="ctx" title="前后文">↕</button>`}</span>
    </div>`;
  }

  async function pageTranscript(arg, my) {
    const days = await api("/api/days");
    if (stale(my)) return;
    let day = arg && /^\d{4}-\d{2}-\d{2}$/.test(arg.split("?")[0]) ? arg.split("?")[0] : (days[0] ? days[0].day : today());
    const hlId = (arg.match(/u=(\d+)/) || [])[1];
    let speakerFilter = "";
    let data = null;
    const listHtml = () => days.length ? days.map((d) => `<div class="item ${d.day === day ? "active" : ""}" data-day="${d.day}">
        <div class="t"><span>${esc(d.day)}</span><span class="small muted">${esc(dayLabel(d.day))}</span></div>
        <div class="s">${d.utterances ? `${d.utterances} 条 · ${d.first}–${d.last}` : "尚未转写"}${d.pending ? ` · <span class="badge warn">${d.pending} 段待转写</span>` : ""}${d.failed ? ` · <span class="badge danger">${d.failed} 失败</span>` : ""}</div>
      </div>`).join("") : `<div class="empty">还没有任何记录</div>`;
    main.innerHTML = `
      <div class="page-head"><div><h1>记录</h1><div class="sub">按天查看转写内容；悬停一句话可以试听、标记说话人</div></div>
        <div class="row"><input class="input" id="q" placeholder="搜索全部记录（关键词或一句话描述）" style="width:320px"><button class="btn" id="q-btn">搜索</button></div></div>
      <div id="search-results"></div>
      <div class="two"><div class="pane-list" id="day-list">${listHtml()}</div><div id="day-view"><div class="muted">加载中…</div></div></div>`;
    const view = $("#day-view");

    async function loadDay() {
      $$("#day-list .item").forEach((el) => el.classList.toggle("active", el.dataset.day === day));
      view.innerHTML = `<div class="muted">加载中…</div>`;
      data = await api(`/api/transcript/${day}`);
      if (stale(my)) return;
      renderDay();
      if (hlId) { const el = $(`#u${hlId}`); if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("hl"); } }
    }
    function renderDay() {
      const info = days.find((d) => d.day === day) || { pending: 0, failed: 0 };
      const spk = Object.entries(data.speakers).sort((a, b) => b[1] - a[1]);
      const chips = spk.map(([n, c]) => `<span class="chip ${speakerFilter === n ? "on" : ""}" data-spk="${esc(n)}"><span class="spk-c${spkColor(n)}">●</span>${esc(n)} <span class="muted">${c}</span></span>`).join("");
      const sessions = data.sessions.filter((s) => !speakerFilter || s.utterances.some((u) => u.speaker === speakerFilter));
      const body = sessions.length ? sessions.map((s) => {
        const utts = speakerFilter ? s.utterances.filter((u) => u.speaker === speakerFilter) : s.utterances;
        return `<div class="session"><div class="session-head">第 ${s.index} 段 · ${esc(s.start)}–${esc(s.end)} · ${utts.length} 条</div>${utts.map((u) => uttRow(u, { hl: String(u.id) === hlId })).join("")}</div>`;
      }).join("") : `<div class="empty-state"><div class="big">${data.segments.length ? "这天的录音还没有转写" : "这天没有记录"}</div>${info.pending ? `${info.pending} 段等待转写` : ""}</div>`;
      const segs = data.segments;
      const segLine = segs.length ? `<span class="small muted">${segs.length} 段录音 · ${fmtMin(segs.reduce((a, s) => a + (s.duration_s || 0), 0) / 60)}${info.pending ? ` · ${info.pending} 待转写` : ""}${info.failed ? ` · <span class="badge danger">${info.failed} 失败</span>` : ""}</span>` : "";
      view.innerHTML = `
        <div class="card">
          <div class="card-head"><div><h2>${esc(day)} <span class="muted small">${esc(dayLabel(day))}</span></h2>${segLine}</div>
            <div class="row">${info.pending ? `<button class="btn sm primary" data-act="process">转写这天的录音</button>` : ""}${info.failed ? `<button class="btn sm" data-act="retry">重试失败的段</button>` : ""}
              <a class="btn sm" href="#/daily/${day}">日报</a><button class="btn sm" data-act="relabel" title="用当前的声纹库重新识别这天所有语句的说话人（不重新转写）">重新识别说话人</button></div></div>
          ${chips ? `<div class="row mb">${chips}${speakerFilter ? `<button class="link-btn small" data-act="clear-spk">清除筛选</button>` : ""}</div>` : ""}
          ${body}
          ${info.failed ? `<details class="mt"><summary class="muted small">失败的录音段</summary><table class="tbl small">${segs.filter((s) => s.status === "failed").map((s) => `<tr><td>#${s.id}</td><td>${fmtTime(s.started_at)}</td><td>${Math.round(s.duration_s)}s</td><td class="muted">${esc(s.error || "")}</td></tr>`).join("")}</table></details>` : ""}
        </div>`;
    }
    main.onclick = async (e) => {
      const item = e.target.closest("#day-list .item");
      if (item) { day = item.dataset.day; location.hash = `#/transcript/${day}`; return; }
      const chip = e.target.closest("[data-spk]");
      if (chip) { speakerFilter = speakerFilter === chip.dataset.spk ? "" : chip.dataset.spk; renderDay(); return; }
      const act = e.target.closest("[data-act]");
      if (act) {
        const a = act.dataset.act;
        if (a === "clear-spk") { speakerFilter = ""; renderDay(); }
        else if (a === "process") runJob("process");
        else if (a === "retry") runJob("process", { retry_failed: true });
        else if (a === "relabel") { if (await confirmModal("重新识别说话人", `用当前注册的声纹重新识别 ${day} 的所有语句（不重新转写，不花钱）。`)) runJob("relabel", { day }); }
        return;
      }
      const ub = e.target.closest("[data-u]");
      if (ub) {
        const row = ub.closest(".utt"); const id = Number(row.dataset.id);
        const u = data.sessions.flatMap((s) => s.utterances).find((x) => x.id === id);
        if (ub.dataset.u === "play") playUtt(row, id);
        else if (ub.dataset.u === "label") labelUtt(u, () => loadDay());
        else if (ub.dataset.u === "ctx") showContext(id);
        return;
      }
      const hit = e.target.closest(".search-hit");
      if (hit) { location.hash = `#/transcript/${hit.dataset.day}?u=${hit.dataset.id}`; }
    };
    const doSearch = async () => {
      const q = $("#q").value.trim(); const box = $("#search-results");
      if (!q) { box.innerHTML = ""; return; }
      box.innerHTML = `<div class="card mb"><div class="thinking"><span class="spin"></span>检索中…</div></div>`;
      try {
        const r = await api(`/api/search?q=${encodeURIComponent(q)}&limit=30`);
        box.innerHTML = `<div class="card mb"><div class="card-head"><h2>搜索「${esc(q)}」</h2><span class="small muted">${esc(r.how)} · ${r.hits.length} 条${r.expansions && r.expansions.length ? ` · 扩展词：${esc(r.expansions.join("、"))}` : ""}</span><button class="btn sm ghost" id="q-close">关闭</button></div>
          ${r.hits.length ? r.hits.map((h) => `<div class="search-hit" data-day="${h.day}" data-id="${h.id}"><span class="d">${esc(h.day)} ${esc(h.time)}</span><span class="spk-c${spkColor(h.speaker)}" style="font-weight:600">${esc(h.speaker)}</span><span>${esc(h.text)}</span></div>`).join("") : `<div class="muted">没有找到</div>`}</div>`;
        $("#q-close").onclick = () => { box.innerHTML = ""; $("#q").value = ""; };
      } catch (err) { box.innerHTML = `<div class="alert danger">${esc(err.message)}</div>`; }
    };
    $("#q-btn").onclick = doSearch;
    $("#q").onkeydown = (e) => { if (e.key === "Enter") doSearch(); };
    state.refresh = loadDay;
    await loadDay();
  }

  let currentAudio = null;
  function playUtt(row, id) {
    if (currentAudio) { currentAudio.pause(); const old = $(".utt.playing"); if (old) { old.classList.remove("playing"); const p = old.querySelector(".player"); if (p) p.remove(); } }
    const p = document.createElement("div");
    p.className = "player";
    p.innerHTML = `<audio controls autoplay src="/api/audio/utterance/${id}?pad=0.4"></audio><button class="icon-btn" title="关闭">✕</button>`;
    row.querySelector(".text").appendChild(p);
    row.classList.add("playing");
    currentAudio = p.querySelector("audio");
    p.querySelector("button").onclick = () => { currentAudio.pause(); p.remove(); row.classList.remove("playing"); currentAudio = null; };
    currentAudio.onerror = () => toast("音频无法播放（文件可能已删除）", "err");
  }
  async function showContext(id) {
    try {
      const r = await api(`/api/context/${id}?before=8&after=8`);
      modal(`<h2>前后文</h2>${r.utterances.map((u) => uttRow(u, { noActs: true, hl: u.id === id })).join("")}<div class="actions"><button class="btn" data-x="close">关闭</button></div>`,
        (el, close) => { el.querySelector('[data-x="close"]').onclick = close; });
    } catch (e) { toastErr(e); }
  }
  async function labelUtt(u, after) {
    const sp = await api("/api/speakers").catch(() => ({ speakers: [], relation_choices: [], self_id: null }));
    const names = sp.speakers.map((s) => s.name);
    const me = sp.speakers.find((s) => s.id === sp.self_id) || null;
    const choices = sp.relation_choices || [];
    modal(`<h2>这句话是谁说的？</h2>
      <div class="muted small mb">${esc(u.time.slice(0, 5))} ${esc(u.speaker)}：${esc(u.text)}</div>
      <div class="field"><label>姓名（已有的人就选，新名字会自动登记）</label><input class="input" id="lb-name" list="lb-list" placeholder="例如：张三" value="${u.speaker_id ? esc(u.speaker) : ""}"><datalist id="lb-list">${names.map((n) => `<option value="${esc(n)}">`).join("")}</datalist></div>
      <div id="lb-new" class="row mt hidden"><label class="check small"><input type="checkbox" id="lb-self"> ${me ? "这其实是我" : "这是我自己"}</label>${relationSelect(choices, "", { empty: "和我的关系（可选）" })}</div>
      <label class="check small mt"><input type="checkbox" id="lb-whole" checked> 这段录音里同一个声音的其他句子也一起标</label>
      <div class="help mt">会用这句话的声音补充该说话人的声纹，并重新识别同一段录音里的其他语句（${(u.end_s - u.start_s).toFixed(1)} 秒，太短的话识别效果一般，建议选 3 秒以上的句子）。</div>
      ${u.audio ? `<div class="mt"><audio controls src="/api/audio/utterance/${u.id}?pad=0.3" style="width:100%"></audio></div>` : `<div class="alert warn mt">这段音频已被删除，无法提取声纹</div>`}
      <div class="actions"><button class="btn" data-x="cancel">取消</button><button class="btn primary" data-x="ok" ${u.audio ? "" : "disabled"}>标记</button></div>`,
      (el, close) => {
        const inp = el.querySelector("#lb-name"); inp.focus();
        wireRelationSelects(el);
        const newBox = el.querySelector("#lb-new");
        const isNew = () => { const n = inp.value.trim(); return !!n && !names.includes(n); };
        const syncNew = () => newBox.classList.toggle("hidden", !isNew());      // 本人 / 关系 only for a name not registered yet
        inp.oninput = syncNew; inp.onchange = syncNew; syncNew();
        el.querySelector('[data-x="cancel"]').onclick = close;
        const ok = async () => {
          const name = inp.value.trim(); if (!name) return;
          const body = { utterance_id: u.id, name, whole_cluster: el.querySelector("#lb-whole").checked };
          if (isNew()) {
            body.is_self = el.querySelector("#lb-self").checked;
            const rel = relationValue(newBox); if (rel) body.relation = rel;
          }
          const b = el.querySelector('[data-x="ok"]'); b.disabled = true; b.innerHTML = `<span class="spin"></span>处理中…`;
          try {
            const r = await api("/api/speakers/label", { method: "POST", json: body });
            toast(`已标记为 ${name}：${r.assigned != null ? `${r.assigned} 句` : "已更新"}${r.relabeled_changed ? `，这段录音里另有 ${r.relabeled_changed} 句自动更新` : ""}`, "ok", 6000);
            if (body.is_self) await refreshStatus();        // so the 我 badge shows on the very next render, not after the next poll
            close(); if (after) after();
          } catch (e) { toastErr(e); b.disabled = false; b.textContent = "标记"; }
        };
        el.querySelector('[data-x="ok"]').onclick = ok;
        inp.onkeydown = (e) => { if (e.key === "Enter") ok(); };
      });
  }

  // ------------------------------------------------------------------ daily
  async function pageDaily(arg, my) {
    const [list, days] = await Promise.all([api("/api/daily"), api("/api/days")]);
    if (!state.status) await refreshStatus();
    if (stale(my)) return;
    let tab = arg === "notes" ? "notes" : "digest";
    let day = arg && /^\d{4}-\d{2}-\d{2}$/.test(arg) ? arg : (list[0] ? list[0].day : (days[0] ? days[0].day : today()));
    const known = new Set(list.map((d) => d.day));
    const extra = days.filter((d) => !known.has(d.day) && d.utterances).map((d) => ({ day: d.day, missing: true, now_count: d.utterances }));
    const all = [...list, ...extra].sort((a, b) => (a.day < b.day ? 1 : -1));
    let mdView = false;
    main.innerHTML = `
      <div class="page-head"><div><h1>日报</h1><div class="sub">每天 ${esc(state.status ? state.status.daily_time : "")} 自动用大模型总结当天内容并推送</div></div>
        <div class="row"><button class="btn ${tab === "digest" ? "primary" : ""}" data-tab="digest">日报</button><button class="btn ${tab === "notes" ? "primary" : ""}" data-tab="notes">记忆笔记</button></div></div>
      <div id="daily-body"></div>`;
    const body = $("#daily-body");
    const renderList = () => all.length ? all.map((d) => `<div class="item ${d.day === day ? "active" : ""}" data-day="${d.day}">
        <div class="t"><span>${esc(d.day)}</span><span class="small muted">${esc(dayLabel(d.day))}</span></div>
        <div class="s">${d.missing ? `<span class="badge plain">未生成</span> ${d.now_count} 条记录` : `${d.pushed_at ? `<span class="badge ok">已推送</span>` : `<span class="badge plain">未推送</span>`}${d.partial ? ` <span class="badge warn">不完整</span>` : ""}${d.stale ? ` <span class="badge warn">记录有更新</span>` : ""} ${d.utterance_count} 条`}</div></div>`).join("") : `<div class="empty">还没有日报</div>`;

    async function renderDigest() {
      body.innerHTML = `<div class="two"><div class="pane-list" id="dlist">${renderList()}</div><div id="dview"><div class="muted">加载中…</div></div></div>`;
      const v = $("#dview");
      const r = await api(`/api/daily/${day}`);
      if (stale(my)) return;
      const j = r.json || {};
      const running = state.status && state.status.jobs.daily.running;
      const head = `<div class="card-head"><div><h2>${esc(day)} <span class="muted small">${esc(dayLabel(day))}</span></h2>
          ${r.exists ? `<div class="small muted">生成于 ${fmtDT(r.created_at)}${r.model ? ` · ${esc(r.model)}` : ""} · ${r.utterance_count} 条记录${r.pushed_at ? ` · 已推送 ${fmtDT(r.pushed_at)}` : " · 未推送"}</div>` : `<div class="small muted">${r.now_count} 条记录，还没有生成日报</div>`}</div>
        <div class="row">${r.exists ? `<button class="btn sm" data-act="push">推送</button><button class="btn sm" data-act="regen" ${running ? "disabled" : ""}>${r.stale ? "更新日报" : "重新生成"}</button><button class="btn sm ghost" data-act="md">${mdView ? "结构视图" : "Markdown"}</button>` : `<button class="btn sm primary" data-act="regen" ${running || !r.now_count ? "disabled" : ""}>生成日报</button>`}</div></div>`;
      let content = "";
      if (!r.exists) content = `<div class="empty-state"><div class="big">${r.now_count ? "还没有生成" : "这天没有记录"}</div>${r.now_count ? "点右上角「生成日报」，大约需要几十秒到几分钟" : ""}</div>`;
      else if (mdView) content = `<div class="md">${md(r.markdown)}</div>`;
      else content = digestHtml(j, r);
      v.innerHTML = `<div class="card">${running ? `<div class="alert info"><span class="spin"></span>正在生成日报…</div>` : ""}${r.stale && r.exists ? `<div class="alert warn">生成之后记录又有更新（现在 ${r.now_count} 条，日报基于 ${r.utterance_count} 条）。</div>` : ""}${j.partial ? `<div class="alert warn">这份日报不完整：${(j.failed_blocks || []).length} 个时段分析失败，服务会自动重试。</div>` : ""}${head}${content}</div>`;
      v.onclick = async (e) => {
        const b = e.target.closest("[data-act]"); if (!b) return;
        const a = b.dataset.act;
        if (a === "md") { mdView = !mdView; renderDigest(); }
        else if (a === "regen") { await runJob("daily", { day, force: r.exists }); renderDigest(); }
        else if (a === "push") { await runJob("push", { day }); }
      };
    }
    function digestHtml(j, r) {
      const sec = (title, inner, cond = true) => (cond ? `<div class="sec"><h3>${esc(title)}</h3>${inner}</div>` : "");
      const ul = (items) => (items && items.length ? `<ul>${items.map((x) => `<li>${esc(typeof x === "string" ? x : x.text || x.name || JSON.stringify(x))}</li>`).join("")}</ul>` : `<div class="faint">（无）</div>`);
      const todos = (j.todos || []).map((t) => `<div class="todo-item"><span class="box"></span><span>${esc(t.text || t)}${t.owner ? ` <span class="muted small">负责人：${esc(t.owner)}</span>` : ""}${t.due ? ` <span class="muted small">截止：${esc(t.due)}</span>` : ""}</span></div>`).join("");
      const people = (j.people || []).map((p) => `<div class="todo-item" style="gap:6px"><span class="spk-c${spkColor(p.name)}" style="font-weight:600;white-space:nowrap">${esc(p.name)}</span><span class="muted">${esc(p.context || "")}</span></div>`).join("");
      const tl = (j.timeline || []).map((t) => `<div class="tl"><span class="t">${esc(t.time || "")}</span>${esc(t.event || "")}</div>`).join("");
      const stats = [j.utterance_count != null ? `${j.utterance_count} 条记录` : "", j.session_count ? `${j.session_count} 个时段` : "", j.time_range || "", j.speakers && j.speakers.length ? `说话人：${j.speakers.join("、")}` : ""].filter(Boolean).join(" · ");
      return `<div class="digest">${stats ? `<div class="digest-stats">${esc(stats)}</div>` : ""}
        ${sec("概览", `<div>${esc(j.overview || "").replace(/\n/g, "<br>")}</div>`)}
        <div class="grid grid-2">${sec("主要话题", ul(j.topics))}${sec("决定", ul(j.decisions))}</div>
        <div class="grid grid-2">${sec("待办", todos || `<div class="faint">（无）</div>`)}${sec("提到的人", people || `<div class="faint">（无）</div>`)}</div>
        <div class="grid grid-2">${sec("情绪", `<div>${esc(j.mood || "（无）")}</div>`)}${sec("未闭环", ul(j.open_loops))}</div>
        ${sec("时间线", `<div class="timeline">${tl}</div>`, tl.length > 0)}</div>`;
    }
    async function renderNotes() {
      let kind = "";
      const draw = async () => {
        const notes = await api(`/api/notes${kind ? `?kind=${kind}` : ""}`);
        const kinds = [["", "全部"], ["todo", "待办"], ["decision", "决定"], ["person", "人物"], ["topic", "话题"], ["fact", "事实"]];
        body.innerHTML = `<div class="card"><div class="card-head"><div class="row">${kinds.map(([k, l]) => `<span class="chip ${kind === k ? "on" : ""}" data-kind="${k}">${l}</span>`).join("")}</div><span class="small muted">${notes.length} 条 · 由每日分析自动整理，问答时会一并参考</span></div>
          <div class="notes-list">${notes.length ? notes.map((n) => `<div class="note"><span class="kind ${esc(n.kind)}">${esc(n.kind)}</span><div class="grow"><div><strong>${esc(n.title)}</strong> <span class="faint small">${esc(n.day_from || "")}${n.day_to && n.day_to !== n.day_from ? `~${esc(n.day_to)}` : ""}</span></div><div class="muted">${esc(n.content)}</div></div><button class="icon-btn" data-del="${n.id}" title="删除">✕</button></div>`).join("") : `<div class="empty-state">还没有笔记（生成日报后会出现）</div>`}</div></div>`;
        body.onclick = async (e) => {
          const c = e.target.closest("[data-kind]"); if (c) { kind = c.dataset.kind; return draw(); }
          const d = e.target.closest("[data-del]"); if (d && await confirmModal("删除笔记", "删除后问答不再参考这条笔记。", "删除", true)) { try { await api(`/api/notes/${d.dataset.del}`, { method: "DELETE" }); } catch (err) { toastErr(err); } draw(); }
        };
      };
      await draw();
    }
    main.onclick = (e) => {
      const t = e.target.closest("[data-tab]"); if (t) { tab = t.dataset.tab; location.hash = `#/daily/${tab === "notes" ? "notes" : day}`; return; }
      const it = e.target.closest("#dlist .item"); if (it) { day = it.dataset.day; location.hash = `#/daily/${day}`; }
    };
    if (tab === "notes") await renderNotes(); else await renderDigest();
    state.refresh = async () => { if (tab === "digest") await renderDigest(); };
    startPagePoll(async () => { if (tab === "digest" && state.status && state.status.jobs.daily.running) { await refreshStatus(); } }, 5000, my);
  }

  // ------------------------------------------------------------------ chat
  const chatState = { msgs: [], busy: false };
  async function pageChat(arg, my) {
    const st = state.status || await api("/api/status");
    const cfg = await api("/api/config").catch(() => null);
    if (stale(my)) return;
    const mem = cfg ? cfg.values.memory : { strategy: "prefetch", retrieval: "hybrid", query_expansion: false };
    main.innerHTML = `
      <div class="page-head"><div><h1>问答</h1><div class="sub">问任何记录里的事：时间、人物、决定、买了什么……答案会标注依据的时间和说话人</div></div>
        <div class="row"><button class="btn sm ghost" id="chat-clear">清空对话</button></div></div>
      ${!st.api_key ? `<div class="alert warn">没有配置 MiMo API key，无法问答。<span class="right"><a class="btn sm" href="#/settings">去设置</a></span></div>` : ""}
      <div class="chat"><div class="chat-msgs" id="msgs"></div>
        <div class="chat-input"><div class="chat-box"><textarea id="chat-q" rows="1" placeholder="输入问题，回车发送（Shift+回车换行）"></textarea><button class="btn primary" id="chat-send">发送</button></div>
          <div class="chat-opts"><span>策略</span><select id="opt-strategy"><option value="prefetch">prefetch · 先预检索再按需查（默认）</option><option value="rag">rag · 只用预检索，最快</option><option value="agentic">agentic · 模型自己反复检索</option></select>
            <span>检索</span><select id="opt-retrieval"><option value="hybrid">混合（关键词+向量）</option><option value="vector">向量</option><option value="fts">关键词</option></select>
            <label class="check"><input type="checkbox" id="opt-expand"> 查询扩展</label></div></div></div>`;
    $("#opt-strategy").value = mem.strategy || "prefetch";
    $("#opt-retrieval").value = mem.retrieval || "hybrid";
    $("#opt-expand").checked = !!mem.query_expansion;
    const box = $("#msgs");
    const q = $("#chat-q");
    const examples = ["昨天晚上聊了什么？", "最近有什么待办？", "上周三的会上定了什么？", "扫地机买的什么牌子？", "这几天谁来找过我？"];
    const renderMsgs = () => {
      if (!chatState.msgs.length) { box.innerHTML = `<div class="empty-state"><div class="big">试试这样问</div><div class="examples mt">${examples.map((x) => `<span class="chip" data-ex="${esc(x)}">${esc(x)}</span>`).join("")}</div></div>`; return; }
      box.innerHTML = chatState.msgs.map((m, i) => {
        if (m.role === "user") return `<div class="msg user">${esc(m.content)}</div>`;
        const trace = m.tools && m.tools.length ? `<details class="trace"><summary>检索过程 · ${m.tools.length} 次工具调用</summary>${m.tools.map((t) => `<div class="tc"><div>${esc(t.name)}(${esc(JSON.stringify(t.args))})</div><div class="r">${esc(t.result)}${t.lines > 20 ? "\n…" : ""}</div></div>`).join("")}</details>` : "";
        const meta = m.stats ? `<div class="meta"><span>${esc(m.stats.strategy)}</span><span>${m.stats.llm_calls} 次模型调用</span>${m.stats.tool_calls ? `<span>${m.stats.tool_calls} 次工具调用</span>` : ""}<span>${m.stats.seconds} 秒</span>${m.model ? `<span>${esc(m.model)}</span>` : ""}</div>` : "";
        const inner = m.pending ? `<div class="thinking"><span class="spin"></span>${m.tools && m.tools.length ? `正在查记录（${m.tools[m.tools.length - 1].name}）…` : "检索中…"}</div>` : m.error ? `<div class="alert danger" style="margin:0">${esc(m.error)}</div>` : `<div class="md">${md(m.content)}</div>`;
        return `<div class="msg bot"><div class="bubble">${inner}${trace}</div>${meta}</div>`;
      }).join("");
      box.lastElementChild && box.lastElementChild.scrollIntoView({ block: "end" });
    };
    chatState.render = renderMsgs;
    const render = () => { if (chatState.render) chatState.render(); };
    const send = async (text) => {
      text = (text || q.value).trim(); if (!text || chatState.busy) return;
      q.value = ""; q.style.height = "";
      const history = chatState.msgs.filter((m) => !m.error && !m.pending).map((m) => ({ role: m.role, content: m.content }));
      chatState.msgs.push({ role: "user", content: text });
      const bot = { role: "assistant", content: "", tools: [], pending: true };
      chatState.msgs.push(bot);
      chatState.busy = true; $("#chat-send").disabled = true;
      render();
      try {
        const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, history, strategy: $("#opt-strategy").value, retrieval: $("#opt-retrieval").value, expand: $("#opt-expand").checked }) });
        if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.error || res.statusText); }
        const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
        while (true) {
          const { value, done } = await reader.read(); if (done) break;
          buf += dec.decode(value, { stream: true });
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
            if (!line) continue;
            const ev = JSON.parse(line);
            if (ev.type === "start") bot.model = ev.model;
            else if (ev.type === "tool") { bot.tools.push(ev); render(); }
            else if (ev.type === "answer") { bot.content = ev.text || "（空回答）"; bot.stats = ev.stats; bot.pending = false; render(); }
            else if (ev.type === "error") { bot.error = ev.message; bot.pending = false; render(); }
          }
        }
        if (bot.pending) { bot.pending = false; bot.error = "连接中断"; }
      } catch (e) { bot.pending = false; bot.error = e.message; }
      chatState.busy = false;
      const sendBtn = $("#chat-send"); if (sendBtn) sendBtn.disabled = false;
      render();
      if (document.body.contains(q)) q.focus();
    };
    $("#chat-send").onclick = () => send();
    q.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
    q.oninput = () => { q.style.height = "auto"; q.style.height = Math.min(160, q.scrollHeight) + "px"; };
    $("#chat-clear").onclick = () => { chatState.msgs = []; renderMsgs(); };
    box.onclick = (e) => { const c = e.target.closest("[data-ex]"); if (c) send(c.dataset.ex); };
    if (chatState.busy) $("#chat-send").disabled = true;
    renderMsgs(); q.focus();
  }

  // ------------------------------------------------------------------ speakers
  let recState = null;
  function stopRecording() { if (recState) { try { recState.stop(true); } catch (e) { /* ignore */ } recState = null; } }
  // relation picker: <select> of the backend's RELATION_CHOICES plus "自定义…" that turns into a text box
  const CUSTOM_REL = "__custom__";
  function relationSelect(choices, value = "", { empty = "关系（可选）" } = {}) {
    const v = String(value || "");
    const custom = !!v && !choices.includes(v);
    return `<span class="rel-pick"><select class="input" data-rel-sel><option value="">${esc(empty)}</option>${choices.map((c) => `<option value="${esc(c)}" ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}<option value="${CUSTOM_REL}" ${custom ? "selected" : ""}>自定义…</option></select><input class="input ${custom ? "" : "hidden"}" data-rel-custom placeholder="填写关系" value="${esc(custom ? v : "")}"></span>`;
  }
  function wireRelationSelects(root) {
    for (const sel of $$("[data-rel-sel]", root)) {
      const custom = sel.parentElement.querySelector("[data-rel-custom]");
      sel.onchange = () => { const c = sel.value === CUSTOM_REL; custom.classList.toggle("hidden", !c); if (c) custom.focus(); };
    }
  }
  function relationValue(root) {
    const sel = root.querySelector("[data-rel-sel]"); if (!sel) return "";
    if (sel.value === CUSTOM_REL) { const c = root.querySelector("[data-rel-custom]"); return c ? c.value.trim() : ""; }
    return sel.value;
  }
  // "X 和我的关系" dialog; resolves true when something was saved
  function relationModal(spk, choices) {
    return new Promise((resolve) => {
      const auto = spk.relation_source === "auto";
      const initialRel = spk.relation || "";
      modal(`<h2>${esc(spk.name)} 和我的关系</h2>
        <div class="field"><label>关系</label>${relationSelect(choices, initialRel)}</div>
        <div class="field mt"><label>备注（可选）</label><input class="input" id="rm-note" value="${esc(auto ? "" : (spk.relation_note || ""))}" placeholder="例如：同一个项目组的 / 大学同学"></div>
        ${auto && spk.relation_note ? `<div class="help mt">自动推断的理由：${esc(spk.relation_note)}（不改动直接保存会保留这条推断）</div>` : ""}
        <div class="actions">${spk.relation ? `<button class="btn ghost" data-x="clear">清除关系</button>` : ""}<button class="btn" data-x="cancel">取消</button><button class="btn primary" data-x="ok">保存</button></div>`,
        (el, close) => {
          wireRelationSelects(el);
          const patch = (body) => api(`/api/speakers/${spk.id}`, { method: "PATCH", json: body });
          const noteEl = el.querySelector("#rm-note");
          let noteTouched = false;                                   // an untouched note box must not overwrite an auto-inferred reason
          noteEl.addEventListener("input", () => { noteTouched = true; });
          el.querySelector('[data-x="cancel"]').onclick = () => { close(); resolve(false); };
          el.querySelector('[data-x="ok"]').onclick = async () => {
            const rel = relationValue(el); const note = noteEl.value.trim();
            if (rel === initialRel && !noteTouched) { close(); resolve(false); return; }   // nothing changed: no PATCH (keeps source='auto' + its reason)
            try {
              if (rel) await patch({ relation: rel, relation_note: note });
              else { await patch({ relation: "" }); if (note) await patch({ relation_note: note }); }   // "" clears relation + note; a note alone is kept
              close(); resolve(true);
            } catch (e) { toastErr(e); }
          };
          const cl = el.querySelector('[data-x="clear"]'); if (cl) cl.onclick = async () => { try { await patch({ relation: "" }); close(); resolve(true); } catch (e) { toastErr(e); } };
        });
    });
  }

  const voicesAutoScanned = {};            // profile name -> true: the speakers page submits the "voices" scan job at most once per profile per session
  const currentProfile = () => (state.status && state.status.profile && state.status.profile.name) || "";
  const voiceDrafts = {};                  // what the user typed into a voice card, kept across a re-render (keyed by group key)
  const voiceLetter = (i) => (i < 26 ? String.fromCharCode(65 + i) : String(i + 1));
  function stashVoiceDrafts() {
    for (const c of $$(".voice-card")) {
      const f = c.querySelector(".voice-form"); if (!f) continue;
      const d = { name: f.querySelector("[data-vname]").value, sel: f.querySelector("[data-rel-sel]").value, custom: f.querySelector("[data-rel-custom]").value, self: f.querySelector("[data-vself]").checked };
      if (d.name || d.sel || d.self) voiceDrafts[c.dataset.key] = d;
    }
  }

  async function pageSpeakers(arg, my) {
    const data = await api("/api/speakers");
    if (stale(my)) return;
    state.speakers = data.speakers;
    const choices = data.relation_choices || [];
    const me = data.speakers.find((s) => s.id === data.self_id) || null;
    const names = data.speakers.map((s) => s.name);
    const people = [...data.speakers].sort((a, b) => (b.is_self ? 1 : 0) - (a.is_self ? 1 : 0));
    const relationChip = (s) => {
      if (s.is_self) return "";
      if (!s.relation) return `<button class="relation-chip empty" data-act="relation" title="标注 TA 和你的关系">＋ 关系</button>`;
      const auto = s.relation_source === "auto";
      return `<button class="relation-chip ${auto ? "auto" : ""}" data-act="relation" title="${esc(s.relation_note || (auto ? "模型从记录里推断的，点击修改" : "点击修改"))}">${esc(s.relation)}${auto ? `<small>自动</small>` : ""}</button>`;
    };
    const cards = people.map((s) => `<div class="card spk-card ${s.is_self ? "me" : ""}" data-sid="${s.id}">
        <div class="row"><span class="avatar bg-c${spkColor(s.name)}">${esc(s.name.slice(0, 1))}</span><div class="grow"><div class="name">${esc(s.name)}${s.is_self ? `<span class="badge me">我</span>` : ""}${relationChip(s)}</div><div class="small muted">${s.samples} 个声纹样本 · ${s.duration_s.toFixed(0)} 秒语音 · ${s.utterances} 条语句${s.notes ? ` · ${esc(s.notes)}` : ""}</div></div></div>
        <div class="row">${s.is_self ? `<button class="btn sm ghost" data-act="unself" title="不再把 TA 当作本人">取消本人</button>` : `<button class="btn sm" data-act="self" title="日报、推送和问答都站在本人的视角；记录里本人的话就是“我”说的">${me ? "改为本人" : "设为本人"}</button>`}<button class="btn sm" data-act="add" data-name="${esc(s.name)}">补充样本</button><button class="btn sm" data-act="samples">样本列表</button><button class="btn sm" data-act="rename">重命名</button><button class="btn sm ghost" data-act="notes">备注</button><button class="btn sm danger right" data-act="delete">删除</button></div>
        <div class="samples hidden"></div></div>`).join("");
    main.innerHTML = `
      <div class="page-head"><div><h1>说话人</h1><div class="sub">不用单独录样本：下面列出记录里还没认出来的声音，听一下是谁、填上名字就行；登记过的人会自动识别。</div></div>
        <div class="row"><button class="btn" id="infer-rel" title="让模型从记录里的对话推断每个人和你的关系（手动标过的不会改）">自动推断关系</button><button class="btn" id="relabel-all" title="用当前声纹库重新识别所有已转写的记录（不重新转写）">重新识别全部记录</button><button class="btn" id="calib">校准阈值</button><button class="btn ghost" id="add-new">用麦克风 / 文件补充样本</button></div></div>
      <datalist id="spk-names">${names.map((n) => `<option value="${esc(n)}">`).join("")}</datalist>
      <div id="enroll-panel" class="card hidden"></div>
      <div id="voices-box"><div class="card"><div class="thinking"><span class="spin"></span>正在整理未识别的声音…</div></div></div>
      ${people.length ? `<div class="sec-head mt"><h2>已登记的人</h2><span class="small muted">${me ? `本人：${esc(me.name)}` : "还没有标出哪位是本人（卡片上点「设为本人」）"}</span></div><div class="grid grid-2">${cards}</div>` : ""}
      <div class="card mt"><div class="card-head"><h2>提示</h2></div><ul class="muted small" style="margin:0;padding-left:18px"><li>给一组声音起名后，会从这些录音里自动提取声纹样本，之后这个人在新录音里会被自动识别；认错的句子在「记录」页悬停 → 👤 标记即可。</li><li>把自己标为「本人」后，日报和问答会站在你的视角；关系可以手动标，也可以点「自动推断关系」让模型从对话里推断。</li><li>当前识别阈值 ${data.identify_threshold}；样本多了之后「校准阈值」会给出建议值。</li></ul></div>`;

    // ---- unidentified voices
    const voicesBox = $("#voices-box");
    let voices = null;
    const voiceCard = (g, i) => {
      const days = g.days || [];
      const range = days.length > 1 ? `${days[0].slice(5)} ~ ${days[days.length - 1].slice(5)}` : (days[0] || "").slice(5);
      const close = g.closest && g.closest.score >= 0.3 ? g.closest : null;
      const samples = (g.samples || []).map((s) => `<div class="voice-sample ${s.audio ? "" : "noaudio"}" data-uid="${s.utterance_id}">
          <span class="time">${esc(String(s.day || "").slice(5))} ${esc(s.time || "")}</span><span class="text" title="${esc(s.text)}">${esc(s.text) || "<i>（没有文字）</i>"}</span><span class="dur">${Number(s.seconds || 0).toFixed(1)}s</span>
          ${s.audio ? `<button class="icon-btn" data-vplay="${s.utterance_id}" title="播放这句">▶</button>` : `<span class="icon-btn faint" title="音频已删除，无法播放">▶</span>`}</div>`).join("");
      return `<div class="card voice-card" data-key="${esc(g.key)}">
        <div class="card-head"><div><h2>声音 ${voiceLetter(i)}</h2><div class="small muted">${g.utterances} 句 · ${Math.round(g.speech_s)} 秒 · ${g.n_days} 天${range ? `（${esc(range)}）` : ""}${g.segments > 1 ? ` · ${g.segments} 段录音` : ""}</div></div>
          ${close ? `<button class="chip" data-vfill="${esc(close.name)}" title="与已登记声纹的相似度 ${Number(close.score).toFixed(2)}，点一下填入名字（不一定对，先听一听）">可能是 ${esc(close.name)}（${Number(close.score).toFixed(2)}）</button>` : ""}</div>
        <div class="voice-samples">${samples || `<div class="muted small">（没有示例句）</div>`}</div>
        <div class="voice-player"></div>
        <div class="row voice-form"><input class="input" data-vname list="spk-names" placeholder="这是谁？">${relationSelect(choices)}<label class="check small"><input type="checkbox" data-vself> ${me ? "这其实是我" : "这是我自己"}</label><button class="btn primary" data-vok>就是 TA</button></div></div>`;
    };
    const renderVoices = () => {
      const v = voices; if (!v || stale(my)) return;
      const job = state.status && state.status.jobs && state.status.jobs.voices;
      const scanning = !!(v.scanning || (job && job.running));
      const groups = v.groups || [];
      const readOnly = !!(state.status && state.status.service && state.status.service.read_only);
      if (!groups.length && !data.speakers.length && !scanning && !v.missing_segments && !v.unknown_utterances) {
        voicesBox.innerHTML = `<div class="empty-state card"><div class="big">还没有记录</div>录一会儿或导入音频后，这里会列出听到的声音。</div>`;
        return;
      }
      const notes = [];
      if (scanning) notes.push(`<div class="alert info"><span class="spin"></span>正在分析 ${v.missing_segments ? `${v.missing_segments} 段录音` : "录音"}…完成后这里会自动更新。</div>`);
      else if (v.missing_segments) notes.push(`<div class="alert warn">还有 ${v.missing_segments} 段录音没有分析过声音${readOnly ? "（只读模式下无法扫描，请在运行服务的那个界面操作）" : ""}。<span class="right"><button class="btn sm" data-vact="scan" ${readOnly ? "disabled" : ""}>扫描</button></span></div>`);
      if (!groups.length && !scanning && !v.missing_segments && v.unknown_utterances) notes.push(`<div class="small muted mb">有 ${v.unknown_utterances} 句未识别，但它们的音频已不在，无法按声音分组；可以在「记录」页逐句标记。</div>`);
      voicesBox.innerHTML = `<div class="sec-head"><h2>未识别的声音</h2><span class="small muted">${v.unknown_utterances ? `${v.unknown_utterances} 句还没认出是谁说的` : ""}</span></div>${notes.join("")}
        ${groups.length ? `<div class="grid grid-2">${groups.map(voiceCard).join("")}</div>` : `<div class="card muted">没有待确认的声音</div>`}`;
      wireRelationSelects(voicesBox);
      for (const c of $$(".voice-card", voicesBox)) {
        const d = voiceDrafts[c.dataset.key]; if (!d) continue;
        delete voiceDrafts[c.dataset.key];
        const f = c.querySelector(".voice-form");
        f.querySelector("[data-vname]").value = d.name;
        const sel = f.querySelector("[data-rel-sel]"); sel.value = d.sel;
        const custom = f.querySelector("[data-rel-custom]");
        custom.classList.toggle("hidden", d.sel !== CUSTOM_REL);     // not sel.onchange(): that would focus() the custom box and steal focus from another card
        custom.value = d.custom;
        f.querySelector("[data-vself]").checked = d.self;
      }
    };
    const loadVoices = async () => {
      try { voices = await api("/api/voices"); }
      catch (e) { if (!stale(my)) voicesBox.innerHTML = `<div class="alert danger">${esc(e.message)}</div>`; return; }
      if (stale(my)) return;
      if (!state.status) { await refreshStatus(); if (stale(my)) return; }    // need profile name + read_only before deciding to auto-scan
      const readOnly = !!(state.status && state.status.service && state.status.service.read_only);
      const prof = currentProfile();
      if (voices.missing_segments > 0 && !voices.scanning && !readOnly && !voicesAutoScanned[prof]) {
        voicesAutoScanned[prof] = true;
        try { await api("/api/jobs/voices", { method: "POST", json: {} }); voices.scanning = true; refreshStatus().catch(() => {}); }
        catch (e) { toast(`无法自动分析录音：${e.message}`, "err", 6000); }
      }
      renderVoices();
    };
    const playVoiceSample = (card, uid) => {
      $$(".voice-sample.playing").forEach((x) => x.classList.remove("playing"));
      $$(".voice-player").forEach((x) => { x.innerHTML = ""; });
      const row = card.querySelector(`.voice-sample[data-uid="${uid}"]`); if (row) row.classList.add("playing");
      const p = card.querySelector(".voice-player");
      p.innerHTML = `<audio controls autoplay src="/api/audio/utterance/${uid}?pad=0.3"></audio><button class="icon-btn" title="关闭">✕</button>`;
      p.querySelector("button").onclick = () => { p.innerHTML = ""; if (row) row.classList.remove("playing"); };
      p.querySelector("audio").onerror = () => toast("音频无法播放（文件可能已删除）", "err");
    };
    const nameVoice = async (card) => {
      const g = voices && (voices.groups || []).find((x) => x.key === card.dataset.key); if (!g) return;
      const f = card.querySelector(".voice-form");
      const inp = f.querySelector("[data-vname]"); const name = inp.value.trim();
      if (!name) { toast("先填上这是谁", "err"); inp.focus(); return; }
      const isSelf = f.querySelector("[data-vself]").checked;
      if (isSelf && me && me.name !== name && !(await confirmModal("改为本人", `目前本人是「${me.name}」，改成「${name}」？日报和问答会改站在 ${name} 的视角。`, "改为本人"))) return;
      const rel = relationValue(f);
      const btn = f.querySelector("[data-vok]"); btn.disabled = true; btn.innerHTML = `<span class="spin"></span>提取声纹…`;
      try {
        const r = await api("/api/voices/name", { method: "POST", json: { members: g.members, name, is_self: isSelf, relation: rel || null, auto_relabel: true } });
        toast(`已标记为 ${name}：${r.utterances_assigned} 句，补充 ${r.samples_added} 个声纹样本${r.relabel_started ? "；正在用新声纹重新识别其他记录" : ""}`, "ok", 7000);
        stashVoiceDrafts();                                  // keep what was typed into the OTHER cards across the re-render
        delete voiceDrafts[card.dataset.key];                // (after the stash, or the named card's draft would be re-added)
        if (isSelf) await refreshStatus();                   // 我 badge / self_name are read from state.status by the next render
        await pageSpeakers(arg, my);
      } catch (err) { toastErr(err); btn.disabled = false; btn.textContent = "就是 TA"; }
    };

    // a finished job re-renders the page, but only for jobs that change what this page shows, and never while the user is
    // typing into a voice card / relation form (a re-render would replace the input under the cursor); a skipped refresh runs
    // once focus leaves the form.
    const REFRESH_JOBS = new Set(["voices", "relabel", "relations", "process"]);
    const editing = () => {
      const a = document.activeElement;
      if (!a || !main.contains(a)) return false;
      const vb = $("#voices-box");
      return (vb && vb.contains(a)) || a.matches("input,textarea,select");
    };
    let pendingRefresh = false;
    const doRefresh = async () => {
      if (stale(my)) return;
      if (recState || !$("#enroll-panel").classList.contains("hidden") || $("#modal-root").children.length) return;
      if (editing()) { pendingRefresh = true; return; }
      pendingRefresh = false;
      stashVoiceDrafts();
      await pageSpeakers(arg, my);
    };
    state.refresh = async (job) => {
      if (job !== undefined && !REFRESH_JOBS.has(job)) return;        // daily / push / cleanup / reindex do not touch speakers or voices
      await doRefresh();
    };
    main.onfocusout = () => { setTimeout(() => { if (!stale(my) && pendingRefresh && !editing()) doRefresh().catch(() => {}); }, 0); };
    $("#add-new").onclick = () => enrollPanel("");
    $("#infer-rel").onclick = async () => {
      if (state.status && !state.status.api_key) { toast("没有配置 MiMo API key，无法推断关系", "err"); return; }
      if (await confirmModal("自动推断关系", "让模型根据记录里的对话推断每个人和你的关系，只处理语句足够多、还没手动标过的人；结果会标成「自动」，可以随时改。")) runJob("relations");
    };
    $("#relabel-all").onclick = async () => { if (await confirmModal("重新识别全部记录", "用当前登记的声纹重新识别所有已转写的语句，不重新转写、不花钱；记录多的话要几分钟。手动标过的句子不会改。")) runJob("relabel"); };
    $("#calib").onclick = async () => {
      try {
        const r = await api("/api/speakers/calibrate", { method: "POST" });
        const row = (l, s) => `<tr><td>${l}</td><td class="num">${s ? s.n : 0}</td><td class="num">${s ? s.min.toFixed(3) : "-"}</td><td class="num">${s ? s.mean.toFixed(3) : "-"}</td><td class="num">${s ? s.max.toFixed(3) : "-"}</td></tr>`;
        modal(`<h2>识别阈值校准</h2><table class="tbl"><tr><th>类型</th><th class="num">对数</th><th class="num">最小</th><th class="num">平均</th><th class="num">最大</th></tr>${row("同一人", r.same)}${row("不同人", r.cross)}</table>
          <div class="mt">${r.n_embeddings} 个样本 / ${r.n_speakers} 人 · 当前阈值 <strong>${r.current}</strong>${r.suggested != null ? ` · 建议 <strong>${r.suggested.toFixed(2)}</strong>` : ""}</div>
          ${!r.same ? `<div class="alert warn mt">没有同一人的多个样本，无法估计同人分数；请每人至少 2 个样本。</div>` : ""}${!r.cross ? `<div class="alert warn mt">只有一个人，无法估计异人分数。</div>` : ""}${r.overlap ? `<div class="alert warn mt">同人与异人分数区间重叠，会有误判；建议增加样本或检查录音质量。</div>` : ""}
          <div class="actions">${r.suggested != null && Math.abs(r.suggested - r.current) > 0.005 ? `<button class="btn primary" data-x="apply">采用建议值 ${r.suggested.toFixed(2)}</button>` : ""}<button class="btn" data-x="close">关闭</button></div>`,
          (el, close) => {
            el.querySelector('[data-x="close"]').onclick = close;
            const ap = el.querySelector('[data-x="apply"]');
            if (ap) ap.onclick = async () => { try { await api("/api/config", { method: "PUT", json: { speakers: { identify_threshold: Number(r.suggested.toFixed(2)) } } }); toast("已更新 identify_threshold", "ok"); close(); route(); } catch (e) { toastErr(e); } };
          });
      } catch (e) { toastErr(e); }
    };
    main.onclick = async (e) => {
      const vp = e.target.closest("[data-vplay]");
      if (vp) { playVoiceSample(vp.closest(".voice-card"), Number(vp.dataset.vplay)); return; }
      const va = e.target.closest("[data-vact]");
      if (va) { if (va.dataset.vact === "scan" && !va.disabled) { voicesAutoScanned[currentProfile()] = true; await runJob("voices"); await loadVoices(); } return; }
      const vok = e.target.closest("[data-vok]");
      if (vok) { await nameVoice(vok.closest(".voice-card")); return; }
      const vf = e.target.closest("[data-vfill]");
      if (vf) { const inp = vf.closest(".voice-card").querySelector("[data-vname]"); inp.value = vf.dataset.vfill; inp.dispatchEvent(new Event("input")); inp.focus(); return; }
      const b = e.target.closest("[data-act]"); if (!b) return;
      const card = b.closest("[data-sid]"); if (!card) return;
      const sid = Number(card.dataset.sid); const spk = data.speakers.find((s) => s.id === sid);
      const a = b.dataset.act;
      if (a === "add") enrollPanel(spk.name);
      else if (a === "self") {
        const text = me ? `把本人从「${me.name}」改为「${spk.name}」？日报、推送和问答都会站在本人的视角，记录里本人的话就是“我”说的。` : `把「${spk.name}」设为本人（你自己）？日报、推送和问答都会站在本人的视角，记录里本人的话就是“我”说的。`;
        if (await confirmModal(me ? "改为本人" : "设为本人", text, me ? "改为本人" : "设为本人")) { try { await api(`/api/speakers/${sid}`, { method: "PATCH", json: { is_self: true } }); toast(`本人：${spk.name}`, "ok"); await refreshStatus(); route(); } catch (err) { toastErr(err); } }
      }
      else if (a === "unself") { if (await confirmModal("取消本人", `不再把「${spk.name}」当作本人；之后可以再设。`)) { try { await api(`/api/speakers/${sid}`, { method: "PATCH", json: { is_self: false } }); await refreshStatus(); route(); } catch (err) { toastErr(err); } } }
      else if (a === "relation") { if (await relationModal(spk, choices)) { toast("已保存关系", "ok"); route(); } }
      else if (a === "rename") { const n = await promptModal("重命名", { label: "新名字", value: spk.name }); if (n && n !== spk.name) { try { await api(`/api/speakers/${sid}`, { method: "PATCH", json: { name: n } }); toast("已重命名", "ok"); route(); } catch (err) { toastErr(err); } } }
      else if (a === "notes") { const n = await promptModal("备注", { label: "例如：项目组的同事、楼下邻居", value: spk.notes }); if (n !== null) { try { await api(`/api/speakers/${sid}`, { method: "PATCH", json: { notes: n } }); route(); } catch (err) { toastErr(err); } } }
      else if (a === "delete") { if (await confirmModal(`删除 ${spk.name}`, `删除声纹和 ${spk.utterances} 条语句上的名字（语句会变成「${data.unknown_label}」，文字不会丢）。`, "删除", true)) { try { await api(`/api/speakers/${sid}`, { method: "DELETE" }); toast("已删除"); route(); } catch (err) { toastErr(err); } } }
      else if (a === "samples") {
        const box = card.querySelector(".samples");
        if (!box.classList.contains("hidden")) { box.classList.add("hidden"); return; }
        const rows = await api(`/api/speakers/${sid}/samples`);
        box.classList.remove("hidden");
        box.innerHTML = `<table class="tbl small">${rows.map((r) => `<tr><td>#${r.id}</td><td>${esc(r.file)}</td><td class="num">${(r.duration_s || 0).toFixed(1)}s</td><td>${fmtDT(r.created_at)}</td><td class="nowrap">${r.audio ? `<button class="icon-btn" data-play="${r.id}" title="试听">▶</button>` : ""}<button class="icon-btn" data-delsample="${r.id}" title="删除样本">✕</button></td></tr>`).join("")}</table><div class="sample-player"></div>`;
        box.onclick = async (ev) => {
          const p = ev.target.closest("[data-play]"); if (p) { box.querySelector(".sample-player").innerHTML = `<audio controls autoplay src="/api/speakers/samples/${p.dataset.play}/audio"></audio>`; return; }
          const d = ev.target.closest("[data-delsample]"); if (d && await confirmModal("删除样本", "删除这个声纹样本。", "删除", true)) { try { await api(`/api/speakers/samples/${d.dataset.delsample}`, { method: "DELETE" }); route(); } catch (err) { toastErr(err); } }
        };
      }
    };
    await loadVoices();
  }
  function enrollPanel(name) {
    stopRecording();
    const panel = $("#enroll-panel");
    panel.classList.remove("hidden");
    const names = state.speakers.map((s) => s.name);
    panel.innerHTML = `<div class="card-head"><div><h2>${name ? `为 ${esc(name)} 补充样本` : "用麦克风 / 文件补充样本"}</h2><div class="small muted">通常不需要：给上面的「未识别的声音」起名就会自动取样本；这里是识别不准时的补充手段。</div></div><button class="btn sm ghost" id="enroll-close">关闭</button></div>
      <div class="row mb"><div class="field" style="width:220px"><label>姓名</label><input class="input" id="en-name" list="en-list" value="${esc(name)}" placeholder="例如：我 / 张三"><datalist id="en-list">${names.map((n) => `<option value="${esc(n)}">`).join("")}</datalist></div></div>
      <div class="grid grid-2">
        <div><h3 class="mb">用麦克风录一段</h3><div class="row"><button class="btn rec" id="rec-btn">● 开始录音</button><span class="rec-timer" id="rec-timer">0:00</span><div class="level"><i id="rec-level"></i></div></div>
          <div class="small muted mt">用平时说话的音量自然地说 15–30 秒（念一段文字就行），然后点停止。浏览器第一次会询问麦克风权限。</div><div id="rec-result" class="mt"></div></div>
        <div><h3 class="mb">或上传录音文件</h3><input type="file" id="en-file" accept="audio/*,.m4a,.webm,.amr" multiple><div class="small muted mt">wav / mp3 / m4a / 微信语音等都可以；尽量只有这个人说话的片段，10–60 秒。</div><div id="file-result" class="mt"></div></div>
      </div>`;
    $("#enroll-close").onclick = () => { stopRecording(); panel.classList.add("hidden"); };
    const getName = () => { const n = $("#en-name").value.trim(); if (!n) { toast("先填姓名", "err"); $("#en-name").focus(); } return n; };
    const upload = async (blob, filename, where) => {
      const n = getName(); if (!n) return false;
      where.innerHTML = `<div class="thinking"><span class="spin"></span>正在提取声纹…</div>`;
      try {
        const r = await api(`/api/speakers/${encodeURIComponent(n)}/samples?filename=${encodeURIComponent(filename)}`, { method: "POST", body: blob });
        where.innerHTML = `<div class="alert info" style="margin:0">已登记 ${esc(n)}：这段有 ${r.speech_s} 秒语音，现在共 ${r.samples} 个样本。</div>`;
        toast(`已注册 ${n}`, "ok");
        setTimeout(() => { if (state.page === "speakers") pageSpeakers(undefined, routeGen).then(() => enrollPanel(n)); }, 1500);
        return true;
      } catch (e) { where.innerHTML = `<div class="alert danger" style="margin:0">${esc(e.message)}</div>`; return false; }
    };
    $("#en-file").onchange = async (e) => {
      for (const f of e.target.files) { const ok = await upload(f, f.name, $("#file-result")); if (!ok) break; }
      e.target.value = "";
    };
    $("#rec-btn").onclick = async () => {
      if (recState) { recState.stop(false); return; }
      if (!getName()) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { toast("这个浏览器不支持录音，请用上传文件", "err"); return; }
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } }); }
      catch (e) { toast("无法打开麦克风：" + e.message, "err", 6000); return; }
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const chunks = [];
      rec.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = ctx.createMediaStreamSource(stream); const an = ctx.createAnalyser(); an.fftSize = 512; src.connect(an);
      const buf = new Uint8Array(an.frequencyBinCount);
      const t0 = Date.now();
      const timer = setInterval(() => {
        an.getByteTimeDomainData(buf); let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
        const lv = $("#rec-level"); if (lv) lv.style.width = `${Math.min(100, peak * 160)}%`;
        const s = Math.floor((Date.now() - t0) / 1000); const tm = $("#rec-timer"); if (tm) tm.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
        if (s >= 90) recState && recState.stop(false);
      }, 100);
      const btn = $("#rec-btn"); btn.textContent = "■ 停止并注册"; btn.classList.remove("rec");
      recState = { stop: (abort) => {
        clearInterval(timer); try { ctx.close(); } catch (e) { /* ignore */ }
        rec.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop());
          const b = $("#rec-btn"); if (b) { b.textContent = "● 开始录音"; b.classList.add("rec"); }
          const lv = $("#rec-level"); if (lv) lv.style.width = "0";
          recState = null;
          if (abort) return;
          const secs = (Date.now() - t0) / 1000;
          if (secs < 3) { toast("录得太短了，至少 5 秒", "err"); return; }
          const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
          const ext = (rec.mimeType || "audio/webm").includes("ogg") ? "ogg" : (rec.mimeType || "").includes("mp4") ? "mp4" : "webm";
          await upload(blob, `record.${ext}`, $("#rec-result"));
        };
        if (rec.state !== "inactive") rec.stop(); else rec.onstop();
      } };
      rec.start(250);
    };
    $("#en-name").focus();
  }

  // ------------------------------------------------------------------ settings
  async function pageSettings(arg, my) {
    const [cfg, devices, sysDevices] = await Promise.all([api("/api/config"), api("/api/devices"), api("/api/devices/system").catch(() => ({ available: false, devices: [] }))]);
    if (stale(my)) return;
    // device names, not indices: PortAudio renumbers devices when something is plugged in
    const sysOpts = [["", "默认输出设备"], ...sysDevices.devices.map((d) => [d.name, `${d.name}${d.is_default ? "（默认）" : ""}`])];
    { const cur = cfg.values.audio.system_device; if (cur != null && cur !== "" && !sysOpts.some(([k]) => k === String(cur))) sysOpts.push([String(cur), `${cur}（当前配置，未在列表中）`]); }
    const v = cfg.values; const sec = cfg.secrets;
    const secretField = (path, label, help = "", placeholder = "") => {
      const s = sec[path] || { set: false, hint: "" };
      return `<div class="field"><label>${esc(label)} ${s.set ? `<span class="badge ok">已设置 ${esc(s.hint)}</span>` : `<span class="badge plain">未设置</span>`}</label><input class="input" type="password" data-path="${path}" placeholder="${esc(placeholder || (s.set ? "留空表示不改" : ""))}" autocomplete="new-password">${help ? `<div class="help">${esc(help)}</div>` : ""}</div>`;
    };
    const text = (path, label, val, help = "", type = "text", attrs = "") => `<div class="field"><label>${esc(label)}</label><input class="input" type="${type}" data-path="${path}" value="${esc(val ?? "")}" ${attrs}>${help ? `<div class="help">${esc(help)}</div>` : ""}</div>`;
    const num = (path, label, val, help = "", step = "1", min = "") => text(path, label, val, help, "number", `step="${step}" ${min !== "" ? `min="${min}"` : ""}`);
    const sel = (path, label, val, opts, help = "") => `<div class="field"><label>${esc(label)}</label><select class="input" data-path="${path}">${opts.map(([k, l]) => `<option value="${esc(k)}" ${String(val) === String(k) ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>${help ? `<div class="help">${esc(help)}</div>` : ""}</div>`;
    const bool = (path, label, val, help = "") => `<div class="field"><label>&nbsp;</label><label class="check"><input type="checkbox" data-path="${path}" ${val ? "checked" : ""}> ${esc(label)}</label>${help ? `<div class="help">${esc(help)}</div>` : ""}</div>`;
    const devOpts = [["", "系统默认输入"], ...devices.map((d) => [String(d.index), `${d.index}: ${d.name} (${d.hostapi})`])];
    const devVal = v.audio.device == null ? "" : String(v.audio.device);
    if (devVal && !devOpts.some(([k]) => k === devVal)) devOpts.push([devVal, `${devVal}（当前配置，未在列表中）`]);
    const channels = new Set(v.push.channels || []);
    const chan = (k, l) => `<label class="check"><input type="checkbox" data-chan="${k}" ${channels.has(k) ? "checked" : ""}> ${l}</label>`;
    main.innerHTML = `
      <div class="page-head"><div><h1>设置</h1><div class="sub">档案「${esc(cfg.profile_label)}」${cfg.base_config_path ? `：这里保存的值写入 <code>${esc(cfg.path)}</code>，没改过的项继承根目录 <code>config.yaml</code>（API key 等共享设置建议放在那里，所有档案通用）` : `：保存后写入 <code>${esc(cfg.path)}</code>`}；保存后立即生效（录音设置会自动重启录音）${cfg.env_api_key ? "。<strong>当前 API key 来自环境变量 MIMO_API_KEY，文件里的会被忽略</strong>" : ""}</div></div></div>
      <form id="cfg-form" onsubmit="return false">
        <div class="card"><div class="card-head"><h2>MiMo API</h2><div class="row"><button class="btn sm" id="test-api" type="button">测试连接</button></div></div>
          <div class="form-grid">${secretField("mimo.api_key", "API key", "在 MiMo 开放平台申请；也可以用环境变量 MIMO_API_KEY", "sk-…")}
            ${text("mimo.base_url", "接口地址", v.mimo.base_url)}${text("mimo.chat_model", "纠错 / 问答模型", v.mimo.chat_model, "mimo-v2.6-flash 便宜；mimo-v2.6-pro 更强")}
            ${text("mimo.analysis_model", "日报模型", v.mimo.analysis_model, "留空 = 同上")}${text("mimo.asr_model", "语音转写模型", v.mimo.asr_model)}
            ${sel("mimo.asr_language", "转写语言", v.mimo.asr_language, [["auto", "自动"], ["zh", "中文"], ["en", "英文"]])}</div></div>
        <div class="card"><div class="card-head"><h2>录音</h2></div>
          <div class="form-grid">${sel("audio.device", "麦克风", devVal, devOpts, "拔插设备后重新打开这页会刷新列表")}
            ${num("audio.vad_threshold", "语音检测阈值", v.audio.vad_threshold, "0–1，越高越保守（环境嘈杂时调高）", "0.05", "0")}
            ${num("audio.min_silence_s", "结束片段的静音秒数", v.audio.min_silence_s, "", "0.5", "0.5")}${num("audio.max_segment_s", "单片段最长秒数", v.audio.max_segment_s, "", "30", "30")}
            ${num("audio.gain", "输入增益", v.audio.gain, "录得太小声就调大，如 2", "0.1", "0.1")}</div>
          <div class="mt" style="border-top:1px solid var(--border);padding-top:12px"><h3 class="mb">系统声音（会议 / 通话对方的声音）</h3>
          <div class="form-grid">${bool("audio.system_audio", "同时录系统声音", v.audio.system_audio, "把扬声器 / 耳机里放出来的声音和麦克风混在一起录；总览页也有开关。用耳机效果最好，外放时对方的声音会被录两次")}
            ${sel("audio.system_device", "录哪个输出设备", v.audio.system_device == null ? "" : String(v.audio.system_device), sysOpts, sysDevices.available ? "" : "当前环境没有可用的系统声音源（需要 Windows 和 PyAudioWPatch）")}
            ${num("audio.system_gain", "系统声音音量倍数", v.audio.system_gain ?? 1, "对方声音太小就调大", "0.1", "0.1")}</div></div></div>
        <div class="card"><div class="card-head"><h2>存储</h2></div>
          <div class="form-grid">${num("storage.keep_audio_days", "音频保留天数", v.storage.keep_audio_days, "到期删除已转写的音频；0 = 永久。文字永不删除", "1", "0")}
            ${num("storage.max_disk_gb", "音频占用上限 (GB)", v.storage.max_disk_gb, "超过后从最旧的已转写音频开始删", "1", "0")}${bool("storage.cleanup_enabled", "定时清理无语音片段", v.storage.cleanup_enabled)}</div></div>
        <div class="card"><div class="card-head"><h2>说话人识别</h2></div>
          <div class="form-grid">${num("speakers.identify_threshold", "识别阈值", v.speakers.identify_threshold, "相似度 ≥ 此值才认定为注册过的人；说话人页可校准", "0.01", "0")}
            ${num("speakers.cluster_threshold", "聚类阈值", v.speakers.cluster_threshold, "越小分出的人越多", "0.05", "0")}${text("speakers.unknown_label", "未识别时显示", v.speakers.unknown_label)}
            ${num("speakers.voice_group_threshold", "未识别声音分组阈值", v.speakers.voice_group_threshold ?? 0.55, "0–1，越高分得越细（同一个人可能被分成几组）", "0.01", "0")}
            ${bool("speakers.infer_relations", "日报后自动推断关系", v.speakers.infer_relations ?? true, "让模型从记录里的对话推断每个人和本人的关系；手动标的不会被改")}
            ${num("speakers.relation_min_utterances", "推断关系至少需要的语句数", v.speakers.relation_min_utterances ?? 15, "语句太少的人先不推断", "1", "1")}
            ${num("speakers.relation_min_confidence", "推断关系的置信度门槛", v.speakers.relation_min_confidence ?? 0.6, "0–1，低于此值不写入", "0.05", "0")}</div></div>
        <div class="card"><div class="card-head"><h2>转写与纠错</h2></div>
          <div class="form-grid">${num("pipeline.process_interval_min", "自动转写间隔（分钟）", v.pipeline.process_interval_min, "", "1", "1")}${bool("correction.enabled", "用大模型纠正转写错别字", v.correction.enabled, "人名、术语以词表为准")}
            ${num("mimo.asr_concurrency", "转写并发数", v.mimo.asr_concurrency, "", "1", "1")}</div></div>
        <div class="card"><div class="card-head"><h2>记忆问答</h2></div>
          <div class="form-grid">${sel("memory.strategy", "问答策略", v.memory.strategy, [["prefetch", "prefetch · 先预检索再按需查（推荐）"], ["rag", "rag · 只用预检索，最快"], ["agentic", "agentic · 模型自己反复检索"]])}
            ${sel("memory.retrieval", "检索方式", v.memory.retrieval, [["hybrid", "混合（关键词+向量，推荐）"], ["vector", "向量"], ["fts", "关键词"]], "向量需要本地 90 MB 模型，首次使用自动下载")}
            ${num("memory.prefetch_k", "预检索条数", v.memory.prefetch_k, "", "1", "1")}${num("memory.max_tool_rounds", "最多工具调用轮数", v.memory.max_tool_rounds, "", "1", "1")}
            ${bool("memory.query_expansion", "检索前让模型补同义词", v.memory.query_expansion, "每次多一次调用，约 2–5 秒")}${text("memory.chat_model", "问答模型", v.memory.chat_model, "留空 = MiMo 纠错 / 问答模型")}</div></div>
        <div class="card"><div class="card-head"><h2>日报与推送</h2><div class="row"><button class="btn sm" id="test-push" type="button">发测试消息</button></div></div>
          <div class="form-grid">${text("analysis.daily_time", "每日分析时间", v.analysis.daily_time, "HH:MM，本地时间", "time")}${bool("analysis.push_daily", "生成后自动推送", v.analysis.push_daily)}
            ${num("analysis.catchup_days", "停机后补几天日报", v.analysis.catchup_days ?? 7, "", "1", "1")}</div>
          <div class="mt"><div class="small muted mb">推送渠道（可多选）</div><div class="row">${chan("toast", "Windows 通知")}${chan("feishu", "飞书机器人")}${chan("wecom", "企业微信机器人")}${chan("serverchan", "Server酱")}${chan("pushplus", "PushPlus")}${chan("wxpusher", "WxPusher")}${chan("ntfy", "ntfy")}</div></div>
          <div class="form-grid mt">${secretField("push.feishu.webhook", "飞书 webhook")}${secretField("push.feishu.secret", "飞书签名密钥", "未开启签名校验则留空")}${secretField("push.wecom.webhook", "企业微信 webhook")}
            ${secretField("push.serverchan.sendkey", "Server酱 SendKey")}${secretField("push.pushplus.token", "PushPlus token")}${secretField("push.wxpusher.spt", "WxPusher SPT")}
            ${text("push.ntfy.topic", "ntfy 主题", v.push.ntfy.topic, "建议用随机字符串")}${text("push.ntfy.server", "ntfy 服务器", v.push.ntfy.server)}</div></div>
        <div class="card"><div class="card-head"><h2>其他</h2></div>
          <div class="form-grid">${num("ui.port", "界面端口", v.ui.port, "改后需重启", "1", "1024")}${bool("ui.open_browser", "audio-note run 启动时打开浏览器", v.ui.open_browser)}${sel("log.level", "日志级别", v.log.level, [["DEBUG", "DEBUG"], ["INFO", "INFO"], ["WARNING", "WARNING"], ["ERROR", "ERROR"]], "改后需重启")}</div></div>
        <div class="sticky-save"><button class="btn primary" id="save-cfg" type="button">保存设置</button><span class="small muted" id="save-note"></span></div>
      </form>
      <div class="card mt"><div class="card-head"><div><h2>词表</h2><div class="small muted">人名、产品名、术语，每行一个。转写纠错时优先使用，检索时当作一个词。<code>${esc(cfg.glossary_path)}</code></div></div><div class="row"><button class="btn sm" id="save-gl">保存词表</button></div></div><textarea class="input mono" id="glossary" rows="8" placeholder="张三&#10;科沃斯&#10;灰度发布"></textarea></div>
      <div class="card mt"><div class="card-head"><div><h2>导入录音文件</h2><div class="small muted">手机录音、会议录音等，会按语音切段、进入和麦克风录音一样的转写流程。文件名里有时间（如 20261031_0900）就自动用它，否则按文件修改时间，也可以手动填。</div></div></div>
        <div class="row"><input type="file" id="imp-file" accept="audio/*,.m4a,.amr,.mp4"><input class="input" id="imp-start" type="datetime-local" title="录音开始时间（可选）"><button class="btn" id="imp-btn">导入</button></div><div id="imp-result" class="mt"></div></div>
      <div class="card mt"><div class="kv"><span class="k">配置文件</span><span class="mono">${esc(cfg.path)}</span><span class="k">数据目录</span><span class="mono">${esc(cfg.data_dir)}</span><span class="k">模型目录</span><span class="mono">${esc(cfg.models_dir)}</span></div></div>`;
    api("/api/glossary").then((g) => { $("#glossary").value = g.text; }).catch(() => {});
    const collect = () => {
      const out = {};
      for (const el of $$("[data-path]")) {
        const [a, b, c] = el.dataset.path.split(".");
        let val;
        if (el.type === "checkbox") val = el.checked;
        else if (el.type === "number") { if (el.value === "") continue; val = Number(el.value); }
        else if (el.type === "password") { if (!el.value) continue; val = el.value; }
        else val = el.value;
        if (el.dataset.path === "audio.device") val = val === "" ? null : (/^\d+$/.test(val) ? Number(val) : val);
        if (el.dataset.path === "audio.system_device") val = val === "" ? null : val;
        out[a] = out[a] || {};
        if (c) { out[a][b] = out[a][b] || {}; out[a][b][c] = val; } else out[a][b] = val;
      }
      out.push = out.push || {};
      out.push.channels = $$("[data-chan]").filter((e) => e.checked).map((e) => e.dataset.chan);
      return out;
    };
    $("#save-cfg").onclick = async () => {
      const b = $("#save-cfg"); b.disabled = true;
      try {
        const r = await api("/api/config", { method: "PUT", json: collect() });
        $("#save-note").textContent = r.saved.length ? `已保存（${r.saved.length} 项）${r.notes.length ? "：" + r.notes.join("；") : ""}` : "没有改动";
        toast(r.saved.length ? "设置已保存" : "没有需要保存的修改", "ok");
        await refreshStatus();
        $$('[data-path][type="password"]').forEach((e) => { e.value = ""; });
        if (r.saved.some((k) => k.endsWith("api_key") || k.includes("webhook") || k.includes("token") || k.includes("sendkey") || k.includes("spt"))) route();
      } catch (e) { toastErr(e); } finally { b.disabled = false; }
    };
    $("#test-api").onclick = async () => {
      const b = $("#test-api"); b.disabled = true; b.innerHTML = `<span class="spin"></span>测试中`;
      try { const r = await api("/api/config/test-api", { method: "POST" }); toast(`连接正常：${r.model} 在 ${r.seconds}s 内回复「${r.reply}」`, "ok", 6000); }
      catch (e) { toastErr(e); } finally { b.disabled = false; b.textContent = "测试连接"; }
    };
    $("#test-push").onclick = async () => {
      const b = $("#test-push"); b.disabled = true;
      try { const r = await api("/api/push/test", { method: "POST", json: {} }); const parts = Object.entries(r.channels).map(([k, v]) => `${k}：${v ? "失败 " + v : "已发送"}`); toast(parts.join("；"), r.ok ? "ok" : "err", 7000); }
      catch (e) { toastErr(e); } finally { b.disabled = false; }
    };
    $("#save-gl").onclick = async () => {
      try { const r = await api("/api/glossary", { method: "PUT", json: { text: $("#glossary").value, reindex: true } }); toast(`词表已保存（${r.terms} 个词）${r.job ? "，正在更新检索索引" : ""}`, "ok"); }
      catch (e) { toastErr(e); }
    };
    $("#imp-btn").onclick = async () => {
      const f = $("#imp-file").files[0]; if (!f) { toast("先选择文件", "err"); return; }
      const box = $("#imp-result"); box.innerHTML = `<div class="thinking"><span class="spin"></span>正在导入 ${esc(f.name)}（切分语音段）…</div>`;
      const start = $("#imp-start").value ? `&start=${encodeURIComponent($("#imp-start").value.replace("T", " "))}` : "";
      try {
        const r = await api(`/api/import?filename=${encodeURIComponent(f.name)}${start}`, { method: "POST", body: f });
        box.innerHTML = `<div class="alert info" style="margin:0">已导入 ${r.segments} 段语音。<span class="right"><button class="btn sm primary" id="imp-proc">立即转写</button></span></div>`;
        $("#imp-proc").onclick = () => runJob("process");
      } catch (e) { box.innerHTML = `<div class="alert danger" style="margin:0">${esc(e.message)}</div>`; }
    };
  }

  // ------------------------------------------------------------------ logs
  async function pageLogs(arg, my) {
    let which = "service", auto = true, filter = "";
    main.innerHTML = `<div class="page-head"><div><h1>日志</h1><div class="sub" id="log-path"></div></div>
      <div class="row"><select class="input" id="log-which"><option value="service">服务 / 界面</option><option value="cli">命令行</option></select><input class="input" id="log-filter" placeholder="过滤文本"><label class="check"><input type="checkbox" id="log-auto" checked> 自动刷新</label><button class="btn" id="log-refresh">刷新</button></div></div>
      <div class="logbox" id="logbox"></div>`;
    const load = async () => {
      const r = await api(`/api/logs?lines=400&which=${which}`);
      if (stale(my)) return;
      $("#log-path").textContent = `${r.path} · ${fmtBytes(r.size)}`;
      const box = $("#logbox");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
      const lines = filter ? r.lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase())) : r.lines;
      box.innerHTML = lines.length ? lines.map((l) => { const m = /\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s/.exec(l); return `<div class="ln ${m ? m[1] : ""}">${esc(l)}</div>`; }).join("") : `<div class="muted">（空）</div>`;
      if (atBottom || !box.dataset.init) { box.scrollTop = box.scrollHeight; box.dataset.init = "1"; }
    };
    $("#log-which").onchange = (e) => { which = e.target.value; load(); };
    $("#log-filter").oninput = (e) => { filter = e.target.value; load(); };
    $("#log-auto").onchange = (e) => { auto = e.target.checked; };
    $("#log-refresh").onclick = load;
    await load();
    startPagePoll(async () => { if (auto) await load(); }, 3000, my);
  }

  const PAGES = { dashboard: pageDashboard, transcript: pageTranscript, daily: pageDaily, chat: pageChat, speakers: pageSpeakers, settings: pageSettings, logs: pageLogs };

  // boot
  refreshStatus().then(() => { if (state.status) state.status.unknown_label = undefined; });
  pollTimer = setInterval(() => { if (!document.hidden) refreshStatus(); }, 5000);
  route();
})();
