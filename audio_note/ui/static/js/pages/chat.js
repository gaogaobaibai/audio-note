// 问答: ask anything about the records. Protocol: POST /api/chat -> newline-delimited JSON events
//   start {strategy, retrieval, model} · tool {name, args, result, lines} · answer {text, stats} · error {message} · ping
// Newer servers may add delta {text} (streamed answer text) and sources {citations}; unknown event types are
// ignored. The conversation lives in this module, so it survives switching pages (not a reload); a request that is
// still running when the page is left keeps running and its answer is there on return.
import { html, raw, when } from "../core/html.js";
import { $, render, on } from "../core/dom.js";
import { ndjson, isAbort } from "../core/api.js";
import { md } from "../core/md.js";
import { on as busOn } from "../core/bus.js";
import { prefs, loadConfig, hasApiKey } from "../core/store.js";
import { ensureStatus } from "../core/status.js";
import { icon } from "../ui/icons.js";
import { button, chip, emptyState, notice, spinner } from "../ui/components.js";
import { openPopover } from "../ui/popover.js";

const chatState = { msgs: [], busy: false, render: null };
busOn("profile:switched", () => { chatState.msgs = []; if (chatState.render) chatState.render(); });

const EXAMPLES = ["昨天晚上聊了什么？", "最近有什么待办？", "上周三的会上定了什么？", "这几天谁来找过我？", "最近提到过哪些要买的东西？"];
const STRATEGIES = [["prefetch", "先检索，再按需补查（推荐）"], ["rag", "只检索一次（最快）"], ["agentic", "让模型自己反复查（最细）"]];
const RETRIEVALS = [["hybrid", "关键词 + 语义（推荐）"], ["vector", "只按语义"], ["fts", "只按关键词"]];
const TOOL_LABEL = { search: "搜索记录", search_utterances: "搜索记录", search_notes: "查笔记", search_memory: "查记忆", get_day: "读某一天", get_context: "看前后文", list_days: "看有哪些天", list_todos: "查待办", get_session: "读一段对话", get_person: "查这个人" };

function msgHtml(m) {
  if (m.role === "user") return html`<div class="msg user"><div class="bubble">${m.content}</div></div>`;
  const tools = m.tools || [];
  const trace = tools.length ? html`<details class="trace"><summary>查了 ${tools.length} 次记录</summary>
    ${tools.map((t) => html`<div class="trace-call"><div class="trace-name">${TOOL_LABEL[t.name] || t.name}<code>${JSON.stringify(t.args ?? {})}</code></div><pre class="trace-res">${t.result || ""}${t.lines > 20 ? "\n…" : ""}</pre></div>`)}</details>` : "";
  const s = m.stats;
  const meta = s ? html`<div class="msg-meta">${[`${s.seconds ?? "?"} 秒`, s.llm_calls != null ? `${s.llm_calls} 次模型调用` : "", s.tool_calls ? `${s.tool_calls} 次查询` : "", m.model || ""].filter(Boolean).join(" · ")}</div>` : "";
  let inner;
  if (m.error) inner = notice({ kind: "danger", compact: true, body: m.error });
  else if (m.pending && !m.content) inner = spinner(tools.length ? `正在查记录（${TOOL_LABEL[tools[tools.length - 1].name] || tools[tools.length - 1].name}）…` : "正在想…");
  else inner = html`<div class="md">${raw(md(m.content))}</div>${when(m.pending, () => html`<span class="typing" aria-hidden="true"></span>`)}`;
  return html`<div class="msg bot"><div class="msg-ic" aria-hidden="true">${icon("sparkle", 16)}</div><div class="msg-main"><div class="bubble">${inner}</div>${trace}${meta}</div></div>`;
}

export default {
  async mount(ctx) {
    await ensureStatus();
    const cfg = await loadConfig().catch(() => null);
    if (!ctx.alive()) return;
    const mem = (cfg && cfg.values && cfg.values.memory) || {};
    const opts = {
      strategy: prefs.get("chat.strategy", mem.strategy || "prefetch"),
      retrieval: prefs.get("chat.retrieval", mem.retrieval || "hybrid"),
      expand: prefs.get("chat.expand", !!mem.query_expansion),
    };

    render(ctx.el, html`<div class="page chat">
      ${when(!hasApiKey(), () => notice({ kind: "warn", body: "还没有配置 MiMo API key，暂时不能问答。", actions: html`<a class="btn sm" href="#/settings/mimo">去设置</a>` }))}
      <div class="chat-msgs" id="chat-msgs" role="log" aria-live="polite" aria-label="对话"></div>
      <form class="chat-input" id="chat-form">
        <label class="sr-only" for="chat-q">问题</label>
        <textarea id="chat-q" rows="1" placeholder="问点什么：时间、人物、决定、买了什么……（回车发送，Shift+回车换行）"></textarea>
        <div class="chat-bar">
          <button type="button" class="btn ghost sm" data-act="opts" title="怎么找相关记录">${icon("sliders", 14)}<span>检索方式</span></button>
          <button type="button" class="btn ghost sm" data-act="clear" title="清空这次对话">${icon("trash", 14)}<span>清空</span></button>
          ${button({ label: "发送", icon: "send", kind: "primary", size: "sm", type: "submit", attrs: { id: "chat-send" } })}
        </div>
      </form>
    </div>`);
    const box = $("#chat-msgs"), q = $("#chat-q"), sendBtn = $("#chat-send");
    sendBtn.classList.add("push");

    const paint = () => {
      if (!chatState.msgs.length) {
        render(box, emptyState({ icon: "chat", title: "问问你的记录", text: "答案会标出依据的时间和说话人。可以从这些开始：", action: html`${EXAMPLES.map((x) => chip({ label: x, attrs: { "data-ex": x } }))}` }));
      } else {
        const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
        render(box, html`${chatState.msgs.map(msgHtml)}`);
        if (nearBottom) window.scrollTo(0, document.documentElement.scrollHeight);
      }
      sendBtn.disabled = chatState.busy;
    };
    chatState.render = () => { if (ctx.alive()) paint(); };

    const send = async (text) => {
      text = String(text ?? q.value).trim();
      if (!text || chatState.busy) return;
      q.value = "";
      q.style.height = "";
      const history = chatState.msgs.filter((m) => !m.error && !m.pending).map((m) => ({ role: m.role, content: m.content }));
      const bot = { role: "assistant", content: "", tools: [], pending: true };
      chatState.msgs.push({ role: "user", content: text }, bot);
      chatState.busy = true;
      const repaint = () => { if (chatState.render) chatState.render(); };
      repaint();
      window.scrollTo(0, document.documentElement.scrollHeight);
      try {
        for await (const ev of ndjson("/api/chat", { json: { question: text, history, strategy: opts.strategy, retrieval: opts.retrieval, expand: !!opts.expand } })) {
          if (ev.type === "start") bot.model = ev.model;
          else if (ev.type === "tool") { bot.tools.push(ev); repaint(); }
          else if (ev.type === "delta") { if (typeof ev.text === "string") { bot.content += ev.text; repaint(); } }
          else if (ev.type === "sources") bot.citations = ev.citations;
          else if (ev.type === "answer") { bot.content = ev.text || bot.content || "（空回答）"; bot.stats = ev.stats; bot.pending = false; repaint(); }
          else if (ev.type === "error") { bot.error = ev.message || "出错了"; bot.pending = false; repaint(); }
          // ping and anything newer: ignored
        }
        if (bot.pending) { bot.pending = false; if (!bot.content) bot.error = "连接中断"; }
      } catch (e) {
        bot.pending = false;
        bot.error = isAbort(e) ? "已停止" : (e.message || String(e));
      }
      chatState.busy = false;
      repaint();
      if (ctx.alive() && document.body.contains(q)) q.focus();
    };

    on(ctx.el, "submit", "#chat-form", (e) => { e.preventDefault(); send(); }, { signal: ctx.signal });
    q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
    q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = `${Math.min(180, q.scrollHeight)}px`; });
    on(ctx.el, "click", "[data-ex]", (e, b) => send(b.dataset.ex), { signal: ctx.signal });
    on(ctx.el, "click", "[data-act]", (e, b) => {
      if (b.dataset.act === "clear") { if (!chatState.busy) { chatState.msgs = []; paint(); q.focus(); } return; }
      if (b.dataset.act !== "opts") return;
      openPopover(b, html`<div class="pop-pad">
        <div class="field"><label for="opt-strategy">回答方式</label><select class="input" id="opt-strategy">${STRATEGIES.map(([k, l]) => html`<option value="${k}" ${opts.strategy === k ? html`selected` : ""}>${l}</option>`)}</select></div>
        <div class="field"><label for="opt-retrieval">怎么找相关记录</label><select class="input" id="opt-retrieval">${RETRIEVALS.map(([k, l]) => html`<option value="${k}" ${opts.retrieval === k ? html`selected` : ""}>${l}</option>`)}</select></div>
        <label class="check"><input type="checkbox" id="opt-expand" ${opts.expand ? html`checked` : ""}> 先让模型补同义词再找（多花 2–5 秒）</label>
        <p class="help">只影响这个浏览器里的问答；默认值在「设置 → 记忆问答」。</p></div>`, {
        label: "检索方式", align: "start", className: "pop-panel",
        onMount: (p) => p.el.addEventListener("change", () => {
          opts.strategy = prefs.set("chat.strategy", p.el.querySelector("#opt-strategy").value);
          opts.retrieval = prefs.set("chat.retrieval", p.el.querySelector("#opt-retrieval").value);
          opts.expand = prefs.set("chat.expand", p.el.querySelector("#opt-expand").checked);
        }),
      });
    }, { signal: ctx.signal });

    paint();
    q.focus();
    return () => { chatState.render = null; };
  },
};
