// 待确认 — placeholder until the inbox package lands (questions about uncertain words, voices, memory conflicts).
import { html } from "../core/html.js";
import { render } from "../core/dom.js";
import { inboxOpen } from "../core/store.js";
import { emptyState } from "../ui/components.js";

export default {
  async mount(ctx) {
    const paint = () => {
      const n = inboxOpen();
      render(ctx.el, html`<div class="page"><div class="card">${emptyState({
        icon: "inbox",
        title: n ? `有 ${n} 件事想问你` : "现在没有需要你确认的事",
        text: "下一版会在这里列出系统拿不准的地方：听不清的词、反复出现却不认识的声音、前后矛盾的记忆。每个问题点一下就能回答，回答过的以后会自动照做。",
        action: html`<a class="btn" href="#/transcript">先去看看记录</a>`,
      })}</div></div>`);
    };
    paint();
    ctx.on("inbox:changed", paint);
  },
};
