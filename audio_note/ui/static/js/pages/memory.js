// 记忆 — placeholder until the memory package lands (items, todos, people, lexicon, "about me", forgetting).
import { html } from "../core/html.js";
import { render } from "../core/dom.js";
import { emptyState } from "../ui/components.js";

export default {
  async mount(ctx) {
    render(ctx.el, html`<div class="page"><div class="card">${emptyState({
      icon: "book",
      title: "记忆正在搬家",
      text: "下一版这里会列出它记住的事：待办、决定、常提到的人、专有名词的写法，还有一份「关于我」。每一条都能看到出处，可以改、可以删。",
      action: html`<a class="btn" href="#/daily/notes">查看现有的记忆笔记</a>`,
    })}</div></div>`);
  },
};
