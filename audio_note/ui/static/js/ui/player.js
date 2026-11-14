// The ONE audio player. A single <audio> element, a queue, and a bar at the bottom of the window.
//   player.play(clip)                       play one clip
//   player.play(clips, {index: 3})          play a list from position 3 and auto-advance to the end (a session)
//   clip = {key, src, title, sub?, uttId?, cls?}   — build with clips.utterance(u) / .segment() / .sample() / .negative()
// Keys: Space play / pause, J next, K previous (not while typing). The sentence being played gets the class
// "playing" wherever it is on the page: any element with data-utt="<utterance id>" or data-clip="<clip key>".
// Dialogs embed a small transport bound to the same player with mountInline().
// While it plays, the player tells the server (POST /api/playback, then a heartbeat), so the recorder does not
// record our own playback again as new speech (audio/playback.py).
import { html } from "../core/html.js";
import { $, $$, render, reveal } from "../core/dom.js";
import { emit, on as busOn } from "../core/bus.js";
import { prefs } from "../core/store.js";
import { audioUrl } from "../core/api.js";
import { fmtClock } from "../core/format.js";
import { clamp } from "../core/util.js";
import { bind } from "../core/shortcuts.js";
import { icon } from "./icons.js";
import { toast } from "./toast.js";

export const RATES = [0.75, 1, 1.25, 1.5, 2];
const st = { queue: [], index: -1, playing: false, time: 0, duration: 0, rate: 1 };
let audio = null, bar = null, errorsInRow = 0, ui = null;

// ---- self-playback guard: "this tab is playing" reports; the server stops trusting a tab TTL_S after its last one
const HEARTBEAT_MS = 2000;
const TAB = `t${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`;
let beat = null, seq = 0;
function sendPlayback(playing) {
  try {
    // seq: reports can overtake each other on the way; the server ignores one older than what it already has
    fetch("/api/playback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tab: TAB, playing, seq: ++seq }), keepalive: true })
      .catch(() => { /* the server is gone or busy: nothing to protect then */ });
  } catch (e) { /* never disturb playback */ }
}
function guardOn() {
  sendPlayback(true);
  if (!beat) beat = setInterval(() => sendPlayback(true), HEARTBEAT_MS);
}
function guardOff() {
  if (!beat) return;
  clearInterval(beat);
  beat = null;
  sendPlayback(false);
}

const current = () => (st.index >= 0 ? st.queue[st.index] || null : null);
const snapshot = () => ({ clip: current(), index: st.index, count: st.queue.length, playing: st.playing, time: st.time, duration: st.duration, rate: st.rate });
const rateLabel = (r) => `${Number(r).toFixed(r % 1 === 0 ? 1 : 2).replace(/0$/, "").replace(/\.$/, ".0")}×`;

export const clips = {
  /** A sentence. `u` needs {id, speaker?, time?, text?}. */
  utterance(u, { pad = 0.4, cls = "" } = {}) {
    const time = String(u.time || "").slice(0, 5);
    return { key: `utt:${u.id}`, src: audioUrl.utterance(u.id, pad), uttId: Number(u.id), title: [u.speaker, time].filter(Boolean).join(" · ") || "录音", sub: u.text || "", cls };
  },
  segment(segmentId, from, to, title = "录音片段") {
    return { key: `seg:${segmentId}:${from}:${to}`, src: audioUrl.segment(segmentId, from, to), title, sub: "" };
  },
  sample: (id, title = "声纹样本") => ({ key: `sample:${id}`, src: audioUrl.sample(id), title, sub: "" }),
  negative: (id, title = "负样本") => ({ key: `neg:${id}`, src: audioUrl.negative(id), title, sub: "" }),
};

/** Put the "playing" class on whatever shows the current clip (call again after re-rendering rows). */
function mark(root = document) {
  for (const el of $$(".playing[data-utt], .playing[data-clip]", root)) el.classList.remove("playing");
  const c = current();
  if (!c) return;
  if (c.uttId != null) for (const el of $$(`[data-utt="${c.uttId}"]`, root)) el.classList.add("playing");
  for (const el of $$("[data-clip]", root)) if (el.dataset.clip === c.key) el.classList.add("playing");
}

function paint() {
  if (!bar || !ui) return;
  const c = current();
  bar.hidden = !c;
  document.documentElement.classList.toggle("has-player", !!c);
  if (!c) return;
  ui.title.textContent = c.title || "";
  ui.sub.textContent = c.sub || "";
  ui.sub.title = c.sub || "";
  render(ui.toggle, st.playing ? icon("pause", 18) : icon("play", 18));
  ui.toggle.setAttribute("aria-label", st.playing ? "暂停" : "播放");
  ui.toggle.title = st.playing ? "暂停（空格）" : "播放（空格）";
  ui.prev.disabled = st.index <= 0;
  ui.next.disabled = st.index >= st.queue.length - 1;
  ui.count.textContent = st.queue.length > 1 ? `${st.index + 1} / ${st.queue.length}` : "";
  ui.rate.textContent = rateLabel(st.rate);
  paintTime();
}
function paintTime() {
  if (!ui) return;
  const d = st.duration || 0;
  ui.time.textContent = `${fmtClock(st.time)} / ${d ? fmtClock(d) : "–:––"}`;
  if (document.activeElement !== ui.seek || !ui.seeking) ui.seek.value = d ? String(Math.round((st.time / d) * 1000)) : "0";
  ui.seek.style.setProperty("--pct", `${d ? clamp((st.time / d) * 100, 0, 100).toFixed(1) : 0}%`);
}
function changed() { paint(); mark(); emit("player:state", snapshot()); }

function load(i, { auto = false } = {}) {
  st.index = i;
  st.time = 0;
  st.duration = 0;
  const c = current();
  if (!c) { stop(); return; }
  audio.src = c.src;
  audio.playbackRate = st.rate;
  const p = audio.play();
  if (p && p.catch) p.catch(() => { /* autoplay refused or the source failed: the error event reports the latter */ });
  changed();
  if (auto && c.uttId != null) {
    const row = document.getElementById(`u${c.uttId}`);
    if (row) { const r = row.getBoundingClientRect(); if (r.top < 96 || r.bottom > window.innerHeight - 96) reveal(row); }
  }
}

function stop() {
  if (audio) { audio.pause(); audio.removeAttribute("src"); try { audio.load(); } catch (e) { /* ignore */ } }
  st.queue = []; st.index = -1; st.playing = false; st.time = 0; st.duration = 0;
  changed();
}

export const player = {
  /** Play a clip, or a list of clips starting at `index` (auto-advances through the list). */
  play(clipOrList, { index = 0 } = {}) {
    const list = (Array.isArray(clipOrList) ? clipOrList : [clipOrList]).filter((c) => c && c.src);
    if (!list.length || !audio) return;
    errorsInRow = 0;
    st.queue = list;
    load(clamp(index, 0, list.length - 1));
  },
  toggle() {
    if (!current()) return;
    if (audio.paused) { const p = audio.play(); if (p && p.catch) p.catch(() => {}); } else audio.pause();
  },
  pause() { if (audio && !audio.paused) audio.pause(); },
  next() { if (st.index < st.queue.length - 1) load(st.index + 1); },
  prev() {
    if (st.time > 2 || st.index <= 0) { if (current()) { audio.currentTime = 0; } return; }
    load(st.index - 1);
  },
  stop,
  seek(sec) { if (current() && Number.isFinite(sec)) { audio.currentTime = clamp(sec, 0, st.duration || sec); st.time = audio.currentTime; paintTime(); emit("player:time", snapshot()); } },
  setRate(r) {
    st.rate = RATES.includes(Number(r)) ? Number(r) : 1;
    prefs.set("rate", st.rate);
    if (audio) audio.playbackRate = st.rate;
    changed();
  },
  cycleRate() { this.setRate(RATES[(RATES.indexOf(st.rate) + 1) % RATES.length]); },
  state: snapshot,
  isCurrent: (key) => { const c = current(); return !!c && c.key === key; },
  /** Toggle if `clip` is the current one, otherwise start it (the play button of a row). */
  playOrToggle(clipOrList, opts = {}) {
    const list = Array.isArray(clipOrList) ? clipOrList : [clipOrList];
    const want = list[clamp(opts.index || 0, 0, list.length - 1)];
    if (want && this.isCurrent(want.key)) this.toggle(); else this.play(clipOrList, opts);
  },
  mark,
};

/**
 * Small transport for dialogs: play / pause button, seek bar, time — bound to the global player for `clip`.
 * Returns {time(), duration(), setClip(clip), destroy()}; destroy() stops the clip if it is still the current one.
 */
export function mountInline(el, clip, { label = "试听" } = {}) {
  let c = clip;
  render(el, html`<div class="pl-inline">
    <button type="button" class="icon-btn solid" data-pl="toggle" aria-label="${label}">${icon("play", 16)}</button>
    <input type="range" class="pl-seek" min="0" max="1000" value="0" step="1" aria-label="播放进度">
    <span class="pl-time">0:00</span>
  </div>`);
  const btn = el.querySelector("[data-pl]"), seek = el.querySelector(".pl-seek"), time = el.querySelector(".pl-time");
  const mine = () => !!c && player.isCurrent(c.key);
  const update = () => {
    const on = mine();
    render(btn, on && st.playing ? icon("pause", 16) : icon("play", 16));
    btn.setAttribute("aria-label", on && st.playing ? "暂停" : label);
    const d = on ? st.duration : 0, t = on ? st.time : 0;
    seek.disabled = !on || !d;
    if (document.activeElement !== seek) seek.value = d ? String(Math.round((t / d) * 1000)) : "0";
    seek.style.setProperty("--pct", `${d ? clamp((t / d) * 100, 0, 100).toFixed(1) : 0}%`);
    time.textContent = on && d ? `${fmtClock(t)} / ${fmtClock(d)}` : "0:00";
  };
  const offs = [busOn("player:state", update), busOn("player:time", update)];
  btn.addEventListener("click", () => { if (c) player.playOrToggle(c); });
  seek.addEventListener("input", () => { if (mine() && st.duration) player.seek((Number(seek.value) / 1000) * st.duration); });
  update();
  return {
    time: () => (mine() ? st.time : 0),
    duration: () => (mine() ? st.duration : 0),
    pause: () => { if (mine()) player.pause(); },
    setClip(next) { if (mine()) player.stop(); c = next; update(); },
    destroy() { for (const off of offs) off(); if (mine()) player.stop(); },
  };
}

export function initPlayer() {
  bar = $("#player");
  if (!bar || audio) return;
  st.rate = RATES.includes(Number(prefs.get("rate", 1))) ? Number(prefs.get("rate", 1)) : 1;
  render(bar, html`<div class="player-inner" role="region" aria-label="播放器">
    <audio preload="auto"></audio>
    <div class="player-ctl">
      <button type="button" class="icon-btn" data-p="prev" aria-label="上一句" title="上一句（K）">${icon("prev", 16)}</button>
      <button type="button" class="icon-btn solid" data-p="toggle" aria-label="播放">${icon("play", 18)}</button>
      <button type="button" class="icon-btn" data-p="next" aria-label="下一句" title="下一句（J）">${icon("next", 16)}</button>
    </div>
    <div class="player-info"><div class="player-title clip"></div><div class="player-sub clip"></div></div>
    <input type="range" class="pl-seek player-seek" min="0" max="1000" value="0" step="1" aria-label="播放进度">
    <span class="player-time"></span>
    <span class="player-count"></span>
    <button type="button" class="btn ghost sm player-rate" data-p="rate" aria-label="播放速度" title="播放速度"></button>
    <button type="button" class="icon-btn" data-p="close" aria-label="关闭播放器" title="关闭播放器">${icon("x", 16)}</button>
  </div>`);
  audio = bar.querySelector("audio");
  ui = {
    title: bar.querySelector(".player-title"), sub: bar.querySelector(".player-sub"), toggle: bar.querySelector('[data-p="toggle"]'),
    prev: bar.querySelector('[data-p="prev"]'), next: bar.querySelector('[data-p="next"]'), seek: bar.querySelector(".player-seek"),
    time: bar.querySelector(".player-time"), count: bar.querySelector(".player-count"), rate: bar.querySelector('[data-p="rate"]'), seeking: false,
  };
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("[data-p]");
    if (!b) return;
    const a = b.dataset.p;
    if (a === "toggle") player.toggle();
    else if (a === "prev") player.prev();
    else if (a === "next") player.next();
    else if (a === "rate") player.cycleRate();
    else if (a === "close") player.stop();
  });
  ui.seek.addEventListener("input", () => { ui.seeking = true; if (st.duration) player.seek((Number(ui.seek.value) / 1000) * st.duration); });
  ui.seek.addEventListener("change", () => { ui.seeking = false; });
  audio.addEventListener("play", () => { guardOn(); st.playing = true; changed(); });
  audio.addEventListener("playing", () => { guardOn(); errorsInRow = 0; });
  audio.addEventListener("pause", () => { guardOff(); st.playing = false; changed(); });
  audio.addEventListener("emptied", () => { if (audio.paused) guardOff(); });
  window.addEventListener("pagehide", () => guardOff());
  audio.addEventListener("timeupdate", () => { st.time = audio.currentTime || 0; paintTime(); emit("player:time", snapshot()); });
  audio.addEventListener("durationchange", () => { st.duration = Number.isFinite(audio.duration) ? audio.duration : 0; paintTime(); emit("player:time", snapshot()); });
  audio.addEventListener("ended", () => {
    st.playing = false;
    if (st.index < st.queue.length - 1) load(st.index + 1, { auto: true }); else changed();
  });
  audio.addEventListener("error", () => {
    guardOff();
    if (!current()) return;                      // stop() clears the source, which also fires "error"
    errorsInRow++;
    st.playing = false;
    if (errorsInRow === 1) toast("音频无法播放（文件可能已删除）", { kind: "err" });
    if (errorsInRow < 3 && st.index < st.queue.length - 1) load(st.index + 1, { auto: true }); else changed();
  });
  bind("space", () => { if (!current()) return false; player.toggle(); return true; }, { desc: "播放 / 暂停", group: "播放", overlay: true });
  bind("j", () => { if (!current()) return false; player.next(); return true; }, { desc: "下一句", group: "播放", overlay: true });
  bind("k", () => { if (!current()) return false; player.prev(); return true; }, { desc: "上一句", group: "播放", overlay: true });
  busOn("profile:switched", () => stop());
  paint();
}
