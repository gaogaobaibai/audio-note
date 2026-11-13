// Dependency-free charts as inline SVG / CSS bars. Pure functions returning SafeHtml; colours come from CSS
// (currentColor and the .c0 … .c11 / .c-me / .c-unk speaker classes), so both themes work without redraw.
// No DOM access: node can import this file directly.
import { SafeHtml, html } from "../core/html.js";
import { esc } from "../core/util.js";

const f1 = (n) => (Math.round(n * 10) / 10).toString();

/** Tiny trend line with a soft area and a dot on the last value. */
export function sparkline(values, { w = 96, h = 28, label = "" } = {}) {
  const v = (values || []).map((x) => Number(x) || 0);
  if (v.length < 2) return new SafeHtml("");
  const max = Math.max(...v, 1e-9), pad = 3;
  const x = (i) => pad + (i * (w - 2 * pad)) / (v.length - 1);
  const y = (val) => h - pad - (val / max) * (h - 2 * pad);
  const pts = v.map((val, i) => `${f1(x(i))},${f1(y(val))}`);
  const area = `M${f1(x(0))},${h - pad} L${pts.join(" L")} L${f1(x(v.length - 1))},${h - pad} Z`;
  return new SafeHtml(`<svg class="chart spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(label || "趋势")}">`
    + `<path d="${area}" fill="currentColor" opacity=".12"/>`
    + `<polyline points="${pts.join(" ")}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>`
    + `<circle cx="${f1(x(v.length - 1))}" cy="${f1(y(v[v.length - 1]))}" r="2.4" fill="currentColor"/></svg>`);
}

/**
 * Vertical bars with a label under each: items [{label, value, title?, on?}] ("on" marks the highlighted bar, e.g. today).
 * Bars carry <title> tooltips; the chart as a whole is one labelled image.
 */
export function bars(items, { w = 320, h = 96, label = "" } = {}) {
  const list = items || [];
  if (!list.length) return new SafeHtml("");
  const max = Math.max(...list.map((d) => Number(d.value) || 0), 1e-9);
  const padB = 18, padT = 4, slot = w / list.length, bw = Math.max(4, Math.min(22, slot * 0.5));
  let out = `<svg class="chart bars" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label || "柱状图")}" preserveAspectRatio="xMidYMid meet">`;
  out += `<line x1="0" x2="${w}" y1="${h - padB + 0.5}" y2="${h - padB + 0.5}" class="axis"/>`;
  list.forEach((d, i) => {
    const val = Number(d.value) || 0;
    const bh = val > 0 ? Math.max(2, (val / max) * (h - padB - padT)) : 0;
    const cx = slot * i + slot / 2;
    out += `<g class="bar${d.on ? " on" : ""}"><title>${esc(d.title || `${d.label}: ${val}`)}</title>`
      + `<rect x="${f1(cx - slot / 2)}" y="0" width="${f1(slot)}" height="${h}" fill="transparent"/>`
      + (bh ? `<rect class="bar-fill" x="${f1(cx - bw / 2)}" y="${f1(h - padB - bh)}" width="${f1(bw)}" height="${f1(bh)}" rx="2.5"/>` : "")
      + `<text x="${f1(cx)}" y="${h - 5}" text-anchor="middle">${esc(d.label)}</text></g>`;
  });
  return new SafeHtml(out + "</svg>");
}

/** Horizontal bars with text labels: items [{label, value, text?, cls?}] (cls = a speaker colour class). */
export function hbars(items, { max = 0 } = {}) {
  const list = (items || []).filter((d) => Number(d.value) > 0);
  const top = max || Math.max(...list.map((d) => Number(d.value)), 1e-9);
  return html`<div class="hbars">${list.map((d) => html`<div class="hbar ${d.cls || ""}">
    <span class="hbar-label clip" title="${d.label}">${d.label}</span>
    <span class="hbar-track"><i style="width:${Math.max(2, (Number(d.value) / top) * 100).toFixed(1)}%"></i></span>
    <span class="hbar-val">${d.text ?? d.value}</span>
  </div>`)}</div>`;
}

/** Ring gauge (storage use): pct 0–100. */
export function ring(pct, { size = 44, stroke = 5, label = "" } = {}) {
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  const r = (size - stroke) / 2, c = 2 * Math.PI * r;
  return new SafeHtml(`<svg class="chart ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${esc(label || `${p.toFixed(0)}%`)}">`
    + `<circle cx="${size / 2}" cy="${size / 2}" r="${f1(r)}" fill="none" class="ring-track" stroke-width="${stroke}"/>`
    + `<circle cx="${size / 2}" cy="${size / 2}" r="${f1(r)}" fill="none" class="ring-fill" stroke-width="${stroke}" stroke-linecap="round"`
    + ` stroke-dasharray="${f1((p / 100) * c)} ${f1(c)}" transform="rotate(-90 ${size / 2} ${size / 2})"/></svg>`);
}

const hhmm = (min) => `${String(Math.floor(min / 60)).padStart(2, "0")}:${String(Math.floor(min % 60)).padStart(2, "0")}`;

/**
 * "Who spoke when" ribbon for one day.
 *   spans: [{start: seconds since midnight, dur: seconds, cls: speaker colour class, name}]
 * Speech is summed into `bucketMin`-minute buckets; each bucket is a stacked column (height = how much speech,
 * colours = who). The axis covers whole hours around the first and last sentence (at least `minHours`).
 * Each bucket <g> carries data-t="HH:MM" so a page can jump to that time, and a <title> tooltip.
 */
export function dayStrip(spans, { bucketMin = 10, h = 36, minHours = 6, label = "" } = {}) {
  const list = (spans || []).filter((s) => Number.isFinite(s.start) && s.dur > 0);
  if (!list.length) return new SafeHtml("");
  const first = Math.min(...list.map((s) => s.start)), last = Math.max(...list.map((s) => s.start + s.dur));
  let h0 = Math.floor(first / 3600), h1 = Math.min(24, Math.ceil(last / 3600));
  while (h1 - h0 < minHours) { if (h0 > 0) h0--; if (h1 - h0 < minHours && h1 < 24) h1++; if (h0 === 0 && h1 === 24) break; }
  const m0 = h0 * 60, m1 = h1 * 60, n = Math.ceil((m1 - m0) / bucketMin);
  const buckets = Array.from({ length: n }, () => new Map());
  for (const s of list) {
    let t = s.start, left = s.dur;
    while (left > 0) {                                         // a long sentence can straddle a bucket boundary
      const bi = Math.min(n - 1, Math.max(0, Math.floor((t / 60 - m0) / bucketMin)));
      const bucketEnd = (m0 + (bi + 1) * bucketMin) * 60;
      const part = Math.min(left, Math.max(0.001, bucketEnd - t));
      const key = `${s.cls}|${s.name}`;
      buckets[bi].set(key, (buckets[bi].get(key) || 0) + part);
      t += part; left -= part;
    }
  }
  const totals = buckets.map((b) => [...b.values()].reduce((a, v) => a + v, 0));
  const max = Math.max(...totals, 1e-9);
  const W = 1000, plotH = h, bw = W / n, gap = Math.min(1.5, bw * 0.15);
  let out = `<svg class="chart strip" viewBox="0 0 ${W} ${h}" preserveAspectRatio="none" role="img" aria-label="${esc(label || "这一天谁在什么时候说话")}">`;
  out += `<rect x="0" y="0" width="${W}" height="${plotH}" class="strip-bg" rx="3"/>`;
  buckets.forEach((b, i) => {
    if (!totals[i]) return;
    const t = hhmm(m0 + i * bucketMin);
    const colH = Math.max(3, (Math.sqrt(totals[i] / max)) * (plotH - 2));     // sqrt: quiet buckets stay visible next to a busy one
    const parts = [...b.entries()].sort((a, c) => c[1] - a[1]);
    const names = parts.slice(0, 4).map(([k, v]) => `${k.split("|")[1]} ${Math.round(v)} 秒`).join("，");
    out += `<g class="strip-b" data-t="${t}"><title>${esc(`${t} · ${names}`)}</title><rect x="${f1(i * bw)}" y="0" width="${f1(bw)}" height="${plotH}" fill="transparent"/>`;
    let y = plotH - 1;
    for (const [k, v] of parts) {
      const ph = (v / totals[i]) * colH;
      y -= ph;
      out += `<rect class="spk-fill ${esc(k.split("|")[0])}" x="${f1(i * bw + gap / 2)}" y="${f1(y)}" width="${f1(bw - gap)}" height="${f1(ph)}"/>`;
    }
    out += "</g>";
  });
  out += "</svg>";
  // hour labels are HTML (SVG text would be stretched by preserveAspectRatio="none")
  const hours = h1 - h0, step = hours > 16 ? 4 : hours > 8 ? 2 : 1;
  let ticks = "";
  for (let hr = h0; hr <= h1; hr += step) ticks += `<span style="left:${(((hr - h0) / hours) * 100).toFixed(2)}%">${hr}</span>`;
  return new SafeHtml(`<div class="strip-wrap">${out}<div class="strip-axis" aria-hidden="true">${ticks}</div></div>`);
}
