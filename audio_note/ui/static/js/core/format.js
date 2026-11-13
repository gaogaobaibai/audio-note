// Formatting helpers (dates, durations, sizes, loudness). No DOM access: node can import this file directly.

const pad2 = (n) => String(n).padStart(2, "0");

/** "N 分钟" below one hour, "N.N 小时" from there. */
export const fmtMin = (m) => (m >= 60 ? `${(m / 60).toFixed(1)} 小时` : `${Math.round(m)} 分钟`);

export function fmtBytes(n) {
  if (n == null || !Number.isFinite(Number(n))) return "-";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let v = Number(n), i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`;
}

/** ISO-ish timestamp ("2026-11-03T09:12:33" or with a space) -> "2026-11-03 09:12"; empty -> "-". Pure slicing, no Date. */
export const fmtDT = (s) => (s ? String(s).replace("T", " ").slice(0, 16) : "-");
/** ISO-ish timestamp -> "09:12"; empty -> "-". */
export const fmtTime = (s) => (s ? String(s).replace("T", " ").slice(11, 16) : "-");

/** Seconds -> "m:ss" (or "h:mm:ss"). */
export function fmtClock(sec) {
  const s = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${pad2(m)}:${pad2(r)}` : `${m}:${pad2(r)}`;
}

/** Seconds -> "45 秒" / "12 分" / "1 小时 5 分". */
export function fmtDur(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  if (s < 60) return `${s} 秒`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} 分`;
  return m % 60 ? `${Math.floor(m / 60)} 小时 ${m % 60} 分` : `${m / 60} 小时`;
}

/** Loudness in dBFS shown as "−32 dB" (real minus sign); null / NaN -> "". */
export function fmtDb(v) {
  if (v == null || !Number.isFinite(Number(v))) return "";
  const n = Math.round(Number(v));
  return `${n < 0 ? "−" : ""}${Math.abs(n)} dB`;
}

export function dayOf(date) { return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`; }
/** Local "YYYY-MM-DD" (optionally of a given Date, for tests). */
export const today = (now = new Date()) => dayOf(now);
export const isDay = (s) => /^\d{4}-\d{2}-\d{2}$/.test(String(s || ""));
export function addDays(day, n) {
  const d = new Date(`${day}T00:00:00`);
  d.setDate(d.getDate() + n);
  return dayOf(d);
}

export const WEEK = ["日", "一", "二", "三", "四", "五", "六"];
/** "今天 周三" / "昨天 周二" / "前天 周一" / "周日". */
export function dayLabel(day, now = new Date()) {
  if (!isDay(day)) return "";
  const d = new Date(`${day}T00:00:00`);
  const t = new Date(`${dayOf(now)}T00:00:00`);
  const diff = Math.round((t - d) / 86400000);
  const w = `周${WEEK[d.getDay()]}`;
  if (diff === 0) return `今天 ${w}`;
  if (diff === 1) return `昨天 ${w}`;
  if (diff === 2) return `前天 ${w}`;
  return w;
}
/** "11月3日" (with the year when it is not the current one). */
export function dayTitle(day, now = new Date()) {
  if (!isDay(day)) return String(day || "");
  const [y, m, d] = day.split("-").map(Number);
  return y === now.getFullYear() ? `${m}月${d}日` : `${y}年${m}月${d}日`;
}

/**
 * Turn what a person types into a day: "今天" / "昨天" / "前天", "11-03", "11/3", "11月3日", "2026-11-03", "3号".
 * A month-day without a year means the most recent such date that is not in the future. Returns "YYYY-MM-DD" or null.
 */
export function parseDayQuery(text, now = new Date()) {
  const q = String(text || "").trim();
  if (!q) return null;
  const rel = { "今天": 0, "今日": 0, "昨天": -1, "昨日": -1, "前天": -2, "大前天": -3 };
  if (q in rel) return addDays(dayOf(now), rel[q]);
  const valid = (y, m, d) => {
    const dt = new Date(y, m - 1, d);
    return dt.getFullYear() === y && dt.getMonth() === m - 1 && dt.getDate() === d ? dayOf(dt) : null;
  };
  let m = /^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$/.exec(q);
  if (m) return valid(Number(m[1]), Number(m[2]), Number(m[3]));
  m = /^(\d{1,2})[-/.月](\d{1,2})[日号]?$/.exec(q);
  if (m) {
    const day = valid(now.getFullYear(), Number(m[1]), Number(m[2]));
    if (!day) return null;
    return day > dayOf(now) ? valid(now.getFullYear() - 1, Number(m[1]), Number(m[2])) : day;
  }
  m = /^(\d{1,2})[日号]$/.exec(q);
  if (m) {
    const day = valid(now.getFullYear(), now.getMonth() + 1, Number(m[1]));
    if (!day) return null;
    if (day <= dayOf(now)) return day;
    const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    return valid(prev.getFullYear(), prev.getMonth() + 1, Number(m[1]));
  }
  return null;
}

/** Seconds since local midnight of an "…T09:12:33" timestamp (NaN when it has no time part). */
export function secOfDay(ts) {
  const m = /[T ](\d{2}):(\d{2})(?::(\d{2}))?/.exec(String(ts || ""));
  return m ? Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3] || 0) : NaN;
}
/** Milliseconds of a local ISO-ish timestamp, NaN when unparsable. */
export function tsMs(ts) {
  if (!ts) return NaN;
  return new Date(String(ts).replace(" ", "T")).getTime();
}
