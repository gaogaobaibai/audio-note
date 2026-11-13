// Tiny in-page event bus. Event names used across the app:
//   "status" (status, previous)        every successful /api/status read
//   "online" / "offline"               connection to the local server came back / was lost
//   "job:done" {name, job} / "job:error" {name, job}      a background job finished
//   "live:<type>" {…}                  server events from /api/events (SPEC 4.3): recording, segment, utterances,
//                                      utterance, job, session, inbox, memory, daily, speakers, lexicon, config, profile
//   "live:mode" "sse" | "poll"         how updates currently arrive
//   "resync"                           the event stream reconnected: anything shown may be stale, reload it
//   "utterances:changed" {day?, ids?, reason}   a mutation made on this page touched sentences
//   "speakers:changed" {reason}        speakers / samples / relations changed
//   "inbox:changed" {open}             number of open inbox questions
//   "config:saved" {saved}             settings were saved
//   "profile:switched" {name}
//   "player:state" {…}                 global audio player (see ui/player.js)
//   "theme" {pref, resolved}
const listeners = new Map();

/** Subscribe; returns the unsubscribe function. */
export function on(type, fn) {
  if (!listeners.has(type)) listeners.set(type, new Set());
  listeners.get(type).add(fn);
  return () => { const set = listeners.get(type); if (set) set.delete(fn); };
}

export function once(type, fn) {
  const off = on(type, (...args) => { off(); fn(...args); });
  return off;
}

/** Call every listener of `type`; a throwing (or rejecting) listener never stops the others. */
export function emit(type, ...args) {
  const set = listeners.get(type);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      const r = fn(...args);
      if (r && typeof r.catch === "function") r.catch((e) => console.error(`[bus] ${type}`, e));
    } catch (e) {
      console.error(`[bus] ${type}`, e);
    }
  }
}
