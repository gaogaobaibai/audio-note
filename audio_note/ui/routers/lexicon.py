"""Lexicon: canonical spellings, their aliases (mis-hearings / nicknames) and retroactive fixes.

    GET    /api/lexicon?q=&kind=&status=            -> {terms: [term], counts}
    POST   /api/lexicon   {term, kind?, note?, hotword?, aliases?: [str | {alias, kind?, scope?}]}  -> {ok, term, created}
    PATCH  /api/lexicon/{id}  {term?, kind?, note?, hotword?, status?, pinyin?}                      -> {ok, term}
    DELETE /api/lexicon/{id}                        -> {ok, deleted, disabled}
    POST   /api/lexicon/{id}/aliases  {alias, kind?, scope?}        -> {ok, alias, term, matches}
    DELETE /api/lexicon/aliases/{id}                -> {ok}
    POST   /api/lexicon/aliases/{id}/apply  {dry_run?, limit?}      -> {ok, alias, term, matches, total, applied, dry_run}

term = {id, term, kind, note, pinyin, speaker_id, source, status, hotword, hits, last_hit_at, created_at,
updated_at, aliases: [{id, lexicon_id, alias, kind, scope, source, hits}]} (speech.lexicon.Term.as_dict).
kind: person | term | project | org | place | product | other; status: active | candidate | disabled;
alias kind: mishear (rewritten to the term) | aka (another name, never rewritten); scope: auto | hint.

The rules live in audio_note.speech.lexicon. Every change publishes a `lexicon` event; applying an alias to the
history rewrites sentences (through revisions, undoable) and goes through deps.rewriting.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ..deps import RT, bool_arg, int_arg, rewriting

router = APIRouter(prefix="/api")


def _str(body: dict[str, Any], key: str) -> str | None:
    v = body.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise HTTPException(400, f"{key} 应为文字")
    return v


def _http(e: Exception) -> HTTPException:
    return HTTPException(404 if isinstance(e, LookupError) else 400, str(e))


@router.get("/lexicon")
def lexicon_list(rt: RT, q: str | None = None, kind: str | None = None, status: str | None = None) -> dict[str, Any]:
    """The entries (status: empty = active + candidate, "all", or one status). glossary.txt and the enrolled
    speakers are brought in first, so the page is complete on the very first visit."""
    from ...speech import lexicon

    if kind and kind not in lexicon.TERM_KINDS:
        raise HTTPException(400, f"kind 只能是 {' / '.join(lexicon.TERM_KINDS)}")
    if status and status != "all" and status not in lexicon.TERM_STATUSES:
        raise HTTPException(400, f"status 只能是 {' / '.join(lexicon.TERM_STATUSES)} / all")
    lexicon.sync_sources_throttled(rt.db, rt.cfg, every_s=2.0)
    return {"terms": lexicon.list_terms(rt.db, q=q, kind=kind or None, status=status or None), "counts": lexicon.counts(rt.db)}


@router.post("/lexicon")
def lexicon_add(rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Add an entry (or update the one with this spelling): {term, kind?, note?, hotword?, aliases?}."""
    from ...speech import lexicon

    term = _str(body, "term")
    if not term or not term.strip():
        raise HTTPException(400, "请填写词条")
    hotword = body.get("hotword")
    hotword = None if hotword is None else bool_arg(hotword, "hotword")
    aliases = body.get("aliases") or []
    if not isinstance(aliases, list):
        raise HTTPException(400, "aliases 应为列表")
    with rt.busy():
        try:
            before = lexicon.Lexicon.load(rt.db).is_term(" ".join(term.split()))
            tid = lexicon.upsert_term(rt.db, term, kind=_str(body, "kind") or "term", note=_str(body, "note"), source="user",
                                      hotword=hotword, events=rt.events)
            for a in aliases:
                if isinstance(a, str):
                    a = {"alias": a}
                if not isinstance(a, dict) or not isinstance(a.get("alias"), str):
                    raise ValueError("别名应为文字")
                lexicon.add_alias(rt.db, tid, a["alias"], kind=a.get("kind") or "mishear", scope=a.get("scope") or None,
                                  source="user", cfg=rt.cfg, events=rt.events)
        except (LookupError, ValueError) as e:
            raise _http(e)
    return {"ok": True, "term": lexicon.get_term(rt.db, tid), "created": not before}


@router.patch("/lexicon/{term_id}")
def lexicon_patch(term_id: int, rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Edit an entry: any of {term, kind, note, hotword, status, pinyin}. status 'disabled' switches it off
    (its aliases stop applying), 'active' brings it back or confirms a candidate."""
    from ...speech import lexicon

    hotword = body.get("hotword")
    hotword = None if hotword is None else bool_arg(hotword, "hotword")
    with rt.busy():
        try:
            term = lexicon.update_term(rt.db, int(term_id), term=_str(body, "term"), kind=_str(body, "kind"), note=_str(body, "note"),
                                       hotword=hotword, status=_str(body, "status"), pinyin_key=_str(body, "pinyin"),
                                       events=rt.events)
        except (LookupError, ValueError) as e:
            raise _http(e)
    return {"ok": True, "term": term}


@router.delete("/lexicon/aliases/{alias_id}")
def lexicon_alias_delete(alias_id: int, rt: RT) -> dict[str, Any]:
    from ...speech import lexicon

    with rt.busy():
        if not lexicon.delete_alias(rt.db, int(alias_id), events=rt.events):
            raise HTTPException(404, "没有这个别名")
    return {"ok": True}


@router.delete("/lexicon/{term_id}")
def lexicon_delete(term_id: int, rt: RT) -> dict[str, Any]:
    """Remove an entry with its aliases. The name of an enrolled speaker / a line of the glossary file would
    come straight back, so those are switched off instead: {"deleted": false, "disabled": true}."""
    from ...speech import lexicon

    with rt.busy():
        try:
            res = lexicon.delete_term(rt.db, rt.cfg, int(term_id), events=rt.events)
        except (LookupError, ValueError) as e:
            raise _http(e)
    return {"ok": True, **res}


@router.post("/lexicon/aliases/{alias_id}/apply")
def lexicon_alias_apply(alias_id: int, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Rewrite this alias to its entry in the stored sentences. {dry_run: true} only lists what would change
    (matches: [{utterance_id, day, time, before, after}], at most 200 listed; total counts all)."""
    from ...speech import lexicon

    dry = bool_arg(body.get("dry_run"), "dry_run")
    limit = int_arg(body.get("limit"), "limit", 1)
    try:
        if dry:
            res = lexicon.apply_alias_to_history(rt.db, rt.cfg, int(alias_id), dry_run=True, limit=limit)
        else:
            with rewriting(rt):
                res = lexicon.apply_alias_to_history(rt.db, rt.cfg, int(alias_id), limit=limit, source="lexicon", events=rt.events)
                _close_rule_questions(rt, int(alias_id))
    except (LookupError, ValueError) as e:
        raise _http(e)
    res.pop("match_ids", None)
    return {"ok": True, **res}


def _close_rule_questions(rt: Any, alias_id: int) -> None:
    """The history was just fixed by hand: the waiting "change the other N places too?" question is moot."""
    from ... import inbox

    for r in rt.db.query("SELECT id FROM inbox_items WHERE kind='rule' AND dedupe_key=? AND status IN ('open','queued','snoozed')",
                         (f"alias:{alias_id}",)):
        try:
            inbox.dismiss(rt.db, int(r["id"]), events=rt.events)
        except (LookupError, ValueError):
            pass


@router.post("/lexicon/{term_id}/aliases")
def lexicon_alias_add(term_id: int, rt: RT, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Add an alias: {alias, kind?: mishear | aka, scope?: auto | hint}. Without scope the server decides
    (a common word, a single character or an ambiguous spelling is only a hint). `matches` = how many stored
    sentences contain it (what POST .../apply would change)."""
    from ...speech import lexicon

    alias = _str(body, "alias")
    if not alias or not alias.strip():
        raise HTTPException(400, "请填写别名")
    with rt.busy():
        try:
            aid = lexicon.add_alias(rt.db, int(term_id), alias, kind=_str(body, "kind") or "mishear", scope=_str(body, "scope") or None,
                                    source="user", cfg=rt.cfg, events=rt.events)
            got = lexicon.get_alias(rt.db, aid) or {}
            matches = lexicon.count_alias_matches(rt.db, aid) if got.get("kind") == "mishear" else 0
        except (LookupError, ValueError) as e:
            raise _http(e)
    return {"ok": True, "alias": got, "term": lexicon.get_term(rt.db, int(term_id)), "matches": matches}
