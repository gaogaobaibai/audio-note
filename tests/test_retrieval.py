"""Tests for the retrieval layer: embeddings, vector store, fusion, Retriever modes, query expansion,
chat strategies, pipeline embedding hook, reindex CLI and the benchmark helpers. No network: the
embedding model is the dependency-free HashEmbedder and the LLM is scripted."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.memory import index, vectors
from audio_note.memory.chat import MemoryChat
from audio_note.memory.embed import Embedder, HashEmbedder, get_embedder, normalise, stable_hash
from audio_note.memory.expand import QueryExpander, clean_terms
from audio_note.memory.index import Hit, tokenize
from audio_note.memory.retrieval import HOW, Retriever, rrf_fuse
from audio_note.memory.tools import TOOLS, run_tool

UTTS = [
    ("2026-10-27", "20:30:00", "我", "买了，科沃斯T30，昨天下的单，明天能到。"),
    ("2026-10-27", "20:30:10", "小雨", "礼物你买了吗？妈想要个扫地机器人。"),
    ("2026-10-27", "12:10:00", "我", "今天去楼下那家兰州拉面。"),
    ("2026-10-28", "09:30:00", "李强", "五百不够，灰度二零零零人至少要撑到一千QPS。"),
    ("2026-10-28", "12:10:00", "我", "走，今天去吃黄焖鸡。"),
    ("2026-10-29", "09:30:00", "王敏", "回滚方案我昨晚写好了，三分钟内可以回滚。"),
    ("2026-10-29", "12:10:00", "我", "今天吃啥？楼下新开了一家麻辣香锅。"),
]


@pytest.fixture
def mcfg(cfg_factory):
    return cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "hybrid", "max_tool_rounds": 3}})


@pytest.fixture
def mdb(mcfg):
    index.register_terms(["王敏", "李强", "小雨"])
    d = DB(mcfg.db_path)
    seg = d.add_segment(mcfg.audio_dir / "x.opus", datetime(2026, 10, 27, 12, 0, 0), 600.0)
    rows = []
    for i, (day, t, spk, text) in enumerate(UTTS):
        rows.append({
            "segment_id": seg, "start_s": float(i), "end_s": float(i) + 0.5, "abs_start": f"{day}T{t}", "day": day,
            "speaker_name": spk, "text_raw": text, "text": text, "text_seg": tokenize(text),
        })
    d.ids = d.add_utterances(rows)  # type: ignore[attr-defined]
    return d


def _text(db: DB, uid: int) -> str:
    return db.one("SELECT text FROM utterances WHERE id=?", (uid,))["text"]


# --- embeddings ------------------------------------------------------------------------
def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=64)
    a = e.embed_passages(["科沃斯扫地机器人", "科沃斯扫地机器人", "兰州拉面"])
    assert a.shape == (3, 64) and a.dtype == np.float32
    assert np.allclose(a[0], a[1])
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(a[0] @ a[2]) < float(a[0] @ e.embed_query("扫地机器人"))
    assert e.embed_passages([]).shape == (0, 64)
    assert stable_hash("abc") == stable_hash("abc") and stable_hash("abc") != stable_hash("abd")
    z = normalise(np.zeros(4))
    assert np.all(z == 0)


def test_get_embedder_selects_by_config(cfg_factory):
    assert isinstance(get_embedder(cfg_factory({"memory": {"embedding_model": "hash"}})), HashEmbedder)
    e = get_embedder(cfg_factory({"memory": {"embedding_model": "BAAI/bge-small-zh-v1.5", "embedding_threads": 3}}))
    assert isinstance(e, Embedder) and e.name == "BAAI/bge-small-zh-v1.5" and e.threads == 3
    assert e.cache_dir.name == "embeddings" and e._model is None  # lazy: nothing loaded yet


# --- vector store ----------------------------------------------------------------------
def test_db_vectors_roundtrip_and_model_mismatch(mdb):
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert mdb.add_vectors("m1", [(mdb.ids[0], v)]) == 1
    row = mdb.one("SELECT * FROM utterance_vectors WHERE utterance_id=?", (mdb.ids[0],))
    stored = np.frombuffer(row["vec"], dtype=np.float32)
    assert row["model"] == "m1" and row["dim"] == 2 and np.allclose(stored, [0.6, 0.8])  # L2-normalised
    missing = mdb.utterances_without_vectors("m1")
    assert {r["id"] for r in missing} == set(mdb.ids[1:])
    # a vector from another model counts as missing for m2
    assert len(mdb.utterances_without_vectors("m2")) == len(UTTS)
    assert mdb.vector_count("m1") == 1 and mdb.vector_count() == 1
    # upsert replaces
    mdb.add_vectors("m2", [(mdb.ids[0], np.array([1.0, 0.0]))])
    assert mdb.vector_count("m1") == 0 and mdb.vector_count("m2") == 1
    # cascade on utterance delete
    mdb.execute("DELETE FROM utterances WHERE id=?", (mdb.ids[0],))
    assert mdb.vector_count() == 0


def test_index_missing_batches_and_limit(mdb):
    e = HashEmbedder(32)
    assert vectors.missing_count(mdb, e.name) == len(UTTS)
    assert vectors.index_missing(mdb, e, batch=2, limit=3) == 3
    assert mdb.vector_count(e.name) == 3 and vectors.missing_count(mdb, e.name) == len(UTTS) - 3
    assert vectors.index_missing(mdb, e, batch=2) == len(UTTS) - 3
    assert vectors.index_missing(mdb, e) == 0
    assert mdb.vector_count(e.name) == len(UTTS)


@pytest.mark.parametrize("use_sql", [True, False])
def test_vector_search_paths_and_filters(mdb, monkeypatch, use_sql):
    if use_sql and not DB.vec_available:
        pytest.skip("sqlite-vec not loaded")
    monkeypatch.setattr(DB, "vec_available", use_sql)
    e = HashEmbedder(64)
    vectors.index_missing(mdb, e)
    hits = vectors.search(mdb, e, "科沃斯扫地机器人", limit=3)
    assert len(hits) == 3 and all(h.source == "vector" for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert "科沃斯" in hits[0].text or "扫地机器人" in hits[0].text
    # filters
    assert all(h.day == "2026-10-28" for h in vectors.search(mdb, e, "吃饭", day_from="2026-10-28", day_to="2026-10-28"))
    assert all(h.speaker_name == "王敏" for h in vectors.search(mdb, e, "回滚", speaker="王敏"))
    assert vectors.search(mdb, e, "回滚", speaker="王敏")[0].text.startswith("回滚方案")
    assert vectors.search(mdb, e, "", limit=5) == []
    assert vectors.search(mdb, e, "回滚", min_score=0.999) == []
    # vectors of a different dimension are ignored, not mixed in
    assert vectors.search(mdb, HashEmbedder(16), "回滚") == []


def test_rrf_fuse_orders_by_fused_rank_and_joins_sources():
    def h(i, src):
        return Hit(id=i, segment_id=1, day="d", abs_start="t", start_s=0, end_s=1, speaker_name=None, text=str(i), source=src)
    a = [h(1, "fts"), h(2, "fts"), h(3, "fts")]
    b = [h(3, "vector"), h(1, "vector"), h(4, "vector")]
    fused = rrf_fuse([a, b], k=10)
    assert [x.id for x in fused] == [1, 3, 2, 4]
    assert fused[0].source == "fts+vector" and fused[2].source == "fts" and fused[3].source == "vector"
    assert fused[0].score == pytest.approx(1 / 11 + 1 / 12)
    # weights tip the balance
    fused_w = rrf_fuse([a, b], k=10, weights=[0.1, 1.0])
    assert fused_w[0].id == 3
    assert rrf_fuse([], k=10) == [] and [x.id for x in rrf_fuse([a])] == [1, 2, 3]


# --- Retriever ---------------------------------------------------------------------------
def test_retriever_modes(mcfg, mdb):
    fts = Retriever(mcfg, mdb, mode="fts")
    r = fts.search("扫地机器人")
    assert r.how == "fts" and r.how_label == HOW["fts"] and [h.text for h in r.hits] == [UTTS[1][3]]
    assert fts.embedder() is None

    vec = Retriever(mcfg, mdb, mode="vector")
    r = vec.search("扫地机器人 科沃斯", limit=2)
    assert r.how == "vector" and len(r.hits) == 2 and mdb.vector_count("hash") == len(UTTS)  # auto-indexed
    assert all(h.source == "vector" for h in r.hits)

    hyb = Retriever(mcfg, mdb)   # config default = hybrid
    assert hyb.mode == "hybrid"
    r = hyb.search("扫地机器人", limit=3)
    assert r.how == "hybrid" and r.hits[0].text == UTTS[1][3] and r.hits[0].source == "fts+vector"
    # hybrid uses the strict keyword list: a natural-language question with no AND match -> vector only
    r2 = hyb.search("妈的生日礼物买的是什么", limit=3)
    assert all("fts" not in h.source for h in r2.hits) and r2.hits
    # LIKE fallback when nothing matches (a term jieba cuts differently + no vector hit above zero is unlikely,
    # so force it through the fts retriever)
    r3 = fts.search("T30")
    assert r3.hits and r3.how in ("fts", "like")
    assert fts.search("").hits == []


def test_retriever_hybrid_can_disable_strict(mcfg, mdb):
    mcfg["memory"]["hybrid_fts_strict"] = False
    hyb = Retriever(mcfg, mdb)
    r = hyb.search("想要礼物的是谁", limit=5)   # no AND match; the loose OR list now joins the fusion
    assert any("fts" in h.source for h in r.hits)


def test_retriever_falls_back_when_model_unavailable(mcfg, mdb, monkeypatch, caplog):
    import audio_note.memory.retrieval as rmod

    def boom(cfg):
        raise RuntimeError("no model")

    monkeypatch.setattr(rmod, "get_embedder", boom)
    for mode in ("vector", "hybrid"):
        r = Retriever(mcfg, mdb, mode=mode)
        res = r.search("扫地机器人")
        assert res.how == "fts" and [h.text for h in res.hits] == [UTTS[1][3]]
        assert r.embedder() is None and r.ensure_indexed() == 0
    assert "falling back to keyword search" in caplog.text
    assert Retriever(mcfg, mdb, mode="bogus").mode == "fts"


def test_search_terms_or_query(mdb):
    hits = index.search_terms(mdb, ["机器人", "T30", "的"], limit=10)
    texts = [h.text for h in hits]
    assert UTTS[0][3] in texts and UTTS[1][3] in texts and len(hits) == 2
    assert index.search_terms(mdb, ["", "。"]) == []
    assert index.search_terms(mdb, []) == []


# --- query expansion -----------------------------------------------------------------------
class ScriptedClient:
    def __init__(self, data=None, err=None):
        self.data, self.err, self.calls = data, err, []

    def json(self, system, user, **kw):
        self.calls.append((system, user, kw))
        if self.err:
            raise self.err
        return self.data

    def chat(self, messages, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))], usage=None)


def test_clean_terms():
    assert clean_terms({"terms": [" 科沃斯 ", "扫地机", "科沃斯", 3, "", "“机器人”", "x" * 13, "扫地机器人到货"]}, "扫地机器人到货了吗") \
        == ["科沃斯"]   # 机器人 and 扫地机 are already in the query
    assert clean_terms(["a", "b", "c"], "q", max_terms=2) == ["a", "b"]
    assert clean_terms({"terms": "nope"}, "q") == [] and clean_terms(None, "q") == []


def test_query_expander_caches_and_survives_errors():
    c = ScriptedClient({"terms": ["科沃斯", "吸尘器"]})
    x = QueryExpander(c, model="m")
    assert x.expand("扫地机器人") == ["科沃斯", "吸尘器"]
    assert x.expand(" 扫地机器人 ") == ["科沃斯", "吸尘器"] and len(c.calls) == 1
    assert c.calls[0][2]["model"] == "m" and "扫地机器人" in c.calls[0][1]
    assert x.expand("") == []
    bad = QueryExpander(ScriptedClient(err=LLMError("down")))
    assert bad.expand("q") == [] and bad.expand("q") == []
    worse = QueryExpander(ScriptedClient(err=ValueError("weird")))
    assert worse.expand("q") == []


def test_retriever_expansion_adds_keyword_hits(mcfg, mdb):
    client = ScriptedClient({"terms": ["科沃斯"]})
    r = Retriever(mcfg, mdb, client, mode="fts", expand=True)
    res = r.search("扫地机 到货", limit=5)
    assert res.expansions == ["科沃斯"]
    assert UTTS[0][3] in [h.text for h in res.hits]
    # config flag path + no client -> no expander
    mcfg["memory"]["query_expansion"] = True
    assert Retriever(mcfg, mdb).expander is None
    assert Retriever(mcfg, mdb, client).expander is not None
    assert Retriever(mcfg, mdb, client, expand=False).expander is None


# --- tool integration ------------------------------------------------------------------------
def test_search_tool_uses_retriever_and_reports_method(mcfg, mdb):
    out = run_tool("search_transcripts", {"query": "扫地机器人"}, mdb, mcfg)  # no retriever -> keyword
    assert HOW["fts"] in out and "扫地机器人" in out
    client = ScriptedClient({"terms": ["科沃斯"]})
    r = Retriever(mcfg, mdb, client, expand=True)
    out = run_tool("search_transcripts", {"query": "扫地机", "day_from": "2026-10-27", "day_to": "2026-10-27"}, mdb, mcfg, r)
    assert HOW["hybrid"] in out and "已扩展：科沃斯" in out and "科沃斯T30" in out
    assert "2026-10-28" not in out
    out = run_tool("search_transcripts", {"query": "zzzz不存在的词"}, mdb, mcfg, Retriever(mcfg, mdb, mode="fts"))
    assert "没有找到" in out
    assert "找到" in run_tool("search_transcripts", {"query": "zzzz不存在的词"}, mdb, mcfg, r)  # vectors always rank something
    # other tools ignore the retriever argument
    assert "2026-10-27" in run_tool("list_days", {}, mdb, mcfg, r)
    assert "search_transcripts" in [t["function"]["name"] for t in TOOLS]


# --- chat strategies ------------------------------------------------------------------------------
def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(cid, name, args):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeChat:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def chat(self, messages, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        msg = self.responses.pop(0) if self.responses else _msg("fallback")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    def json(self, *a, **k):
        raise AssertionError("not used")


def test_chat_prefetch_puts_hits_in_system_prompt(mcfg, mdb):
    client = FakeChat([_msg("科沃斯T30，10-27 20:30 我说的。")])
    chat = MemoryChat(mcfg, mdb, client)       # config default strategy = prefetch, retrieval = hybrid
    assert chat.strategy == "prefetch" and chat.retriever.mode == "hybrid"
    chat.now = datetime(2026, 10, 30, 21, 0)
    ans = chat.ask("扫地机器人买的哪个牌子？")
    assert ans.startswith("科沃斯T30")
    sysmsg = client.calls[0]["messages"][0]["content"]
    assert "2026-10-30 21:00（周五）" in sysmsg
    assert "## 预检索结果" in sysmsg and "科沃斯T30" in sysmsg and "[#" in sysmsg
    assert client.calls[0]["tools"] == TOOLS
    assert chat.last_stats == {"strategy": "prefetch", "llm_calls": 1, "tool_calls": 0, "rounds": 1,
                               "prefetched": chat.last_stats["prefetched"]}
    assert chat.last_stats["prefetched"] >= 1


def test_chat_rag_single_call_without_tools(mcfg, mdb):
    client = FakeChat([_msg(None, [_call("c1", "list_days", {})]), _msg("never")])
    chat = MemoryChat(mcfg, mdb, client, strategy="rag")
    ans = chat.ask("扫地机器人买的哪个牌子？")
    assert len(client.calls) == 1 and client.calls[0]["tools"] is None
    assert ans == ""  # a tool call without tools yields no content; the loop must not run tools
    assert chat.last_stats["llm_calls"] == 1 and chat.last_stats["tool_calls"] == 0
    sysmsg = client.calls[0]["messages"][0]["content"]
    assert "只根据下面「预检索结果」回答" in sysmsg and "科沃斯" in sysmsg


def test_chat_agentic_has_no_prefetch_and_counts_tools(mcfg, mdb):
    client = FakeChat([_msg(None, [_call("c1", "search_transcripts", {"query": "科沃斯"})]), _msg("答案")])
    chat = MemoryChat(mcfg, mdb, client, strategy="agentic")
    assert chat.ask("q") == "答案"
    assert "预检索结果" not in client.calls[0]["messages"][0]["content"]
    assert chat.last_stats == {"strategy": "agentic", "llm_calls": 2, "tool_calls": 1, "rounds": 2, "prefetched": 0}
    tool_msg = [m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]
    assert HOW["hybrid"] in tool_msg["content"]   # the tool went through the chat's hybrid retriever


def test_chat_unknown_strategy_and_prefetch_failure(mcfg, mdb, monkeypatch):
    client = FakeChat([_msg("ok")])
    chat = MemoryChat(mcfg, mdb, client, strategy="weird")
    assert chat.strategy == "prefetch"
    monkeypatch.setattr(chat.retriever, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert chat.ask("q") == "ok"
    assert "预检索结果" not in client.calls[0]["messages"][0]["content"]


# --- pipeline hook ----------------------------------------------------------------------------------
def test_pipeline_embeds_new_utterances(mcfg, mdb):
    from tests.test_pipeline import EchoCorrector, FakeASR, FakeEngine, Turn, make_segment
    from audio_note.speech.pipeline import Processor

    seg_id = make_segment(mcfg, mdb, datetime(2026, 11, 1, 8, 0, 0), 3.0)
    p = Processor(mcfg, mdb, client=object(), engine=FakeEngine([Turn(0.0, 1.4, 0), Turn(1.6, 2.9, 1)]),
                  asr=FakeASR("你好世界"), corrector=EchoCorrector())
    before = mdb.vector_count("hash")
    assert p.process_segment(seg_id) == 2
    assert mdb.vector_count("hash") == before + 2 + len(UTTS)  # also back-fills the fixture rows
    assert vectors.missing_count(mdb, "hash") == 0


def test_pipeline_skips_embedding_when_fts_or_broken(mcfg, mdb, monkeypatch, caplog):
    from tests.test_pipeline import EchoCorrector, FakeASR, FakeEngine, Turn, make_segment
    from audio_note.speech.pipeline import Processor

    mcfg["memory"]["retrieval"] = "fts"
    seg_id = make_segment(mcfg, mdb, datetime(2026, 11, 1, 8, 0, 0), 3.0)
    p = Processor(mcfg, mdb, client=object(), engine=FakeEngine([Turn(0.0, 2.9, 0)]), asr=FakeASR("你好"), corrector=EchoCorrector())
    assert p.process_segment(seg_id) == 1 and mdb.vector_count() == 0

    mcfg["memory"]["retrieval"] = "hybrid"
    import audio_note.memory.embed as emod

    monkeypatch.setattr(emod, "get_embedder", lambda cfg: (_ for _ in ()).throw(RuntimeError("no model")))
    seg2 = make_segment(mcfg, mdb, datetime(2026, 11, 1, 9, 0, 0), 3.0)
    p2 = Processor(mcfg, mdb, client=object(), engine=FakeEngine([Turn(0.0, 2.9, 0)]), asr=FakeASR("你好"), corrector=EchoCorrector())
    assert p2.process_segment(seg2) == 1
    assert "embedding skipped" in caplog.text and p2._embedder is False
    seg3 = make_segment(mcfg, mdb, datetime(2026, 11, 1, 10, 0, 0), 3.0)
    assert p2.process_segment(seg3) == 1     # does not retry the model every segment


# --- CLI -----------------------------------------------------------------------------------------------
def test_cli_reindex(mcfg, mdb, capsys):
    from audio_note.cli import main

    assert main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "关键词索引已重建" in out and "向量索引完成" in out
    assert mdb.vector_count("hash") == len(UTTS)
    assert main(["reindex"]) == 0 and "已是最新" in capsys.readouterr().out
    assert main(["reindex", "--no-vectors"]) == 0 and "跳过" in capsys.readouterr().out


def test_cli_chat_flags_build_retriever(mcfg, mdb, monkeypatch):
    from audio_note import cli

    seen = {}

    class FakeMC:
        def __init__(self, cfg, db, client, retriever=None, strategy=None):
            seen["retriever"], seen["strategy"] = retriever, strategy

        def ask(self, question, history=None):
            return "答"

    import audio_note.memory.chat as cmod

    monkeypatch.setattr(cmod, "MemoryChat", FakeMC)
    monkeypatch.setattr(cli, "_client", lambda cfg: ScriptedClient({"terms": []}))
    assert cli.main(["chat", "--strategy", "rag", "--retrieval", "fts", "--expand", "问题"]) == 0
    assert seen["strategy"] == "rag" and seen["retriever"].mode == "fts" and seen["retriever"].expander is not None
    assert cli.main(["chat", "--no-expand", "问题"]) == 0
    assert seen["strategy"] is None and seen["retriever"].mode == "hybrid" and seen["retriever"].expander is None


# --- benchmark helpers -----------------------------------------------------------------------------------
def test_bench_helpers(mdb):
    import importlib.util
    import sys

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("memory_bench", root / "scripts" / "memory_bench.py")
    mb = importlib.util.module_from_spec(spec)
    sys.modules["memory_bench"] = mb
    spec.loader.exec_module(mb)

    qs, now = mb.load_questions(root / "scripts" / "memory_questions.yaml")
    assert now.startswith("2026-10-30") and len(qs) >= 20
    assert all(q["n"] >= 1 and isinstance(q["expect"], list) for q in qs)
    assert [q["n"] for q in mb.load_questions(root / "scripts" / "memory_questions.yaml", {2, 5})[0]] == [2, 5]

    ev = mb.resolve_evidence(mdb, ["科沃斯T30", "不存在的证据"])
    assert ev[0] == {mdb.ids[0]} and ev[1] == set()
    assert mb.evidence_found([mdb.ids[0], 99], ev, k=1) == 1 and mb.evidence_found([99, mdb.ids[0]], ev, k=1) == 0
    assert mb.answer_score("买的是科沃斯，2899 元", [["科沃斯"], ["2899", "两千八百九十九"], ["没有"]]) == (2, 3)
    assert mb.answer_score("", []) == (0, 0)


def test_cli_reindex_in_fresh_interpreter(mcfg, mdb):
    """Regression: `_lazy("memory", "index")` only worked when another import had already loaded the
    submodule; a fresh `python -m audio_note reindex` must work on its own."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, AUDIO_NOTE_CONFIG=os.environ["AUDIO_NOTE_CONFIG"], PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "audio_note", "reindex", "--no-vectors"], capture_output=True,
                       text=True, encoding="utf-8", env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "关键词索引已重建" in r.stdout
