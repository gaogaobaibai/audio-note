"""MemoryChat.repl(): console loop driven by scripted input (no network, no real terminal)."""
from __future__ import annotations

from types import SimpleNamespace

from audio_note.db import DB
from audio_note.llm.client import LLMError
from audio_note.memory.chat import MemoryChat


class ScriptedConsole:
    """Stands in for rich.console.Console: feeds inputs, records prints."""

    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.printed: list[str] = []

    def input(self, prompt=""):
        if not self.inputs:
            raise EOFError
        v = self.inputs.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    def print(self, *args, **kw):
        self.printed.append(" ".join(str(a) for a in args))

    def status(self, *a, **k):
        class _S:
            def __enter__(self_):
                return self_

            def __exit__(self_, *exc):
                return False
        return _S()


class Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=a, tool_calls=None))], usage=None)


def test_repl_commands_history_and_errors(cfg_factory, monkeypatch):
    import rich.console

    cfg = cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "fts", "strategy": "agentic"}})
    db = DB(cfg.db_path)
    console = ScriptedConsole(["", "/help", "第一问", "第二问", "/clear", "出错的问题", "/quit", "never reached"])
    monkeypatch.setattr(rich.console, "Console", lambda *a, **k: console)
    client = Client(["答一", "答二", LLMError("boom")])
    chat = MemoryChat(cfg, db, client)
    chat.repl(max_history=2)

    assert chat.on_tool is None                      # observer removed on exit
    assert len(client.calls) == 3
    # second question carried the first exchange as history; after /clear the third had none
    roles2 = [m["role"] for m in client.calls[1]]
    assert roles2 == ["system", "user", "assistant", "user"]
    roles3 = [m["role"] for m in client.calls[2]]
    assert roles3 == ["system", "user"]
    joined = "\n".join(console.printed)
    assert "命令" in joined and "对话已清空" in joined and "LLM 调用失败" in joined and "boom" in joined


def test_repl_eof_and_keyboard_interrupt_exit_cleanly(cfg_factory, monkeypatch):
    import rich.console

    cfg = cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "fts"}})
    db = DB(cfg.db_path)
    for stop in (EOFError(), KeyboardInterrupt()):
        console = ScriptedConsole([stop])
        monkeypatch.setattr(rich.console, "Console", (lambda c: (lambda *a, **k: c))(console))
        chat = MemoryChat(cfg, db, Client([]))
        chat.repl()                                   # no exception
        assert chat.on_tool is None


def test_repl_shows_tool_calls(cfg_factory, monkeypatch):
    import json
    import rich.console

    cfg = cfg_factory({"memory": {"embedding_model": "hash", "retrieval": "fts", "strategy": "agentic"}})
    db = DB(cfg.db_path)
    console = ScriptedConsole(["有哪些天", "/q"])
    monkeypatch.setattr(rich.console, "Console", lambda *a, **k: console)

    class ToolClient(Client):
        def chat(self, messages, **kw):
            self.calls.append(messages)
            if len(self.calls) == 1:
                tc = SimpleNamespace(id="c1", type="function",
                                     function=SimpleNamespace(name="list_days", arguments=json.dumps({})))
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))], usage=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="没有记录", tool_calls=None))], usage=None)

    chat = MemoryChat(cfg, db, ToolClient([]))
    chat.repl()
    joined = "\n".join(console.printed)
    assert "list_days" in joined and "没有任何转写记录" in joined
