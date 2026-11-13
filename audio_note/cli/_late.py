"""Late binding between the command modules and the `audio_note.cli` facade.

Tests (and embedders) replace names on the package object - `monkeypatch.setattr(cli, "console", ...)`,
`cli._client`, `cli._optional_engine`, `cli.load_config`, `cli.PROJECT_ROOT`, `cli.cmd_<name>` - and expect
the commands to see the replacement. A command module must therefore never bind those objects at import
time:

    console = Late("console")        # every attribute access / call is forwarded to the package's current value
    path = pkg().PROJECT_ROOT / ...  # plain values (a Path cannot be proxied) are read through pkg()
    set_defaults(func=handler(cmd))  # the package's current `cmd.__name__` if it re-exports cmd, else cmd itself
"""
from __future__ import annotations

import sys
from typing import Any, Callable

PKG = __name__.rsplit(".", 1)[0]          # "audio_note.cli"


def pkg() -> Any:
    """The `audio_note.cli` package module (looked up on every call, never cached)."""
    return sys.modules[PKG]


class Late:
    """Proxy for an attribute of the package that is resolved at use time."""

    __slots__ = ("_name",)

    def __init__(self, name: str):
        object.__setattr__(self, "_name", name)

    def _target(self) -> Any:
        return getattr(sys.modules[PKG], object.__getattribute__(self, "_name"))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._target(), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._target(), attr)

    def __call__(self, *a: Any, **k: Any) -> Any:
        return self._target()(*a, **k)

    def __repr__(self) -> str:
        return f"Late({object.__getattribute__(self, '_name')!r})"


def handler(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The command handler argparse should dispatch to, decided when the parser is built.

    A handler the facade re-exports is looked up there by name, so one replaced on `audio_note.cli`
    wins (as it did when everything lived in one module). Any other function - a command module added
    later that the facade does not re-export, or one that merely shares a name - is returned as is."""
    facade = sys.modules.get(PKG)
    if getattr(facade, "EXPORTED_HANDLERS", {}).get(fn.__name__) is fn:
        return getattr(facade, fn.__name__, fn)
    return fn
