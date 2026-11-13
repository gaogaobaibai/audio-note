"""Frontend (audio_note/ui/static): structure, safety and the DOM-free core modules.

The page is native ES modules with no build step, so these tests are the only "compiler":
  * every relative import resolves to a file, nothing is imported from outside the package (no CDN);
  * index.html / style.css reference only files that exist;
  * markup reaches the DOM only through core/dom.js (which accepts nothing but the auto-escaping html`` template);
  * the design tokens keep a contrast of at least 4.5:1 for every text colour, in both themes;
  * core/{util,format,md,html}.js and the pure helpers of ui/ run under node (skipped when node is not installed).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "audio_note" / "ui" / "static"
JS = STATIC / "js"
CSS = STATIC / "css"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(not NODE, reason="node not installed")


def _js_files() -> list[Path]:
    return sorted([STATIC / "app.js", *JS.rglob("*.js")])


def _rel(p: Path) -> str:
    return p.relative_to(STATIC).as_posix()


def _strip_comments(src: str) -> str:
    """Drop /* … */ blocks and // line comments (good enough for import / sink scanning; strings stay)."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(re.sub(r"(^|[^:\"'`])//.*$", r"\1", ln) for ln in src.splitlines())


# --- layout ------------------------------------------------------------------------------------------

def test_entry_files_exist_and_stay_thin():
    assert "audio-note" in (STATIC / "index.html").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'import "./js/main.js";' in app and len(_strip_comments(app).strip().splitlines()) == 1
    style = (STATIC / "style.css").read_text(encoding="utf-8")
    imports = re.findall(r'@import url\("([^"]+)"\);', style)
    assert imports == ["css/tokens.css", "css/base.css", "css/components.css", "css/pages.css"]
    for name in imports:
        assert (STATIC / name).is_file(), name
    assert _strip_comments(style).replace("@import", "").count("{") == 0, "style.css is only the entry: rules live in css/"


def test_expected_modules_are_present():
    want = [f"js/core/{n}.js" for n in ("util", "format", "md", "html", "dom", "api", "bus", "store", "router", "status",
                                        "events", "theme", "shortcuts")]
    want += [f"js/ui/{n}.js" for n in ("icons", "components", "modal", "toast", "charts", "player", "palette", "speaker",
                                      "utterance", "popover", "shell")]
    want += [f"js/pages/{n}.js" for n in ("dashboard", "timeline", "inbox", "memory", "daily", "chat", "people", "settings")]
    want += ["js/main.js"]
    missing = [w for w in want if not (STATIC / w).is_file()]
    assert not missing, missing


def test_index_html_references_only_existing_assets():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="(/static/[^"]+)"', html)
    assert "/static/app.js?v=__V__" in refs and "/static/style.css?v=__V__" in refs
    for ref in refs:
        path = ref.split("?", 1)[0][len("/static/"):]
        assert (STATIC / path).is_file(), ref
    assert '<script type="module" src="/static/app.js?v=__V__">' in html
    # ids the shell code looks up (js/ui/shell.js, player.js, modal.js, toast.js, popover.js)
    for el_id in ("app", "rail", "nav", "rail-toggle", "topbar", "page-title", "palette-btn", "rec-pill", "bell-btn",
                  "theme-btn", "profile-btn", "banner", "main", "tabbar", "player", "modal-root", "pop-root", "toasts"):
        assert f'id="{el_id}"' in html, el_id
    # nothing is loaded from another host
    assert not re.findall(r'(?:href|src)="(?:https?:)?//', html)


def test_no_external_assets_anywhere():
    for p in [*CSS.glob("*.css"), STATIC / "style.css"]:
        css = _strip_comments(p.read_text(encoding="utf-8"))
        assert not re.search(r"url\(\s*[\"']?(?:https?:)?//", css), p.name
        assert "@font-face" not in css, p.name
    for p in _js_files():
        src = _strip_comments(p.read_text(encoding="utf-8"))
        assert not re.search(r"""(?:from\s*|import\s*\(?\s*)["'](?:https?:)?//""", src), _rel(p)
        assert not re.search(r"""(?:fetch|EventSource|WebSocket)\(\s*["'`](?:https?|wss?):""", src), _rel(p)


def test_text_files_are_utf8_lf():
    for p in [STATIC / "index.html", STATIC / "style.css", *CSS.glob("*.css"), *_js_files()]:
        raw = p.read_bytes()
        assert b"\r" not in raw, f"{_rel(p)} has CR line endings"
        raw.decode("utf-8")


# --- imports -----------------------------------------------------------------------------------------

# static imports / re-exports are one statement per line in this code base (anchoring them keeps the word "import"
# inside strings and markup out); dynamic imports are import("…") with a literal specifier
STATIC_IMPORT_RE = re.compile(
    r"""^[ \t]*(?:import[ \t]+(?:[^"'`;\n]*?[ \t]from[ \t]+)?|export[ \t]+[^"'`;\n=]*?[ \t]from[ \t]+)(["'])([^"'\n]+)\1[ \t]*;?[ \t]*$""", re.M)
DYNAMIC_IMPORT_RE = re.compile(r"""\bimport\(\s*(["'])([^"'\n]+)\1\s*\)""")


def _import_specs(src: str) -> list[str]:
    static = [spec for _q, spec in STATIC_IMPORT_RE.findall(src)]
    return static + [spec for _q, spec in DYNAMIC_IMPORT_RE.findall(src)]


def test_import_scanner_sees_what_the_browser_sees():
    src = 'import "./a.js";\nimport { x, y as z } from "../b.js";\nimport * as m from \'./c.js\';\nexport { q } from "./d.js";\n' \
          'const s = { id: "import", title: "x" };\nconst t = `data-x="foo import"><div class="row">`;\nload: () => import("./e.js"),\n' \
          'export const k = "./not-an-import.js";\n'
    assert _import_specs(src) == ["./a.js", "../b.js", "./c.js", "./d.js", "./e.js"]
    # an import() whose argument is not a literal would escape the scanner: there must be none
    for p in _js_files():
        code = _strip_comments(p.read_text(encoding="utf-8"))
        assert len(re.findall(r"\bimport\(", code)) == len(DYNAMIC_IMPORT_RE.findall(code)), _rel(p)


def test_every_import_resolves_to_a_file():
    seen = 0
    for p in _js_files():
        src = _strip_comments(p.read_text(encoding="utf-8"))
        for spec in _import_specs(src):
            seen += 1
            assert spec.startswith("./") or spec.startswith("../"), f"{_rel(p)}: non-relative import {spec!r}"
            target = (p.parent / spec).resolve()
            assert target.is_file(), f"{_rel(p)}: import {spec!r} does not exist"
            assert STATIC.resolve() in target.parents, f"{_rel(p)}: import {spec!r} leaves the static directory"
            assert target.suffix == ".js", f"{_rel(p)}: import {spec!r} must name the .js file"
    assert seen > 100          # the regex still finds the imports


def test_named_imports_exist_in_the_imported_module():
    """`import { a, b } from "./x.js"`: x.js must export a and b (a typo otherwise only fails in the browser)."""
    export_re = re.compile(r"\bexport\s+(?:async\s+)?(?:function\*?|const|let|class)\s+([A-Za-z_$][\w$]*)")
    export_list_re = re.compile(r"\bexport\s*\{([^}]*)\}")
    cache: dict[Path, set[str]] = {}

    def exports(path: Path) -> set[str]:
        if path not in cache:
            src = _strip_comments(path.read_text(encoding="utf-8"))
            names = set(export_re.findall(src))
            # `export const a = 1, b = 2;` style lists are not used; `export { a, b as c }` is
            for group in export_list_re.findall(src):
                for part in group.split(","):
                    part = part.strip()
                    if part:
                        names.add(part.split(" as ")[-1].strip())
            if re.search(r"\bexport\s+default\b", src):
                names.add("default")
            cache[path] = names
        return cache[path]

    named_re = re.compile(r"""\bimport\s*\{([^}]*)\}\s*from\s*(["'])([^"'\n]+)\2""")
    checked = 0
    for p in _js_files():
        src = _strip_comments(p.read_text(encoding="utf-8"))
        for names, _q, spec in named_re.findall(src):
            target = (p.parent / spec).resolve()
            have = exports(target)
            for part in names.split(","):
                name = part.strip().split(" as ")[0].strip()
                if not name:
                    continue
                checked += 1
                assert name in have, f"{_rel(p)}: {spec} has no export named {name!r}"
    assert checked > 200


def test_page_modules_export_a_default_with_mount():
    main = (JS / "main.js").read_text(encoding="utf-8")
    pages = re.findall(r'load: \(\) => import\("(\./pages/[\w/]+\.js)"\)', main)
    assert len(pages) >= 8
    for spec in pages:
        src = (JS / spec).read_text(encoding="utf-8")
        assert re.search(r"export default \{\s*async mount\(ctx\)", src), spec


# --- safety ------------------------------------------------------------------------------------------

SINKS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "createContextualFragment", "srcdoc",
         "DOMParser", "setHTMLUnsafe")


def test_markup_reaches_the_dom_only_through_core_dom():
    """innerHTML & co. may appear in core/dom.js only; that module accepts nothing but SafeHtml."""
    allowed = {"js/core/dom.js"}
    for p in _js_files():
        if _rel(p) in allowed:
            continue
        src = p.read_text(encoding="utf-8")
        for sink in SINKS:
            assert sink not in src, f"{_rel(p)} uses {sink}: build markup with html`` and core/dom.js render()/patch()/insert()"
        assert not re.search(r"\beval\s*\(|new Function\s*\(", src), _rel(p)
    dom = (JS / "core" / "dom.js").read_text(encoding="utf-8")
    assert "instanceof SafeHtml" in dom and "throw new TypeError" in dom
    # every assignment in dom.js goes through markup(), the SafeHtml gate
    assert len(re.findall(r"\.innerHTML\s*=", dom)) == len(re.findall(r"\.innerHTML\s*=\s*(?:s|markup\(safe\));", dom)) > 0
    assert re.findall(r"insertAdjacentHTML\((.*)\);", dom) == ["position, markup(safe)"]


def test_no_inline_event_handlers_in_markup():
    for p in [STATIC / "index.html", *_js_files()]:
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"""<[a-z][^>]*\son[a-z]+\s*=\s*["']""", src), f"{_rel(p)}: inline on…= handler"
        assert "javascript:" not in src, _rel(p)


# --- design tokens -----------------------------------------------------------------------------------

def _lum(hex_color: str) -> float:
    n = int(hex_color.lstrip("#"), 16)
    out = []
    for v in ((n >> 16) & 255, (n >> 8) & 255, n & 255):
        c = v / 255
        out.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def _contrast(a: str, b: str) -> float:
    x, y = _lum(a), _lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def _token_blocks() -> dict[str, dict[str, str]]:
    css = (CSS / "tokens.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def block(selector: str) -> dict[str, str]:
        i = css.index(selector)
        body = css[css.index("{", i) + 1:css.index("}", i)]
        return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))

    light = block(":root {")
    dark = block(':root[data-theme="dark"] {')
    auto = block(':root:not([data-theme="light"]) {')
    return {"light": light, "dark": dark, "auto": auto}


def test_dark_tokens_are_identical_in_both_places():
    b = _token_blocks()
    assert b["dark"] == b["auto"]
    assert b["dark"].keys() <= b["light"].keys() | {"color-scheme"}
    assert {k for k in b["light"] if k.startswith("--") and re.fullmatch(r"#[0-9A-Fa-f]{6}", b["light"][k])} == \
           {k for k in b["dark"] if re.fullmatch(r"#[0-9A-Fa-f]{6}", b["dark"][k])}


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_tokens_keep_aa_contrast(theme: str):
    t = _token_blocks()[theme]
    surfaces = ["--bg", "--surface", "--surface-2"]
    pairs: list[tuple[str, str]] = []
    for fg in ("--text", "--text-2", "--text-3", "--accent-text", "--rec-text", "--ok-text", "--warn-text", "--danger-text",
               "--accent", "--ok", "--warn", "--danger", "--rec"):
        pairs += [(fg, bg) for bg in surfaces]
    pairs += [(fg, "--surface-3") for fg in ("--text", "--text-2", "--text-3")]
    for k in ("accent", "rec", "ok", "warn", "danger"):
        pairs += [(f"--on-{k}", f"--{k}"), (f"--{k}-text", f"--{k}-soft"), ("--text", f"--{k}-soft"), ("--text-2", f"--{k}-soft")]
    pairs += [("--on-accent", "--accent-hover")]
    for i in range(12):
        pairs += [(f"--spk-{i}", bg) for bg in surfaces] + [(f"--spk-{i}", f"--spk-{i}-bg"), ("--text", f"--spk-{i}-bg")]
    bad = [(fg, bg, round(_contrast(t[fg], t[bg]), 2)) for fg, bg in pairs if _contrast(t[fg], t[bg]) < 4.5]
    assert not bad, bad


def test_css_uses_only_defined_tokens():
    defined = set(_token_blocks()["light"]) | set(re.findall(r"(--[\w-]+)\s*:", (CSS / "tokens.css").read_text(encoding="utf-8")))
    # set from code / per element: speaker colour, avatar size, seek position, scrollbar width kept while a dialog is open
    local = {"--ink", "--tint", "--sz", "--pct", "--sbw"}
    for p in CSS.glob("*.css"):
        used = set(re.findall(r"var\((--[\w-]+)", p.read_text(encoding="utf-8")))
        assert used <= defined | local, (p.name, sorted(used - defined - local))
    for p in _js_files():
        used = set(re.findall(r"var\((--[\w-]+)", p.read_text(encoding="utf-8")))
        assert used <= defined | local, (_rel(p), sorted(used - defined - local))


# --- node: DOM-free modules --------------------------------------------------------------------------

def _node(script: str) -> dict:
    res = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, encoding="utf-8",
                         timeout=60, cwd=str(JS))
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout)


def _uri(rel: str) -> str:
    return json.dumps((JS / rel).as_uri())


@needs_node
def test_core_modules_import_without_a_dom():
    mods = ["core/util.js", "core/format.js", "core/md.js", "core/html.js", "ui/icons.js", "ui/charts.js"]
    script = ("const out = {};"
              + "".join(f"out[{json.dumps(m)}] = Object.keys(await import({_uri(m)})).length;" for m in mods)
              + "out.dom = typeof document + typeof window; process.stdout.write(JSON.stringify(out));")
    out = _node(script)
    assert out.pop("dom") == "undefinedundefined"
    assert all(n > 0 for n in out.values()), out


@needs_node
def test_every_module_can_be_imported_without_touching_the_dom():
    """No module may read document / window at import time: boot order then never matters and node can load them all."""
    mods = [_rel(p)[3:] for p in _js_files() if _rel(p).startswith("js/") and _rel(p) != "js/main.js"]
    script = ("const bad = [];"
              f"for (const m of {json.dumps(mods)}) {{ try {{ await import(new URL(m, {json.dumps(JS.as_uri() + '/')}).href); }}"
              " catch (e) { bad.push([m, String(e && e.message)]); } }"
              "process.stdout.write(JSON.stringify({bad}));")
    assert _node(script)["bad"] == []


@needs_node
def test_util():
    out = _node(f"""
      import * as u from {_uri("core/util.js")};
      const d = u.debounce(() => {{}}, 5);
      const o = {{}}; u.setPath(o, "a.b.c", 1); u.setPath(o, "a.d", null);
      process.stdout.write(JSON.stringify({{
        esc: u.esc(`<a href="x" title='y'>&</a>`), escNull: u.esc(null) + u.esc(undefined), escNum: u.esc(0),
        qs: u.qs({{q: "a b&c", limit: 8, x: null, y: undefined, z: "", f: false, zero: 0}}), qsEmpty: u.qs({{}}) + u.qs(null),
        clamp: [u.clamp(5, 0, 3), u.clamp(-1, 0, 3), u.clamp(2, 0, 3)],
        num: [u.num("3.5"), u.num(""), u.num(null), u.num("x", 7), u.num(Infinity, 0), u.num(0)],
        eq: [u.deepEqual({{a: [1, {{b: 2}}]}}, {{a: [1, {{b: 2}}]}}), u.deepEqual([1, 2], [2, 1]), u.deepEqual(null, {{}}), u.deepEqual("", null), u.deepEqual([], {{}})],
        get: [u.getPath({{a: {{b: {{c: 3}}}}}}, "a.b.c"), u.getPath({{a: null}}, "a.b") === undefined, u.getPath({{}}, "x") === undefined],
        set: o,
        group: [...u.groupBy([1, 2, 3, 4], (n) => n % 2).entries()],
        diff: u.diffChars("今天去星和计画开会", "今天去星河计划开会"),
        diffSame: u.diffChars("abc", "abc"), diffIns: u.diffChars("ac", "abc"), diffDel: u.diffChars("abc", "ac"),
        diffEmpty: [u.diffChars("", ""), u.diffChars("", "x"), u.diffChars("x", "")],
        diffBig: u.diffChars("a".repeat(600), "b".repeat(600)),
        hash: [u.hashStr("张三") === u.hashStr("张三"), u.hashStr("") === 0, Number.isInteger(u.hashStr("李四"))],
        uid: u.uid("x") !== u.uid("x"), deb: typeof d.cancel + typeof d.flush,
      }}));""")
    assert out["esc"] == "&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;" and out["escNull"] == "" and out["escNum"] == "0"
    assert out["qs"] == "?q=a%20b%26c&limit=8&f=false&zero=0" and out["qsEmpty"] == ""
    assert out["clamp"] == [3, 0, 2]
    assert out["num"] == [3.5, None, None, 7, 0, 0]
    assert out["eq"] == [True, False, False, False, False]
    assert out["get"] == [3, True, True] and out["set"] == {"a": {"b": {"c": 1}, "d": None}}
    assert out["group"] == [[1, [1, 3]], [0, [2, 4]]]
    assert out["diff"] == [{"op": "eq", "a": "今天去星", "b": "今天去星"}, {"op": "rep", "a": "和", "b": "河"},
                           {"op": "eq", "a": "计", "b": "计"}, {"op": "rep", "a": "画", "b": "划"}, {"op": "eq", "a": "开会", "b": "开会"}]
    assert out["diffSame"] == [{"op": "eq", "a": "abc", "b": "abc"}]
    assert out["diffIns"] == [{"op": "eq", "a": "a", "b": "a"}, {"op": "ins", "a": "", "b": "b"}, {"op": "eq", "a": "c", "b": "c"}]
    assert out["diffDel"] == [{"op": "eq", "a": "a", "b": "a"}, {"op": "del", "a": "b", "b": ""}, {"op": "eq", "a": "c", "b": "c"}]
    assert out["diffEmpty"] == [[], [{"op": "ins", "a": "", "b": "x"}], [{"op": "del", "a": "x", "b": ""}]]
    assert out["diffBig"] is None                                  # too long for the table: callers fall back
    assert out["hash"] == [True, True, True] and out["uid"] is True and out["deb"] == "functionfunction"
    # every diff reassembles both inputs
    assert "".join(op["a"] for op in out["diff"]) == "今天去星和计画开会" and "".join(op["b"] for op in out["diff"]) == "今天去星河计划开会"


@needs_node
def test_format():
    out = _node(f"""
      import * as f from {_uri("core/format.js")};
      const now = new Date(2026, 10, 11, 9, 30);          // Wednesday 2026-11-11
      const p = (s) => f.parseDayQuery(s, now);
      process.stdout.write(JSON.stringify({{
        min: [f.fmtMin(0.4), f.fmtMin(59.6), f.fmtMin(60), f.fmtMin(135)],
        bytes: [f.fmtBytes(null), f.fmtBytes(0), f.fmtBytes(1023), f.fmtBytes(1536), f.fmtBytes(28.2 * 1024 ** 2), f.fmtBytes(3 * 1024 ** 4)],
        dt: [f.fmtDT("2026-11-03T09:12:33"), f.fmtDT("2026-11-03 09:12:33.5"), f.fmtDT(""), f.fmtDT(null)],
        time: [f.fmtTime("2026-11-03T09:12:33"), f.fmtTime(null)],
        clock: [f.fmtClock(0), f.fmtClock(65.9), f.fmtClock(3723), f.fmtClock(-4), f.fmtClock("x")],
        dur: [f.fmtDur(45), f.fmtDur(59.6), f.fmtDur(720), f.fmtDur(3600), f.fmtDur(3900)],
        db: [f.fmtDb(-32.4), f.fmtDb(0), f.fmtDb(3), f.fmtDb(null), f.fmtDb("x")],
        today: f.today(now), isDay: [f.isDay("2026-11-03"), f.isDay("2026-11-3"), f.isDay(""), f.isDay(null)],
        add: [f.addDays("2026-11-01", -1), f.addDays("2026-12-31", 1), f.addDays("2026-03-01", -1)],
        label: [f.dayLabel("2026-11-11", now), f.dayLabel("2026-11-10", now), f.dayLabel("2026-11-09", now), f.dayLabel("2026-11-01", now), f.dayLabel("nope", now)],
        title: [f.dayTitle("2026-11-03", now), f.dayTitle("2025-12-25", now), f.dayTitle("x", now)],
        parse: [p("今天"), p("昨天"), p("前天"), p("11-03"), p("11/3"), p("11月3日"), p("2026-11-03"), p("2025年12月25日"), p("12-25"), p("3号"), p("28号"),
                p("02-30"), p("13-01"), p("hello"), p(""), p(" 昨天 ")],
        sec: [f.secOfDay("2026-11-03T09:12:33"), f.secOfDay("2026-11-03 00:00"), Number.isNaN(f.secOfDay("2026-11-03"))],
        ms: [f.tsMs("2026-11-03 09:12:33") === new Date(2026, 10, 3, 9, 12, 33).getTime(), Number.isNaN(f.tsMs("")), Number.isNaN(f.tsMs("zzz"))],
      }}));""")
    assert out["min"] == ["0 分钟", "60 分钟", "1.0 小时", "2.3 小时"]
    assert out["bytes"] == ["-", "0 B", "1023 B", "1.5 KB", "28.2 MB", "3.0 TB"]
    assert out["dt"] == ["2026-11-03 09:12", "2026-11-03 09:12", "-", "-"] and out["time"] == ["09:12", "-"]
    assert out["clock"] == ["0:00", "1:05", "1:02:03", "0:00", "0:00"]
    assert out["dur"] == ["45 秒", "1 分", "12 分", "1 小时", "1 小时 5 分"]
    assert out["db"] == ["−32 dB", "0 dB", "3 dB", "", ""]
    assert out["today"] == "2026-11-11" and out["isDay"] == [True, False, False, False]
    assert out["add"] == ["2026-10-31", "2027-01-01", "2026-02-28"]
    assert out["label"] == ["今天 周三", "昨天 周二", "前天 周一", "周日", ""]
    assert out["title"] == ["11月3日", "2025年12月25日", "x"]
    assert out["parse"] == ["2026-11-11", "2026-11-10", "2026-11-09", "2026-11-03", "2026-11-03", "2026-11-03", "2026-11-03",
                            "2025-12-25", "2025-12-25", "2026-11-03", "2026-10-28", None, None, None, None, "2026-11-10"]
    assert out["sec"] == [9 * 3600 + 12 * 60 + 33, 0, True] and out["ms"] == [True, True, True]


@needs_node
def test_html_template_escapes_everything_but_safe_html():
    out = _node(f"""
      import {{ html, raw, join, when, attrs, cls, SafeHtml, isSafe }} from {_uri("core/html.js")};
      const evil = `<img src=x onerror=alert(1)>"'&`;
      const s = (v) => String(v);
      process.stdout.write(JSON.stringify({{
        text: s(html`<p title="${{evil}}">${{evil}}</p>`),
        nested: s(html`<ul>${{[1, 2].map((n) => html`<li>${{n}}</li>`)}}</ul>`),
        mixedArray: s(html`${{["<b>", html`<i>ok</i>`, null, false, 3]}}`),
        nothing: s(html`[${{null}}${{undefined}}${{false}}${{true}}${{""}}${{0}}]`),
        raw: s(html`${{raw("<hr>")}}${{raw(null)}}${{raw(html`<b>x</b>`)}}`),
        join: s(join([html`<a>1</a>`, "<2>", null, html`<a>3</a>`], " · ")), joinSafeSep: s(join(["a", "b"], raw("<br>"))),
        when: s(when(true, html`<b>y</b>`)) + s(when(0, html`<b>n</b>`)) + s(when(1, () => html`<i>fn</i>`)),
        attrs: s(attrs({{disabled: true, hidden: false, title: evil, "data-id": 3, nope: null, "bad name": 1, onclick: undefined}})),
        cls: cls("utt", {{far: true, hl: false}}, null, "x"),
        safe: [isSafe(html`a`), isSafe("a"), html`a` instanceof SafeHtml, Object.isFrozen(html`a`)],
        obj: s(html`${{{{toString() {{ return "<o>"; }}}}}}`),
      }}));""")
    esc_evil = "&lt;img src=x onerror=alert(1)&gt;&quot;&#39;&amp;"
    assert out["text"] == f'<p title="{esc_evil}">{esc_evil}</p>'
    assert out["nested"] == "<ul><li>1</li><li>2</li></ul>"
    assert out["mixedArray"] == "&lt;b&gt;<i>ok</i>3"
    assert out["nothing"] == "[0]"
    assert out["raw"] == "<hr><b>x</b>"
    assert out["join"] == "<a>1</a> · &lt;2&gt; · <a>3</a>" and out["joinSafeSep"] == "a<br>b"
    assert out["when"] == "<b>y</b><i>fn</i>"
    assert out["attrs"] == f' disabled title="{esc_evil}" data-id="3"'
    assert out["cls"] == "utt far x"
    assert out["safe"] == [True, False, True, True]
    assert out["obj"] == "&lt;o&gt;"                               # only real SafeHtml instances pass unescaped


@needs_node
def test_md():
    cases = {
        "xss_list": "- <input onfocus=alert(1) autofocus> hi",
        "xss_para": "<script>alert(1)</script> **b** *i* `c<d>`",
        "task": "- [x] <b>done</b>\n- [ ] todo",
        "table": "| a | <i> |\n|---|---|\n| <u> | b |",
        "heading": "###### deep <x>\n# top",
        "code": "```\n<b>raw</b>\n```",
        "quote": "> quoted **x**\n> more",
        "link": "[t](https://example.com/a?b=1) [bad](javascript:alert(1))",
        "ordered": "1. one\n2) two\n   continued",
        "lone_pipe": "| not a table\nnext",
        "hr": "a\n\n---\n\nb",
        "empty": "",
    }
    out = _node(f"""
      import {{ md }} from {_uri("core/md.js")};
      const cases = {json.dumps(cases)};
      const res = {{}};
      for (const [k, v] of Object.entries(cases)) res[k] = md(v);
      res.nullish = md(null) + md(undefined);
      process.stdout.write(JSON.stringify(res));""")
    for key, html in out.items():
        assert "<script" not in html and "<input onfocus" not in html and "<b>" not in html and "<u>" not in html and "<x>" not in html, (key, html)
    assert out["xss_list"] == "<ul><li>&lt;input onfocus=alert(1) autofocus&gt; hi</li></ul>"
    assert "<strong>b</strong>" in out["xss_para"] and "<em>i</em>" in out["xss_para"] and "<code>c&lt;d&gt;</code>" in out["xss_para"]
    assert '<input type="checkbox" disabled checked>&lt;b&gt;done&lt;/b&gt;' in out["task"] and '<input type="checkbox" disabled >todo' in out["task"]
    assert "<table>" in out["table"] and "<th>&lt;i&gt;</th>" in out["table"] and "<td>&lt;u&gt;</td>" in out["table"]
    assert out["heading"] == "<h3>deep &lt;x&gt;</h3>\n<h1>top</h1>"
    assert out["code"] == "<pre><code>&lt;b&gt;raw&lt;/b&gt;</code></pre>"
    assert out["quote"] == "<blockquote><p>quoted <strong>x</strong><br>more</p></blockquote>"
    assert '<a href="https://example.com/a?b=1" target="_blank" rel="noopener noreferrer">t</a>' in out["link"] and "href=\"javascript" not in out["link"]
    assert out["ordered"] == "<ol><li>one</li><li>two continued</li></ol>"
    assert out["lone_pipe"] == "<p>| not a table</p>\n<p>next</p>"       # used to loop forever
    assert out["hr"] == "<p>a</p>\n<hr>\n<p>b</p>"
    assert out["empty"] == "" and out["nullish"] == ""


@needs_node
def test_icons_and_charts_are_safe_strings():
    out = _node(f"""
      import {{ icon, ICON_NAMES }} from {_uri("ui/icons.js")};
      import {{ sparkline, bars, hbars, ring, dayStrip }} from {_uri("ui/charts.js")};
      const s = (v) => String(v);
      const evil = `"><script>x</script>`;
      process.stdout.write(JSON.stringify({{
        names: ICON_NAMES,
        icon: s(icon("play", 16)), labelled: s(icon("alert", 18, {{label: evil}})), unknown: s(icon("no-such-icon")).length > 0,
        spark: s(sparkline([1, 3, 2, 5], {{label: evil}})), sparkShort: s(sparkline([1])), sparkZero: s(sparkline([0, 0, 0])),
        bars: s(bars([{{label: evil, value: 3, title: evil}}, {{label: "b", value: 0, on: true}}])), barsEmpty: s(bars([])),
        hbars: s(hbars([{{label: evil, value: 2, text: evil, cls: "c1"}}, {{label: "zero", value: 0}}])),
        ring: s(ring(150)) + s(ring(-3)),
        strip: s(dayStrip([{{start: 9 * 3600 + 60, dur: 30, cls: "c1", name: evil}}, {{start: 9 * 3600 + 500, dur: 400, cls: "c-me", name: "我"}}, {{start: NaN, dur: 5, cls: "c2", name: "x"}}])),
        stripEmpty: s(dayStrip([])),
      }}));""")
    assert len(out["names"]) == len(set(out["names"])) > 40
    assert out["icon"].startswith("<svg") and 'aria-hidden="true"' in out["icon"] and 'width="16"' in out["icon"]
    assert 'role="img"' in out["labelled"] and "<script>" not in out["labelled"] and out["unknown"] is True
    for key in ("spark", "bars", "hbars", "strip"):
        assert "<script>" not in out[key] and "&lt;script&gt;" in out[key], key
    assert out["sparkShort"] == "" and "NaN" not in out["sparkZero"] and out["barsEmpty"] == "" and out["stripEmpty"] == ""
    assert "zero" not in out["hbars"]
    assert "NaN" not in out["ring"] and "NaN" not in out["strip"] and "NaN" not in out["bars"]
    assert 'data-t="09:00"' in out["strip"] and 'data-t="09:10"' in out["strip"] and out["strip"].count('class="strip-b"') == 2


@needs_node
def test_every_icon_name_used_in_the_code_exists():
    names = set(_node(f"import {{ ICON_NAMES }} from {_uri('ui/icons.js')}; process.stdout.write(JSON.stringify({{n: ICON_NAMES}}));")["n"])
    used: dict[str, str] = {}
    for p in _js_files():
        if _rel(p) == "js/ui/icons.js":
            continue
        src = p.read_text(encoding="utf-8")
        for name in re.findall(r"""\bicon\(\s*["']([\w-]+)["']""", src) + re.findall(r"""\bicon:\s*["']([\w-]+)["']""", src):
            used[name] = _rel(p)
    assert len(used) > 25
    missing = {n: f for n, f in used.items() if n not in names}
    assert not missing, missing


@needs_node
def test_pure_helpers_of_the_ui_layer():
    out = _node(f"""
      import {{ colorClassFor, rememberSpeakers, PALETTE_SIZE }} from {_uri("ui/speaker.js")};
      import {{ matches }} from {_uri("ui/palette.js")};
      import {{ backoffDelay }} from {_uri("core/events.js")};
      import {{ sessionsOf, SESSION_GAP_S }} from {_uri("pages/timeline.js")};
      import {{ groupTurns }} from {_uri("ui/utterance.js")};
      import {{ parseHash, hrefFor, definePage, alias }} from {_uri("core/router.js")};
      import {{ jobSummary }} from {_uri("core/status.js")};
      import {{ extraFields, SECTIONS }} from {_uri("pages/settings/schema.js")};
      rememberSpeakers([{{id: 14, name: "李四"}}]);
      const u = (id, t, extra = {{}}) => ({{id, abs_start: `2026-11-03T${{t}}`, time: t, start_s: 0, end_s: 2, speaker_id: null, speaker: "未知", segment_id: 1, cluster: 0, ...extra}});
      const day = [u(1, "09:00:00", {{speaker_id: 3}}), u(2, "09:00:10", {{speaker_id: 3}}), u(3, "09:00:20"), u(4, "09:00:25", {{cluster: 1}}), u(5, "09:02:00", {{speaker_id: 3}}),
                   u(6, "09:04:00", {{speaker_id: 3}}), u(7, "09:09:03", {{speaker_id: 3}}), u(8, "23:59:59", {{speaker_id: 4}})];
      definePage("transcript", {{title: "t", aliases: ["timeline"]}}); definePage("settings", {{title: "s"}}); definePage("dashboard", {{title: "d"}}); alias("logs", "settings", ["logs"]);
      const ph = (h) => {{ const r = parseHash(h); return [r.page, r.params, [...r.query.entries()], r.known]; }};
      const extra = extraFields({{performance: {{mode: "balanced", threads: 0, defer_on_battery: false, weird: [1]}}, mimo: {{chat_model: "x", listen_model: "y", api_key: ""}}, audio: {{new_nullable: null}}}},
                                {{audio: {{new_nullable: 1.5}}}}, {{"mimo.api_key": {{set: true}}, "push.new.token": {{set: false}}}});
      process.stdout.write(JSON.stringify({{
        colors: [colorClassFor({{id: 1}}), colorClassFor({{id: 12}}), colorClassFor({{id: 13}}), colorClassFor({{name: "李四"}}), colorClassFor({{id: 5, isSelf: true}}), colorClassFor({{known: false, id: 2}}), colorClassFor({{}}), colorClassFor({{id: 0}})],
        nameStable: colorClassFor({{name: "王敏"}}) === colorClassFor({{name: "王敏"}}), size: PALETTE_SIZE,
        match: [matches({{title: "开始录音", keywords: "record start"}}, "录音"), matches({{title: "开始录音", keywords: "record start"}}, "REC start"), matches({{title: "开始录音"}}, "日报"), matches({{title: "x"}}, "  ")],
        backoff: [0, 1, 2, 3, 4, 5, 9].map((n) => backoffDelay(n, 0.5)), jitter: [backoffDelay(3, 0), backoffDelay(3, 1)],
        gap: SESSION_GAP_S, sessions: sessionsOf(day).map((s) => [s.key, s.start, s.end, s.utts.map((x) => x.id), Math.round(s.seconds)]),
        bySid: sessionsOf(day.map((x, i) => ({{...x, session_id: i < 3 ? 7 : 8}}))).map((s) => s.utts.length), none: sessionsOf([]),
        turns: groupTurns(day.slice(0, 6)).map((t) => t.map((x) => x.id)),
        hash: [ph("#/transcript/2026-11-10?u=42"), ph("#/timeline/2026-11-10"), ph(""), ph("#/nope/x"), ph("#/logs"), ph("#/settings/%E5%BD%95"), ph("#/settings/%E0%A4%A")],
        href: [hrefFor("transcript", "2026-11-10", {{u: 42}}), hrefFor("daily"), hrefFor("daily", null), hrefFor("settings", "a b", {{q: ""}})],
        jobs: [jobSummary("process", {{processed: 2, utterances: 9, failed: 1}}), jobSummary("nope", {{}}), jobSummary("process", null), jobSummary("loudness", {{computed: 3, threshold: -40, below: 2}})],
        extra, sectionIds: SECTIONS.map((s) => s.id), dupPaths: SECTIONS.flatMap((s) => s.fields.map((f) => f.path)).filter((p, i, a) => a.indexOf(p) !== i),
      }}));""")
    assert out["colors"] == ["c0", "c11", "c0", "c1", "c-me", "c-unk", "c-unk", "c11"] and out["nameStable"] is True and out["size"] == 12
    assert out["match"] == [True, True, False, True]
    assert out["backoff"] == [1000, 2000, 4000, 8000, 16000, 30000, 30000] and out["jitter"] == [6800, 9200]
    # 5-minute gap: 09:04:02 -> 09:09:03 is 5 min 1 s (new session); 09:02:02 -> 09:04:00 is not
    assert out["gap"] == 300
    assert out["sessions"] == [["1", "09:00", "09:04", [1, 2, 3, 4, 5, 6], 242], ["7", "09:09", "09:09", [7], 2], ["8", "23:59", "23:59", [8], 2]]
    assert out["bySid"] == [3, 5] and out["none"] == []
    # same speaker within a minute -> one block; two unidentified voices share one block; a 95 s pause splits
    assert out["turns"] == [[1, 2], [3, 4], [5], [6]]
    assert out["hash"][0] == ["transcript", ["2026-11-10"], [["u", "42"]], True]
    assert out["hash"][1] == ["transcript", ["2026-11-10"], [], True]
    assert out["hash"][2] == ["dashboard", [], [], True] and out["hash"][3] == ["dashboard", [], [], False]
    assert out["hash"][4] == ["settings", ["logs"], [], True] and out["hash"][5] == ["settings", ["录"], [], True]
    assert out["hash"][6][0] == "settings"                          # a malformed % escape must not throw
    assert out["href"] == ["#/transcript/2026-11-10?u=42", "#/daily", "#/daily", "#/settings/a%20b"]
    assert out["jobs"] == ["：处理 2 段，9 条语句，1 段失败", "", "", "：补算 3 句音量，2 句低于门槛 −40 dB"]
    extra = out["extra"]
    assert [f["path"] for f in extra["other"]] == ["performance.mode", "performance.threads", "performance.defer_on_battery"]
    assert [f["type"] for f in extra["other"]] == ["text", "number", "bool"] and all(f["advanced"] for f in extra["other"])
    assert [f["path"] for f in extra["mimo"]] == ["mimo.listen_model"]          # chat_model is listed, api_key is a secret
    assert extra["audio"] == [{"path": "audio.new_nullable", "type": "number", "label": "audio.new_nullable", "step": "any", "nullable": True, "advanced": True, "auto": True}]
    assert [f["path"] for f in extra["daily"]] == ["push.new.token"] and extra["daily"][0]["type"] == "secret"
    assert len(out["sectionIds"]) == len(set(out["sectionIds"])) and out["dupPaths"] == []


@needs_node
def test_api_helpers_with_a_fake_fetch():
    out = _node(f"""
      const calls = [];
      const enc = new TextEncoder();
      const stream = (chunks) => new ReadableStream({{ start(c) {{ for (const x of chunks) c.enqueue(enc.encode(x)); c.close(); }} }});
      const json = (status, body) => new Response(JSON.stringify(body), {{status, headers: {{"content-type": "application/json"}}}});
      const routes = {{
        "/ok": () => json(200, {{a: 1}}),
        "/bad": () => json(400, {{error: "不行", mismatch: true, score: 0.3}}),
        "/boom": () => new Response("oops", {{status: 500, statusText: "Server Error", headers: {{"content-type": "text/plain"}}}}),
        "/file": () => new Response("RIFF", {{status: 200, headers: {{"content-type": "audio/wav"}}}}),
        "/down": () => {{ throw new TypeError("fetch failed"); }},
        "/chat": () => new Response(stream(['{{"type":"start","model":"m"}}\\n{{"type":"to', 'ol","name":"s"}}\\nnot json\\n\\n{{"type":"delta","text":"你"}}\\n', '{{"type":"answer","text":"好"}}']), {{status: 200}}),
        "/chat409": () => json(409, {{error: "没有配置 key"}}),
      }};
      globalThis.fetch = async (url, init = {{}}) => {{ calls.push([url, init.method || "GET", init.headers && init.headers["Content-Type"], typeof init.body]); return routes[String(url).split("?")[0]](); }};
      const {{ api, ApiError, ndjson, isMismatch, isAbort, audioUrl, readEventStream }} = await import({_uri("core/api.js")});
      const err = async (p) => {{ try {{ await p; return null; }} catch (e) {{ return {{name: e.name, msg: e.message, status: e.status, data: e.data, network: e.network, mismatch: isMismatch(e), api: e instanceof ApiError}}; }} }};
      const out = {{}};
      out.ok = await api("/ok", {{query: {{q: "a b", n: 2, skip: null}}}});
      out.post = await api("/ok", {{method: "POST", json: {{x: 1}}}});
      out.bad = await err(api("/bad"));
      out.boom = await err(api("/boom"));
      out.down = await err(api("/down"));
      out.file = (await api("/file")) instanceof Response;
      const ctl = new AbortController(); ctl.abort();
      globalThis.fetch = async (url, init = {{}}) => {{ if (init.signal && init.signal.aborted) {{ const e = new Error("aborted"); e.name = "AbortError"; throw e; }} calls.push([url]); return routes[String(url).split("?")[0]](); }};
      const ab = await err(api("/ok", {{signal: ctl.signal}}));
      out.abort = [ab.name, ab.api, isAbort({{name: "AbortError"}}), isAbort(new Error("x"))];
      out.events = []; for await (const ev of ndjson("/chat", {{json: {{question: "q"}}}})) out.events.push(ev);
      out.chat409 = await err((async () => {{ for await (const ev of ndjson("/chat409", {{json: {{}}}})) void ev; }})());
      out.calls = calls.slice(0, 2);
      out.audio = [audioUrl.utterance(7), audioUrl.utterance(7, 0), audioUrl.segment(3, 1.5, 2), audioUrl.sample(4), audioUrl.negative(5)];
      // server-sent events: chunks split anywhere, CRLF, comments, multi-line data, sticky ids
      const sse = [];
      let ticks = 0;
      await readEventStream(new Response(stream(['data: {{"type":"hello","seq":5}}\\n\\n: keep', '-alive\\n\\nid: 6\\r\\ndata: {{"a":', '1}}\\r\\n\\r\\nevent: job\\ndata: line1\\ndata: line2\\n\\ndata: tail-without-blank-line\\n'])),
                            {{onEvent: (e) => sse.push(e), onActivity: () => ticks++}});
      out.sse = sse; out.ticks = ticks > 0;
      process.stdout.write(JSON.stringify(out));""")
    assert out["ok"] == {"a": 1} and out["post"] == {"a": 1} and out["file"] is True
    assert out["calls"] == [["/ok?q=a%20b&n=2", "GET", None, "undefined"], ["/ok", "POST", "application/json", "string"]]
    assert out["bad"] == {"name": "ApiError", "msg": "不行", "status": 400, "data": {"error": "不行", "mismatch": True, "score": 0.3},
                          "network": False, "mismatch": True, "api": True}
    assert out["boom"]["msg"] == "500 Server Error" and out["boom"]["status"] == 500 and out["boom"]["data"] is None
    assert out["down"]["network"] is True and out["down"]["status"] == 0 and "无法连接" in out["down"]["msg"]
    assert out["abort"] == ["AbortError", False, True, False]            # the caller's own abort is not turned into an ApiError
    assert out["events"] == [{"type": "start", "model": "m"}, {"type": "tool", "name": "s"}, {"type": "delta", "text": "你"},
                             {"type": "answer", "text": "好"}]            # split lines joined, the malformed line skipped
    assert out["chat409"]["msg"] == "没有配置 key" and out["chat409"]["status"] == 409
    assert out["audio"] == ["/api/audio/utterance/7?pad=0.3", "/api/audio/utterance/7?pad=0", "/api/audio/segment/3?from=1.5&to=2",
                            "/api/speakers/samples/4/audio", "/api/speakers/negatives/5/audio"]
    assert out["sse"] == [{"event": "message", "data": '{"type":"hello","seq":5}', "id": ""},
                          {"event": "message", "data": '{"a":1}', "id": "6"},
                          {"event": "job", "data": "line1\nline2", "id": "6"}]       # an unterminated last event is not dispatched
    assert out["ticks"] is True


@needs_node
def test_live_events_fall_back_to_polling_and_resume_with_the_last_id():
    """SSE client: hello + events re-emitted on the bus; a dropped stream -> "poll" mode and a reconnect that asks for
    the events after the last sequence number; a server without the endpoint (404) -> polling, no reconnect storm."""
    out = _node(f"""
      globalThis.window = {{ addEventListener() {{}} }};
      const enc = new TextEncoder();
      const calls = [];
      const first = () => new Response(new ReadableStream({{ start(c) {{
        c.enqueue(enc.encode('data: {{"type":"hello","seq":5,"inbox":{{"open":2}},"gap":false}}\\n\\n: keep-alive\\n\\n'));
        c.enqueue(enc.encode('id: 6\\ndata: {{"seq":6,"id":99,"type":"segment","status":"new"}}\\n\\nid: 7\\ndata: {{"seq":7,"type":"utterances","day":"2026-11-03","ids":[1,2]}}\\n\\n'));
        c.close();
      }} }}), {{status: 200, headers: {{"content-type": "text/event-stream; charset=utf-8"}}}});
      const missing = () => new Response('{{"error":"x"}}', {{status: 404, headers: {{"content-type": "application/json"}}}});
      const queue = [first, missing];
      globalThis.fetch = async (url, init = {{}}) => {{ calls.push([String(url), (init.headers || {{}})["Last-Event-ID"] || null]); return (queue.shift() || missing)(); }};
      const bus = await import({_uri("core/bus.js")});
      const {{ store }} = await import({_uri("core/store.js")});
      const {{ startLive, liveMode }} = await import({_uri("core/events.js")});
      const seen = [], modes = [], typed = [];
      let resync = 0;
      bus.on("live", (e) => seen.push([e.type, e.seq ?? null, e.id ?? null]));
      bus.on("live:mode", (m) => modes.push(m));
      bus.on("live:segment", (e) => typed.push(e.status));
      bus.on("live:utterances", (e) => typed.push(e.day));
      bus.on("resync", () => resync++);
      startLive(); startLive();
      await new Promise((r) => setTimeout(r, 2200));          // first stream ends at once; the reconnect comes ~1 s later
      process.stdout.write(JSON.stringify({{calls, seen, modes, typed, resync, mode: liveMode(), stored: store.get("live")}}));
      process.exit(0);""")
    assert out["seen"] == [["hello", 5, None], ["segment", 6, 99], ["utterances", 7, None]]
    assert out["typed"] == ["new", "2026-11-03"]
    assert out["modes"] == ["sse", "poll"] and out["mode"] == "poll" and out["stored"] == "poll"
    # one reconnect, resuming after bus sequence 7 (the event's own "id": 99 is a segment id, not the resume point)
    assert out["calls"] == [["/api/events", None], ["/api/events?since=7", "7"]]
    assert out["resync"] == 0                                    # only a hello after the first one (or a gap) asks pages to reload


# --- the server still serves it ----------------------------------------------------------------------

def test_server_serves_the_module_tree(cfg_factory, tmp_path: Path):
    from fastapi.testclient import TestClient

    from audio_note.runtime import Runtime
    from audio_note.ui.app import create_app

    rt = Runtime(cfg_factory({"memory": {"retrieval": "fts", "embedding_model": "hash"}}))
    try:
        client = TestClient(create_app(rt))
        r = client.get("/")
        assert r.status_code == 200 and "audio-note" in r.text and "__V__" not in r.text
        refs = re.findall(r'(?:href|src)="(/static/[^"]+)"', r.text)
        assert any(ref.startswith("/static/app.js?v=") for ref in refs)
        for ref in refs:
            a = client.get(ref)
            assert a.status_code == 200, ref
            if ref.split("?")[0].endswith(".js"):
                assert "javascript" in a.headers["content-type"], (ref, a.headers["content-type"])     # module scripts need a JS MIME type
            if ref.split("?")[0].endswith(".css"):
                assert a.headers["content-type"].startswith("text/css"), ref
        for path in ("/static/css/tokens.css", "/static/js/pages/timeline.js", "/static/js/pages/settings/schema.js"):
            assert client.get(path).status_code == 200, path
        assert client.get("/static/js/nope.js").status_code == 404
    finally:
        rt.shutdown()
        import logging

        from audio_note.logging_setup import _remove_own_handlers

        _remove_own_handlers(logging.getLogger())
