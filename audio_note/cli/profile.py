"""Profile commands: `profile list|create|use|delete` (one profile = its own recordings, records, speakers)."""
from __future__ import annotations

import argparse
from typing import Any

from rich.table import Table

from ..config import (DEFAULT_PROFILE, Config, create_profile, delete_profile, list_profiles, profile_config_path,
                      set_current_profile)
from ._late import Late, handler
from .helpers import CLIError, add_json_flag, emit_json, wants_json

console = Late("console")
load_config = Late("load_config")


def cmd_profile_list(cfg: Config, args: argparse.Namespace) -> int:
    from ..runtime import profile_label

    cur = cfg.get("_profile")
    rows: list[dict[str, Any]] = []
    for name in list_profiles():
        row: dict[str, Any] = {"name": name, "label": profile_label(name), "current": name == cur, "data_dir": None,
                               "utterances": None, "days": None, "speakers": None, "last_day": None, "error": None}
        try:
            c = load_config(profile=name)
            row.update(_profile_counts(c) or {})
            row["data_dir"] = str(c.data_dir)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        rows.append(row)
    if wants_json(args):
        emit_json({"current": cur, "config_path": cfg.get("_config_path"), "profiles": rows})
        return 0
    t = Table(title="档案（每个档案有自己的录音、记录、说话人；API key 等共享设置放根目录 config.yaml）")
    for col in ("", "档案", "数据目录", "记录"):
        t.add_column(col)
    for row in rows:
        if row["error"] is not None:
            info, data_dir = f"[red]{row['error']}[/red]", "?"
        else:
            info = f"{row['utterances']} 条 / {row['days']} 天" if row["utterances"] is not None else "（还没有数据）"
            data_dir = row["data_dir"]
        t.add_row("●" if row["current"] else "", row["label"], data_dir, info)
    console.print(t)
    if cur is None:
        console.print(f"当前使用的是自定义配置 {cfg.get('_config_path')}（--config / AUDIO_NOTE_CONFIG），不属于任何档案。")
    else:
        console.print("● = 当前档案。切换：audio-note profile use <名字>；一次性：audio-note --profile <名字> <命令>")
    return 0


def _profile_counts(c: Config) -> dict[str, int] | None:
    """Row counts of a profile's database without creating it (None when the db does not exist)."""
    import sqlite3

    if not c.db_path.exists():
        return None
    con = sqlite3.connect(f"file:{c.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT day), MAX(day) FROM utterances").fetchone()
        spk = con.execute("SELECT COUNT(*) FROM speakers").fetchone()
        return {"utterances": int(row[0] or 0), "days": int(row[1] or 0), "speakers": int(spk[0] or 0), "last_day": row[2]}
    except sqlite3.Error:
        return None
    finally:
        con.close()


def cmd_profile_create(cfg: Config, args: argparse.Namespace) -> int:
    src = None
    if args.copy_from:
        src = profile_config_path(args.copy_from)
        if not src.exists():
            raise CLIError(f"档案不存在: {args.copy_from}")
    try:
        path = create_profile(args.name, copy_from=src)
    except (ValueError, FileExistsError, OSError) as e:
        raise CLIError(str(e)) from e
    console.print(f"已创建档案 [bold]{args.name}[/bold]：{path}")
    if args.use:
        set_current_profile(args.name)
        console.print(f"已切换到 {args.name}。")
    else:
        console.print(f"切换：audio-note profile use {args.name}")
    return 0


def cmd_profile_use(cfg: Config, args: argparse.Namespace) -> int:
    try:
        set_current_profile(args.name)
    except ValueError as e:
        raise CLIError(str(e)) from e
    c = load_config(profile=args.name)
    console.print(f"当前档案：[bold]{args.name}[/bold]（数据目录 {c.data_dir}）。正在运行的服务 / 界面需要重启或在页面上切换。")
    return 0


def cmd_profile_delete(cfg: Config, args: argparse.Namespace) -> int:
    from ..service import running_pid

    if args.name == DEFAULT_PROFILE:
        raise CLIError("默认档案不能删除")
    try:
        c = load_config(profile=args.name)
    except FileNotFoundError as e:
        raise CLIError(str(e)) from e
    pid = running_pid(c)
    if pid:
        raise CLIError(f"档案 {args.name} 的服务正在运行 (pid {pid})，先停止再删除")
    if not args.yes:
        n = _profile_counts(c)
        console.print(f"将删除档案 [bold]{args.name}[/bold] 的全部数据：{c.data_dir}"
                      + (f"（{n['utterances']} 条记录 / {n['days']} 天 / {n['speakers']} 个说话人）" if n else ""))
        try:
            ans = console.input("确认请输入档案名: ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != args.name:
            console.print("已取消。")
            return 1
    try:
        delete_profile(args.name)
    except (ValueError, FileNotFoundError, OSError) as e:
        raise CLIError(str(e)) from e
    console.print(f"已删除档案 {args.name}。")
    return 0


def register(sub: Any) -> None:
    pp = sub.add_parser("profile", help="档案：给不同的人 / 用途分开的录音、记录和说话人")
    psub = pp.add_subparsers(dest="profile_cmd", metavar="<sub>")
    psub.required = True
    add_json_flag(psub.add_parser("list", help="列出档案")).set_defaults(func=handler(cmd_profile_list))
    s = psub.add_parser("create", help="新建档案（profiles/<名字>/，API key 等从根目录 config.yaml 继承）")
    s.add_argument("name")
    s.add_argument("--copy-from", default=None, metavar="NAME", help="复制某个档案的设置")
    s.add_argument("--use", action="store_true", help="创建后立即切换过去")
    s.set_defaults(func=handler(cmd_profile_create))
    s = psub.add_parser("use", help="切换当前档案（之后的命令和界面都用它）")
    s.add_argument("name")
    s.set_defaults(func=handler(cmd_profile_use))
    s = psub.add_parser("delete", help="删除档案及其全部数据")
    s.add_argument("name")
    s.add_argument("--yes", "-y", action="store_true", help="不询问")
    s.set_defaults(func=handler(cmd_profile_delete))
