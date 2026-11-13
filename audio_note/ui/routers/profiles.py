"""Profiles: independent config + data sets (one per person / purpose), switchable without a restart.

    GET    /api/profiles
    POST   /api/profiles {name, copy_from?, switch?}
    POST   /api/profiles/{name}/switch
    DELETE /api/profiles/{name} {confirm: <name>}
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ...config import (DEFAULT_PROFILE, canonical_profile_name, create_profile, delete_profile, list_profiles, load_config,
                       profile_config_path, profile_exists, valid_profile_name)
from ...runtime import profile_label
from ..deps import RT, HolderDep, bool_arg

router = APIRouter(prefix="/api")


def profile_counts(cfg: Any) -> dict[str, Any] | None:
    """Row counts of a profile's database read-only (None when it has no database yet)."""
    import sqlite3

    if not cfg.db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{cfg.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT day), MAX(day) FROM utterances").fetchone()
        spk = con.execute("SELECT COUNT(*) FROM speakers").fetchone()
        return {"utterances": int(row[0] or 0), "days": int(row[1] or 0), "last_day": row[2], "speakers": int(spk[0] or 0)}
    except sqlite3.Error:
        return None
    finally:
        con.close()


def profile_info(rt: Any, name: str) -> dict[str, Any]:
    d: dict[str, Any] = {"name": name, "label": profile_label(name), "current": rt.profile == name,
                         "config_path": str(profile_config_path(name)), "default": name == DEFAULT_PROFILE}
    try:
        cfg = load_config(profile=name)
        d["data_dir"] = str(cfg.data_dir)
        d["stats"] = profile_counts(cfg)
        from ...service import running_pid        # looked up per call: tests replace service.running_pid

        pid = running_pid(cfg)
        d["running_pid"] = pid if pid and pid != os.getpid() else None
    except Exception as e:  # noqa: BLE001
        d["error"] = str(e)
    return d


@router.get("/profiles")
def profiles_list(rt: RT) -> dict[str, Any]:
    return {"current": rt.profile, "current_label": profile_label(rt.profile),
            "custom_config": str(rt.config_path) if rt.profile is None else None,
            "profiles": [profile_info(rt, n) for n in list_profiles()]}


@router.post("/profiles")
def profiles_create(holder: HolderDep, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    if not valid_profile_name(name):
        raise HTTPException(400, "档案名只能用中英文、数字、下划线、短横线（1–32 字），且不能叫 default / current")
    switch = bool_arg(body.get("switch"), "switch")
    src = None
    copy_from = body.get("copy_from")
    if copy_from:
        if not profile_exists(str(copy_from)):
            raise HTTPException(404, f"档案不存在: {copy_from}")
        src = profile_config_path(str(copy_from))
    try:
        path = create_profile(name, copy_from=src)
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    out: dict[str, Any] = {"ok": True, "name": name, "config_path": str(path)}
    if switch:
        holder.switch(name)
        out["switched"] = True
    return out


@router.post("/profiles/{name}/switch")
def profiles_switch(name: str, holder: HolderDep) -> dict[str, Any]:
    name = canonical_profile_name(name) or name
    if not profile_exists(name):
        raise HTTPException(404, f"档案不存在: {name}")
    try:
        holder.switch(name)          # JobError (a job is running / a request is busy) -> 409 via the error handler
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except OSError as e:
        raise HTTPException(500, f"无法打开档案 {name}：{e}")
    st = holder.rt.status()          # the runtime AFTER the switch (a per-request `rt` would be the old one)
    return {"ok": True, "profile": st["profile"], "service": st["service"]}


@router.delete("/profiles/{name}")
def profiles_delete(name: str, rt: RT, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    name = canonical_profile_name(name) or name
    if name == DEFAULT_PROFILE:
        raise HTTPException(400, "默认档案不能删除")
    if not profile_exists(name):
        raise HTTPException(404, f"档案不存在: {name}")
    if rt.profile == name:
        raise HTTPException(409, "不能删除正在使用的档案，先切换到别的档案")
    if str(body.get("confirm") or "") != name:
        raise HTTPException(400, "需要在 confirm 里重复档案名以确认删除")
    info = profile_info(rt, name)
    if info.get("running_pid"):
        raise HTTPException(409, f"这个档案的服务正在另一个进程运行 (pid {info['running_pid']})，先停止它")
    try:
        delete_profile(name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except OSError as e:          # ProfileInUse: nothing was deleted
        raise HTTPException(409, str(e))
    return {"ok": True, "deleted": name}
