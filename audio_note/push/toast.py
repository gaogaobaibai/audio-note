"""Windows toast notification with zero dependencies (PowerShell + Windows.UI.Notifications).

The toast XML (ToastText02 template) is built in Python with XML escaping and handed to
PowerShell base64-encoded; the PowerShell script itself is passed via ``-EncodedCommand``
(UTF-16LE base64), so arbitrary title/body text can never be interpreted as code.
"""
from __future__ import annotations

import base64
import logging
import subprocess
import sys
from typing import Any
from xml.sax.saxutils import escape

from .base import ChannelNotConfigured, Message, markdown_to_text

logger = logging.getLogger(__name__)

# CreateToastNotifier(appId).Show() silently drops the toast on Windows 10/11 unless the AUMID is
# registered (Start-menu shortcut or HKCU\Software\Classes\AppUserModelId\<id>). PowerShell's own
# AUMID is always registered, so we show toasts under it (the same trick BurntToast uses).
APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
APP_NAME = "audio-note"
BODY_MAX_CHARS = 200
TITLE_MAX_CHARS = 64
TIMEOUT_S = 20

_PS_TEMPLATE = """\
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$xmlText = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{xml_b64}'))
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($xmlText)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app_id}').Show($toast)
"""


def is_windows() -> bool:
    return sys.platform.startswith("win")


def plain_body(msg: Message) -> str:
    """Markdown stripped to text and truncated to BODY_MAX_CHARS."""
    text = markdown_to_text(msg.body_md)
    if msg.url and msg.url not in text:
        text = f"{text}\n{msg.url}" if text else msg.url
    text = " ".join(text.split())  # toasts have no room for layout
    if len(text) > BODY_MAX_CHARS:
        text = text[: BODY_MAX_CHARS - 1] + "…"
    return text


def build_xml(title: str, body: str) -> str:
    """ToastText02 XML with escaped title/body (pure)."""
    return (
        '<toast><visual><binding template="ToastText02">'
        f'<text id="1">{escape(title, {chr(34): "&quot;"})}</text>'
        f'<text id="2">{escape(body, {chr(34): "&quot;"})}</text>'
        "</binding></visual></toast>"
    )


def build_payload(msg: Message) -> dict[str, str]:
    """Pure: {"title", "body", "xml", "script"} for ``msg``."""
    title = " ".join((msg.title or APP_NAME).split())[:TITLE_MAX_CHARS]
    body = plain_body(msg)
    xml = build_xml(title, body)
    return {"title": title, "body": body, "xml": xml, "script": build_script(xml)}


def build_script(xml: str, app_id: str = APP_ID) -> str:
    xml_b64 = base64.b64encode(xml.encode("utf-8")).decode("ascii")
    return _PS_TEMPLATE.format(xml_b64=xml_b64, app_id=app_id.replace("'", "''"))


def encode_command(script: str) -> str:
    """PowerShell ``-EncodedCommand`` value: base64 of UTF-16LE script."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def build_command(script: str) -> list[str]:
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-NoLogo",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encode_command(script),
    ]


def show_toast(title: str, body: str) -> None:
    """Run the PowerShell toast; raises RuntimeError on failure."""
    xml = build_xml(title, body)
    cmd = build_command(build_script(xml))
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=TIMEOUT_S,
            creationflags=creationflags,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"toast: 无法运行 PowerShell: {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"toast: PowerShell 退出码 {proc.returncode}: {err[:300]}")


class ToastChannel:
    name = "toast"

    def __init__(self, cfg: Any) -> None:
        if not is_windows():
            raise ChannelNotConfigured("toast 仅支持 Windows")
        self.app_id = APP_ID

    def send(self, msg: Message) -> None:
        p = build_payload(msg)
        try:
            show_toast(p["title"], p["body"])
        except RuntimeError as e:
            logger.error("Windows toast 失败: %s", e)
            raise
