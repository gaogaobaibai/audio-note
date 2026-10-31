"""Run the FastAPI app with uvicorn on a background thread, so it can live inside the service process.

    srv = UIServer(create_app(runtime), host, port)
    srv.start()          # raises RuntimeError when the port is taken
    ...
    srv.stop()
"""
from __future__ import annotations

import logging
import socket
import threading
import time
import webbrowser
from typing import Any

log = logging.getLogger(__name__)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, int(port)))
            return True
        except OSError:
            return False


class UIServer:
    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 8765):
        self.app = app
        self.host = host
        self.port = int(port)
        self._thread: threading.Thread | None = None
        self._server: Any = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}"

    def start(self, timeout: float = 10.0) -> str:
        import uvicorn

        if not port_free(self.host, self.port):
            raise RuntimeError(f"端口 {self.port} 已被占用（可能另一个 audio-note 界面在运行，或改 config.yaml 的 ui.port）")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", access_log=False,
                                lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run, name="ui-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                log.info("web UI at %s", self.url)
                return self.url
            if not self._thread.is_alive():
                break
            time.sleep(0.05)
        raise RuntimeError(f"界面服务没有在 {timeout:.0f}s 内启动（端口 {self.port}）")

    def _run(self) -> None:
        try:
            self._server.run()
        except Exception as e:  # noqa: BLE001
            log.exception("web UI server crashed: %s", e)

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)
        self._thread = None

    def open_browser(self) -> bool:
        try:
            return bool(webbrowser.open(self.url))
        except Exception as e:  # noqa: BLE001
            log.warning("cannot open browser: %s", e)
            return False
