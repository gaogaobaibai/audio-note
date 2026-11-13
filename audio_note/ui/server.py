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

#: seconds uvicorn waits for open responses when it stops (the event streams end as soon as the bus closes)
GRACEFUL_SHUTDOWN_S = 3


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
        bus = self._bus()
        if bus is not None:
            try:
                bus.reopen()      # a previous stop() on the same holder closed it
            except Exception:  # noqa: BLE001
                pass
        # open event streams (GET /api/events) never end by themselves: stop() closes the bus first, and the
        # graceful-shutdown timeout bounds whatever is still connected after that
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", access_log=False,
                                lifespan="off", timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S)
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

    def _bus(self) -> Any | None:
        """The event bus of the app's RuntimeHolder (None for an app without one)."""
        holder = getattr(getattr(self.app, "state", None), "holder", None)
        return getattr(holder, "events", None)

    def stop(self, timeout: float = 5.0) -> None:
        bus = self._bus()
        if bus is not None:
            try:
                bus.close()       # ends every event stream, so the server has no long-lived response left
            except Exception as e:  # noqa: BLE001
                log.debug("event bus close failed: %s", e)
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
