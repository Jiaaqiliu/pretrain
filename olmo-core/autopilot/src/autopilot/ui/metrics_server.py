"""Lightweight metrics server for receiving training metrics from running jobs.

Running training jobs send metrics to this server via HTTP POST.
The server feeds them into the monitoring pipeline.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional

from autopilot.monitoring.metrics import MetricsCollector, MetricsSnapshot
from autopilot.utils.logging import get_logger

log = get_logger("ui.metrics_server")


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for incoming training metrics."""

    collector: Optional[MetricsCollector] = None
    on_error: Optional[Callable] = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        if self.path == "/metrics":
            self._handle_metrics(data)
        elif self.path == "/error":
            self._handle_error(data)
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"healthy")
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_metrics(self, data: dict) -> None:
        experiment_id = data.get("experiment_id", "unknown")
        step = data.get("step", 0)
        timestamp = data.get("timestamp", 0.0)
        metrics = data.get("metrics", {})

        snapshot = MetricsSnapshot(timestamp=timestamp, step=step, metrics=metrics)

        if MetricsHandler.collector:
            MetricsHandler.collector.record(experiment_id, snapshot)

    def _handle_error(self, data: dict) -> None:
        experiment_id = data.get("experiment_id", "unknown")
        error = data.get("error", "unknown error")
        log.error(f"Training error from {experiment_id}: {error}")

        if MetricsHandler.on_error:
            MetricsHandler.on_error(experiment_id, error)

    def log_message(self, format, *args):
        pass  # suppress default logging


class MetricsServer:
    """Background HTTP server for collecting metrics from training jobs."""

    def __init__(
        self,
        collector: MetricsCollector,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_error: Optional[Callable] = None,
    ):
        self._collector = collector
        self._host = host
        self._port = port
        self._on_error = on_error
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        MetricsHandler.collector = self._collector
        MetricsHandler.on_error = self._on_error

        self._server = HTTPServer((self._host, self._port), MetricsHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info(f"Metrics server started at {self.url}")

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            log.info("Metrics server stopped")
