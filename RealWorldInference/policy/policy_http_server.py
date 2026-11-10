"""Policy HTTP Server — standalone request/response server for sim evaluation.

The main inference link (SEP/policy/slave) uses the reverse-WS SEP protocol
in ``policy/policy_server.py``. Simulation evaluation (RoboTwin, RoboCasa, LIBERO,
RMBench, RoboMemArena …) instead runs many single-GPU client-server pairs on one
box, where a plain HTTP request/response loop is simpler and more robust than a
WS channel through SEP/sep. This module restores that HTTP server, decoupled
from the production WS path.

Endpoints (matched by ``policy/policies/http_client/policy.py``):
  POST /infer  — Observation bytes → Action bytes (session-locked)
  POST /reset  — reset policy state, release session lock
  GET  /health — readiness probe (run_eval.sh waits on this)
  GET  /config — wire-config handshake (executor obs encoding)

Single-threaded on purpose: torch.compile / CUDA Graph state lives in TLS, so a
second concurrent request on another thread trips an AssertionError. Sim workers
issue one in-flight request per server, so serial handling is sufficient.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m policy.policy_http_server --config policy/config_falcon.yaml
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy.policies import BasePolicy, load_policy
from utils.logging_utils import setup_logger
from utils.message import Action, Observation

logger = setup_logger("PolicyHTTPServer")

# Single-process single-policy: module-level state is enough, mirrors policy_server.py.
_policy: Optional[BasePolicy] = None
_policy_id: str = "policy_default"

# --- Session lock ---
# Binds the policy to the first session_id seen and rejects others with 409 until
# /reset (or TTL expiry). Keeps a stray second worker from corrupting AR state.
_current_session_id: Optional[str] = None
_session_lock = threading.Lock()
_last_active_time: float = 0.0
SESSION_TTL = 120  # seconds


def _do_infer(data: bytes, session_id: str) -> tuple[bytes, float]:
    """Run one inference. Returns (action_bytes, inference_time).

    Raises PermissionError(409 detail) if the policy is bound to another session.
    """
    global _current_session_id, _last_active_time
    assert _policy is not None

    with _session_lock:
        now = time.time()
        if _current_session_id and now - _last_active_time > SESSION_TTL:
            logger.warning("Session %s TTL expired, auto-releasing", _current_session_id)
            _current_session_id = None
            _policy.reset()

        if _current_session_id is None:
            _current_session_id = session_id
            logger.info("Session bound: %s", session_id)
        elif _current_session_id != session_id:
            raise PermissionError(f"Policy occupied by session {_current_session_id}")

        _last_active_time = now

    _policy.set_session_id(session_id)
    observation = Observation.from_bytes(data)
    inference_start = time.time()
    _policy.update_obs(observation)
    action = _policy.get_action()
    inference_time = time.time() - inference_start
    return action.to_bytes(), inference_time


def _do_reset() -> None:
    """Reset policy state and unconditionally release the session lock.

    Any caller can reset and kick out the current session. Fine for single-worker
    eval; add session_id validation here if multi-session support is needed.
    """
    global _current_session_id
    assert _policy is not None
    with _session_lock:
        old = _current_session_id
        _current_session_id = None
    _policy.reset()
    logger.info("Policy reset (was session=%s)", old)


class PolicyHandler(BaseHTTPRequestHandler):
    """HTTP handler mirroring the pre-refactor Flask endpoints."""

    # Silence the default per-request stderr logging; we log meaningful events ourselves.
    def log_message(self, fmt, *args):  # noqa: A003 - signature fixed by base class
        pass

    def _send_bytes(self, status: int, body: bytes, content_type: str,
                    extra_headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, obj) -> None:
        import json
        self._send_bytes(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):  # noqa: N802 - name fixed by base class
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "policy_id": _policy_id,
                "current_session": _current_session_id,
            })
        elif self.path == "/config":
            assert _policy is not None
            self._send_json(200, _policy.get_wire_config().to_dict())
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):  # noqa: N802 - name fixed by base class
        if self.path == "/infer":
            self._handle_infer()
        elif self.path == "/reset":
            _do_reset()
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def _handle_infer(self) -> None:
        session_id = self.headers.get("X-Session-Id", "anonymous")
        server_start = time.time()
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length) if length > 0 else b""

        try:
            action_bytes, inference_time = _do_infer(data, session_id)
        except PermissionError as exc:
            self._send_bytes(409, str(exc).encode("utf-8"), "text/plain")
            return
        except Exception as exc:  # noqa: BLE001 - report any inference failure as 500
            logger.error("Inference failed: %s", exc, exc_info=True)
            self._send_bytes(500, str(exc).encode("utf-8"), "text/plain")
            return

        server_total = time.time() - server_start
        logger.info("Inference: %.3fs | total: %.3fs | session: %s",
                    inference_time, server_total, session_id)
        self._send_bytes(
            200, action_bytes, "application/octet-stream",
            extra_headers={
                "X-Inference-Time": f"{inference_time:.6f}",
                "X-Server-Total-Time": f"{server_total:.6f}",
            },
        )


def main():
    global _policy, _policy_id

    parser = argparse.ArgumentParser(description="Policy HTTP Server (sim evaluation)")
    parser.add_argument("--config", type=str, default="policy/config.yaml")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    policy_config = config.get("policy", {}) or {}
    policy_type = policy_config.get("type", "unknown")
    _policy_id = policy_config.get("policy_id", f"policy_{policy_type}")

    server_config = config.get("server", {}) or {}
    host = server_config.get("host", "0.0.0.0")
    port = int(server_config.get("port", 24851))

    logger.info("Loading policy type=%s policy_id=%s ...", policy_type, _policy_id)
    _policy = load_policy(policy_type, policy_config)
    logger.info("Policy loaded")

    # Single-threaded HTTPServer: requests serialize, matching the old
    # flask app.run(threaded=False) so CUDA Graph TLS state stays valid.
    httpd = HTTPServer((host, port), PolicyHandler)
    logger.info("Starting on %s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting.")
    finally:
        httpd.server_close()
        if _policy is not None:
            _policy.close()


if __name__ == "__main__":
    main()
