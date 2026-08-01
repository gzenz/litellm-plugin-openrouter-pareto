"""Tiny HTTP server that always returns HTTP 429, for the 429 integration test.

Used ONLY as a deterministic test fixture to drive a real 429 through the
litellm -> async_log_failure_event -> RateLimitCooldown -> reroute wiring that
the unit tests (which stub the exception) cannot cover. It is not a real LLM;
real OpenRouter carries every happy-path case in the e2e. Listens on
127.0.0.1:8888 by default.

The server counts POST requests to /v1/chat/completions and exposes the count
at GET /count, so the test can prove the fixture was (and was not) reached.

Run: python3 tests/e2e/four29.py [port]
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_hit_count: list[int] = [0]
_hit_lock = threading.Lock()


class _FourTwoNine(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            with _hit_lock:
                _hit_count[0] += 1
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                self.rfile.read(length)
            body = json.dumps({"error": {"message": "rate limit", "code": 429}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/count":
            with _hit_lock:
                current = _hit_count[0]
            body = json.dumps({"count": current}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"four29: {args[0] if args else ''}\n")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    server = ThreadingHTTPServer(("127.0.0.1", port), _FourTwoNine)
    print(f"four29 listening on 127.0.0.1:{port} (always 429, /count for hits)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
