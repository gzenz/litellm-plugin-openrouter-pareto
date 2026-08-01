"""Tiny HTTP server that returns a minimal OpenAI-format 200, for the 429
integration test's healthy alternative. Used ONLY as a deterministic "healthy
provider" fixture so the cooldown reroute lands somewhere that reliably succeeds,
without depending on real OpenRouter's rate-limit state. It is not a real LLM.

Counts POST /v1/chat/completions requests at GET /count, so the test can prove
the healthy deployment was (and was not) reached.

Run: python3 tests/e2e/healthy_fixture.py [port]
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_hit_count: list[int] = [0]
_hit_lock = threading.Lock()


class _Healthy(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            with _hit_lock:
                _hit_count[0] += 1
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                self.rfile.read(length)
            body = json.dumps({
                "id": "cmpl-healthy-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "z-ai/glm-5.2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "healthy"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "provider": "BaseTen",
            }).encode()
            self.send_response(200)
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
        sys.stderr.write(f"healthy: {args[0] if args else ''}\n")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8889
    server = ThreadingHTTPServer(("127.0.0.1", port), _Healthy)
    print(f"healthy_fixture listening on 127.0.0.1:{port} (always 200, /count for hits)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
