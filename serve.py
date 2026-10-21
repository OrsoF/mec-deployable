"""Minimal synchronous JSON API; no model uploads, no shell execution.

This service is a reference integration adapter, NOT a hardened internet-facing
server. Run behind an authenticated reverse proxy with rate limits and TLS.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from deploy import add_engine_args
from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import parse_instance

MAX_BYTES = 1024 * 1024


def handler_factory(engine: DecisionEngine):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                return self.respond(200, {"status": "ok"})
            if self.path == "/ready":
                return self.respond(
                    200,
                    {
                        "status": "ready",
                        "strategy": engine.strategy,
                        "model_loaded": engine.runtime is not None,
                        "warnings": [engine.checkpoint_error]
                        if engine.checkpoint_error
                        else [],
                    },
                )
            self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/decide":
                return self.respond(404, {"error": "not found"})
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self.respond(415, {"error": "application/json required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.respond(400, {"error": "invalid Content-Length"})
            if size <= 0 or size > MAX_BYTES:
                return self.respond(413, {"error": "request must be 1..1048576 bytes"})
            try:
                payload = json.loads(self.rfile.read(size))
                problem = parse_instance(payload)
            except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                return self.respond(
                    400, {"error": "invalid instance", "detail": str(exc)}
                )
            try:
                with lock:
                    result = engine.decide(problem)
                return self.respond(200, result)
            except ValueError as exc:
                return self.respond(
                    422, {"error": "decision rejected", "detail": str(exc)}
                )
            except RuntimeError as exc:
                return self.respond(
                    503, {"error": "decision unavailable", "detail": str(exc)}
                )

    lock = Lock()  # Model RNG / VecNormalize are not thread-safe.
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    add_engine_args(parser)
    args = parser.parse_args()
    engine = DecisionEngine(
        strategy=args.strategy,
        checkpoint=args.checkpoint,
        paths=args.paths,
        episodes=args.episodes,
        stochastic=args.stochastic,
        seed=args.seed,
    )
    with ThreadingHTTPServer((args.host, args.port), handler_factory(engine)) as server:
        print(
            f"Listening on http://{args.host}:{args.port} (strategy={args.strategy})",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
