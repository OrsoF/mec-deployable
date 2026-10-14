"""Reference temporal JSON API, not hardened for untrusted public clients.

POST /decide receives an online snapshot (NO realized future trajectories).
Startup loads the trusted checkpoint once, if configured. See README for TLS
and authentication requirements. The client must persist `next_previous`.
"""
from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Lock
from pathlib import Path
from network_ppo.temporal_inference import TemporalDecisionEngine

MAX_BYTES=1024*1024


def handler_factory(engine):
    lock=Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,body):
            data=json.dumps(body,allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path in ("/health","/ready"):
                return self.respond(200,{"status":"ready","strategy":engine.strategy,
                                         "ppo_loaded":engine.policy is not None})
            return self.respond(404,{"error":"not found"})
        def do_POST(self):
            if self.path!="/decide":
                return self.respond(404,{"error":"not found"})
            if self.headers.get("Content-Type","").split(";")[0].strip().lower()!="application/json":
                return self.respond(415,{"error":"application/json required"})
            try:
                size=int(self.headers.get("Content-Length","0"))
            except ValueError:
                return self.respond(400,{"error":"invalid Content-Length"})
            if not 0<size<=MAX_BYTES:
                return self.respond(413,{"error":"request too large or empty"})
            try:
                payload=json.loads(self.rfile.read(size))
                with lock:
                    decision=engine.decide(payload)
                self.respond(200,decision)
            except (ValueError,TypeError,KeyError,IndexError) as e:
                self.respond(400,{"error":"invalid snapshot","detail":str(e)})
            except RuntimeError as e:
                self.respond(422,{"error":"no feasible decision","detail":str(e)})
    return Handler


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=8081)
    ap.add_argument("--strategy",choices=("keep","greedy","myopic","mpc","ppo","auto"),default="mpc")
    ap.add_argument("--horizon",type=int,default=4)
    ap.add_argument("--beam-width",type=int,default=2)
    ap.add_argument("--max-candidates",type=int,default=9)
    ap.add_argument("--max-planning-ms",type=float,default=400)
    ap.add_argument("--checkpoint",type=Path)
    args=ap.parse_args()
    engine=TemporalDecisionEngine(strategy=args.strategy,horizon=args.horizon,
                                  beam_width=args.beam_width,max_candidates=args.max_candidates,max_planning_ms=args.max_planning_ms,
                                  checkpoint=args.checkpoint)
    with ThreadingHTTPServer((args.host,args.port),handler_factory(engine)) as server:
        print(f"Temporal API on http://{args.host}:{args.port}",flush=True)
        server.serve_forever()


if __name__=='__main__':
    main()
