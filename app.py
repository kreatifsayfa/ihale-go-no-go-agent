"""Small web demo for the İhale Go/No-Go Agent."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import run
from decider_client import Decider, DeciderUnavailable
from samples.data import COMPANY_PROFILE, DEMO_TENDERS

app = FastAPI(title="İhale Go/No-Go Agent")
_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
RATE_PER_HOUR = int(os.environ.get("RATE_PER_HOUR", "20"))


class RunRequest(BaseModel):
    notice_tr: str


def _rate_limit(ip: str) -> None:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_PER_HOUR:
        raise HTTPException(429, "Demo limit reached, try again later.")
    q.append(now)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/samples")
def samples() -> dict:
    return {"tenders": DEMO_TENDERS, "company": COMPANY_PROFILE}


@app.get("/api/health")
def health() -> dict:
    try:
        return {"ok": True, "decider": Decider().health()}
    except DeciderUnavailable as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/run")
def run_agent(req: RunRequest, request: Request) -> dict:
    if len(req.notice_tr) > 6000:
        raise HTTPException(400, "Notice too long for the demo.")
    _rate_limit(request.headers.get("cf-connecting-ip") or request.client.host)
    if not _lock.acquire(timeout=90):
        raise HTTPException(503, "Busy, try again in a minute.")
    try:
        return run(req.notice_tr)
    finally:
        _lock.release()
