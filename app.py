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


@app.get("/api/llmcheck")
def llmcheck(request: Request) -> dict:
    """One Bedrock call without retries, to see the real provider error."""
    import boto3
    from botocore.config import Config

    from agent import MODEL_ID, REGION

    _rate_limit(request.headers.get("cf-connecting-ip") or request.client.host)
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(retries={"max_attempts": 1}, read_timeout=30, connect_timeout=10))
    started = time.perf_counter()
    try:
        r = client.converse(modelId=MODEL_ID, messages=[{"role": "user", "content": [{"text": "Say ok."}]}],
                            inferenceConfig={"maxTokens": 5})
        return {"ok": True, "model": MODEL_ID, "ms": round((time.perf_counter() - started) * 1000),
                "text": r["output"]["message"]["content"][0]["text"]}
    except Exception as exc:
        return {"ok": False, "model": MODEL_ID, "error": f"{type(exc).__name__}: {exc}"[:500]}


@app.post("/api/run")
def run_agent(req: RunRequest, request: Request) -> dict:
    if len(req.notice_tr) > 6000:
        raise HTTPException(400, "Notice too long for the demo.")
    _rate_limit(request.headers.get("cf-connecting-ip") or request.client.host)
    if not _lock.acquire(timeout=90):
        raise HTTPException(503, "Busy, try again in a minute.")
    try:
        return run(req.notice_tr)
    except DeciderUnavailable as exc:
        raise HTTPException(503, f"Decider unavailable: {exc}") from exc
    except Exception as exc:  # surface model/provider errors to the demo UI
        raise HTTPException(502, f"{type(exc).__name__}: {exc}"[:400]) from exc
    finally:
        _lock.release()
