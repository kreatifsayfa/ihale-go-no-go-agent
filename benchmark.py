"""Strands Decider 2B vs. a Bedrock LLM on the same Go/No-Go decision.

For each labelled brief in samples/data.py, ask:
  * the decider: one forward pass, typed questions, calibrated confidence
  * the LLM:     the same question as a prompt, answer parsed from text
and report accuracy, latency and estimated cost. Results are written to results/benchmark.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import boto3

from decider_client import Decider
from decision_patterns.go_no_go import build_state, decide
from samples.data import COMPANY_PROFILE, benchmark_cases

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
# USD per 1M tokens for the LLM baseline; set to your model's on-demand price.
PRICE_IN = float(os.environ.get("LLM_PRICE_IN_PER_M", "1.0"))
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT_PER_M", "5.0"))

LLM_PROMPT = """Decide whether the bidder company should bid on this public tender.
Answer with exactly one word: go, no_go, or human_review.
go = the company meets every mandatory requirement and can realistically deliver.
no_go = the company clearly fails at least one mandatory requirement.
human_review = it is borderline or unclear; a person should decide.

{state}"""


def ask_llm(client, state: str) -> tuple[str, float, int, int]:
    started = time.perf_counter()
    resp = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": LLM_PROMPT.format(state=state)}]}],
        inferenceConfig={"maxTokens": 10, "temperature": 0},
    )
    ms = (time.perf_counter() - started) * 1000
    text = resp["output"]["message"]["content"][0]["text"].strip().lower()
    label = next((k for k in ("human_review", "no_go", "go") if k in text.replace("-", "_").replace(" ", "_")), "unparsed")
    usage = resp.get("usage", {})
    return label, ms, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


def main() -> None:
    decider = Decider()
    client = boto3.client("bedrock-runtime", region_name=REGION)
    rows = []
    for case in benchmark_cases():
        state = build_state(case["brief"], COMPANY_PROFILE)
        d = decide(decider, state)
        llm_label, llm_ms, tin, tout = ask_llm(client, state)
        rows.append({
            "id": case["id"], "kind": case["kind"], "label": case["label"],
            "decider_action": d.action, "decider_verdict": d.verdict, "decider_confidence": d.confidence,
            "decider_ms": d.latency_ms, "llm_label": llm_label, "llm_ms": round(llm_ms, 1),
            "llm_tokens_in": tin, "llm_tokens_out": tout,
        })
        print(rows[-1])

    def acc(key: str) -> float:
        return round(sum(r[key] == r["label"] for r in rows) / len(rows), 3)

    confident = [r for r in rows if r["decider_confidence"] >= 0.9]
    llm_cost = sum(r["llm_tokens_in"] * PRICE_IN + r["llm_tokens_out"] * PRICE_OUT for r in rows) / 1e6
    summary = {
        "cases": len(rows),
        "llm_model": MODEL_ID,
        "decider_accuracy_after_policy": acc("decider_action"),
        "decider_raw_verdict_accuracy": acc("decider_verdict"),
        "decider_accuracy_when_confidence_ge_0_9": round(
            sum(r["decider_verdict"] == r["label"] for r in confident) / len(confident), 3) if confident else None,
        "decider_cases_with_confidence_ge_0_9": len(confident),
        "llm_accuracy": acc("llm_label"),
        "decider_median_ms": statistics.median(r["decider_ms"] for r in rows),
        "llm_median_ms": statistics.median(r["llm_ms"] for r in rows),
        "llm_estimated_cost_usd_for_all_cases": round(llm_cost, 5),
        "decider_cost_usd": 0.0,
    }
    print(json.dumps(summary, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
