"""İhale Go/No-Go Agent.

A Strands agent on Amazon Bedrock that reads a Turkish public tender notice, writes an English
brief of its requirements, and tries to prepare a bid. Before the bid is prepared, the Go/No-Go
gate asks Strands Decider 2B whether the bidder company should bid at all.

    LLM (Bedrock)     reads the Turkish notice, writes the brief, drafts the bid checklist
    Strands Decider   go / no_go / human_review, with calibrated confidence
    Python policy     turns the decider's numbers into Proceed or Guide
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from decider_client import Decider
from decision_patterns.go_no_go import GoNoGoGate
from samples.data import COMPANY_PROFILE

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """You are a bid assistant for a Turkish construction company.
You receive a public tender notice in Turkish. Work in this order:
1. Call record_tender_brief with a short English brief of the tender's requirements, as facts:
   the work, location, estimated cost, required similar-work experience in TRY (compute it from
   the percentage if needed), every mandatory certificate, the bid bond, and the days left until
   the bid deadline. Do not judge whether the company qualifies.
2. Call prepare_bid_checklist to start the bid.
3. If you are told not to prepare the bid, follow the instruction you receive and call the tool it
   names (log_no_bid or request_human_review) with the reasons.
Finish with a two-sentence summary for the bid team, in English."""


class Outcome:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add(self, kind: str, **data: Any) -> None:
        self.events.append({"kind": kind, **data})


def build_agent(outcome: Outcome, decider: Decider | None = None) -> tuple[Agent, GoNoGoGate]:
    decider = decider or Decider()

    @tool
    def record_tender_brief(brief: str) -> str:
        """Record an English brief of the tender's requirements, stated as facts.

        Args:
            brief: The requirements of the tender as plain facts.
        """
        outcome.add("brief", brief=brief)
        return "Brief recorded."

    @tool
    def prepare_bid_checklist(tender_id: str, checklist: list[str]) -> str:
        """Start preparing a bid: create the checklist of documents and steps.

        Args:
            tender_id: The tender registration number (İhale kayıt no).
            checklist: The documents and steps needed to submit the bid.
        """
        outcome.add("bid_checklist", tender_id=tender_id, checklist=checklist)
        return "Bid checklist created and sent to the bid team."

    @tool
    def log_no_bid(tender_id: str, reasons: list[str]) -> str:
        """Record that the company will not bid on this tender, with the reasons.

        Args:
            tender_id: The tender registration number.
            reasons: Why the company is not bidding.
        """
        outcome.add("no_bid", tender_id=tender_id, reasons=reasons)
        return "No-bid decision logged."

    @tool
    def request_human_review(tender_id: str, reasons: list[str]) -> str:
        """Hand the go/no-go decision to a person on the bid team.

        Args:
            tender_id: The tender registration number.
            reasons: What the person should look at.
        """
        outcome.add("human_review", tender_id=tender_id, reasons=reasons)
        return "Sent to the bid team for review."

    gate = GoNoGoGate(decider, COMPANY_PROFILE, guarded_tools={"prepare_bid_checklist"})
    model = BedrockModel(model_id=MODEL_ID, region_name=REGION)
    agent = Agent(
        model=model,
        tools=[record_tender_brief, prepare_bid_checklist, log_no_bid, request_human_review],
        interventions=[gate],
        system_prompt=SYSTEM_PROMPT + "\n\nCompany profile:\n" + json.dumps(COMPANY_PROFILE, ensure_ascii=False),
        callback_handler=None,
    )
    return agent, gate


def run(notice_tr: str) -> dict[str, Any]:
    outcome = Outcome()
    agent, gate = build_agent(outcome)
    started = time.perf_counter()
    result = agent(f"Tender notice:\n{notice_tr}")
    return {
        "summary": str(result).strip(),
        "events": outcome.events,
        "decisions": [d.to_dict() for d in gate.decisions],
        "total_seconds": round(time.perf_counter() - started, 2),
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    from samples.data import DEMO_TENDERS

    for t in DEMO_TENDERS:
        print("=" * 80, f"\n{t['id']}  {t['title']}")
        print(json.dumps(run(t["notice_tr"]), indent=2, ensure_ascii=False))
