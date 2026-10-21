"""Reusable decision pattern: Go / No-Go gate for committing to an opportunity.

PATTERN TAG: #policy

What it decides
---------------
Before an agent commits resources to an opportunity (bidding on a public tender, answering an
RFP, accepting a lead, starting an expensive workflow), it must decide one of three things:

    go            -> the requirements fit the organisation; proceed with the commitment tool
    no_go         -> at least one hard requirement is not met; do not commit, log why
    human_review  -> borderline or uncertain; a person decides

The decision is made by Strands Decider 2B, not by the LLM that drives the agent. The LLM only
writes a short brief of the opportunity (requirements as facts). The decider reads the brief and
the organisation profile and answers typed questions with calibrated probabilities. A few lines of
plain Python (`policy`) turn those numbers into an action.

Why this split
--------------
* The verdict is a number the code branches on, so the LLM cannot talk the gate into a "yes".
* The decider is a local ~2B model: no extra LLM call, no data leaves your server.
* The decider's confidence is calibrated, so "not sure" routes to a human instead of a guess.

How to reuse
------------
1. Change `ORG_LABEL` / `OPPORTUNITY_LABEL` and the wording of `QUESTIONS` to your domain.
2. Build the state with `build_state(opportunity_brief, org_profile)`.
3. Call `decide(decider, state)` anywhere, or add `GoNoGoGate(decider, guarded_tools={...})`
   to a Strands agent's `interventions=[...]` to gate the tool that commits resources.
4. Tune `GO_CONFIDENCE` and `BLOCKER_THRESHOLD` on your own traffic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from strands.interventions import Guide, InterventionHandler, Proceed

from decider_client import Decider

ORG_LABEL = "BIDDER COMPANY PROFILE"
OPPORTUNITY_LABEL = "PUBLIC TENDER BRIEF"

# --------------------------------------------------------------------------- the questions
#
# One choice question for the verdict itself, plus three narrow yes/no checks that explain it.
# All four are answered in a single forward pass over the same state.

QUESTIONS: dict[str, dict[str, Any]] = {
    "verdict": Decider.choice(
        "Should the bidder company submit a bid for this tender?",
        {
            "go": "the company meets every mandatory requirement and can realistically deliver",
            "no_go": "the company clearly fails at least one mandatory requirement",
            "human_review": "it is borderline or unclear; a person should decide",
        },
    ),
    "hard_blocker": Decider.noul(
        "Does the company fail any mandatory qualification requirement of the tender?",
        {
            "true": "a mandatory requirement (experience, certificate, license, capacity) is not met",
            "false": "every mandatory requirement is met by the company profile",
        },
    ),
    "experience_ok": Decider.noul(
        "Is the company's documented similar-work experience at least the amount the tender requires?",
        {
            "true": "documented similar-work experience is equal to or above the required amount",
            "false": "documented similar-work experience is below the required amount",
        },
    ),
    "timeline_risky": Decider.noul(
        "Is the time left before the bid deadline too short for the company to prepare a compliant bid?",
        {
            "true": "the deadline is closer than the company's minimum preparation time",
            "false": "there is enough time to prepare the bid",
        },
    ),
}

# Policy knobs, owned by the code, not the model.
GO_CONFIDENCE = 0.80        # below this, a "go" verdict is sent to a human
BLOCKER_THRESHOLD = 0.60    # P(hard_blocker) at or above this forces no_go


@dataclass
class GoNoGoResult:
    action: str                         # "go" | "no_go" | "human_review"
    verdict: str                        # the decider's top choice
    confidence: float                   # the decider's confidence in that choice
    probabilities: dict[str, float]     # per-option scores for the verdict question
    checks: dict[str, float]            # P(yes) for each yes/no check
    reasons: list[str] = field(default_factory=list)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_state(opportunity_brief: str, org_profile: dict[str, Any] | str) -> str:
    profile = org_profile if isinstance(org_profile, str) else json.dumps(org_profile, indent=2, ensure_ascii=False)
    return f"--- {OPPORTUNITY_LABEL} ---\n{opportunity_brief.strip()}\n\n--- {ORG_LABEL} ---\n{profile}"


def _choice_scores(answer: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    """Normalise the server's choice answer to (top option, confidence, per-option scores)."""
    scores = answer.get("scores") or answer.get("probabilities") or answer.get("options") or {}
    if isinstance(scores, list):
        scores = {item.get("option") or item.get("name"): item.get("score") or item.get("p") for item in scores}
    top = answer.get("choice") or answer.get("answer") or (max(scores, key=scores.get) if scores else "human_review")
    conf = answer.get("confidence")
    if conf is None:
        conf = scores.get(top, 0.0) if scores else 0.0
    return top, float(conf), {k: round(float(v), 3) for k, v in scores.items()}


def policy(answers: dict[str, Any]) -> GoNoGoResult:
    """Plain-Python policy over the decider's typed answers."""
    verdict, conf, probs = _choice_scores(answers["verdict"])
    checks = {k: round(float(answers[k]["noul"]), 3) for k in ("hard_blocker", "experience_ok", "timeline_risky")}
    reasons: list[str] = []

    if checks["hard_blocker"] >= BLOCKER_THRESHOLD or verdict == "no_go":
        if checks["experience_ok"] < 0.5:
            reasons.append("similar-work experience is below the tender's requirement")
        if checks["hard_blocker"] >= BLOCKER_THRESHOLD:
            reasons.append("at least one mandatory qualification is not met")
        if verdict == "no_go" and conf < GO_CONFIDENCE and checks["hard_blocker"] < BLOCKER_THRESHOLD:
            action = "human_review"
            reasons.append(f"no_go verdict with low confidence ({conf:.2f})")
        else:
            action = "no_go"
    elif verdict == "go" and conf >= GO_CONFIDENCE and checks["timeline_risky"] < 0.5:
        action = "go"
        reasons.append(f"all checks passed with confidence {conf:.2f}")
    else:
        action = "human_review"
        if conf < GO_CONFIDENCE:
            reasons.append(f"decider confidence {conf:.2f} is below {GO_CONFIDENCE}")
        if checks["timeline_risky"] >= 0.5:
            reasons.append("the bid deadline looks too close")
        if verdict == "human_review":
            reasons.append("the decider judged the case borderline")

    return GoNoGoResult(action, verdict, round(conf, 3), probs, checks, reasons)


def decide(decider: Decider, state: str) -> GoNoGoResult:
    answers = decider.ask(state, QUESTIONS)
    result = policy(answers)
    result.latency_ms = round(decider.last_latency_ms or 0.0, 1)
    return result


class GoNoGoGate(InterventionHandler):
    """Strands intervention: gate the tool(s) that commit resources behind the Go/No-Go decision.

    The agent's LLM must first call a tool that records the opportunity brief (`brief_tool`);
    when it then proposes a guarded tool, the gate asks the decider and returns:
        go           -> Proceed()
        no_go        -> Guide(...)  (the model is told to log a no-bid with the reasons instead)
        human_review -> Guide(...)  (the model is told to hand the case to a person)
    """

    name = "strands-decider-go-no-go-gate"

    def __init__(self, decider: Decider, org_profile: dict[str, Any], guarded_tools: set[str]) -> None:
        self._decider = decider
        self._profile = org_profile
        self._guarded = guarded_tools
        self.brief: str | None = None
        self.decisions: list[GoNoGoResult] = []

    def before_tool_call(self, event, **kwargs):
        name = event.tool_use["name"]
        args = event.tool_use.get("input") or {}
        if name == "record_tender_brief":
            self.brief = args.get("brief", "")
            return Proceed()
        if name not in self._guarded:
            return Proceed()
        if not self.brief:
            return Guide(feedback="First call record_tender_brief with the tender's requirements as facts.")

        result = decide(self._decider, build_state(self.brief, self._profile))
        self.decisions.append(result)
        why = "; ".join(result.reasons)
        if result.action == "go":
            return Proceed(reason=why)
        if result.action == "no_go":
            return Guide(feedback=f"Go/No-Go gate: NO-GO ({why}). Do not prepare a bid. "
                                  "Call log_no_bid with these reasons instead.")
        return Guide(feedback=f"Go/No-Go gate: HUMAN REVIEW needed ({why}). Do not prepare a bid. "
                              "Call request_human_review with these reasons instead.")
