# İhale Go/No-Go Agent

**Should this company bid on this public tender?** A Strands agent on Amazon Bedrock reads a Turkish
public procurement notice and tries to start a bid. Before it can, a **Go/No-Go gate** asks
**Strands Decider 2B** whether the company should bid at all. The decider answers with calibrated
probabilities in one local forward pass, and a few lines of plain Python decide what happens next.

Live demo: **https://gonogo.ihaleon.com**

Built for the AWS Agent Experience Hackathon. Tags: `#strands-decider-2b` `#policy`

---

## The problem

Turkish construction and services companies look at hundreds of public tenders a month on EKAP, the
national e-procurement platform. Most of them are a clear "no": a mandatory certificate is missing, the
required similar-work experience is too high, or the deadline is too close. Preparing a bid costs days
of work and a cash bid bond, so the go / no-go call is the expensive first step of every bid.

This is a good fit for an agent, but not for an agent where the LLM makes the call. An LLM that is asked
"should we bid?" tends to say yes, explains its way around a missing certificate, and gives no usable
signal for "I am not sure". The decision needs to be a number the code can branch on.

## How it works

```
Turkish tender notice
        │
        ▼
┌──────────────────────────┐   record_tender_brief(brief)
│ Strands Agent            │──────────────────────────────┐
│ Claude on Amazon Bedrock │                              │
└──────────────────────────┘                              ▼
        │ prepare_bid_checklist(...)          ┌──────────────────────────┐
        ▼                                     │ GoNoGoGate               │
┌──────────────────────────┐  before_tool_call│ (InterventionHandler)    │
│ intervention: GoNoGoGate │─────────────────▶│ state = brief + profile  │
└──────────────────────────┘                  └────────────┬─────────────┘
                                                           ▼
                                      ┌──────────────────────────────────┐
                                      │ Strands Decider 2B  (local, CPU) │
                                      │ verdict: choice go/no_go/review  │
                                      │ hard_blocker, experience_ok,     │
                                      │ timeline_risky: P(yes)           │
                                      └────────────┬─────────────────────┘
                                                   ▼
                                      plain-Python policy()
                       go → Proceed   no_go → Guide(log_no_bid)   review → Guide(request_human_review)
```

1. **The LLM reads and writes, it does not decide.** Claude on Bedrock reads the Turkish notice and calls
   `record_tender_brief` with an English brief of the requirements, stated as facts: work, location,
   estimated cost, required similar-work experience in TRY, mandatory certificates, bid bond, days left.
2. **The LLM proposes the commitment.** It calls `prepare_bid_checklist`, the tool that commits the bid
   team's time.
3. **The gate intercepts.** `GoNoGoGate.before_tool_call` builds a state from the brief and the company
   profile and asks Strands Decider 2B four typed questions in one forward pass.
4. **Code decides.** `policy()` turns the probabilities into an action:
   * **go** → `Proceed()`, the checklist is created.
   * **no_go** → `Guide(...)`, the model is told to call `log_no_bid` with the reasons.
   * **human_review** → `Guide(...)`, the model is told to call `request_human_review`.

The LLM never sees a "should we bid?" question, so it cannot argue its way past a missing certificate.

## The reusable decision pattern (`#policy`)

[`decision_patterns/go_no_go.py`](decision_patterns/go_no_go.py) is a self-contained **Go / No-Go gate
for committing to an opportunity**. Public tenders are one example; the same shape fits RFP responses,
inbound sales leads, grant calls, vendor onboarding, or any expensive workflow an agent is about to start.

### The questions

| key | type | asks |
|---|---|---|
| `verdict` | choice | go / no_go / human_review |
| `hard_blocker` | noul | does the organisation fail any mandatory requirement? |
| `experience_ok` | noul | is documented experience at least the required amount? |
| `timeline_risky` | noul | is the deadline too close to prepare a compliant bid? |

The three yes/no checks explain the verdict and give the policy independent signals, so a single
confident but wrong verdict does not decide alone.

### The policy

```python
GO_CONFIDENCE = 0.80       # below this, "go" goes to a human
BLOCKER_THRESHOLD = 0.60   # P(hard_blocker) at or above this forces no_go

if hard_blocker >= BLOCKER_THRESHOLD or verdict == "no_go":
    no_go   (a low-confidence no_go with no blocker goes to human_review)
elif verdict == "go" and confidence >= GO_CONFIDENCE and timeline_risky < 0.5:
    go
else:
    human_review
```

Thresholds live in code, are versioned in git, and can be tuned on real traffic without touching a
prompt.

### Reuse it in three steps

```python
from strands import Agent
from decider_client import Decider
from decision_patterns.go_no_go import GoNoGoGate, build_state, decide

# 1. as a function, anywhere
result = decide(Decider(), build_state(opportunity_brief, org_profile))
print(result.action, result.confidence, result.reasons)

# 2. as a Strands intervention that gates the tool which commits resources
agent = Agent(
    tools=[record_tender_brief, prepare_bid_checklist, log_no_bid, request_human_review],
    interventions=[GoNoGoGate(Decider(), org_profile, guarded_tools={"prepare_bid_checklist"})],
)
```

3. Change `ORG_LABEL`, `OPPORTUNITY_LABEL` and the wording of `QUESTIONS` for your domain, then tune the
   two thresholds.

## Benchmark

[`benchmark.py`](benchmark.py) runs 30 labelled tender briefs (10 go, 10 no_go, 10 human_review) through
both the decider and a Bedrock LLM asked the same question as a prompt. Labels come from explicit rules
in [`samples/data.py`](samples/data.py) so anyone can check them.

Results are written to `results/benchmark.json`. See [Results](#results).

### Results

_Filled in from `results/benchmark.json` after the run on the demo server._

## Run it

### With Docker Compose

```bash
export AWS_BEARER_TOKEN_BEDROCK=...        # a Bedrock API key
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
docker compose up --build
# open http://localhost:8000
```

The `decider` service downloads the model (about 4.5 GB) from Hugging Face on first start and runs it on
CPU. It needs roughly 8 GB of RAM.

### Without Docker

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu --port 8099

pip install -r requirements.txt
python agent.py          # runs the 4 demo tenders
python benchmark.py      # decider vs LLM
uvicorn app:app          # web demo
```

## Repository layout

| path | what |
|---|---|
| `decision_patterns/go_no_go.py` | the reusable Go/No-Go pattern: questions, policy, Strands intervention |
| `decider_client.py` | small stdlib client for the Strands Decider HTTP server |
| `agent.py` | the Strands agent on Amazon Bedrock |
| `app.py`, `static/index.html` | web demo |
| `benchmark.py` | decider vs LLM: accuracy, latency, cost |
| `samples/data.py` | synthetic company profile, demo tenders, labelled benchmark set |
| `decider/Dockerfile`, `docker-compose.yml` | deployment |

## Data

All tenders, amounts and the bidder company are **synthetic**. The notices are written in the style of
real EKAP notices. The idea comes from [İhaleOn](https://ihaleon.com), a Turkish public-tender tracking
product built by KREATIF TECHNOLOGY LLC.

## License

Apache-2.0. `decider_client.py` is adapted from the Strands Decider examples (Apache-2.0).
