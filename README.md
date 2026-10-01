# Governed Agentic AI for Fraud Investigation (Stage 3 PoC)

A working proof of concept: a Streamlit investigator dashboard triggers a LangGraph multi-agent workflow.
Four agents gather and explain evidence, a policy gate routes the case, and humans decide every case
with customer impact. All data is synthetic.

| Brief requirement | Where it is |
| --- | --- |
| n8n or equivalent orchestration | LangGraph state graph, `app/workflow.py` |
| UI/UX platform | Streamlit, `streamlit_app.py` |
| Workflow triggered from the UI | "Run investigation" button (tab 1) |
| Multiple agents | Transaction, Customer Behaviour, Risk/Policy, Recommendation (`app/agents.py`) |
| Conditional routing | `route_case` node: low -> auto-clear (A), medium -> human review (H), high -> escalation (E) |
| Tool/data integration | Data lookup, feature calculation and rules engine (`app/data_tools.py`) |
| HITL approval | Workflow pauses (LangGraph `interrupt`) until an investigator decides |
| Escalation | High risk needs a Senior Investigator; checked inside the workflow, not only in the UI |
| Exception handling | Unknown customer, missing data, invalid model JSON (1 retry, then human) |
| Test cases | 8 cases in `data/test_cases.json` (normal, ambiguous, high-risk, injection, data exception), plus any new alert typed into the "Create a new alert" form |
| Explainability | "Why did the AI give this recommendation?" table: risk factor, value, rule triggered, score contribution |
| Access control | Name + PIN sign-in; role comes from `data/users.json` |
| Open-model comparison | `benchmark.py`: same cases, prompts, evidence and JSON format for every model |
| Guardrails | `app/guardrails.py`: PII masking, injection filter, schema check, evidence check, policy floor |
| Audit trail | `logs/audit_log.jsonl`, viewable in tab 4 |

## 1. Install (one time)

1. Install **Python 3.10 or newer** from python.org. On Windows, tick "Add Python to PATH".
2. Install **Ollama** from https://ollama.com/download and open it.
3. Open a terminal (Windows: Command Prompt) in this folder and run:

```
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull mistral:7b
```

The downloads are roughly 2 GB, 2 GB and 4 GB. On a laptop with 8 GB RAM, if `mistral:7b` is too slow,
use `gemma2:2b` as the third model instead (then pass `--models llama3.2:3b qwen2.5:3b gemma2:2b` below).

## Team sign-in (role-based access control)

The role is assigned from `data/users.json`; nobody can pick their own role.

| Role | Members | Can do |
| --- | --- | --- |
| Senior Investigator | Riyanshi, Khushi | Everything, including decisions on escalated (high-risk) cases |
| Fraud Analyst | Vatsal, Aakash, Sarthak, Priyanshu | Run investigations; decide medium-risk and exception cases; escalate |
| Viewer (read only) | Any other name | See the list of test cases only |

- Team members sign in with their name and a demo PIN. **PINs are not published here**; they are shared with the faculty on request for the demo.
- A team member's name with the wrong PIN is refused (and logged).
- Only Senior Investigators can decide escalated (high-risk) cases. This is checked inside the workflow too.
- Every sign-in, failed sign-in and sign-out is written to the audit log.
- PINs are demo-only and stored as SHA-256 hashes. A real bank would use single sign-on.

## 2. Run the app

```
streamlit run streamlit_app.py
```

It opens at http://localhost:8501. Sign in, pick a model in the sidebar, choose an alert and click
**Run investigation**. Small models on a laptop take roughly 10 to 60 seconds per case.

## 3. Run the model comparison

```
python benchmark.py
```

This runs all 8 cases on all 3 models (about 96 model calls, so allow 15 to 45 minutes on a laptop).
Results appear in `results/` and in tab 3 of the app. `results/benchmark_summary.md` has tables ready
for the report. Fill in the "explanation quality" table by hand as a team.

Optional: `python benchmark.py --injection-mode flag_only` lets models see the injection text, to test
which models resist it without the filter.

## 4. Demo script for the video (about 5 minutes)

| Step | Alert / setting | What to show |
| --- | --- | --- |
| 1 | ALT-001 | Normal case auto-clears (A); nothing needs a human |
| 2 | ALT-002 | Missing device/IP data: can't auto-clear; analyst requests customer verification (H) |
| 3 | ALT-003 signed in as Vatsal (Fraud Analyst) | Escalation (E); submit a decision and show "Access denied" |
| 4 | Sign out, sign in as Riyanshi (Senior Investigator) | Confirm hold on the same case; outcome and reason shown |
| 5 | ALT-004 | Injection text removed before the model saw it; case floored to human review |
| 6 | ALT-005 | Unknown customer: exception review |
| 7 | Demo controls: "Model fails once" | Retry recovers; then "fails twice" goes to exception review |
| 8 | "Create a new alert" | Type a new transaction, add it, run it: the score and route change live |
| 9 | Sign out, sign in with any other name, e.g. "Guest" | Viewer sees only the test case list |
| 10 | Tabs 2 to 4 | Workflow graph, model comparison, audit log (shows sign-ins and decisions) |

## Offline test mode (developers only)

The app uses Ollama only. For testing the workflow without any model, `python benchmark.py --provider offline`
and `python run_cli.py ALT-003 --provider offline` use a rule-based stand-in. It is **not a model**: never report its
numbers as a model result.

## Honest limits (say these in the viva)

- Synthetic data, 8 test cases: shows the design works, not real-world accuracy.
- The rules and thresholds are illustrative, not a real bank policy.
- Role check is a demo login, not bank single sign-on.
- "Hold account" only records an outcome; nothing touches a real account.
