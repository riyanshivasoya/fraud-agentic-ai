# Screenshot checklist (Stage 3, numbered and captioned)

Take these with your real models, not offline test mode. Use the captions as written.

| Fig | What to capture | Caption |
| --- | --- | --- |
| 1 | `docs/workflow_graph.png` (or tab 2) | LangGraph workflow with conditional routing, HITL and exception paths |
| 2 | `app/workflow.py`, the `build_graph` function | Orchestration code: nodes and conditional edges |
| 3 | `app/agents.py`, the AGENTS prompts | Agent roles, prompts and JSON output schemas (same for all models) |
| 4 | Tab 1, alert queue | Synthetic alert queue with 8 test cases |
| 5 | Tab 1, "Run investigation" progress box expanded | Workflow triggered from the UI; agents run in order |
| 6 | ALT-001 result | Normal case: low risk, auto-cleared (A) |
| 7 | ALT-002 result and decision form | Ambiguous case: missing data blocks auto-clear; human review (H) |
| 8 | ALT-003, Fraud Analyst submits | Escalation access control: analyst denied |
| 9 | ALT-003, Senior Investigator confirms hold | Escalation (E): senior confirms hold, reason logged |
| 10 | ALT-004 guardrail banner | Prompt injection removed before reaching the model (DQ2) |
| 11 | ALT-005 | Data exception: unknown customer routed to exception review |
| 12 | Demo controls "fails once", agent card with "Retry used" | Exception handling: invalid JSON rejected, retry succeeds |
| 13 | Demo controls "fails twice" | Exception handling: repeated model failure routed to a human |
| 14 | Any agent's "Prompt and raw model reply" | Exact prompt and raw model output |
| 15 to 17 | Same case (e.g. ALT-003) run on each of the 3 models | Model output comparison on the same case: Llama / Qwen / Mistral |
| 18 | Terminal running `python benchmark.py` | Benchmark run: 8 cases x 3 models |
| 19 | Tab 3 summary table | Model comparison metrics |
| 20 | Tab 3 charts | Accuracy and latency by model |
| 21 | Tab 4 audit log | Audit trail of agent outputs and human decisions |
| 22 | Tab 5 masked data and guardrail table | PII masking and guardrails in the PoC |
| 23 | Dashboard counters at the top of tab 1 | Case dashboard: risk levels and cases awaiting a human |
| 24 | "Why did the AI give this recommendation?" table | Explainability: each rule's contribution to the risk score |
| 25 | "Create a new alert" form and its result | A new scenario investigated live |
| 26 | Sign-in page, and the viewer page (non-team name) | Role-based access control: viewers see only the test case list |
| 27 | Tab 4 filtered to sign-in events | Audit of sign-ins and failed sign-ins |
