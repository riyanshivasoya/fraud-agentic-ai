"""Run one alert from the terminal. Useful for testing without the UI.

    python run_cli.py ALT-003 --provider ollama --model qwen2.5:3b
    python run_cli.py ALT-002 --provider offline --decide verify_with_customer --role "Fraud Analyst"
"""
import argparse
import json

from app.workflow import Command, build_graph, new_run, pending_interrupt

p = argparse.ArgumentParser()
p.add_argument("alert_id")
p.add_argument("--provider", default="ollama", choices=["ollama", "openai", "offline"])
p.add_argument("--model", default="qwen2.5:3b")
p.add_argument("--fault", choices=["first_attempt", "all_attempts"])
p.add_argument("--decide", help="action to take at the human step, e.g. verify_with_customer, confirm_hold")
p.add_argument("--role", default="Senior Investigator")
a = p.parse_args()
if a.provider == "offline":
    a.model = "offline-test-mode"

graph = build_graph()
run_id, state, cfg = new_run(a.alert_id, {"provider": a.provider, "model": a.model}, fault=a.fault)
for step in graph.stream(state, cfg, stream_mode="updates"):
    for node in step:
        print("  ->", node)
pause = pending_interrupt(graph, cfg)
if pause:
    print("\nPAUSED for", pause["label"], "| allowed roles:", pause["allowed_roles"], "| options:", list(pause["options"]))
    if a.decide:
        for step in graph.stream(Command(resume={"role": a.role, "user": "cli-user", "action": a.decide,
                                                 "comment": "Decision entered from the command line test."}),
                                 cfg, stream_mode="updates"):
            for node in step:
                print("  ->", node)
        pause = pending_interrupt(graph, cfg)
        if pause:
            print("PAUSED again for", pause["label"])
v = graph.get_state(cfg).values
print(json.dumps({k: v.get(k) for k in ("run_id", "llm_risk_level", "final_level", "final_action", "route",
                                          "route_reasons", "outcome", "status", "data_error", "failed_agent")}, indent=1))
