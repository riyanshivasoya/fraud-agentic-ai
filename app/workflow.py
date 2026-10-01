"""LangGraph orchestration: the governed fraud investigation workflow.

Alert -> Orchestrator intake -> Transaction Agent -> Customer Behaviour Agent -> Risk/Policy Agent
      -> Recommendation Agent -> Policy gate (routing) -> Auto-clear (A) | Human review (H) | Escalation (E)
Any data or model failure -> Exception review (H).
"""
import hashlib
import uuid
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import config
from .agents import ALLOWED_FLAGS, run_agent
from .audit import log_event
from .data_tools import compute_features, get_alert, get_customer, load_rules, max_level, run_rules_engine
from .guardrails import ACTION_FOR_LEVEL, governed_action, mask_customer, sanitise_transaction

AGENT_ORDER = ["transaction", "behaviour", "risk", "recommendation"]
AGENT_ACTOR = {
    "transaction": "Transaction Analysis Agent",
    "behaviour": "Customer Behaviour Agent",
    "risk": "Risk / Policy Agent",
    "recommendation": "Recommendation Agent",
}


class CaseState(TypedDict, total=False):
    run_id: str
    alert_id: str
    model: dict
    fault: Optional[str]
    fault_agent: Optional[str]
    injection_mode: str
    audit_path: Optional[str]
    # evidence
    txn: dict
    safe_txn: dict
    masked_customer: dict
    features: dict
    missing_fields: list
    injection_findings: list
    rules_result: dict
    data_error: Optional[str]
    # agent work
    agent_results: dict
    failed_agent: Optional[str]
    # decisions
    llm_risk_level: Optional[str]
    llm_action: Optional[str]
    final_level: Optional[str]
    final_action: Optional[str]
    route: Optional[str]
    route_reasons: list
    human_decisions: list
    next_step: Optional[str]
    outcome: Optional[str]
    status: str


def _log(state, actor, event, details=None):
    path = state.get("audit_path")
    log_event(state["run_id"], state["alert_id"], actor, event, details, path=Path(path) if path else None)


# ------------------------------------------------------------------ orchestrator intake (A)

def intake(state: CaseState):
    txn = get_alert(state["alert_id"])
    if txn is None:
        err = f"Alert {state['alert_id']} not found"
        _log(state, "Orchestrator", "data_exception", {"error": err})
        return {"data_error": err, "status": "exception"}
    customer = get_customer(txn["customer_id"])
    if customer is None:
        err = f"Customer {txn['customer_id']} not found in customer master; agents cannot compare behaviour"
        _log(state, "Orchestrator", "data_exception", {"error": err})
        return {"txn": txn, "data_error": err, "status": "exception", "agent_results": {}}

    features, missing = compute_features(txn, customer)
    safe_txn, findings = sanitise_transaction(txn, state.get("injection_mode", config.INJECTION_MODE))
    safe_txn = {k: v for k, v in safe_txn.items() if k != "customer_id"}
    rules_result = run_rules_engine(features, missing, bool(findings), load_rules())
    _log(state, "Orchestrator", "case_opened", {
        "model": state["model"], "missing_fields": missing, "injection_findings": findings,
        "rule_score": rules_result["rule_score"], "policy_level": rules_result["policy_level"],
        "pii_masked": True,
    })
    return {
        "txn": txn, "safe_txn": safe_txn, "masked_customer": mask_customer(customer),
        "features": features, "missing_fields": missing, "injection_findings": findings,
        "rules_result": rules_result, "agent_results": {}, "status": "in_progress",
        "route_reasons": [], "human_decisions": [],
    }


# ------------------------------------------------------------------ the four agents (A)

def _evidence(agent, s):
    r = s["agent_results"]
    flagged = [f["field"] for f in s.get("injection_findings", [])]
    rules = s["rules_result"]
    if agent == "transaction":
        return {"transaction": s["safe_txn"], "facts": s["features"], "missing_fields": s["missing_fields"],
                "text_guardrail_flags": flagged, "allowed_red_flags": ALLOWED_FLAGS}
    if agent == "behaviour":
        return {"customer_profile": s["masked_customer"], "facts": s["features"],
                "transaction_agent_findings": r["transaction"]["output"]}
    if agent == "risk":
        return {"transaction_findings": r["transaction"]["output"], "behaviour_findings": r["behaviour"]["output"],
                "rules_engine": {k: rules[k] for k in ("rule_score", "triggered_rules", "floors_applied", "thresholds")},
                "allowed_rule_ids": [h["id"] for h in rules["triggered_rules"]] + [f["id"] for f in rules["floors_applied"]],
                "missing_fields": s["missing_fields"], "text_guardrail_flags": flagged}
    return {"risk_assessment": r["risk"]["output"], "transaction_findings": r["transaction"]["output"],
            "behaviour_findings": r["behaviour"]["output"],
            "allowed_actions": ["clear", "verify_with_customer", "hold_and_escalate"]}


def make_agent_node(agent):
    def node(state: CaseState):
        fault = state.get("fault") if state.get("fault_agent") in (None, agent) else None
        rule_ids = [h["id"] for h in state["rules_result"]["triggered_rules"]] + \
                   [f["id"] for f in state["rules_result"]["floors_applied"]]
        result = run_agent(agent, _evidence(agent, state), state["model"], state["features"], rule_ids, fault=fault)
        _log(state, AGENT_ACTOR[agent], "agent_completed" if result["ok"] else "agent_failed", {
            "model": state["model"]["model"], "output": result["output"],
            "attempts": len(result["attempts"]), "json_valid_first_try": result["json_valid_first_try"],
            "evidence_problems": result["evidence_problems"], "latency_s": result["latency_s"],
            "error": result.get("error"),
        })
        results = dict(state["agent_results"])
        results[agent] = result
        update = {"agent_results": results}
        if not result["ok"]:
            update["failed_agent"] = agent
            update["status"] = "exception"
        return update
    node.__name__ = f"{agent}_agent"
    return node


# ------------------------------------------------------------------ policy gate (A)

def route_case(state: CaseState):
    llm_level = state["agent_results"]["risk"]["output"]["risk_level"]
    llm_action = state["agent_results"]["recommendation"]["output"]["recommended_action"]
    policy_level = state["rules_result"]["policy_level"]
    final_level = max_level(llm_level, policy_level)
    final_action = governed_action(llm_action, final_level)
    reasons = []
    if final_level != llm_level:
        floors = ", ".join(f["id"] for f in state["rules_result"]["floors_applied"]) or "none"
        reasons.append(f"Policy raised risk from {llm_level} (model) to {final_level} "
                       f"(rule score {state['rules_result']['rule_score']}, floors: {floors}).")
    if final_action != llm_action:
        reasons.append(f"Action changed from '{llm_action}' to '{final_action}' to match {final_level} risk policy.")
    route = {"clear": "auto_clear", "verify_with_customer": "human_review", "hold_and_escalate": "escalation"}[final_action]
    _log(state, "Policy gate", "case_routed", {
        "llm_risk_level": llm_level, "policy_level": policy_level, "final_level": final_level,
        "llm_action": llm_action, "final_action": final_action, "route": route, "reasons": reasons,
    })
    return {"llm_risk_level": llm_level, "llm_action": llm_action, "final_level": final_level,
            "final_action": final_action, "route": route, "route_reasons": reasons,
            "status": "awaiting_human" if route != "auto_clear" else "in_progress"}


def auto_clear(state: CaseState):
    n = load_rules().get("qa_sample_every_n_auto_clears", 3)
    qa = int(hashlib.sha256(state["run_id"].encode()).hexdigest(), 16) % n == 0
    _log(state, "Policy gate", "auto_cleared", {"qa_sample": qa, "note": "No customer impact; logged for quality review"})
    return {"outcome": "CLEARED_AUTOMATICALLY" + (" (selected for QA review)" if qa else ""), "next_step": "finalize"}


# ------------------------------------------------------------------ human steps (H / E)

HUMAN_OPTIONS = {
    "human_review": {
        "label": "Human review (H)",
        "roles": ["Fraud Analyst", "Senior Investigator"],
        "options": {
            "clear": "Clear the alert (genuine transaction)",
            "verify_with_customer": "Contact customer to verify",
            "hold_and_escalate": "Escalate to Senior Investigator",
        },
    },
    "exception_review": {
        "label": "Exception review (H)",
        "roles": ["Fraud Analyst", "Senior Investigator"],
        "options": {
            "clear": "Clear after manual checks",
            "verify_with_customer": "Contact customer to verify",
            "hold_and_escalate": "Escalate to Senior Investigator",
        },
    },
    "escalation": {
        "label": "Escalation (E)",
        "roles": ["Senior Investigator"],
        "options": {
            "confirm_hold": "Hold account and open fraud case",
            "verify_with_customer": "Downgrade: contact customer to verify",
            "clear": "Clear the alert (genuine transaction)",
        },
    },
}

OUTCOMES = {
    "clear": "CLEARED_BY_HUMAN",
    "verify_with_customer": "CUSTOMER_VERIFICATION_REQUESTED",
    "confirm_hold": "ACCOUNT_HOLD_AND_FRAUD_CASE_OPENED",
}


def _human_step(state: CaseState, stage: str):
    spec = HUMAN_OPTIONS[stage]
    payload = {
        "stage": stage, "label": spec["label"], "allowed_roles": spec["roles"], "options": spec["options"],
        "final_level": state.get("final_level"), "recommended": state.get("final_action"),
        "data_error": state.get("data_error"), "failed_agent": state.get("failed_agent"), "error": None,
    }
    rejected = []
    while True:
        decision = interrupt(payload)
        problem = None
        if decision.get("role") not in spec["roles"]:
            problem = f"Access denied: {spec['label']} needs one of {spec['roles']}; you are '{decision.get('role')}'."
        elif decision.get("action") not in spec["options"]:
            problem = f"'{decision.get('action')}' is not an allowed action here."
        elif len(str(decision.get("comment", "")).strip()) < 10:
            problem = "A comment of at least 10 characters is required for the audit trail."
        if problem is None:
            break
        rejected.append({"decision": decision, "reason": problem})
        payload = {**payload, "error": problem}

    for r in rejected:  # logged once, after the accepted decision
        _log(state, r["decision"].get("user") or r["decision"].get("role"), "decision_rejected", r)
    record = {"stage": stage, **decision}
    _log(state, f"{decision.get('user', 'unknown')} ({decision['role']})", "human_decision", record)
    decisions = list(state.get("human_decisions", [])) + [record]
    if decision["action"] == "hold_and_escalate":
        return {"human_decisions": decisions, "next_step": "escalation", "status": "awaiting_human"}
    return {"human_decisions": decisions, "outcome": OUTCOMES[decision["action"]], "next_step": "finalize"}


def human_review(state: CaseState):
    return _human_step(state, "human_review")


def exception_review(state: CaseState):
    return _human_step(state, "exception_review")


def escalation(state: CaseState):
    return _human_step(state, "escalation")


def finalize(state: CaseState):
    _log(state, "Orchestrator", "case_closed", {"outcome": state.get("outcome"), "route": state.get("route")})
    return {"status": "closed"}


# ------------------------------------------------------------------ graph

def _after(agent):
    nxt = {"transaction": "behaviour_agent", "behaviour": "risk_agent",
           "risk": "recommendation_agent", "recommendation": "route_case"}[agent]

    def decide(state: CaseState):
        return "exception_review" if state.get("failed_agent") == agent else nxt
    return decide, {nxt: nxt, "exception_review": "exception_review"}


def build_graph(checkpointer=None):
    g = StateGraph(CaseState)
    g.add_node("intake", intake)
    for a in AGENT_ORDER:
        g.add_node(f"{a}_agent", make_agent_node(a))
    for name, fn in [("route_case", route_case), ("auto_clear", auto_clear), ("human_review", human_review),
                     ("escalation", escalation), ("exception_review", exception_review), ("finalize", finalize)]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: "exception_review" if s.get("data_error") else "transaction_agent",
                            {"transaction_agent": "transaction_agent", "exception_review": "exception_review"})
    for a in AGENT_ORDER:
        decide, targets = _after(a)
        g.add_conditional_edges(f"{a}_agent", decide, targets)
    g.add_conditional_edges("route_case", lambda s: s["route"],
                            {"auto_clear": "auto_clear", "human_review": "human_review", "escalation": "escalation"})
    g.add_edge("auto_clear", "finalize")
    for node in ("human_review", "exception_review"):
        g.add_conditional_edges(node, lambda s: s.get("next_step", "finalize"),
                                {"escalation": "escalation", "finalize": "finalize"})
    g.add_edge("escalation", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def new_run(alert_id, model_cfg, fault=None, fault_agent=None, injection_mode=None, audit_path=None):
    run_id = f"{alert_id}-{uuid.uuid4().hex[:8]}"
    state = {"run_id": run_id, "alert_id": alert_id, "model": model_cfg, "fault": fault,
             "fault_agent": fault_agent, "injection_mode": injection_mode or config.INJECTION_MODE,
             "audit_path": str(audit_path) if audit_path else None}
    return run_id, state, {"configurable": {"thread_id": run_id}}


def pending_interrupt(graph, cfg):
    snap = graph.get_state(cfg)
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


__all__ = ["build_graph", "new_run", "pending_interrupt", "Command", "HUMAN_OPTIONS", "ACTION_FOR_LEVEL"]
