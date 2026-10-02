"""Investigator dashboard. The UI triggers the LangGraph workflow and handles every human step.

Run:  streamlit run streamlit_app.py
"""
import json

import pandas as pd
import streamlit as st

from app import config
from app.agents import AGENTS
from app.audit import log_event, read_log, summarise_runs
from app.auth import authenticate, can_run, load_users
from app.data_tools import (add_custom_alert, clear_custom_alerts, explain_score, load_alerts, load_customers,
                            load_rules, load_test_cases)
from app.guardrails import mask_customer
from app.llm import list_ollama_models
from app.workflow import AGENT_ORDER, Command, build_graph, new_run, pending_interrupt

st.set_page_config(page_title="Fraud Investigation Assistant", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.hero {background: linear-gradient(100deg, #0b2e6b 0%, #1f5fd1 100%); color: #fff; padding: 20px 26px;
       border-radius: 14px; margin-bottom: 14px;}
.hero-title {font-size: 1.75rem; font-weight: 700; letter-spacing: -0.3px;}
.hero-sub {font-size: 0.95rem; opacity: 0.9; margin-top: 4px;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
       padding: 10px 14px; background: rgba(31,95,209,0.07);}
div[data-testid="stMetricValue"] {font-weight: 700;}
.stTabs [data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
       background-color: #1f5fd1 !important; border-color: #1f5fd1 !important; color: #fff !important;}
div[data-testid="stExpander"] details {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

ROUTE_LABEL = {
    "auto_clear": ("A", "Auto-cleared", "green"),
    "human_review": ("H", "Human review", "orange"),
    "escalation": ("E", "Escalated to Senior Investigator", "red"),
    "exception_review": ("H", "Exception review", "orange"),
}
LEVEL_COLOR = {"low": "green", "medium": "orange", "high": "red"}
NODE_LABEL = {
    "intake": "Orchestrator: case opened, data loaded, PII masked, rules engine run",
    "transaction_agent": "Transaction Analysis Agent",
    "behaviour_agent": "Customer Behaviour Agent",
    "risk_agent": "Risk / Policy Agent",
    "recommendation_agent": "Recommendation Agent",
    "route_case": "Policy gate: risk routing",
    "auto_clear": "Auto-clear (A)",
    "human_review": "Human review (H)",
    "exception_review": "Exception review (H)",
    "escalation": "Escalation (E)",
    "finalize": "Case closed and logged",
}


@st.cache_resource
def get_graph():
    return build_graph()  # one checkpointer, so paused cases survive page reruns


graph = get_graph()
ss = st.session_state
ss.setdefault("run", None)
ss.setdefault("history", [])
ss.setdefault("user", None)

# ------------------------------------------------------------------ page 1: sign-in
if ss.user is None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.header("🛡️ Fraud Investigation Assistant")
        st.caption("Governed Agentic AI for banking fraud investigation · proof of concept on synthetic data")
        st.subheader("Sign in")
        with st.form("login"):
            login_name = st.text_input("Name")
            login_pin = st.text_input("PIN", type="password", autocomplete="off")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                found = authenticate(login_name, login_pin)
                log_event("-", "-", login_name or "(blank)", "sign_in" if found else "sign_in_failed",
                          {"role": found["role"], "guest": found.get("guest", False)} if found else {})
                if found:
                    ss.user = found
                    st.rerun()
                st.error("Sign-in refused: enter your name, and the correct PIN if you are a team member.")
    st.stop()

user, role = ss.user["name"], ss.user["role"]


def sidebar_identity():
    st.header("Signed in")
    st.markdown(f"**{user}**  \n:blue[{role}]")
    if st.button("Sign out"):
        log_event("-", "-", user, "sign_out")
        ss.user = None  # the open case stays, so a senior can pick up an escalated case after an analyst
        st.rerun()


# ------------------------------------------------------------------ viewer page: test case list only
if not can_run(role):
    with st.sidebar:
        sidebar_identity()
    st.title("Fraud alert test cases")
    st.caption("View-only access. Investigations, decisions, model settings and audit records "
               "are available only to Fraud Analysts and Senior Investigators.")
    alerts = load_alerts()
    tc = pd.DataFrame(load_test_cases())
    view = tc.merge(alerts[["alert_id", "timestamp", "amount", "channel", "city", "alert_rule"]],
                    on="alert_id", how="left")
    view = view[["alert_id", "case_type", "timestamp", "amount", "channel", "city", "alert_rule",
                 "expected_route", "why"]].rename(columns={
                     "alert_id": "Alert", "case_type": "Test case", "timestamp": "Time", "amount": "Amount (INR)",
                     "channel": "Channel", "city": "City", "alert_rule": "Why flagged",
                     "expected_route": "Expected route", "why": "What it tests"})
    st.dataframe(view, hide_index=True, width="stretch")
    st.stop()

# ------------------------------------------------------------------ page 2: investigation workspace
with st.sidebar:
    sidebar_identity()

    st.header("Model")
    provider = "ollama"
    st.caption("Provider: Ollama (local open models)")
    available = list_ollama_models()
    if available:
        default = next((m for m in config.DEFAULT_MODELS if m in available), available[0])
        model = st.selectbox("Model", available, index=available.index(default), label_visibility="collapsed")
    else:
        st.warning(f"Ollama not reachable at {config.OLLAMA_HOST}. Start the Ollama app.")
        model = st.text_input("Model", value=config.DEFAULT_MODELS[0], label_visibility="collapsed")

    with st.expander("Demo controls"):
        fault_label = st.selectbox("Simulate model failure", ["None", "Model fails once (retry recovers)",
                                                              "Model fails twice (goes to exception review)"])
        fault = {"None": None, "Model fails once (retry recovers)": "first_attempt",
                 "Model fails twice (goes to exception review)": "all_attempts"}[fault_label]
        injection_mode = st.radio("Injection guardrail", ["redact", "flag_only"],
                                  help="redact: suspicious text never reaches the model. flag_only: model sees it, flagged.")

st.markdown('<div class="hero"><div class="hero-title">🛡️ Governed Agentic AI for Fraud Investigation</div>'
            '<div class="hero-sub">Agents gather and explain evidence · humans decide · '
            'A = autonomous · H = human-in-the-loop · E = escalation · synthetic data only</div></div>',
            unsafe_allow_html=True)

tab_inv, tab_flow, tab_cmp, tab_audit, tab_data = st.tabs(
    ["🔎 1. Investigate", "🧭 2. Workflow graph", "📊 3. Model comparison", "🧾 4. Audit log",
     "🛡️ 5. Data and guardrails"])


# ------------------------------------------------------------------ helpers

def stream_and_show(inp, cfg):
    with st.status("Workflow running...", expanded=True) as status:
        for update in graph.stream(inp, cfg, stream_mode="updates"):
            for node, out in update.items():
                if node == "__interrupt__":
                    st.write("⏸️ Paused: waiting for a human decision")
                    continue
                extra = ""
                if isinstance(out, dict) and out.get("agent_results"):
                    agent = node.replace("_agent", "")
                    r = out["agent_results"].get(agent)
                    if r:
                        extra = f" ({r['latency_s']} s, {len(r['attempts'])} attempt(s))" + ("" if r["ok"] else " FAILED")
                st.write(f"✔ {NODE_LABEL.get(node, node)}{extra}")
        status.update(label="Workflow step complete", state="complete", expanded=False)


def show_agent(agent, r):
    spec = AGENTS[agent]
    title = f"{spec['title']}  ·  {'OK' if r['ok'] else 'FAILED'}  ·  {r['latency_s']} s"
    with st.expander(title, expanded=agent in ("risk", "recommendation")):
        if r["ok"]:
            out = r["output"]
            if agent == "transaction":
                st.write("**Red flags:**", ", ".join(out["red_flags"]) or "none")
                st.write(out["summary"])
            elif agent == "behaviour":
                st.write(f"**Deviation:** :{ {'none': 'green', 'low': 'green', 'moderate': 'orange', 'high': 'red'}[out['deviation_level']] }[{out['deviation_level']}]")
                for o in out["observations"]:
                    st.write("-", o)
            elif agent == "risk":
                st.write(f"**Model risk level:** :{LEVEL_COLOR[out['risk_level']]}[{out['risk_level'].upper()}]  "
                         f"· score {int(out['risk_score'])} · rules cited: {', '.join(out['rules_cited']) or 'none'}")
                st.write(out["rationale"])
            else:
                st.write(f"**Recommended:** `{out['recommended_action']}` · confidence {out['confidence']:.2f}")
                for reason in out["reasons"]:
                    st.write("-", reason)
            st.caption("Evidence cited: " + ", ".join(out["evidence_fields"]))
        else:
            st.error(f"Agent failed after {len(r['attempts'])} attempt(s): {r.get('error')}")
        probs = r.get("evidence_problems", [])
        false_claims = [p for p in probs if p["type"] == "hallucination"]
        naming = [p for p in probs if p["type"] == "citation"]
        if false_claims:
            st.warning(f"**Evidence check: {len(false_claims)} claim(s) do not match the data**\n\n"
                       + "\n".join(f"- {p['detail']}" for p in false_claims))
        elif r["ok"]:
            st.caption("✓ Evidence check passed: no claims contradict the data.")
        if naming:
            st.caption("Note: the model " + "; ".join(p["detail"] for p in naming)
                       + ". This is a naming slip, not a false claim.")
        if len(r["attempts"]) > 1:
            st.info(f"Retry used: first reply rejected ({'; '.join(r['attempts'][0].get('schema_errors', ['error']))}).")
        with st.popover("Prompt and raw model reply"):
            st.code(r["prompt"]["system"], language="text")
            st.code(r["prompt"]["user"], language="text")
            for a in r["attempts"]:
                st.caption(f"Attempt {a['attempt']}")
                st.code(a.get("raw") or a.get("error", ""), language="json")


def show_case(values, cfg):
    route = values.get("route") or ("exception_review" if values.get("status") == "exception" else None)
    rules = values.get("rules_result") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rule score", rules.get("rule_score", "-"))
    c2.metric("Model risk level", (values.get("llm_risk_level") or "-").upper())
    c3.metric("Final risk level (policy)", (values.get("final_level") or "-").upper())
    if route:
        code, label, color = ROUTE_LABEL[route]
        c4.markdown(f"**Route**  \n:{color}[**[{code}] {label}**]")

    if values.get("data_error"):
        st.error(f"Data exception: {values['data_error']}")
    if values.get("failed_agent"):
        failed = (values.get("agent_results") or {}).get(values["failed_agent"], {})
        tries = len(failed.get("attempts", []))
        st.error(f"Model exception: {AGENTS[values['failed_agent']]['title']} could not give a valid answer "
                 f"({tries} attempt{'s' if tries != 1 else ''}). The case was sent to a human. "
                 f"Reason: {failed.get('error', 'unknown')}")
    for reason in values.get("route_reasons") or []:
        st.warning(f"Guardrail: {reason}")
    if values.get("missing_fields"):
        st.info(f"Missing evidence: {', '.join(values['missing_fields'])}. Case cannot be auto-cleared (DQ1).")
    for f in values.get("injection_findings") or []:
        st.warning(f"Injection guardrail: instructions found in '{f['field']}' and "
                   f"{'removed before the model saw them' if values.get('injection_mode') == 'redact' else 'flagged'} (DQ2).")
    if values.get("masked_customer"):
        st.caption("PII masked: the models saw customer_ref and a masked account number, never the name or phone.")

    if rules:
        st.subheader("Why did the AI give this recommendation?")
        st.dataframe(pd.DataFrame(explain_score(values["features"], rules, values["txn"], values["masked_customer"])),
                     hide_index=True, width="stretch")
        st.caption("The score is calculated by the rules engine (a tool), line by line, so it can be checked by hand. "
                   "The AI agents explain it and may raise the risk, but can never lower it below this policy level.")

    results = values.get("agent_results") or {}
    if results:
        st.subheader("Agent findings")
        for agent in AGENT_ORDER:
            if agent in results:
                show_agent(agent, results[agent])

    pause = pending_interrupt(graph, cfg)
    if pause:
        st.subheader(f"Decision needed: {pause['label']}")
        st.write(f"Allowed roles: **{', '.join(pause['allowed_roles'])}** · you are **{role}**")
        if pause.get("recommended"):
            st.write(f"System recommendation: `{pause['recommended']}` (final risk {pause.get('final_level')})")
        if pause.get("error"):
            st.error(pause["error"])
        if role is None:
            st.info("Sign in (left sidebar) to make a decision.")
            return
        if role not in pause["allowed_roles"]:
            st.info("Your role cannot decide this case. You can still submit to see the server-side access check.")
        with st.form(f"decision-{values['run_id']}-{len(values.get('human_decisions') or [])}"):
            options = pause["options"]
            action = st.radio("Decision", list(options), format_func=lambda k: options[k])
            comment = st.text_area("Reason (required, min 10 characters, goes to the audit log)")
            if st.form_submit_button("Submit decision", type="primary"):
                stream_and_show(Command(resume={"role": role, "user": user, "action": action, "comment": comment}), cfg)
                st.rerun()
    elif values.get("status") == "closed":
        st.success(f"Outcome: **{values.get('outcome')}**. Every step is in the audit log (tab 4).")
        for d in values.get("human_decisions") or []:
            st.write(f"- {d['stage']}: **{d['action']}** by {d['user']} ({d['role']}): {d['comment']}")


# ------------------------------------------------------------------ tab 1: investigate
with tab_inv:
    alerts = load_alerts()
    cases = {c["alert_id"]: c for c in load_test_cases()}

    # dashboard counters, from the audit log
    runs = summarise_runs(read_log())

    def waiting(r):
        if r["closed"] or r["route"] in (None, "auto_clear"):
            return False
        return pending_interrupt(graph, {"configurable": {"thread_id": r["run_id"]}}) is not None

    m = st.columns(7)
    m[0].metric("Alerts", len(alerts), help="Alerts in the queue")
    m[1].metric("Cases run", len(runs), help="Investigations run so far (from the audit log)")
    m[2].metric("Low risk", sum(r["final_level"] == "low" for r in runs))
    m[3].metric("Medium risk", sum(r["final_level"] == "medium" for r in runs))
    m[4].metric("High risk", sum(r["final_level"] == "high" for r in runs))
    m[5].metric("Pending", sum(waiting(r) for r in runs), help="Cases waiting for a human decision")
    m[6].metric("Closed", sum(r["closed"] for r in runs))

    st.subheader("Alert queue")
    queue = alerts[["alert_id", "timestamp", "customer_id", "amount", "channel", "city", "alert_rule"]].copy()
    queue.insert(1, "test case", queue["alert_id"].map(
        lambda a: cases.get(a, {}).get("case_type", "Custom (created in the UI)")))
    st.dataframe(queue, hide_index=True, width="stretch")

    if "pending_pick" in ss:
        ss["alert_pick"] = ss.pop("pending_pick")
    left, right = st.columns([3, 1])
    alert_id = left.selectbox("Select alert", queue["alert_id"], key="alert_pick",
                              format_func=lambda a: f"{a} · {cases.get(a, {}).get('case_type', 'Custom')}")
    right.write("")
    right.write("")
    allowed_to_run = role is not None and can_run(role)
    if right.button("Run investigation", type="primary", width="stretch", disabled=not allowed_to_run,
                    help=None if allowed_to_run else "Sign in as a Fraud Analyst or Senior Investigator to run cases."):
        run_id, state, cfg = new_run(alert_id, {"provider": provider, "model": model},
                                     fault=fault, injection_mode=injection_mode)
        ss.run = {"run_id": run_id, "cfg": cfg, "alert_id": alert_id, "model": model}
        ss.history.insert(0, ss.run)
        stream_and_show(state, cfg)
        st.rerun()  # refresh the dashboard counters


    with st.expander("Create a new alert (test any scenario live)"):
        customers_df = load_customers()
        with st.form("new_alert"):
            c1, c2, c3 = st.columns(3)
            cust = c1.selectbox("Customer", list(customers_df["customer_id"]) + ["C999 (not in customer master)"])
            amount = c1.number_input("Amount (INR)", min_value=1, value=25000, step=1000)
            channel = c1.selectbox("Channel", ["UPI", "POS", "CARD_ONLINE", "NETBANKING", "ATM"])
            city = c2.text_input("City", value="Pune")
            country = c2.text_input("Country code", value="IN")
            device_choice = c2.radio("Device", ["Customer's usual device", "New device", "Missing (no data)"])
            ip_country = c3.text_input("IP country (leave blank = missing)", value="IN")
            txn_date = c3.date_input("Date")
            txn_time = c3.time_input("Time")
            c4, c5 = st.columns(2)
            new_ben = c4.checkbox("New beneficiary")
            velocity = c4.number_input("Transactions in the last hour", min_value=0, value=1)
            merchant = c5.text_input("Merchant / payee", value="Test merchant")
            remarks = c5.text_input("Remarks (free text)", value="")
            if st.form_submit_button("Add alert to queue"):
                cid = cust.split(" ")[0]
                row = customers_df[customers_df["customer_id"] == cid]
                usual = row.iloc[0]["usual_device_id"] if len(row) else "DEV-UNKNOWN"
                device = {"Customer's usual device": usual, "New device": "DEV-NEW-" + str(int(amount) % 9973),
                          "Missing (no data)": ""}[device_choice]
                new_id = add_custom_alert({
                    "customer_id": cid, "timestamp": f"{txn_date}T{txn_time.strftime('%H:%M')}:00",
                    "amount": int(amount), "currency": "INR", "channel": channel, "merchant_category": "Manual test",
                    "merchant_name": merchant, "city": city, "country": country, "device_id": device,
                    "ip_country": ip_country.strip(), "beneficiary_new": "Y" if new_ben else "N",
                    "txns_last_1h": int(velocity), "alert_rule": "Manual test alert", "remarks": remarks,
                })
                log_event("-", new_id, user or "(signed out)", "custom_alert_created", {"amount": int(amount), "city": city})
                ss["pending_pick"] = new_id
                st.rerun()
        if st.button("Delete all custom alerts"):
            clear_custom_alerts()
            st.rerun()

    if ss.run:
        cfg = ss.run["cfg"]
        values = graph.get_state(cfg).values
        st.divider()
        st.subheader(f"Case {ss.run['run_id']} · model: {ss.run['model']}")
        show_case(values, cfg)

    if len(ss.history) > 1:
        with st.expander("Earlier runs this session"):
            pick = st.selectbox("Open run", [h["run_id"] for h in ss.history])
            if st.button("Open"):
                ss.run = next(h for h in ss.history if h["run_id"] == pick)
                st.rerun()

# ------------------------------------------------------------------ tab 2: workflow graph
with tab_flow:
    st.subheader("LangGraph workflow (generated from the running code)")
    mermaid = graph.get_graph().draw_mermaid()
    png = config.BASE_DIR / "docs" / "workflow_graph.png"
    _, gmid, _ = st.columns([1, 2, 1])
    with gmid:
        if png.exists():
            st.image(str(png), caption="Solid arrows = fixed path · dotted arrows = conditional routing",
                     width="stretch")
    st.markdown("**What each step does**")
    st.dataframe(pd.DataFrame([
        ("intake", "Orchestrator", "A"), ("transaction_agent", "Transaction Analysis Agent", "A"),
        ("behaviour_agent", "Customer Behaviour Agent", "A"), ("risk_agent", "Risk / Policy Agent", "A"),
        ("recommendation_agent", "Recommendation Agent", "A"), ("route_case", "Policy gate (conditional routing)", "A"),
        ("auto_clear", "Low risk, logged, QA sampled", "A"), ("human_review", "Fraud Analyst decides", "H"),
        ("exception_review", "Missing data or model failure", "H"), ("escalation", "Senior Investigator decides", "E"),
    ], columns=["Node", "Role", "Label"]), hide_index=True, width="stretch")
    st.download_button("Download graph (Mermaid)", mermaid, file_name="workflow_graph.mmd")

# ------------------------------------------------------------------ tab 3: model comparison
with tab_cmp:
    st.subheader("Open-model comparison")
    summary_path = config.RESULTS_DIR / "benchmark_summary.csv"
    runs_path = config.RESULTS_DIR / "benchmark_runs.csv"
    if not summary_path.exists():
        st.info("No results yet. In a terminal run:  `python benchmark.py`")
    else:
        summary = pd.read_csv(summary_path)
        runs = pd.read_csv(runs_path)
        if "offline-test-mode" in summary["Model"].values:
            st.warning("These results come from offline test mode, not a model. Run `python benchmark.py` with Ollama.")
        st.caption("Same 8 cases, prompts, evidence and JSON schema for every model; temperature 0.")
        st.dataframe(summary.set_index("Model").T.astype(str), width="stretch")
        a, b = st.columns(2)
        with a:
            st.write("**Accuracy (%)**")
            st.bar_chart(summary.set_index("Model")[["Risk classification accuracy, model alone (%)",
                                                     "Routing accuracy with guardrails (%)",
                                                     "Valid JSON first try (%)"]], stack=False)
        with b:
            st.write("**Average model time per case (s)**")
            st.bar_chart(summary.set_index("Model")[["Avg model time per case (s)"]])
        st.write("**Per case**")
        st.dataframe(runs[["model", "alert_id", "case_type", "expected_level", "model_level", "final_level",
                           "expected_route", "actual_route", "route_correct", "hallucinations", "problems",
                           "model_latency_s"]], hide_index=True, width="stretch")
        st.download_button("Download per-case results (CSV)", runs.to_csv(index=False), "benchmark_runs.csv")

# ------------------------------------------------------------------ tab 4: audit log
with tab_audit:
    st.subheader("Audit trail")
    log = read_log()
    if not log:
        st.info("No events yet. Run an investigation.")
    else:
        df = pd.DataFrame(log)
        df["details"] = df["details"].map(lambda d: json.dumps(d, default=str)[:400])
        runs_list = ["All"] + list(dict.fromkeys(df["run_id"][::-1]))
        pick = st.selectbox("Filter by case run", runs_list)
        view = df if pick == "All" else df[df["run_id"] == pick]
        st.dataframe(view.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download audit log (JSONL)", config.AUDIT_LOG.read_text(encoding="utf-8"), "audit_log.jsonl")

# ------------------------------------------------------------------ tab 5: data and guardrails
with tab_data:
    st.subheader("Synthetic data")
    customers = load_customers()
    st.write("**What the models see (PII masked)**")
    st.dataframe(pd.DataFrame([mask_customer(c) for c in customers.to_dict("records")]), hide_index=True,
                 width="stretch")
    if role == "Senior Investigator" and st.toggle("Show unmasked synthetic records (Senior Investigator only)"):
        st.dataframe(customers, hide_index=True, width="stretch")
    st.write("**Test cases**")
    st.dataframe(pd.DataFrame(load_test_cases()), hide_index=True, width="stretch")
    st.subheader("Fraud policy")
    st.caption("Set by the bank's fraud policy owner (data/rules.json), not by the AI model.")
    policy = load_rules()
    t = policy["thresholds"]
    st.write("**Risk thresholds**")
    st.dataframe(pd.DataFrame([
        ("Low", f"below {t['medium']}", "Auto-clear (A), logged and sampled for quality review"),
        ("Medium", f"{t['medium']} to {t['high'] - 1}", "Human review by a Fraud Analyst (H)"),
        ("High", f"{t['high']} and above", "Escalation to a Senior Investigator (E)"),
    ], columns=["Risk level", "Rule score (0-100)", "What happens"]), hide_index=True, width="stretch")
    st.write("**Scoring rules**")
    rows = [(r["id"], r["description"], f"+{r['points']}") for r in policy["amount_tiers"]]
    rows += [(r["id"], r["description"], f"+{r['points']}") for r in policy["rules"]]
    st.dataframe(pd.DataFrame(rows, columns=["Rule", "What it checks", "Points"]), hide_index=True, width="stretch")
    st.caption("Only the highest amount rule (R01A, R01B or R01C) counts. The total is capped at 100.")
    st.write("**Safety floors (minimum risk level, whatever the score)**")
    st.dataframe(pd.DataFrame([(r["id"], r["description"], r["floor"].capitalize())
                               for r in policy["hard_rules"] + policy["data_floors"]],
                              columns=["Floor", "When it applies", "Minimum risk level"]),
                 hide_index=True, width="stretch")
    st.subheader("Guardrails in this proof of concept")
    st.dataframe(pd.DataFrame([
        ("Data", "PII masking", "Name and phone never sent; account number masked"),
        ("Data", "Injection filter", "Instruction-like text in remarks is removed or flagged; case floored to medium"),
        ("Data", "Missing-data floor", "Missing device or IP data blocks auto-clear"),
        ("Model", "JSON schema check", "Every agent reply is validated; one retry with the errors, then human review"),
        ("Model", "Evidence check", "Claims that contradict the data or cite rules that did not fire are flagged"),
        ("Model", "Bias instruction", "Agents told not to judge by name, gender, religion, region or nationality"),
        ("Action", "Policy floor", "The model can raise risk, never lower it below the rules engine"),
        ("Action", "No autonomous customer impact", "Only low-risk cases auto-clear; holds need a Senior Investigator"),
        ("Oversight", "Role-based approval", "Checked inside the workflow, not only in the UI"),
        ("Audit", "Audit log", "Every agent output, model, time, override and human decision"),
    ], columns=["Type", "Guardrail", "How it works"]), hide_index=True, width="stretch")
