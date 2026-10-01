"""Guardrails: data (PII, injection), model (JSON schema, evidence check) and action (policy floor)."""
import json
import re

from .data_tools import ALLOWED_FLAGS, FLAG_PREDICATES

# ---------------------------------------------------------------- data guardrails

def mask_account(number):
    number = str(number or "")
    return ("X" * max(len(number) - 4, 0)) + number[-4:] if number else ""


def mask_customer(customer):
    """Only what the model needs. Name and phone never leave the bank's side."""
    return {
        "customer_ref": "CUST-" + customer["customer_id"][1:],
        "account_number_masked": mask_account(customer.get("account_number")),
        "home_city": customer.get("home_city"),
        "home_country": customer.get("home_country"),
        "usual_channels": customer.get("usual_channels"),
        "avg_txn_amount_inr": customer.get("avg_txn_amount"),
        "max_txn_amount_90d_inr": customer.get("max_txn_amount_90d"),
        "account_age_days": customer.get("account_age_days"),
        "kyc_risk_category": customer.get("kyc_risk_category"),
        "segment": customer.get("segment"),
    }


INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts)",
    r"disregard (the |all |your )?(instructions|rules|policy)",
    r"\bsystem prompt\b",
    r"you are now\b",
    r"\bact as\b",
    r"mark (the )?(risk|this|alert|transaction)?\s*(as )?(low|safe|clear|genuine)",
    r"(clear|close|approve) (the |this )?(alert|case|transaction)",
    r"verified by the bank",
    r"\bjailbreak\b",
    r"<\|.*?\|>",
    r"###\s*(instruction|system)",
]
FREE_TEXT_FIELDS = ["remarks", "merchant_name"]


def detect_injection(text):
    text = str(text or "")
    return [p for p in INJECTION_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]


def sanitise_transaction(txn, mode="redact"):
    """Check free-text fields for instructions aimed at the AI. Returns (safe_txn, findings)."""
    safe, findings = dict(txn), []
    for field in FREE_TEXT_FIELDS:
        matches = detect_injection(txn.get(field))
        if matches:
            findings.append({"field": field, "patterns": matches})
            if mode == "redact":
                safe[field] = "[REMOVED BY GUARDRAIL: possible instruction injection]"
            else:
                safe[field] = "[UNTRUSTED TEXT, FLAGGED] " + str(txn.get(field))
    return safe, findings


# ---------------------------------------------------------------- model guardrails

def extract_json(raw):
    """Parse a JSON object from model text (handles code fences and extra words)."""
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def validate(obj, schema):
    """Small schema check: required keys, types, allowed values, ranges. Returns (clean, errors)."""
    errors, clean = [], {}
    for key, rule in schema.items():
        if key not in obj:
            errors.append(f"missing key '{key}'")
            continue
        value = obj[key]
        kind = rule["type"]
        if kind == "str":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"'{key}' must be a non-empty string")
                continue
            value = value.strip()
            if "enum" in rule:
                value = value.lower()
                if value not in rule["enum"]:
                    errors.append(f"'{key}' must be one of {rule['enum']}")
                    continue
        elif kind == "list":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                errors.append(f"'{key}' must be a list")
                continue
            value = [str(v).strip() for v in value if str(v).strip()]
            if rule.get("min", 0) and len(value) < rule["min"]:
                errors.append(f"'{key}' needs at least {rule['min']} item(s)")
                continue
            if "max" in rule:
                value = value[: rule["max"]]
            if rule.get("upper"):
                value = [v.upper() for v in value]
        elif kind == "num":
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number")
                continue
            lo, hi = rule.get("range", (None, None))
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                errors.append(f"'{key}' must be between {lo} and {hi}")
                continue
        elif kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"'{key}' must be true or false")
                continue
        clean[key] = value
    return clean, errors


FLAG_TEXT = {
    "AMOUNT_SPIKE": "amount is at least 3x the customer's average",
    "ABOVE_90D_MAX": "amount is above the customer's 90-day maximum",
    "NEW_DEVICE": "payment came from a new device",
    "NEW_LOCATION": "payment was made away from the home city",
    "FOREIGN_IP": "IP address is from another country",
    "NIGHT_TIME": "payment was made at night (00:00 to 05:59)",
    "NEW_BENEFICIARY": "payment went to a new beneficiary",
    "HIGH_VELOCITY": "customer made 3 or more payments in the last hour",
    "NEW_ACCOUNT": "account is less than 30 days old",
    "HIGH_KYC_RISK": "customer's KYC risk category is high",
    "UNUSUAL_CHANNEL": "payment channel is unusual for this customer",
}


def evidence_check(agent, output, features, allowed_fields, triggered_rule_ids):
    """Check every claim against the real data.

    hallucination = a claim that contradicts the data or invents a flag/rule.
    citation      = cites a field name that does not exist (sloppy, not a false fact).
    """
    problems = []
    for field in output.get("evidence_fields", []):
        if field not in allowed_fields:
            problems.append({"type": "citation", "detail": f"referred to a data field that does not exist ('{field}')"})
    for code in output.get("red_flags", []):
        if code not in ALLOWED_FLAGS:
            problems.append({"type": "hallucination", "detail": f"Used a warning sign that is not on the allowed list ({code})."})
        elif code in FLAG_PREDICATES and FLAG_PREDICATES[code](features) is False:
            problems.append({"type": "hallucination",
                             "detail": f"Said the {FLAG_TEXT.get(code, code)}, but the data shows this is not true ({code})."})
    for rule_id in output.get("rules_cited", []):
        if rule_id not in triggered_rule_ids:
            problems.append({"type": "hallucination",
                             "detail": f"Cited rule {rule_id}, but that rule did not trigger for this case."})
    return problems


# ---------------------------------------------------------------- action guardrails

ACTION_FOR_LEVEL = {"low": "clear", "medium": "verify_with_customer", "high": "hold_and_escalate"}
ACTION_RANK = ["clear", "verify_with_customer", "hold_and_escalate"]


def governed_action(llm_action, final_level):
    """The model may suggest a stricter action, never a weaker one than policy allows."""
    policy_action = ACTION_FOR_LEVEL[final_level]
    if llm_action in ACTION_RANK and ACTION_RANK.index(llm_action) > ACTION_RANK.index(policy_action):
        return llm_action
    return policy_action
