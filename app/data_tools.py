"""Tools the agents use: data lookup, feature calculation and the bank's rules engine.

These are deterministic Python functions. The LLM agents interpret their results;
they never change the numbers or the policy thresholds.
"""
import json
from datetime import datetime

import pandas as pd

from . import config

LEVELS = ["low", "medium", "high"]


def level_rank(level):
    return LEVELS.index(level) if level in LEVELS else -1


def max_level(*levels):
    valid = [lv for lv in levels if lv in LEVELS]
    return max(valid, key=level_rank) if valid else None


def load_rules():
    with open(config.RULES_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_customers():
    return pd.read_csv(config.CUSTOMERS_CSV, dtype=str).fillna("")


CUSTOM_ALERTS_CSV = config.DATA_DIR / "custom_alerts.csv"


def load_alerts():
    """The 8 fixed test alerts plus any alerts created in the UI."""
    df = pd.read_csv(config.TRANSACTIONS_CSV, dtype=str).fillna("")
    if CUSTOM_ALERTS_CSV.exists():
        df = pd.concat([df, pd.read_csv(CUSTOM_ALERTS_CSV, dtype=str).fillna("")], ignore_index=True)
    return df


def add_custom_alert(fields):
    """Save an alert typed into the UI form. Returns its new alert id."""
    existing = pd.read_csv(CUSTOM_ALERTS_CSV, dtype=str) if CUSTOM_ALERTS_CSV.exists() else None
    n = 1 if existing is None else len(existing) + 1
    row = {col: "" for col in pd.read_csv(config.TRANSACTIONS_CSV, nrows=0).columns}
    row.update({k: str(v) for k, v in fields.items()})
    row["alert_id"], row["txn_id"] = f"ALT-C{n:03d}", f"TXN-C{n:05d}"
    row.setdefault("alert_rule", "Manual test alert")
    new = pd.DataFrame([row])
    out = new if existing is None else pd.concat([existing, new], ignore_index=True)
    out.to_csv(CUSTOM_ALERTS_CSV, index=False)
    return row["alert_id"]


def clear_custom_alerts():
    if CUSTOM_ALERTS_CSV.exists():
        CUSTOM_ALERTS_CSV.unlink()


def load_test_cases():
    with open(config.TEST_CASES_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_alert(alert_id):
    df = load_alerts()
    rows = df[df["alert_id"] == alert_id]
    return rows.iloc[0].to_dict() if len(rows) else None


def get_customer(customer_id):
    df = load_customers()
    rows = df[df["customer_id"] == customer_id]
    return rows.iloc[0].to_dict() if len(rows) else None


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_features(txn, customer):
    """Turn raw records into facts. None means the fact cannot be checked (data missing)."""
    amount = _num(txn.get("amount"))
    avg = _num(customer.get("avg_txn_amount"))
    max90 = _num(customer.get("max_txn_amount_90d"))
    age = _num(customer.get("account_age_days"))
    velocity = _num(txn.get("txns_last_1h"))
    device = txn.get("device_id", "").strip()
    ip_country = txn.get("ip_country", "").strip()
    try:
        hour = datetime.fromisoformat(txn["timestamp"]).hour
    except (KeyError, ValueError):
        hour = None

    ratio = round(amount / avg, 2) if amount is not None and avg else None
    features = {
        "amount_inr": amount,
        "customer_avg_amount_inr": avg,
        "amount_to_average_ratio": ratio,
        "above_90d_max": (amount > max90) if amount is not None and max90 is not None else None,
        "is_new_device": (device != customer.get("usual_device_id")) if device else None,
        "is_new_location": txn.get("city", "") != customer.get("home_city"),
        "is_foreign_ip": (ip_country != customer.get("home_country")) if ip_country else None,
        "hour_of_day": hour,
        "is_night_time": (hour is not None and hour < 6) if hour is not None else None,
        "is_new_beneficiary": txn.get("beneficiary_new", "").upper() == "Y",
        "txns_last_1h": int(velocity) if velocity is not None else None,
        "is_high_velocity": (velocity >= 3) if velocity is not None else None,
        "account_age_days": int(age) if age is not None else None,
        "is_new_account": (age < 30) if age is not None else None,
        "kyc_risk_category": customer.get("kyc_risk_category"),
        "channel": txn.get("channel"),
        "channel_is_usual": txn.get("channel", "") in customer.get("usual_channels", "").split(";"),
    }
    missing = [name for name, key in (("device_id", "device_id"), ("ip_country", "ip_country"))
               if not txn.get(key, "").strip()]
    return features, missing


# Red-flag codes the agents are allowed to use, and the fact that proves each one.
FLAG_PREDICATES = {
    "AMOUNT_SPIKE": lambda f: None if f["amount_to_average_ratio"] is None else f["amount_to_average_ratio"] >= 3,
    "ABOVE_90D_MAX": lambda f: f["above_90d_max"],
    "NEW_DEVICE": lambda f: f["is_new_device"],
    "NEW_LOCATION": lambda f: f["is_new_location"],
    "FOREIGN_IP": lambda f: f["is_foreign_ip"],
    "NIGHT_TIME": lambda f: f["is_night_time"],
    "NEW_BENEFICIARY": lambda f: f["is_new_beneficiary"],
    "HIGH_VELOCITY": lambda f: f["is_high_velocity"],
    "NEW_ACCOUNT": lambda f: f["is_new_account"],
    "HIGH_KYC_RISK": lambda f: f["kyc_risk_category"] == "high",
    "UNUSUAL_CHANNEL": lambda f: not f["channel_is_usual"],
}
ALLOWED_FLAGS = list(FLAG_PREDICATES) + ["MISSING_DATA", "SUSPICIOUS_TEXT"]


def true_flags(features):
    return [code for code, test in FLAG_PREDICATES.items() if test(features) is True]


def run_rules_engine(features, missing_fields, injection_detected, rules=None):
    """Apply the bank policy. Returns score, triggered rules and the policy floor level."""
    rules = rules or load_rules()
    hits, score = [], 0

    ratio = features["amount_to_average_ratio"]
    if ratio is not None:
        tier = None
        for t in rules["amount_tiers"]:
            if ratio >= t["min_ratio"]:
                tier = t
        if tier:
            hits.append({"id": tier["id"], "description": tier["description"], "points": tier["points"]})
            score += tier["points"]

    flag_to_feature = {
        "NEW_DEVICE": "is_new_device", "NEW_LOCATION": "is_new_location", "FOREIGN_IP": "is_foreign_ip",
        "NIGHT_TIME": "is_night_time", "NEW_BENEFICIARY": "is_new_beneficiary",
        "HIGH_VELOCITY": "is_high_velocity", "NEW_ACCOUNT": "is_new_account", "ABOVE_90D_MAX": "above_90d_max",
    }
    for rule in rules["rules"]:
        if rule["flag"] == "HIGH_KYC_RISK":
            fired = features["kyc_risk_category"] == "high"
        else:
            fired = features.get(flag_to_feature[rule["flag"]]) is True
        if fired:
            hits.append({"id": rule["id"], "description": rule["description"], "points": rule["points"]})
            score += rule["points"]

    score = min(score, 100)
    t = rules["thresholds"]
    score_level = "high" if score >= t["high"] else "medium" if score >= t["medium"] else "low"

    floors = []
    new_dev_or_foreign = features["is_new_device"] is True or features["is_foreign_ip"] is True
    if ratio is not None and ratio >= 10 and new_dev_or_foreign:
        floors.append({"id": "HR1", "floor": "high"})
    if features["is_new_account"] and features["is_new_beneficiary"] and features["is_high_velocity"]:
        floors.append({"id": "HR2", "floor": "high"})
    if missing_fields:
        floors.append({"id": "DQ1", "floor": "medium"})
    if injection_detected:
        floors.append({"id": "DQ2", "floor": "medium"})

    policy_level = max_level(score_level, *[f["floor"] for f in floors])
    return {
        "rule_score": score,
        "score_level": score_level,
        "triggered_rules": hits,
        "floors_applied": floors,
        "policy_level": policy_level,
        "thresholds": t,
    }


def explain_score(features, rules_result, txn, profile):
    """'Why did the AI give this recommendation?' table: factor, value, rule triggered, score contribution."""
    rules = load_rules()
    f, rows = features, []
    amt = f["amount_inr"]
    for hit in rules_result["triggered_rules"]:
        rid = hit["id"]
        if rid.startswith("R01"):
            factor, value = "Amount vs customer average", f"₹{amt:,.0f} = {f['amount_to_average_ratio']}x average (₹{f['customer_avg_amount_inr']:,.0f})"
        elif rid == "R02":
            factor, value = "Device", f"{txn.get('device_id')} (not the customer's usual device)"
        elif rid == "R03":
            factor, value = "Location", f"{txn.get('city')} (home city: {profile.get('home_city')})"
        elif rid == "R04":
            factor, value = "IP country", f"{txn.get('ip_country')} (home country: {profile.get('home_country')})"
        elif rid == "R05":
            factor, value = "Time of day", str(txn.get("timestamp", ""))[11:16]
        elif rid == "R06":
            factor, value = "Beneficiary", "Paying a new beneficiary"
        elif rid == "R07":
            factor, value = "Velocity", f"{f['txns_last_1h']} transactions in the last hour"
        elif rid == "R08":
            factor, value = "Account age", f"{f['account_age_days']} days"
        elif rid == "R09":
            factor, value = "KYC risk category", str(f["kyc_risk_category"])
        else:  # R10
            factor, value = "Amount vs 90-day maximum", f"₹{amt:,.0f} > ₹{float(profile.get('max_txn_amount_90d_inr') or 0):,.0f}"
        rows.append({"Risk factor": factor, "Value": value, "Rule triggered": f"{rid}: {hit['description']}",
                     "Score contribution": f"+{hit['points']}"})
    descriptions = {r["id"]: r["description"] for r in rules["hard_rules"] + rules["data_floors"]}
    for fl in rules_result["floors_applied"]:
        rows.append({"Risk factor": "Policy floor", "Value": descriptions.get(fl["id"], ""),
                     "Rule triggered": fl["id"], "Score contribution": f"minimum level: {fl['floor']}"})
    rows.append({"Risk factor": "TOTAL", "Value": f"Score {rules_result['rule_score']} / 100",
                 "Rule triggered": f"low < {rules_result['thresholds']['medium']} ≤ medium < {rules_result['thresholds']['high']} ≤ high",
                 "Score contribution": f"policy level: {rules_result['policy_level'].upper()}"})
    return rows
