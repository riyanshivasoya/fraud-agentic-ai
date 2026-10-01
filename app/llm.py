"""Model providers. Every agent calls models through `call_model`, so all models get
identical prompts, evidence and output format.

Providers
- ollama:   open-weight models running locally (Llama, Qwen, Mistral, Gemma...)
- openai:   any OpenAI-compatible endpoint serving open models (Groq, OpenRouter, LM Studio)
- offline:  NOT a model. A rule-based stand-in used only to test the workflow when no model
            is available. Never report its results as a model comparison.
"""
import json
import re
import time

import requests

from . import config


class ModelError(Exception):
    pass


def list_ollama_models():
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def _ollama(model, system, user):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": config.LLM_TEMPERATURE, "num_ctx": 4096},
    }
    try:
        r = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise ModelError(f"Cannot reach Ollama at {config.OLLAMA_HOST}. Is it running? ({e})")
    if r.status_code != 200:
        raise ModelError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


def _openai_compatible(model, system, user):
    if not config.OPENAI_API_KEY:
        raise ModelError("Set OPENAI_COMPAT_API_KEY to use a hosted endpoint.")
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    body = {
        "model": model,
        "temperature": config.LLM_TEMPERATURE,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    url = f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
    if r.status_code == 400:  # some providers do not support JSON mode; retry without it
        body.pop("response_format")
        r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise ModelError(f"API error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _offline(agent, user):
    """Deterministic stand-in: reads the evidence block and answers like a careful model."""
    from .data_tools import max_level, true_flags

    ev = json.loads(re.search(r"<evidence>\s*(\{.*\})\s*</evidence>", user, re.DOTALL).group(1))
    if agent == "transaction":
        flags = true_flags(ev["facts"])
        if ev.get("missing_fields"):
            flags = flags + ["MISSING_DATA"]
        return json.dumps({
            "red_flags": flags,
            "summary": f"{len(flags)} red flag(s) found in the transaction." if flags else "No red flags found.",
            "evidence_fields": ["amount_to_average_ratio", "is_new_device", "is_new_location"],
        })
    if agent == "behaviour":
        n = len(true_flags(ev["facts"]))
        level = "none" if n == 0 else "low" if n <= 1 else "moderate" if n <= 3 else "high"
        return json.dumps({
            "deviation_level": level,
            "observations": [f"{n} behaviour signal(s) differ from the customer's usual pattern."],
            "evidence_fields": ["avg_txn_amount_inr", "home_city", "usual_channels"],
        })
    if agent == "risk":
        rules = ev["rules_engine"]
        t, score = rules["thresholds"], rules["rule_score"]
        level = "high" if score >= t["high"] else "medium" if score >= t["medium"] else "low"
        level = max_level(level, *[f["floor"] for f in rules["floors_applied"]])
        return json.dumps({
            "risk_level": level,
            "risk_score": score,
            "rules_cited": [h["id"] for h in rules["triggered_rules"]][:4],
            "rationale": f"Rules engine score {score} gives {level} risk under policy thresholds.",
            "evidence_fields": ["rule_score", "triggered_rules", "floors_applied"],
        })
    if agent == "recommendation":
        level = ev["risk_assessment"]["risk_level"]
        action = {"low": "clear", "medium": "verify_with_customer", "high": "hold_and_escalate"}[level]
        return json.dumps({
            "recommended_action": action,
            "reasons": [ev["risk_assessment"]["rationale"]],
            "confidence": 0.8,
            "evidence_fields": ["risk_level", "rationale"],
        })
    raise ModelError(f"unknown agent {agent}")


def call_model(provider, model, system, user, agent, fault=None, attempt=1):
    """Returns (raw_text, latency_seconds). `fault` lets the demo simulate a broken model."""
    start = time.perf_counter()
    if fault == "all_attempts" or (fault == "first_attempt" and attempt == 1):
        raw = "Sorry, I think this looks risky but I cannot answer in JSON."
    elif provider == "ollama":
        raw = _ollama(model, system, user)
    elif provider == "openai":
        raw = _openai_compatible(model, system, user)
    elif provider == "offline":
        raw = _offline(agent, user)
    else:
        raise ModelError(f"unknown provider '{provider}'")
    return raw, round(time.perf_counter() - start, 2)
