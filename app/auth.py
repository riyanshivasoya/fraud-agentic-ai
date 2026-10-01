"""Role-based access control: the role comes from the access list, never from the user's own choice."""
import hashlib
import json

from . import config

USERS_JSON = config.DATA_DIR / "users.json"


def load_users():
    with open(USERS_JSON, encoding="utf-8") as f:
        return json.load(f)["users"]


VIEWER = "Viewer (read only)"


def authenticate(name, pin):
    """Team members: name + correct PIN gives their listed role; wrong PIN is refused.
    Anyone else: signs in as a read-only Viewer (no PIN needed). Blank name is refused."""
    clean = (name or "").strip()
    if not clean:
        return None
    digest = hashlib.sha256(str(pin or "").strip().encode()).hexdigest()
    for u in load_users():
        if u["name"].lower() == clean.lower():
            return {"name": u["name"], "role": u["role"]} if u["pin_sha256"] == digest else None
    return {"name": clean, "role": VIEWER, "guest": True}


def can_run(role):
    return role in ("Fraud Analyst", "Senior Investigator")
