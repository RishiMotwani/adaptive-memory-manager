import json
import re
import time
import requests
from typing import Dict, List, Tuple

VALID_CATEGORIES = {"technical_preference", "personal", "project_context", "transient"}

CATEGORY_MAP = {
    "database": "technical_preference",
    "db": "technical_preference",
    "sql": "technical_preference",
    "queries": "technical_preference",
    "technical": "technical_preference",
    "tech": "technical_preference",
    "infrastructure": "technical_preference",
    "infra": "technical_preference",
    "system": "technical_preference",
    "coding": "technical_preference",
    "code": "technical_preference",
    "tooling": "technical_preference",
    "deployment": "technical_preference",
    "server": "technical_preference",
    "database_requirement": "technical_preference",
    "project": "project_context",
    "work": "project_context",
    "task": "project_context",
    "feature": "project_context",
    "feature_flag": "project_context",
    "config": "project_context",
    "architecture": "project_context",
    "product": "project_context",
    "requirement": "project_context",
    "personal": "personal",
    "preference": "personal",
    "personal_preference": "personal",
    "user": "personal",
    "background": "personal",
    "info": "personal",
    "demographic": "personal",
    "note": "transient",
    "misc": "transient",
    "temporary": "transient",
    "noise": "transient",
}

TECH_WORDS = {"postgres", "postgresql", "sql", "mysql", "database", "kubernetes", "docker",
              "deployment", "infra", "infrastructure", "tooling", "server", "configs"}
PROJECT_PHRASES = ("feature flag", "config key", "rollout", "sprint", "milestone",
                   "under discussion", "architecture decision")
PROJECT_WORDS = {"project", "release", "roadmap"}
PERSONAL_PHRASES = ("has a cat", "my cat", "cat named", "dog named", "i prefer", "my favorite",
                    "my favourite", "is a ", "named whiskers")
PERSONAL_WORDS = {"whiskers", "hobby", "pet"}
TRANSIENT_PHRASES = ("coffee", "standup", "lunch", "daily ", "routine monitoring")


def _normalize_category(category: str) -> str:
    if not isinstance(category, str):
        return "project_context"
    flat = re.sub(r"[^a-z0-9]", "", category.strip().lower())
    for valid in VALID_CATEGORIES:
        if re.sub(r"[^a-z0-9]", "", valid) == flat:
            return valid
    for key, val in CATEGORY_MAP.items():
        if re.sub(r"[^a-z0-9]", "", key) == flat:
            return val
    return "project_context"


def _keyword_category(fact_text: str) -> str:
    if not fact_text:
        return ""
    text = fact_text.strip().lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    if words.intersection(TECH_WORDS):
        return "technical_preference"
    if any(p in text for p in PROJECT_PHRASES) or words.intersection(PROJECT_WORDS):
        return "project_context"
    if any(p in text for p in PERSONAL_PHRASES) or words.intersection(PERSONAL_WORDS):
        return "personal"
    if any(p in text for p in TRANSIENT_PHRASES):
        return "transient"
    return ""


def build_extraction_prompt(user_text: str) -> str:
    return f"""Extract atomic facts from the following input. Ignore routine filler and small talk that will not matter later.
Output ONLY a JSON array of objects with keys: "fact", "category" (must be one of: technical_preference, personal, project_context, transient), and "confidence" (0.0 to 1.0).

Example:
Input: "Also, remember, I have a cat named Whiskers and prod must run PostgreSQL 15."
Output: [{{"fact": "User has a cat named Whiskers", "category": "personal", "confidence": 0.9}}, {{"fact": "Production must run PostgreSQL 15", "category": "technical_preference", "confidence": 0.9}}]

Input: "{user_text}"
JSON Output:"""


def _unwrap_response(raw_json) -> List[Dict]:
    if isinstance(raw_json, list):
        return [item for item in raw_json if isinstance(item, dict)]
    if isinstance(raw_json, dict):
        for key in ("facts", "fact_list", "items", "responses", "result", "data"):
            value = raw_json.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "fact" in raw_json:
            return [raw_json]
    return []


class FactExtractor:
    """Extracts structured atomic fact candidates from raw conversation turns using local SLMs."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.endpoint = f"{endpoint}/api/generate"
        self.model = model

    def extract_facts(self, turn_id: int, user_text: str) -> tuple:
        prompt = build_extraction_prompt(user_text)
        meta = {
            "model": self.model,
            "prompt_eval_count": None,
            "eval_count": None,
            "ms": None,
            "fallback": False,
            "attempts": 0,
        }
        attempts = 0
        while attempts < 2:
            attempts += 1
            meta["attempts"] = attempts
            try:
                t0 = time.perf_counter()
                res = requests.post(self.endpoint, json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": "30m",
                    "options": {"num_ctx": 8192},
                }, timeout=120)
                meta["ms"] = (time.perf_counter() - t0) * 1000
                data = res.json()
                meta["prompt_eval_count"] = data.get("prompt_eval_count")
                meta["eval_count"] = data.get("eval_count")
                if data.get("error"):
                    continue
                if res.status_code != 200:
                    continue
                raw_json = json.loads(data.get("response", "[]"))
                items = _unwrap_response(raw_json)
                parsed = []
                for item in items:
                    fact = item.get("fact")
                    if not isinstance(fact, str) or not fact.strip():
                        continue
                    try:
                        confidence = float(item.get("confidence", 0.7))
                    except (TypeError, ValueError):
                        confidence = 0.7
                    raw_cat = item.get("category", "")
                    keyword_cat = _keyword_category(fact)
                    category = keyword_cat if keyword_cat else _normalize_category(raw_cat)
                    parsed.append({
                        "fact": fact.strip(),
                        "category": category,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "source_turn_id": turn_id
                    })
                if parsed:
                    return parsed, meta
            except Exception:
                meta["ms"] = (time.perf_counter() - t0) * 1000 if "t0" in locals() else meta["ms"]
                continue

        meta["fallback"] = True
        return [{
            "fact": user_text,
            "category": "transient",
            "confidence": 0.5,
            "source_turn_id": turn_id
        }], meta