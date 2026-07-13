"""
Groq LLM helper for dynamic recommendation wording.

The API key is loaded from environment / .env only — never hard-code secrets here.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from .config import PROJECT_ROOT

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _load_dotenv_file() -> None:
    """Minimal .env loader (no dependency required for the key itself)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Do not override a key already provided by the shell/CI.
        if key and key not in os.environ:
            os.environ[key] = value


@lru_cache(maxsize=1)
def get_groq_settings() -> tuple[str | None, str]:
    _load_dotenv_file()
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_KEY")
    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    return api_key, model


def llm_available() -> bool:
    api_key, _ = get_groq_settings()
    return bool(api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_recommendation_copy(theme: str, confidence: float, severity: str, priority: str, facts: dict) -> dict[str, str] | None:
    """
    Ask Groq to write one action sentence + one evidence sentence from the fact pack.
    Returns None on any failure so the caller can fall back to local NLG.
    """
    api_key, model = get_groq_settings()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    # Keep the prompt compact and force JSON so parsing stays simple for coursework.
    user_payload = {
        "theme": theme,
        "gru_confidence": round(float(confidence), 4),
        "severity": severity,
        "priority": priority,
        "facts": facts,
    }
    system_prompt = (
        "You are a retail analytics assistant for an academic BI+AI project on Olist e-commerce. "
        "Write concise, professional recommendation copy. Use ONLY the provided facts and theme — "
        "do not invent metrics. Return strict JSON with keys: recommendation, evidence. "
        "recommendation = 1-2 sentences with a concrete business action. "
        "evidence = 1 sentence citing GRU confidence/severity and 2-4 key numbers from facts."
    )

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            temperature=0.4,
            max_tokens=280,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Compose recommendation JSON for this payload:\n"
                        + json.dumps(user_payload, ensure_ascii=True)
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        recommendation = str(parsed.get("recommendation", "")).strip()
        evidence = str(parsed.get("evidence", "")).strip()
        if not recommendation or not evidence:
            return None
        return {
            "recommendation": recommendation,
            "evidence": evidence,
            "model": model,
            "provider": "groq",
        }
    except Exception:
        # Silent fallback — local NLG still produces usable text for demos.
        return None
