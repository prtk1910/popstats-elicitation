"""OpenRouter client with ledger — mirrors gtfsplan.api for this standalone project."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute(
            """CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, model TEXT, request_json TEXT,
            response_json TEXT, content TEXT,
            prompt_tokens INT, completion_tokens INT, latency_ms INT, ts TEXT)"""
        )
        self.con.commit()

    def get(self, key: str):
        with self.lock:
            row = self.con.execute(
                "SELECT response_json, content FROM calls WHERE key = ?", (key,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key, model, request_json, response_json, content, ptok, ctok, lat):
        with self.lock:
            self.con.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
            (key, model, request_json, response_json, content, ptok, ctok, lat),
        )
        self.con.commit()


class OpenRouterClient:
    def __init__(self, ledger_path: Path, model: str | None = None):
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key or "replace-with" in self.key:
            raise RuntimeError("OPENROUTER_API_KEY missing; set it in the root .env")
        self.model = model or os.environ.get("OX_MODEL", "stealth/ox-alpha")
        self.ledger = Ledger(ledger_path)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.key}",
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "research-evals"),
                "Content-Type": "application/json",
            }
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = 12,
    ):
        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        hit = self.ledger.get(key)
        if hit is not None:
            body = json.loads(hit[0])
            usage = body.get("usage", {})
            ch = (body.get("choices") or [{}])[0]
            return {
                "key": key, "cached": True, "content": hit[1],
                "reasoning": (ch.get("message") or {}).get("reasoning") or "",
                "finish_reason": ch.get("finish_reason") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": 0,
            }
        delay = 3.0
        last_err = None
        deadline = time.monotonic() + 1200
        for _ in range(max_retries):
            if time.monotonic() > deadline:
                raise RuntimeError(f"call deadline (20 min) exceeded; last error: {last_err}")
            t0 = time.monotonic()
            try:
                r = self.session.post(API_URL, data=json.dumps(payload), timeout=300)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(delay, 90)); delay = min(delay * 1.6, 90)
                continue
            if r.status_code == 200:
                body = r.json()
                ch = (body.get("choices") or [{}])[0]
                msg = ch.get("message", {}) or {}
                usage = body.get("usage", {})
                lat = int((time.monotonic() - t0) * 1000)
                content = msg.get("content") or ""
                self.ledger.put(key, body.get("model", self.model),
                                json.dumps(payload, sort_keys=True), json.dumps(body),
                                content, usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0), lat)
                return {
                    "key": key, "cached": False, "content": content,
                    "reasoning": msg.get("reasoning") or "",
                    "finish_reason": ch.get("finish_reason") or "",
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "latency_ms": lat,
                }
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(min(delay, 90)); delay = min(delay * 1.6, 90)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"max retries exceeded; last error: {last_err}")


def load_client(root: Path) -> OpenRouterClient:
    return OpenRouterClient(root / "artifacts" / "ledger.sqlite")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    return t


def chat_json(
    client: OpenRouterClient,
    messages: list[dict],
    schema: dict,
    max_tokens: int = 4000,
    validation_retries: int = 3,
) -> dict:
    """Chat with strict-JSON enforcement: response_format first, then schema-in-prompt
    fallback with local jsonschema validation. ox-alpha ignores response_format, so
    expect the fallback path to carry production traffic."""
    import jsonschema

    def validate(content: str) -> dict:
        obj = json.loads(_strip_fences(content))
        jsonschema.validate(obj, schema.get("json_schema", {}).get("schema", schema))
        return obj

    r = client.chat(messages, max_tokens=max_tokens,
                    response_format=schema)
    try:
        return validate(r["content"])
    except (json.JSONDecodeError, jsonschema.ValidationError):
        pass

    schema_text = json.dumps(schema.get("json_schema", {}).get("schema", schema), indent=2)
    augmented = list(messages) + [
        {"role": "assistant", "content": r["content"]},
        {
            "role": "user",
            "content": (
                "Your previous reply was not machine-parseable. Respond again with "
                "ONLY a single valid JSON object — no prose, no markdown, no code "
                "fences — conforming exactly to this JSON Schema:\n" + schema_text
            ),
        },
    ]
    for _ in range(validation_retries):
        r = client.chat(augmented, max_tokens=max_tokens)
        try:
            return validate(r["content"])
        except (json.JSONDecodeError, jsonschema.ValidationError):
            augmented.append({"role": "assistant", "content": r["content"]})
            augmented.append({
                "role": "user",
                "content": ("Still invalid. Output ONLY the raw JSON object matching the "
                            "schema. No other text."),
            })
    raise RuntimeError("chat_json: model failed to produce schema-valid JSON after retries")
