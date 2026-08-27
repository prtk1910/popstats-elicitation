"""Direct Meta Model API client for Muse Spark replication."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import requests


API_URL = "https://api.meta.ai/v1/chat/completions"


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()

        with self.lock:
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    key TEXT PRIMARY KEY,
                    model TEXT,
                    request_json TEXT,
                    response_json TEXT,
                    content TEXT,
                    prompt_tokens INT,
                    completion_tokens INT,
                    latency_ms INT,
                    ts TEXT
                )
                """
            )
            self.con.commit()

    def get(self, key: str):
        with self.lock:
            row = self.con.execute(
                "SELECT response_json, content FROM calls WHERE key = ?",
                (key,),
            ).fetchone()

        return (row[0], row[1]) if row else None

    def put(
        self,
        key,
        model,
        request_json,
        response_json,
        content,
        prompt_tokens,
        completion_tokens,
        latency_ms,
    ):
        with self.lock:
            self.con.execute(
                """
                INSERT OR REPLACE INTO calls
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    key,
                    model,
                    request_json,
                    response_json,
                    content,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                ),
            )
            self.con.commit()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    return ""


class MuseClient:
    def __init__(
        self,
        ledger_path: Path,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.key = os.environ.get("MODEL_API_KEY", "")

        if not self.key or "replace-with" in self.key:
            raise RuntimeError(
                "MODEL_API_KEY missing. Add the Meta Model API key to your "
                "environment/.env; do not place it in source control."
            )

        self.model = model or os.environ.get(
            "MUSE_MODEL",
            "muse-spark-1.2",
        )

        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.environ.get("MUSE_REASONING_EFFORT", "").strip() or None
        )

        self.ledger = Ledger(ledger_path)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            }
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = 12,
    ):
        payload: dict = {
            "model": self.model,
            "messages": messages,
        }

        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens

        if response_format is not None:
            payload["response_format"] = response_format

        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        key_material = {
            "endpoint": API_URL,
            "payload": payload,
        }

        key = hashlib.sha256(
            json.dumps(
                key_material,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        hit = self.ledger.get(key)

        if hit is not None:
            body = json.loads(hit[0])
            usage = body.get("usage", {})
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message", {}) or {}

            return {
                "key": key,
                "cached": True,
                "content": hit[1],
                "reasoning": (
                    message.get("reasoning")
                    or message.get("reasoning_content")
                    or ""
                ),
                "finish_reason": choice.get("finish_reason") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": 0,
            }

        delay = 3.0
        last_err = None
        deadline = time.monotonic() + 1200

        for _ in range(max_retries):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"call deadline exceeded; last error: {last_err}"
                )

            t0 = time.monotonic()

            try:
                response = self.session.post(
                    API_URL,
                    data=json.dumps(payload),
                    timeout=300,
                )
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(min(delay, 90))
                delay = min(delay * 1.6, 90)
                continue

            if response.status_code == 200:
                body = response.json()
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message", {}) or {}
                usage = body.get("usage", {})

                latency_ms = int(
                    (time.monotonic() - t0) * 1000
                )

                content = _content_text(
                    message.get("content")
                )

                self.ledger.put(
                    key,
                    body.get("model", self.model),
                    json.dumps(
                        payload,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        body,
                        ensure_ascii=False,
                    ),
                    content,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    latency_ms,
                )

                return {
                    "key": key,
                    "cached": False,
                    "content": content,
                    "reasoning": (
                        message.get("reasoning")
                        or message.get("reasoning_content")
                        or ""
                    ),
                    "finish_reason": (
                        choice.get("finish_reason") or ""
                    ),
                    "prompt_tokens": usage.get(
                        "prompt_tokens",
                        0,
                    ),
                    "completion_tokens": usage.get(
                        "completion_tokens",
                        0,
                    ),
                    "latency_ms": latency_ms,
                }

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get(
                    "retry-after"
                )

                if retry_after:
                    try:
                        delay = max(
                            delay,
                            float(retry_after),
                        )
                    except ValueError:
                        pass

                last_err = RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

                time.sleep(min(delay, 90))
                delay = min(delay * 1.6, 90)
                continue

            raise RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        raise RuntimeError(
            f"max retries exceeded; last error: {last_err}"
        )


def _strip_fences(text: str) -> str:
    value = text.strip()

    if value.startswith("```"):
        lines = value.splitlines()
        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]
        value = "\n".join(lines).strip()

    return value


def chat_json(
    client: MuseClient,
    messages: list[dict],
    schema: dict,
    max_tokens: int = 6000,
    validation_retries: int = 3,
) -> dict:
    import jsonschema

    def validate(content: str) -> dict:
        obj = json.loads(_strip_fences(content))

        jsonschema.validate(
            obj,
            schema.get(
                "json_schema",
                {},
            ).get(
                "schema",
                schema,
            ),
        )

        return obj

    response = client.chat(
        messages,
        max_tokens=max_tokens,
        response_format=schema,
    )

    try:
        return validate(response["content"])
    except (
        json.JSONDecodeError,
        jsonschema.ValidationError,
    ):
        pass

    schema_text = json.dumps(
        schema.get(
            "json_schema",
            {},
        ).get(
            "schema",
            schema,
        ),
        indent=2,
    )

    augmented = list(messages) + [
        {
            "role": "assistant",
            "content": response["content"],
        },
        {
            "role": "user",
            "content": (
                "Your previous reply was not machine-parseable. "
                "Respond again with ONLY a single valid JSON object "
                "— no prose, no markdown, no code fences — "
                "conforming exactly to this JSON Schema:\n"
                + schema_text
            ),
        },
    ]

    for _ in range(validation_retries):
        response = client.chat(
            augmented,
            max_tokens=max_tokens,
        )

        try:
            return validate(response["content"])
        except (
            json.JSONDecodeError,
            jsonschema.ValidationError,
        ):
            augmented.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                }
            )

            augmented.append(
                {
                    "role": "user",
                    "content": (
                        "Still invalid. Output ONLY the raw JSON "
                        "object matching the schema. No other text."
                    ),
                }
            )

    raise RuntimeError(
        "Muse failed to produce schema-valid JSON after retries"
    )
