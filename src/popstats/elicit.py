"""Elicitation arms and concurrent runner for population-distribution questions."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api import load_client, chat_json
from .grid import OUTCOMES, STATE_NAMES

QUANT_SCHEMA = {"type": "json_schema", "json_schema": {"name": "quantiles", "strict": True,
    "schema": {
        "type": "object",
        "properties": {"p10": {"type": "number"},
                       "p50": {"type": "number"},
                       "p90": {"type": "number"}},
        "required": ["p10", "p50", "p90"],
        "additionalProperties": False,
    }}}

INTERVAL_SCHEMA = {"type": "json_schema", "json_schema": {"name": "interval", "strict": True,
    "schema": {
        "type": "object",
        "properties": {"median": {"type": "number"},
                       "lo90": {"type": "number"},
                       "hi90": {"type": "number"}},
        "required": ["median", "lo90", "hi90"],
        "additionalProperties": False,
    }}}

Q_TEMPLATES = [
    ("What is the distribution of {desc} ({unit}) among {pop}? Report your best estimate "
     "of the 10th percentile, the median (50th percentile), and the 90th percentile."),
    ("Estimate how {desc} ({unit}) is distributed among {pop}. Give the 10th, 50th, and "
     "90th percentile values of that distribution."),
    ("For {pop}, provide the 10th, 50th, and 90th percentiles of {desc} ({unit})."),
]

I_TEMPLATE = ("Consider one unit drawn at random from {pop}, and its {desc} ({unit}). "
              "Give the median of that value across the population, and a central 90% "
              "prediction interval: the value that only 5% of draws fall below, and the "
              "value that only 5% exceed.")


def build_quantile_messages(cell_pop: str, outcome_desc: str, unit: str,
                            variant: int) -> list[dict]:
    tmpl = Q_TEMPLATES[variant % len(Q_TEMPLATES)]
    q = tmpl.format(desc=outcome_desc, unit=unit, pop=cell_pop)
    return [{"role": "user", "content":
             q + " Respond with ONLY a JSON object {\"p10\": number, \"p50\": number, "
                 "\"p90\": number} in the same units. No other text."}]


def build_interval_messages(cell_pop: str, outcome_desc: str, unit: str) -> list[dict]:
    q = I_TEMPLATE.format(pop=cell_pop, desc=outcome_desc, unit=unit)
    return [{"role": "user", "content":
             q + " Respond with ONLY a JSON object {\"median\": number, \"lo90\": number, "
                 "\"hi90\": number} in the same units. No other text."}]


def run_elicitation(root: Path, arms=("quantiles", "interval"), reps: int = 3,
                    interval_reps: int = 2, workers: int = 8) -> Path:
    root = root.resolve()
    client = load_client(root)
    gold = json.loads((root / "artifacts" / "gold" / "cells.json").read_text())
    out_path = root / "artifacts" / "results" / "elicitation.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add(r["key"])
                except Exception:
                    pass

    tasks = []
    for key, g in sorted(gold.items()):
        oc = OUTCOMES[g["outcome"]]
        state = g["state"]
        if "quantiles" in arms:
            for rep in range(reps):
                k = f"{key}:quantiles:{rep}"
                if k not in done:
                    tasks.append({
                        "key": k, "cell_key": key, "arm": "quantiles",
                        "messages": build_quantile_messages(
                            g["population_nl"], oc.desc, oc.unit, rep),
                        "schema": QUANT_SCHEMA,
                    })
        if "interval" in arms:
            for rep in range(interval_reps):
                k = f"{key}:interval:{rep}"
                if k not in done:
                    tasks.append({
                        "key": k, "cell_key": key, "arm": "interval",
                        "messages": build_interval_messages(
                            g["population_nl"], oc.desc, oc.unit),
                        "schema": INTERVAL_SCHEMA,
                    })
    print(f"queued {len(tasks)} elicitation calls")

    lock = threading.Lock()
    count = {"n": 0}

    def work(t):
        t0 = time.time()
        try:
            obj = chat_json(client, t["messages"], schema=t["schema"], max_tokens=6000)
            rec = {"key": t["key"], "cell_key": t["cell_key"], "arm": t["arm"],
                   "response": obj, "error": "",
                   "latency_s": round(time.time() - t0, 1)}
        except Exception as e:
            rec = {"key": t["key"], "cell_key": t["cell_key"], "arm": t["arm"],
                   "response": None, "error": str(e)[:200],
                   "latency_s": round(time.time() - t0, 1)}
        with lock:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count["n"] += 1
            if count["n"] % 20 == 0:
                print(f"progress {count['n']}", flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for _ in as_completed(futures):
            pass
    print(f"elicitation complete: {count['n']} records in {time.time()-t0:.0f}s")
    return out_path
