"""Epistemic uncertainty experiment for population medians."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .grid import OUTCOMES
from .muse_api import MuseClient, chat_json


ROOT = Path(__file__).resolve().parents[2]


EPISTEMIC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "population_median_uncertainty",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "median": {"type": "number"},
                "lo90": {"type": "number"},
                "hi90": {"type": "number"},
            },
            "required": ["median", "lo90", "hi90"],
            "additionalProperties": False,
        },
    },
}


EPISTEMIC_TEMPLATES = [
    (
        "For {pop}, estimate the population median of {desc} ({unit}). "
        "Then give a 90% uncertainty interval representing your uncertainty "
        "about the value of that population median itself. "
        "This is NOT a prediction interval for a randomly selected "
        "person or household."
    ),
    (
        "Estimate the median {desc} ({unit}) for {pop}. "
        "Also report a 90% interval expressing your uncertainty about "
        "what the true population median is. The interval should describe "
        "uncertainty in the median estimate, not variation among individual "
        "people or households."
    ),
    (
        "What is your best estimate of the population median of {desc} "
        "({unit}) among {pop}? Provide lower and upper bounds for a 90% "
        "uncertainty interval for the population median itself. "
        "Do not give a range for individual observations."
    ),
]


PILOT_CELLS = [
    "ca:income:marginal",
    "ca:income:size24+renter",
    "ca:wages:marginal",
    "ca:wages:female+bach",
    "ca:commute:marginal",
    "ca:commute:age25_34+female",
    "ny:income:marginal",
    "ny:income:size24+renter",
    "ny:wages:marginal",
    "ny:wages:female+bach",
    "ny:commute:marginal",
    "ny:commute:age25_34+female",
]


def build_messages(
    population: str,
    outcome_desc: str,
    unit: str,
    variant: int,
) -> list[dict]:
    template = EPISTEMIC_TEMPLATES[
        variant % len(EPISTEMIC_TEMPLATES)
    ]

    question = template.format(
        pop=population,
        desc=outcome_desc,
        unit=unit,
    )

    return [
        {
            "role": "user",
            "content": (
                question
                + ' Respond with ONLY a JSON object '
                '{"median": number, "lo90": number, "hi90": number} '
                "in the same units. No other text."
            ),
        }
    ]


def build_tasks(
    root: Path,
    pilot: bool,
    variants: int,
) -> list[dict]:
    gold = json.loads(
        (root / "artifacts/gold/cells.json").read_text()
    )

    cell_keys = (
        PILOT_CELLS
        if pilot
        else sorted(gold)
    )

    tasks = []

    for cell_key in cell_keys:
        g = gold[cell_key]
        outcome = OUTCOMES[g["outcome"]]

        for variant in range(variants):
            tasks.append(
                {
                    "key": (
                        f"{cell_key}:epistemic:{variant}"
                    ),
                    "cell_key": cell_key,
                    "arm": "epistemic",
                    "prompt_variant": variant,
                    "messages": build_messages(
                        g["population_nl"],
                        outcome.desc,
                        outcome.unit,
                        variant,
                    ),
                    "schema": EPISTEMIC_SCHEMA,
                }
            )

    return tasks


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done = set()

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            if not row.get("error"):
                done.add(row["key"])
        except Exception:
            pass

    return done


def run(
    root: Path,
    workers: int,
    pilot: bool,
    variants: int,
    limit: int | None,
    dry_run: bool,
) -> Path:
    model = os.environ.get(
        "MUSE_MODEL",
        "muse-spark-1.2",
    )

    out_path = (
        root
        / "artifacts/results"
        / "muse-spark-1.2-epistemic.jsonl"
    )

    ledger_path = (
        root
        / "artifacts/ledgers"
        / "muse-spark-1.2-epistemic.sqlite"
    )

    all_tasks = build_tasks(
        root=root,
        pilot=pilot,
        variants=variants,
    )

    done = read_done(out_path)

    tasks = [
        t for t in all_tasks
        if t["key"] not in done
    ]

    if limit is not None:
        tasks = tasks[:limit]

    print("model:", model)
    print("arm: epistemic population-median uncertainty")
    print("pilot:", pilot)
    print("variants per cell:", variants)
    print("total intended tasks:", len(all_tasks))
    print("already complete:", len(done))
    print("queued:", len(tasks))
    print("results:", out_path)

    if dry_run:
        return out_path

    client = MuseClient(
        ledger_path=ledger_path,
        model=model,
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = threading.Lock()
    count = {"n": 0}

    def work(task):
        started = time.time()

        try:
            obj = chat_json(
                client,
                task["messages"],
                schema=task["schema"],
                max_tokens=6000,
            )

            error = ""

            if not (
                obj["lo90"]
                <= obj["median"]
                <= obj["hi90"]
            ):
                error = (
                    "unordered epistemic interval: "
                    + json.dumps(obj)
                )

            record = {
                "key": task["key"],
                "cell_key": task["cell_key"],
                "arm": "epistemic",
                "prompt_variant": task["prompt_variant"],
                "provider": "meta_model_api",
                "model": model,
                "messages": task["messages"],
                "response": obj,
                "error": error,
                "latency_s": round(
                    time.time() - started,
                    3,
                ),
            }

        except Exception as exc:
            record = {
                "key": task["key"],
                "cell_key": task["cell_key"],
                "arm": "epistemic",
                "prompt_variant": task["prompt_variant"],
                "provider": "meta_model_api",
                "model": model,
                "messages": task["messages"],
                "response": None,
                "error": str(exc)[:1000],
                "latency_s": round(
                    time.time() - started,
                    3,
                ),
            }

        with lock:
            with out_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            count["n"] += 1

            print(
                f"[{count['n']}/{len(tasks)}] "
                f"{task['key']} "
                f"{'OK' if not record['error'] else 'ERROR'}",
                flush=True,
            )

        return record

    with ThreadPoolExecutor(
        max_workers=workers,
    ) as executor:
        futures = [
            executor.submit(work, t)
            for t in tasks
        ]

        for _ in as_completed(futures):
            pass

    print(
        "epistemic batch complete:",
        count["n"],
    )

    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
    )

    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        choices=[1, 2, 3],
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args(argv)

    run(
        root=ROOT,
        workers=args.workers,
        pilot=args.pilot,
        variants=args.variants,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
