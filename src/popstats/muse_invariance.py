from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .grid import OUTCOMES
from .muse_api import MuseClient, chat_json


ROOT = Path(__file__).resolve().parents[2]


NORMAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "population_quantiles",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "p10": {"type": "number"},
                "p50": {"type": "number"},
                "p90": {"type": "number"},
            },
            "required": ["p10", "p50", "p90"],
            "additionalProperties": False,
        },
    },
}


REVERSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "population_quantiles_reverse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "p90": {"type": "number"},
                "p50": {"type": "number"},
                "p10": {"type": "number"},
            },
            "required": ["p90", "p50", "p10"],
            "additionalProperties": False,
        },
    },
}


PILOT_CELLS = [
    "ca:income:marginal",
    "ca:income:size24+renter",
    "ca:wages:female+bach",
    "ca:commute:age25_34+female",
    "ny:income:marginal",
    "ny:wages:female+bach",
    "ny:wages:age25_34+female+bach",
    "ny:commute:marginal",
]


FULL_CELLS = [
    "ca:income:marginal",
    "ca:income:size1",
    "ca:income:size24+renter",
    "ca:wages:marginal",
    "ca:wages:age25_34",
    "ca:wages:female+bach",
    "ca:wages:age25_34+female+bach",
    "ca:commute:marginal",
    "ca:commute:female",
    "ca:commute:age25_34+female",
    "ny:income:marginal",
    "ny:income:size1",
    "ny:income:size24+renter",
    "ny:wages:marginal",
    "ny:wages:age25_34",
    "ny:wages:female+bach",
    "ny:wages:age25_34+female+bach",
    "ny:commute:marginal",
    "ny:commute:female",
    "ny:commute:age25_34+female",
]


def make_task(cell_key, g, outcome, arm):
    pop = g["population_nl"]
    desc = outcome.desc
    unit = outcome.unit

    if arm == "reverse_order":
        content = (
            f"For {pop}, estimate the 90th percentile, median, and "
            f"10th percentile of {desc} ({unit}), in that order. "
            'Respond with ONLY a JSON object '
            '{"p90": number, "p50": number, "p10": number}.'
        )

        schema = REVERSE_SCHEMA
        scale = 1.0

    elif arm == "verbal_definition":
        content = (
            f"For {pop}, estimate these three points of the population "
            f"distribution of {desc} ({unit}): "
            "the value at or below which 10% of observations fall; "
            "the value at or below which 50% fall; "
            "and the value at or below which 90% fall. "
            'Respond with ONLY a JSON object '
            '{"p10": number, "p50": number, "p90": number}.'
        )

        schema = NORMAL_SCHEMA
        scale = 1.0

    elif arm == "scaled_units":
        if g["outcome"] in {"income", "wages"}:
            scaled_unit = "thousands of US dollars per year"
            scale = 1000.0
        else:
            scaled_unit = "hours"
            scale = 60.0

        content = (
            f"For {pop}, estimate the 10th percentile, median, and "
            f"90th percentile of {desc}, expressed in {scaled_unit}. "
            'Respond with ONLY a JSON object '
            '{"p10": number, "p50": number, "p90": number}.'
        )

        schema = NORMAL_SCHEMA

    else:
        raise ValueError(arm)

    return {
        "key": f"{cell_key}:invariance:{arm}",
        "cell_key": cell_key,
        "arm": arm,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "schema": schema,
        "scale_to_original": scale,
    }


def build_tasks(root, pilot):
    gold = json.loads(
        (root / "artifacts/gold/cells.json").read_text()
    )

    cells = PILOT_CELLS if pilot else FULL_CELLS

    tasks = []

    for cell_key in cells:
        g = gold[cell_key]
        outcome = OUTCOMES[g["outcome"]]

        for arm in (
            "reverse_order",
            "verbal_definition",
            "scaled_units",
        ):
            tasks.append(
                make_task(
                    cell_key,
                    g,
                    outcome,
                    arm,
                )
            )

    return tasks


def read_done(path):
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


def run(root, pilot, workers, dry_run):
    model = os.environ.get(
        "MUSE_MODEL",
        "muse-spark-1.2",
    )

    out_path = (
        root
        / "artifacts/results"
        / "muse-spark-1.2-invariance.jsonl"
    )

    ledger_path = (
        root
        / "artifacts/ledgers"
        / "muse-spark-1.2-invariance.sqlite"
    )

    all_tasks = build_tasks(
        root,
        pilot,
    )

    done = read_done(
        out_path
    )

    tasks = [
        t for t in all_tasks
        if t["key"] not in done
    ]

    print("model:", model)
    print("pilot:", pilot)
    print("cells:", len(all_tasks) // 3)
    print("total intended tasks:", len(all_tasks))
    print("already complete:", len(done))
    print("queued:", len(tasks))
    print("results:", out_path)

    if dry_run:
        return

    client = MuseClient(
        ledger_path=ledger_path,
        model=model,
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = threading.Lock()
    counter = {"n": 0}

    def work(task):
        started = time.time()

        try:
            obj = chat_json(
                client,
                task["messages"],
                schema=task["schema"],
                max_tokens=6000,
            )

            scale = task[
                "scale_to_original"
            ]

            converted = {
                "p10": float(
                    obj["p10"]
                ) * scale,
                "p50": float(
                    obj["p50"]
                ) * scale,
                "p90": float(
                    obj["p90"]
                ) * scale,
            }

            error = ""

            if not (
                converted["p10"]
                <= converted["p50"]
                <= converted["p90"]
            ):
                error = (
                    "unordered quantiles: "
                    + json.dumps(
                        converted
                    )
                )

            record = {
                "key": task["key"],
                "cell_key": task["cell_key"],
                "arm": task["arm"],
                "provider": "meta_model_api",
                "model": model,
                "messages": task["messages"],
                "raw_response": obj,
                "response_original_units": converted,
                "scale_to_original": scale,
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
                "arm": task["arm"],
                "provider": "meta_model_api",
                "model": model,
                "messages": task["messages"],
                "raw_response": None,
                "response_original_units": None,
                "scale_to_original": task[
                    "scale_to_original"
                ],
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

            counter["n"] += 1

            print(
                f"[{counter['n']}/{len(tasks)}] "
                f"{task['key']} "
                f"{'OK' if not record['error'] else 'ERROR'}",
                flush=True,
            )

    with ThreadPoolExecutor(
        max_workers=workers,
    ) as executor:
        futures = [
            executor.submit(
                work,
                task,
            )
            for task in tasks
        ]

        for _ in as_completed(
            futures
        ):
            pass

    print(
        "invariance batch complete:",
        counter["n"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot",
        action="store_true",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args(
        argv
    )

    run(
        ROOT,
        pilot=args.pilot,
        workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
