"""Exact original-prompt replication using Meta Muse Spark 1.2."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .elicit import (
    INTERVAL_SCHEMA,
    QUANT_SCHEMA,
    build_interval_messages,
    build_quantile_messages,
)
from .grid import OUTCOMES
from .muse_api import MuseClient, chat_json


ROOT = Path(__file__).resolve().parents[2]


def build_tasks(root: Path) -> list[dict]:
    gold = json.loads(
        (
            root
            / "artifacts"
            / "gold"
            / "cells.json"
        ).read_text()
    )

    tasks = []

    for cell_key, cell in sorted(gold.items()):
        outcome = OUTCOMES[cell["outcome"]]

        for variant in range(3):
            tasks.append(
                {
                    "key": (
                        f"{cell_key}:quantiles:{variant}"
                    ),
                    "cell_key": cell_key,
                    "arm": "quantiles",
                    "prompt_variant": variant,
                    "messages": build_quantile_messages(
                        cell["population_nl"],
                        outcome.desc,
                        outcome.unit,
                        variant,
                    ),
                    "schema": QUANT_SCHEMA,
                }
            )

        tasks.append(
            {
                "key": f"{cell_key}:interval:0",
                "cell_key": cell_key,
                "arm": "interval",
                "prompt_variant": 0,
                "messages": build_interval_messages(
                    cell["population_nl"],
                    outcome.desc,
                    outcome.unit,
                ),
                "schema": INTERVAL_SCHEMA,
            }
        )

    return tasks


def read_done(path: Path) -> set[str]:
    done = set()

    if not path.exists():
        return done

    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if not row.get("error"):
                    done.add(row["key"])
            except Exception:
                pass

    return done


def run(
    root: Path,
    workers: int = 8,
    limit: int | None = None,
    dry_run: bool = False,
) -> Path:
    model = os.environ.get(
        "MUSE_MODEL",
        "muse-spark-1.2",
    )

    reasoning_effort = (
        os.environ.get(
            "MUSE_REASONING_EFFORT",
            "",
        ).strip()
        or None
    )

    out_path = (
        root
        / "artifacts"
        / "results"
        / "muse-spark-1.2-original-replication.jsonl"
    )

    ledger_path = (
        root
        / "artifacts"
        / "ledgers"
        / "muse-spark-1.2.sqlite"
    )

    all_tasks = build_tasks(root)
    done = read_done(out_path)

    tasks = [
        task
        for task in all_tasks
        if task["key"] not in done
    ]

    if limit is not None:
        tasks = tasks[:limit]

    quantile_count = sum(
        task["arm"] == "quantiles"
        for task in all_tasks
    )

    interval_count = sum(
        task["arm"] == "interval"
        for task in all_tasks
    )

    print("model:", model)
    print("provider: Meta Model API")
    print(
        "reasoning effort:",
        reasoning_effort or "<omitted/model default>",
    )
    print("total unique tasks:", len(all_tasks))
    print("quantile tasks:", quantile_count)
    print("interval tasks:", interval_count)
    print("already complete:", len(done))
    print("queued:", len(tasks))
    print("results:", out_path)
    print("ledger:", ledger_path)

    if dry_run:
        return out_path

    client = MuseClient(
        ledger_path=ledger_path,
        model=model,
        reasoning_effort=reasoning_effort,
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

            record = {
                "key": task["key"],
                "cell_key": task["cell_key"],
                "arm": task["arm"],
                "prompt_variant": task[
                    "prompt_variant"
                ],
                "provider": "meta_model_api",
                "model": model,
                "reasoning_effort": (
                    reasoning_effort
                ),
                "messages": task["messages"],
                "response": obj,
                "error": "",
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
                "prompt_variant": task[
                    "prompt_variant"
                ],
                "provider": "meta_model_api",
                "model": model,
                "reasoning_effort": (
                    reasoning_effort
                ),
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

            counter["n"] += 1

            print(
                f"[{counter['n']}/{len(tasks)}] "
                f"{task['key']} "
                f"{'OK' if not record['error'] else 'ERROR'}",
                flush=True,
            )

        return record

    with ThreadPoolExecutor(
        max_workers=workers,
    ) as executor:
        futures = [
            executor.submit(work, task)
            for task in tasks
        ]

        for _ in as_completed(futures):
            pass

    print(
        "replication batch complete:",
        counter["n"],
    )

    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
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
        ROOT,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
