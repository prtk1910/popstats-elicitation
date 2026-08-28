"""Year-explicit Muse experiment for temporal population knowledge."""

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
YEARS = (2019, 2021, 2023)


QUANTILE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "temporal_population_quantiles",
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


def build_tasks(root: Path) -> list[dict]:
    tasks = []

    for year in YEARS:
        path = (
            root
            / "artifacts/gold/temporal"
            / f"cells-{year}.json"
        )

        cells = json.loads(
            path.read_text()
        )

        for cell_key, cell in sorted(cells.items()):
            outcome = OUTCOMES[cell["outcome"]]

            if cell["outcome"] in {"income", "wages"}:
                unit_text = (
                    f"{year} US dollars per year"
                )
                clarification = (
                    f"Express monetary values in {year} dollars, "
                    "not dollars adjusted to another year. "
                )
            else:
                unit_text = outcome.unit
                clarification = ""

            content = (
                f"In {year}, for {cell['population_nl']}, "
                f"estimate the 10th percentile, median, and "
                f"90th percentile of {outcome.desc} "
                f"({unit_text}). "
                f"{clarification}"
                "Respond with ONLY a JSON object "
                '{"p10": number, "p50": number, "p90": number}.'
            )

            tasks.append({
                "key": f"{cell_key}:temporal:{year}",
                "cell_key": cell_key,
                "year": year,
                "outcome": cell["outcome"],
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "schema": QUANTILE_SCHEMA,
            })

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
    *,
    root: Path,
    workers: int,
    dry_run: bool,
):
    model = os.environ.get(
        "MUSE_MODEL",
        "muse-spark-1.2",
    )

    out_path = (
        root
        / "artifacts/results"
        / "muse-spark-1.2-temporal.jsonl"
    )

    ledger_path = (
        root
        / "artifacts/ledgers"
        / "muse-spark-1.2-temporal.sqlite"
    )

    all_tasks = build_tasks(root)

    done = read_done(out_path)

    tasks = [
        t for t in all_tasks
        if t["key"] not in done
    ]

    print("model:", model)
    print("years:", YEARS)
    print("cells per year:", 40)
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

            error = ""

            if not (
                float(obj["p10"])
                <= float(obj["p50"])
                <= float(obj["p90"])
            ):
                error = (
                    "unordered quantiles: "
                    + json.dumps(obj)
                )

            record = {
                "key": task["key"],
                "cell_key": task["cell_key"],
                "year": task["year"],
                "outcome": task["outcome"],
                "arm": "temporal",
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
                "year": task["year"],
                "outcome": task["outcome"],
                "arm": "temporal",
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
            ) as f:
                f.write(
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
            executor.submit(work, task)
            for task in tasks
        ]

        for _ in as_completed(futures):
            pass

    print(
        "temporal batch complete:",
        counter["n"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args(argv)

    run(
        root=ROOT,
        workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
