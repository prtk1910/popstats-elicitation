"""Pipeline entry points for popstats-elicitation."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .pums import download, extract_to_parquet

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "artifacts"


def _each_kind(args, fn):
    kinds = ("h", "p") if args.file == "both" else (args.file,)
    for kind in kinds:
        fn(args.state, args.year, kind)


def fetch_one(state: str, year: int, kind: str) -> None:
    zp = download(DATA_ROOT, state, year, kind)
    print(
        f"[{state}/{kind}] downloaded -> "
        f"{zp.name} ({zp.stat().st_size / 1e6:.1f} MB)"
    )


def build_one(state: str, year: int, kind: str) -> None:
    pq = extract_to_parquet(DATA_ROOT, state, year, kind)
    print(
        f"[{state}/{kind}] parquet -> "
        f"{pq} ({pq.stat().st_size / 1e6:.1f} MB)"
    )


def cmd_fetch(args) -> None:
    _each_kind(args, fetch_one)


def cmd_build(args) -> None:
    _each_kind(args, build_one)


def cmd_gold(args) -> None:
    from .grid import build_all_gold

    path = build_all_gold(
        ROOT,
        year=args.year,
        force=args.force,
    )
    print(f"gold complete -> {path}")


def cmd_smoke(_args) -> None:
    from popstats.gold import weighted_quantile

    assert (
        weighted_quantile(
            [1.0, 2.0],
            [1.0, 1.0],
            0.5,
        )
        == 1.0
    )

    pq_dir = DATA_ROOT / "parquet"

    have = (
        sorted(
            str(p.relative_to(DATA_ROOT))
            for p in pq_dir.glob("*/*.parquet")
        )
        if pq_dir.exists()
        else []
    )

    print(
        "smoke OK: weighted quantiles functional; "
        f"extracts present: {have or 'none'}"
    )


def cmd_run(args) -> None:
    from .elicit import run_elicitation

    run_elicitation(
        DATA_ROOT.parent,
        arms=tuple(args.arms.split(",")),
        reps=args.reps,
        interval_reps=args.interval_reps,
        workers=args.workers,
    )


def cmd_paper(_args) -> None:
    from .analysis import write_paper
    from .paperio import md_to_docx

    md = write_paper(
        ROOT,
        config_note="stealth/ox-alpha via OpenRouter",
    )
    docx = md_to_docx(
        md,
        ROOT / "PAPER.docx",
    )

    print(f"paper written: {md} and {docx}")


def cmd_clean(_args) -> None:
    for sub in ("raw", "work"):
        p = DATA_ROOT / sub

        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="popstats")
    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
    )

    def add_year(parser):
        parser.add_argument(
            "--year",
            type=int,
            default=2023,
        )

    p = sub.add_parser("fetch")
    p.add_argument(
        "--state",
        choices=["ca", "ny"],
        required=True,
    )
    p.add_argument(
        "--file",
        choices=["h", "p", "both"],
        default="both",
    )
    add_year(p)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("build")
    p.add_argument(
        "--state",
        choices=["ca", "ny"],
        required=True,
    )
    p.add_argument(
        "--file",
        choices=["h", "p", "both"],
        default="both",
    )
    add_year(p)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("gold")
    add_year(p)
    p.add_argument(
        "--force",
        action="store_true",
        help="recompute gold even when compatible cached gold exists",
    )
    p.set_defaults(fn=cmd_gold)

    sub.add_parser("smoke").set_defaults(
        fn=cmd_smoke
    )

    p = sub.add_parser("run")
    p.add_argument(
        "--arms",
        default="quantiles,interval",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=3,
    )
    p.add_argument(
        "--interval-reps",
        type=int,
        default=2,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    p.set_defaults(fn=cmd_run)

    sub.add_parser("paper").set_defaults(
        fn=cmd_paper
    )

    sub.add_parser("clean").set_defaults(
        fn=cmd_clean
    )

    args = ap.parse_args(argv)

    try:
        args.fn(args)
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
