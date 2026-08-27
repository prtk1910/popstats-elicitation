"""Conditioning-depth grid definition and gold computation against ACS PUMS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

STATE_NAMES = {"ca": "California", "ny": "New York State"}


@dataclass(frozen=True)
class Outcome:
    key: str
    table: str          # 'h' or 'p'
    expr: str
    weight: str
    unit: str
    desc: str           # plain-language description for elicitation
    base_filter: str


@dataclass(frozen=True)
class Dim:
    key: str
    sql: str
    nl: str


@dataclass(frozen=True)
class Cell:
    outcome: str
    dims: tuple[str, ...]
    filter_sql: str
    population_nl: str


OUTCOMES = {
    "income": Outcome(
        "income",
        "h",
        "HINCP",
        "WGTP",
        "US dollars per year",
        "total annual household income",
        "HINCP > 0",
    ),
    "wages": Outcome(
        "wages",
        "p",
        "WAGP",
        "PWGTP",
        "US dollars per year",
        "annual wage and salary earnings of the individual",
        "WAGP >= 1000",
    ),
    "commute": Outcome(
        "commute",
        "p",
        "JWMNP",
        "PWGTP",
        "minutes",
        "one-way travel time to work",
        "JWMNP > 0",
    ),
}

PERSON_BASE = "(ESR IN (1, 2, 4) OR ESR IS NULL)"

DIMS = {
    "income": [
        Dim("size1", "NP = 1", "single-person households"),
        Dim(
            "size24",
            "NP BETWEEN 2 AND 4",
            "households of two to four people",
        ),
        Dim("renter", "TEN IN (2, 3)", "renter households"),
    ],
    "wages": [
        Dim(
            "age25_34",
            "AGEP BETWEEN 25 AND 34",
            "workers aged 25 to 34",
        ),
        Dim(
            "age55_64",
            "AGEP BETWEEN 55 AND 64",
            "workers aged 55 to 64",
        ),
        Dim("female", "SEX = 2", "women"),
        Dim(
            "bach",
            "CAST(SCHL AS INTEGER) >= 21",
            "holders of a bachelor's degree or higher",
        ),
    ],
    "commute": [
        Dim(
            "age25_34",
            "AGEP BETWEEN 25 AND 34",
            "workers aged 25 to 34",
        ),
        Dim(
            "age55_64",
            "AGEP BETWEEN 55 AND 64",
            "workers aged 55 to 64",
        ),
        Dim("female", "SEX = 2", "women"),
    ],
}

DEPTH_COMBOS = {
    "income": [
        [],
        ["size1"],
        ["size24"],
        ["renter"],
        ["size24", "renter"],
    ],
    "wages": [
        [],
        ["age25_34"],
        ["age55_64"],
        ["female"],
        ["bach"],
        ["female", "bach"],
        ["age25_34", "female"],
        ["age55_64", "bach"],
        ["age25_34", "female", "bach"],
    ],
    "commute": [
        [],
        ["age25_34"],
        ["age55_64"],
        ["female"],
        ["age25_34", "female"],
        ["female", "age55_64"],
    ],
}


def build_cells(state: str) -> list[Cell]:
    out = []

    for okey, combos in DEPTH_COMBOS.items():
        oc = OUTCOMES[okey]
        dim_map = {d.key: d for d in DIMS[okey]}

        for combo in combos:
            parts = [oc.base_filter]
            parts += [dim_map[k].sql for k in combo]

            nl_dims = [dim_map[k].nl for k in combo]

            if okey == "income":
                base_nl = f"households in {STATE_NAMES[state]}"
            else:
                base_nl = f"people in {STATE_NAMES[state]}"

                if okey == "wages":
                    base_nl += " with wage earnings of at least $1,000"
                elif okey == "commute":
                    base_nl += " who commute to work"

            pop = base_nl

            if nl_dims:
                pop += ", specifically " + " and ".join(nl_dims)

            filt = " AND ".join(f"({part})" for part in parts)

            out.append(
                Cell(
                    okey,
                    tuple(combo),
                    filt,
                    pop,
                )
            )

    return out


def compute_cell_gold(
    parquet_path: str,
    cell: Cell,
    oc: Outcome,
    year: int = 2023,
) -> dict:
    """Compute one population cell using ACS SDR replicate weights."""
    from .gold import compute_gold

    g = compute_gold(
        parquet_path,
        oc.expr,
        oc.weight,
        filter_sql=cell.filter_sql,
    )

    return {
        "outcome": oc.key,
        "dims": list(cell.dims),
        "filter_sql": cell.filter_sql,
        "population_nl": cell.population_nl,
        "year": year,
        "n": g.n,
        "gold": {
            "q10": g.q10,
            "q50": g.q50,
            "q90": g.q90,
        },
        "ci": g.ci,
        "se": g.se,
        "moe90": g.moe90,
        "uncertainty_method": g.uncertainty_method,
    }


def gold_output_path(root: Path, year: int) -> Path:
    """Keep the historical 2023 filename while allowing later temporal gold."""
    if year == 2023:
        return root / "artifacts" / "gold" / "cells.json"

    return root / "artifacts" / "gold" / f"cells-{year}.json"


def _usable_cached_cell(
    value: dict,
    year: int,
) -> bool:
    """Old bootstrap-gold cells must not be silently reused."""
    return (
        value.get("year") == year
        and value.get("uncertainty_method")
        == "acs_sdr_replicate_weights"
        and "se" in value
        and "moe90" in value
    )


def build_all_gold(
    root: Path,
    year: int = 2023,
    force: bool = False,
) -> Path:
    """Compute gold for all cells using official ACS replicate weights.

    The original 2023 benchmark remains at artifacts/gold/cells.json.
    Additional years use cells-<year>.json for later temporal experiments.
    """
    out_path = gold_output_path(root, year)

    cached: dict[str, dict] = {}

    if out_path.exists() and not force:
        try:
            cached = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            cached = {}

    # Require the complete CA + NY benchmark rather than silently producing
    # a partial gold file.
    required = [
        root
        / "artifacts"
        / "parquet"
        / state
        / f"{year}_{table}.parquet"
        for state in ("ca", "ny")
        for table in ("h", "p")
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "required PUMS Parquet extracts are missing:\n"
            f"{formatted}\n"
            "Run make fetch and make build first."
        )

    results: dict[str, dict] = {}

    for state in ("ca", "ny"):
        for okey, oc in OUTCOMES.items():
            pq = (
                root
                / "artifacts"
                / "parquet"
                / state
                / f"{year}_{oc.table}.parquet"
            )

            for cell in build_cells(state):
                if cell.outcome != okey:
                    continue

                key = (
                    f"{state}:{okey}:"
                    f"{'+'.join(cell.dims) or 'marginal'}"
                )

                old = cached.get(key)

                if (
                    old is not None
                    and not force
                    and _usable_cached_cell(old, year)
                ):
                    results[key] = old
                    continue

                g = compute_cell_gold(
                    str(pq),
                    cell,
                    oc,
                    year=year,
                )
                g["state"] = state

                results[key] = g

                print(
                    f"{key}: "
                    f"n={g['n']:,} "
                    f"q50={g['gold']['q50']:,.0f} "
                    f"SE={g['se']['q50']:,.2f} "
                    f"{oc.unit}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))

    print(
        f"gold written: {len(results)} cells "
        f"for {year} -> {out_path}"
    )

    return out_path
