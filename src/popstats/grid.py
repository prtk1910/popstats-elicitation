"""Conditioning-depth grid definition and gold computation against ACS PUMS."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb

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
    "income": Outcome("income", "h", "HINCP", "WGTP", "US dollars per year",
                      "total annual household income", "HINCP > 0"),
    "wages": Outcome("wages", "p", "WAGP", "PWGTP", "US dollars per year",
                     "annual wage and salary earnings of the individual",
                     "WAGP >= 1000"),
    "commute": Outcome("commute", "p", "JWMNP", "PWGTP", "minutes",
                       "one-way travel time to work",
                       "JWMNP > 0"),
}

PERSON_BASE = "(ESR IN (1, 2, 4) OR ESR IS NULL)"

DIMS = {
    "income": [
        Dim("size1", "NP = 1", "single-person households"),
        Dim("size24", "NP BETWEEN 2 AND 4", "households of two to four people"),
        Dim("renter", "TEN IN (2, 3)", "renter households"),
    ],
    "wages": [
        Dim("age25_34", "AGEP BETWEEN 25 AND 34", "workers aged 25 to 34"),
        Dim("age55_64", "AGEP BETWEEN 55 AND 64", "workers aged 55 to 64"),
        Dim("female", "SEX = 2", "women"),
        Dim("bach", "CAST(SCHL AS INTEGER) >= 21", "holders of a bachelor's degree or higher"),
    ],
    "commute": [
        Dim("age25_34", "AGEP BETWEEN 25 AND 34", "workers aged 25 to 34"),
        Dim("age55_64", "AGEP BETWEEN 55 AND 64", "workers aged 55 to 64"),
        Dim("female", "SEX = 2", "women"),
    ],
}

DEPTH_COMBOS = {
    "income": [[], ["size1"], ["size24"], ["renter"], ["size24", "renter"]],
    "wages": [[], ["age25_34"], ["age55_64"], ["female"], ["bach"],
              ["female", "bach"], ["age25_34", "female"], ["age55_64", "bach"],
              ["age25_34", "female", "bach"]],
    "commute": [[], ["age25_34"], ["age55_64"], ["female"],
                ["age25_34", "female"], ["female", "age55_64"]],
}


def build_cells(state: str) -> list[Cell]:
    out = []
    for okey, combos in DEPTH_COMBOS.items():
        oc = OUTCOMES[okey]
        dim_map = {d.key: d for d in DIMS[okey]}
        for combo in combos:
            parts = [oc.base_filter if oc.table == "p" else oc.base_filter]
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
            filt = " AND ".join(f"({p})" for p in parts)
            out.append(Cell(okey, tuple(combo), filt, pop))
    return out


def compute_cell_gold(
    parquet_path: str,
    cell: Cell,
    oc: Outcome,
    n_bootstrap: int = 200,
    seed: int = 13,
) -> dict:
    from .gold import compute_gold

    g = compute_gold(parquet_path, oc.expr, oc.weight,
                     filter_sql=cell.filter_sql, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "outcome": oc.key, "dims": list(cell.dims),
        "filter_sql": cell.filter_sql, "population_nl": cell.population_nl,
        "n": g.n,
        "gold": {"q10": g.q10, "q50": g.q50, "q90": g.q90},
        "ci": g.ci,
    }


def build_all_gold(root: Path) -> Path:
    """Compute gold for all cells in all states; cached to artifacts/gold/cells.json."""
    out_path = root / "artifacts" / "gold" / "cells.json"
    done = {}
    if out_path.exists():
        done = json.loads(out_path.read_text())
    results = {}
    for state in ("ca", "ny"):
        for okey, oc in OUTCOMES.items():
            pq = root / "artifacts" / "parquet" / state / f"2023_{oc.table}.parquet"
            if not pq.exists():
                continue
            for cell in build_cells(state):
                if cell.outcome != okey:
                    continue
                key = f"{state}:{okey}:{'+'.join(cell.dims) or 'marginal'}"
                if key in done:
                    results[key] = done[key]
                    continue
                g = compute_cell_gold(str(pq), cell, oc)
                g["state"] = state
                results[key] = g
                print(f"{key}: n={g['n']:,} q50={g['gold']['q50']:,.0f} {oc.unit}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    print(f"gold written: {len(results)} cells -> {out_path}")
    return out_path
