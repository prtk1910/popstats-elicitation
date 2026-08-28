"""ACS PUMS download, selective extraction, Parquet reduction."""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests


STATE_ABBR = {"ca": "ca", "ny": "ny"}
FIPS = {"ca": "06", "ny": "36"}
BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year"
FILE_KINDS = ("h", "p")  # housing, person

MAX_ZIP_BYTES = 500 * 1024 * 1024

# ACS supplies 80 successive-difference replicate weights for variance
# estimation. Preserve them in the reduced Parquet extracts so gold
# uncertainty can use the official survey design rather than a row bootstrap.
PERSON_REPLICATE_WEIGHTS = [
    f"PWGTP{i}" for i in range(1, 81)
]

HOUSING_REPLICATE_WEIGHTS = [
    f"WGTP{i}" for i in range(1, 81)
]


PERSON_COLUMNS = [
    "PWGTP", "ADJINC",
    *PERSON_REPLICATE_WEIGHTS,
    "AGEP",
    "SEX",
    "SCHL",
    "ESR",
    "COW",
    "JWMNP",
    "WAGP",
    "PERNP",
    "PINCP",
    "RAC1P",
    "HISP",
    "CIT",
    "PUMA",
]


HOUSING_COLUMNS = [
    "WGTP", "ADJINC",
    *HOUSING_REPLICATE_WEIGHTS,
    "HINCP",
    "TEN",
    "GRPIP",
    "VALP",
    "NP",
    "BDSP",
    "RMSP",
    "VEHCP",
    "PUMA",
]


@dataclass(frozen=True)
class PumsPaths:
    state: str
    year: int
    root: Path
    kind: str = "h"

    @property
    def zip_path(self) -> Path:
        return (
            self.root
            / "raw"
            / f"csv_{self.kind}{STATE_ABBR[self.state]}.zip"
        )

    @property
    def parquet_path(self) -> Path:
        return (
            self.root
            / "parquet"
            / self.state
            / f"{self.year}_{self.kind}.parquet"
        )


def make_paths(
    root: Path,
    state: str,
    year: int,
    kind: str,
) -> PumsPaths:
    assert kind in FILE_KINDS

    return PumsPaths(
        state=state,
        year=year,
        root=root,
        kind=kind,
    )


def zip_url(
    state: str,
    year: int,
    kind: str,
) -> str:
    return (
        f"{BASE.format(year=year)}/"
        f"csv_{kind}{STATE_ABBR[state]}.zip"
    )


def inner_csv_name(
    state: str,
    kind: str,
) -> str:
    return f"psam_{kind}{FIPS[state]}.csv"


def download(
    root: Path,
    state: str,
    year: int,
    kind: str,
    chunk: int = 1 << 20,
) -> Path:
    paths = make_paths(
        root,
        state,
        year,
        kind,
    )

    paths.zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if paths.zip_path.exists():
        return paths.zip_path

    part = paths.zip_path.with_suffix(".zip.part")

    with requests.get(
        zip_url(state, year, kind),
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        if total and total > MAX_ZIP_BYTES:
            raise RuntimeError(
                f"{state}/{kind}: "
                f"{total}B exceeds guard"
            )

        written = 0

        with open(part, "wb") as fh:
            for chunk_bytes in response.iter_content(chunk):
                if not chunk_bytes:
                    continue

                written += len(chunk_bytes)

                if written > MAX_ZIP_BYTES:
                    raise RuntimeError(
                        f"{state}/{kind}: "
                        "exceeded size guard"
                    )

                fh.write(chunk_bytes)

    part.replace(paths.zip_path)

    return paths.zip_path


def extract_to_parquet(
    root: Path,
    state: str,
    year: int,
    kind: str,
    columns: list[str] | None = None,
) -> Path:
    import duckdb

    paths = make_paths(
        root,
        state,
        year,
        kind,
    )

    want = set(
        columns
        or (
            PERSON_COLUMNS
            if kind == "p"
            else HOUSING_COLUMNS
        )
    )

    work = (
        paths.root
        / "work"
        / state
    )

    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(paths.zip_path) as zf:
        expected_name = inner_csv_name(
            state,
            kind,
        )

        name = next(
            n
            for n in zf.namelist()
            if n.split("/")[-1].lower()
            == expected_name.lower()
        )

        zf.extract(
            name,
            work,
        )

    csv_path = work / inner_csv_name(
        state,
        kind,
    )

    # Some archives may contain the CSV inside a directory.
    if not csv_path.exists():
        candidates = list(
            work.rglob(
                inner_csv_name(
                    state,
                    kind,
                )
            )
        )

        if not candidates:
            raise FileNotFoundError(
                f"could not find extracted "
                f"{inner_csv_name(state, kind)}"
            )

        csv_path = candidates[0]

    with csv_path.open() as fh:
        header = (
            fh.readline()
            .strip()
            .split(",")
        )

    keep = [
        column
        for column in header
        if column in want
    ]

    # Fail loudly if the official ACS replicate weights are unexpectedly
    # absent from the source data.
    expected_replicates = (
        PERSON_REPLICATE_WEIGHTS
        if kind == "p"
        else HOUSING_REPLICATE_WEIGHTS
    )

    missing_replicates = [
        column
        for column in expected_replicates
        if column not in keep
    ]

    if missing_replicates:
        preview = ", ".join(
            missing_replicates[:5]
        )

        suffix = (
            f" ... ({len(missing_replicates)} total)"
            if len(missing_replicates) > 5
            else ""
        )

        raise RuntimeError(
            f"{state}/{kind}: ACS replicate "
            f"weights missing from source: "
            f"{preview}{suffix}"
        )

    pq = paths.parquet_path

    pq.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    try:
        collist = ", ".join(
            f'"{column}"'
            for column in keep
        )

        con.execute(
            f"""
            COPY (
                SELECT {collist}
                FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    header=true,
                    sample_size=-1
                )
            )
            TO '{pq.as_posix()}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )
    finally:
        con.close()

    csv_path.unlink(
        missing_ok=True
    )

    paths.zip_path.unlink(
        missing_ok=True
    )

    shutil.rmtree(
        work,
        ignore_errors=True,
    )

    return pq
