"""Stream each source, write per-source text files into slm/data/curated/."""
from pathlib import Path
from typing import Iterator

from slm.curate.sources import (
    iter_code_python,
    iter_code_cloud,
    iter_sql,
    iter_cosmopedia_de,
    iter_openhermes_de,
)

OUT_DIR = Path("slm/data/curated")
SEP = "\n\n<|endoftext|>\n\n"

# First-pass sandbox budgets — see PLAN.md §1 ratios.
# Total raw text target: ~30-50MB, well under the 2GB cache budget.
BUDGETS = {
    "code":       1000,   # codeparrot Python (35%)
    "cloud":       500,   # codeparrot Python filtered for boto3/google.cloud (25%)
    "sql":         800,   # b-mc2/sql-create-context (compact rows; lifts SQL coverage)
    "cosmopedia":  500,   # cosmopedia-v2 DE-keyword filter (25%)
    "openhermes":  300,   # OpenHermes 2.5 DE-keyword filter (15%)
}


def write_source(name: str, gen: Iterator[str]) -> tuple[int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.txt"
    n_docs = 0
    n_chars = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in gen:
            f.write(doc)
            f.write(SEP)
            n_docs += 1
            n_chars += len(doc)
    print(f"{name:12s}: {n_docs:5d} docs | {n_chars:>12,} chars -> {path}")
    return n_docs, n_chars


def main() -> None:
    total_chars = 0
    total_chars += write_source("code",       iter_code_python(BUDGETS["code"]))[1]
    total_chars += write_source("cloud",      iter_code_cloud(BUDGETS["cloud"]))[1]
    total_chars += write_source("sql",        iter_sql(BUDGETS["sql"]))[1]
    total_chars += write_source("cosmopedia", iter_cosmopedia_de(BUDGETS["cosmopedia"]))[1]
    total_chars += write_source("openhermes", iter_openhermes_de(BUDGETS["openhermes"]))[1]
    print(f"\nTOTAL raw text: {total_chars:,} chars (~{total_chars/1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()