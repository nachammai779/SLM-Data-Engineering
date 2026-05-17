"""One streaming generator per source. Each yields raw text docs."""
from typing import Iterator
from datasets import load_dataset

from slm.curate.filters import has_de_keyword, has_cloud_import


def iter_code_python(limit: int) -> Iterator[str]:
    """Open Python corpus from codeparrot (dedup'd, content-bundled)."""
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    n = 0
    for ex in ds:
        if n >= limit:
            return
        c = ex.get("content")
        if c:
            yield c
            n += 1


def iter_code_cloud(limit: int, scan_cap: int = 50_000) -> Iterator[str]:
    """Python files importing boto3 or google.cloud. Scans up to scan_cap docs."""
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    n = 0
    scanned = 0
    for ex in ds:
        if n >= limit or scanned >= scan_cap:
            return
        scanned += 1
        c = ex.get("content") or ""
        if has_cloud_import(c):
            yield c
            n += 1


def iter_sql(limit: int) -> Iterator[str]:
    """Text-to-SQL pairs as a single concatenated block per example."""
    ds = load_dataset("b-mc2/sql-create-context", streaming=True, split="train")
    n = 0
    for ex in ds:
        if n >= limit:
            return
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        a = ex.get("answer", "")
        yield f"-- Schema\n{ctx}\n-- Question\n{q}\n-- SQL\n{a}\n"
        n += 1


def iter_cosmopedia_de(limit: int, scan_cap: int = 50_000) -> Iterator[str]:
    """Synthetic textbook / how-to text filtered for data-engineering relevance."""
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus", name="cosmopedia-v2",
        streaming=True, split="train",
    )
    n = 0
    scanned = 0
    for ex in ds:
        if n >= limit or scanned >= scan_cap:
            return
        scanned += 1
        t = ex.get("text") or ""
        if has_de_keyword(t):
            yield t
            n += 1


def iter_openhermes_de(limit: int, scan_cap: int = 30_000) -> Iterator[str]:
    """Conversations from OpenHermes 2.5, filtered for DE relevance."""
    ds = load_dataset("teknium/OpenHermes-2.5", streaming=True, split="train")
    n = 0
    scanned = 0
    for ex in ds:
        if n >= limit or scanned >= scan_cap:
            return
        scanned += 1
        convo = ex.get("conversations") or []
        text = "\n".join(c.get("value", "") for c in convo if isinstance(c, dict))
        if text and has_de_keyword(text):
            yield text
            n += 1