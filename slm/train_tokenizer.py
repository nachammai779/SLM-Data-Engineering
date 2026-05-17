"""Train a custom BPE tokenizer on the corpus.

Enforces the three constraints from PLAN.md §2.B:
  B.1 Whitespace preservation — ByteLevel encoding maps each byte to a unique
      unicode char, so consecutive spaces / tabs / newlines survive intact.
  B.2 Llama-3-style regex pre-tokenization — splits BEFORE BPE so it cannot
      glue digits, punctuation, or snake_case tokens together.
  B.3 Hard vocab cap — keeps the embedding layer small enough to fit a 60M budget.
"""
from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from slm.config import TrainConfig, smoke_config, real_config


# Strict split rule: each digit standalone, every non-alphanumeric char isolated,
# whitespace runs grouped (ByteLevel preserves them byte-by-byte after this).
#
#   \p{L}+         letter runs (words like `bucket`, `name`, `def`)
#   \p{N}          single digit (so `2026` → `2`,`0`,`2`,`6` — math stays compositional)
#   [^\s\p{L}\p{N}] one non-alphanumeric non-whitespace char (each `_`, `(`, `,` isolated)
#   \s+            any whitespace run (preserved literally by ByteLevel)
SPLIT_PATTERN = r"\p{L}+|\p{N}|[^\s\p{L}\p{N}]|\s+"

SPECIAL_TOKENS = ["<unk>", "<pad>", "<eos>"]


def build_trainer(vocab_size: int) -> BpeTrainer:
    return BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )


def build_tokenizer_skeleton() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Sequence([
        Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = ByteLevelDecoder()
    return tok


def corpus_files(corpus_dir: Path) -> list[str]:
    exts = {".txt", ".py", ".sql", ".md"}
    files = [str(p) for p in sorted(corpus_dir.rglob("*"))
             if p.is_file() and p.suffix in exts]
    if not files:
        raise FileNotFoundError(f"No corpus files under {corpus_dir}")
    return files


def train(cfg: TrainConfig, vocab_size: int, out_path: Path) -> Tokenizer:
    files = corpus_files(Path(cfg.corpus_dir))
    print(f"Training BPE on {len(files)} files -> vocab_size={vocab_size}")

    tok = build_tokenizer_skeleton()
    trainer = build_trainer(vocab_size)
    tok.train(files, trainer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"Saved -> {out_path}")
    print(f"Actual vocab size: {tok.get_vocab_size()}")
    return tok


def smoke_demo(tok: Tokenizer) -> None:
    """Sanity-check the three constraints on representative code snippets."""
    samples = [
        "def extract_s3_to_postgres(bucket_name, secret_arn):",
        "SELECT COUNT(*) FROM events WHERE event_time >= '2026-05-16';",
        "    df = pd.read_csv(path, chunksize=10000)",  # 4-space indent
        "x = 2026 + 42",
    ]
    print("\n=== Constraint checks (each cell = one token, round-tripped) ===")
    for s in samples:
        ids = tok.encode(s).ids
        pieces = [tok.decode([i]) for i in ids]
        print(f"\nINPUT  : {s!r}")
        print(f"N_TOKS : {len(ids)}")
        print(f"PIECES : {[repr(p) for p in pieces]}")


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        cfg = real_config()
        tok = train(cfg, vocab_size=cfg.vocab_size, out_path=Path(cfg.tokenizer_path))
    else:
        cfg = smoke_config()
        tok = train(cfg, vocab_size=cfg.vocab_size, out_path=Path(cfg.tokenizer_path))
    smoke_demo(tok)