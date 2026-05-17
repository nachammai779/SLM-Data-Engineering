"""Read text files from corpus_dir, tokenize, write train.bin/val.bin memmaps."""
from pathlib import Path
import numpy as np

from slm.config import TrainConfig, smoke_config, real_config
from slm.tokenizer import build_tokenizer


CORPUS_EXTS = {".txt", ".py", ".sql", ".md"}


def read_corpus(corpus_dir: Path) -> list[str]:
    docs = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix in CORPUS_EXTS:
            docs.append(p.read_text(encoding="utf-8"))
    if not docs:
        raise FileNotFoundError(f"No corpus files under {corpus_dir}")
    return docs


def tokenize(cfg: TrainConfig) -> None:
    corpus_dir = Path(cfg.corpus_dir)
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer(cfg.tokenizer_name, path=getattr(cfg, "tokenizer_path", None))
    if tok.vocab_size > cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size {tok.vocab_size} exceeds config vocab_size {cfg.vocab_size}; "
            "increase TrainConfig.vocab_size or retrain the tokenizer smaller."
        )
    if tok.vocab_size < cfg.vocab_size:
        print(f"NOTE: tokenizer has {tok.vocab_size} tokens; "
              f"model will allocate {cfg.vocab_size} embedding rows ({cfg.vocab_size - tok.vocab_size} unused).")

    docs = read_corpus(corpus_dir)
    print(f"Loaded {len(docs)} files from {corpus_dir}")

    all_ids: list[int] = []
    for d in docs:
        all_ids.extend(tok.encode(d))
        all_ids.append(tok.eot_token)
    n = len(all_ids)
    print(f"Total tokens: {n:,} (tokenizer={cfg.tokenizer_name}, vocab={tok.vocab_size})")

    n_val = max(int(n * cfg.val_fraction), cfg.block_size + 1)
    train_ids = np.array(all_ids[:-n_val], dtype=np.uint16)
    val_ids = np.array(all_ids[-n_val:], dtype=np.uint16)

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(
        f"Wrote {train_path} ({len(train_ids):,} tokens) "
        f"and {val_path} ({len(val_ids):,} tokens)"
    )


if __name__ == "__main__":
    import sys
    cfg = real_config() if "--real" in sys.argv else smoke_config()
    tokenize(cfg)