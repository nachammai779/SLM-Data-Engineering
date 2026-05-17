from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


@dataclass
class TrainConfig:
    corpus_dir: str = "slm/toy_corpus"
    data_dir: str = "slm/data"
    out_dir: str = "slm/runs/smoke"

    tokenizer_name: str = "bpe"
    tokenizer_path: str = "slm/tokenizers/bpe_smoke.json"
    vocab_size: int = 1024

    val_fraction: float = 0.1

    batch_size: int = 8
    block_size: int = 64
    max_iters: int = 200
    eval_iters: int = 20
    eval_interval: int = 50

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_accum_steps: int = 4
    grad_clip: float = 1.0

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


def smoke_config() -> TrainConfig:
    return TrainConfig()


def real_config() -> TrainConfig:
    """Sandbox preset for the curated multi-source corpus."""
    return TrainConfig(
        corpus_dir="slm/data/curated",
        data_dir="slm/data/real",
        out_dir="slm/runs/real",
        tokenizer_path="slm/tokenizers/bpe_real.json",
        vocab_size=4096,
        batch_size=16,
        block_size=128,
        max_iters=500,
        eval_iters=30,
        eval_interval=100,
        warmup_steps=50,
        grad_accum_steps=4,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )


def gpt_config(cfg: TrainConfig) -> GPTConfig:
    return GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )