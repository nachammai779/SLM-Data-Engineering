from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    swiglu_inner: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0
    bias: bool = False  # retained for backward-compat in older callers; unused in current model


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

    z_loss_weight: float = 0.0
    compile: bool = False
    dtype: str = "float32"          # smoke: cpu/fp32; GPU preset switches to bfloat16

    # Architecture
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    swiglu_inner: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.1
    bias: bool = False


def smoke_config() -> TrainConfig:
    return TrainConfig()


def real_config() -> TrainConfig:
    """60M-target preset for GPU training on the curated corpus (RunPod)."""
    return TrainConfig(
        corpus_dir="slm/data/curated",
        data_dir="slm/data/real",
        out_dir="slm/runs/real",
        tokenizer_path="slm/tokenizers/bpe_real.json",
        vocab_size=16384,
        # data
        batch_size=32,
        block_size=2048,
        # training
        max_iters=8000,
        eval_iters=50,
        eval_interval=200,
        warmup_steps=200,
        learning_rate=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        grad_accum_steps=2,
        grad_clip=1.0,
        z_loss_weight=1e-4,
        compile=True,
        dtype="bfloat16",
        # architecture (~55M params, target ~60M)
        n_layer=6,
        n_head=12,
        n_embd=768,
        swiglu_inner=2048,
        rope_base=10000.0,
        dropout=0.0,
        bias=False,
    )


def gpt_config(cfg: TrainConfig) -> GPTConfig:
    return GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        swiglu_inner=cfg.swiglu_inner,
        rope_base=cfg.rope_base,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )