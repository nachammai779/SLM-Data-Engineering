"""End-to-end training loop. Run `python -m slm.tokenize_corpus [--real]` first."""
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from slm.config import TrainConfig, gpt_config, smoke_config, real_config
from slm.data import get_batch
from slm.model import GPT


DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def make_autocast_ctx(device: str, dtype_name: str):
    if device == "cpu" or dtype_name == "float32":
        return nullcontext()
    return torch.amp.autocast(device_type=device, dtype=DTYPE_MAP[dtype_name])


def estimate_loss(model: GPT, cfg: TrainConfig, device: str, ctx) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split, cfg, device)
                with ctx:
                    _, loss = model(X, Y)  # no z-loss during eval
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: TrainConfig) -> GPT:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = GPT(gpt_config(cfg)).to(device)
    print(f"Model: {model.num_params() / 1e6:.2f}M params (non-tied) | device={device} | dtype={cfg.dtype}")

    if cfg.compile and device == "cuda":
        print("torch.compile(model) ...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        eps=1e-9,
    )
    warmup = LinearLR(optimizer, total_iters=max(1, cfg.warmup_steps))
    decay = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.max_iters - cfg.warmup_steps),
        eta_min=cfg.min_lr,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[cfg.warmup_steps]
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / "best.pt"

    losses_csv = out_dir / "losses.csv"
    with open(losses_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    ctx = make_autocast_ctx(device, cfg.dtype)
    t0 = time.time()
    for step in range(1, cfg.max_iters + 1):
        # === train step ===
        X, y = get_batch("train", cfg, device)
        with ctx:
            _, loss = model(X, y, z_loss_weight=cfg.z_loss_weight)
        (loss / cfg.grad_accum_steps).backward()

        if step % cfg.grad_accum_steps == 0 or step == cfg.max_iters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # === eval / checkpoint ===
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, cfg, device, ctx)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.5f} | {elapsed:7.1f}s"
            )

            with open(losses_csv, "a", newline="") as f:
                csv.writer(f).writerow([step, losses["train"], losses["val"], lr, elapsed])

            # checkpoint every eval (for graphs / presentations)
            ckpt = {
                "step": step,
                "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
            }
            torch.save(ckpt, out_dir / f"step_{step:07d}.pt")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(ckpt, best_path)

    print(f"FINAL best_val {best_val:.4f} | losses.csv -> {losses_csv}")
    return model


if __name__ == "__main__":
    import sys
    cfg = real_config() if "--real" in sys.argv else smoke_config()
    train(cfg)