"""End-to-end training loop. Run `python -m slm.tokenize_corpus` first."""
from pathlib import Path
import time

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from slm.config import TrainConfig, gpt_config, smoke_config
from slm.data import get_batch
from slm.model import GPT


def estimate_loss(model: GPT, cfg: TrainConfig, device: str) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split, cfg, device)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: TrainConfig) -> GPT:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = GPT(gpt_config(cfg)).to(device)
    print(f"Model: {model.num_params() / 1e6:.2f}M params | device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        eps=1e-9,
    )
    warmup = LinearLR(optimizer, total_iters=cfg.warmup_steps)
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

    t0 = time.time()
    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 and step > 0:
            losses = estimate_loss(model, cfg, device)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.5f} | {elapsed:.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg}, best_path)

        X, y = get_batch("train", cfg, device)
        _, loss = model(X, y)
        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0 or step + 1 == cfg.max_iters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    final = estimate_loss(model, cfg, device)
    print(f"FINAL | train {final['train']:.4f} | val {final['val']:.4f} | best_val {best_val:.4f}")
    return model


if __name__ == "__main__":
    train(smoke_config())