"""Plot training/validation loss curves from a run's losses.csv."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    steps, train, val, lr = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
            lr.append(float(row["lr"]))
    return steps, train, val, lr


def plot(csv_path: Path, vocab_size: int, out_path: Path) -> None:
    steps, train, val, lr = load_csv(csv_path)
    uniform = math.log(vocab_size)
    best_val = min(val)
    best_step = steps[val.index(best_val)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, train, marker="o", linewidth=2, label="Train loss", color="#1f77b4")
    ax.plot(steps, val, marker="s", linewidth=2, label="Val loss", color="#d62728")
    ax.axhline(uniform, linestyle="--", color="gray", alpha=0.7,
               label=f"Uniform baseline = log({vocab_size}) = {uniform:.2f}")

    ax.annotate(
        f"Best val: {best_val:.3f}\n@ step {best_step}",
        xy=(best_step, best_val),
        xytext=(best_step - 600, best_val + 0.7),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    final_gap = train[-1] - val[-1]
    ax.text(
        0.98, 0.95,
        f"Final train: {train[-1]:.3f}\nFinal val:   {val[-1]:.3f}\nGap:         {abs(final_gap):.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("SLM Training — 55M params, 50M tokens, RTX 4090")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    print(f"  Final train loss: {train[-1]:.4f}")
    print(f"  Final val   loss: {val[-1]:.4f}")
    print(f"  Best  val   loss: {best_val:.4f} @ step {best_step}")
    print(f"  Train-val gap:    {abs(final_gap):.4f}")
    print(f"  Perplexity (val): {math.exp(val[-1]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="slm/runs/real/losses.csv")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--out", default="slm/runs/real/loss_curve.png")
    args = ap.parse_args()
    plot(Path(args.csv), args.vocab, Path(args.out))