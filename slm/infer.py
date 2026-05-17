"""Load best.pt and generate completions for representative DE prompts."""
import argparse
from pathlib import Path

import torch

from slm.config import GPTConfig
from slm.model import GPT
from slm.tokenizer import BPETokenizer


PROMPTS = [
    # 1. Python docstring -> implementation
    (
        "docstring -> implementation",
        'def extract_s3_to_postgres(bucket: str, key: str, table: str):\n'
        '    """\n'
        '    Stream a CSV from S3 into PostgreSQL using psycopg2 COPY.\n'
        '    """\n',
    ),
    # 2. SQL schema -> analytical query
    (
        "SQL schema -> query",
        "CREATE TABLE orders (\n"
        "    order_id   BIGSERIAL PRIMARY KEY,\n"
        "    user_id    UUID NOT NULL,\n"
        "    amount     NUMERIC(12,2),\n"
        "    order_date DATE NOT NULL\n"
        ");\n"
        "-- Top 10 users by total spend in the last 90 days:\n"
        "SELECT ",
    ),
    # 3. boto3 partial -> body completion
    (
        "boto3 partial",
        "import boto3\n\n"
        "def list_all_objects(bucket: str, prefix: str) -> list[str]:\n"
        '    """List every object key under a prefix, handling pagination."""\n'
        "    s3 = boto3.client(\"s3\")\n",
    ),
    # 4. Pandas natural-language comment -> code
    (
        "pandas comment -> code",
        "# Read a Parquet file with pandas, filter rows where amount > 100,\n"
        "# group by country, and write the result back as Parquet.\n"
        "import pandas as pd\n"
        "df = ",
    ),
]


def load_model(ckpt_path: Path, device: str) -> tuple[GPT, BPETokenizer]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    gcfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        swiglu_inner=cfg.swiglu_inner,
        rope_base=cfg.rope_base,
        dropout=0.0,
        bias=cfg.bias,
    )
    model = GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer(cfg.tokenizer_path)
    print(f"Loaded best.pt: step={ckpt.get('step', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}, "
          f"params={model.num_params() / 1e6:.2f}M")
    return model, tok


def generate(
    model: GPT, tok: BPETokenizer, prompt: str, device: str,
    max_new: int = 120, temperature: float = 0.8, top_k: int = 50,
) -> str:
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new, temperature=temperature, top_k=top_k)
    out_ids = y[0].tolist()
    return tok.decode(out_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="slm/runs/real/best.pt")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    torch.manual_seed(0)

    model, tok = load_model(Path(args.ckpt), device)
    print(f"device={device} | max_new={args.max_new} | T={args.temperature} | top_k={args.top_k}")

    for label, prompt in PROMPTS:
        print("\n" + "=" * 78)
        print(f"PROMPT [{label}]")
        print("-" * 78)
        print(prompt, end="")
        completion = generate(
            model, tok, prompt, device,
            max_new=args.max_new, temperature=args.temperature, top_k=args.top_k,
        )
        # The decoded string includes the prompt; print only the new portion.
        new_text = completion[len(prompt):] if completion.startswith(prompt) else completion
        print("\n--- COMPLETION ---")
        print(new_text)


if __name__ == "__main__":
    main()