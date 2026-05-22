# SLM for Data Engineering

A ~60M-parameter Small Language Model trained from scratch, specialized for data-engineering workloads: Python (pandas / polars / PySpark / DuckDB), PostgreSQL, AWS (S3, Glue, Athena, Redshift, Secrets Manager via `boto3`), and GCP (BigQuery, GCS, Cloud Run, Pub/Sub via `google-cloud` client libs). Designed for edge deployment (CLI / IDE plugin / laptop) after INT4/INT8 quantization.

The goal is **depth over breadth**: a tiny model that produces precise SDK calls, correct PostgreSQL, and idiomatic pipeline code — not a general-purpose chatbot.

## Status

Initial training run complete on RunPod GPU. Corpus: **57.8M tokens** across 4 HuggingFace sources (codeparrot-clean Python, Cosmopedia v2, OpenHermes 2.5, b-mc2/sql-create-context), tokenized with a custom **8,192-token BPE** vocab.

| Metric | Value |
|---|---|
| Parameters | ~55M |
| Final val loss | 2.28 (vs uniform baseline 9.70) |
| Final val perplexity | 9.76 — ~1,700× better than chance |
| Context window | 2,048 |
| Precision | BF16 autocast, fp32 master weights |

Training was stable end-to-end (no NaNs, no spikes). The train–val gap widened monotonically — classic early-overfitting signature at ~1:1 token-to-parameter ratio, ~20× under Chinchilla-optimal. More data is the highest-leverage next investment.

Full loss curve, qualitative samples, and interpretation: [`slm/runs/real/ANALYSIS.md`](slm/runs/real/ANALYSIS.md).

## Layout

```
PLAN.md                     # Full project plan (corpus mix, tokenizer, architecture, eval strategy)
slm/
  config.py                 # GPTConfig + TrainConfig dataclasses
  model.py                  # Transformer: RoPE, SwiGLU, RMSNorm, no-bias, z-loss
  train.py                  # Training loop (cosine LR, grad accum, gradient clip)
  infer.py                  # Sampling / generation
  tokenizer.py              # Custom BPE tokenizer with whitespace-preserving pre-tokenization
  train_tokenizer.py
  tokenize_corpus.py        # Materializes train.bin / val.bin from curated text
  plot_losses.py
  curate/                   # Streaming HuggingFace dataset curation
    sources.py              # Per-source generators (codeparrot, cosmopedia, openhermes, sql)
    filters.py              # DE-keyword filtering, cloud-import detection
    build.py                # Mix sources to per-bucket token budgets
  runs/real/                # Curated artifacts from the published training run
    ANALYSIS.md
    losses.csv
    loss_curve.png
  requirements.txt
discovery-to-deployment/    # Separate course project — own git repo
```

## Running

Tokenizer + training expect a CUDA GPU for the real config. The smoke config in `slm/config.py` runs on CPU for sanity checks.

```bash
cd slm
pip install -r requirements.txt

# 1. Curate the corpus (streams from HuggingFace, writes slm/data/curated/*.txt)
python -m curate.build

# 2. Train the BPE tokenizer on the curated corpus
python train_tokenizer.py

# 3. Tokenize the corpus into train.bin / val.bin
python tokenize_corpus.py

# 4. Train
python train.py

# 5. Sample from a checkpoint
python infer.py
```

Model checkpoints (`*.pt`) are gitignored. The `slm/runs/real/` directory is kept in git for the curated analysis artifacts only.

## Plan and design rationale

See [`PLAN.md`](PLAN.md) for:
- Corpus design (target 20B–30B tokens across 4 buckets)
- Tokenizer strategy (vocab pruned to 16k–24k; whitespace-preserving BPE; Llama-3-style pre-tokenization split regex)
- Architecture choices (RoPE, 2k/4k context, INT4/INT8 quantization-aware path)
- Four "capability pillars" — Python pipelines, PostgreSQL, AWS, GCP
- Evaluation suite: HumanEval (Python subset), custom text-to-SQL with execution, SDK-compilation tests

## Starting point

The model code began from `discovery-to-deployment/notebooks/Small_Language_Model_Scratch_HW.ipynb` (a from-scratch SLM tutorial notebook) and was extended into the modular `slm/` package documented above.