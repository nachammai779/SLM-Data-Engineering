# SLM for Data Engineering — Plan

**Boilerplate:** `discovery-to-deployment/notebooks/Small_Language_Model_Scratch_HW.ipynb` (starting point for the from-scratch model code).

**Target:** ~60M parameter model focused on Python, PostgreSQL, AWS, and GCP for data-engineering tasks. Goal is dense mapping of syntax, API signatures, and architectural logic — not general knowledge. Should excel at synthesizing boilerplate, generating correct SQL schemas, writing precise SDK calls (boto3 / google-cloud), and handling PySpark/pandas logic.

---

## 1. Token & Curation Mix

Curated pre-training corpus of ~20B–30B tokens.

| Component | Target Allocation | Source Datasets (Hugging Face) |
|---|---|---|
| Python & SQL Syntax | 35% | The Stack v2 / StarCoder (filter `.py`, `.sql`). Include Python-Edu for clean, well-commented structures. |
| Data Engineering Logic | 25% | Cosmopedia v2 (Textbook & Howto). Extract sections on data engineering, database design, normalization, ACID properties, distributed systems. |
| Cloud APIs (AWS & GCP) | 25% | Open markdown docs / GitHub SDK code for Boto3 (AWS) and Google Cloud Client Libraries. |
| Instruction / Multi-Turn | 15% | OpenHermes 2.5 / Code-Feedback (filtered to data-engineering tasks — transformations, queries). |

---

## 2. Advanced Curation Strategies ("Edge Synthesizer")

### A. Docstring + Implementation Masking
Train heavily on functions structured as docstring-first, so the model learns to jump from natural-language intent to low-level implementation:

```python
def extract_s3_to_postgres(bucket_name: str, secret_arn: str, table_name: str):
    """
    Extracts a CSV file from AWS S3 using boto3 and streams it into a PostgreSQL table.
    """
    # Force the model to fill in the rest
```

Pair code blocks directly with markdown explanations during pre-training.

### B. Heavy Tokenizer Pruning
- Start from a base tokenizer (Llama or Qwen).
- Keep syntax characters, whitespace (`\t`, spaces), common programming keywords, common English.
- Drop tokens unrelated to tech/logic/basic English.
- **Target vocabulary: ~16,000–24,000 tokens.** Frees millions of embedding parameters to redirect into deeper hidden layers.

#### The catch: you must customize the tokenizer
A stock BPE tokenizer trained on general web text (e.g., GPT-2) will perform poorly on code. Three constraints to enforce when training or pruning:

**B.1 — Strict whitespace preservation.**
The BPE training script must NOT collapse runs of whitespace into a single space, and must NOT strip trailing spaces. In code, `' '` (one space) is fundamentally different from `'    '` (four spaces — Python indentation). Treat each whitespace run as semantically distinct.

**B.2 — Regex pre-tokenization (enforce split rules).**
Before BPE starts merging characters, run raw text through a regex splitter to prevent **toxic merges** (code + punctuation glued into one token). Use a **Llama-3 style** split regex so:
- **Numbers split into individual digits** — keeps `2026` from becoming one unchangeable token (which would destroy arithmetic and indexing ability).
- **Punctuation and brackets stay isolated** (`.`, `(`, `)`, `[`, `]`) unless they form a known operator (`==`, `!=`, `->`, `:=`).
- **snake_case stays decomposed** — `bucket_name` tokenizes cleanly as `bucket` + `_` + `name`, not as a single rare token the model can never reuse.

**B.3 — Hard cap the vocabulary.**
Typical large-model BPE vocab is 32k–128k. For a 60M model that's catastrophic:

> Embedding params = vocab_size × d_model
>
> At `d_model = 768` and `vocab = 100,000`, the embedding layer alone is **76.8M params** — already over budget before a single transformer layer exists.

Target vocab: **16,000–24,000 tokens.**
At `vocab = 16,000` and `d_model = 768`, the embedding layer is **~12.2M params**, leaving ~48M for the transformer stack to learn actual logic, cloud SDK structure, and SQL syntax.

---

## 3. Specific Capability Mapping (Four Pillars)

### Pillar 1 — Python Data Pipelines
- **Libraries:** pandas, polars, PySpark, DuckDB.
- **Must generate:** dataframe transformations, memory-efficient generators, JSON parsing, chunked file reads.

### Pillar 2 — PostgreSQL Dialect
- **Concepts:** DDL generation, indexing, window functions (`ROW_NUMBER()`, `PARTITION BY`), JSONB querying, `COPY FROM`.
- **Must generate:** well-optimized analytical queries, safe migrations without syntax hallucinations.

### Pillar 3 — AWS Data Services
- **Components:** S3, AWS Glue, Athena, Redshift, Secrets Manager.
- **Must generate:** boto3 boilerplate for object upload/download, Glue job triggers, secure credential fetches.

### Pillar 4 — GCP Data Services
- **Components:** BigQuery, Cloud Storage (GCS), Cloud Run, Pub/Sub.
- **Must generate:** `google-cloud-bigquery` client code, streaming ingestion scripts, bucket interactions.

---

## 4. Architectural Fine-Tuning for Edge Speed

- **Context window:** 2,048 or 4,096 tokens. Edge synthesizers need local context, not long history.
- **RoPE:** Use Rotary Position Embeddings so context scales gracefully during long debugging sessions.
- **Quantization-first training:** 60M @ FP16 ≈ 120MB. INT4/INT8 via llama.cpp or ONNX Runtime → 30MB–60MB. Hundreds of tokens/sec on CPU cache.

---

## 5. Evaluation Strategy

Custom test suite — skip generic MMLU.

1. **HumanEval (Python subset)** — basic coding logic.
2. **Custom Text-to-SQL** — feed schemas, check correct PostgreSQL output.
3. **SDK Compilation test** — generate 100 boto3 / google-cloud snippets, programmatically check argument alignment against real SDK specs.