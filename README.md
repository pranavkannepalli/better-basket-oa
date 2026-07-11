# Better Basket Product Matcher

CSV-based product matching pipeline for mapping Store A items to Store B items.

The final output is a matches CSV at:

```text
artifacts/full-llm-local-embeddings/matches.csv
```

## Architecture

The matcher does the following:

1. Loads Store A and Store B CSV catalogs.
2. Normalizes product records into structured fields.
3. Resolves exact matches (if any) from global IDs and provider IDs (if they don't exist, it attempts to parse the URL to find them).
4. Builds a candidate retrieval index over Store B.
5. Retrieves a small candidate set for each unmatched Store A item.
6. Scores candidate pairs with deterministic rules.
7. Uses LLM reranking only for promising shortlists.
8. Writes checkpointed intermediate outputs and final CSV exports.

Core modules:

- `matcher/io.py`: CSV loading, product normalization, output writing
- `matcher/exact.py`: UPC/global ID and provider ID matching
- `matcher/retrieval.py`: TF-IDF, brand/size indexes, FAISS-backed embedding retrieval
- `matcher/scoring.py`: deterministic pair scoring and contradiction checks
- `matcher/llm.py`: structured LLM shortlist reranking
- `matcher/pipeline.py`: orchestration, checkpointing, process workers, progress logging
- `matcher/cli.py`: command-line entry point

## Design Choices

The first implementation was the simplest direct matcher: normalize both catalogs, generate candidate pairs, score them, and write results. That was correct, but doing direct comparisons for every (store A, store B) pair was slow for the full dataset.

The pipeline now avoids such comparisons. Store B is indexed once, then each Store A item retrieves a limited candidate set using:

- TF-IDF text similarity
- brand indexes
- size indexes
- local embeddings
- FAISS HNSW search

Deterministic scoring uses the following product attributes:

- token/name overlap
- category overlap
- brand/private-label handling
- size and unit compatibility
- pack count compatibility
- form flags such as fresh/frozen/dry/refrigerated
- attribute flags such as organic, gluten-free, low sodium, etc.

I ended up adding some hard contradiction checks to reduce false positives for mismatched product types, sizes, pack counts, diet variants, forms, categories, and cheese varieties (this would be a growing db that I would preferably not maintain by hand, but instead have an LLM feedback loop).

LLM usage is intentionally limited. The deterministic pipeline builds the shortlist first, and the LLM only reranks candidates above a configured score threshold. This keeps runtime and cost lower while still helping on ambiguous matches.

## Runtime Optimization

The main bottleneck was the comparison stage.

The initial embedding + FAISS approach projected roughly 8 hours end-to-end on the full dataset. The pipeline was reworked to use process workers for CPU-heavy retrieval and scoring, while parent-thread workers handle LLM calls. Query embedding work can run inside workers, and Store B retrieval indexes/embeddings are cached between runs.

Key runtime features:

- process worker mode for CPU-bound scoring
- configurable worker count
- cached retrieval index
- cached Store B embeddings
- FAISS HNSW CPU index for embedding search
- checkpoint/resume support
- incremental CSV writes
- atomic final output writes
- progress logs with ETA, CPU time, observed worker count, pending LLM calls, and RSS memory

The final full run processed `233,199` Store A rows in about `3,790` seconds (with cached embeddings for store B), with the comparison stage finishing in just over an hour on my PC (Ryzen 5 5600, 32 GB RAM, didn't even use 3070). Comparable cloud VMs (GCP c4a-standard-8 or AWS m7i.2xlarge) sit at around $0.40/hr usage with overall cost including LLM calls about $10 for this run if my math is correct.

## Productionization Notes

This implementation is intentionally file-based for the OA. In production, I would replace local CSV/artifact flow with persistent storage and incremental processing.

Expected production changes:

- As different async workers to batch load catalogs from a database instead of local CSVs.
- Add the following system to append new entries to either store:
    - Store A:
        - Just do one pass with one worker on the cached store B retrieval index.
    - Store B:
        - Find nearest products in Store B, identify what products in Store A those products are linked with, and then let an LLM decide for each pair.
- Use a managed vector/index layer such as Pinecone, Kuzu, pgvector, or another vector database depending on infrastructure constraints.
- Run matching incrementally for new or changed products instead of recomputing the full catalog.
- Add human review queues for medium-confidence matches and contradiction-heavy cases to build those hard contradiction checks I talked about about.
- Add metrics around match acceptance rate, rerun cost, LLM usage, latency, and drift.

## Usage

Install dependencies:

```bash
pip install -e .
```

Run the full default pipeline:

```bash
python run_matcher.py
```

Fast deterministic benchmark path:

```bash
python run_matcher.py \
  --no-llm \
  --no-embeddings \
  --worker-mode process \
  --max-workers 8 \
  --checkpoint-every 1000
```

Default higher-recall path:

```bash
python run_matcher.py \
  --worker-mode process \
  --max-workers 8 \
  --checkpoint-every 1000
```

Useful output files:

- `matches.csv`: final high-confidence submission file
- `matches_detailed.csv`: submitted matches with product details and confidence
- `match_decisions.csv`: all decisions, including low-confidence rows
- `run.log`: runtime log
- `pipeline-checkpoint.json`: resumable checkpoint

## Final Run

Final run summary:

- Store A decisions: `233,199`
- Submitted matches: `37,394`
- Confidence cutoff: `0.70`
- Runtime: `3,790.6s`
- Quality counts: `33,272 high`, `16,657 medium`, `183,270 low`
- Decision sources: `56,909 LLM`, `187 rules`, `176,103 fallback`
