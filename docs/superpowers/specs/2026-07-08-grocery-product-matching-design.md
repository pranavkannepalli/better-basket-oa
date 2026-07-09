# Grocery Product Matching Design

## Goal

Build a Python matching pipeline that takes grocery store A and grocery store B item catalogs and produces one best match in B for every item in A. The pipeline should maximize useful coverage, rely heavily on GPT-5 nano for semantic judgment, and still use deterministic retrieval and scoring to keep runtime and API cost practical.

## Inputs

- `grocery_store_a_items_final.csv`
- `grocery_store_b_items_final.csv`

Observed scale:

- Store A: 233,199 rows
- Store B: 55,516 rows

Observed fields that matter:

- `item_id`
- `name`
- `brand_raw`
- `description`
- `item_info`
- `tags`
- `subcategory`
- `is_private_label` or equivalent store-brand signals
- `sizing_comp`
- `size_raw`
- `url`

The catalogs contain grocery and non-grocery rows. The first version should still attempt to match all rows.

## Output Contract

The pipeline returns exactly one row for every item in store A.

Primary output CSV:

- `item_id_a`
- `item_id_b`
- `confidence`
- `match_quality`
- `decision_source`
- `review_flag`

Definitions:

- `confidence`: normalized score in `[0, 1]`
- `match_quality`: `high`, `medium`, or `low`
- `decision_source`: `rules`, `llm`, or `fallback`
- `review_flag`: boolean for weak matches that may need manual inspection

Secondary output:

- detailed JSONL logs explaining why each match won
- aggregate run summary with counts and coverage stats

## Constraints

## Functional

- One best `B` row per `A` row
- Wide use of GPT-5 nano in scoring, not only as a last resort
- Match quality can be weak, but every `A` row still gets a best available `B` row
- Product names may differ materially, so the system must use more than name matching

## Practical

- A literal cross join of all A rows to all B rows is too large
- The system must approximate "compare against all B rows" through efficient retrieval
- The implementation must be executable in Python
- The pipeline should cache reusable work to make reruns cheaper

## Matching Philosophy

The system should pick the B item that is most likely to be:

1. the exact same national-brand item, if available
2. otherwise the closest customer-equivalent product
3. otherwise the least-bad best guess in B

Customer-equivalent means a shopper would consider the two products essentially interchangeable for the intended purchase, even when brands differ. This is especially important for private label, produce, fresh, frozen, and loose goods.

## Architecture

The pipeline has seven stages:

1. normalization and feature extraction
2. broad candidate retrieval over all B rows
3. deterministic feature scoring
4. GPT-5 nano semantic scoring
5. final one-best-match resolution
6. persistence and reusable learning artifacts
7. output generation and summary reporting

## Stage 1: Normalization And Feature Extraction

Build structured product records from each CSV row.

### Parsed fields

- raw name
- normalized name
- raw brand
- normalized brand
- category path from `item_info`
- extracted description summary
- size value
- size unit
- pack count
- form and storage flags
- private-label signal
- key tags and attribute flags

### Normalization goals

- lowercase, strip punctuation noise, normalize whitespace
- normalize units such as `oz`, `fl oz`, `lb`, `ct`
- parse embedded quantities from titles and descriptions
- infer useful traits like `organic`, `frozen`, `plain`, `whole milk`, `boneless`, `unscented`
- separate informative tokens from weak marketing words

### Derived features

- `tokens_core`
- `tokens_full`
- `brand_norm`
- `size_value`
- `size_unit`
- `pack_count`
- `category_path`
- `form_flags`
- `attribute_flags`
- `private_label_flag`

## Stage 2: Broad Candidate Retrieval

All B rows should be searchable for every A row, but not fully pair-scored.

The retrieval layer should create a wide candidate pool per A item using multiple independent retrieval methods, then merge them.

### Retrieval methods

- lexical retrieval on normalized title and attribute tokens
- category-aware retrieval
- brand-aware retrieval for national brands
- size-aware retrieval
- semantic embedding retrieval

### Candidate strategy

For each A row:

- retrieve broad candidates across all B rows
- merge candidates from all retrieval methods
- deduplicate them
- keep a wide top `K` set, expected roughly `100-300`

This is the practical replacement for a literal full cross join.

## Stage 3: Deterministic Feature Scoring

Apply cheap structured scoring to the candidate pool before LLM-heavy scoring.

### Positive signals

- similar core product tokens
- same or compatible category path
- same national brand
- private-label compatibility
- similar normalized size
- similar pack count
- compatible product form and storage type
- overlapping important attributes such as flavor, fat content, organic, variety, cut, scent, count

### Negative signals

- contradictory category or department
- incompatible size units where conversion is impossible
- conflicts like dog food vs human food
- strong attribute contradictions such as `paste` vs `sauce`, `frozen` vs `fresh`, `beef` vs `chicken`
- severe count or concentration mismatch

### Output

Each candidate receives:

- deterministic score
- feature breakdown
- contradiction flags

This stage does not pick the final match alone. Its main job is to remove obviously bad pairs and rank the rest for LLM review.

## Stage 4: GPT-5 Nano Semantic Scoring

GPT-5 nano is an integral part of the pipeline and should score a wide share of finalists, not only edge cases.

### LLM role

For each A item, send the strongest remaining candidates to GPT-5 nano with compact structured inputs. The model should judge both exact-match likelihood and customer-equivalent likelihood.

### LLM input

For the A item and each candidate B item:

- raw name
- normalized name
- brand
- category path
- size and unit
- pack count
- private-label signal
- form/storage flags
- short distilled description
- key extracted attributes

### LLM tasks

- score whether the pair is the same exact national-brand item
- score whether the pair is a customer-equivalent substitute
- identify major conflicts
- provide a confidence score in `[0, 1]`
- provide short reason codes

### LLM usage pattern

- use the LLM on a broad final candidate set, not just tiny leftovers
- keep prompts structured and compact
- prefer batch-style or repeated small pairwise judgments over huge free-form prompts
- cache results by stable pair fingerprint

## Stage 5: Final One-Best-Match Resolution

For each A item:

1. retrieve top candidates
2. apply deterministic scoring and contradiction filtering
3. send finalists to GPT-5 nano
4. combine deterministic and LLM outputs
5. choose exactly one best B item

### Resolution rules

- if a candidate has very strong structured and LLM agreement, mark `high`
- if a candidate is a plausible customer-equivalent with some ambiguity, mark `medium`
- if no candidate is strong, still emit the best available choice and mark `low`

### Tie-breaks

If the top candidates are close:

- run an LLM tie-break comparison among the top few
- prefer the candidate with the best customer-equivalence judgment
- use deterministic features as secondary tie-breakers

## Stage 6: Fallback Heuristics

Fallback does not mean "drop the row." It means "choose the least-bad best guess when strong evidence is missing."

Fallback features may include:

- family-level category similarity
- size ratio compatibility
- package count similarity
- price-shape heuristics if pricing becomes available later
- previously learned accepted patterns from earlier runs

Fallback decisions should be clearly labeled:

- `decision_source = fallback`
- `match_quality = low`
- `review_flag = true`

## Stage 7: Persistence And Reusable Learning

Persist artifacts locally so reruns are cheaper and improve over time.

### Persisted artifacts

- normalized product tables
- embeddings
- candidate retrieval outputs
- deterministic feature scores
- GPT pairwise results
- final match decisions

### Reusable learned patterns

- accepted brand mappings
- synonym and alias tables
- category-specific size tolerances
- private-label equivalence patterns
- conflict penalties by category

This persistence layer should improve both speed and matching quality on later runs.

## Scoring Strategy

The final score should be a weighted combination where the LLM carries substantial weight.

### Intended balance

- deterministic retrieval and scoring ensure recall and control cost
- GPT-5 nano is the main semantic judge for the final decision
- deterministic features remain important for hard contradictions and factual signals like size, count, and form

The exact weights can be tuned during implementation, but the default posture should be:

- use the LLM heavily
- do not rely mainly on fuzzy title similarity

## Logging

Every final match should have an explainable record.

Per-match log fields should include:

- `item_id_a`
- `item_id_b`
- deterministic score
- LLM score
- final combined score
- top reason codes
- contradiction flags
- retrieval methods that surfaced the candidate
- final `match_quality`
- final `decision_source`
- `review_flag`

## Summary Reporting

At the end of a run, print and save:

- total rows in A
- total rows in B
- total emitted matches
- count and percentage of `high`, `medium`, `low`
- count by `decision_source`
- number of GPT calls
- GPT cache hit rate
- runtime by pipeline stage

## Non-Goals For V1

- perfect precision for every low-signal item
- human review UI
- complex online learning system
- full pricing model integration unless price data is later added

## Implementation Notes

- Python implementation should favor small modules and clear data flow
- the first version should prefer existing libraries for parsing, retrieval, and caching over custom infrastructure
- the pipeline should be resumable if long-running
- logs and caches should be written incrementally to avoid losing progress

## Risks

- non-grocery rows may produce noisy matches
- missing or inconsistent size data may weaken structured scoring
- overly wide LLM usage may increase runtime and cost if candidate pruning is weak
- private-label equivalence remains subjective in some categories

## Mitigations

- strong contradiction filters
- category-aware retrieval
- compact structured prompts
- persistent cache
- clear confidence and review flags

## Deliverables

- executable Python matching pipeline
- final match CSV with one row per A item
- detailed logs
- final count summary
