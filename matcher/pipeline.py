from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from matcher.config import Settings
from matcher.embeddings import embedding_cache_path
from matcher.embeddings import load_embedding_cache
from matcher.embeddings import load_local_embedding_model
from matcher.embeddings import save_embedding_cache
from matcher.exact import (
    build_global_id_index,
    build_provider_id_index,
    choose_exact_global_id_match,
    choose_provider_id_match,
)
from matcher.llm import build_openai_client, score_shortlist_with_llm_cached
from matcher.persistence import append_match_log, load_checkpoint, open_cache, save_checkpoint
from matcher.io import dataframe_to_products
from matcher.retrieval import get_or_build_retrieval_index, retrieve_candidates
from matcher.retrieval import attach_embedding_matrix
from matcher.resolution import choose_best_match
from matcher.scoring import HARD_CONTRADICTIONS, blend_scores, deterministic_only_score, score_candidate_pair
from matcher.schemas import MatchDecision

_settings = Settings()


def _llm_choice_is_acceptable(llm_result: dict, chosen_pair) -> bool:
    if llm_result["confidence"] < _settings.medium_quality_threshold:
        return False
    contradictions = set(llm_result.get("contradictions", [])) | set(chosen_pair.contradiction_flags)
    if contradictions & HARD_CONTRADICTIONS:
        return False
    return True


def _append_decision_log(output_dir: str, decision) -> None:
    append_match_log(
        Path(output_dir) / "match_logs.jsonl",
        {
            "item_id_a": decision.item_id_a,
            "item_id_b": decision.item_id_b,
            "confidence": decision.confidence,
            "match_quality": decision.match_quality,
            "decision_source": decision.decision_source,
            "review_flag": decision.review_flag,
        },
    )


def _fallback_decision(item_a, products_b):
    item_b = products_b[0]
    return choose_best_match(
        [
            score_candidate_pair(item_a, item_b).model_copy(
                update={"combined_score": 0.0, "reason_codes": ["processing_failure_fallback"]}
            )
        ]
    )


def _with_retries(fn, attempts: int):
    last_exc = None
    for _ in range(max(attempts, 1)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - pipeline should preserve one output row per A item.
            last_exc = exc
    raise last_exc


def _score_unmatched_item(
    item_a,
    products_b,
    index,
    client,
    cache,
    model: str,
    llm_enabled: bool,
    retrieval_k: int,
    llm_top_n: int,
    llm_min_deterministic: float,
    query_embedding=None,
):
    candidates = retrieve_candidates(item_a, products_b, index=index, top_k=retrieval_k, query_embedding=query_embedding)
    candidates_by_id = {candidate.item_id: candidate for candidate in candidates}
    scored = []
    forced_item_id_b = None
    for item_b in candidates:
        pair = score_candidate_pair(item_a, item_b)
        pair.combined_score = deterministic_only_score(pair.deterministic_score)
        scored.append(pair)

    scored.sort(key=lambda item: item.combined_score or 0.0, reverse=True)
    llm_candidates = (
        scored[:llm_top_n]
        if llm_enabled and scored and (scored[0].combined_score or 0.0) >= llm_min_deterministic
        else []
    )

    if llm_enabled and llm_candidates:
        shortlist_items = [candidates_by_id[pair.item_id_b] for pair in llm_candidates]
        llm_result = score_shortlist_with_llm_cached(client, cache, model, item_a, shortlist_items)
        chosen_item_id_b = llm_result["chosen_item_id_b"] or shortlist_items[0].item_id
        for pair in llm_candidates:
            if pair.item_id_b != chosen_item_id_b:
                continue
            pair.llm_score = llm_result["confidence"]
            pair.reason_codes.extend(llm_result["reason_codes"])
            pair.contradiction_flags.extend(llm_result["contradictions"])
            pair.combined_score = blend_scores(
                pair.deterministic_score,
                llm_result["confidence"],
                max(llm_result["exact_match_score"], llm_result["substitute_match_score"]),
            )
            if _llm_choice_is_acceptable(llm_result, pair):
                forced_item_id_b = chosen_item_id_b
            break

    return choose_best_match(scored, forced_item_id_b=forced_item_id_b)


def run_pipeline(
    rows_a,
    rows_b,
    llm_enabled: bool = False,
    output_dir: str = "artifacts",
    retrieval_k: int = _settings.retrieval_k,
    llm_top_n: int = _settings.llm_top_n,
    llm_min_deterministic: float = _settings.llm_min_deterministic,
    embedding_model: str = _settings.embedding_model,
    embedding_batch_size: int = _settings.embedding_batch_size,
    max_workers: int = _settings.max_workers,
    item_retry_attempts: int = _settings.item_retry_attempts,
    checkpoint_every: int = 100,
):
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    exact_index_b = build_global_id_index(products_b)
    provider_index_b = build_provider_id_index(products_b)
    model = os.environ.get("OPENAI_MODEL", _settings.llm_model)
    checkpoint_every = max(checkpoint_every, 1)
    checkpoint_path = Path(output_dir) / "pipeline-checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    item_ids_a = [item.item_id for item in products_a]
    decisions = [None] * len(products_a)
    if checkpoint and checkpoint.get("item_ids_a") == item_ids_a:
        saved_decisions = checkpoint.get("decisions", [])
        for position, saved_decision in enumerate(saved_decisions[: len(decisions)]):
            if saved_decision is not None:
                decisions[position] = MatchDecision.model_validate(saved_decision)
        restored_count = sum(decision is not None for decision in decisions)
        print(f"Resumed {restored_count}/{len(decisions)} decisions from {checkpoint_path}")
    elif checkpoint:
        print(f"Ignoring checkpoint with different input rows: {checkpoint_path}")

    completed = sum(decision is not None for decision in decisions)
    next_progress_log = ((completed // checkpoint_every) + 1) * checkpoint_every

    def record_progress(position, decision) -> None:
        nonlocal completed, next_progress_log
        if decisions[position] is not None:
            return
        decisions[position] = decision
        completed += 1
        if completed >= next_progress_log:
            save_checkpoint(
                checkpoint_path,
                {
                    "item_ids_a": item_ids_a,
                    "decisions": [decision.model_dump() if decision is not None else None for decision in decisions],
                },
            )
            print(f"Progress: {completed}/{len(decisions)} decisions complete; checkpoint saved to {checkpoint_path}")
            next_progress_log = ((completed // checkpoint_every) + 1) * checkpoint_every

    unmatched = []

    for position, item_a in enumerate(products_a):
        if decisions[position] is not None:
            continue
        exact_decision = choose_exact_global_id_match(item_a, exact_index_b)
        if exact_decision is not None:
            record_progress(position, exact_decision)
            continue

        provider_decision = choose_provider_id_match(item_a, provider_index_b)
        if provider_decision is not None:
            record_progress(position, provider_decision)
            continue

        unmatched.append((position, item_a))

    if unmatched:
        retrieval_index_path = _settings.retrieval_index_path or Path(output_dir) / "retrieval-index-b.pkl"
        index = get_or_build_retrieval_index(products_b, retrieval_index_path)
        client = build_openai_client() if llm_enabled else None
        cache = open_cache(str(Path(output_dir) / "cache")) if llm_enabled else None
        query_embeddings = {}
        if embedding_model:
            local_embedding_model = load_local_embedding_model(embedding_model)
            embeddings_path = embedding_cache_path(output_dir, embedding_model, index["dataset_signature"])
            embeddings_b = load_embedding_cache(embeddings_path, products_b)
            if embeddings_b is None:
                embeddings_b = local_embedding_model.embed_products(products_b, batch_size=embedding_batch_size)
                save_embedding_cache(embeddings_path, embeddings_b, products_b)
            attach_embedding_matrix(index, embeddings_b)
            unmatched_products = [item_a for _, item_a in unmatched]
            query_embeddings = local_embedding_model.embed_products(unmatched_products, batch_size=embedding_batch_size)

        def process(item_a):
            try:
                return _with_retries(
                    lambda: _score_unmatched_item(
                        item_a,
                        products_b,
                        index,
                        client,
                        cache,
                        model,
                        llm_enabled,
                        retrieval_k,
                        llm_top_n,
                        llm_min_deterministic,
                        query_embeddings.get(item_a.item_id),
                    ),
                    item_retry_attempts,
                )
            except Exception:  # noqa: BLE001 - preserve deliverable coverage on unexpected row failures.
                return _fallback_decision(item_a, products_b)

        if max_workers > 1 and len(unmatched) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for (position, _), decision in zip(unmatched, executor.map(lambda pair: process(pair[1]), unmatched)):
                    record_progress(position, decision)
        else:
            for position, item_a in unmatched:
                record_progress(position, process(item_a))

    save_checkpoint(
        checkpoint_path,
        {
            "item_ids_a": item_ids_a,
            "decisions": [decision.model_dump() if decision is not None else None for decision in decisions],
        },
    )
    print(f"Completed {completed}/{len(decisions)} decisions; checkpoint saved to {checkpoint_path}")

    for decision in decisions:
        _append_decision_log(output_dir, decision)

    return decisions


def summarize_decisions(decisions):
    quality_counts = Counter(decision.match_quality for decision in decisions)
    source_counts = Counter(decision.decision_source for decision in decisions)
    return {
        "total_matches": len(decisions),
        "by_quality": dict(quality_counts),
        "by_source": dict(source_counts),
    }
