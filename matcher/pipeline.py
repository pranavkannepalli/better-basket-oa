from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import multiprocessing
import os
from pathlib import Path
import threading
import time
from typing import Callable

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
from matcher.schemas import MatchDecision, ProductRecord

_settings = Settings()
_PROCESS_WORKER_STATE = {}


def _log_elapsed(message: str, started: float) -> None:
    print(f"{message} in {time.perf_counter() - started:.1f}s")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


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


def _score_unmatched_item_process(payload):
    item_a, query_embedding = payload
    started_cpu = time.process_time()
    try:
        decision = _with_retries(
            lambda: _score_unmatched_item(
                item_a,
                _PROCESS_WORKER_STATE["products_b"],
                _PROCESS_WORKER_STATE["index"],
                None,
                None,
                _PROCESS_WORKER_STATE["model"],
                False,
                _PROCESS_WORKER_STATE["retrieval_k"],
                _PROCESS_WORKER_STATE["llm_top_n"],
                _PROCESS_WORKER_STATE["llm_min_deterministic"],
                query_embedding,
            ),
            _PROCESS_WORKER_STATE["item_retry_attempts"],
        )
    except Exception:  # noqa: BLE001 - preserve deliverable coverage on unexpected row failures.
        decision = _fallback_decision(item_a, _PROCESS_WORKER_STATE["products_b"])
    return decision, os.getpid(), time.process_time() - started_cpu


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
    best_rule_score = (scored[0].combined_score or 0.0) if scored else 0.0
    requires_llm_eval = llm_enabled and best_rule_score >= _settings.medium_quality_threshold
    llm_candidates = (
        scored[:llm_top_n]
        if llm_enabled and scored and best_rule_score >= llm_min_deterministic
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
            if requires_llm_eval or _llm_choice_is_acceptable(llm_result, pair):
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
    worker_mode: str = "thread",
    item_retry_attempts: int = _settings.item_retry_attempts,
    checkpoint_every: int = 100,
    progress_callback: Callable[
        [list[MatchDecision | None], dict[str, ProductRecord], dict[str, ProductRecord]], None
    ]
    | None = None,
):
    started = time.perf_counter()
    print(f"Normalizing products: A={len(rows_a)} rows, B={len(rows_b)} rows")
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    del rows_a, rows_b
    products_a_by_id = {product.item_id: product for product in products_a}
    products_b_by_id = {product.item_id: product for product in products_b}
    _log_elapsed(f"Normalized products: A={len(products_a)}, B={len(products_b)}", started)

    started = time.perf_counter()
    print("Building exact-match indexes")
    exact_index_b = build_global_id_index(products_b)
    provider_index_b = build_provider_id_index(products_b)
    _log_elapsed("Built exact-match indexes", started)

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
    progress_started = time.perf_counter()
    progress_cpu_started = time.process_time()
    process_worker_cpu_seconds = 0.0
    progress_start_completed = completed
    scoring_worker_ids = set()
    scoring_worker_lock = threading.Lock()

    def record_progress(position, decision, worker_id=None, worker_cpu_seconds=0.0) -> None:
        nonlocal completed, next_progress_log, process_worker_cpu_seconds
        if decisions[position] is not None:
            return
        decisions[position] = decision
        completed += 1
        if worker_id is not None:
            with scoring_worker_lock:
                scoring_worker_ids.add(worker_id)
        process_worker_cpu_seconds += worker_cpu_seconds
        if completed >= next_progress_log:
            save_checkpoint(
                checkpoint_path,
                {
                    "item_ids_a": item_ids_a,
                    "decisions": [decision.model_dump() if decision is not None else None for decision in decisions],
                },
            )
            if progress_callback is not None:
                progress_callback(decisions, products_a_by_id, products_b_by_id)
            elapsed = time.perf_counter() - progress_started
            parent_cpu_elapsed = time.process_time() - progress_cpu_started
            cpu_elapsed = (
                process_worker_cpu_seconds + parent_cpu_elapsed
                if worker_mode == "process"
                else parent_cpu_elapsed
            )
            completed_since_start = completed - progress_start_completed
            rate = completed_since_start / elapsed if elapsed > 0 else 0.0
            effective_cores = cpu_elapsed / elapsed if elapsed > 0 else 0.0
            remaining = len(decisions) - completed
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"Progress: {completed}/{len(decisions)} decisions complete "
                f"({rate:.2f}/s, elapsed {_format_duration(elapsed)}, "
                f"ETA {_format_duration(eta)}, CPU {_format_duration(cpu_elapsed)}, "
                f"effective cores={effective_cores:.1f}, workers configured={max_workers}, "
                f"workers observed={len(scoring_worker_ids)}); checkpoint saved to {checkpoint_path}"
            )
            next_progress_log = ((completed // checkpoint_every) + 1) * checkpoint_every

    unmatched = []

    started = time.perf_counter()
    print("Scanning exact/provider matches")
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
    _log_elapsed(
        f"Exact/provider scan complete: {completed} completed, {len(unmatched)} need retrieval",
        started,
    )
    del exact_index_b, provider_index_b
    gc.collect()

    if unmatched:
        retrieval_index_path = _settings.retrieval_index_path or Path(output_dir) / "retrieval-index-b.pkl"
        started = time.perf_counter()
        print(f"Loading/building retrieval index: {retrieval_index_path}")
        index = get_or_build_retrieval_index(products_b, retrieval_index_path)
        matrix = index.get("matrix")
        matrix_info = "no text matrix" if matrix is None else f"text nnz={matrix.nnz}"
        _log_elapsed(f"Retrieval index ready: {len(index.get('items_b', []))} items, {matrix_info}", started)

        started = time.perf_counter()
        print(f"Preparing LLM/cache: enabled={llm_enabled}, model={model}")
        client = build_openai_client() if llm_enabled else None
        cache = open_cache(str(Path(output_dir) / "cache")) if llm_enabled else None
        _log_elapsed("LLM/cache ready", started)

        local_embedding_model = None
        if embedding_model:
            started = time.perf_counter()
            print(f"Loading embedding model: {embedding_model}")
            local_embedding_model = load_local_embedding_model(embedding_model)
            _log_elapsed("Embedding model ready", started)

            embeddings_path = embedding_cache_path(output_dir, embedding_model, index["dataset_signature"])
            started = time.perf_counter()
            print(f"Loading embedding cache: {embeddings_path}")
            embeddings_b = load_embedding_cache(embeddings_path, products_b)
            if embeddings_b is None:
                _log_elapsed("Embedding cache miss", started)
                started = time.perf_counter()
                print(f"Building embeddings for {len(products_b)} B products")
                embeddings_b = local_embedding_model.embed_products(products_b, batch_size=embedding_batch_size)
                _log_elapsed("Built B embeddings", started)
                started = time.perf_counter()
                print(f"Saving embedding cache: {embeddings_path}")
                save_embedding_cache(embeddings_path, embeddings_b, products_b)
                _log_elapsed("Saved embedding cache", started)
            else:
                _log_elapsed(f"Loaded embedding cache: shape={embeddings_b.shape}", started)
            started = time.perf_counter()
            print("Attaching embedding matrix")
            attach_embedding_matrix(index, embeddings_b)
            del embeddings_b
            gc.collect()
            _log_elapsed(f"Attached embedding matrix: backend={index.get('embedding_backend')}", started)
        else:
            print("Embeddings disabled")

        def process(item_a, query_embedding=None):
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
                        query_embedding,
                    ),
                    item_retry_attempts,
                )
            except Exception:  # noqa: BLE001 - preserve deliverable coverage on unexpected row failures.
                return _fallback_decision(item_a, products_b)

        def process_batch(batch, executor=None) -> None:
            batch_embeddings = (
                local_embedding_model.embed_products(
                    [item_a for _, item_a in batch], batch_size=embedding_batch_size
                )
                if local_embedding_model is not None
                else None
            )

            def process_pair(item_with_embedding):
                (_, item_a), query_embedding = item_with_embedding
                with scoring_worker_lock:
                    scoring_worker_ids.add(threading.get_ident())
                return process(item_a, query_embedding), None, 0.0

            pairs_with_embeddings = [
                (
                    (position, item_a),
                    batch_embeddings[offset] if batch_embeddings is not None else None,
                )
                for offset, (position, item_a) in enumerate(batch)
            ]
            if worker_mode == "process" and executor is not None:
                results = executor.map(
                    _score_unmatched_item_process,
                    [(item_a, query_embedding) for (_, item_a), query_embedding in pairs_with_embeddings],
                )
            else:
                results = (
                    executor.map(process_pair, pairs_with_embeddings)
                    if executor
                    else map(process_pair, pairs_with_embeddings)
                )
            for (position, _), (decision, worker_id, worker_cpu_seconds) in zip(batch, results):
                record_progress(position, decision, worker_id, worker_cpu_seconds)
            del batch_embeddings

        batch_size = max(embedding_batch_size, 1)
        if worker_mode == "process" and llm_enabled:
            print("Process worker mode is CPU-only; falling back to thread workers because LLM is enabled")
            worker_mode = "thread"
        print(
            f"Scoring retrieval rows: {len(unmatched)} items, "
            f"batch_size={batch_size}, workers={max_workers}, worker_mode={worker_mode}, llm={llm_enabled}"
        )
        progress_started = time.perf_counter()
        progress_cpu_started = time.process_time()
        progress_start_completed = completed
        process_worker_cpu_seconds = 0.0
        if max_workers > 1 and len(unmatched) > 1:
            if worker_mode == "process":
                global _PROCESS_WORKER_STATE
                _PROCESS_WORKER_STATE = {
                    "products_b": products_b,
                    "index": index,
                    "model": model,
                    "retrieval_k": retrieval_k,
                    "llm_top_n": llm_top_n,
                    "llm_min_deterministic": llm_min_deterministic,
                    "item_retry_attempts": item_retry_attempts,
                }
                context = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    for start in range(0, len(unmatched), batch_size):
                        process_batch(unmatched[start : start + batch_size], executor)
                _PROCESS_WORKER_STATE = {}
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for start in range(0, len(unmatched), batch_size):
                        process_batch(unmatched[start : start + batch_size], executor)
        else:
            for start in range(0, len(unmatched), batch_size):
                process_batch(unmatched[start : start + batch_size])

    save_checkpoint(
        checkpoint_path,
        {
            "item_ids_a": item_ids_a,
            "decisions": [decision.model_dump() if decision is not None else None for decision in decisions],
        },
    )
    if progress_callback is not None:
        progress_callback(decisions, products_a_by_id, products_b_by_id)
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
