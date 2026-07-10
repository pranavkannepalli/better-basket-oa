from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import gc
import hashlib
import multiprocessing
import os
import platform
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
    _log(f"{message} in {time.perf_counter() - started:.1f}s")


def _log(message: str) -> None:
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024
    return None


def _item_ids_digest(item_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for item_id in item_ids:
        digest.update(item_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_matches(checkpoint: dict, item_ids: list[str], item_ids_digest: str) -> bool:
    if checkpoint.get("item_ids_a") == item_ids:
        return True
    return checkpoint.get("item_count") == len(item_ids) and checkpoint.get("item_ids_digest") == item_ids_digest


def _checkpoint_payload(item_ids: list[str], item_ids_digest: str, decisions: list[MatchDecision | None]) -> dict:
    return {
        "version": 2,
        "item_count": len(item_ids),
        "item_ids_digest": item_ids_digest,
        "decisions_by_position": {
            str(position): decision.model_dump()
            for position, decision in enumerate(decisions)
            if decision is not None
        },
    }


def _resolve_process_start_method(requested: str) -> str:
    if requested != "auto":
        method = requested
    elif platform.system() == "Darwin":
        method = "spawn"
    else:
        method = "fork"
    if method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"Process start method {method!r} is not available on this platform")
    return method


def _log_embedding_progress(label: str, completed: int, total: int, elapsed: float) -> None:
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0 else 0.0
    _log(
        f"{label}: {completed}/{total} products "
        f"({rate:.1f}/s, elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)})"
    )


def _log_embedding_batch_start(label: str, start: int, end: int, total: int) -> None:
    _log(f"{label}: embedding products {start + 1}-{end}/{total}")


def _worker_log(state: dict, message: str) -> None:
    line = f"[process-worker pid={os.getpid()}] {message}"
    _log(line)
    worker_log_path = state.get("worker_log_path")
    if worker_log_path:
        log_path = Path(worker_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def _init_process_worker(state: dict) -> None:
    global _PROCESS_WORKER_STATE
    _PROCESS_WORKER_STATE = state
    started = time.perf_counter()
    embedding_model_name = state.get("embedding_model_name")
    if not embedding_model_name:
        _worker_log(state, "ready: no embedding model needed")
        return

    _worker_log(state, f"loading embedding model: {embedding_model_name}")
    local_embedding_model = load_local_embedding_model(embedding_model_name)
    _PROCESS_WORKER_STATE["local_embedding_model"] = local_embedding_model
    embeddings_path = state.get("embedding_cache_path")
    if not embeddings_path:
        _worker_log(state, f"ready: embedding model loaded in {time.perf_counter() - started:.1f}s")
        return

    cache_started = time.perf_counter()
    _worker_log(state, f"loading embedding cache: {embeddings_path}")
    embeddings = load_embedding_cache(Path(embeddings_path), state["products_b"])
    if embeddings is None:
        _worker_log(state, "embedding cache miss; building worker-local B embeddings")
        embeddings = local_embedding_model.embed_products(
            state["products_b"],
            batch_size=state.get("embedding_batch_size", _settings.embedding_batch_size),
            progress_callback=lambda completed, total, elapsed: _worker_log(
                state,
                (
                    f"worker B embedding progress: {completed}/{total} "
                    f"({completed / elapsed if elapsed > 0 else 0.0:.1f}/s, "
                    f"elapsed {_format_duration(elapsed)})"
                ),
            ),
            batch_start_callback=lambda start, end, total: _worker_log(
                state,
                f"worker B embedding batch: products {start + 1}-{end}/{total}",
            ),
            progress_interval=max(state.get("embedding_batch_size", _settings.embedding_batch_size) * 5, 500),
        )
    else:
        _worker_log(
            state,
            f"loaded embedding cache: shape={embeddings.shape} in {time.perf_counter() - cache_started:.1f}s",
        )
    attach_started = time.perf_counter()
    _worker_log(state, "attaching worker-local embedding matrix")
    attach_embedding_matrix(state["index"], embeddings)
    _worker_log(
        state,
        (
            f"ready: backend={state['index'].get('embedding_backend')} "
            f"attach={time.perf_counter() - attach_started:.1f}s total={time.perf_counter() - started:.1f}s"
        ),
    )


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


def _score_deterministic_candidates(item_a, products_b, index, retrieval_k: int, query_embedding=None):
    candidates = retrieve_candidates(item_a, products_b, index=index, top_k=retrieval_k, query_embedding=query_embedding)
    candidates_by_id = {candidate.item_id: candidate for candidate in candidates}
    scored = []
    for item_b in candidates:
        pair = score_candidate_pair(item_a, item_b)
        pair.combined_score = deterministic_only_score(pair.deterministic_score)
        scored.append(pair)
    scored.sort(key=lambda item: item.combined_score or 0.0, reverse=True)
    return scored, candidates_by_id


def _finish_scored_item_with_llm(
    item_a,
    scored,
    candidates_by_id,
    client,
    cache,
    model: str,
    llm_enabled: bool,
    llm_top_n: int,
    llm_min_deterministic: float,
):
    forced_item_id_b = None
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


def _score_unmatched_item_process(payload):
    position, item_a, query_embedding = payload
    started_cpu = time.process_time()
    try:
        local_embedding_model = _PROCESS_WORKER_STATE.get("local_embedding_model")
        if query_embedding is None and local_embedding_model is not None:
            query_embedding = local_embedding_model.embed_products([item_a], batch_size=1)[0]
        llm_enabled = _PROCESS_WORKER_STATE["llm_enabled"]
        scored, candidates_by_id = _with_retries(
            lambda: _score_deterministic_candidates(
                item_a,
                _PROCESS_WORKER_STATE["products_b"],
                _PROCESS_WORKER_STATE["index"],
                _PROCESS_WORKER_STATE["retrieval_k"],
                query_embedding,
            ),
            _PROCESS_WORKER_STATE["item_retry_attempts"],
        )
        best_rule_score = (scored[0].combined_score or 0.0) if scored else 0.0
        if llm_enabled and scored and best_rule_score >= _PROCESS_WORKER_STATE["llm_min_deterministic"]:
            return (
                "llm",
                position,
                item_a,
                scored,
                candidates_by_id,
                os.getpid(),
                time.process_time() - started_cpu,
            )
        decision = choose_best_match(scored)
    except Exception:  # noqa: BLE001 - preserve deliverable coverage on unexpected row failures.
        decision = _fallback_decision(item_a, _PROCESS_WORKER_STATE["products_b"])
    return ("decision", position, decision, None, None, os.getpid(), time.process_time() - started_cpu)


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
    scored, candidates_by_id = _score_deterministic_candidates(
        item_a,
        products_b,
        index,
        retrieval_k,
        query_embedding,
    )
    return _finish_scored_item_with_llm(
        item_a,
        scored,
        candidates_by_id,
        client,
        cache,
        model,
        llm_enabled,
        llm_top_n,
        llm_min_deterministic,
    )


def run_pipeline(
    rows_a,
    rows_b,
    llm_enabled: bool = False,
    output_dir: str = "artifacts",
    retrieval_k: int = _settings.retrieval_k,
    llm_top_n: int = _settings.llm_top_n,
    llm_min_deterministic: float = _settings.llm_min_deterministic,
    llm_backlog_limit: int = _settings.llm_backlog_limit,
    llm_workers: int = _settings.llm_workers,
    embedding_model: str = _settings.embedding_model,
    embedding_batch_size: int = _settings.embedding_batch_size,
    max_workers: int = _settings.max_workers,
    worker_mode: str = "thread",
    process_start_method: str = "auto",
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
    item_ids_a_digest = _item_ids_digest(item_ids_a)
    decisions = [None] * len(products_a)
    if checkpoint and _checkpoint_matches(checkpoint, item_ids_a, item_ids_a_digest):
        if "decisions_by_position" in checkpoint:
            for position_text, saved_decision in checkpoint.get("decisions_by_position", {}).items():
                position = int(position_text)
                if 0 <= position < len(decisions):
                    decisions[position] = MatchDecision.model_validate(saved_decision)
        else:
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
    pending_llm_finishes = set()
    use_process_pool = False
    resolved_process_start_method = None

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
            save_checkpoint(checkpoint_path, _checkpoint_payload(item_ids_a, item_ids_a_digest, decisions))
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
            rss_mb = _current_rss_mb()
            rss_text = f", parent RSS={rss_mb:.0f} MB" if rss_mb is not None else ""
            print(
                f"Progress: {completed}/{len(decisions)} decisions complete "
                f"({rate:.2f}/s, elapsed {_format_duration(elapsed)}, "
                f"ETA {_format_duration(eta)}, CPU {_format_duration(cpu_elapsed)}, "
                f"effective cores={effective_cores:.1f}, workers configured={max_workers}, "
                f"workers observed={len(scoring_worker_ids)}, pending LLM={len(pending_llm_finishes)}"
                f"{rss_text}); checkpoint saved to {checkpoint_path}"
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
        use_process_pool = worker_mode == "process" and max_workers > 1 and len(unmatched) > 1
        if use_process_pool:
            resolved_process_start_method = _resolve_process_start_method(process_start_method)
            _log(f"Process worker start method: {resolved_process_start_method}")
            if llm_enabled:
                _log(
                    "Hybrid mode active: process workers do deterministic retrieval/scoring; "
                    "parent threads finish LLM rows"
                )
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
        embeddings_path = None
        if embedding_model:
            spawn_process_embeddings = use_process_pool and resolved_process_start_method == "spawn"
            embeddings_path = embedding_cache_path(output_dir, embedding_model, index["dataset_signature"])
            if not spawn_process_embeddings:
                started = time.perf_counter()
                print(f"Loading embedding model: {embedding_model}")
                local_embedding_model = load_local_embedding_model(embedding_model)
                _log_elapsed("Embedding model ready", started)

            started = time.perf_counter()
            print(f"Loading embedding cache: {embeddings_path}")
            print("Embedding cache is scoped to this output directory")
            embeddings_b = load_embedding_cache(embeddings_path, products_b)
            if embeddings_b is None:
                _log_elapsed("Embedding cache miss", started)
                progress_interval = max(embedding_batch_size * 5, 500)
                _log(
                    f"Cold B-embedding build started: {len(products_b)} products, "
                    f"batch_size={embedding_batch_size}, progress_every={progress_interval}"
                )
                if local_embedding_model is None:
                    started = time.perf_counter()
                    print(f"Loading embedding model: {embedding_model}")
                    local_embedding_model = load_local_embedding_model(embedding_model)
                    _log_elapsed("Embedding model ready", started)
                started = time.perf_counter()
                print(f"Building embeddings for {len(products_b)} B products")
                embeddings_b = local_embedding_model.embed_products(
                    products_b,
                    batch_size=embedding_batch_size,
                    progress_callback=lambda completed, total, elapsed: _log_embedding_progress(
                        "B embedding build progress",
                        completed,
                        total,
                        elapsed,
                    ),
                    batch_start_callback=lambda start, end, total: _log_embedding_batch_start(
                        "B embedding batch",
                        start,
                        end,
                        total,
                    ),
                    progress_interval=progress_interval,
                )
                _log_elapsed("Built B embeddings", started)
                started = time.perf_counter()
                print(f"Saving embedding cache: {embeddings_path}")
                save_embedding_cache(embeddings_path, embeddings_b, products_b)
                _log_elapsed("Saved embedding cache", started)
            else:
                _log_elapsed(f"Loaded embedding cache: shape={embeddings_b.shape}", started)
            if not spawn_process_embeddings:
                started = time.perf_counter()
                print("Attaching embedding matrix")
                attach_embedding_matrix(index, embeddings_b)
                _log_elapsed(f"Attached embedding matrix: backend={index.get('embedding_backend')}", started)
            else:
                print("Embedding matrix will attach inside spawned process workers")
                local_embedding_model = None
            del embeddings_b
            gc.collect()
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

        llm_finish_executor = None
        resolved_llm_backlog_limit = (
            llm_backlog_limit if llm_backlog_limit > 0 else max(max_workers * 64, max_workers + 1)
        )
        resolved_llm_workers = llm_workers if llm_workers > 0 else max_workers
        resolved_llm_backlog_limit = max(resolved_llm_backlog_limit, resolved_llm_workers)

        def record_worker_activity(worker_id, worker_cpu_seconds) -> None:
            nonlocal process_worker_cpu_seconds
            if worker_id is not None:
                with scoring_worker_lock:
                    scoring_worker_ids.add(worker_id)
            process_worker_cpu_seconds += worker_cpu_seconds

        def finish_llm_item(position, item_a, scored, candidates_by_id):
            decision = _finish_scored_item_with_llm(
                item_a,
                scored,
                candidates_by_id,
                client,
                cache,
                model,
                llm_enabled,
                llm_top_n,
                llm_min_deterministic,
            )
            return position, decision

        def submit_llm_finish(position, item_a, scored, candidates_by_id) -> None:
            if llm_finish_executor is None:
                position, decision = finish_llm_item(position, item_a, scored, candidates_by_id)
                record_progress(position, decision)
                return
            pending_llm_finishes.add(
                llm_finish_executor.submit(finish_llm_item, position, item_a, scored, candidates_by_id)
            )
            drain_llm_finishes(wait_for_capacity=True)

        def drain_llm_finishes(block: bool = False, wait_for_capacity: bool = False) -> None:
            if not pending_llm_finishes:
                return
            if block:
                for future in as_completed(tuple(pending_llm_finishes)):
                    position, decision = future.result()
                    record_progress(position, decision)
                pending_llm_finishes.clear()
                return

            if wait_for_capacity:
                while len(pending_llm_finishes) >= resolved_llm_backlog_limit:
                    completed_futures, _ = wait(pending_llm_finishes, return_when=FIRST_COMPLETED)
                    pending_llm_finishes.difference_update(completed_futures)
                    for future in completed_futures:
                        position, decision = future.result()
                        record_progress(position, decision)

            completed_futures = {future for future in pending_llm_finishes if future.done()}
            pending_llm_finishes.difference_update(completed_futures)
            for future in completed_futures:
                position, decision = future.result()
                record_progress(position, decision)

        def process_batch(batch, executor=None) -> None:
            batch_embeddings = (
                local_embedding_model.embed_products(
                    [item_a for _, item_a in batch], batch_size=embedding_batch_size
                )
                if local_embedding_model is not None and not use_process_pool
                else None
            )

            def process_pair(item_with_embedding):
                (position, item_a), query_embedding = item_with_embedding
                with scoring_worker_lock:
                    scoring_worker_ids.add(threading.get_ident())
                return ("decision", position, process(item_a, query_embedding), None, None, None, 0.0)

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
                    [
                        (position, item_a, query_embedding)
                        for (position, item_a), query_embedding in pairs_with_embeddings
                    ],
                )
            else:
                results = (
                    executor.map(process_pair, pairs_with_embeddings)
                    if executor
                    else map(process_pair, pairs_with_embeddings)
                )
            for result in results:
                kind, position, payload, scored, candidates_by_id, worker_id, worker_cpu_seconds = result
                if kind == "llm":
                    record_worker_activity(worker_id, worker_cpu_seconds)
                    submit_llm_finish(position, payload, scored, candidates_by_id)
                    drain_llm_finishes()
                else:
                    record_progress(position, payload, worker_id, worker_cpu_seconds)
            del batch_embeddings

        batch_size = max(embedding_batch_size, 1)
        print(
            f"Scoring retrieval rows: {len(unmatched)} items, "
            f"batch_size={batch_size}, workers={max_workers}, worker_mode={worker_mode}, "
            f"process_start_method={resolved_process_start_method or 'n/a'}, llm={llm_enabled}, "
            f"llm_workers={resolved_llm_workers if llm_enabled else 0}, "
            f"llm_backlog_limit={resolved_llm_backlog_limit if llm_enabled else 0}"
        )
        if use_process_pool and embedding_model:
            if resolved_process_start_method == "spawn":
                worker_log_path = Path(output_dir) / "process-workers.log"
                _log(
                    "Embedding process mode: spawned workers load the model/cache and build worker-local "
                    f"FAISS; worker setup log: {worker_log_path}"
                )
            else:
                _log("Embedding process mode: fork workers share parent embedding state copy-on-write")
        progress_started = time.perf_counter()
        progress_cpu_started = time.process_time()
        progress_start_completed = completed
        process_worker_cpu_seconds = 0.0
        if max_workers > 1 and len(unmatched) > 1:
            if worker_mode == "process":
                global _PROCESS_WORKER_STATE
                worker_state = {
                    "products_b": products_b,
                    "index": index,
                    "model": model,
                    "llm_enabled": llm_enabled,
                    "retrieval_k": retrieval_k,
                    "llm_top_n": llm_top_n,
                    "llm_min_deterministic": llm_min_deterministic,
                    "item_retry_attempts": item_retry_attempts,
                    "worker_log_path": str(Path(output_dir) / "process-workers.log"),
                }
                if embedding_model:
                    if resolved_process_start_method == "spawn":
                        worker_state["embedding_model_name"] = embedding_model
                        worker_state["embedding_cache_path"] = str(embeddings_path)
                        worker_state["embedding_batch_size"] = embedding_batch_size
                    else:
                        worker_state["local_embedding_model"] = local_embedding_model
                if resolved_process_start_method != "spawn":
                    _PROCESS_WORKER_STATE = worker_state
                context = multiprocessing.get_context(resolved_process_start_method)
                if llm_enabled:
                    llm_finish_executor = ThreadPoolExecutor(max_workers=resolved_llm_workers)
                try:
                    _log(
                        f"Starting process pool: max_workers={max_workers}, "
                        f"start_method={resolved_process_start_method}, "
                        f"llm_finish_threads={resolved_llm_workers if llm_enabled else 0}"
                    )
                    executor_kwargs = {
                        "max_workers": max_workers,
                        "mp_context": context,
                    }
                    if resolved_process_start_method == "spawn":
                        executor_kwargs["initializer"] = _init_process_worker
                        executor_kwargs["initargs"] = (worker_state,)
                    with ProcessPoolExecutor(**executor_kwargs) as executor:
                        for start in range(0, len(unmatched), batch_size):
                            process_batch(unmatched[start : start + batch_size], executor)
                            drain_llm_finishes()
                    drain_llm_finishes(block=True)
                finally:
                    if llm_finish_executor is not None:
                        llm_finish_executor.shutdown(wait=True)
                    _PROCESS_WORKER_STATE = {}
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for start in range(0, len(unmatched), batch_size):
                        process_batch(unmatched[start : start + batch_size], executor)
        else:
            for start in range(0, len(unmatched), batch_size):
                process_batch(unmatched[start : start + batch_size])

    save_checkpoint(
        checkpoint_path, _checkpoint_payload(item_ids_a, item_ids_a_digest, decisions)
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
