"""Command-line entry point for the matcher package."""

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

from matcher.config import Settings
from matcher.io import (
    append_detailed_submission_csv,
    append_matches_csv,
    append_submission_csv,
    load_catalog_csv,
    write_detailed_submission_csv,
    write_matches_csv,
    write_submission_csv,
)
from matcher.pipeline import run_pipeline, summarize_decisions


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Match Store A products to Store B products.")
    parser.add_argument("--input-a", default=str(settings.input_a_path))
    parser.add_argument("--input-b", default=str(settings.input_b_path))
    parser.add_argument("--output-dir", default=str(settings.output_dir))
    parser.add_argument("--limit-a", type=int, default=None)
    parser.add_argument("--llm", dest="llm_enabled", action="store_true", default=True)
    parser.add_argument("--no-llm", dest="llm_enabled", action="store_false")
    parser.add_argument("--max-workers", type=int, default=settings.max_workers)
    parser.add_argument("--worker-mode", choices=["thread", "process"], default="thread")
    parser.add_argument("--process-start-method", choices=["auto", "fork", "spawn"], default="auto")
    parser.add_argument("--item-retry-attempts", type=int, default=settings.item_retry_attempts)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--retrieval-k", type=int, default=settings.retrieval_k)
    parser.add_argument("--llm-top-n", type=int, default=settings.llm_top_n)
    parser.add_argument("--llm-min-deterministic", type=float, default=settings.llm_min_deterministic)
    parser.add_argument("--llm-backlog-limit", type=int, default=settings.llm_backlog_limit)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--embedding-batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--no-embeddings", dest="embedding_model", action="store_const", const="")
    parser.add_argument("--min-confidence", type=float, default=settings.min_confidence)
    return parser


def _load_catalog_frame(path: str, limit: int | None = None):
    started = time.perf_counter()
    print(f"Loading input: {path}")
    frame = load_catalog_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    print(f"Loaded {len(frame)} rows from {path} in {time.perf_counter() - started:.1f}s")
    return frame


def _write_outputs_atomic(
    output_dir: Path,
    decisions,
    products_a_by_id,
    products_b_by_id,
    min_confidence: float,
) -> None:
    completed_decisions = [decision for decision in decisions if decision is not None]
    tmp_submission_path = output_dir / "matches.csv.tmp"
    tmp_detailed_submission_path = output_dir / "matches_detailed.csv.tmp"
    tmp_decisions_path = output_dir / "match_decisions.csv.tmp"
    write_submission_csv(tmp_submission_path, completed_decisions, min_confidence=min_confidence)
    write_detailed_submission_csv(
        tmp_detailed_submission_path,
        completed_decisions,
        products_a_by_id,
        products_b_by_id,
        min_confidence=min_confidence,
    )
    write_matches_csv(tmp_decisions_path, completed_decisions, products_a_by_id, products_b_by_id)
    tmp_submission_path.replace(output_dir / "matches.csv")
    tmp_detailed_submission_path.replace(output_dir / "matches_detailed.csv")
    tmp_decisions_path.replace(output_dir / "match_decisions.csv")


class CheckpointCsvWriter:
    def __init__(self, output_dir: Path, min_confidence: float) -> None:
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.written_positions: set[int] = set()
        self.products_a_by_id = None
        self.products_b_by_id = None
        self.decisions_path = output_dir / "checkpoint_match_decisions.csv"
        self.submission_path = output_dir / "checkpoint_matches.csv"
        self.detailed_path = output_dir / "checkpoint_matches_detailed.csv"
        self.written_item_ids_a = self._load_written_item_ids()

    def _load_written_item_ids(self) -> set[str]:
        if not self.decisions_path.exists():
            return set()
        with self.decisions_path.open(newline="", encoding="utf-8") as handle:
            return {row["item_id_a"] for row in csv.DictReader(handle) if row.get("item_id_a")}

    def write_new(self, decisions, products_a_by_id, products_b_by_id) -> None:
        self.products_a_by_id = products_a_by_id
        self.products_b_by_id = products_b_by_id

        def new_decisions():
            for position, decision in enumerate(decisions):
                if (
                    decision is not None
                    and position not in self.written_positions
                    and decision.item_id_a not in self.written_item_ids_a
                ):
                    yield decision

        append_submission_csv(self.submission_path, new_decisions(), self.min_confidence)
        append_detailed_submission_csv(
            self.detailed_path,
            new_decisions(),
            products_a_by_id,
            products_b_by_id,
            self.min_confidence,
        )
        append_matches_csv(self.decisions_path, new_decisions(), products_a_by_id, products_b_by_id)
        for position, decision in enumerate(decisions):
            if decision is not None:
                self.written_positions.add(position)
                self.written_item_ids_a.add(decision.item_id_a)

    def write_final(self, decisions) -> None:
        if self.products_a_by_id is None or self.products_b_by_id is None:
            raise RuntimeError("Cannot write final CSV outputs before checkpoint products are available")
        _write_outputs_atomic(
            self.output_dir,
            decisions,
            self.products_a_by_id,
            self.products_b_by_id,
            self.min_confidence,
        )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"Writing outputs to {output_dir}")
            started = time.perf_counter()
            checkpoint_csv_writer = CheckpointCsvWriter(output_dir, args.min_confidence)
            progress_callback = checkpoint_csv_writer.write_new
            decisions = run_pipeline(
                _load_catalog_frame(args.input_a, args.limit_a),
                _load_catalog_frame(args.input_b),
                llm_enabled=args.llm_enabled,
                output_dir=args.output_dir,
                retrieval_k=args.retrieval_k,
                llm_top_n=args.llm_top_n,
                llm_min_deterministic=args.llm_min_deterministic,
                llm_backlog_limit=args.llm_backlog_limit,
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                max_workers=args.max_workers,
                worker_mode=args.worker_mode,
                process_start_method=args.process_start_method,
                item_retry_attempts=args.item_retry_attempts,
                checkpoint_every=args.checkpoint_every,
                progress_callback=progress_callback,
            )

            print("Writing final CSV outputs")
            checkpoint_csv_writer.write_final(decisions)
            print(summarize_decisions(decisions))
            print(f"Run finished in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
