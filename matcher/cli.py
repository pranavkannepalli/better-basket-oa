"""Command-line entry point for the matcher package."""

from pathlib import Path

from matcher.io import load_catalog_csv, write_matches_csv
from matcher.pipeline import run_pipeline


def main() -> int:
    input_a = load_catalog_csv("grocery_store_a_items_final.csv").to_dict("records")
    input_b = load_catalog_csv("grocery_store_b_items_final.csv").to_dict("records")
    decisions = run_pipeline(input_a, input_b, llm_enabled=True)

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matches_csv(output_dir / "matches.csv", decisions)
    print(f"matches={len(decisions)}")
    return 0
