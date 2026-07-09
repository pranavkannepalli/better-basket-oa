from pathlib import Path

from matcher.persistence import append_match_log
from matcher.pipeline import run_pipeline, summarize_decisions
from matcher.io import write_matches_csv
from matcher.schemas import MatchDecision


def test_run_pipeline_returns_one_match_per_input_row(tmp_path):
    rows_a = [
        {
            "item_id": "a1",
            "name": "Organic Tomato Sauce 8 oz",
            "brand_raw": "Great Value",
            "description": "",
            "item_info": '{"category_0":"Grocery","category_1":"Sauces"}',
            "sizing_comp": '{"size_user_friendly":"8 oz"}',
            "tags": "[]",
        },
        {
            "item_id": "a2",
            "name": "Dog Food",
            "brand_raw": "Pedigree",
            "description": "",
            "item_info": '{"category_0":"Pets"}',
            "sizing_comp": '{}',
            "tags": "[]",
        },
    ]
    rows_b = [
        {
            "item_id": "b1",
            "name": "Wegmans Organic Tomato Sauce 8 oz",
            "brand_raw": "Wegmans",
            "description": "",
            "item_info": '{"category_0":"Grocery","category_1":"Sauces"}',
            "sizing_comp": '{"size_user_friendly":"8 oz"}',
            "tags": "[]",
        },
        {
            "item_id": "b2",
            "name": "Pedigree Dog Food",
            "brand_raw": "Pedigree",
            "description": "",
            "item_info": '{"category_0":"Pets"}',
            "sizing_comp": '{}',
            "tags": "[]",
        },
    ]
    matches = run_pipeline(rows_a, rows_b, llm_enabled=False, output_dir=str(tmp_path))
    assert len(matches) == 2
    assert {match.item_id_a for match in matches} == {"a1", "a2"}


def test_write_matches_csv_outputs_expected_columns(tmp_path: Path):
    path = tmp_path / "matches.csv"
    decisions = [
        MatchDecision(
            item_id_a="a1",
            item_id_b="b1",
            confidence=0.92,
            match_quality="high",
            decision_source="llm",
            review_flag=False,
        )
    ]
    write_matches_csv(path, decisions)
    content = path.read_text()
    assert "item_id_a,item_id_b,confidence,match_quality,decision_source,review_flag" in content


def test_append_match_log_writes_jsonl(tmp_path: Path):
    path = tmp_path / "match_logs.jsonl"
    append_match_log(path, {"item_id_a": "a1", "item_id_b": "b1", "combined_score": 0.91})
    content = path.read_text().strip()
    assert '"item_id_a":"a1"' in content


def test_summarize_decisions_counts_quality_bands():
    decisions = [
        MatchDecision(
            item_id_a="a1",
            item_id_b="b1",
            confidence=0.91,
            match_quality="high",
            decision_source="llm",
            review_flag=False,
        ),
        MatchDecision(
            item_id_a="a2",
            item_id_b="b2",
            confidence=0.42,
            match_quality="low",
            decision_source="fallback",
            review_flag=True,
        ),
    ]
    summary = summarize_decisions(decisions)
    assert summary["total_matches"] == 2
    assert summary["by_quality"]["high"] == 1
    assert summary["by_quality"]["low"] == 1
