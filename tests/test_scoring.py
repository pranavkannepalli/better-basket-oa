import pytest

from matcher.scoring import blend_scores
from matcher.scoring import score_candidate_pair
from matcher.llm import build_pair_prompt
from matcher.llm import parse_llm_response
from matcher.schemas import ProductRecord


def test_score_candidate_pair_rewards_family_match():
    item_a = ProductRecord(
        item_id="a1",
        name="Organic Tomato Sauce",
        tokens_core=["tomato", "sauce"],
        category_path=["Grocery"],
        attribute_flags=["organic"],
    )
    item_b = ProductRecord(
        item_id="b1",
        name="Wegmans Organic Tomato Sauce",
        tokens_core=["tomato", "sauce"],
        category_path=["Grocery"],
        attribute_flags=["organic"],
    )
    score = score_candidate_pair(item_a, item_b)
    assert score.deterministic_score > 0.7


def test_score_candidate_pair_penalizes_hard_conflict():
    item_a = ProductRecord(
        item_id="a1",
        name="Dog Food",
        tokens_core=["dog", "food"],
        category_path=["Pets"],
    )
    item_b = ProductRecord(
        item_id="b1",
        name="Tomato Sauce",
        tokens_core=["tomato", "sauce"],
        category_path=["Grocery"],
    )
    score = score_candidate_pair(item_a, item_b)
    assert "category_conflict" in score.contradiction_flags


def test_build_pair_prompt_includes_structured_fields():
    item_a = ProductRecord(item_id="a1", name="Organic Tomato Sauce", brand_norm="great value", category_path=["Grocery"], size_value=8.0, size_unit="oz")
    item_b = ProductRecord(item_id="b1", name="Wegmans Organic Tomato Sauce", brand_norm="wegmans", category_path=["Grocery"], size_value=8.0, size_unit="oz")
    prompt = build_pair_prompt(item_a, item_b)
    assert '"item_id":"a1"' in prompt
    assert '"item_id":"b1"' in prompt
    assert '"size_unit":"oz"' in prompt


def test_parse_llm_response_reads_confidence():
    raw = '{"exact_match_score":0.2,"substitute_match_score":0.9,"confidence":0.86,"reason_codes":["private_label_match"],"contradictions":[]}'
    parsed = parse_llm_response(raw)
    assert parsed["confidence"] == 0.86
    assert parsed["substitute_match_score"] == 0.9


def test_blend_scores_uses_weighted_average():
    blended = blend_scores(0.5, 0.8, 0.9)
    assert blended == pytest.approx(0.74)
