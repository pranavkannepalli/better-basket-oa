from matcher.parsing import parse_item_info, parse_sizing_comp
from matcher.schemas import MatchDecision, ProductRecord


def test_product_record_defaults():
    product = ProductRecord(item_id="1", name="Tomato Sauce")
    assert product.brand_norm == ""
    assert product.private_label_flag is False


def test_match_decision_requires_ids():
    decision = MatchDecision(
        item_id_a="1",
        item_id_b="2",
        confidence=0.75,
        match_quality="medium",
        decision_source="llm",
        review_flag=False,
    )
    assert decision.item_id_a == "1"
    assert decision.item_id_b == "2"


def test_parse_item_info_extracts_category_path():
    raw = '{"category_0":"Grocery","category_1":"Sauces","category_2":"Tomato Sauce"}'
    assert parse_item_info(raw) == ["Grocery", "Sauces", "Tomato Sauce"]


def test_parse_sizing_comp_extracts_user_friendly_size():
    raw = '{"size_user_friendly":"16 fl. oz.","billed_by_weight":false}'
    parsed = parse_sizing_comp(raw)
    assert parsed["size_user_friendly"] == "16 fl. oz."
