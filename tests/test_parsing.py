from matcher.schemas import ProductRecord, MatchDecision


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
