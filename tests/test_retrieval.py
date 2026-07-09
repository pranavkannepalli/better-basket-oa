from matcher.retrieval import build_retrieval_index, retrieve_candidates
from matcher.schemas import ProductRecord


def test_retrieve_candidates_prefers_same_family():
    item_a = ProductRecord(
        item_id="a1",
        name="Organic Tomato Sauce",
        tokens_core=["tomato", "sauce"],
        category_path=["Grocery", "Sauces"],
    )
    items_b = [
        ProductRecord(
            item_id="b1",
            name="Organic Tomato Sauce",
            tokens_core=["tomato", "sauce"],
            category_path=["Grocery", "Sauces"],
        ),
        ProductRecord(
            item_id="b2",
            name="Dog Biscuits",
            tokens_core=["dog", "biscuits"],
            category_path=["Pets"],
        ),
    ]
    candidates = retrieve_candidates(item_a, items_b, top_k=2)
    assert candidates[0].item_id == "b1"


def test_retrieval_uses_brand_and_size_signals():
    items_b = [
        ProductRecord(
            item_id="b1",
            name="Chobani Greek Yogurt Honey",
            brand_norm="chobani",
            tokens_core=["greek", "yogurt", "honey"],
            size_value=5.3,
            size_unit="oz",
        ),
        ProductRecord(
            item_id="b2",
            name="Random Yogurt",
            brand_norm="other",
            tokens_core=["yogurt"],
            size_value=32.0,
            size_unit="oz",
        ),
    ]
    item_a = ProductRecord(
        item_id="a1",
        name="Chobani Whole Milk Greek Yogurt Honey Blended 5.3 oz",
        brand_norm="chobani",
        tokens_core=["greek", "yogurt", "honey"],
        size_value=5.3,
        size_unit="oz",
    )
    index = build_retrieval_index(items_b)
    candidates = retrieve_candidates(item_a, items_b, index=index, top_k=2)
    assert candidates[0].item_id == "b1"
