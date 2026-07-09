from matcher.retrieval import retrieve_candidates
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
