from matcher.scoring import score_candidate_pair
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
