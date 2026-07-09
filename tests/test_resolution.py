from matcher.resolution import choose_best_match
from matcher.schemas import CandidateScore


def test_choose_best_match_marks_medium_range_correctly():
    candidates = [
        CandidateScore(item_id_a="a1", item_id_b="b1", deterministic_score=0.4, llm_score=0.7, combined_score=0.64),
        CandidateScore(item_id_a="a1", item_id_b="b2", deterministic_score=0.35, llm_score=0.55, combined_score=0.51),
    ]
    decision = choose_best_match(candidates, high_quality_threshold=0.85, medium_quality_threshold=0.55)
    assert decision.match_quality == "medium"
    assert decision.decision_source == "llm"


def test_choose_best_match_marks_low_quality_for_weak_best_guess():
    candidates = [
        CandidateScore(item_id_a="a1", item_id_b="b1", deterministic_score=0.2, llm_score=0.28, combined_score=0.27),
        CandidateScore(item_id_a="a1", item_id_b="b2", deterministic_score=0.1, llm_score=0.14, combined_score=0.13),
    ]
    decision = choose_best_match(candidates)
    assert decision.item_id_b == "b1"
    assert decision.match_quality == "low"
    assert decision.review_flag is True
