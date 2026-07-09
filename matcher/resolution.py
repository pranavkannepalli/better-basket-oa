from matcher.schemas import MatchDecision


def choose_best_match(candidates):
    best = sorted(candidates, key=lambda item: item.combined_score or 0.0, reverse=True)[0]
    confidence = best.combined_score or 0.0
    if confidence >= 0.85:
        match_quality = "high"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    elif confidence >= 0.55:
        match_quality = "medium"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    else:
        match_quality = "low"
        decision_source = "fallback"
        review_flag = True
    return MatchDecision(
        item_id_a=best.item_id_a,
        item_id_b=best.item_id_b,
        confidence=confidence,
        match_quality=match_quality,
        decision_source=decision_source,
        review_flag=review_flag,
    )
