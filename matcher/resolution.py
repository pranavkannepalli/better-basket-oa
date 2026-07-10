from matcher.schemas import MatchDecision
from matcher.config import Settings


_settings = Settings()


def choose_best_match(
    candidates,
    forced_item_id_b: str | None = None,
    high_quality_threshold: float = _settings.high_quality_threshold,
    medium_quality_threshold: float = _settings.medium_quality_threshold,
):
    if forced_item_id_b is not None:
        best = next(
            (candidate for candidate in candidates if candidate.item_id_b == forced_item_id_b),
            None,
        )
        if best is None:
            best = sorted(candidates, key=lambda item: item.combined_score or 0.0, reverse=True)[0]
    else:
        best = sorted(candidates, key=lambda item: item.combined_score or 0.0, reverse=True)[0]
    confidence = round(best.combined_score or 0.0, 4)
    if confidence >= high_quality_threshold:
        match_quality = "high"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    elif confidence >= medium_quality_threshold:
        match_quality = "medium"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    else:
        match_quality = "low"
        decision_source = "llm" if best.llm_score is not None else "fallback"
        review_flag = True
    return MatchDecision(
        item_id_a=best.item_id_a,
        item_id_b=best.item_id_b,
        confidence=confidence,
        match_quality=match_quality,
        decision_source=decision_source,
        review_flag=review_flag,
    )
