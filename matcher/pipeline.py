from pathlib import Path

from matcher.llm import build_openai_client, score_pair_with_llm
from matcher.persistence import append_match_log
from matcher.io import dataframe_to_products
from matcher.retrieval import build_retrieval_index, retrieve_candidates
from matcher.resolution import choose_best_match
from matcher.scoring import blend_scores, score_candidate_pair


def run_pipeline(rows_a, rows_b, llm_enabled: bool = False, output_dir: str = "artifacts"):
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    index = build_retrieval_index(products_b)
    client = build_openai_client() if llm_enabled else None
    decisions = []

    for item_a in products_a:
        candidates = retrieve_candidates(item_a, products_b, index=index, top_k=25)
        scored = []
        for item_b in candidates:
            pair = score_candidate_pair(item_a, item_b)
            if llm_enabled:
                llm_result = score_pair_with_llm(client, "gpt-5-nano", item_a, item_b)
                pair.llm_score = llm_result["confidence"]
                pair.reason_codes.extend(llm_result["reason_codes"])
                pair.contradiction_flags.extend(llm_result["contradictions"])
                pair.combined_score = blend_scores(
                    pair.deterministic_score,
                    llm_result["confidence"],
                    llm_result["substitute_match_score"],
                )
            else:
                pair.combined_score = blend_scores(pair.deterministic_score, 0.0, 0.0)
            scored.append(pair)
        decision = choose_best_match(scored)
        append_match_log(
            Path(output_dir) / "match_logs.jsonl",
            {
                "item_id_a": decision.item_id_a,
                "item_id_b": decision.item_id_b,
                "confidence": decision.confidence,
                "match_quality": decision.match_quality,
                "decision_source": decision.decision_source,
                "review_flag": decision.review_flag,
            },
        )
        decisions.append(decision)

    return decisions
