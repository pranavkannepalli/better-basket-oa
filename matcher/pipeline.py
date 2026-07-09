from matcher.io import dataframe_to_products
from matcher.retrieval import build_retrieval_index, retrieve_candidates
from matcher.resolution import choose_best_match
from matcher.scoring import blend_scores, score_candidate_pair


def run_pipeline(rows_a, rows_b, llm_enabled: bool = False):
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    index = build_retrieval_index(products_b)
    decisions = []

    for item_a in products_a:
        candidates = retrieve_candidates(item_a, products_b, index=index, top_k=25)
        scored = []
        for item_b in candidates:
            pair = score_candidate_pair(item_a, item_b)
            llm_score = pair.llm_score or 0.0
            pair.combined_score = blend_scores(
                pair.deterministic_score,
                llm_score if llm_enabled else 0.0,
                llm_score,
            )
            scored.append(pair)
        decisions.append(choose_best_match(scored))

    return decisions
