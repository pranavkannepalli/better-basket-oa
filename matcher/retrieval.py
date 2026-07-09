from collections import defaultdict

from rapidfuzz.fuzz import token_set_ratio


def _category_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)


def build_retrieval_index(items_b):
    by_brand = defaultdict(list)
    by_unit = defaultdict(list)

    for item in items_b:
        if item.brand_norm:
            by_brand[item.brand_norm].append(item)
        if item.size_unit:
            by_unit[item.size_unit].append(item)

    return {"by_brand": by_brand, "by_unit": by_unit}


def retrieve_candidates(item_a, items_b, index=None, top_k: int = 200):
    search_space = list(items_b)
    if index:
        if item_a.brand_norm in index["by_brand"]:
            search_space.extend(index["by_brand"][item_a.brand_norm])
        if item_a.size_unit in index["by_unit"]:
            search_space.extend(index["by_unit"][item_a.size_unit])

    deduped = {item.item_id: item for item in search_space}.values()
    scored = []
    query = " ".join(item_a.tokens_core)
    for item_b in deduped:
        text_score = token_set_ratio(query, " ".join(item_b.tokens_core)) / 100.0
        cat_score = _category_overlap(item_a.category_path, item_b.category_path)
        brand_score = 1.0 if item_a.brand_norm and item_a.brand_norm == item_b.brand_norm else 0.0
        size_score = (
            1.0
            if item_a.size_value
            and item_b.size_value
            and item_a.size_value == item_b.size_value
            and item_a.size_unit == item_b.size_unit
            else 0.0
        )
        score = (0.5 * text_score) + (0.2 * cat_score) + (0.2 * brand_score) + (0.1 * size_score)
        scored.append((score, item_b))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]
