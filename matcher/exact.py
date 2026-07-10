from collections import defaultdict

from matcher.schemas import MatchDecision, ProductRecord

PROVIDER_MATCH_KEYS = {"ic_item_id", "ic_product_id", "ext_id"}


def build_global_id_index(products_b: list[ProductRecord]) -> dict[str, list[ProductRecord]]:
    index = defaultdict(list)
    for product in products_b:
        for global_id in product.global_ids:
            index[global_id].append(product)
    return dict(index)


def build_provider_id_index(products_b: list[ProductRecord]) -> dict[tuple[str, str], list[ProductRecord]]:
    index = defaultdict(list)
    for product in products_b:
        for key, value in product.source_ids.items():
            if key not in PROVIDER_MATCH_KEYS:
                continue
            index[(key, value)].append(product)
    return dict(index)


def choose_exact_global_id_match(
    item_a: ProductRecord,
    global_id_index_b: dict[str, list[ProductRecord]],
) -> MatchDecision | None:
    for global_id in item_a.global_ids:
        candidates = global_id_index_b.get(global_id, [])
        if not candidates:
            continue
        item_b = sorted(candidates, key=lambda item: item.item_id)[0]
        return MatchDecision(
            item_id_a=item_a.item_id,
            item_id_b=item_b.item_id,
            confidence=1.0,
            match_quality="high",
            decision_source="upc",
            review_flag=False,
        )
    return None


def choose_provider_id_match(
    item_a: ProductRecord,
    provider_id_index_b: dict[tuple[str, str], list[ProductRecord]],
) -> MatchDecision | None:
    for key, value in item_a.source_ids.items():
        if key not in PROVIDER_MATCH_KEYS:
            continue
        candidates = provider_id_index_b.get((key, value), [])
        if not candidates:
            continue
        item_b = sorted(candidates, key=lambda item: item.item_id)[0]
        return MatchDecision(
            item_id_a=item_a.item_id,
            item_id_b=item_b.item_id,
            confidence=0.96,
            match_quality="high",
            decision_source="provider_id",
            review_flag=False,
        )
    return None
