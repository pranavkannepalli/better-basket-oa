from collections import defaultdict
import hashlib
import pickle
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer


def _category_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)


def _product_text(item) -> str:
    parts = [
        item.name,
        item.description,
        item.brand_norm,
        " ".join(item.tokens_core),
        " ".join(item.category_path),
        " ".join(item.attribute_flags),
        " ".join(item.form_flags),
        str(item.size_value or ""),
        item.size_unit,
    ]
    return " ".join(part for part in parts if part).strip()


def _dataset_signature(items_b) -> str:
    digest = hashlib.sha256()
    for item in items_b:
        digest.update(item.item_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(item.brand_norm.encode("utf-8"))
        digest.update(b"\n")
        digest.update(_product_text(item).encode("utf-8"))
        digest.update(b"\n")
    digest.update(f"count:{len(items_b)}".encode("utf-8"))
    return digest.hexdigest()


def _size_key(item):
    if item.size_value is None or not item.size_unit:
        return None
    return (item.size_unit, round(float(item.size_value), 3))


def build_retrieval_index(items_b):
    by_brand = defaultdict(list)
    by_size = defaultdict(list)
    corpus = []

    for item in items_b:
        if item.brand_norm:
            by_brand[item.brand_norm].append(item)
        size_key = _size_key(item)
        if size_key:
            by_size[size_key].append(item)
        corpus.append(_product_text(item))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus) if corpus else None

    return {
        "by_brand": by_brand,
        "by_size": by_size,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "items_b": list(items_b),
        "dataset_signature": _dataset_signature(items_b),
    }


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def attach_embedding_matrix(index, embeddings_by_item_id: dict[str, list[float]]) -> None:
    items_b = index.get("items_b", [])
    vectors = []
    item_indices = []
    for idx, item in enumerate(items_b):
        vector = embeddings_by_item_id.get(item.item_id)
        if vector is None:
            continue
        vectors.append(vector)
        item_indices.append(idx)

    if not vectors:
        index["embedding_matrix"] = None
        index["embedding_item_indices"] = []
        return

    index["embedding_matrix"] = _normalize_matrix(np.asarray(vectors, dtype=np.float32))
    index["embedding_item_indices"] = item_indices


def save_retrieval_index(path, index) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle)


def load_retrieval_index(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def get_or_build_retrieval_index(items_b, path=None):
    expected_signature = _dataset_signature(items_b)
    if path is not None:
        index_path = Path(path)
        if index_path.exists():
            try:
                cached = load_retrieval_index(index_path)
            except (pickle.PickleError, EOFError, AttributeError, ValueError):
                cached = None
            if cached is not None and cached.get("dataset_signature") == expected_signature:
                return cached

    index = build_retrieval_index(items_b)
    if path is not None:
        save_retrieval_index(path, index)
    return index


def _semantic_candidates(item_a, index, limit: int) -> list:
    matrix = index.get("matrix")
    vectorizer = index.get("vectorizer")
    items_b = index.get("items_b", [])
    if matrix is None or not items_b:
        return []

    query = vectorizer.transform([_product_text(item_a)])
    similarities = (matrix @ query.T).toarray().ravel()
    if len(similarities) <= limit:
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -limit)[-limit:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    return [items_b[int(idx)] for idx in top_indices if similarities[int(idx)] > 0]


def _embedding_candidates(query_embedding, index, limit: int) -> list:
    matrix = index.get("embedding_matrix")
    item_indices = index.get("embedding_item_indices", [])
    items_b = index.get("items_b", [])
    if query_embedding is None or matrix is None or not item_indices:
        return []

    query = np.asarray(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    similarities = matrix @ (query / norm)
    if len(similarities) <= limit:
        top_positions = np.argsort(similarities)[::-1]
    else:
        top_positions = np.argpartition(similarities, -limit)[-limit:]
        top_positions = top_positions[np.argsort(similarities[top_positions])[::-1]]
    return [items_b[item_indices[int(pos)]] for pos in top_positions if similarities[int(pos)] > 0]


def retrieve_candidates(item_a, items_b, index=None, top_k: int = 200, query_embedding=None):
    search_space = []
    if index:
        search_space.extend(_embedding_candidates(query_embedding, index, max(top_k * 3, 200)))
        search_space.extend(_semantic_candidates(item_a, index, max(top_k * 3, 200)))
        if item_a.brand_norm in index["by_brand"]:
            search_space.extend(index["by_brand"][item_a.brand_norm])
        size_key = _size_key(item_a)
        if size_key and size_key in index["by_size"]:
            search_space.extend(index["by_size"][size_key])
    else:
        search_space.extend(items_b)

    if not search_space:
        search_space = list(items_b)

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
        semantic_score = token_set_ratio(_product_text(item_a), _product_text(item_b)) / 100.0
        score = (
            (0.35 * text_score)
            + (0.25 * semantic_score)
            + (0.15 * cat_score)
            + (0.15 * brand_score)
            + (0.10 * size_score)
        )
        scored.append((score, item_b))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]
