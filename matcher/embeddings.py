from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Callable

import numpy as np

from matcher.retrieval import _product_text

DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class LocalEmbeddingModel:
    model_name: str
    encoder: object

    def embed_products(
        self,
        products,
        batch_size: int = 128,
        progress_callback: Callable[[int, int, float], None] | None = None,
        batch_start_callback: Callable[[int, int, int], None] | None = None,
        progress_interval: int = 5000,
    ) -> np.ndarray:
        """Embed products into one compact float32 matrix without retaining all text."""
        if not products:
            return np.empty((0, 0), dtype=np.float32)

        matrix = None
        batch_size = max(batch_size, 1)
        progress_interval = max(progress_interval, batch_size)
        started = time.perf_counter()
        next_progress = min(len(products), progress_interval)
        for start in range(0, len(products), batch_size):
            batch = products[start : start + batch_size]
            if batch_start_callback is not None:
                batch_start_callback(start, start + len(batch), len(products))
            texts = [_product_text(product) for product in batch]
            vectors = np.asarray(
                list(self.encoder.embed(texts, batch_size=batch_size)),
                dtype=np.float32,
            )
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if matrix is None:
                matrix = np.empty((len(products), vectors.shape[1]), dtype=np.float32)
            matrix[start : start + len(batch)] = vectors
            completed = start + len(batch)
            if progress_callback is not None and (completed >= next_progress or completed == len(products)):
                progress_callback(completed, len(products), time.perf_counter() - started)
                while next_progress <= completed and next_progress < len(products):
                    next_progress += progress_interval
        return matrix


def _load_fastembed_model(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require fastembed. "
            "Install dependencies with `pip install -e .` before running embedding retrieval."
        ) from exc
    return TextEmbedding(model_name=model_name)


def load_local_embedding_model(model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(model_name=model_name, encoder=_load_fastembed_model(model_name))


def embedding_cache_path(output_dir: str, model_name: str, dataset_signature: str) -> Path:
    model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    return Path(output_dir) / "cache" / f"embeddings-b-{model_digest}-{dataset_signature[:16]}.npz"


def load_embedding_cache(path: Path, products) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        item_ids = [str(value) for value in data["item_ids"]]
        expected_item_ids = [product.item_id for product in products]
        if item_ids != expected_item_ids:
            return None
        matrix = np.asarray(data["matrix"], dtype=np.float32)
    return matrix


def save_embedding_cache(path: Path, matrix: np.ndarray, products) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(matrix) != len(products):
        raise ValueError("Embedding matrix must contain one row for every product")
    item_ids = np.asarray([product.item_id for product in products])
    np.savez_compressed(path, item_ids=item_ids, matrix=matrix)
