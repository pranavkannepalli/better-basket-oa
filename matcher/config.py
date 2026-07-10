import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from matcher.embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL

load_dotenv()


class Settings(BaseModel):
    input_a_path: Path = Path(os.environ.get("INPUT_A_PATH", "grocery_store_a_items_final.csv"))
    input_b_path: Path = Path(os.environ.get("INPUT_B_PATH", "grocery_store_b_items_final.csv"))
    output_dir: Path = Path(os.environ.get("OUTPUT_DIR", "artifacts/full-llm-local-embeddings"))
    cache_dir: Path = Path(os.environ.get("CACHE_DIR", "artifacts/full-llm-local-embeddings/cache"))
    retrieval_index_path: Path | None = (
        Path(os.environ["RETRIEVAL_INDEX_PATH"]) if "RETRIEVAL_INDEX_PATH" in os.environ else None
    )
    llm_model: str = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
    embedding_batch_size: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "128"))
    retrieval_k: int = int(os.environ.get("RETRIEVAL_K", "30"))
    llm_top_n: int = int(os.environ.get("LLM_TOP_N", "8"))
    llm_min_deterministic: float = float(os.environ.get("LLM_MIN_DETERMINISTIC", "0.30"))
    max_workers: int = int(os.environ.get("MAX_WORKERS", "2"))
    item_retry_attempts: int = int(os.environ.get("ITEM_RETRY_ATTEMPTS", "2"))
    min_confidence: float = float(os.environ.get("MIN_CONFIDENCE", "0.62"))
    high_quality_threshold: float = float(os.environ.get("HIGH_QUALITY_THRESHOLD", "0.72"))
    medium_quality_threshold: float = float(os.environ.get("MEDIUM_QUALITY_THRESHOLD", "0.40"))
