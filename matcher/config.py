from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    input_a_path: Path = Path("grocery_store_a_items_final.csv")
    input_b_path: Path = Path("grocery_store_b_items_final.csv")
    output_dir: Path = Path("artifacts")
    cache_dir: Path = Path("artifacts/cache")
    llm_model: str = "gpt-5-nano"
    retrieval_k: int = 200
    llm_top_n: int = 24
    high_quality_threshold: float = 0.85
    medium_quality_threshold: float = 0.55
