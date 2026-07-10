from pathlib import Path

import orjson

from diskcache import Cache


def open_cache(path: str) -> Cache:
    return Cache(path)


def append_match_log(path, payload: dict):
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(orjson.dumps(payload))
        handle.write(b"\n")


def load_checkpoint(path: str | Path) -> dict | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        return orjson.loads(checkpoint_path.read_bytes())
    except orjson.JSONDecodeError:
        return None


def save_checkpoint(path: str | Path, payload: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    temporary_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    temporary_path.replace(checkpoint_path)
