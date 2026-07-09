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
