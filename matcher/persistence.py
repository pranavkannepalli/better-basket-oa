from diskcache import Cache


def open_cache(path: str) -> Cache:
    return Cache(path)
