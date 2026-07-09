import json


def _safe_json_loads(raw: str):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _category_sort_key(item: tuple[str, str]) -> tuple[int, int | str]:
    key, _ = item
    suffix = key.removeprefix("category_")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def parse_item_info(raw: str) -> list[str]:
    data = _safe_json_loads(raw)
    if not isinstance(data, dict):
        return []
    return [
        value
        for key, value in sorted(data.items(), key=_category_sort_key)
        if key.startswith("category_") and value
    ]


def parse_sizing_comp(raw: str) -> dict:
    data = _safe_json_loads(raw)
    return data if isinstance(data, dict) else {}
