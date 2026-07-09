import json


def _safe_json_loads(raw: str):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def parse_item_info(raw: str) -> list[str]:
    data = _safe_json_loads(raw)
    return [
        value
        for key, value in sorted(data.items())
        if key.startswith("category_") and value
    ]


def parse_sizing_comp(raw: str) -> dict:
    data = _safe_json_loads(raw)
    return data if isinstance(data, dict) else {}
