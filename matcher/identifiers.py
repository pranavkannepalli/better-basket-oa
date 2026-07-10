import re
from urllib.parse import urlparse

EXPLICIT_GLOBAL_ID_KEYS = {
    "upc",
    "upc_code",
    "product_upc",
    "gtin",
    "gtin12",
    "gtin13",
    "gtin14",
    "ean",
    "ean13",
    "barcode",
    "bar_code",
}

PROVIDER_ID_KEYS = {"ic_item_id", "ic_product_id", "ext_id"}

_DIGITS_RE = re.compile(r"\D+")
_WALMART_ID_RE = re.compile(r"/ip/(?:[^/]+/)?(?P<id>\d+)(?:[/?#]|$)")
_WEGMANS_SHOP_ID_RE = re.compile(r"/shop/product/(?P<id>\d+)(?:-|[/?#]|$)")
_WEGMANS_PRODUCT_ID_RE = re.compile(r"/product/(?P<id>\d+)(?:[/?#]|$)")


def normalize_identifier(value: object) -> str:
    return _DIGITS_RE.sub("", str(value or ""))


def is_valid_gtin(value: object) -> bool:
    digits = normalize_identifier(value)
    if len(digits) not in {8, 12, 13, 14}:
        return False
    check_digit = int(digits[-1])
    body = [int(char) for char in digits[:-1]]
    total = 0
    for index, digit in enumerate(reversed(body)):
        total += digit * (3 if index % 2 == 0 else 1)
    return (10 - (total % 10)) % 10 == check_digit


def extract_global_ids(row: dict[str, object], item_info: dict[str, object] | None = None) -> tuple[list[str], list[str]]:
    global_ids = []
    flags = []
    sources = [row]
    if item_info:
        sources.append(item_info)

    for source in sources:
        for key, value in source.items():
            normalized_key = str(key).strip().lower()
            if normalized_key not in EXPLICIT_GLOBAL_ID_KEYS:
                continue
            identifier = normalize_identifier(value)
            if is_valid_gtin(identifier):
                global_ids.append(identifier)
                flags.append("explicit_global_id")
            elif identifier:
                flags.append("invalid_global_id")

    return sorted(set(global_ids)), sorted(set(flags))


def extract_source_ids(row: dict[str, object], item_info: dict[str, object] | None = None) -> tuple[dict[str, str], list[str]]:
    source_ids = {}
    flags = []
    if item_info:
        for key in PROVIDER_ID_KEYS:
            value = normalize_identifier(item_info.get(key))
            if value:
                source_ids[key] = value
                flags.append("provider_id_only")

    source_product_id = extract_source_product_id(str(row.get("url", "") or ""))
    if source_product_id:
        source_ids["url_product_id"] = source_product_id
        flags.append("url_source_id")

    return source_ids, sorted(set(flags))


def extract_source_product_id(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    target = parsed.path
    host = parsed.netloc.lower()
    patterns = []
    if "walmart.com" in host:
        patterns.append(_WALMART_ID_RE)
    if "wegmans.com" in host:
        patterns.extend([_WEGMANS_SHOP_ID_RE, _WEGMANS_PRODUCT_ID_RE])
    patterns.extend([_WALMART_ID_RE, _WEGMANS_SHOP_ID_RE, _WEGMANS_PRODUCT_ID_RE])
    for pattern in patterns:
        match = pattern.search(target)
        if match:
            return match.group("id")
    return ""
