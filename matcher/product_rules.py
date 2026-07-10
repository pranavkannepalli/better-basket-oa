GENERAL_PRODUCT_TYPES = {
    "apparel": {"bodysuit", "clothing", "dress", "onesie", "pant", "pants", "romper", "shirt"},
    "bread_loaf": {"loaf", "sliced"},
    "bread_roll": {"bun", "buns", "roll", "rolls"},
    "cake": {"cake", "cakes"},
    "candy": {"candy", "gum"},
    "candy_heart": {"heart", "hearts"},
    "candy_kiss": {"kiss", "kisses"},
    "cheese": {"cheese"},
    "chili": {"chili"},
    "cookie": {"cookie", "cookies", "macaroon", "macaroons"},
    "diaper": {"diaper", "diapers"},
    "grain_bulgur": {"bulgur"},
    "grain_couscous": {"couscous"},
    "lotion": {"cream", "lotion"},
    "meatball": {"meatball", "meatballs"},
    "milk": {"milk"},
    "muffin": {"muffin", "muffins"},
    "pasta": {"linguine", "macaroni", "noodle", "noodles", "pasta", "spaghetti"},
    "soup": {"minestrone", "soup"},
}

EXCLUSIVE_PRODUCT_TYPES = {
    "apparel",
    "cake",
    "candy",
    "candy_heart",
    "candy_kiss",
    "cheese",
    "chili",
    "cookie",
    "diaper",
    "grain_bulgur",
    "grain_couscous",
    "lotion",
    "meatball",
    "milk",
    "muffin",
    "pasta",
    "soup",
}
VARIANT_PRODUCT_TYPES = {
    "candy_heart",
    "candy_kiss",
}

DIET_TOKENS = {"diet", "sugar-free", "sugarfree", "zero"}
BEVERAGE_TOKENS = {"cola", "drink", "energy", "pepsi", "pop", "red", "bull", "soda"}
CHEESE_VARIETIES = {
    "american",
    "asiago",
    "blue",
    "cheddar",
    "colby",
    "feta",
    "gouda",
    "gruyere",
    "havarti",
    "jack",
    "manchego",
    "mozzarella",
    "munster",
    "parmesan",
    "pepperjack",
    "provolone",
    "ricotta",
    "romano",
    "swiss",
}


def product_type_groups(tokens: set[str], category_path: list[str]) -> set[str]:
    text_values = {str(value).strip().lower() for value in tokens}
    for category in category_path:
        text_values.update(str(category).lower().replace("&", " ").split())
    return {
        group
        for group, keywords in GENERAL_PRODUCT_TYPES.items()
        if text_values & keywords
    }


def product_types_compatible(groups_a: set[str], groups_b: set[str]) -> bool:
    if not groups_a or not groups_b:
        return True
    variant_a = groups_a & VARIANT_PRODUCT_TYPES
    variant_b = groups_b & VARIANT_PRODUCT_TYPES
    if variant_a and variant_b and not (variant_a & variant_b):
        return False
    if {"bread_loaf", "bread_roll"} <= (groups_a | groups_b):
        return False
    if groups_a & groups_b:
        return True
    exclusive_a = groups_a & EXCLUSIVE_PRODUCT_TYPES
    exclusive_b = groups_b & EXCLUSIVE_PRODUCT_TYPES
    if exclusive_a and exclusive_b:
        return False
    return True


def diet_variant_conflict(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not ((tokens_a | tokens_b) & BEVERAGE_TOKENS):
        return False
    return bool(tokens_a & DIET_TOKENS) != bool(tokens_b & DIET_TOKENS)


def cheese_variety_conflict(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if "cheese" not in tokens_a or "cheese" not in tokens_b:
        return False
    varieties_a = tokens_a & CHEESE_VARIETIES
    varieties_b = tokens_b & CHEESE_VARIETIES
    return bool(varieties_a and varieties_b and not (varieties_a & varieties_b))
