from matcher.normalize import extract_size, normalize_brand, normalize_name


def test_normalize_brand_lowercases_and_trims():
    assert normalize_brand("  Great Value ") == "great value"


def test_normalize_name_strips_noise():
    assert normalize_name("Chobani Whole Milk Greek Yogurt, Honey Blended 5.3 oz") == "chobani whole milk greek yogurt honey blended 5.3 oz"


def test_extract_size_reads_float_and_unit():
    assert extract_size("16 fl. oz.") == (16.0, "fl oz")
