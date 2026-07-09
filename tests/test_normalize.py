from matcher.io import dataframe_to_products
from matcher.normalize import extract_size, normalize_brand, normalize_name
from matcher.schemas import ProductRecord


def test_normalize_brand_lowercases_and_trims():
    assert normalize_brand("  Great Value ") == "great value"


def test_normalize_name_strips_noise():
    assert normalize_name("Chobani Whole Milk Greek Yogurt, Honey Blended 5.3 oz") == "chobani whole milk greek yogurt honey blended 5.3 oz"


def test_extract_size_reads_float_and_unit():
    assert extract_size("16 fl. oz.") == (16.0, "fl oz")


def test_dataframe_to_products_builds_product_record():
    rows = [
        {
            "item_id": "10",
            "name": "Wegmans Dressing, Basil Vinaigrette",
            "brand_raw": "Wegmans",
            "description": "Gluten free.",
            "item_info": '{"category_0":"Grocery","category_1":"Salad Dressing"}',
            "sizing_comp": '{"size_user_friendly":"16 fl. oz."}',
            "tags": "['_internal_any_gluten_free']",
        }
    ]
    products = dataframe_to_products(rows)
    assert isinstance(products[0], ProductRecord)
    assert products[0].brand_norm == "wegmans"
    assert products[0].category_path == ["Grocery", "Salad Dressing"]
    assert products[0].size_value == 16.0
