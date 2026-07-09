from matcher.io import load_catalog_csv
from matcher.parsing import parse_item_info, parse_sizing_comp
from matcher.schemas import MatchDecision, ProductRecord


def test_load_catalog_csv_preserves_strings_and_empty_cells(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("item_id,name,price\n1,Tomato Sauce,\n2,,3.99\n")

    df = load_catalog_csv(path)

    assert df.to_dict("records") == [
        {"item_id": "1", "name": "Tomato Sauce", "price": ""},
        {"item_id": "2", "name": "", "price": "3.99"},
    ]
    assert all(isinstance(value, str) for value in df.to_numpy().ravel())


def test_product_record_defaults():
    product = ProductRecord(item_id="1", name="Tomato Sauce")
    assert product.brand_norm == ""
    assert product.private_label_flag is False


def test_match_decision_requires_ids():
    decision = MatchDecision(
        item_id_a="1",
        item_id_b="2",
        confidence=0.75,
        match_quality="medium",
        decision_source="llm",
        review_flag=False,
    )
    assert decision.item_id_a == "1"
    assert decision.item_id_b == "2"


def test_parse_item_info_extracts_category_path():
    raw = '{"category_0":"Grocery","category_1":"Sauces","category_2":"Tomato Sauce"}'
    assert parse_item_info(raw) == ["Grocery", "Sauces", "Tomato Sauce"]


def test_parse_item_info_orders_category_keys_numerically():
    raw = '{"category_2":"Sauces","category_10":"Tomato Sauce","category_1":"Grocery"}'
    assert parse_item_info(raw) == ["Grocery", "Sauces", "Tomato Sauce"]


def test_parse_item_info_returns_empty_list_for_non_object_json():
    assert parse_item_info('["Grocery", "Sauces"]') == []


def test_parse_item_info_filters_non_string_category_values():
    raw = '{"category_0":"Grocery","category_1":2,"category_2":false,"category_3":"Sauces"}'
    assert parse_item_info(raw) == ["Grocery", "Sauces"]


def test_parse_item_info_returns_empty_list_for_invalid_json():
    assert parse_item_info("{not json}") == []


def test_parse_sizing_comp_extracts_user_friendly_size():
    raw = '{"size_user_friendly":"16 fl. oz.","billed_by_weight":false}'
    parsed = parse_sizing_comp(raw)
    assert parsed["size_user_friendly"] == "16 fl. oz."


def test_parse_sizing_comp_returns_empty_dict_for_invalid_json():
    assert parse_sizing_comp("{not json}") == {}


def test_parse_sizing_comp_returns_empty_dict_for_non_object_json():
    assert parse_sizing_comp('["size_user_friendly", "16 fl. oz."]') == {}


def test_parsers_return_fallbacks_for_none():
    assert parse_item_info(None) == []
    assert parse_sizing_comp(None) == {}


def test_parsers_return_fallbacks_for_truthy_non_string_input():
    assert parse_item_info(1) == []
    assert parse_sizing_comp(1) == {}
