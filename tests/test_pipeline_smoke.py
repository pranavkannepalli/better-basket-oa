from matcher.pipeline import run_pipeline


def test_run_pipeline_returns_one_match_per_input_row(tmp_path):
    rows_a = [
        {
            "item_id": "a1",
            "name": "Organic Tomato Sauce 8 oz",
            "brand_raw": "Great Value",
            "description": "",
            "item_info": '{"category_0":"Grocery","category_1":"Sauces"}',
            "sizing_comp": '{"size_user_friendly":"8 oz"}',
            "tags": "[]",
        },
        {
            "item_id": "a2",
            "name": "Dog Food",
            "brand_raw": "Pedigree",
            "description": "",
            "item_info": '{"category_0":"Pets"}',
            "sizing_comp": '{}',
            "tags": "[]",
        },
    ]
    rows_b = [
        {
            "item_id": "b1",
            "name": "Wegmans Organic Tomato Sauce 8 oz",
            "brand_raw": "Wegmans",
            "description": "",
            "item_info": '{"category_0":"Grocery","category_1":"Sauces"}',
            "sizing_comp": '{"size_user_friendly":"8 oz"}',
            "tags": "[]",
        },
        {
            "item_id": "b2",
            "name": "Pedigree Dog Food",
            "brand_raw": "Pedigree",
            "description": "",
            "item_info": '{"category_0":"Pets"}',
            "sizing_comp": '{}',
            "tags": "[]",
        },
    ]
    matches = run_pipeline(rows_a, rows_b, llm_enabled=False)
    assert len(matches) == 2
    assert {match.item_id_a for match in matches} == {"a1", "a2"}
