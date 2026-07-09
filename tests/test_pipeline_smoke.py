from importlib.util import find_spec


def test_required_packages_are_importable():
    assert find_spec("matcher") is not None
    assert find_spec("rapidfuzz") is not None
    assert find_spec("sklearn") is not None
