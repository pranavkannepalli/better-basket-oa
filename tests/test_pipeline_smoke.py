def test_required_packages_are_importable():
    from importlib import metadata

    import matcher
    import rapidfuzz
    import sklearn

    entry_points = metadata.entry_points(group="console_scripts")
    bb_match = [entry_point for entry_point in entry_points if entry_point.name == "bb-match"]

    assert matcher.__version__ == "0.1.0"
    assert bb_match
    assert bb_match[0].value == "matcher.cli:main"
    assert rapidfuzz.__version__
    assert sklearn.__version__
