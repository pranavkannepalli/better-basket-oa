def test_required_packages_are_importable():
    import matcher
    import rapidfuzz
    import sklearn
    from matcher.cli import main

    assert matcher.__version__ == "0.1.0"
    assert callable(main)
    assert rapidfuzz.__version__
    assert sklearn.__version__
