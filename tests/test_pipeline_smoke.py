import matcher
import rapidfuzz
import sklearn


def test_required_packages_are_importable():
    assert matcher.__version__ == "0.1.0"
    assert rapidfuzz.__version__
    assert sklearn.__version__
