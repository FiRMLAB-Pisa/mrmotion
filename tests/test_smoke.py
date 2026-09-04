"""The package imports and reports a version."""

import mrmotion


def test_the_package_reports_a_version():
    assert isinstance(mrmotion.__version__, str)
    assert mrmotion.__version__
