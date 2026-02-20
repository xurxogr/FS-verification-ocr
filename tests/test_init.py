"""Tests for verification_ocr package initialization."""

import verification_ocr


class TestPackageInit:
    """Tests for package initialization."""

    def test_version_exists(self) -> None:
        """
        Test that __version__ is defined.

        Returns:
            None
        """
        assert hasattr(verification_ocr, "__version__")

    def test_version_is_string(self) -> None:
        """
        Test that __version__ is a string.

        Returns:
            None
        """
        assert isinstance(verification_ocr.__version__, str)

    def test_version_format(self) -> None:
        """
        Test that __version__ follows semantic versioning format.

        Returns:
            None
        """
        version = verification_ocr.__version__
        parts = version.split(".")
        assert len(parts) >= 2
        # Major and minor should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()
