"""Tests for verification model."""

from verification_ocr.models.verification import Verification


class TestVerification:
    """Tests for Verification model."""

    def test_create_verification_default(self) -> None:
        """
        Test creating a Verification with defaults.

        Returns:
            None
        """
        verification = Verification()
        assert verification.name is None
        assert verification.level is None
        assert verification.regiment is None
        assert verification.colonial is None
        assert verification.shard is None
        assert verification.ingame_time is None

    def test_create_verification_with_values(self) -> None:
        """
        Test creating a Verification with values.

        Returns:
            None
        """
        verification = Verification(
            name="TestPlayer",
            level=25,
            regiment=True,
            colonial=False,
            shard="ABLE",
        )
        assert verification.name == "TestPlayer"
        assert verification.level == 25
        assert verification.regiment is True
        assert verification.colonial is False
        assert verification.shard == "ABLE"

    def test_verification_to_dict(self) -> None:
        """
        Test Verification model_dump.

        Returns:
            None
        """
        verification = Verification(
            name="Player1",
            level=10,
        )
        data = verification.model_dump()
        assert data["name"] == "Player1"
        assert data["level"] == 10
        assert data["regiment"] is None

    def test_verification_partial_data(self) -> None:
        """
        Test Verification with partial data.

        Returns:
            None
        """
        verification = Verification(
            name="PartialPlayer",
            colonial=True,
        )
        assert verification.name == "PartialPlayer"
        assert verification.level is None
        assert verification.colonial is True
