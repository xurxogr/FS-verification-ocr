"""Tests for verification model."""

from verification_ocr.enums import Faction
from verification_ocr.models.verification import Verification


class TestVerification:
    """Tests for Verification model."""

    def test_create_verification_default(self) -> None:
        """
        Test creating a Verification with defaults.

        """
        verification = Verification()
        assert verification.name is None
        assert verification.level is None
        assert verification.regiment is None
        assert verification.faction is None
        assert verification.shard is None
        assert verification.ingame_time is None
        assert verification.war_number is None
        assert verification.current_ingame_time is None

    def test_create_verification_with_values(self) -> None:
        """
        Test creating a Verification with values.

        """
        verification = Verification(
            name="TestPlayer",
            level=25,
            regiment="[TAG] Test Regiment",
            faction=Faction.WARDENS,
            shard="ABLE",
            war_number=132,
            current_ingame_time="268, 14:30",
        )
        assert verification.name == "TestPlayer"
        assert verification.level == 25
        assert verification.regiment == "[TAG] Test Regiment"
        assert verification.faction == Faction.WARDENS
        assert verification.shard == "ABLE"
        assert verification.war_number == 132
        assert verification.current_ingame_time == "268, 14:30"

    def test_verification_to_dict(self) -> None:
        """
        Test Verification model_dump.

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

        """
        verification = Verification(
            name="PartialPlayer",
            faction=Faction.COLONIAL,
        )
        assert verification.name == "PartialPlayer"
        assert verification.level is None
        assert verification.faction == Faction.COLONIAL
