"""Tests for response models."""

from verification_ocr.models import (
    HealthResponse,
    Verification,
    VerificationResponse,
    WarResponse,
)


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_create_health_response(self) -> None:
        """
        Test creating a HealthResponse.

        """
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"

    def test_health_response_with_tesseract_version(self) -> None:
        """
        Test HealthResponse with tesseract_version.

        """
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            tesseract_version="tesseract 5.0.0",
        )
        assert response.tesseract_version == "tesseract 5.0.0"

    def test_health_response_tesseract_version_default_none(self) -> None:
        """
        Test HealthResponse tesseract_version defaults to None.

        """
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
        )
        assert response.tesseract_version is None

    def test_health_response_to_dict(self) -> None:
        """
        Test HealthResponse model_dump.

        """
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
        )
        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestVerificationResponse:
    """Tests for VerificationResponse model."""

    def test_create_verification_response_success(self) -> None:
        """
        Test creating a successful VerificationResponse.

        """
        response = VerificationResponse(
            success=True,
            verification=Verification(name="TestUser", level=10),
        )
        assert response.success is True
        assert response.error is None
        assert response.verification.name == "TestUser"

    def test_verification_response_default_values(self) -> None:
        """
        Test VerificationResponse default values.

        """
        response = VerificationResponse(success=True)
        assert response.error is None
        assert response.verification is None

    def test_verification_response_failure(self) -> None:
        """
        Test VerificationResponse for failure case.

        """
        response = VerificationResponse(
            success=False,
            error="No name found in any of the images",
        )
        assert response.success is False
        assert response.error == "No name found in any of the images"
        assert response.verification is None

    def test_verification_response_to_dict(self) -> None:
        """
        Test VerificationResponse model_dump.

        """
        response = VerificationResponse(
            success=True,
            verification=Verification(name="Player1"),
        )
        data = response.model_dump()
        assert data["success"] is True
        assert data["verification"]["name"] == "Player1"

    def test_verification_response_with_full_verification(self) -> None:
        """
        Test VerificationResponse with full verification data.

        """
        response = VerificationResponse(
            success=True,
            verification=Verification(
                name="TestPlayer",
                level=25,
                regiment="[TAG] My Regiment",
                colonial=False,
                shard="ABLE",
            ),
        )
        assert response.verification.name == "TestPlayer"
        assert response.verification.level == 25
        assert response.verification.regiment == "[TAG] My Regiment"
        assert response.verification.colonial is False
        assert response.verification.shard == "ABLE"


class TestWarResponse:
    """Tests for WarResponse model."""

    def test_create_war_response(self) -> None:
        """
        Test creating a WarResponse.

        """
        response = WarResponse(
            war_number=132,
            war_day=48,
            war_hour=12,
            war_minute=30,
            start_time=1770663602746,
        )
        assert response.war_number == 132
        assert response.war_day == 48
        assert response.war_hour == 12
        assert response.war_minute == 30
        assert response.start_time == 1770663602746

    def test_war_response_default_values(self) -> None:
        """
        Test WarResponse default values are None.

        """
        response = WarResponse()
        assert response.war_number is None
        assert response.war_day is None
        assert response.war_hour is None
        assert response.war_minute is None
        assert response.start_time is None

    def test_war_response_to_dict(self) -> None:
        """
        Test WarResponse model_dump.

        """
        response = WarResponse(
            war_number=132,
            war_day=48,
            war_hour=15,
            war_minute=45,
        )
        data = response.model_dump()
        assert data["war_number"] == 132
        assert data["war_day"] == 48
        assert data["war_hour"] == 15
        assert data["war_minute"] == 45

    def test_war_response_partial_values(self) -> None:
        """
        Test WarResponse with partial values.

        """
        response = WarResponse(war_number=132)
        assert response.war_number == 132
        assert response.war_day is None
        assert response.war_hour is None
        assert response.war_minute is None
        assert response.start_time is None
