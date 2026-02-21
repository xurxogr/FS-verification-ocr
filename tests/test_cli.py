"""Tests for CLI module."""

import argparse
import sys
from unittest.mock import patch

from verification_ocr.cli import main, run_server


class TestMain:
    """Tests for main function."""

    def test_main_no_args_shows_help(self) -> None:
        """
        Test that main with no args shows help and returns 0.

        """
        with patch.object(sys, "argv", ["vocr"]):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                result = main()
                mock_help.assert_called_once()
                assert result == 0

    def test_main_server_command(self) -> None:
        """
        Test that main with server command calls run_server.

        """
        with patch.object(sys, "argv", ["vocr", "server"]):
            with patch("verification_ocr.cli.run_server", return_value=0) as mock_run:
                result = main()
                mock_run.assert_called_once()
                assert result == 0

    def test_main_server_with_host(self) -> None:
        """
        Test that main passes host argument to run_server.

        """
        with patch.object(sys, "argv", ["vocr", "server", "--host", "127.0.0.1"]):
            with patch("verification_ocr.cli.run_server", return_value=0) as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.host == "127.0.0.1"

    def test_main_server_with_port(self) -> None:
        """
        Test that main passes port argument to run_server.

        """
        with patch.object(sys, "argv", ["vocr", "server", "--port", "9000"]):
            with patch("verification_ocr.cli.run_server", return_value=0) as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.port == 9000

    def test_main_server_with_reload(self) -> None:
        """
        Test that main passes reload argument to run_server.

        """
        with patch.object(sys, "argv", ["vocr", "server", "--reload"]):
            with patch("verification_ocr.cli.run_server", return_value=0) as mock_run:
                main()
                args = mock_run.call_args[0][0]
                assert args.reload is True

    def test_main_unknown_command_shows_help(self) -> None:
        """
        Test that unknown command shows help.

        """
        with patch.object(sys, "argv", ["vocr"]):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                result = main()
                mock_help.assert_called_once()
                assert result == 0


class TestRunServer:
    """Tests for run_server function."""

    def test_run_server_calls_uvicorn(self) -> None:
        """
        Test that run_server calls uvicorn.run.

        """
        args = argparse.Namespace(host=None, port=None, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            result = run_server(args)

            mock_uvicorn.assert_called_once()
            assert result == 0

    def test_run_server_uses_custom_host(self) -> None:
        """
        Test that run_server uses custom host.

        """
        args = argparse.Namespace(host="127.0.0.1", port=None, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"

    def test_run_server_uses_custom_port(self) -> None:
        """
        Test that run_server uses custom port.

        """
        args = argparse.Namespace(host=None, port=9000, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs["port"] == 9000

    def test_run_server_uses_reload(self) -> None:
        """
        Test that run_server uses reload setting.

        """
        args = argparse.Namespace(host=None, port=None, reload=True)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs["reload"] is True

    def test_run_server_uses_default_host_from_settings(self) -> None:
        """
        Test that run_server uses default host from settings when not provided.

        """
        args = argparse.Namespace(host=None, port=None, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            # Default from settings is 127.0.0.1 (secure default)
            assert call_kwargs["host"] == "127.0.0.1"

    def test_run_server_uses_default_port_from_settings(self) -> None:
        """
        Test that run_server uses default port from settings when not provided.

        """
        args = argparse.Namespace(host=None, port=None, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            # Default from settings is 8000
            assert call_kwargs["port"] == 8000

    def test_run_server_passes_correct_app_string(self) -> None:
        """
        Test that run_server passes correct app string to uvicorn.

        """
        args = argparse.Namespace(host=None, port=None, reload=False)

        with patch("verification_ocr.cli.uvicorn.run") as mock_uvicorn:
            run_server(args)

            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs["app"] == "verification_ocr.api.server:app"
