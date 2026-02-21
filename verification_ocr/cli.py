"""CLI entry point for the verification service."""

import argparse
import sys

import uvicorn

from verification_ocr.core.settings import get_settings


def main() -> int:
    """
    Main CLI entry point.

        int: Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Verification OCR Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default=None, help="Server host")
    server_parser.add_argument("--port", type=int, default=None, help="Server port")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command == "server":
        return run_server(args)
    else:
        parser.print_help()
        return 0


def run_server(args: argparse.Namespace) -> int:
    """
    Run the API server.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

        int: Exit code (0 for success).
    """
    settings = get_settings()

    host = args.host or settings.api_server.host
    port = args.port or settings.api_server.port

    uvicorn.run(
        app="verification_ocr.api.server:app",
        host=host,
        port=port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
