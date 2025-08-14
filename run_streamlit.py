#!/usr/bin/env python3
"""
Launcher script for the Streamlit viral genomics protocol comparison dashboard.

This script provides a command-line interface to start the Streamlit application
with various configuration options.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dashboard.log')
        ]
    )


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Viral Genomics Protocol Comparison Dashboard (Streamlit)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data directory (will be set as default in the app)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to bind the server to (default: 8501)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (don't open browser)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Viral Genomics Protocol Comparison Dashboard (Streamlit)")

    # Build streamlit command
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app.py",
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--logger.level", args.log_level.lower()
    ]

    # Add headless option if specified
    if args.headless:
        streamlit_cmd.extend(["--server.headless", "true"])

    # Print startup information
    print("\n" + "="*60)
    print("  Viral Genomics Protocol Comparison Dashboard")
    print("  (Streamlit Version)")
    print("="*60)
    print(f"  URL: http://{args.host}:{args.port}")

    if args.data_path:
        print(f"  Data path: {args.data_path}")
        # Create environment variable for default data path
        import os
        os.environ['DEFAULT_DATA_PATH'] = args.data_path
    else:
        print("  Data path: Configure via web interface")

    print("="*60)
    print("  Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    # Start the application
    try:
        subprocess.run(streamlit_cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error("Error running dashboard: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
