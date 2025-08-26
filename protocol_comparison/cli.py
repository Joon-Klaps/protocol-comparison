#!/usr/bin/env python3
"""
Command Line Interface for the viral genomics protocol comparison dashboard.

This module provides the main entry point for the CLI with subcommands for:
- run: Run the Streamlit application
- convert-data: Convert data files to parquet format
- convert-parquet: Convert parquet files to CSV
- info: Show package information
- version: Show version information
"""

import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path
import importlib.metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Package information
PACKAGE_NAME = "protocol-comparison"

def get_version() -> str:
    """Get the package version."""
    try:
        return importlib.metadata.version("protocol-comparison")
    except importlib.metadata.PackageNotFoundError:
        # Fallback to version in __init__.py
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "1.0.0"

def cmd_version() -> None:
    """Print version information."""
    print(f"{PACKAGE_NAME} version {get_version()}")

def cmd_info() -> None:
    """Print package information."""
    print(f"""
{PACKAGE_NAME} - Viral Genomics Protocol Comparison Dashboard
Version: {get_version()}
Author: Joon Klaps
Description: A modular Streamlit dashboard for comparing viral genomics analysis protocols

Available commands:
  run            Run the Streamlit application
  convert-data   Convert data files to parquet format
  convert-parquet Convert parquet files to CSV format
  info           Show this information
  version        Show version

For help on a specific command, use: {PACKAGE_NAME} <command> --help
""")

def cmd_run(args: argparse.Namespace) -> None:
    """Run the Streamlit application."""
    try:
        # Import the app module
        from . import app

        # Build the streamlit command
        app_path = Path(app.__file__)
        cmd = ["streamlit", "run", str(app_path)]

        # Add port if specified
        if args.port:
            cmd.extend(["--server.port", str(args.port)])

        # Add data directory if specified
        if args.data:
            os.environ['PROTOCOL_DATA_DIR'] = str(Path(args.data).resolve())

        # Add other streamlit options
        if args.host:
            cmd.extend(["--server.address", args.host])

        if args.browser:
            cmd.append("--server.headless=false")
        else:
            cmd.append("--server.headless=true")

        logger.info(f"Starting Streamlit app with command: {' '.join(cmd)}")
        logger.info(f"Data directory: {os.environ.get('PROTOCOL_DATA_DIR', 'Not specified')}")

        # Run streamlit
        subprocess.run(cmd)

    except ImportError as e:
        logger.error(f"Failed to import app module: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Streamlit is not installed. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

def cmd_convert_data(args: argparse.Namespace) -> None:
    """Convert data files to parquet format."""
    try:
        from . import convert_data

        # Create arguments namespace for convert_data
        convert_args = argparse.Namespace()

        # Map our arguments to convert_data arguments
        convert_args.input_dir = args.input if args.input else "../../data/"
        convert_args.output_dir = args.output if args.output else "../../data/app"
        convert_args.log_level = args.log_level if args.log_level else "INFO"
        convert_args.force_alignment = args.force if args.force else False

        logger.info(f"Converting data from {convert_args.input_dir} to {convert_args.output_dir}")

        # Call the main function from convert_data module
        result = convert_data.main(convert_args)

        if result != 0:
            logger.error("Data conversion failed")
            sys.exit(result)
        else:
            logger.info("Data conversion completed successfully")

    except ImportError as e:
        logger.error(f"Failed to import convert_data module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data conversion failed: {e}")
        sys.exit(1)

def cmd_convert_parquet(args: argparse.Namespace) -> None:
    """Convert parquet files to CSV format."""
    import pandas as pd
    import shutil
    from pathlib import Path

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting parquet files from {input_dir} to {output_dir}")

    # Find and convert parquet files
    parquet_files = list(input_dir.glob("**/*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {input_dir}")

    for parquet_file in parquet_files:
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)

            # Create output path
            relative_path = parquet_file.relative_to(input_dir)
            csv_path = output_dir / relative_path.with_suffix('.csv')

            # Create parent directories if needed
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Write CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Converted {parquet_file.name} -> {csv_path.name}")

        except Exception as e:
            logger.error(f"Failed to convert {parquet_file}: {e}")

    # Copy non-parquet files
    for file_path in input_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix != ".parquet":
            relative_path = file_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(file_path, output_path)
                logger.info(f"Copied {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to copy {file_path}: {e}")

    logger.info("Conversion completed successfully")

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog=PACKAGE_NAME,
        description="Viral Genomics Protocol Comparison Dashboard CLI"
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Streamlit application"
    )
    run_parser.add_argument(
        "--data",
        type=str,
        help="Path to data directory"
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the application on (default: 8501)"
    )
    run_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the application to (default: localhost)"
    )
    run_parser.add_argument(
        "--browser",
        action="store_true",
        help="Open browser automatically"
    )
    run_parser.set_defaults(func=cmd_run)

    # Convert data command
    convert_data_parser = subparsers.add_parser(
        "convert-data",
        help="Convert data files to parquet format"
    )
    convert_data_parser.add_argument(
        "--input",
        type=str,
        help="Input data directory containing nf-core/viralmetagenome output files"
    )
    convert_data_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for converted files"
    )
    convert_data_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    convert_data_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-alignment of sequences even if output files already exist"
    )
    convert_data_parser.set_defaults(func=cmd_convert_data)

    # Convert parquet command
    convert_parquet_parser = subparsers.add_parser(
        "convert-parquet",
        help="Convert parquet files to CSV format"
    )
    convert_parquet_parser.add_argument(
        "input",
        type=str,
        help="Input directory containing parquet files"
    )
    convert_parquet_parser.add_argument(
        "output",
        type=str,
        help="Output directory for CSV files"
    )
    convert_parquet_parser.set_defaults(func=cmd_convert_parquet)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show package information"
    )
    info_parser.set_defaults(func=lambda args: cmd_info())

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.set_defaults(func=lambda args: cmd_version())

    return parser

def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        # No command provided, show help
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()