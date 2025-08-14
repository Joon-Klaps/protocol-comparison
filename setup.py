#!/usr/bin/env python3
"""
Setup script for the viral genomics protocol comparison dashboard.

This script helps with initial setup and testing of the analysis platform.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(command: list, description: str) -> bool:
    """
    Run a shell command and return success status.

    Args:
        command: Command to run as list
        description: Description of what the command does

    Returns:
        True if successful, False otherwise
    """
    print(f"⚡ {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def install_dependencies() -> bool:
    """Install required Python packages."""
    return run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing dependencies"
    )


def generate_sample_data(num_samples: int = 10) -> bool:
    """Generate sample data for testing."""
    return run_command(
        [sys.executable, "generate_sample_data.py", "--num-samples", str(num_samples)],
        f"Generating sample data for {num_samples} samples"
    )


def test_import() -> bool:
    """Test if all modules can be imported."""
    print("⚡ Testing module imports...")
    try:
        # Test individual module imports
        import pandas as pd
        import numpy as np
        import streamlit
        import plotly
        import streamlit_option_menu

        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def check_data_structure(data_path: Path) -> bool:
    """Check if data directory has the expected structure."""
    print(f"⚡ Checking data structure in {data_path}...")

    required_dirs = [
        'consensus', 'coverage', 'depth', 'read_stats',
        'mapping', 'contamination', 'references'
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        if not (data_path / dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("✅ Data directory structure is correct")
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for viral genomics analysis platform"
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate sample data for testing"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)"
    )

    parser.add_argument(
        "--check-data",
        type=str,
        help="Check data directory structure"
    )

    parser.add_argument(
        "--test-imports",
        action="store_true",
        help="Test if all required modules can be imported"
    )

    parser.add_argument(
        "--full-setup",
        action="store_true",
        help="Run full setup (install deps, generate data, test)"
    )

    args = parser.parse_args()

    print("🧬 Viral Genomics Protocol Comparison Dashboard Setup")
    print("=" * 60)

    success = True

    if args.full_setup or args.install_deps:
        success &= install_dependencies()
        print()

    if args.full_setup or args.test_imports:
        success &= test_import()
        print()

    if args.full_setup or args.generate_data:
        success &= generate_sample_data(args.num_samples)
        print()

        # Also check the generated data structure
        data_path = Path("sample_data")
        if data_path.exists():
            success &= check_data_structure(data_path)

    if args.check_data:
        data_path = Path(args.check_data)
        success &= check_data_structure(data_path)

    print("=" * 60)

    if success:
        print("🎉 Setup completed successfully!")

        if args.full_setup or args.generate_data:
            print("\n📊 To start the dashboard with sample data:")
            print("   streamlit run streamlit_app.py")
            print("   # or")
            print("   python run_streamlit.py --data-path sample_data")
            print("   # or")
            print("   make run")

        print("\n📚 For more information, see README.md")

    else:
        print("❌ Setup encountered some issues. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
