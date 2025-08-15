#! /usr/bin/env python3
"""
A script to convert output files from nf-core/viralmetagenome to the formats needed for this analysis platform.
"""

import pandas as pd
import logging
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Callable


def load_excel(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load the main Excel file containing sample information.

    Args:
        file_path: Path to the main Excel file.
        sheet_name: Name of the sheet to load.

    Returns:
        DataFrame with sample information.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except KeyError:
        logging.error("Sheet '%s' not found in %s", sheet_name, file_path)
        sys.exit(1)


def extract_mapping(file_path: Path) -> pd.DataFrame:
    """
    Extract mapping statistics from the main DataFrame.

    Args:
        file_path: Path to the Excel file.

    Returns:
        DataFrame with mapping statistics.
    """
    df = load_excel(file_path, "mapping")

    columns_of_interest = [
        "sample",
        "cluster",
        "species",
        "segment",
        "(samtools Raw) reads mapped (R1+R2)",
        "(samtools Raw) reads mapped %",
        "(samtools Raw) reads unmapped (R1+R2)",
        "(samtools Raw) reads unmapped %"
    ]

    for column in columns_of_interest:
        if column not in df.columns:
            logging.error("Column '%s' not found in the main DataFrame", column)
            sys.exit(1)

    mapping_stats = df[columns_of_interest].dropna()
    return pd.DataFrame(mapping_stats)


def extract_reads(file_path: Path) -> pd.DataFrame:
    """
    Extract read statistics from the main DataFrame.

    Args:
        file_path: Path to the Excel file.

    Returns:
        DataFrame with read statistics.
    """
    df = load_excel(file_path, "samples")

    columns_of_interest = [
        "sample",
        "FastQC (Raw). Seqs (R1,R2)",
        "FastQC (Post-trimming). Seqs (R1,R2)",
        "FastQC (post-Host-removal). Seqs (R1,R2)",
    ]

    for column in columns_of_interest:
        if column not in df.columns:
            logging.error("Column '%s' not found in the main DataFrame", column)
            sys.exit(1)

    read_stats = df[columns_of_interest].dropna()
    return pd.DataFrame(read_stats)

def find_locations(directory: Path) -> Dict[str, Path]:
    """
    Find all relevant files in the given directory.

    Args:
        directory: Path to the directory to search.

    Returns:
        Dictionary mapping file types to their Paths.
    """
    locations = {}

    main_excel_path = directory / "RUN1-6.rbind.xlsx"
    if main_excel_path.exists():
        locations["mainexcel"] = main_excel_path

    return locations


def write_dfs(output_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Write DataFrames to TSV files in the specified output directory.

    Args:
        output_dfs: Dictionary mapping output names to DataFrames.
        output_dir: Path to the output directory.
    """
    for name, df in output_dfs.items():
        if df.empty:
            logging.warning("DataFrame for '%s' is empty, skipping", name)
            continue

        output_dir_name = output_dir / name
        output_dir_name.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_name / f"{name}.tsv"
        df.to_csv(output_path, sep='\t', index=False)
        logging.info("Data for '%s' saved to %s", name, output_path)


def main(args: argparse.Namespace) -> int:
    """
    Main function to handle the conversion process.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(level=args.log_level.upper())

    output_dfs = {}

    # Find all relevant files
    locations = find_locations(Path(args.input_dir))

    if not locations:
        logging.error("No relevant files found in %s", args.input_dir)
        return 1

    if locations["mainexcel"]:
        output_dfs["mapping"] = extract_mapping(Path(locations["mainexcel"]))
        output_dfs["reads"] = extract_reads(Path(locations["mainexcel"]))

    if output_dfs:
        write_dfs(output_dfs, Path(args.output_dir))
    else:
        logging.error("No output DataFrames to write.")
        return 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert nf-core/viralmetagenome output files to analysis platform format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/joonklaps/Desktop/School/PhD/projects/LVE-BE002-PIPELINE/LVE-BE02-Supplmentary/results/HPC-results/TMP-RUN001-004/inrahost-analysis/data/viralmetagenome",
        help="Input directory containing nf-core/viralmetagenome output files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/joonklaps/Desktop/School/PhD/projects/LVE-BE002-PIPELINE/LVE-BE02-Supplmentary/results/HPC-results/TMP-RUN001-004/inrahost-analysis/data/app",
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()
    sys.exit(main(args))