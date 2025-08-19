#! /usr/bin/env python3
"""
A script to convert output files from nf-core/viralmetagenome to Parquet format for optimized analysis platform performance.
"""

import pandas as pd
import logging
import sys
import os
import argparse
import re
from pathlib import Path
from typing import Dict, Union, List
from tqdm import tqdm

def load_excel(file_path: Path, sheet_name: str, **kwargs) -> pd.DataFrame:
    """
    Load the main Excel file containing sample information.

    Args:
        file_path: Path to the main Excel file.
        sheet_name: Name of the sheet to load.

    Returns:
        DataFrame with sample information.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except (KeyError, FileNotFoundError, ValueError) as e:
        logging.error("Error loading sheet '%s' from %s: %s", sheet_name, file_path, e)
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
        "(samtools Raw) reads unmapped %",
        "(umitools) deduplicated reads (R1,R2)",
        "(umitools) total UMIs",
        "(umitools) unique UMIs"
    ]

    for column in columns_of_interest:
        if column not in df.columns:
            logging.error("Column '%s' not found in the main DataFrame", column)
            sys.exit(1)

    mapping_stats = df[columns_of_interest].copy()
    mapping_stats["(umitools) deduplicated reads (R1,R2)"] = mapping_stats["(umitools) deduplicated reads (R1,R2)"] * 2

    # Calculate estimated PCR cycles: Total UMIs ÷ Unique UMIs
    # This represents the average number of copies of each read (amplification level)
    mapping_stats["(umitools) estimated PCR cycles"] = (
        pd.to_numeric(mapping_stats["(umitools) total UMIs"], errors="coerce") /
        pd.to_numeric(mapping_stats["(umitools) unique UMIs"], errors="coerce")
    )

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

    df["FastQC (Raw). Seqs (R1,R2)"] = df["FastQC (Raw). Seqs (R1,R2)"] *2
    df["FastQC (Post-trimming). Seqs (R1,R2)"] = df["FastQC (Post-trimming). Seqs (R1,R2)"] *2
    df["FastQC (post-Host-removal). Seqs (R1,R2)"] = df["FastQC (post-Host-removal). Seqs (R1,R2)"] *2

    read_stats = df[columns_of_interest]
    return pd.DataFrame(read_stats)

def extract_comparison(file_path: Path) -> pd.DataFrame:
    """
    Extract comparison excel data.

    Args:
        file_path: Path to the Excel file.

    Returns:
        DataFrame with comparison information.
    """
    df = load_excel(file_path, "Sheet1", )

    # Replace "NA" to NaN, "Yes" to True & "No" to False
    df = df.replace("NA", pd.NA)
    df = df.replace("Yes", True)
    df = df.replace("No", False)

    return pd.DataFrame(df)


def extract_custom_vcf_key(filename: str) -> str:
    """
    Extract the key from custom VCF filename using regex pattern.

    Args:
        filename: The VCF filename to extract key from.

    Returns:
        Extracted key or original filename if pattern not found.

    Example:
        'merged-reads_20250505_0.1.3dev_agitated_feynman-LVE0001_NGA-2018-IRR-120-MK117940.1-CONSTRAINT_constraint.vcf.tsv'
        -> 'LVE0001_NGA-2018-IRR-120-MK117940.1'
    """
    # Pattern to match: (LVE\d+_[A-Z0-9\-_\.]+)-CONSTRAINT_constraint.vcf.tsv
    pattern = r'(LVE0\d+_[A-z0-9\-_\.]+)-CONSTRAINT_constraint\.vcf\.tsv'
    match = re.search(pattern, filename)

    if match:
        return match.group(1)
    else:
        logging.warning("Could not extract key from filename: %s", filename)
        # Fallback to original behavior
        return filename.replace("_constraint.vcf.tsv", "")


def extract_custom_vcf(file_path: Path) -> pd.DataFrame:
    """
    Extract custom VCF data from TSV file.

    Args:
        file_path: Path to the custom VCF TSV file.

    Returns:
        DataFrame with custom VCF data.
    """
    df = pd.read_csv(file_path, sep='\t')
    df["depth"] = df["A"] + df["C"] + df["G"] + df["T"]

    # Use vectorized division
    df['freqA'] = df['A'] / df['depth']
    df['freqC'] = df['C'] / df['depth']
    df['freqG'] = df['G'] / df['depth']
    df['freqT'] = df['T'] / df['depth']

    # Handle division by zero
    df = df.fillna(0)
    return df

def find_locations(directory: Path) -> Dict[str, Union[Path, List[Path]]]:
    """
    Find all relevant files in the given directory.

    Args:
        directory: Path to the directory to search.

    Returns:
        Dictionary mapping file types to their Paths.
    """
    locations = {}
    viralmetagenome = directory / "viralmetagenome"

    main_excel_path = directory/ "viralmetagenome" / "RUN1-6.rbind.xlsx"
    if main_excel_path.exists():
        logging.info("Found main Excel file: %s", main_excel_path)
        locations["mainexcel"] = main_excel_path
    else:
        logging.warning("Main Excel file not found in %s", directory)

    comparison_excels = directory / "comparison-excels" / "raw"
    if comparison_excels.exists() and comparison_excels.is_dir():
        logging.info("Found comparison Excel files: %s", comparison_excels.glob("*.xlsx"))
        locations["comparison_excels"] = [
            f for f in comparison_excels.glob("*.xlsx") if not f.name.startswith("~")
        ]
    else:
        logging.warning("No comparison Excel files found in %s", comparison_excels)

    custom_vcfs = [f for f in viralmetagenome.rglob("*/custom-vcf/*/*_constraint.vcf.tsv") if "seqruns-collapsed" not in str(f)]
    if custom_vcfs:
        logging.info("Found custom VCF file: %s", custom_vcfs[0])
        locations["custom_vcfs"] = custom_vcfs
    else:
        logging.warning("No custom VCF files found in %s", viralmetagenome)


    return locations


def write_dfs(output_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Write DataFrames to Parquet files in the specified output directory.

    Args:
        output_dfs: Dictionary mapping output names to DataFrames.
        output_dir: Path to the output directory.
    """
    for name, df in tqdm(output_dfs.items(), desc="Writing DataFrames"):
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(df, pd.Series) or isinstance(df, pd.DataFrame):

            if df.empty:
                logging.warning("DataFrame for '%s' is empty, skipping", name)
                continue

            output_path = output_dir / f"{name}.parquet"
            df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
            logging.debug("Data for '%s' saved to %s", name, output_path)

        elif isinstance(df, Dict):
            output_dir_name = output_dir / name
            write_dfs(df, output_dir_name)

        else:
            logging.warning("DataFrame for '%s' is of unsupported type, skipping", name)


def main(cli_args: argparse.Namespace) -> int:
    """
    Main function to handle the conversion process.

    Args:
        cli_args: Parsed command line arguments.
    """
    logging.basicConfig(level=cli_args.log_level.upper())

    output_dfs = {}

    # Find all relevant files
    locations = find_locations(Path(cli_args.input_dir))

    if not locations:
        logging.error("No relevant files found in %s", cli_args.input_dir)
        return 1

    if locations.get("mainexcel"):
        main_excel = locations["mainexcel"]
        if isinstance(main_excel, Path):
            output_dfs["mapping"] = extract_mapping(main_excel)
            output_dfs["reads"] = extract_reads(main_excel)

    if locations.get("comparison_excels"):
        comparison_excels = locations["comparison_excels"]
        if isinstance(comparison_excels, list):
            output_dfs["comparison_excels"] = {
                comp_excel.stem.replace("SeqID_analysis-outline_to-do_", ""): extract_comparison(comp_excel)
                for comp_excel in comparison_excels
            }
    if locations.get("custom_vcfs"):
        custom_vcfs = locations["custom_vcfs"]
        if isinstance(custom_vcfs, list):
            output_dfs["custom_vcfs"] = {
                extract_custom_vcf_key(custom_vcf.name): extract_custom_vcf(custom_vcf)
                for custom_vcf in tqdm(custom_vcfs, desc="Extracting custom VCFs")
            }

    if output_dfs:
        write_dfs(output_dfs, Path(cli_args.output_dir))
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
        default="/Users/joonklaps/Desktop/School/PhD/projects/LVE-BE002-PIPELINE/LVE-BE02-Supplmentary/results/HPC-results/TMP-RUN001-004/inrahost-analysis/data/",
        help="Input directory containing nf-core/viralmetagenome output files, & other optional supporting files"
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