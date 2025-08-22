#! /usr/bin/env python3
"""
A script to convert output files from nf-core/viralmetagenome to Parquet format for optimized analysis platform performance.
"""

import pandas as pd
import logging
import sys
import argparse
import re
import subprocess
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

def extract_contigs(file_path: Path) -> pd.DataFrame:
    """
    Extract contig statistics from the main DataFrame.

    Args:
        file_path: Path to the Excel file.

    Returns:
        DataFrame with contig statistics.
    """
    df = load_excel(file_path, "contigs")

    columns_of_interest = [
        "sample",
        "cluster",
        "step",
        "(annotation) acronym",
        "(annotation) segment",
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

    contig_stats = df[columns_of_interest].copy()
    return pd.DataFrame(contig_stats)


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

def find_locations(directory: Path) -> Dict[str, Union[Path, List[Path], Dict]]:
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

    if custom_vcfs := [f for f in viralmetagenome.rglob("*/custom-vcf/*/*_constraint.vcf.tsv") if "seqruns-collapsed" not in str(f)]:
        logging.info("Found custom VCF file: %s", custom_vcfs[0])
        locations["custom_vcfs"] = custom_vcfs
    else:
        logging.warning("No custom VCF files found in %s", viralmetagenome)

    if denovo_seq_lasv := [f for f in viralmetagenome.rglob("*/lasvdedup-out/dedup/**/good/*.fasta") if "seqruns-collapsed" not in str(f)]:
        # Sanity check: within every 'good' directory, there should only be a single sequence
        good_dirs = {}
        for f in denovo_seq_lasv:
            good_dir = f.parent
            good_dirs.setdefault(str(good_dir), []).append(f)

        multiple_seqs = [files for files in good_dirs.values() if len(files) > 1]

        if multiple_seqs:
            logging.warning("Multiple sequences found in some 'good' directories. Attempting to filter using annotations.")

            if annotations := [f for f in viralmetagenome.rglob("*/lasvdedup-out/dedup/*.figtree.ann")]:
                annotation_df = pd.concat([pd.read_csv(f, sep='\t') for f in annotations], ignore_index=True)
                good_df = annotation_df[annotation_df["classification"] == "good"]
                good_df["clean_name"] = good_df["sequence_name"].apply(lambda x: x.replace("_R_", "").split(".")[0])

                # Filter the original list based on annotations
                denovo_seq_lasv = [f for f in denovo_seq_lasv if any(contig in str(f) for contig in good_df["clean_name"])]
                logging.info("Filtered sequences based on annotations. Remaining: %s", len(denovo_seq_lasv))

                # Re-check after filtering
                good_dirs = {}
                for f in denovo_seq_lasv:
                    good_dir = f.parent
                    good_dirs.setdefault(str(good_dir), []).append(f)

                multiple_seqs = [files for files in good_dirs.values() if len(files) > 1]

                if multiple_seqs:
                    logging.error("Multiple sequences still found after filtering:")
                    for files in multiple_seqs:
                        logging.error([str(f) for f in files])
                    sys.exit(1)
                else:
                    logging.info("Successfully resolved multiple sequences using annotations.")
            else:
                logging.error("Multiple sequences found but no annotation files available for filtering:")
                for files in multiple_seqs:
                    logging.error([str(f) for f in files])
                sys.exit(1)
        else:
            logging.info("Sanity check passed: Single sequence found in each 'good' directory.")

        # Now create the L and S lists from the (potentially filtered) sequence list
        dn_lasv_l = [f for f in denovo_seq_lasv if "LASV-L" in str(f)]
        dn_lasv_s = [f for f in denovo_seq_lasv if "LASV-S" in str(f)]

        locations["alignment"] = {
            "denovo": {"LASV": {
                "L": dn_lasv_l,
                "S": dn_lasv_s
            }}
        }
    else:
        logging.warning("No de novo sequence files found for LASV  %s", viralmetagenome)

    if seq_all := [f for f in viralmetagenome.rglob("*.consensus.fasta") if "seqruns-collapsed" not in str(f)]:
        logging.info("Found de sequence files: %s", len(seq_all))
        hazv_seq_all = [f for f in seq_all if "it2" in str(f)]
        mapping_seq_all = [f for f in seq_all if "constraint" in str(f)]
        locations["alignment"]["denovo"]["HAZV"] = {
            "L": hazv_seq_all, "S": hazv_seq_all, "M": hazv_seq_all
        }
        locations["alignment"]["mapping"] = {
            "LASV": {"S": mapping_seq_all, "L": mapping_seq_all},
            "HAZV": {"S": mapping_seq_all, "L": mapping_seq_all, "M": mapping_seq_all}
        }
    else:
        logging.warning("No de novo sequence files found for it2 dir %s", viralmetagenome)

    return locations


def generate_alignments(alignment: Dict, contig_df: pd.DataFrame, mapping_df: pd.DataFrame, output_dir: Path):
    """Extract alignment data from FASTA files."""

    BLACKLIST_SAMPLES = ["LVE00288", "LVE00290", "LVE00385"]

    # Step 1: Create three lookup dataframes
    hazv_contigs = contig_df[(contig_df["(annotation) acronym"] == "HAZV") & (~contig_df["sample"].isin(BLACKLIST_SAMPLES))].copy()
    lasv_mapping = mapping_df[(mapping_df["species"] == "LASV") & (~mapping_df["sample"].isin(BLACKLIST_SAMPLES))].copy()
    hazv_mapping = mapping_df[(mapping_df["species"] == "HAZV") & (~mapping_df["sample"].isin(BLACKLIST_SAMPLES))].copy()

    logging.debug(f"HAZV contigs: {hazv_contigs.shape}")
    logging.debug(f"LASV mapping: {lasv_mapping.shape}")
    logging.debug(f"HAZV mapping: {hazv_mapping.shape}")

    # Step 2: Create alignment_name column for each dataframe
    hazv_contigs['alignment_name'] = hazv_contigs['sample'] + '_' + hazv_contigs['cluster']
    lasv_mapping['alignment_name'] = lasv_mapping['sample'] + "_" + lasv_mapping['cluster']
    hazv_mapping['alignment_name'] = hazv_mapping['sample'] + "_" + hazv_mapping['cluster']

    logging.debug(f"Sample HAZV contig alignment_names: {hazv_contigs['alignment_name'].head().tolist()}")
    logging.debug(f"Sample LASV mapping alignment_names: {lasv_mapping['alignment_name'].head().tolist()}")
    logging.debug(f"Sample HAZV mapping alignment_names: {hazv_mapping['alignment_name'].head().tolist()}")

    # Step 3: For each segment, create sets of alignment names
    def get_alignment_names_by_segment(df, segment_col):
        """Get alignment names grouped by segment."""
        segments = {}
        for segment in ['L', 'S', 'M']:
            segment_df = df[df[segment_col] == segment]
            segments[segment] = set(segment_df['alignment_name'])  # For contigs, use sample_cluster
        return segments

    hazv_contig_segments = get_alignment_names_by_segment(hazv_contigs, '(annotation) segment')
    lasv_mapping_segments = get_alignment_names_by_segment(lasv_mapping, 'segment')
    hazv_mapping_segments = get_alignment_names_by_segment(hazv_mapping, 'segment')

    logging.debug(f"HAZV contig segments: {dict((k, len(v)) for k, v in hazv_contig_segments.items())}")
    logging.debug(f"LASV mapping segments: {dict((k, len(v)) for k, v in lasv_mapping_segments.items())}")
    logging.debug(f"HAZV mapping segments: {dict((k, len(v)) for k, v in hazv_mapping_segments.items())}")


    def filter_paths_by_alignment_names(paths: list, valid_alignment_names: set) -> list:
        """Filter paths to only include those with valid alignment names."""
        return [l for l in paths if any(s in str(l) for s in valid_alignment_names)]

    # Step 5: Apply filtering to each alignment category
    result = {}

    if denovo := alignment.get("denovo"):
        result["denovo"] = {}

        # LASV denovo - DON'T filter, just pass through all files
        if "LASV" in denovo:
            result["denovo"]["LASV"] = {}
            for segment in ['L', 'S']:  # LASV only has L and S segments
                if segment in denovo["LASV"]:
                    # No filtering for LASV denovo - pass through all files
                    result["denovo"]["LASV"][segment] = denovo["LASV"][segment]
                    logging.debug(f"LASV denovo {segment}: {len(denovo['LASV'][segment])} files (no filtering)")

        # HAZV denovo - filter by segments using sample_cluster pattern
        if "HAZV" in denovo:
            result["denovo"]["HAZV"] = {}
            for segment in ['L', 'S', 'M']:
                if segment in denovo["HAZV"]:
                    valid_names = hazv_contig_segments[segment]
                    filtered_paths = filter_paths_by_alignment_names(denovo["HAZV"][segment], valid_names)
                    result["denovo"]["HAZV"][segment] = filtered_paths
                    logging.debug(f"HAZV denovo {segment}: {len(filtered_paths)} files from {len(denovo['HAZV'][segment])} total")

    if mapping := alignment.get("mapping"):
        result["mapping"] = {}

        # LASV mapping - filter by sample names only
        if "LASV" in mapping:
            result["mapping"]["LASV"] = {}
            for segment in ['L', 'S']:
                if segment in mapping["LASV"]:
                    valid_samples = lasv_mapping_segments[segment]
                    filtered_paths = filter_paths_by_alignment_names(mapping["LASV"][segment], valid_samples)
                    result["mapping"]["LASV"][segment] = filtered_paths
                    logging.debug(f"LASV mapping {segment}: {len(filtered_paths)} files from {len(mapping['LASV'][segment])} total")

        # HAZV mapping - filter by sample names only
        if "HAZV" in mapping:
            result["mapping"]["HAZV"] = {}
            for segment in ['L', 'S', 'M']:
                if segment in mapping["HAZV"]:
                    valid_samples = hazv_mapping_segments[segment]
                    filtered_paths = filter_paths_by_alignment_names(mapping["HAZV"][segment], valid_samples)
                    result["mapping"]["HAZV"][segment] = filtered_paths
                    logging.debug(f"HAZV mapping {segment}: {len(filtered_paths)} files from {len(mapping['HAZV'][segment])} total")

    return align_sequences(result, output_dir)

def align_sequences(data: Dict, output_dir: Path):
    """Recursively align sequences using MAFFT."""
    logging.info("Starting sequence alignment")

    def normalize_sequence_name(seq_name):
        """Remove '_R_' prefix from sequence name for comparison."""
        return seq_name.replace('_R_', '')

    def get_sequence_names_from_fasta(file_path):
        """Extract sequence names from a FASTA file, normalized."""
        names = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith('>'):
                        seq_name = line[1:].strip()
                        names.add(normalize_sequence_name(seq_name))
        except FileNotFoundError:
            pass
        return names

    def sequences_match(temp_input_path, output_file_path):
        """Check if sequences in output file match those in temp input."""
        if not output_file_path.exists():
            return False

        input_names = get_sequence_names_from_fasta(temp_input_path)
        output_names = get_sequence_names_from_fasta(output_file_path)

        return input_names == output_names

    def process_item(item, path_parts=None):
        if path_parts is None:
            path_parts = []

        if isinstance(item, list):
            # Found sequences to align
            if len(item) <= 1:
                return item[0] if item else None

            # Create temporary multi-FASTA file by concatenating all input files
            temp_input = output_dir / "/".join(path_parts + ["temp_input.fasta"])
            output_file = output_dir / "/".join(path_parts + ["aligned.fasta"])
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Concatenate all FASTA files into one
            with open(temp_input, "w", encoding="utf-8") as temp_f:
                for fasta_path in item:
                    with open(fasta_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        temp_f.write(content)
                        if not content.endswith('\n'):
                            temp_f.write('\n')

            # Check if output already exists and contains the same sequences
            if sequences_match(temp_input, output_file):
                logging.info(f"Output file {output_file} already exists with matching sequences, skipping alignment")
                temp_input.unlink()  # Clean up temporary file
                return output_file

            # Run MAFFT alignment on the combined file
            logging.info(f"Running MAFFT alignment for {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                subprocess.run(["mafft","--adjustdirection", "--thread", "8", str(temp_input)], stdout=f, check=True)

            # Clean up temporary file
            temp_input.unlink()
            return output_file

        elif isinstance(item, dict):
            # Recursively process dictionary
            return {key: process_item(value, path_parts + [key])
                   for key, value in item.items()}

        else:
            # Single file, return as-is
            return item

    return process_item(data)

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
            # logging.debug("Data for '%s' saved to %s", name, output_path)

        elif isinstance(df, Dict):
            output_dir_name = output_dir / name
            write_dfs(df, output_dir_name)

        else:
            logging.warning("DataFrame for '%s' is of unsupported type, skipping", name)


def main(cli_args: argparse.Namespace) -> int:
    """
    Main function to handle the convearsion process.

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
            output_dfs["contigs"] = extract_contigs(main_excel)
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
    if (locations.get("alignment") and
    "mapping" in output_dfs and not output_dfs["mapping"].empty and
    "contigs" in output_dfs and not output_dfs["contigs"].empty):
        alignment_data = locations["alignment"]
        if isinstance(alignment_data, dict):
            alignments = generate_alignments(alignment_data, output_dfs["contigs"], output_dfs["mapping"], output_dir=Path(cli_args.output_dir) / "alignments")
            logging.info("Alignment completed")

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