#! /usr/bin/env python3
"""
A script to convert output files from nf-core/viralmetagenome to the formats needed for this analysis platform.
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
import sys


def load_main_excel(file_path: Path) -> pd.DataFrame:
    """
    Load the main Excel file containing sample information.

    Args:
        file_path: Path to the main Excel file.

    Returns:
        DataFrame with sample information.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=None)
        return df['main']
    except KeyError:
        logging.error(f"Sheet 'main' not found in {file_path}")
        sys.exit(1)

def extract_mapping_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract mapping statistics from the main DataFrame.

    Args:
        df: DataFrame containing sample information.

    Returns:
        DataFrame with mapping statistics.
    """
    if 'mapping_stats' not in df.columns:
        logging.error("Column 'mapping_stats' not found in the main DataFrame")
        sys.exit(1)

    return df

    # mapping_stats = df[['sample_id', 'mapping_stats']].dropna()
    # mapping_stats['mapping_stats'] = mapping_stats['mapping_stats'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # return pd.json_normalize(mapping_stats['mapping_stats']).assign(sample_id=mapping_stats['sample_id'])