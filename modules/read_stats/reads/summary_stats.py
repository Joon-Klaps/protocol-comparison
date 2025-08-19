"""
Read processing summary statistics.

This module calculates statistics for read count changes through the processing pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import math

from ...base import DataManager

logger = logging.getLogger(__name__)


class ReadProcessingDataManager(DataManager):
    """Data manager specifically for read processing statistics."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.stats_dir = self.data_path

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load read processing data files.

        Returns:
            Dictionary containing read processing DataFrames
        """
        data = {}

        # Load read count summaries
        read_counts = self.stats_dir / "reads.parquet"
        if read_counts.exists():
            reads_df = pd.read_parquet(read_counts)
            # Rename columns for easier handling
            reads_df = reads_df.rename(columns={
                'FastQC (Raw). Seqs (R1,R2)': 'raw_reads',
                'FastQC (Post-trimming). Seqs (R1,R2)': 'post_trimming_reads',
                'FastQC (post-Host-removal). Seqs (R1,R2)': 'post_host_removal_reads'
            })
            data['reads'] = reads_df

        return data

    def get_available_samples(self) -> List[str]:
        """Get available sample IDs from read processing data."""
        samples = set()

        for _, df in self.load_data().items():
            if 'sample' in df.columns:
                samples.update(df['sample'].unique())
            elif 'sample_id' in df.columns:
                samples.update(df['sample_id'].unique())

        return sorted(list(samples))


class ReadProcessingSummaryStats:
    """
    Calculator for read processing pipeline statistics.

    Handles:
    - Read count changes through processing steps
    - Trimming efficiency analysis
    - Host removal efficiency analysis
    - Overall pipeline efficiency
    """

    def __init__(self, data_path: Path):
        """
        Initialize read processing summary stats calculator.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = ReadProcessingDataManager(data_path)
        self.data = self.data_manager.load_data()

    def calculate_efficiency_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate efficiency statistics for read processing pipeline.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of efficiency statistics
        """
        if "reads" not in self.data or self.data["reads"].empty:
            return {}

        reads_df = self.data["reads"]
        if sample_ids:
            reads_df = reads_df[reads_df['sample'].isin(sample_ids)]

        # Calculate read loss statistics
        stats = {
            'trimming_efficiency': {
                'mean_loss_pct': ((reads_df['raw_reads'] - reads_df['post_trimming_reads']) / reads_df['raw_reads'] * 100).mean(),
                'median_loss_pct': ((reads_df['raw_reads'] - reads_df['post_trimming_reads']) / reads_df['raw_reads'] * 100).median(),
                'std_loss_pct': ((reads_df['raw_reads'] - reads_df['post_trimming_reads']) / reads_df['raw_reads'] * 100).std(),
            },
            'host_removal_efficiency': {
                'mean_loss_pct': ((reads_df['post_trimming_reads'] - reads_df['post_host_removal_reads']) / reads_df['post_trimming_reads'] * 100).mean(),
                'median_loss_pct': ((reads_df['post_trimming_reads'] - reads_df['post_host_removal_reads']) / reads_df['post_trimming_reads'] * 100).median(),
                'std_loss_pct': ((reads_df['post_trimming_reads'] - reads_df['post_host_removal_reads']) / reads_df['post_trimming_reads'] * 100).std(),
            },
            'overall_efficiency': {
                'mean_retention_pct': (reads_df['post_host_removal_reads'] / reads_df['raw_reads'] * 100).mean(),
                'median_retention_pct': (reads_df['post_host_removal_reads'] / reads_df['raw_reads'] * 100).median(),
                'std_retention_pct': (reads_df['post_host_removal_reads'] / reads_df['raw_reads'] * 100).std(),
            },
            'sample_count': len(reads_df),
            'mean_raw_reads': math.floor(reads_df['raw_reads'].mean()),
            'mean_final_reads': math.floor(reads_df['post_host_removal_reads'].mean())
        }

        return stats

    def get_sample_details(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get detailed read counts for each sample.

        Args:
            sample_ids: Optional list of sample IDs to include

        Returns:
            DataFrame with sample details
        """
        if "reads" not in self.data or self.data["reads"].empty:
            return pd.DataFrame()

        reads_df = self.data["reads"].copy()
        if sample_ids:
            reads_df = reads_df[reads_df['sample'].isin(sample_ids)]

        # Add calculated columns
        reads_df['trimming_loss_pct'] = (reads_df['raw_reads'] - reads_df['post_trimming_reads']) / reads_df['raw_reads'] * 100
        reads_df['host_removal_loss_pct'] = (reads_df['post_trimming_reads'] - reads_df['post_host_removal_reads']) / reads_df['post_trimming_reads'] * 100
        reads_df['overall_retention_pct'] = reads_df['post_host_removal_reads'] / reads_df['raw_reads'] * 100

        return reads_df

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export read processing summary statistics to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export efficiency statistics
        stats = self.calculate_efficiency_stats(sample_ids)
        if stats:
            stats_df = pd.DataFrame.from_dict(stats, orient='index')
            stats_df.to_csv(output_path / 'read_processing_efficiency_summary.csv')

        # Export detailed sample data
        sample_details = self.get_sample_details(sample_ids)
        if not sample_details.empty:
            sample_details.to_csv(output_path / 'read_processing_sample_details.csv', index=False)

        logger.info("Read processing summary statistics exported to %s", output_path)
