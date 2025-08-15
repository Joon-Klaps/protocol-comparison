"""
Mapping summary statistics.

This module calculates statistics for read mapping efficiency and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ...base import DataManager

logger = logging.getLogger(__name__)


class MappingDataManager(DataManager):
    """Data manager specifically for mapping statistics."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.mapping_dir = self.data_path / "mapping"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load mapping statistics data files.

        Returns:
            Dictionary containing mapping statistics DataFrames
        """
        data = {}

        # Load mapping statistics
        mapping_stats = self.mapping_dir / "mapping.tsv"
        if mapping_stats.exists():
            mapping_df = pd.read_csv(mapping_stats, sep='\t')
            # Rename columns for easier handling
            mapping_df = mapping_df.rename(columns={
                '(samtools Raw) reads mapped (R1+R2)': 'reads_mapped',
                '(samtools Raw) reads mapped %': 'reads_mapped_pct',
                '(samtools Raw) reads unmapped (R1+R2)': 'reads_unmapped',
                '(samtools Raw) reads unmapped %': 'reads_unmapped_pct'
            })
            data['mapping'] = mapping_df

        return data

    def get_available_samples(self) -> List[str]:
        """Get available sample IDs from mapping data."""
        samples = set()

        for _, df in self.load_data().items():
            if 'sample' in df.columns:
                samples.update(df['sample'].unique())
            elif 'sample_id' in df.columns:
                samples.update(df['sample_id'].unique())

        return sorted(list(samples))


class MappingSummaryStats:
    """
    Calculator for read mapping statistics.

    Handles:
    - Mapping efficiency per species and segment
    - UMI mapping analysis
    - Contamination detection (HAZV vs LASV)
    - Hierarchical statistics by species-segment combinations
    """

    def __init__(self, data_path: Path):
        """
        Initialize mapping summary stats calculator.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = MappingDataManager(data_path)
        self.data = self.data_manager.load_data()

    def calculate_species_segment_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create mapping statistics overview grouped by segment and species.
        Filters out combinations where max reads mapped < 200.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            dict: Nested dictionary with statistics per species per segment
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return {}

        mapping_df = self.data["mapping"]
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        # Group by species and segment
        grouped = mapping_df.groupby(['species', 'segment'])

        stats = {}

        for (species, segment), group in grouped:
            # Calculate statistics for this species-segment combination
            min_idx = group['reads_mapped'].idxmin()
            max_idx = group['reads_mapped'].idxmax()

            reads_mapped_stats = {
                'mean_mapping_reads': group['reads_mapped'].mean(),
                'min_mapping_reads': group['reads_mapped'].min(),
                'min_mapping_reads_sample': group.loc[min_idx, 'sample'],
                'max_mapping_reads': group['reads_mapped'].max(),
                'max_mapping_reads_sample': group.loc[max_idx, 'sample'],
                'std_mapping_reads': group['reads_mapped'].std(),
                'sample_count': len(group)
            }

            # Filter out combinations where max reads mapped < 200
            if reads_mapped_stats['max_mapping_reads'] >= 200:
                # Initialize species in stats if not exists
                if species not in stats:
                    stats[species] = {}

                # Add segment stats for this species
                stats[species][segment] = reads_mapped_stats

        return stats

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export mapping summary statistics to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export species-segment statistics
        species_stats = self.calculate_species_segment_stats(sample_ids)
        if species_stats:
            # Flatten nested dictionary
            flat_stats = []
            for species, segments in species_stats.items():
                for segment, stats in segments.items():
                    row = {'species': species, 'segment': segment}
                    row.update(stats)
                    flat_stats.append(row)

            if flat_stats:
                stats_df = pd.DataFrame(flat_stats)
                stats_df.to_csv(output_path / 'mapping_species_segment_summary.csv', index=False)

        # Export detailed data
        if "mapping" in self.data and not self.data["mapping"].empty:
            mapping_df = self.data["mapping"]
            if sample_ids:
                mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]
            mapping_df.to_csv(output_path / 'mapping_detailed.csv', index=False)

        logger.info("Mapping summary statistics exported to %s", output_path)
