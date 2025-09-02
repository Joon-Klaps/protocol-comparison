"""
Mapping summary statistics.

This module calculates statistics for read mapping efficiency and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import math

from ...base import DataManager
from ....sample_selection import label_for_sample

logger = logging.getLogger(__name__)


class MappingDataManager(DataManager):
    """Data manager specifically for mapping statistics."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.mapping_dir = self.data_path

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load mapping statistics data files.

        Returns:
            Dictionary containing mapping statistics DataFrames
        """
        data = {}

        # Load mapping statistics
        mapping_stats = self.mapping_dir / "mapping.parquet"
        if mapping_stats.exists():
            mapping_df = pd.read_parquet(mapping_stats)
            # Rename columns for easier handling
            mapping_df = mapping_df.rename(columns={
                '(samtools Raw) reads mapped (R1+R2)': 'reads_mapped',
                '(samtools Raw) reads mapped %': 'reads_mapped_pct',
                '(samtools Raw) reads unmapped (R1+R2)': 'reads_unmapped',
                '(samtools Raw) reads unmapped %': 'reads_unmapped_pct',
                '(umitools) deduplicated reads (R1,R2)': 'umi_mapping_reads',
                '(umitools) total UMIs': 'total_UMIs',
                '(umitools) unique UMIs': 'unique_UMIs',
                '(umitools) estimated PCR cycles': 'estimated_PCR_cycles'
            })
            mapping_df["umi_mapping_reads"] = mapping_df["umi_mapping_reads"] * 2
            data['mapping'] = mapping_df

        # Load annotation with sample categories if available
        annotation_fp = self.mapping_dir / "metadata.parquet"
        if annotation_fp.exists():
            try:
                annotation_df = pd.read_parquet(annotation_fp)
                # Normalize sample column name
                annotation_df = annotation_df.rename(columns={'LVE_SeqID': 'sample'})
                annotation_df = annotation_df[annotation_df['sample'].str.contains("LVE", na=False)]
                if annotation_df is not None:
                    data['annotation'] = annotation_df
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Failed to load annotation.parquet: %s", e)

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

    @staticmethod
    def _normalize_species_name(species: str) -> str:
        s = str(species).strip().upper()
        if 'LAS' in s:
            return 'LASV'
        if 'HAZ' in s:
            return 'HAZV'
        return s

    @staticmethod
    def _expected_species_for_category(category: str) -> List[str]:
        """Return expected species for a given sample category.

        - 'POS HAZV' -> ['HAZV']
        - contains 'LASSA' -> ['LASV']
        - contains 'NEG' -> [] (check both)
        """
        if not isinstance(category, str):
            return []
        c = category.strip().upper()
        if 'POS' in c or 'HAZ' in c:
            return ['HAZV']
        if 'LASSA' in c or 'LASV' in c:
            return ['LASV']
        if 'NEG' in c:
            return []
        return []

    # ------------------------------
    # Internal helpers (modularized)
    # ------------------------------
    def _normalize_mapping(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        df = mapping_df.copy()
        if 'sample' in df.columns:
            df['sample'] = df['sample'].astype(str).str.strip()
        # Normalize species
        df['species_norm'] = df['species'].map(self._normalize_species_name)
        return df

    def _aggregate_pct_by_sample_species(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        if 'reads_mapped_pct' not in mapping_df.columns:
            raise KeyError("reads_mapped_pct not present in mapping data")
        return (mapping_df.groupby(['sample', 'species_norm'])['reads_mapped_pct']
                .sum(min_count=1)
                .reset_index(name='pct'))

    def _get_annotation_with_category(self, all_samples: List[str]) -> pd.DataFrame:
        """Return annotation with columns: sample, category, expected_species.

        - Detect category column case-insensitively among common variants.
        - If missing or empty, category is set to '' (unknown) so both species are considered.
        """
        annotation_df = self.data.get('annotation')
        if annotation_df is None or annotation_df.empty:
            ann = pd.DataFrame({'sample': list(all_samples), 'category': [''] * len(all_samples)})
        else:
            ann = annotation_df.copy()
            # Ensure we have the canonical 'sample' column already normalized by load_data


            if 'sample' not in ann.columns and 'LVE_SeqID' in ann.columns:
                ann = ann.rename(columns={'LVE_SeqID': 'sample'})
            if 'category' not in ann.columns and 'Sample_Catagory' in ann.columns:
                ann = ann.rename(columns={'Sample_Catagory': 'category'})

            if 'sample' not in ann.columns and 'category' in ann.columns:
                ann = pd.DataFrame({'sample': list(all_samples), 'category': [''] * len(all_samples)})

            # Find a category-like column (case-insensitive)
            ann['sample'] = ann['sample'].astype(str).str.strip()
            # Normalize blanks/NaNs
            ann['category'] = ann['category'].astype(str).fillna('').str.strip()
            ann = ann.drop_duplicates(subset=['sample'], keep='first')
            # Align to provided sample list (preserve order)
            ann = ann.set_index('sample').reindex(all_samples).reset_index()
            ann['category'] = ann['category'].fillna('')

        # Expected species per sample from category
        ann['expected_species'] = ann['category'].apply(self._expected_species_for_category)
        return ann

    def _build_contamination_matrix(self, agg: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
        # Join expected species and keep off-target rows only
        merged = agg.merge(ann[['sample', 'expected_species']], on='sample', how='left')
        merged['is_contam'] = merged.apply(
            lambda r: (r['species_norm'] not in (r['expected_species'] or [])), axis=1
        )
        contam = merged[merged['is_contam']].copy()
        if contam.empty:
            # Build an empty matrix with LASV/HAZV cols and all samples as index
            matrix = pd.DataFrame(index=ann['sample'].tolist(), columns=['LASV', 'HAZV'])
            return matrix

        matrix = contam.pivot_table(index='sample', columns='species_norm', values='pct', aggfunc='sum')
        # Ensure all samples present (and desired columns)
        matrix = matrix.reindex(ann['sample'].tolist())
        desired_cols = ['LASV', 'HAZV']
        for col in desired_cols:
            if col not in matrix.columns:
                matrix[col] = pd.NA
        first_cols = [c for c in desired_cols if c in matrix.columns]
        other_cols = [c for c in matrix.columns if c not in first_cols]
        return matrix[first_cols + other_cols]

    def _flag_above_threshold(self, matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        flagged: List[Dict[str, Any]] = []
        for samp, row in matrix.iterrows():
            for sp, val in row.items():
                if pd.notna(val) and float(val) > threshold:
                    flagged.append({
                        'sample': str(samp),
                        'label': label_for_sample(str(samp)),
                        'species': sp,
                        'contamination_pct': float(val)
                    })
        return flagged

    def compute_contamination_metrics(self, sample_ids: Optional[List[str]] = None, threshold: float = 10.0) -> Dict[str, Any]:
        """Compute contamination percentages per sample/species and flag high contamination.

        Returns dict with keys:
        - matrix: DataFrame index=sample, columns species (e.g., LASV, HAZV) with contamination % (NaN where not applicable)
        - flagged: list of dicts with sample, label, species, contamination_pct for values > threshold
        - threshold: the threshold used
        - sample_category_map: dict mapping sample -> category (original string) for hover labels
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return {"matrix": pd.DataFrame(), "flagged": [], "threshold": threshold, "sample_category_map": {}}

        # Prepare and filter mapping data
        mapping_df = self._normalize_mapping(self.data["mapping"])
        if sample_ids:
            sample_ids = [str(s) for s in sample_ids]
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        if mapping_df.empty:
            return {"matrix": pd.DataFrame(), "flagged": [], "threshold": threshold, "sample_category_map": {}}

        try:
            agg = self._aggregate_pct_by_sample_species(mapping_df)
        except KeyError as e:
            logger.warning("%s; cannot compute contamination", e)
            return {"matrix": pd.DataFrame(), "flagged": [], "threshold": threshold, "sample_category_map": {}}

        # Build annotation (category + expected species), aligned to samples present in mapping
        all_samples = agg['sample'].astype(str).str.strip().unique().tolist()
        ann = self._get_annotation_with_category(all_samples)

        # Contamination matrix and flags
        matrix = self._build_contamination_matrix(agg, ann)
        flagged = self._flag_above_threshold(matrix, threshold)

        # Category map for hover
        sample_category_map = dict(zip(ann['sample'].astype(str), ann['category']))

        return {"matrix": matrix, "flagged": flagged, "threshold": threshold, "sample_category_map": sample_category_map}

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
                'mean_mapping_reads': math.floor(group['reads_mapped'].mean(skipna=True)),
                'mean_umi_mapping_reads': math.floor(group['umi_mapping_reads'].mean(skipna=True)),
                'mean_PCR_cycles': math.floor(group['estimated_PCR_cycles'].mean(skipna=True)),
                'min_mapping_reads': group['reads_mapped'].min(skipna=True),
                'min_mapping_reads_sample': label_for_sample(str(group.loc[min_idx, 'sample'])),
                'max_mapping_reads': group['reads_mapped'].max(skipna=True),
                'max_mapping_reads_sample': label_for_sample(str(group.loc[max_idx, 'sample'])),
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
