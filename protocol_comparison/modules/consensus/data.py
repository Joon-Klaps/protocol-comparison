"""
Consensus data management module.

This module provides data management functionality specifically for consensus analysis,
including loading alignment data and managing consensus-related datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

# Optional MSA backend (pyfamsa)
try:
    from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FamsaAligner = None  # type: ignore
    FamsaSequence = None  # type: ignore

from ..base import DataManager

logger = logging.getLogger(__name__)

# Global cache for consensus data managers to avoid duplicate loading
_consensus_data_cache = {}
_identity_matrix_cache = {}
_alignment_stats_cache = {}


def clear_consensus_cache():
    """Clear all global consensus data caches."""
    global _consensus_data_cache, _identity_matrix_cache, _alignment_stats_cache
    _consensus_data_cache = {}
    _identity_matrix_cache = {}
    _alignment_stats_cache = {}
    logger.info("All consensus data caches cleared")


class ConsensusDataManager(DataManager):
    """
    Data manager specifically for consensus analysis.

    Handles loading and management of consensus sequence data from alignment files,
    organized by assembly method (mapping/denovo), species (HAZV/LASV), and
    segments (S/M/L). Uses caching to avoid reloading data multiple times.
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus data manager with hierarchical structure support.

        Args:
            data_path: Path to the data directory containing alignment files
        """
        super().__init__(data_path)
        self.alignment_dir = self.data_path / "alignments"

        # Discover available alignment files and organize by hierarchy
        self.alignment_structure = self._discover_alignment_files()

        # Use cache key based on the entire structure
        self.cache_key = str(self.alignment_dir.absolute())

        # Check if we already have cached data for this structure
        if self.cache_key in _consensus_data_cache:
            logger.debug("Using cached consensus data for %s", self.cache_key)
            self._alignment_data = _consensus_data_cache[self.cache_key]['alignment_data']
            self._alignment_stats = _consensus_data_cache[self.cache_key]['alignment_stats']
            self._flat_data = _consensus_data_cache[self.cache_key]['flat_data']
        else:
            self._alignment_data = {}
            self._alignment_stats = {}
            self._flat_data = None

        # cache for on-the-fly realignments: key -> {sample_id: SeqRecord}
        self._realigned_cache: Dict[str, Dict[str, SeqRecord]] = {}

        self._validate_consensus_data_path()

    def _extract_sample_id(self, record_id: str) -> str:
        """
        Extract sample ID from FASTA record ID.

        Args:
            record_id: The FASTA record identifier

        Returns:
            Extracted sample ID
        """
        return record_id.replace("_R_", "").split("_")[0]

    def _validate_consensus_data_path(self) -> None:
        """Validate that the consensus data files exist."""
        if not self.alignment_dir.exists():
            logger.warning("Alignment directory does not exist: %s", self.alignment_dir)
            return

    def _discover_alignment_files(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        """
        Discover all alignment files in the hierarchical structure.

        Returns:
            Dictionary with structure: {method: {species: {segment: path}}}
        """
        structure = {}

        if not self.alignment_dir.exists():
            logger.warning("Alignment directory does not exist: %s", self.alignment_dir)
            return structure

        # Expected structure: ../alignments/{method}/{species}/{segment}/aligned.fasta
        for path in self.alignment_dir.rglob("aligned.fasta"):
            try:
                # Get the parts from the Path object
                # path.parts gives us a tuple of all path components
                # Or we can use path.parent to navigate up the directory tree

                segment = path.parent.name          # Directory containing aligned.fasta
                species = path.parent.parent.name   # Species directory
                method = path.parent.parent.parent.name  # Assembly method directory

                # Create nested dictionary structure if it doesn't exist
                if method not in structure:
                    structure[method] = {}

                if species not in structure[method]:
                    structure[method][species] = {}

                # Store the full file path
                structure[method][species][segment] = path

            except (IndexError, AttributeError) as e:
                logger.warning("Could not parse path structure for %s: %s", path, e)
                continue

        logger.info("Discovered %d assembly methods with alignment files", len(structure))
        return structure

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load consensus-related data files.
        Implementation of abstract method from base class.

        Returns:
            Dictionary containing consensus data DataFrames with flattened structure
        """
        if self._flat_data is None:
            logger.info("Loading consensus data for the first time for %s", self.cache_key)

            # Load alignment data and calculate statistics for all files
            self._load_all_alignment_data()

            # Create flat data structure for base class compatibility
            flat_data = {}

            # Cache all data
            _consensus_data_cache[self.cache_key] = {
                'alignment_data': self._alignment_data,
                'alignment_stats': self._alignment_stats,
                'flat_data': flat_data
            }
            self._flat_data = flat_data
        else:
            logger.debug("Using cached flat consensus data for %s", self.cache_key)

        return self._flat_data

    def _load_all_alignment_data(self) -> None:
        """Load alignment data for all discovered files using tuple-based structure."""
        self._alignment_data: Dict[Tuple[str, str, str], Dict[str, SeqRecord]] = {}

        for method, species_dict in self.alignment_structure.items():
            for species, segment_dict in species_dict.items():
                for segment, file_path in segment_dict.items():
                    # Load data for this combination
                    alignment_data = self._load_single_alignment_data(file_path)
                    if alignment_data:
                        # Store using tuple key
                        key = (method, species, segment)
                        self._alignment_data[key] = alignment_data

    def _load_single_alignment_data(self, alignment_file: Path) -> Optional[Dict[str, SeqRecord]]:
        """
        Load alignment data from a single FASTA file.

        Args:
            alignment_file: Path to the alignment file

        Returns:
            List of SeqRecord objects or None if loading fails
        """
        if not alignment_file.exists():
            logger.error("Alignment file not found: %s", alignment_file)
            return None

        try:
            # Parse sequences and handle duplicates by keeping only the first occurrence
            sequences = {}
            for record in SeqIO.parse(alignment_file, "fasta"):
                key = self._extract_sample_id(record.id)
                if key not in sequences:
                    sequences[key] = record
                else:
                    logger.debug("Duplicate sequence ID found, keeping first occurrence: %s", record.id)
            logger.info("Loaded %d sequences from alignment file %s", len(sequences.keys()), alignment_file.name)
            return sequences
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Error loading alignment file %s: %s", alignment_file, e)
            return None

    @property
    def alignment_data(self) -> Dict[Tuple[str, str, str], Dict[str, SeqRecord]]:
        """
        Lazily loads and returns the alignment data.

        Returns:
            Dictionary with structure: {(method, species, segment): {sample_id: SeqRecord}}
        """
        if not self._alignment_data:
            logger.info("Loading alignment data for the first time for %s", self.cache_key)
            self._load_all_alignment_data()

            # Update cache if needed
            if self.cache_key not in _consensus_data_cache:

                _consensus_data_cache[self.cache_key] = {
                    'alignment_data': self._alignment_data,
                    'alignment_stats': {},
                    'flat_data': {}
                }

        else:
            logger.debug("Using cached alignment data for %s", self.cache_key)

        return self._alignment_data

    def get_alignment_data_for(self, method: str, species: str, segment: str) -> Optional[Dict[str, SeqRecord]]:
        """
        Get alignment data for specific assembly method, species, and segment.

        Args:
            method: Assembly method (mapping/denovo)
            species: Species (HAZV/LASV)
            segment: Segment (S/M/L)

        Returns:
            Dict of SeqRecord objects or None
        """
        key = (method, species, segment)
        return self.alignment_data.get(key)

    def get_available_samples(self) -> List[str]:
        """
        Get available sample IDs from alignment data.
        Implementation of abstract method from base class.

        Returns:
            Sorted list of sample identifiers from all loaded alignment data
        """
        return self.get_available_samples_filtered(None, None, None)

    def get_available_samples_filtered(self, method: Optional[List[str]], species: Optional[List[str]], segment: Optional[List[str]]) -> List[str]:
        """
        Get available sample IDs from alignment data with optional filtering.

        Args:
            method: Optional list of assembly methods to filter by
            species: Optional list of species to filter by
            segment: Optional list of segments to filter by

        Returns:
            Sorted list of sample identifiers
        """
        sample_ids = set()
        if not self.alignment_data:
            return []

        for (m, sp, seg), sample_dict in self.alignment_data.items():
            # Apply filters if specified
            if method and m not in method:
                continue
            if species and sp not in species:
                continue
            if segment and seg not in segment:
                continue

            # Add all sample IDs from this combination
            sample_ids.update(sample_dict.keys())

        return sorted(sample_ids)

    def get_sequence_data(self, method: str, species: str, segment: str, sample_id: str) -> Optional[Any]:
        """
        Get sequence record for a specific sample.

        Args:
            sample_id: Sample identifier
            sequence_id: Optional specific sequence identifier

        Returns:
            SeqRecord object or None if not found
        """
        alignment_data = self.get_alignment_data_for(method, species, segment)
        return alignment_data.get(sample_id) if alignment_data else None

    def filter_alignment_by_samples(self, method: str, species: str, segment: str,
                                  sample_ids: List[str], remove_gap_columns: bool = True,
                                  realign: bool = True,
                                  guide_tree: str = "upgma") -> Optional[Dict[str, SeqRecord]]:
        """
        Filter alignment by specified sample IDs and optionally remove gap-only columns.

        Args:
            method: Assembly method ('mapping' or 'denovo')
            species: Species name (e.g., 'HAZV', 'LASV')
            segment: Segment name (e.g., 'S', 'M', 'L')
            sample_ids: List of sample IDs to include in the filtered alignment
            remove_gap_columns: Whether to remove columns that are all gaps in filtered sequences

        Returns:
            Filtered alignment as dict with {sample_id: SeqRecord} structure, or None if no data
        """
        # Get alignment data for the specified combination
        alignment_data = self.get_alignment_data_for(method, species, segment)
        if not alignment_data:
            logger.warning("No alignment data found for %s/%s/%s", method, species, segment)
            return None

        # If requested, realign the selected sequences on the fly using pyfamsa
        if realign and sample_ids:
            try:
                realigned = self._realign_subset((method, species, segment), sample_ids, guide_tree=guide_tree)
                if remove_gap_columns:
                    realigned = self._remove_gap_only_columns(realigned)
                return realigned
            except Exception as e:
                logger.warning("Realignment failed for %s/%s/%s; falling back to subset with optional gap filtering: %s",
                               method, species, segment, e)

        # Fallback: Filter sequences by sample IDs from original alignment
        # alignment_data is a dict with {sample_id: SeqRecord} structure
        filtered_alignment = {}
        for sample_id, record in alignment_data.items():
            if sample_id in sample_ids:
                filtered_alignment[sample_id] = record

        if not filtered_alignment:
            logger.warning("No sequences found for provided sample IDs")
            return None

        logger.debug("Filtered to %d sequences from %d original sequences",
                   len(filtered_alignment), len(alignment_data))

        # Remove gap-only columns if requested
        if remove_gap_columns:
            filtered_alignment = self._remove_gap_only_columns(filtered_alignment)

        return filtered_alignment

    def _realign_subset(self, key: Tuple[str, str, str], sample_ids: List[str], guide_tree: str = "upgma") -> Dict[str, SeqRecord]:
        """Realign a subset of sequences using pyfamsa and cache the result.

        Args:
            key: Tuple (method, species, segment)
            sample_ids: List of sample IDs to realign
            guide_tree: Guide tree method for FAMSA (e.g., 'upgma' or 'nj')

        Returns:
            Dict[sample_id, SeqRecord] of aligned sequences
        """
        method, species, segment = key
        alignment_data = self.get_alignment_data_for(method, species, segment)
        if not alignment_data:
            raise ValueError(f"No alignment data available for {key}")

        # Build a deterministic cache key
        ordered_ids = tuple(sample_ids)
        cache_key = f"realign::{method}::{species}::{segment}::{guide_tree}::{hash(ordered_ids)}"

        if cache_key in self._realigned_cache:
            logger.debug("Using cached realigned subset for %s", cache_key)
            return self._realigned_cache[cache_key]

        if FamsaAligner is None or FamsaSequence is None:
            raise ImportError("pyfamsa is not installed or failed to import")

        # Collect un-gapped sequences for the selected IDs
        famsa_inputs: List[Any] = []
        id_preserve: List[str] = []
        for sid in ordered_ids:
            if sid not in alignment_data:
                logger.debug("Sample %s not found in alignment for %s; skipping", sid, key)
                continue
            raw_seq = str(alignment_data[sid].seq).replace('-', '')
            famsa_inputs.append(FamsaSequence(sid.encode('utf-8'), raw_seq.encode('utf-8')))
            id_preserve.append(sid)

        if not famsa_inputs:
            raise ValueError("No valid samples found to realign")

        # Run alignment
        aligner = FamsaAligner(guide_tree=guide_tree)  # type: ignore[arg-type]
        msa = aligner.align(famsa_inputs)

        # Map back to SeqRecords preserving the requested order
        realigned: Dict[str, SeqRecord] = {}
        for seq in msa:
            sid = seq.id.decode('utf-8')
            aligned_seq = seq.sequence.decode('utf-8')
            # Use original record metadata if available
            if sid in alignment_data:
                orig = alignment_data[sid]
                realigned[sid] = SeqRecord(Seq(aligned_seq), id=orig.id, description=orig.description)
            else:
                realigned[sid] = SeqRecord(Seq(aligned_seq), id=sid, description="")

        # Ensure order respects requested sample_ids where present
        ordered_realigned = {sid: realigned[sid] for sid in id_preserve if sid in realigned}

        # Cache and return
        self._realigned_cache[cache_key] = ordered_realigned
        logger.info("Realigned %d sequences for %s/%s/%s", len(ordered_realigned), method, species, segment)
        return ordered_realigned

    def filter_all_alignments_by_samples(self, sample_ids:List[str]) -> Dict[str, Dict[str, Dict[str, Optional[Dict[str, SeqRecord]]]]]:
        """
        Filter all alignments by specified sample IDs.

        Args:
            sample_ids: List of sample IDs to include in the filtered alignments

        Returns:
            Dictionary containing filtered alignments for each method/species/segment combination
            with structure: {method: {species: {segment: {sample_id: SeqRecord}}}}
        """
        filtered_alignments = {}
        for method, species_dict in self.alignment_structure.items():
            for species, segment_dict in species_dict.items():
                for segment, _ in segment_dict.items():
                    filtered = self.filter_alignment_by_samples(method, species, segment, sample_ids)
                    if filtered:
                        filtered_alignments.setdefault(method, {}).setdefault(species, {})[segment] = filtered
        return filtered_alignments

    def _remove_gap_only_columns(self, alignment: Dict[str, SeqRecord]) -> Dict[str, SeqRecord]:
        """
        Remove columns that contain only gaps ('-') from alignment.

        Args:
            alignment: Dict with {sample_id: SeqRecord} structure

        Returns:
            Alignment with gap-only columns removed, maintaining the same structure
        """
        if not alignment:
            return alignment

        try:
            # Convert sequences to numpy array for efficient processing
            sample_ids = list(alignment.keys())
            sequences = np.array([list(str(alignment[sample_id].seq)) for sample_id in sample_ids])

            # Find columns that are not all gaps
            non_gap_columns = ~np.all(sequences == '-', axis=0)

            # Filter sequences
            filtered_sequences = sequences[:, non_gap_columns]

            # Create new SeqRecord objects maintaining the dictionary structure
            filtered_alignment = {}
            for i, sample_id in enumerate(sample_ids):
                original_record = alignment[sample_id]
                new_seq = ''.join(filtered_sequences[i])
                new_record = SeqRecord(
                    Seq(new_seq),
                    id=original_record.id,
                    description=original_record.description
                )
                filtered_alignment[sample_id] = new_record

            logger.debug("Removed gap-only columns: %d -> %d positions",
                       sequences.shape[1], filtered_sequences.shape[1])
            return filtered_alignment

        except ImportError:
            logger.error("NumPy not available - cannot filter gap columns")
            return alignment
        except (ValueError, TypeError) as e:
            logger.error("Error filtering gap columns: %s", e)
            return alignment

    def compute_pairwise_identity_matrix(self, alignment: Dict[str, SeqRecord]) -> np.ndarray:
        """
        Compute pairwise identity matrix with caching.

        Args:
            alignment: Dictionary with {sample_id: SeqRecord} structure

        Returns:
            Identity matrix as numpy array
        """
        if not alignment:
            return np.array([])

        # Create cache key based on sample IDs and their sequences
        sample_ids = sorted(alignment.keys())
        cache_key = f"global_pid_{hash(tuple(sample_ids))}_{hash(tuple(str(alignment[sid].seq) for sid in sample_ids))}"

        # Check cache first
        if cache_key in _identity_matrix_cache:
            logger.debug("Using cached global PID matrix for %d samples", len(sample_ids))
            return _identity_matrix_cache[cache_key]

        # Get all sample IDs
        num_samples = len(sample_ids)

        # Initialize identity matrix
        identity_matrix = np.zeros((num_samples, num_samples), dtype=float)

        # Compute pairwise identity
        for i, sample1 in enumerate(sample_ids):
            for j, sample2 in enumerate(sample_ids):
                if i == j:
                    identity_matrix[i, j] = 1.0
                elif j < i:
                    val = self.compute_sequence_identity(alignment[sample1], alignment[sample2])
                    identity_matrix[i, j] = val
                    identity_matrix[j, i] = val

        # Cache the result
        _identity_matrix_cache[cache_key] = identity_matrix
        logger.debug("Cached global PID matrix for %d samples", len(sample_ids))

        return identity_matrix

    def compute_sequence_identity(self, seq_record1: SeqRecord, seq_record2: SeqRecord) -> float:
        """
        Compute the identity between aligned sequence records using numpy.
        """
        seq1 = np.array(list(str(seq_record1.seq)))
        seq2 = np.array(list(str(seq_record2.seq)))

        # Check if sequences are empty
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0

        # Compute identity
        matches = np.sum(seq1 == seq2)
        total = len(seq1)
        return matches / total if total > 0 else 0.0

    def compute_sequence_identity_local(self, seq_record1: SeqRecord, seq_record2: SeqRecord) -> float:
        """
        Compute the local identity between aligned sequence records, ignoring positions with gaps ('-') or 'N'.

        Args:
            seq_record1: First sequence record
            seq_record2: Second sequence record

        Returns:
            Local identity score (0.0 to 1.0)
        """
        seq1 = np.array(list(str(seq_record1.seq)))
        seq2 = np.array(list(str(seq_record2.seq)))

        # Check if sequences are empty
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0

        # Create mask for positions that are not gaps or N (case insensitive)
        valid_positions = ~np.isin(seq1, ['-', 'N', 'n']) & ~np.isin(seq2, ['-', 'N', 'n'])

        # If no valid positions, return 0
        try:
            if not np.any(valid_positions):
                return 0.0
        except ValueError:
            # Handle ambiguous truth value error
            if np.sum(valid_positions) == 0:
                return 0.0

        # Compute identity only on valid positions
        matches = np.sum(seq1[valid_positions] == seq2[valid_positions])
        total_valid = np.sum(valid_positions)
        return matches / total_valid if total_valid > 0 else 0.0

    def compute_pairwise_identity_matrix_local(self, alignment: Dict[str, SeqRecord]) -> np.ndarray:
        """
        Compute pairwise identity matrix using local identity (ignoring gaps and Ns) with caching.

        Args:
            alignment: Dictionary with {sample_id: SeqRecord} structure

        Returns:
            Identity matrix as numpy array
        """
        if not alignment:
            return np.array([])

        # Create cache key based on sample IDs and their sequences
        sample_ids = sorted(alignment.keys())
        cache_key = f"local_pid_{hash(tuple(sample_ids))}_{hash(tuple(str(alignment[sid].seq) for sid in sample_ids))}"

        # Check cache first
        if cache_key in _identity_matrix_cache:
            logger.debug("Using cached local PID matrix for %d samples", len(sample_ids))
            return _identity_matrix_cache[cache_key]

        # Get all sample IDs
        num_samples = len(sample_ids)

        # Initialize identity matrix
        identity_matrix = np.zeros((num_samples, num_samples), dtype=float)

        # Compute pairwise identity
        for i, sample1 in enumerate(sample_ids):
            for j, sample2 in enumerate(sample_ids):
                if i == j:
                    identity_matrix[i, j] = 1.0
                elif j < i:
                    val = self.compute_sequence_identity_local(alignment[sample1], alignment[sample2])
                    identity_matrix[i, j] = val
                    identity_matrix[j, i] = val

        # Cache the result
        _identity_matrix_cache[cache_key] = identity_matrix
        logger.debug("Cached local PID matrix for %d samples", len(sample_ids))

        return identity_matrix

    def format_alignment_for_dash_bio(self, alignment: Dict[str, SeqRecord]) -> str:
        """
        Format alignment data as FASTA string for dash_bio AlignmentChart.

        Args:
            alignment: Dict with {sample_id: SeqRecord} structure

        Returns:
            FASTA format string for dash_bio
        """
        fasta_lines = []
        for sample_id, record in alignment.items():
            fasta_lines.append(f">{sample_id}")
            fasta_lines.append(str(record.seq))
        return '\n'.join(fasta_lines)

    def get_most_divergent_sample(self, method: str, species: str, segment: str, sample_ids: Optional[List[str]] = None) -> Optional[str]:
        """
        Find the most divergent sample based on lowest mean identity with others.

        Args:
            method: Assembly method
            species: Species name
            segment: Segment name
            sample_ids: Optional list of sample IDs to consider

        Returns:
            Sample ID of most divergent sample or None
        """
        alignment_data = self.get_alignment_data_for(method, species, segment)
        if not alignment_data:
            return None

        # Filter by sample_ids if provided
        if sample_ids:
            alignment_data = {sid: record for sid, record in alignment_data.items() if sid in sample_ids}

        if len(alignment_data) < 2:
            return None

        try:
            identity_matrix = self.compute_pairwise_identity_matrix(alignment_data)
            sample_ids_list = list(alignment_data.keys())

            # Calculate mean identity for each sample (excluding self-comparison)
            mean_identities = []
            for i in range(len(sample_ids_list)):
                # Get all identities for this sample except diagonal
                identities = [identity_matrix[i, j] for j in range(len(sample_ids_list)) if i != j]
                mean_identity = np.mean(identities) if identities else 0.0
                mean_identities.append(mean_identity)

            # Find sample with lowest mean identity
            most_divergent_idx = np.argmin(mean_identities)
            return sample_ids_list[most_divergent_idx]

        except Exception as e:
            logger.error("Error finding most divergent sample: %s", e)
            return None

    def get_alignment_summary_stats(self, tuple_key: Tuple[str, str, str], sample_ids: List[str]) -> Dict[str, Any]:
        """
        Get summary statistics for an alignment with caching.

        Args:
            tuple_key: Tuple of (method, species, segment)
            sample_ids: List of sample IDs to consider

        Returns:
            Dictionary with alignment statistics
        """
        method, species, segment = tuple_key

        # Create cache key based on the combination and sample IDs
        sorted_sample_ids = sorted(sample_ids) if sample_ids else []
        cache_key = f"stats_{method}_{species}_{segment}_{hash(tuple(sorted_sample_ids))}"

        # Check cache first
        if cache_key in _alignment_stats_cache:
            logger.debug("Using cached alignment stats for %s", tuple_key)
            return _alignment_stats_cache[cache_key]

        alignment_data = self.filter_alignment_by_samples(method, species, segment, sample_ids=sample_ids)
        if not alignment_data:
            _alignment_stats_cache[cache_key] = {}
            return {}

        # Calculate basic stats
        first_seq = next(iter(alignment_data.values()))
        alignment_length = len(str(first_seq.seq)) if first_seq.seq else 0
        num_samples = len(alignment_data)

        most_divergent = "NA"
        # Find most divergent sample
        if len(alignment_data.keys()) > 2:
            most_divergent = self.get_most_divergent_sample(method, species, segment, list(alignment_data.keys()))

        stats = {
            'Alignment Length': alignment_length,
            'Number of Samples': num_samples,
            'Most Divergent Sample': most_divergent,
        }

        # Cache the result
        _alignment_stats_cache[cache_key] = stats
        logger.debug("Cached alignment stats for %s", tuple_key)

        return stats


    def get_available_combinations(self) -> List[Dict[str, str]]:
        """
        Get all available assembly method/species/segment combinations.

        Returns:
            List of dictionaries with 'method', 'species', 'segment' keys
        """
        combinations = []
        for (method, species, segment) in self.alignment_data.keys():
            combinations.append({
                'method': method,
                'species': species,
                'segment': segment
            })
        return combinations

    def get_available_combination_tuples(self) -> List[Tuple[str, str, str]]:
        """
        Get all available assembly method/species/segment combinations as tuples.

        Returns:
            List of (method, species, segment) tuples
        """
        return list(self.alignment_data.keys())

    def get_samples_for_combination(self, method: str, species: str, segment: str) -> List[str]:
        """
        Get sample IDs available for a specific assembly method/species/segment combination.

        Args:
            method: Assembly method ('mapping' or 'denovo')
            species: Species name (e.g., 'HAZV', 'LASV')
            segment: Segment name (e.g., 'S', 'M', 'L')

        Returns:
            List of sample IDs available for this combination
        """
        # Ensure data is loaded
        self.load_data()
        return self.get_available_samples_filtered([method], [species], [segment])

    def export_filtered_alignment(self, method: str, species: str, segment: str,
                                sample_ids: List[str], output_path: Path,
                                remove_gap_columns: bool = True) -> bool:
        """
        Filter alignment and export to FASTA file.

        Args:
            method: Assembly method ('mapping' or 'denovo')
            species: Species name (e.g., 'HAZV', 'LASV')
            segment: Segment name (e.g., 'S', 'M', 'L')
            sample_ids: List of sample IDs to include
            output_path: Path where to save the filtered alignment
            remove_gap_columns: Whether to remove gap-only columns

        Returns:
            True if export was successful, False otherwise
        """
        # Get filtered alignment
        filtered_alignment = self.filter_alignment_by_samples(
            method, species, segment, sample_ids, remove_gap_columns
        )

        if not filtered_alignment:
            logger.error("No filtered alignment data to export")
            return False

        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to FASTA file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Convert dictionary to list of SeqRecord objects for SeqIO.write
                sequences_to_write = list(filtered_alignment.values())
                SeqIO.write(sequences_to_write, f, "fasta")

            logger.info("Exported %d sequences to %s", len(filtered_alignment), output_path)
            return True

        except Exception as e:
            logger.error("Error exporting alignment to %s: %s", output_path, e)
            return False

    def filter_gap_columns(self, method: str, species: str, segment: str,
                          sample_ids: Optional[List[str]] = None) -> Optional[Dict[str, SeqRecord]]:
        """
        Remove columns that contain only gaps ('-') from alignment.

        Args:
            method: Assembly method ('mapping' or 'denovo')
            species: Species name (e.g., 'HAZV', 'LASV')
            segment: Segment name (e.g., 'S', 'M', 'L')
            sample_ids: Optional list of sample IDs to include in filtering

        Returns:
            Filtered alignment with gap-only columns removed, maintaining {sample_id: SeqRecord} structure
        """
        # Get alignment data for the specified combination
        alignment_data = self.get_alignment_data_for(method, species, segment)
        if alignment_data is None:
            return None

        # Filter by sample IDs if provided
        if sample_ids:
            filtered_alignment_data = {sample_id: record for sample_id, record in alignment_data.items()
                                     if sample_id in sample_ids}
        else:
            # Use all sequences
            filtered_alignment_data = alignment_data.copy()

        if not filtered_alignment_data:
            return None

        # Use the existing _remove_gap_only_columns method which now works with dictionaries
        return self._remove_gap_only_columns(filtered_alignment_data)


