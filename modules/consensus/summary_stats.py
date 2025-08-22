"""
Summary statistics for consensus sequence analysis.
Simplified module focusing on basic alignment statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..data import ConsensusDataManager

def calculate_alignment_summaries(
    data_manager: ConsensusDataManager,
    selected_keys: List[Tuple[str, str, str]],
    selected_samples: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate simplified alignment summary statistics.

    Args:
        data_manager: ConsensusDataManager instance with tuple-based structure
        selected_keys: List of (method, species, segment) tuples
        selected_samples: Optional list of sample IDs to include

    Returns:
        Dictionary with alignment statistics
    """
    summaries = {}

    for key in selected_keys:
        alignment_data = data_manager.alignment_data.get(key, {})

        if not alignment_data:
            continue

        # Filter samples if specified
        if selected_samples:
            alignment_data = {
                sample_id: seq_record
                for sample_id, seq_record in alignment_data.items()
                if sample_id in selected_samples
            }

        if not alignment_data:
            continue

        # Get basic stats using data manager helper
        stats = data_manager.get_alignment_summary_stats(key, list(alignment_data.keys()))

        # Add key information
        method, species, segment = key
        stats['method'] = method
        stats['species'] = species
        stats['segment'] = segment

        summaries[f"{method}_{species}_{segment}"] = stats

    return summaries


def get_overall_summary(summaries: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create overall summary comparing methods.

    Args:
        summaries: Dictionary of alignment summaries

    Returns:
        Overall summary statistics
    """
    if not summaries:
        return {}

    # Group by method for comparison
    methods = {}
    for key, stats in summaries.items():
        method = stats.get('method', 'unknown')
        if method not in methods:
            methods[method] = []
        methods[method].append(stats)

    overall = {
        'total_alignments': len(summaries),
        'methods_compared': list(methods.keys()),
        'method_summary': {}
    }

    # Calculate method-level summaries
    for method, method_stats in methods.items():
        total_samples = sum(s.get('total_samples', 0) for s in method_stats)
        avg_length = np.mean([s.get('alignment_length', 0) for s in method_stats])

        overall['method_summary'][method] = {
            'alignments': len(method_stats),
            'total_samples': total_samples,
            'avg_alignment_length': round(avg_length, 1)
        }

    return overall


def create_comparison_table(summaries: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a comparison table from alignment summaries.

    Args:
        summaries: Dictionary of alignment summaries

    Returns:
        DataFrame with comparison statistics
    """
    if not summaries:
        return pd.DataFrame()

    rows = []
    for key, stats in summaries.items():
        row = {
            'Method': stats.get('method', 'Unknown'),
            'Species': stats.get('species', 'Unknown'),
            'Segment': stats.get('segment', 'Unknown'),
            'Samples': stats.get('total_samples', 0),
            'Length': stats.get('alignment_length', 0),
            'Most_Divergent': stats.get('most_divergent_sample', 'N/A')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(['Method', 'Species', 'Segment']).reset_index(drop=True)
