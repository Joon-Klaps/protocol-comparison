#!/usr/bin/env python3
"""
Sample selection management for viral genomics protocol comparison.

This module handles:
- Loading and parsing preconfigured sample selections from TSV files
- Managing nested selection structure by dataset/run and condition
- Providing sample selection UI components for Streamlit
"""

import pandas as pd
import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PreconfiguredSelections:
    """Manages preconfigured sample selections from comparison TSV files."""

    def __init__(self, data_path: str):
        """
        Initialize with data path.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.comparison_excels_path = self.data_path / "comparison_excels"
        self.selections = {}

    def _process_parquet_file(self, parquet_file: Path):
        df = pd.read_parquet(parquet_file)

        if df.empty:
            return

        required_columns = ['Condition_group-to-check', 'LVE_codes-to-compare', 'Comment_additional-information']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Required columns not found in %s", parquet_file.name)
            return

        file_prefix = parquet_file.stem.replace('LVE_CAP_', '').replace('_V01', '')

        # Use pandas vectorized operations instead of iterating over rows
        df = df.rename(columns={
            'Condition_group-to-check': 'condition',
            'LVE_codes-to-compare': 'codes_raw',
            'Comment_additional-information': 'comment'
        })

        df['codes_list'] = df['codes_raw'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip()])
        df['source_file'] = parquet_file.name
        df['file_prefix'] = file_prefix

        # Filter out rows with empty conditions or codes
        filtered_df = df[df['condition'].notna() & (df['codes_list'].str.len() > 0)]

        if file_prefix not in self.selections:
            self.selections[file_prefix] = {}

        for index, row in filtered_df.iterrows():
            selection_key = row['condition']
            self.selections[file_prefix][selection_key] = {
                'condition': row['condition'],
                'lve_codes': row['codes_list'],
                'comment': row['comment'],
                'source_file': row['source_file'],
                'file_prefix': row['file_prefix']
            }

    def load_preconfigured_selections(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load preconfigured selections from TSV files.

        Returns:
            Nested dictionary: selections[file_prefix][selection_key] = selection_data
        """
        self.selections = {}

        if not self.comparison_excels_path.exists():
            logger.warning("Comparison excels path does not exist: %s", self.comparison_excels_path)
            return self.selections

        parquet_files = list(self.comparison_excels_path.glob("*.parquet"))
        if not parquet_files:
            logger.info("No Parquet files found in %s", self.comparison_excels_path)
            return self.selections

        for parquet_file in parquet_files:
            try:
                self._process_parquet_file(parquet_file)
            except Exception as e:
                logger.warning(f"Error processing file {parquet_file.name}: {e}")

        return self.selections

    def get_available_file_prefixes(self) -> List[str]:
        """Get list of available file prefixes, sorted numerically."""
        prefixes = list(self.selections.keys())

        # Sort prefixes numerically (RUN001, RUN002, etc.)
        def sort_key(prefix):
            if prefix.startswith('RUN'):
                try:
                    return int(prefix.replace('RUN', ''))
                except ValueError:
                    return float('inf')  # Put non-numeric at end
            return prefix

        return sorted(prefixes, key=sort_key)

    def get_selections_for_file(self, file_prefix: str) -> Dict[str, Dict[str, Any]]:
        """Get all selections for a specific file prefix."""
        return self.selections.get(file_prefix, {})

    def get_selection_by_file_and_key(self, file_prefix: str, selection_key: str) -> Optional[Dict[str, Any]]:
        """Get selection details by file prefix and selection key."""
        return self.selections.get(file_prefix, {}).get(selection_key)

    def get_all_selections_flat(self) -> Dict[str, Dict[str, Any]]:
        """Get all selections in a flat dictionary format for backward compatibility."""
        flat_selections = {}
        for file_prefix, file_selections in self.selections.items():
            for selection_key, selection_data in file_selections.items():
                combined_key = f"{file_prefix}: {selection_key}"
                flat_selections[combined_key] = selection_data
        return flat_selections

    def get_available_selections(self) -> List[str]:
        """Get list of available selection keys in flat format."""
        return list(self.get_all_selections_flat().keys())


class SampleSelectionManager:
    """Manages sample selection UI and logic for Streamlit app."""

    def __init__(self, data_path: str):
        """
        Initialize sample selection manager.

        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        self.preconfigured_selections = None

    def load_preconfigured_selections(self) -> Optional[PreconfiguredSelections]:
        """Load preconfigured selections if available."""
        try:
            self.preconfigured_selections = PreconfiguredSelections(self.data_path)
            self.preconfigured_selections.load_preconfigured_selections()
            return self.preconfigured_selections
        except Exception as e:
            logger.warning("Error loading preconfigured selections: %s", e)
            return None

    def get_selection_info_for_sidebar(self) -> tuple[int, int]:
        """
        Get selection info for sidebar display.

        Returns:
            Tuple of (total_selections, num_datasets)
        """
        if not self.preconfigured_selections:
            return 0, 0

        file_prefixes = self.preconfigured_selections.get_available_file_prefixes()
        total_selections = sum(len(self.preconfigured_selections.get_selections_for_file(fp))
                             for fp in file_prefixes)
        return total_selections, len(file_prefixes)

    def render_sample_selection(self, available_samples: List[str]) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """
        Render sample selection interface.

        Args:
            available_samples: List of available sample IDs

        Returns:
            Tuple of (selected sample IDs or None for all samples, selected preconfigured info)
        """
        if not available_samples:
            return None, None

        col1, col2 = st.columns([2, 1])

        with col1:
            selection_options = ["All samples", "Custom selection"]
            if self.preconfigured_selections and self.preconfigured_selections.get_available_file_prefixes():
                selection_options.append("Preconfigured selections")

            selection_type = st.radio(
                "Selection mode:",
                selection_options,
                horizontal=True
            )

        with col2:
            st.metric("Available Samples", len(available_samples))

        selected_preconfigured_info = None

        if selection_type == "Custom selection":
            # Individual sample selection
            selected_samples = st.multiselect(
                "Select samples:",
                options=available_samples,
                default=available_samples[:5] if len(available_samples) > 5 else available_samples,
                help="Choose specific samples for analysis"
            )
            return selected_samples if selected_samples else None, None

        elif selection_type == "Preconfigured selections" and self.preconfigured_selections:
            return self._render_preconfigured_selection(available_samples)

        return None, None  # Return None for all samples

    def _render_preconfigured_selection(self, available_samples: List[str]) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """Render preconfigured selection interface."""
        # File prefix selection
        available_file_prefixes = self.preconfigured_selections.get_available_file_prefixes()

        if not available_file_prefixes:
            st.info("No preconfigured selections available")
            return None, None

        col1, col2 = st.columns([1, 1])

        with col1:
            selected_file_prefix = st.radio(
                "Choose dataset/run:",
                options=available_file_prefixes,
                help="Select which dataset or run to use for comparison"
            )

        with col2:
            if selected_file_prefix:
                # Get selections for the chosen file
                file_selections = self.preconfigured_selections.get_selections_for_file(selected_file_prefix)

                if file_selections:
                    selected_condition = st.selectbox(
                        "Choose comparison condition:",
                        options=list(file_selections.keys()),
                        help="Select from predefined sample comparisons"
                    )

                    if selected_condition:
                        return self._process_selected_condition(
                            selected_file_prefix,
                            selected_condition,
                            available_samples
                        )
                else:
                    st.info(f"No comparisons available for {selected_file_prefix}")

        return None, None

    def _process_selected_condition(self, file_prefix: str, condition: str, available_samples: List[str]) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """Process a selected preconfigured condition."""
        config_info = self.preconfigured_selections.get_selection_by_file_and_key(file_prefix, condition)

        if not config_info:
            return None, None

        # Display configuration details
        with st.expander("📋 Selection Details", expanded=True):
            st.markdown(f"**Dataset:** {file_prefix}")
            st.markdown(f"**Condition:** {config_info['condition']}")
            if config_info['comment']:
                st.markdown(f"**Description:** {config_info['comment']}")
            st.markdown(f"**Source:** {config_info['source_file']}")
            st.markdown(f"**Samples:** {', '.join(config_info['lve_codes'])}")

            # Show which samples are available vs missing
            available_set = set(available_samples)
            requested_set = set(config_info['lve_codes'])

            available_requested = list(requested_set & available_set)
            missing_requested = list(requested_set - available_set)

            if available_requested:
                st.success(f"✅ Available samples ({len(available_requested)}): {', '.join(available_requested)}")

            if missing_requested:
                st.warning(f"⚠️ Missing samples ({len(missing_requested)}): {', '.join(missing_requested)}")

        # Return only the samples that are actually available
        return available_requested if available_requested else None, config_info

    def render_sidebar_info(self, selected_preconfigured_info: Optional[Dict[str, Any]]) -> None:
        """Render selection info in sidebar."""
        if selected_preconfigured_info:
            with st.sidebar:
                st.markdown("---")
                st.subheader("🔍 Current Selection")
                st.markdown(f"**{selected_preconfigured_info['file_prefix']}**")
                st.markdown(f"*{selected_preconfigured_info['condition']}*")
                if selected_preconfigured_info['comment']:
                    st.caption(selected_preconfigured_info['comment'])
