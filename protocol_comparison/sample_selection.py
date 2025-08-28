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
import re
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

        df['codes_list'] = df['codes_raw'].apply(
            lambda x: [c for c in str(x).replace(',', ' ').split() if c != '']
        )
        df['source_file'] = parquet_file.name
        df['file_prefix'] = file_prefix

        # Filter out rows with empty conditions or codes
        filtered_df = df[df['condition'].notna() & (df['codes_list'].str.len() > 0)]

        if file_prefix not in self.selections:
            self.selections[file_prefix] = {}

        for _, row in filtered_df.iterrows():
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
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                logger.warning("Error processing file %s: %s", parquet_file.name, e)

        return self.selections

    def get_available_file_prefixes(self) -> List[str]:
        """Get list of available file prefixes, sorted numerically."""
        prefixes = list(self.selections.keys())

        # Sort prefixes numerically (RUN001, RUN002, etc.)
        def sort_key(prefix):
            if prefix.startswith('RUN'):
                try:
                    return (0, int(prefix.replace('RUN', '')))
                except ValueError:
                    return (1, prefix)
            return (1, prefix)

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
        except (OSError, ValueError, RuntimeError) as e:
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
        """Render sample selection UI and return the chosen samples and optional config info.

        Returns (selected_samples | None for all, selected_preconfigured_info | None).
        """
        if not available_samples:
            st.session_state["sample_order"] = []
            return None, None

        col1, col2 = st.columns([2, 1])

        with col1:
            # Put "Custom selection" first so the default app startup only selects a small subset (first 5)
            selection_options = ["Custom selection", "All samples"]
            if self.preconfigured_selections and self.preconfigured_selections.get_available_file_prefixes():
                selection_options.append("Preconfigured selections")

            selection_type = st.radio(
                "Selection mode:",
                selection_options,
                horizontal=True
            )

        with col2:
            st.metric("Available Samples", len(available_samples))

        if selection_type == "Custom selection":
            pasted = st.text_area(
                "Paste LVE sample IDs (comma/space separated)",
                placeholder="LVE00101, LVE00102 LVE00103, LVE00104 LVE00132",
                height=100,
                help="You can separate IDs using commas, spaces, or both. Duplicates will be removed while preserving order."
            )

            if pasted and pasted.strip():
                selected_samples, missing_requested = self._parse_pasted_ids(pasted, available_samples)

                with st.expander("Parsed selection summary", expanded=True):
                    st.success(f"Found {len(selected_samples)} valid sample(s)")
                    if missing_requested:
                        st.warning(f"Ignoring {len(missing_requested)} missing ID(s): {', '.join(missing_requested)}")

                st.session_state["sample_order"] = selected_samples
                return selected_samples if selected_samples else None, None

            # Fallback to multiselect when no paste provided
            selected_samples = st.multiselect(
                "Select samples:",
                options=available_samples,
                default=available_samples[:5] if len(available_samples) > 5 else available_samples,
                help="Choose specific samples for analysis or paste IDs above"
            )
            if selected_samples:
                st.session_state["sample_order"] = selected_samples
            return selected_samples if selected_samples else None, None

        if selection_type == "Preconfigured selections" and self.preconfigured_selections:
            return self._render_preconfigured_selection(available_samples)

        # All samples selected
        st.session_state["sample_order"] = list(available_samples)
        return None, None

    def _render_preconfigured_selection(self, available_samples: List[str]) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """Render preconfigured selection interface."""
        if not self.preconfigured_selections:
            st.info("No preconfigured selections available")
            return None, None
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
        if not self.preconfigured_selections:
            return None, None
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

            # Show which samples are available vs missing, preserving requested order
            requested_order = config_info['lve_codes']
            available_set = set(available_samples)
            available_requested = [s for s in requested_order if s in available_set]
            missing_requested = [s for s in requested_order if s not in available_set]

            if available_requested:
                st.success(f"✅ Available samples ({len(available_requested)}): {', '.join(available_requested)}")

            if missing_requested:
                st.warning(f"⚠️ Missing samples ({len(missing_requested)}): {', '.join(missing_requested)}")

        # Persist desired plotting order and return only the samples that are actually available
        st.session_state["sample_order"] = available_requested
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

    # ---------------------------
    # Helper methods
    # ---------------------------
    def _parse_pasted_ids(self, text: str, available_samples: List[str]) -> tuple[List[str], List[str]]:
        """Parse a large pasted string of sample IDs using commas and/or spaces as separators.

        - Splits on one or more commas/whitespace.
        - Normalizes case for matching, but returns canonical IDs as found in available_samples.
        - De-duplicates while preserving the original order from the pasted string.

        Returns:
            (selected_in_order, missing_in_order)
        """
        # Build a case-insensitive lookup to preserve canonical IDs as present in available_samples
        avail_map = {s.upper(): s for s in available_samples}

        # Split by commas and/or whitespace, collapse empties
        tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]

        seen = set()
        selected: List[str] = []
        missing: List[str] = []

        for tok in tokens:
            key = tok.upper()
            if key in seen:
                continue
            seen.add(key)
            if key in avail_map:
                selected.append(avail_map[key])
            else:
                # Keep original token in missing for user feedback
                missing.append(tok)

        return selected, missing


# ---------------------------
# Plot-order utilities
# ---------------------------
def get_current_sample_order(default: Optional[List[str]] = None) -> List[str]:
    """Return the current desired sample order from session state."""
    if default is None:
        default = []
    order = st.session_state.get("sample_order", default)
    return list(order) if isinstance(order, list) else default


def apply_sample_order(df: pd.DataFrame, sample_col: str = "sample_id") -> pd.DataFrame:
    """Apply the current sample order to a DataFrame for consistent plotting.

    - Sets 'sample_col' to a pandas Categorical with the saved order
    - Returns a sorted copy of the DataFrame
    """
    order = get_current_sample_order()
    if not order or sample_col not in df.columns:
        return df
    ordered_df = df.copy()
    ordered_df[sample_col] = pd.Categorical(ordered_df[sample_col], categories=order, ordered=True)
    return ordered_df.sort_values(sample_col)
