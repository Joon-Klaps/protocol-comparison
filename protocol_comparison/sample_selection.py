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

        # Normalize file prefix: remove leading "LVE_CAP_" and strip any trailing version like "_V01", "_V2a", etc.
        file_prefix = parquet_file.stem
        file_prefix = re.sub(r'^LVE_CAP_', '', file_prefix, flags=re.IGNORECASE)
        # remove trailing "_V..." (case-insensitive)
        file_prefix = re.sub(r'_[Vv].*$', '', file_prefix)

        # Use pandas vectorized operations instead of iterating over rows
        df = df.rename(columns={
            'Condition_group-to-check': 'condition',
            'LVE_codes-to-compare': 'codes_raw',
            'Comment_additional-information': 'comment',
            # Optional column for alternative naming
            'Plot_names-for-legends_with-volume': 'aliases_raw',
            'Plot_names-for-legends_short': 'aliases_short'
        })

        df['codes_list'] = df['codes_raw'].apply(
            lambda x: [c for c in str(x).replace(',', ' ').split() if c != '']
        )
        # Parse aliases list if available; keep None when absent
        if 'aliases_raw' in df.columns:
            df['aliases_list'] = df['aliases_raw'].apply(
                lambda x: [c.strip() for c in str(x).split(',') if c.strip() != ''] if pd.notna(x) else []
            )
        else:
            df['aliases_list'] = [[] for _ in range(len(df))]

        # Parse short aliases list if available
        if 'aliases_short' in df.columns:
            df['aliases_short_list'] = df['aliases_short'].apply(
                lambda x: [c.strip() for c in str(x).split(',') if c.strip() != ''] if pd.notna(x) else []
            )
        else:
            df['aliases_short_list'] = [[] for _ in range(len(df))]

        df['source_file'] = parquet_file.name
        df['file_prefix'] = file_prefix

        # Filter out rows with empty conditions or codes
        filtered_df = df[df['condition'].notna() & (df['codes_list'].str.len() > 0)]

        if file_prefix not in self.selections:
            self.selections[file_prefix] = {}

        for _, row in filtered_df.iterrows():
            selection_key = row['condition']
            # Build alias mappings if lengths are compatible (>0 and same length)
            aliases_list = row.get('aliases_list', []) if isinstance(row.get('aliases_list', []), list) else []
            aliases_short_list = row.get('aliases_short_list', []) if isinstance(row.get('aliases_short_list', []), list) else []

            alias_by_code = {}
            alias_short_by_code = {}

            # Process full aliases (with volume)
            if aliases_list:
                if len(aliases_list) != len(row['codes_list']):
                    logger.warning(
                        "Alias list length (%d) does not match codes list length (%d) in %s - condition '%s'",
                        len(aliases_list), len(row['codes_list']), parquet_file.name, selection_key
                    )
                # Zip will truncate to min length
                alias_by_code = {code: alias for code, alias in zip(row['codes_list'], aliases_list)}

            # Process short aliases
            if aliases_short_list:
                if len(aliases_short_list) != len(row['codes_list']):
                    logger.warning(
                        "Short alias list length (%d) does not match codes list length (%d) in %s - condition '%s'",
                        len(aliases_short_list), len(row['codes_list']), parquet_file.name, selection_key
                    )
                # Zip will truncate to min length
                alias_short_by_code = {code: alias for code, alias in zip(row['codes_list'], aliases_short_list)}

            self.selections[file_prefix][selection_key] = {
                'condition': row['condition'],
                'lve_codes': row['codes_list'],
                'comment': row['comment'],
                'source_file': row['source_file'],
                'file_prefix': row['file_prefix'],
                # Optional alias data
                'aliases_list': aliases_list,
                'alias_by_code': alias_by_code,
                'aliases_short_list': aliases_short_list,
                'alias_short_by_code': alias_short_by_code
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
                # Clear any alias mapping for custom selection
                st.session_state.pop("sample_alias_map", None)
                st.session_state.pop("sample_alias_order", None)
                st.session_state.pop("alias_volume_map", None)
                st.session_state.pop("alias_short_map", None)
                st.session_state.pop("display_mode", None)
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
                # Clear any alias mapping for custom selection
                st.session_state.pop("sample_alias_map", None)
                st.session_state.pop("sample_alias_order", None)
                st.session_state.pop("alias_volume_map", None)
                st.session_state.pop("alias_short_map", None)
                st.session_state.pop("display_mode", None)
            return selected_samples if selected_samples else None, None

        if selection_type == "Preconfigured selections" and self.preconfigured_selections:
            return self._render_preconfigured_selection(available_samples)

        # All samples selected
        st.session_state["sample_order"] = list(available_samples)
        # Clear any alias mapping for all-samples mode
        st.session_state.pop("sample_alias_map", None)
        st.session_state.pop("sample_alias_order", None)
        st.session_state.pop("alias_volume_map", None)
        st.session_state.pop("alias_short_map", None)
        st.session_state.pop("display_mode", None)
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

        # Set up display mode and alias mappings
        alias_by_code = config_info.get('alias_by_code', {})
        alias_short_by_code = config_info.get('alias_short_by_code', {})

        # Store both types of alias mappings in session state
        st.session_state["alias_volume_map"] = alias_by_code
        st.session_state["alias_short_map"] = alias_short_by_code

        # Determine which aliases are available
        has_volume_aliases = bool(alias_by_code)
        has_short_aliases = bool(alias_short_by_code)

        # Set default display mode based on available aliases
        if "display_mode" not in st.session_state:
            if has_short_aliases:
                st.session_state["display_mode"] = "alias_short"
            elif has_volume_aliases:
                st.session_state["display_mode"] = "alias_volume"
            else:
                st.session_state["display_mode"] = "lve_ids"

        # Set up current alias mapping based on display mode
        current_display_mode = st.session_state.get("display_mode", "lve_ids")

        if current_display_mode == "alias_volume" and has_volume_aliases:
            alias_map = {code: alias_by_code.get(code, code) for code in available_requested}
        elif current_display_mode == "alias_short" and has_short_aliases:
            alias_map = {code: alias_short_by_code.get(code, code) for code in available_requested}
        else:
            alias_map = {}

        if alias_map:
            st.session_state["sample_alias_map"] = alias_map
            st.session_state["sample_alias_order"] = [alias_map.get(code, code) for code in available_requested]
        else:
            # Clear any previous alias mapping
            st.session_state.pop("sample_alias_map", None)
            st.session_state.pop("sample_alias_order", None)

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

                # If alias mapping is active, show display mode selector
                alias_by_code = selected_preconfigured_info.get('alias_by_code', {})
                alias_short_by_code = selected_preconfigured_info.get('alias_short_by_code', {})

                has_volume_aliases = bool(alias_by_code)
                has_short_aliases = bool(alias_short_by_code)

                # Create display options based on available aliases
                display_options = ["LVE IDs"]
                if has_volume_aliases:
                    display_options.append("Alias w/ Volume")
                if has_short_aliases:
                    display_options.append("Alias Short")

                # Map display names to internal modes
                display_mode_map = {
                    "LVE IDs": "lve_ids",
                    "Alias w/ Volume": "alias_volume",
                    "Alias Short": "alias_short"
                }

                current_mode = st.session_state.get("display_mode", "lve_ids")
                # Convert internal mode back to display name for the radio button
                reverse_map = {v: k for k, v in display_mode_map.items()}
                current_display_name = reverse_map.get(current_mode, "LVE IDs")

                selected_display = st.radio(
                    "Display mode:",
                    options=display_options,
                    index=display_options.index(current_display_name) if current_display_name in display_options else 0,
                    help="Choose how to display sample names",
                    horizontal=True,
                    key="display_mode_radio"
                )

                # Update session state with the selected mode
                st.session_state["display_mode"] = display_mode_map[selected_display]

                # Update alias mappings based on new display mode
                update_current_alias_mappings()

                # Show current mode indicator
                if selected_display == "LVE IDs":
                    st.markdown("✅ Displaying raw LVE sample IDs")
                elif selected_display == "Alias w/ Volume":
                    st.markdown("✅ Using aliases with volume information")
                elif selected_display == "Alias Short":
                    st.markdown("✅ Using short alias names")

                # Display mapping in the current selection order
                ordered_samples = st.session_state.get("sample_order", [])
                current_alias_map = st.session_state.get("sample_alias_map", {})

                if ordered_samples:
                    with st.expander("View current mapping", expanded=False):
                        # Build a small two-column table-like text
                        for sid in ordered_samples:
                            if selected_display == "LVE IDs":
                                display_name = sid
                            else:
                                display_name = current_alias_map.get(sid, sid)
                            st.markdown(f"`{sid}` → **{display_name}**")

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


# ---------------------------
# Alias/label utilities
# ---------------------------
def update_current_alias_mappings():
    """Update the current alias mappings based on the selected display mode."""
    display_mode = st.session_state.get("display_mode", "lve_ids")
    sample_order = get_current_sample_order()

    if not sample_order:
        return

    if display_mode == "lve_ids":
        st.session_state.pop("sample_alias_map", None)
        st.session_state.pop("sample_alias_order", None)
        return

    # Get the appropriate base alias map
    if display_mode == "alias_volume":
        base_alias_map = st.session_state.get("alias_volume_map", {})
    elif display_mode == "alias_short":
        base_alias_map = st.session_state.get("alias_short_map", {})
    else:
        base_alias_map = {}

    if base_alias_map:
        # Create current alias mapping for available samples
        current_alias_map = {code: base_alias_map.get(code, code) for code in sample_order}
        st.session_state["sample_alias_map"] = current_alias_map
        st.session_state["sample_alias_order"] = [current_alias_map.get(code, code) for code in sample_order]
    else:
        st.session_state.pop("sample_alias_map", None)
        st.session_state.pop("sample_alias_order", None)


def get_current_sample_alias_map() -> Dict[str, str]:
    """Return current mapping from sample_id -> display label based on display mode."""
    display_mode = st.session_state.get("display_mode", "lve_ids")

    if display_mode == "lve_ids":
        return {}

    # Get the appropriate alias map based on display mode
    if display_mode == "alias_volume":
        alias_map = st.session_state.get("alias_volume_map", {})
    elif display_mode == "alias_short":
        alias_map = st.session_state.get("alias_short_map", {})
    else:
        alias_map = {}

    return dict(alias_map) if isinstance(alias_map, dict) else {}


def label_for_sample(sample_id: str) -> str:
    """Return display label for a sample based on current display mode."""
    display_mode = st.session_state.get("display_mode", "lve_ids")

    if display_mode == "lve_ids":
        return sample_id

    alias_map = get_current_sample_alias_map()
    if alias_map:
        return alias_map.get(sample_id, sample_id)
    return sample_id


def get_current_label_order() -> List[str]:
    """Return desired label order for display based on current display mode."""
    display_mode = st.session_state.get("display_mode", "lve_ids")

    if display_mode == "lve_ids":
        return get_current_sample_order()

    # Get the appropriate alias order based on display mode
    sample_order = get_current_sample_order()
    alias_map = get_current_sample_alias_map()

    if alias_map and sample_order:
        return [alias_map.get(sample, sample) for sample in sample_order]

    # Fall back to raw sample order
    return get_current_sample_order()


def map_series_to_labels(series: pd.Series) -> pd.Series:
    """Map a pandas Series of sample IDs to display labels based on current display mode."""
    display_mode = st.session_state.get("display_mode", "lve_ids")

    if display_mode == "lve_ids":
        return series

    alias_map = get_current_sample_alias_map()
    if not alias_map:
        return series
    return series.map(lambda s: alias_map.get(s, s))
