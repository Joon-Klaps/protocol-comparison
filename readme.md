# Protocol Comparison Dashboard

A modular Streamlit dashboard for comparing viral genomics analysis protocols.

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Usage

The `protocol-comparison` CLI provides several commands:

### Commands

#### 1. Run the Streamlit Application
```bash
protocol-comparison run --data data/app --port 8888
```

Options:
- `--data`: Path to data directory containing parquet files
- `--port`: Port to run the application on (default: 8501)
- `--host`: Host to bind the application to (default: localhost)
- `--browser`: Open browser automatically

#### 2. Convert Data to Parquet Format
```bash
protocol-comparison convert-data --input data/ --output data/app --log-level INFO
```

Options:
- `--input`: Input data directory containing nf-core/viralmetagenome output files
- `--output`: Output directory for converted files
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--force`: Force re-alignment of sequences even if output files already exist

#### 3. Convert Parquet to CSV
```bash
protocol-comparison convert-parquet input_dir/ output_dir/
```

Converts parquet files to CSV format for compatibility with other tools.

#### 4. Show Package Information
```bash
protocol-comparison info
```

#### 5. Show Version
```bash
protocol-comparison version
```

## Development

This package uses modern Python packaging with `pyproject.toml`:

- All dependencies are specified in `pyproject.toml`
- Install in editable mode with `pip install -e .`
- Optional development dependencies: `pip install -e .[dev]`

## Package Structure

```
protocol_comparison/
├── __init__.py
├── cli.py              # Main CLI entry point
├── app.py              # Streamlit application
├── convert_data.py     # Data conversion utilities
├── sample_selection.py # Sample selection utilities
└── modules/            # Analysis modules
    ├── consensus/
    ├── coverage/
    └── read_stats/
```
