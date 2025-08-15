# Viral Genomics Protocol Comparison Dashboard

A modular analysis platform for comparing different sequencing protocols in viral genomics studies. Available in both **Streamlit** (recommended for free hosting) and **Dash** versions.

## 🚀 Quick Start

### Streamlit Version (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py --num-samples 10

# Run Streamlit app
streamlit run streamlit_app.py
# or
make run
```

### Dash Version

```bash
# Run Dash app
python run_dashboard.py --data-path sample_data
# or
make run-dash
```

## 🌐 Free Hosting Options

### Streamlit Cloud (Recommended)
- **Free hosting** for public GitHub repositories
- **Automatic deployment** from GitHub
- **Built-in sharing** and authentication
- **No server management** required

**Deploy in 3 steps:**
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Other Options
- **Railway**: Automatic deployment detection
- **Heroku**: Free tier available
- **Render**: Simple deployment

## Features

### Current Implementation

- **Consensus Analysis Module**
  - Genome recovery statistics
  - Average Nucleotide Identity (ANI) calculations and matrix visualization
  - Recovery percentage analysis and distribution plots

- **Coverage Analysis Module**
  - Coverage depth analysis with customizable thresholds
  - Overlay coverage plots for multiple samples
  - Segment-specific coverage visualization
  - Depth distribution analysis

- **Read Statistics Module**
  - Mapping efficiency analysis
  - UMI (Unique Molecular Identifier) statistics
  - Contamination detection for LASV and HAZV
  - Per-segment read distribution

### Planned Features (from TODO list)

- [ ] Variant population comparison
- [ ] Enhanced contamination checks
- [ ] Additional virus type support
- [ ] Database integration for cached results
- [ ] Export functionality for analysis results

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your data directory** with the expected structure:
   ```
   data/
   ├── consensus/
   │   ├── ani_comparison.tsv
   │   ├── genome_recovery.tsv
   ├── coverage/
   │   ├── coverage_summary.tsv
   ├── depth/
   │   ├── sample1.depth
   │   ├── sample2.depth
   ├── read_stats/
   │   ├── read_counts.tsv
   │   ├── umi_stats.tsv
   ├── mapping/
   │   ├── mapping_stats.tsv
   ├── contamination/
   │   ├── lasv_contamination.tsv
   │   ├── hazv_contamination.tsv
   └── references/
       ├── reference_mapping.tsv
   ```

## Usage

### Quick Start

```bash
# Run with default settings
python run_dashboard.py

# Run with specific data path
python run_dashboard.py --data-path /path/to/your/data

# Run on specific host and port
python run_dashboard.py --host 0.0.0.0 --port 8080 --debug
```

### Using the Web Interface

1. **Start the dashboard:**
   ```bash
   streamlit run streamlit_app.py
   # or
   python run_streamlit.py --data-path sample_data
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Configure data path** in the sidebar if not specified via command line

4. **Select analysis tab** and choose samples for analysis

5. **Generate visualizations** and explore the results

### Command Line Options

For the Streamlit launcher (`run_streamlit.py`):
- `--data-path`: Path to your data directory
- `--host`: Host to bind the server (default: localhost)
- `--port`: Port to bind the server (default: 8501)
- `--headless`: Run without opening browser
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Architecture

The platform follows a modular object-oriented design:

### Base Classes (`modules/base.py`)

- `DataManager`: Abstract base class for data loading and caching
- `BaseAnalyzer`: Abstract base class for analysis modules

### Analysis Modules

- `ConsensusAnalyzer` (`modules/consensus.py`): Handles consensus sequence analysis
- `CoverageAnalyzer` (`modules/coverage.py`): Handles coverage and depth analysis
- `ReadStatsAnalyzer` (`modules/read_stats.py`): Handles read mapping and contamination analysis

### Main Application (`streamlit_app.py`)

- Streamlit application with sidebar navigation and tabbed interface
- Interactive parameter configuration and sample selection
- Real-time visualization generation with caching
- Responsive design optimized for web deployment

### Configuration (`.streamlit/config.toml`)

- Streamlit-specific configuration and theming
- Performance and caching settings
- Server configuration for deployment## Data Format Requirements

### Expected File Formats

All input files should be tab-separated values (TSV) with headers:

**Mapping Statistics (`mapping/mapping_stats.tsv`):**
```
sample_id	total_reads	mapped_reads	target_mapped_reads
sample1	1000000	800000	700000
```

**Genome Recovery (`consensus/genome_recovery.tsv`):**
```
sample_id	total_bases	covered_bases	segment
sample1	10000	9500	L
```

**Coverage Depth (`depth/*.depth`):**
```
contig	position	depth
segment_L	1	150
segment_L	2	148
```

**UMI Statistics (`read_stats/umi_stats.tsv`):**
```
sample_id	total_umis	target_umis
sample1	50000	45000
```

**Contamination Data (`contamination/lasv_contamination.tsv`):**
```
sample_id	total_reads	contaminant_reads
sample1	1000000	5000
```

## Development

### Adding New Analysis Modules

1. **Create a new analyzer class** inheriting from `BaseAnalyzer`
2. **Implement required methods:**
   - `generate_summary_stats()`
   - `create_visualizations()`
3. **Create corresponding data manager** inheriting from `DataManager`
4. **Add to the main application** in `streamlit_app.py`

### Extending Existing Modules

Each analyzer supports:
- Custom sample selection
- Configurable parameters
- Export functionality
- Caching for performance

## Contributing

When adding new features:

1. Follow the object-oriented modular design
2. Update the TODO list in the README
3. Add appropriate error handling and logging
4. Include data format documentation
5. Test with sample data

## Troubleshooting

### Common Issues

1. **"No data available" errors:**
   - Check that your data directory structure matches the expected format
   - Verify file permissions and paths
   - Check the application logs for specific file loading errors

2. **Import errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (requires Python 3.7+)

3. **Port already in use:**
   - Use a different port: `python run_streamlit.py --port 8502`
   - Check for other running instances

### Logging

The application creates detailed logs. Check the terminal output for troubleshooting information, or use `--log-level DEBUG` for more detailed logging.

## Future Development

The platform is designed for gradual expansion. Priority features include:

1. **Database Integration:** Replace file-based data loading with database connections
2. **Variant Analysis:** Implement population-level variant comparison
3. **Enhanced Visualizations:** Add more interactive plot types
4. **Batch Processing:** Support for automated analysis pipelines
5. **API Endpoints:** RESTful API for programmatic access

## Original Analysis Requirements

Modules to make:

 -[] Consensus comparison - how much recovered in each
 > Genome recovery ~ ANI to the mapping reference
 -[] Consensus comparison - nucleotide level
 > ANI to the other given sequences (DISTANCE PLOT/MATRIX)
 -[] Consensus recovery comparison - overlay coverage plots
 > Coverage plot for each segment
 -[] Variant population comparison
 > Taking a relative frequency of custom mpilup at every position, find the sd across the samples & sum
 -[] Total number of reads recovered for target versus total reads (per segment)
 > How many reads for every segment %wise
 >
 -[] UMI number of reads recovered for target versus total reads (per segment)
 > How many UMIs for every segment %wise
 -[] Genome recovery for target (LASV - for each segment)
 > % of genome that has enough depth
 -[] Genome recovery for target (HAZV - for each segment)
 > % of genome that has enough depth
 -[] Coverage plot  (LASV - for each segments)
 > Coverage plot for each LASV segment
 -[] `Coveragel` plot (HAZV - for each segment)
 > Coverage plot for each HAZV segment
 -[] Contamination check - HAZV presence
 > Check the number of relative reads mapping towards all three HAZV segments
 -[] Contamination check - LASV presence
 > Check the number of relative reads mapping towards all three LASV segments
 -[] Contamination check comment
 > Can be ignored


