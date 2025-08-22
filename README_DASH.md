# 🧬 Viral Genomics Protocol Comparison Dashboard

**Pure Dash Application for Bioinformatics Analysis**

A modern, interactive web application built with Dash for comparing viral genomics protocols, featuring specialized bioinformatics visualizations including sequence alignments, coverage analysis, and read statistics.

## 🎯 Features

- **🧬 Interactive Sequence Alignments**: Full `dash_bio` AlignmentChart integration
- **📊 Coverage Analysis**: Depth and breadth coverage visualizations
- **📈 Read Statistics**: Quality metrics and processing statistics
- **🔗 Identity Matrices**: Hierarchical clustering with interactive dendrograms
- **📱 Responsive Design**: Bootstrap-powered UI components
- **⚡ High Performance**: Optimized for large genomics datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Conda or pip package manager

### Installation & Setup

```bash
# Clone or navigate to the project directory
cd protocol-comparison

# Install dependencies
make install

# Setup and validate data paths
make setup

# Run the application
make run
```

The dashboard will be available at: **http://localhost:8050**

## 📋 Available Commands

```bash
make help          # Show all available commands
make install       # Install Python dependencies
make setup         # Full setup with data validation
make run           # Run the Dash application
make dev           # Run in development mode with debug
make test          # Test components and modules
make clean         # Clean up cache files
make info          # Show system information
```

## 📁 Project Structure

```
protocol-comparison/
├── dash_app.py                    # Main Dash application
├── requirements.txt               # Python dependencies
├── Makefile                       # Build and run commands
├── modules/                       # Analysis modules
│   ├── consensus/                 # Consensus analysis
│   │   ├── dash_adapter.py        # Dash integration
│   │   ├── data.py                # Data management
│   │   ├── visualizations.py     # Plot generation
│   │   └── tab.py                 # Core logic
│   ├── coverage/                  # Coverage analysis
│   └── read_stats/                # Read statistics
├── test_simple.py                 # Component tests
└── streamlit_backup/              # Legacy Streamlit files
```

## 🔧 Configuration

### Data Path Configuration
The application expects data in the following structure:
```
../../data/app/
├── alignments/
│   ├── mapping/
│   │   ├── HAZV/
│   │   │   ├── S/, M/, L/
│   │   └── LASV/
│   │       ├── S/, L/
│   └── denovo/
│       ├── HAZV/
│       └── LASV/
├── contigs.parquet
├── mapping.parquet
└── reads.parquet
```

### Environment Variables
- `DEFAULT_DATA_PATH`: Override default data path location

## 🧪 Testing

Run comprehensive tests:
```bash
make test
```

Test specific components:
```bash
python test_simple.py
```

## 🛠️ Development

### Development Mode
```bash
make dev
```
This enables:
- Auto-reload on file changes
- Debug mode with detailed error messages
- Enhanced logging

### Adding New Modules
1. Create module directory in `modules/`
2. Implement core analysis in `tab.py`
3. Create Dash adapter in `dash_adapter.py`
4. Register in main `dash_app.py`

## 📊 Modules

### 🧬 Consensus Analysis
- Interactive sequence alignments using `dash_bio.AlignmentChart`
- Pairwise identity matrices with hierarchical clustering
- Support for multiple alignment methods (mapping vs. denovo)
- FASTA sequence parsing and validation

### 📊 Coverage Analysis
- Depth and breadth coverage metrics
- Interactive coverage plots
- Sample comparison visualizations

### 📈 Read Statistics
- Quality score distributions
- Read length analysis
- Processing pipeline metrics

## 🎨 UI Components

- **Bootstrap Theme**: Professional, responsive design
- **Interactive Controls**: Dropdowns, multi-select, buttons
- **Loading Indicators**: Smooth user experience
- **Error Handling**: Graceful error messages and recovery
- **Tabbed Interface**: Organized module presentation

## 🔍 Troubleshooting

### Common Issues

**"No module named 'dash_bio'"**
```bash
pip install dash-bio
```

**"Data path not found"**
- Update the data path in `dash_app.py`
- Ensure data structure matches expected format

**"Import errors"**
```bash
make install  # Reinstall dependencies
```

### Debug Mode
Run with debug information:
```bash
make dev
```

## 🔄 Migration from Streamlit

This project was migrated from Streamlit to provide better support for bioinformatics visualizations. The original Streamlit files are preserved in `streamlit_backup/` for reference.

### Why Dash?
- Native `dash_bio` component support
- Better performance for large datasets
- More flexible callback system
- Professional UI components

## 📈 Performance

- Optimized for datasets with 100+ samples
- Lazy loading of alignment data
- Efficient caching mechanisms
- Responsive UI even with large files

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is part of viral genomics research. Please cite appropriately if used in academic work.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run `make test` to validate setup
3. Check logs for detailed error messages
4. Open an issue with detailed description

---

**Built with ❤️ for the viral genomics community**
