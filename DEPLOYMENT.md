# Streamlit Cloud Deployment Configuration

This file contains configuration for deploying the Viral Genomics Protocol Comparison Dashboard to Streamlit Cloud.

## Quick Deploy to Streamlit Cloud

1. **Push your code to GitHub**
2. **Visit [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Set the following configuration:**

### Deployment Settings

- **Repository**: `your-username/your-repo-name`
- **Branch**: `main`
- **Main file path**: `analysis/protocol-comparison/streamlit_app.py`
- **Python version**: `3.9` (or your preferred version)

### Environment Variables (Optional)

If you want to set a default data path:

```
DEFAULT_DATA_PATH=sample_data
```

### Required Files for Deployment

Make sure these files are in your repository:

- `streamlit_app.py` (main application)
- `requirements.txt` (dependencies)
- `modules/` (analysis modules)
- `.streamlit/config.toml` (Streamlit configuration)

### Repository Structure for Deployment

```
your-repo/
├── analysis/
│   └── protocol-comparison/
│       ├── streamlit_app.py          # Main app file
│       ├── requirements.txt          # Dependencies
│       ├── modules/                  # Analysis modules
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── consensus.py
│       │   ├── coverage.py
│       │   └── read_stats.py
│       ├── .streamlit/
│       │   └── config.toml           # Streamlit config
│       ├── generate_sample_data.py   # Sample data generator
│       └── README.md                 # Documentation
```

## Alternative Deployment Options

### 1. Heroku

Create a `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### 2. Railway

Railway will automatically detect the Streamlit app and deploy it.

### 3. Render

Create a `render.yaml`:
```yaml
services:
  - type: web
    name: viral-genomics-dashboard
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

### 4. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data (optional)
python generate_sample_data.py --num-samples 10

# Run the app
streamlit run streamlit_app.py
```

## Environment Variables

- `DEFAULT_DATA_PATH`: Default path to data directory
- `STREAMLIT_SERVER_PORT`: Port for the server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Address to bind to (default: localhost)

## Notes

- Streamlit Cloud provides **free hosting** for public repositories
- Apps automatically update when you push to your repository
- Built-in authentication and sharing features
- No server management required
