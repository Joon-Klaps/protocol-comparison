# Clean Modular Dashboard Architecture

## Overview

This is a **clean, modular Streamlit dashboard** where:
- **X tabs** are created automatically - one per module
- **Clean separation** - no Streamlit code in modules
- **Pure data components** - modules return data structures, not UI
- **Custom HTML support** - for your mini HTML package integration

## Architecture

```
analysis/protocol-comparison/
├── clean_modular_app.py      # Main Streamlit app (ALL UI code here)
├── run_clean_app.py          # Launcher script
└── modules/                  # Framework-agnostic modules
    ├── read_stats/
    │   └── tab.py            # Returns pure data/figures
    ├── consensus/
    │   └── tab.py            # Includes custom HTML support
    └── coverage/
        └── tab.py            # Returns pure data/figures
```

## How It Works

### 1. Module Discovery
The main app automatically discovers modules by scanning the `modules/` directory for folders containing `tab.py` files.

### 2. Tab Generation
Each discovered module becomes a tab in the Streamlit interface. Tabs are ordered by the `order` field in module metadata.

### 3. Clean Component Interface
Each `tab.py` file contains a class with these methods:

```python
class ModuleTab:
    def get_summary_stats(self, selected_samples=None):
        """Returns: Dict with sections of metrics/tables"""

    def get_visualizations(self, selected_samples=None):
        """Returns: Dict with list of plotly figures"""

    def get_raw_data(self, selected_samples=None):
        """Returns: Dict with list of pandas DataFrames"""

    def get_custom_html(self, selected_samples=None):  # Optional
        """Returns: Dict with list of HTML components"""
```

### 4. Framework-Agnostic Design
- **No Streamlit imports** in module files
- **Pure data returns** - just dicts, DataFrames, plotly figures
- **Custom HTML support** - for your mini package integration
- **Easy testing** - modules can be tested independently

## Tab Interface Structure

### Summary Stats Format
```python
{
    "sections": [
        {
            "type": "metrics",
            "title": "Key Metrics",
            "data": {"metric1": "value1", "metric2": "value2"}
        },
        {
            "type": "species_breakdown",
            "title": "Species Analysis",
            "data": {"species1": {"stat1": "val1"}}
        }
    ]
}
```

### Visualizations Format
```python
{
    "figures": [
        {
            "title": "Plot Title",
            "description": "Plot description",
            "figure": plotly_figure_object
        }
    ]
}
```

### Raw Data Format
```python
{
    "tables": [
        {
            "title": "Table Name",
            "data": pandas_dataframe
        }
    ]
}
```

### Custom HTML Format
```python
{
    "components": [
        {
            "title": "Component Title",
            "description": "Component description",
            "html": "<div>Your HTML here</div>"
        }
    ]
}
```

## Running the Application

### Option 1: Using the launcher
```bash
python run_clean_app.py
```

### Option 2: Direct streamlit command
```bash
streamlit run clean_modular_app.py
```

## Module Development

### Creating a New Module

1. **Create module directory**:
   ```bash
   mkdir modules/my_new_module
   ```

2. **Create tab.py** with required interface:
   ```python
   # modules/my_new_module/tab.py

   def get_tab_info():
       return {
           "title": "My Analysis",
           "icon": "📊",
           "description": "My analysis description",
           "order": 4
       }

   def create_tab(data_path):
       return MyAnalysisTab(data_path)

   class MyAnalysisTab:
       def __init__(self, data_path):
           self.data_path = data_path

       def get_available_samples(self):
           # Return list of sample IDs

       def get_summary_stats(self, selected_samples=None):
           # Return summary data structure

       def get_visualizations(self, selected_samples=None):
           # Return visualization data structure

       def get_raw_data(self, selected_samples=None):
           # Return raw data structure
   ```

3. **The module appears automatically** - no need to modify main app!

## Key Benefits

### ✅ Clean Separation
- **UI code** only in main app
- **Data logic** only in modules
- **Easy maintenance** and testing

### ✅ Framework Agnostic
- Modules work with **any frontend**
- Easy to **switch frameworks**
- **Reusable components**

### ✅ Custom HTML Integration
- Built-in support for **HTML components**
- Perfect for your **mini HTML package**
- **Flexible rendering** options

### ✅ Dynamic Discovery
- **Automatic tab generation**
- **No code changes** for new modules
- **Ordered display** with metadata

### ✅ Robust Error Handling
- **Graceful degradation**
- **Module isolation** - one failure doesn't break others
- **Clear error messages**

## File Overview

### clean_modular_app.py
- **Main Streamlit application**
- **Module discovery and loading**
- **Tab rendering and UI**
- **Data path management**
- **Sample selection interface**

### modules/*/tab.py
- **Framework-agnostic tab components**
- **Pure data returns**
- **Consistent interface**
- **Custom HTML support**

### run_clean_app.py
- **Simple launcher script**
- **Error handling for missing dependencies**

This architecture gives you exactly what you requested: **X tabs (one per module)** with **clean separation** where modules provide **pure data components** without any Streamlit dependencies, perfect for integrating with your custom HTML components!
