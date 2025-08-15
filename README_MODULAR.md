# Modular Streamlit Dashboard System

This directory implements a **modular page collection system** for creating complex Streamlit applications. The system allows you to organize analysis components into separate modules, with each module contributing its own pages to a unified dashboard.

## Architecture Overview

### Core Components

1. **`streamlit_base.py`** - Base classes for page components
2. **`main.py`** - Module page manager and registry system
3. **Module directories** - Individual analysis modules (e.g., `read_stats/`)
4. **`modular_streamlit_app.py`** - Example application using the modular system

### Module Structure

```
modules/
├── streamlit_base.py          # Base page component classes
├── main.py                    # Page manager and registry
├── __init__.py               # Module exports
├── read_stats/               # Example analysis module
│   ├── __init__.py
│   ├── streamlit_pages.py    # Page components for this module
│   ├── reads/               # Read processing analysis
│   └── mapping/             # Mapping analysis
├── consensus/               # Future module
└── coverage/                # Future module
```

## Key Features

### 🧩 Modular Architecture
- Each analysis type is a separate module
- Modules can be developed and tested independently
- Easy to add new analysis types

### 📄 Page Components
- Each module contributes `StreamlitPageComponent` instances
- Components handle their own UI rendering and data management
- Consistent interface across all modules

### 🔄 Automatic Discovery
- System automatically finds and registers available analysis pages
- No manual configuration required for new modules
- Registry system manages page organization

### 🎛️ Unified Interface
- All modules accessible through single dashboard
- Consistent navigation and user experience
- Shared data path configuration

## How to Use

### 1. Running the Modular Dashboard

```bash
# Run the example modular dashboard
streamlit run modular_streamlit_app.py

# Or use the run script with data path
python run_streamlit.py --data-path /path/to/data
```

### 2. Creating a New Analysis Module

#### Step 1: Create Module Directory
```bash
mkdir modules/my_analysis
touch modules/my_analysis/__init__.py
```

#### Step 2: Create Page Components (`modules/my_analysis/streamlit_pages.py`)
```python
from ..streamlit_base import StreamlitPageComponent, PageConfig

class MyAnalysisPage(StreamlitPageComponent):
    def __init__(self, data_manager=None):
        config = PageConfig(
            title="My Analysis",
            icon="📈",
            sidebar_title="My Analysis",
            description="Description of my analysis",
            requires_data=True,
            order=10
        )
        super().__init__(config, data_manager)

    def create_analyzer(self):
        # Create your analyzer instance
        return MyAnalyzer(self.data_manager.data_path)

    def render_sidebar(self, **kwargs):
        import streamlit as st
        # Render sidebar controls
        option = st.sidebar.selectbox("Options", ["A", "B"])
        return {'option': option}

    def render_content(self, **kwargs):
        import streamlit as st
        # Render main content
        st.write("My analysis content!")
        st.write(f"Selected option: {kwargs.get('option')}")

def get_my_analysis_pages(data_manager=None):
    """Return list of page components for this module."""
    return [MyAnalysisPage(data_manager)]
```

#### Step 3: Update Module Registry (in `main.py`)
The page will be automatically discovered if you follow the naming convention, or you can add it manually:

```python
# In main.py, add to _get_module_pages method
elif module_name == 'my_analysis':
    from .my_analysis.streamlit_pages import get_my_analysis_pages
    pages = get_my_analysis_pages(data_manager)
```

### 3. Page Component API

#### Base Class: `StreamlitPageComponent`

```python
class StreamlitPageComponent(ABC):
    def __init__(self, config: PageConfig, data_manager=None)

    @abstractmethod
    def create_analyzer(self):
        """Create analyzer instance for this page"""
        pass

    @abstractmethod
    def render_content(self, **kwargs):
        """Render main page content"""
        pass

    def render_sidebar(self, **kwargs):
        """Render sidebar controls (optional)"""
        return {}

    def validate_requirements(self, **kwargs):
        """Validate page requirements (optional)"""
        return True
```

#### Configuration: `PageConfig`

```python
@dataclass
class PageConfig:
    title: str                    # Page title
    icon: str = "📊"             # Icon for navigation
    sidebar_title: str = None     # Title in sidebar (defaults to title)
    description: str = None       # Page description
    requires_data: bool = True    # Whether page needs data
    order: int = 100             # Display order (lower = first)
```

## Example Usage

### Simple Module Integration
```python
from modules import get_global_page_manager

# Initialize page manager with data path
page_manager = get_global_page_manager("/path/to/data")

# Get available pages
available_pages = page_manager.get_available_pages()

# Get navigation menu items
nav_items = page_manager.get_navigation_menu_items()

# Render a specific page
page_manager.render_page("read_stats_0", data_path="/path/to/data")
```

### Complete Streamlit App
```python
import streamlit as st
from modules import get_global_page_manager

def main():
    st.title("My Dashboard")

    # Get page manager
    page_manager = get_global_page_manager(data_path)

    # Create navigation
    nav_items = page_manager.get_navigation_menu_items()

    # Let user select page
    selected_page = st.selectbox("Select Analysis", nav_items)

    # Render selected page
    page_manager.render_page(selected_page)

if __name__ == "__main__":
    main()
```

## Benefits

1. **Maintainability** - Each module is self-contained and independently testable
2. **Scalability** - Easy to add new analysis types without changing existing code
3. **Reusability** - Page components can be reused in different applications
4. **Consistency** - Shared base classes ensure consistent behavior
5. **Flexibility** - Modules can be loaded dynamically based on available data

## File Structure

```
protocol-comparison/
├── modular_streamlit_app.py    # Example modular dashboard
├── streamlit_app.py           # Original monolithic dashboard
├── run_streamlit.py           # Launcher script
├── modules/
│   ├── __init__.py
│   ├── streamlit_base.py      # Base classes
│   ├── main.py               # Page manager
│   ├── base.py               # Data manager base classes
│   ├── read_stats/           # Read statistics module
│   │   ├── __init__.py
│   │   ├── streamlit_pages.py
│   │   ├── reads/
│   │   └── mapping/
│   ├── consensus/            # Consensus analysis module
│   └── coverage/             # Coverage analysis module
└── README_MODULAR.md         # This file
```

This modular system allows you to build complex, maintainable Streamlit applications while keeping individual components focused and testable.
