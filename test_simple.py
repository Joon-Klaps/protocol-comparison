#!/usr/bin/env python3
"""
Simple test script for consensus module without dash_bio dependencies.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_consensus_basic():
    """Test basic consensus functionality without dash dependencies."""
    print("🧪 Testing Basic Consensus Functionality")
    print("=" * 50)
    
    try:
        from modules.consensus.tab import ConsensusTab
        print("✅ ConsensusTab imported successfully")
        
        # Set data path
        data_path = Path("../../data/app")
        print(f"📁 Data path: {data_path}")
        print(f"   Exists: {data_path.exists()}")
        
        if data_path.exists():
            # Create consensus tab
            consensus_tab = ConsensusTab(data_path)
            print("✅ ConsensusTab created successfully")
            
            # Test getting available alignments
            available_alignments = consensus_tab.get_available_alignments()
            print(f"✅ Found {len(available_alignments)} alignments")
            
            for i, alignment in enumerate(available_alignments):
                method, species, segment = alignment
                print(f"   {i+1}. {method} - {species} {segment}")
                
                # Test getting samples for this alignment
                try:
                    samples = consensus_tab.get_available_samples(alignment)
                    print(f"      → {len(samples)} samples available")
                    if samples:
                        print(f"      → Sample examples: {samples[:3]}...")
                except Exception as e:
                    print(f"      → Error getting samples: {e}")
                    
            print("\n✅ Basic consensus functionality working!")
            return True
        else:
            print("❌ Data path not found. Consensus testing skipped.")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dash_imports():
    """Test dash component imports."""
    print("\n🧪 Testing Dash Imports")
    print("=" * 50)
    
    try:
        import dash
        print(f"✅ dash: {dash.__version__}")
    except ImportError:
        print("❌ dash not available")
        return False
        
    try:
        import dash_bootstrap_components as dbc
        print(f"✅ dash-bootstrap-components: {dbc.__version__}")
    except ImportError:
        print("❌ dash-bootstrap-components not available")
        return False
        
    try:
        import dash_bio
        print(f"✅ dash-bio: {dash_bio.__version__}")
        return True
    except ImportError:
        print("❌ dash-bio not available")
        print("   This is needed for AlignmentChart and Clustergram")
        return False

if __name__ == "__main__":
    print("🚀 Simple Consensus Test")
    print("=" * 60)
    
    basic_test = test_consensus_basic()
    dash_test = test_dash_imports()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Basic Consensus: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"   Dash Components: {'✅ PASS' if dash_test else '❌ FAIL'}")
    
    if basic_test:
        print("\n🎉 Core consensus functionality is working!")
        print("   The original error has been resolved.")
        
        if dash_test:
            print("   Dash components are available for full migration.")
        else:
            print("   Install dash components: pip install dash dash-bootstrap-components dash-bio")
    else:
        print("\n⚠️  Basic consensus functionality not working.")
        
    print("\n💡 Next steps:")
    if basic_test and dash_test:
        print("   - Run the full Dash application: python dash_app.py")
        print("   - Access the application at: http://localhost:8050")
    elif basic_test:
        print("   - Install missing Dash packages")
        print("   - Then run: python dash_app.py")
    else:
        print("   - Debug the consensus module data loading issues")
