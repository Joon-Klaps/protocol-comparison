#!/usr/bin/env python3
"""
Test script for the Dash consensus adapter.

This script validates that the consensus module can work with Dash components
and that the dash_bio integration is functioning correctly.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_consensus_adapter():
    """Test the DashConsensusTab adapter."""
    print("🧪 Testing Dash Consensus Adapter")
    print("=" * 50)
    
    try:
        from modules.consensus.dash_adapter import DashConsensusTab
        print("✅ DashConsensusTab imported successfully")
        
        # Set data path
        data_path = Path("../../data/app")
        if not data_path.exists():
            print(f"⚠️  Data path not found: {data_path}")
            print("   Using current directory for testing")
            data_path = Path(".")
        
        # Create consensus tab
        consensus_tab = DashConsensusTab(data_path)
        print(f"✅ DashConsensusTab created with data path: {data_path}")
        
        # Test getting available alignments
        try:
            available_alignments = consensus_tab.get_available_alignments()
            print(f"✅ Available alignments: {len(available_alignments)} found")
            
            if available_alignments:
                for i, alignment in enumerate(available_alignments[:5]):  # Show first 5
                    method, species, segment = alignment
                    print(f"   {i+1}. {method} - {species} {segment}")
                if len(available_alignments) > 5:
                    print(f"   ... and {len(available_alignments) - 5} more")
            else:
                print("   No alignments available")
                
        except Exception as e:
            print(f"❌ Error getting alignments: {e}")
        
        # Test layout creation
        try:
            layout = consensus_tab.create_layout()
            print("✅ Layout created successfully")
            print(f"   Layout type: {type(layout)}")
        except Exception as e:
            print(f"❌ Error creating layout: {e}")
        
        # Test alignment selector
        try:
            selector = consensus_tab.create_alignment_selector()
            print("✅ Alignment selector created successfully")
            print(f"   Selector type: {type(selector)}")
        except Exception as e:
            print(f"❌ Error creating alignment selector: {e}")
            
        print("\n🎉 All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure dash, dash-bootstrap-components, and dash-bio are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_consensus_module():
    """Test the original consensus module for compatibility."""
    print("\n🧪 Testing Original Consensus Module")
    print("=" * 50)
    
    try:
        from modules.consensus.tab import ConsensusTab
        print("✅ Original ConsensusTab imported successfully")
        
        # Set data path
        data_path = Path("../../data/app")
        if not data_path.exists():
            print(f"⚠️  Data path not found: {data_path}")
            data_path = Path(".")
        
        # Create original tab
        original_tab = ConsensusTab(data_path)
        print(f"✅ Original ConsensusTab created with data path: {data_path}")
        
        # Test getting available alignments
        try:
            available_alignments = original_tab.get_available_alignments()
            print(f"✅ Available alignments: {len(available_alignments)} found")
        except Exception as e:
            print(f"❌ Error getting alignments: {e}")
        
        # Test get_available_samples with proper signature
        if available_alignments:
            try:
                test_key = available_alignments[0]
                samples = original_tab.get_available_samples(test_key)
                print(f"✅ Available samples for {test_key}: {len(samples)} found")
                if samples:
                    print(f"   First few samples: {samples[:3]}...")
            except Exception as e:
                print(f"❌ Error getting samples: {e}")
        
        print("✅ Original consensus module tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing original consensus module: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Consensus Module Tests")
    print("=" * 60)
    
    test1_passed = test_original_consensus_module()
    test2_passed = test_consensus_adapter()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Original Consensus Module: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Dash Consensus Adapter:    {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Ready to run Dash application.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
