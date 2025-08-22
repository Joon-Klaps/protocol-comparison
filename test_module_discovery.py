#!/usr/bin/env python3
"""
Test module discovery for the Dash application.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_module_discovery():
    """Test that all modules are properly discovered by the Dash app."""
    print("🧪 Testing Module Discovery for Dash App")
    print("=" * 50)
    
    try:
        # Import the module discovery from dash_app
        from dash_app import ModuleDiscovery
        
        # Set up module discovery
        modules_path = Path(__file__).parent / "modules"
        discovery = ModuleDiscovery(modules_path)
        
        # Discover modules
        available_modules = discovery.discover_modules()
        
        print(f"✅ Found {len(available_modules)} modules:")
        
        for module_name, module_info in available_modules.items():
            print(f"\n📊 Module: {module_name}")
            print(f"   Title: {module_info.get('title', 'N/A')}")
            print(f"   Icon: {module_info.get('icon', 'N/A')}")
            print(f"   Order: {module_info.get('order', 'N/A')}")
            print(f"   Description: {module_info.get('description', 'N/A')}")
            
            # Check if it has a Dash adapter
            dash_adapter = module_info.get('dash_adapter_module')
            if dash_adapter:
                print(f"   ✅ Has Dash adapter")
                
                # Test if we can create the adapter
                try:
                    if hasattr(dash_adapter, f'create_{module_name}_dash_component'):
                        factory_func = getattr(dash_adapter, f'create_{module_name}_dash_component')
                        component = factory_func(Path("../../data/app"))
                        print(f"   ✅ Can create Dash component")
                        
                        # Test layout creation
                        layout = component.create_layout()
                        print(f"   ✅ Can create layout")
                        
                    else:
                        print(f"   ⚠️  Missing factory function")
                        
                except Exception as e:
                    print(f"   ❌ Error creating component: {e}")
            else:
                print(f"   ⚠️  No Dash adapter")
        
        if available_modules:
            print(f"\n🎉 Module discovery successful!")
            return True
        else:
            print(f"\n❌ No modules discovered")
            return False
            
    except Exception as e:
        print(f"❌ Error during module discovery: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_path():
    """Test if the expected data path exists."""
    print("\n🧪 Testing Data Path")
    print("=" * 30)
    
    data_path = Path("../../data/app")
    print(f"📁 Expected data path: {data_path}")
    print(f"   Exists: {data_path.exists()}")
    
    if data_path.exists():
        print(f"   Is directory: {data_path.is_dir()}")
        
        # Check for key subdirectories
        subdirs = ['alignments', 'comparison_excels', 'custom_vcfs']
        for subdir in subdirs:
            subpath = data_path / subdir
            print(f"   {subdir}/: {subpath.exists()}")
        
        return True
    else:
        print("   ❌ Data path does not exist")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Dash Application Setup")
    print("=" * 60)
    
    test1_passed = test_data_path()
    test2_passed = test_module_discovery()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Data Path:        {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Module Discovery: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Dash app should work correctly.")
        print("\n💡 Next steps:")
        print("   1. Run: python dash_app.py")
        print("   2. Open: http://localhost:8050")
        print("   3. Set data path to: ../../data/app")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
