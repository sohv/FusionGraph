#!/usr/bin/env python3
"""
Test runner for the enhanced RAG-knowledgegraph project
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'llama-index',
        'streamlit',
        'torch',
        'transformers',
        'pillow',
        'opencv-python',
        'sentence-transformers',
        'networkx',
        'plotly',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def run_unit_tests():
    """Run unit tests"""
    test_files = [
        "tests/test_image_ingest.py",
        "tests/test_visual_rag.py"
    ]
    
    success = True
    for test_file in test_files:
        if os.path.exists(test_file):
            success &= run_command(f"python {test_file}", f"Running {test_file}")
        else:
            print(f"⚠️ Test file not found: {test_file}")
    
    return success


def run_integration_tests():
    """Run integration tests that require model loading"""
    print("\n🧪 Running integration tests (this may take a while)...")
    return run_command(
        "RUN_INTEGRATION_TESTS=1 python tests/test_image_ingest.py",
        "Integration tests for image ingestion"
    )


def validate_project_structure():
    """Validate that all required files exist"""
    print("📁 Validating project structure...")
    
    required_files = [
        "requirements.txt",
        "ingest/image_ingest.py",
        "pipeline/visual_rag.py",
        "webapp/app.py",
        "webapp/provenance.py",
        "webapp/feedback_sink.py",
        "tools/visual_utils.py",
        "notebooks/visual_rag_demo.ipynb",
        "tests/test_image_ingest.py",
        "tests/test_visual_rag.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present!")
    return True


def test_basic_imports():
    """Test basic imports to ensure modules load correctly"""
    print("📦 Testing basic imports...")
    
    imports_to_test = [
        ("ingest.image_ingest", "ImageIngestor"),
        ("pipeline.visual_rag", "VisualRAGPipeline"),
        ("webapp.feedback_sink", "FeedbackCollector"),
        ("webapp.provenance", "ProvenanceExtractor"),
        ("tools.visual_utils", "VisualUtils")
    ]
    
    sys.path.insert(0, os.getcwd())
    
    success = True
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} - Import Error: {e}")
            success = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} - Attribute Error: {e}")
            success = False
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - Error: {e}")
            success = False
    
    return success


def run_streamlit_test():
    """Test if Streamlit app can be imported and basic syntax is correct"""
    print("🌐 Testing Streamlit app...")
    
    try:
        # Test syntax by importing the module
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Try to check the app file syntax
        with open("webapp/app.py", "r") as f:
            content = f.read()
            compile(content, "webapp/app.py", "exec")
        
        print("✅ Streamlit app syntax is valid")
        return True
        
    except ImportError:
        print("❌ Streamlit is not installed")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error in Streamlit app: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Streamlit app: {e}")
        return False


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test runner for enhanced RAG-knowledgegraph")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick validation tests")
    args = parser.parse_args()
    
    print("🚀 Enhanced RAG-knowledgegraph Test Runner")
    print("=" * 60)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    all_passed = True
    
    # Quick validation tests
    all_passed &= validate_project_structure()
    all_passed &= check_dependencies()
    all_passed &= test_basic_imports()
    all_passed &= run_streamlit_test()
    
    if not args.quick:
        # Unit tests
        all_passed &= run_unit_tests()
        
        # Integration tests (if requested)
        if args.integration:
            all_passed &= run_integration_tests()
    
    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your enhanced RAG system is ready to use!")
        print("\nNext steps:")
        print("1. Run: streamlit run webapp/app.py")
        print("2. Open: jupyter lab notebooks/visual_rag_demo.ipynb")
        print("3. Configure your HuggingFace token")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Please fix the issues above before proceeding")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check file permissions and paths")
        print("3. Verify Python environment and imports")
        exit_code = 1
    
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()