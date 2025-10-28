#!/usr/bin/env python3
"""
Quick start script for Enhanced RAG with Visual capabilities
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path


def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🧠 Enhanced RAG with Knowledge Graph               ║
    ║                                                              ║
    ║              🖼️  Visual RAG + 🌐 Interactive Web UI          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_setup():
    """Check if basic setup is complete"""
    print("🔍 Checking setup...")
    
    # Check if in correct directory
    required_files = ["requirements.txt", "webapp/app.py", "notebooks/visual_rag_demo.ipynb"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("Please run this script from the project root directory.")
        return False
    
    print("✅ Project structure looks good!")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run manually: pip install -r requirements.txt")
        return False


def check_token():
    """Check for HuggingFace token"""
    print("\n🔑 Checking for HuggingFace token...")
    
    # Check environment variable
    if os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN'):
        print("✅ HuggingFace token found in environment!")
        return True
    
    # Check if token is configured in notebook or webapp
    print("⚠️ HuggingFace token not found in environment variables")
    print("You'll need to configure it in the web interface or notebook")
    return False


def show_menu():
    """Show startup menu"""
    print("\n🚀 What would you like to do?")
    print("1. 🌐 Launch Web Interface (Streamlit)")
    print("2. 📓 Open Demo Notebook (Jupyter)")
    print("3. 🧪 Run Tests")
    print("4. 📚 View Documentation")
    print("5. ❌ Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    return choice


def launch_streamlit():
    """Launch Streamlit web interface"""
    print("\n🌐 Launching Web Interface...")
    print("This will open in your default browser...")
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "webapp/app.py"
        ])
        
        # Wait a moment then try to open browser
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass
        
        print("✅ Streamlit is running!")
        print("🌐 Open in browser: http://localhost:8501")
        print("📝 Configure your HuggingFace token in the sidebar")
        print("🛑 Press Ctrl+C to stop")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Streamlit...")
        process.terminate()
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        print("Try running manually: streamlit run webapp/app.py")


def launch_jupyter():
    """Launch Jupyter notebook"""
    print("\n📓 Launching Jupyter Notebook...")
    
    try:
        # Try to start Jupyter Lab first, fallback to notebook
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "jupyter", "lab", "notebooks/visual_rag_demo.ipynb"
            ])
            print("✅ Jupyter Lab started!")
        except:
            process = subprocess.Popen([
                sys.executable, "-m", "jupyter", "notebook", "notebooks/visual_rag_demo.ipynb"
            ])
            print("✅ Jupyter Notebook started!")
        
        print("📓 Demo notebook should open in your browser")
        print("🛑 Press Ctrl+C to stop")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Jupyter...")
        process.terminate()
    except Exception as e:
        print(f"❌ Failed to launch Jupyter: {e}")
        print("Try running manually: jupyter lab notebooks/visual_rag_demo.ipynb")


def run_tests():
    """Run test suite"""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, "run_tests.py", "--quick"], 
                               capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed - check output above")
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")


def show_documentation():
    """Show documentation information"""
    print("\n📚 Documentation & Resources")
    print("=" * 50)
    print("📄 Enhanced README: README_ENHANCED.md")
    print("📓 Demo Notebook: notebooks/visual_rag_demo.ipynb")
    print("🌐 Web Interface: Launch option 1 to explore")
    print("🧪 Tests: run_tests.py for validation")
    print()
    print("📖 Key Features:")
    print("  • 🖼️ Multimodal Visual RAG with image processing")
    print("  • 🌐 Interactive web interface with knowledge graph visualization")
    print("  • 🔍 Enhanced provenance tracking")
    print("  • 💬 User feedback collection system")
    print()
    print("🎯 Quick Start:")
    print("  1. Configure HuggingFace token")
    print("  2. Launch web interface or notebook")
    print("  3. Upload documents and images")
    print("  4. Start querying with natural language!")
    
    input("\nPress Enter to continue...")


def main():
    """Main startup function"""
    print_banner()
    
    # Basic setup check
    if not check_setup():
        sys.exit(1)
    
    # Check dependencies
    try:
        import streamlit
        import torch
        import transformers
        deps_ok = True
    except ImportError:
        print("⚠️ Some dependencies are missing")
        install_choice = input("Install dependencies now? (y/n): ").lower().strip()
        if install_choice == 'y':
            deps_ok = install_dependencies()
        else:
            print("Please install dependencies manually: pip install -r requirements.txt")
            deps_ok = False
    else:
        deps_ok = True
        print("✅ Dependencies are installed")
    
    # Check token
    check_token()
    
    # Main menu loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            if deps_ok:
                launch_streamlit()
            else:
                print("❌ Please install dependencies first")
                
        elif choice == '2':
            if deps_ok:
                launch_jupyter()
            else:
                print("❌ Please install dependencies first")
                
        elif choice == '3':
            run_tests()
            
        elif choice == '4':
            show_documentation()
            
        elif choice == '5':
            print("\n👋 Goodbye! Happy exploring with Visual RAG!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()