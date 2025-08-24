#!/usr/bin/env python3
"""
HOA Auditor Launcher Script

This script launches the HOA Auditor Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the HOA Auditor app."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    if not (script_dir / "app" / "main.py").exists():
        print("Error: app/main.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import fitz  # PyMuPDF
    except ImportError as e:
        print(f"Error: Missing required dependencies. Please install requirements:")
        print("pip install -r requirements.txt")
        print(f"Missing: {e}")
        sys.exit(1)
    
    # Set up environment
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    # Launch Streamlit app
    app_path = script_dir / "app" / "main.py"
    
    print("🏠 Starting HOA Auditor...")
    print("📖 Florida Chapter 718 Compliance Checker")
    print("🌐 Opening in browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 HOA Auditor stopped.")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
