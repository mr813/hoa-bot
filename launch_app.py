#!/usr/bin/env python3
"""
Simple launcher for HOA Auditor that handles Python path correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'

if __name__ == "__main__":
    # Import and run the main function
    from app.main import main
    main()
