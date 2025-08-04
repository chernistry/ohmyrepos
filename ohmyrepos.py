#!/usr/bin/env python
"""Oh My Repos CLI wrapper.

This script serves as an entry point for the Oh My Repos CLI.
It ensures that the project's directory is in the Python path.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project directory to the Python path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

# Import the CLI app
from src.cli import app

if __name__ == "__main__":
    # Run the CLI app
    app() 