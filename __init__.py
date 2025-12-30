import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
