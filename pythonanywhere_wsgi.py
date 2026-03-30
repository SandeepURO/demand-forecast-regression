"""WSGI entry-point template for PythonAnywhere.

Instructions:
1) Upload project folder to PythonAnywhere under /home/<username>/MLL
2) In Web tab -> WSGI configuration file, paste this content
3) Replace <username> below with your PythonAnywhere username
"""

import sys
from pathlib import Path

PROJECT_HOME = Path("/home/<username>/MLL")
if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))

from app import app as application
