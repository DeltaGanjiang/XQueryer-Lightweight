"""Compatibility wrapper for the root inference script.

Folder: src
Purpose: source package for XQueryer Lightweight. Use ../inference.py for the
current experimental-XRD inference workflow.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "inference.py"), run_name="__main__")
