"""Compatibility wrapper for the root trainer.

Folder: src
Purpose: source package for XQueryer Lightweight. Use ../trainer.py for the
current dynamic-simulation training workflow.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "trainer.py"), run_name="__main__")
