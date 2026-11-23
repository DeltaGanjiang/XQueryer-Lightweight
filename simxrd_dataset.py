"""Compatibility entry point for the GanJiang HDF5 reader and spectrum export."""

from simxrd.patterns import GanjiangPatterns, file_sha256, main

__all__ = ["GanjiangPatterns", "file_sha256", "main"]


if __name__ == "__main__":
    main()
