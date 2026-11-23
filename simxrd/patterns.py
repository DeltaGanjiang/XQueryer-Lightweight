"""Read Ganjiang HDF5 spectra and export one spectrum to existing inference inputs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GanjiangPatterns:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self._handle = None
        self._pid = None
        with h5py.File(self.path, "r") as data:
            if data.attrs.get("schema") != "ganjiang-pxrd-1" or not data.attrs.get("complete"):
                raise ValueError(f"Expected a completed Ganjiang PXRD file: {self.path}")
            self.target = str(data.attrs["target"])
            self.db_sha256 = str(data.attrs["db_sha256"])
            self.angles = data["two_theta_deg"][:]
            self.entry_ids = data["entry_id"][:]
            self.valid = data["valid"][:]
            self.element_dim = data["elements"].shape[1]
            self.shape = data["intensity"].shape
            if (self.valid.shape != self.shape[:2] or len(self.entry_ids) != self.shape[0]
                    or len(self.angles) != self.shape[2] or not len(self.entry_ids)
                    or not np.isfinite(self.angles).all() or not np.all(np.diff(self.angles) > 0)
                    or np.any(np.diff(self.entry_ids) <= 0) or self.entry_ids[0] < 1):
                raise ValueError(f"Inconsistent Ganjiang data axes: {self.path}")

    def __getstate__(self):
        return {**self.__dict__, "_handle": None, "_pid": None}

    @property
    def handle(self):
        pid = os.getpid()
        if self._pid != pid or self._handle is None:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = pid
        return self._handle

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def row_for_entry(self, entry_id):
        row = int(np.searchsorted(self.entry_ids, entry_id))
        if row == len(self.entry_ids) or self.entry_ids[row] != entry_id:
            raise KeyError(f"ASE entry {entry_id} not in {self.path}")
        return row

    def pattern(self, row, view, grid=None, *, correct_zero_shift=False):
        if not self.valid[row, view]:
            raise ValueError(f"No positive diffraction for ASE entry {self.entry_ids[row]}, view {view}")
        y = self.handle["intensity"][row, view]
        if grid is not None:
            angles = self.angles
            if correct_zero_shift:
                angles = angles - self.handle["zero_shift_deg"][row, view]
            if not np.array_equal(grid, angles):
                y = np.interp(grid, angles, y, left=0, right=0).astype(np.float32)
        return y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("entry_id", type=int, help="ASE database ID, starting at 1")
    parser.add_argument("--view", type=int, default=0, help="simulated view index, starting at 0")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args()
    data = GanjiangPatterns(args.file)
    if not 0 <= args.view < data.shape[1]:
        parser.error(f"view must be between 0 and {data.shape[1] - 1}")
    y = data.pattern(data.row_for_entry(args.entry_id), args.view)
    if args.format == "csv":
        np.savetxt(sys.stdout, np.column_stack((data.angles, y)), delimiter=",", fmt="%.8g",
                   header="angle,intensity", comments="")
    else:
        json.dump({"two_theta": data.angles.tolist(), "intensities": y.tolist()}, sys.stdout)
        print()
    data.close()


if __name__ == "__main__":
    main()
