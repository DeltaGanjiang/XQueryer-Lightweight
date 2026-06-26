"""Dynamic XRD datasets and preprocessing utilities.

Folder: src/model
Purpose: dataset classes, simulated XRD generation, interpolation, and
experimental XRD loading for the lightweight XQueryer pipeline.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase.db import connect
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

try:
    from Pysimxrd import generator
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    generator = None


XRD_MIN_ANGLE = 10.0
XRD_MAX_ANGLE = 90.0
DEFAULT_XRD_POINTS = 3500
DEFAULT_NUM_CLASSES = 100315


@dataclass(frozen=True)
class SimulationRanges:
    grainsize: tuple[float, float] = (10.0, 100.0)
    prefect_orientation: tuple[float, float] = (-0.3, 0.3)
    thermo_vibration: tuple[float, float] = (0.0, 2.0)
    zero_shift: tuple[float, float] = (0.0, 2.0)
    dis_detector2sample: tuple[float, float] = (300.0, 600.0)
    half_height_slit_detector: tuple[float, float] = (3.0, 8.0)
    half_height_sample: tuple[float, float] = (1.0, 4.0)
    background_ratio: tuple[float, float] = (0.02, 0.2)
    mixture_noise_ratio: tuple[float, float] = (0.01, 0.1)


def split_entry_ids(num_entries: int, split: str, seed: int = 42) -> list[int]:
    ids = list(range(1, num_entries + 1))
    rng = random.Random(seed)
    rng.shuffle(ids)
    train_end = int(num_entries * 0.8)
    val_end = int(num_entries * 0.9)
    if split == "train":
        return ids[:train_end]
    if split == "val":
        return ids[train_end:val_end]
    if split == "test":
        return ids[val_end:]
    if split == "all":
        return ids
    raise ValueError(f"Unknown split: {split}")


def sample_simulation_kwargs(rng: random.Random, ranges: SimulationRanges | None = None) -> dict[str, object]:
    ranges = ranges or SimulationRanges()
    return {
        "grainsize": rng.uniform(*ranges.grainsize),
        "prefect_orientation": [
            rng.uniform(*ranges.prefect_orientation),
            rng.uniform(*ranges.prefect_orientation),
        ],
        "thermo_vibration": rng.uniform(*ranges.thermo_vibration),
        "zero_shift": rng.uniform(*ranges.zero_shift),
        "dis_detector2sample": rng.uniform(*ranges.dis_detector2sample),
        "half_height_slit_detector": rng.uniform(*ranges.half_height_slit_detector),
        "half_height_sample": rng.uniform(*ranges.half_height_sample),
        "xrd": "reciprocal",
        "background_order": 6,
        "background_ratio": rng.uniform(*ranges.background_ratio),
        "mixture_noise_ratio": rng.uniform(*ranges.mixture_noise_ratio),
    }


def interpolate_xrd(
    angles: Iterable[float],
    intensities: Iterable[float],
    num_points: int = DEFAULT_XRD_POINTS,
    min_angle: float = XRD_MIN_ANGLE,
    max_angle: float = XRD_MAX_ANGLE,
) -> np.ndarray:
    angles_np = np.asarray(list(angles), dtype=np.float32)
    intensities_np = np.asarray(list(intensities), dtype=np.float32)
    finite = np.isfinite(angles_np) & np.isfinite(intensities_np)
    angles_np = angles_np[finite]
    intensities_np = intensities_np[finite]
    if len(angles_np) < 2:
        raise ValueError("XRD data must contain at least two finite points.")

    order = np.argsort(angles_np)
    angles_np = angles_np[order]
    intensities_np = intensities_np[order]
    unique_angles, unique_indices = np.unique(angles_np, return_index=True)
    intensities_np = intensities_np[unique_indices]

    target = np.linspace(min_angle, max_angle, num_points, dtype=np.float32)
    interpolator = interp1d(unique_angles, intensities_np, kind="linear", bounds_error=False, fill_value=0.0)
    y = np.asarray(interpolator(target), dtype=np.float32)
    y = np.clip(y, a_min=0.0, a_max=None)
    scale = float(y.max())
    if scale > 0:
        y = y / scale * 100.0
    return y.astype(np.float32)


def element_vector_from_atoms(atoms, max_atomic_number: int = 92) -> np.ndarray:
    vector = np.zeros(max_atomic_number, dtype=np.float32)
    for number in atoms.get_atomic_numbers():
        if 1 <= int(number) <= max_atomic_number:
            vector[int(number) - 1] = 1.0
    return vector


class SimulatedXRDDataset(Dataset):
    """Generate multiple simulated XRD patterns per crystal structure on demand."""

    def __init__(
        self,
        db_path: str | Path = "data/MP500.db",
        split: str = "train",
        simulations_per_entry: int = 20,
        seed: int = 42,
        num_points: int = DEFAULT_XRD_POINTS,
        max_entries: int | None = None,
        cache_structures: bool = False,
    ):
        self.db_path = str(db_path)
        self.split = split
        self.simulations_per_entry = simulations_per_entry
        self.seed = seed
        self.num_points = num_points
        self.cache_structures = cache_structures
        self._db = None
        self._element_cache: dict[int, np.ndarray] = {}

        db = connect(self.db_path)
        self.num_entries = len(db)
        entry_ids = split_entry_ids(self.num_entries, split=split, seed=seed)
        if max_entries is not None:
            entry_ids = entry_ids[:max_entries]
        self.entry_ids = entry_ids

    def __len__(self) -> int:
        return len(self.entry_ids) * self.simulations_per_entry

    @property
    def db(self):
        if self._db is None:
            self._db = connect(self.db_path)
        return self._db

    def _element_vector(self, entry_id: int) -> np.ndarray:
        if self.cache_structures and entry_id in self._element_cache:
            return self._element_cache[entry_id]
        atoms = self.db.get(entry_id).toatoms()
        vector = element_vector_from_atoms(atoms)
        if self.cache_structures:
            self._element_cache[entry_id] = vector
        return vector

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if generator is None:
            raise ImportError("Pysimxrd is required for dynamic XRD simulation.")

        entry_offset = index // self.simulations_per_entry
        sim_offset = index % self.simulations_per_entry
        entry_id = self.entry_ids[entry_offset]
        rng = random.Random(self.seed * 1_000_003 + entry_id * 101 + sim_offset)
        kwargs = sample_simulation_kwargs(rng)
        angles, intensities = generator.parser(self.db, entry_id, **kwargs)
        xrd = interpolate_xrd(angles, intensities, num_points=self.num_points)
        label = int(entry_id - 1)
        return {
            "intensity": torch.from_numpy(xrd),
            "id": torch.tensor(label, dtype=torch.long),
            "element": torch.from_numpy(self._element_vector(entry_id)),
            "entry_id": torch.tensor(entry_id, dtype=torch.long),
        }


class ExperimentalXRDDataset(Dataset):
    """Load user-provided experimental CSV files for inference."""

    def __init__(self, paths: list[str | Path], num_points: int = DEFAULT_XRD_POINTS):
        self.paths = [Path(path) for path in paths]
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        angles: list[float] = []
        intensities: list[float] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                angles.append(float(row["angle"]))
                intensities.append(float(row["intensity"]))
        xrd = interpolate_xrd(angles, intensities, num_points=self.num_points)
        return {
            "intensity": torch.from_numpy(xrd),
            "element": torch.zeros(92, dtype=torch.float32),
            "path": str(path),
        }


ASEDataset = SimulatedXRDDataset
EXPDataset = ExperimentalXRDDataset
