"""Non-Cu target generation following xrdbench/prepare_targets.py.

Kα1/Kα2 lines use a 2:1 intensity ratio. Peak positions and intensities are
recalculated from the crystal at each wavelength, then receive TCH pseudo-Voigt
broadening, a positive background, Poisson counts and Gaussian read noise.
The dataset keeps its original Pysimxrd branch for Cu.
"""

from __future__ import annotations

import numpy as np


# Kα1, Kα2 and nominal Kα wavelengths in Å, exactly as in xrdbench.
RADIATION = {
    "Cu": (1.5406, 1.5444, 1.5418),
    "Co": (1.7890, 1.7929, 1.7903),
    "Mo": (0.7093, 0.7136, 0.7107),
    "Cr": (2.2897, 2.2936, 2.2910),
    "Fe": (1.9360, 1.9400, 1.9374),
    "Ag": (0.5594, 0.5638, 0.5609),
}


def resolve_target(target: str) -> str:
    """Accept element symbols and Kα aliases such as CoKa."""
    value = target.strip().lower()
    for symbol in RADIATION:
        if value in (symbol.lower(), f"{symbol.lower()}ka"):
            return symbol
    raise ValueError(f"Unknown X-ray target {target!r}; choose from {', '.join(RADIATION)}")


def _instrument(rng: np.random.Generator) -> dict[str, float]:
    return {
        "caglioti_U_deg2": float(rng.uniform(0.002, 0.012)),
        "caglioti_V_deg2": 0.0,
        "caglioti_W_deg2": float(rng.uniform(0.0025, 0.010)),
        "zero_shift_deg": float(rng.uniform(-0.025, 0.025)),
        "displacement_cos_theta_deg": float(rng.uniform(-0.025, 0.025)),
    }


def _microstructure(rng: np.random.Generator) -> dict[str, float]:
    return {
        "domain_size_nm": float(np.exp(rng.uniform(np.log(50), np.log(250)))),
        "microstrain": float(rng.uniform(0.00005, 0.0008)),
        "isotropic_B_A2": float(rng.uniform(0.3, 1.2)),
    }


def _tch_width(gaussian: np.ndarray, lorentzian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thompson–Cox–Hastings approximation to a Voigt profile."""
    g, l = gaussian, lorentzian
    width = (
        g**5 + 2.69269 * g**4 * l + 2.42843 * g**3 * l**2
        + 4.47163 * g**2 * l**3 + 0.07842 * g * l**4 + l**5
    ) ** 0.2
    ratio = l / width
    eta = np.clip(1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3, 0, 1)
    return width, eta


def _phase_signal(structure, target: str, grid: np.ndarray, instrument: dict, micro: dict) -> np.ndarray:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    signal = np.zeros(len(grid), dtype=float)
    for wavelength, line_weight in zip(RADIATION[target][:2], (2 / 3, 1 / 3)):
        # Calculate both lines from the structure with a shared absolute scale.
        # Scaling each line independently would change their intensity ratio.
        pattern = XRDCalculator(wavelength=wavelength).get_pattern(
            structure, scaled=False,
            two_theta_range=(max(0.01, float(grid[0]) - 2), min(179.99, float(grid[-1]) + 2)),
        )
        position = np.asarray(pattern.x, dtype=float)
        if not len(position):
            continue
        theta = np.radians(position / 2)
        d = np.asarray(pattern.d_hkls, dtype=float)
        intensity = np.asarray(pattern.y, dtype=float) * np.exp(-micro["isotropic_B_A2"] / (2 * d**2)) * line_weight
        position += instrument["zero_shift_deg"] + instrument["displacement_cos_theta_deg"] * np.cos(theta)
        tangent = np.tan(theta)
        gaussian = np.sqrt(
            instrument["caglioti_U_deg2"] * tangent**2
            + instrument["caglioti_V_deg2"] * tangent
            + instrument["caglioti_W_deg2"]
            + np.degrees(4 * micro["microstrain"] * tangent) ** 2
        )
        lorentzian = np.degrees(0.9 * wavelength / (10 * micro["domain_size_nm"] * np.cos(theta)))
        fwhm, eta = _tch_width(gaussian, lorentzian)
        for center, area, width, fraction in zip(position, intensity, fwhm, eta):
            lower = int(np.searchsorted(grid, center - 40 * width))
            upper = int(np.searchsorted(grid, center + 40 * width, side="right"))
            if lower == upper:
                continue
            delta = (grid[lower:upper] - center) / width
            gaussian_profile = np.sqrt(4 * np.log(2) / np.pi) / width * np.exp(-4 * np.log(2) * delta**2)
            lorentzian_profile = 2 / (np.pi * width) / (1 + 4 * delta**2)
            signal[lower:upper] += area * ((1 - fraction) * gaussian_profile + fraction * lorentzian_profile)
    return signal


def _observe(signal: np.ndarray, grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(signal).all() or signal.max() <= 0:
        raise ValueError("Structure has no positive diffraction signal in the scan range")
    x = (grid - grid[0]) / (grid[-1] - grid[0])
    background = rng.uniform(0.008, 0.035) * (1 + rng.uniform(-0.35, 0.35) * (2 * x - 1))
    background += rng.uniform(0, 0.012) * (2 * x - 1)**2
    background += rng.uniform(0.005, 0.04) * np.exp(-(grid - grid[0]) / rng.uniform(8, 25))
    for _ in range(int(rng.integers(1, 3))):
        center, sigma, amplitude = rng.uniform(17, 65), rng.uniform(4, 13), rng.uniform(0.003, 0.025)
        background += amplitude * np.exp(-0.5 * ((grid - center) / sigma)**2)
    peak_counts = float(np.exp(rng.uniform(np.log(3000), np.log(30000))))
    read_sigma = float(rng.uniform(0.5, 2.5))
    expected = peak_counts * (signal / signal.max() + background)
    counts = np.maximum(rng.poisson(expected) + rng.normal(0, read_sigma, len(grid)), 0)
    normalizer = float(np.sum((counts[1:] + counts[:-1]) / 2))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Simulated observation has no positive integrated intensity")
    return (counts / normalizer).astype(np.float32)


def simulate_target_pattern(atoms, target: str, rng: np.random.Generator, *, min_angle=10.0, max_angle=90.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate one target-specific single-phase pattern on a 0.01° grid.

    Uses a generic non-resonant laboratory model, with no composition-specific
    fluorescence, absorption, preferred orientation or Kβ contribution.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    target = resolve_target(target)
    if not np.isfinite([min_angle, max_angle]).all() or not 0 < min_angle < max_angle < 178:
        raise ValueError("Require 0 < min_angle < max_angle < 178 degrees")
    points = round((max_angle - min_angle) / 0.01) + 1
    grid = np.linspace(min_angle, max_angle, points)
    structure = AseAtomsAdaptor.get_structure(atoms)
    signal = _phase_signal(structure, target, grid, _instrument(rng), _microstructure(rng))
    return grid, _observe(signal, grid, rng)
