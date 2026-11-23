"""Laboratory PXRD from ASE cells, with Diffra-style randomized views.

The line wavelengths match XQueryer-Lightweight. Bragg intensities use
pymatgen's scattering coefficients and Lorentz-polarization correction.
"""

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
from pymatgen.core import Element


RADIATION = {
    "Cu": (1.5406, 1.5444, 1.5418),
    "Co": (1.7890, 1.7929, 1.7903),
    "Mo": (0.7093, 0.7136, 0.7107),
    "Cr": (2.2897, 2.2936, 2.2910),
    "Fe": (1.9360, 1.9400, 1.9374),
    "Ag": (0.5594, 0.5638, 0.5609),
}
LINE_WEIGHTS = (2 / 3, 1 / 3)
CU_ZERO_SHIFT_DEG = (0.0, 4.0)
NON_CU_ZERO_SHIFT_DEG = (-0.05, 0.05)
ORIENTATION_MAX = 0.6
ANGLE_MIN, ANGLE_MAX, POINTS = 10.0, 90.0, 3500
GRID = np.linspace(ANGLE_MIN, ANGLE_MAX, POINTS, dtype=np.float32)


def scattering_coefficients(numbers):
    return np.array([ATOMIC_SCATTERING_PARAMS[Element.from_Z(int(z)).symbol]
                     for z in numbers], dtype=np.float32)


class Simulator:
    def __init__(self, device):
        self.device = torch.device(device)
        self.grid = self.tensor(GRID)
        self.step = (ANGLE_MAX - ANGLE_MIN) / (POINTS - 1)
        x = torch.linspace(-1, 1, POINTS, device=self.device)
        self.background_basis = torch.stack([x ** power for power in range(7)])

    def tensor(self, data, dtype=torch.float32):
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    @torch.inference_mode()
    def reflections(self, atoms, wavelengths, angle_range):
        """Return reciprocal lengths and unscaled |F|², counting both ±hkl."""
        cell = np.asarray(atoms.cell, dtype=np.float64)
        if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-9:
            raise ValueError("missing or singular crystal cell")
        coords = self.tensor(atoms.get_scaled_positions(wrap=True).T.copy())
        numbers, species = np.unique(atoms.numbers, return_inverse=True)
        coefficients = self.tensor(scattering_coefficients(numbers))
        atom_types = self.tensor(species, torch.long)
        charges = self.tensor(numbers)
        reciprocal = self.tensor(np.linalg.inv(cell).T, torch.float64)
        min_angle, max_angle = angle_range
        q_min = 2 * np.sin(np.deg2rad(min_angle / 2)) / max(wavelengths)
        q_max = 2 * np.sin(np.deg2rad(max_angle / 2)) / min(wavelengths)
        h_max, k_max, l_max = np.ceil(q_max * np.linalg.norm(cell, axis=1)).astype(int)
        k_size, l_size = 2 * k_max + 1, 2 * l_max + 1
        plane_size = k_size * l_size
        count = (h_max + 1) * plane_size
        q_values, strengths = [], []
        reflection_batch = max(512, min(32768, 2_000_000 // len(atoms)))
        for first in range(0, count, 131072):
            index = torch.arange(first, min(first + 131072, count), device=self.device)
            h = index // plane_size
            k = index // l_size % k_size - k_max
            l = index % l_size - l_max
            # A real scattering factor gives identical intensities for Friedel pairs.
            half = (h > 0) | (k > 0) | ((k == 0) & (l > 0))
            hkl = torch.stack((h, k, l), dim=1)[half].to(torch.float64)
            q2 = ((hkl @ reciprocal) ** 2).sum(dim=1)
            in_range = (q2 >= q_min**2 - 1e-12) & (q2 <= q_max**2 + 1e-12) & (q2 > 0)
            hkl, q2 = hkl[in_range], q2[in_range]
            for start in range(0, len(hkl), reflection_batch):
                group = slice(start, start + reflection_batch)
                s2 = q2[group].float()[:, None] / 4
                form = charges - 41.78214 * s2 * (
                    coefficients[None, :, :, 0]
                    * torch.exp(-s2[:, :, None] * coefficients[None, :, :, 1])
                ).sum(dim=-1)
                phase = (hkl[group].float() @ coords) * (2 * np.pi)
                form = form[:, atom_types]
                real = (form * torch.cos(phase)).sum(dim=1)
                imag = (form * torch.sin(phase)).sum(dim=1)
                q_values.append(q2[group].sqrt())
                strengths.append(2 * (real.square() + imag.square()))
        if not q_values:
            return np.empty(0), np.empty(0)
        return torch.cat(q_values).cpu().numpy(), torch.cat(strengths).cpu().numpy()

    def peaks(self, atoms, wavelengths, angle_range, weights=None):
        """Merge reflections at coincident angles separately for each Kα line."""
        q, strength = self.reflections(atoms, wavelengths, angle_range)
        order = np.argsort(q)
        q, strength = q[order], strength[order]
        edges = np.sin(np.deg2rad(np.array(angle_range) / 2))
        lines = []
        weights = weights if weights is not None else [1.0] * len(wavelengths)
        for wave, weight in zip(wavelengths, weights, strict=True):
            sin_theta = wave * q / 2
            keep = (sin_theta >= edges[0] - 1e-12) & (sin_theta <= edges[1] + 1e-12)
            sin_theta = sin_theta[keep]
            if not len(sin_theta):
                continue
            theta = np.arcsin(sin_theta)
            positions = np.rad2deg(2 * theta)
            polarization = (1 + np.cos(2 * theta) ** 2) / (sin_theta**2 * np.cos(theta))
            mass = strength[keep] * polarization * weight
            tolerance = 1e-5
            starts = np.r_[0, np.flatnonzero(np.diff(positions) >= tolerance) + 1]
            ends = np.r_[starts[1:], len(positions)]
            # Pymatgen compares with the first angle of each peak, not just its neighbor.
            split = positions[ends - 1] - positions[starts] >= tolerance
            extra = []
            for first, end in zip(starts[split], ends[split]):
                while positions[end - 1] - positions[first] >= tolerance:
                    first = np.searchsorted(positions, positions[first] + tolerance)
                    extra.append(first)
            if extra:
                starts = np.sort(np.r_[starts, extra])
            lines.append(np.stack((positions[starts], np.add.reduceat(mass, starts),
                                   np.full(len(starts), wave))))
        if not lines:
            return np.empty((3, 0), dtype=np.float32)
        result = np.concatenate(lines, axis=1)
        # Pymatgen's relative threshold is 0.001 on the 0–100 scale.
        return result[:, result[1] > result[1].max() * 1e-5].astype(np.float32)

    @torch.inference_mode()
    def broaden(self, peaks, grain_nm, orientation, vibration, zero_shift, *, gaussian_u=0.06**2, gaussian_w=0.06**2):
        """Spread each peak with a TCH pseudo-Voigt on the actual 2θ axis."""
        view_count = len(grain_nm)
        profile = torch.zeros((view_count, POINTS), device=self.device)
        if peaks.shape[1] == 0:
            return profile
        center, area, wave = self.tensor(peaks)
        theta = torch.deg2rad(center / 2)
        cosine = theta.cos()
        lorentz = torch.rad2deg(0.888 * wave / (10 * self.tensor(grain_nm)[:, None] * cosine))
        gaussian = (self.tensor(gaussian_w).reshape(-1, 1)
                    + self.tensor(gaussian_u).reshape(-1, 1) * theta.tan().square()).sqrt()
        width = (gaussian**5 + 2.69269 * gaussian**4 * lorentz + 2.42843 * gaussian**3 * lorentz**2
                 + 4.47163 * gaussian**2 * lorentz**3 + 0.07842 * gaussian * lorentz**4 + lorentz**5) ** 0.2
        ratio = lorentz / width
        eta = (1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3).clamp(0, 1)
        thermal = (16 / 3 * np.pi**2) * (self.tensor(vibration)[:, None] * theta.sin() / wave).square()
        log_strength = area.log() + self.tensor(orientation).log() - thermal
        strength = (log_strength - log_strength.amax(dim=1, keepdim=True)).exp()
        center = center + self.tensor(zero_shift)[:, None]
        radius = min(POINTS - 1, int(torch.ceil(width.max() * 12 / self.step).item()) + 1)
        offsets = torch.arange(-radius, radius + 1, device=self.device)
        for first in range(0, peaks.shape[1], 128):
            block = slice(first, first + 128)
            position = center[:, block, None]
            fwhm = width[:, block, None]
            fraction = eta[:, block, None]
            bins = torch.round((position - ANGLE_MIN) / self.step).long() + offsets
            inside = (bins >= 0) & (bins < POINTS)
            bins = bins.clamp(0, POINTS - 1)
            delta2 = ((self.grid[bins] - position) / fwhm).square()
            gaussian_pdf = np.sqrt(4 * np.log(2) / np.pi) * torch.exp(-4 * np.log(2) * delta2)
            lorentz_pdf = (2 / np.pi) / (1 + 4 * delta2)
            weights = ((1 - fraction) * gaussian_pdf + fraction * lorentz_pdf) / fwhm
            weights *= inside & (delta2 <= 12**2)
            weights *= strength[:, block, None]
            profile.scatter_add_(1, bins.flatten(1), weights.flatten(1))
        return profile

    @torch.inference_mode()
    def views(self, peaks, target, entry_id, count, seed):
        rng = np.random.default_rng(np.random.SeedSequence([seed, tuple(RADIATION).index(target), entry_id]))
        if target == "Cu":
            grain_nm = rng.uniform(10, 190, count) / 10  # Pysimxrd's grainsize is in Å.
            orientation_width = rng.uniform(0, ORIENTATION_MAX, (count, 1))
            orientation = np.clip(rng.normal(1, 0.2, (count, peaks.shape[1])),
                                  1 - orientation_width, 1 + orientation_width)
            vibration = rng.uniform(0, 3, count)
            shift = rng.uniform(*CU_ZERO_SHIFT_DEG, count)
            signal = self.broaden(peaks, grain_nm, orientation, vibration, shift)
        else:
            grain_nm = np.exp(rng.uniform(np.log(22.36), np.log(559.02), count))
            b_factor = rng.uniform(0, 1.8, count)
            # B = 8π² u² / 3 for the isotropic displacement used by Diffra.
            vibration = np.sqrt(3 * b_factor / (8 * np.pi**2))
            shift = rng.uniform(*NON_CU_ZERO_SHIFT_DEG, count)
            microstrain = rng.uniform(0, 1.5e-3, count)
            gaussian_u = rng.uniform(0, 0.020, count) + np.rad2deg(4 * microstrain)**2
            gaussian_w = rng.uniform(0, 0.015, count)
            signal = self.broaden(peaks, grain_nm, 1.0, vibration, shift,
                                  gaussian_u=gaussian_u, gaussian_w=gaussian_w)
        maximum = signal.amax(dim=1, keepdim=True)
        if not torch.isfinite(signal).all().item():
            raise ValueError("non-finite broadened diffraction signal")
        valid = maximum[:, 0] > 0
        signal = signal / torch.where(maximum > 0, maximum, 1)
        background = self.tensor(rng.normal(size=(count, 7))) @ self.background_basis
        background -= background.amin(dim=1, keepdim=True)
        background *= (self.tensor(rng.uniform(0.02, 0.2, (count, 1)))
                       / background.amax(dim=1, keepdim=True).clamp_min(1e-8))
        noise = self.tensor(rng.random((count, POINTS)) * rng.uniform(0, 0.18, (count, 1)))
        observed = signal + background.flip((1,)) + noise
        observed = observed / observed.amax(dim=1, keepdim=True).clamp_min(1e-8) * 100
        observed[~valid] = 0
        return observed.cpu().numpy(), valid.cpu().numpy(), shift.astype(np.float32)

    def simulate(self, atoms, target, entry_id, count, seed):
        waves = RADIATION[target][:2]
        # Include lines just outside the window that can move inside after a zero shift.
        peaks = self.peaks(atoms, waves, (max(0.1, ANGLE_MIN - 4), ANGLE_MAX + 2), LINE_WEIGHTS)
        return self.views(peaks, target, entry_id, count, seed)
