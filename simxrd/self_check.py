"""Check reference peak calculations, augmentation and resumable HDF5 output."""

import hashlib
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase.db import connect
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from generate import DATABASE, SCHEMA, db_catalog, read_block, run_target, verify_final, write_block
from patterns import GanjiangPatterns
from physics import GRID, POINTS, RADIATION, Simulator


def check_physics(device="cuda:0"):
    sim = Simulator(device)
    db = connect(DATABASE, use_lock_file=False)
    for entry_id in (1, 32, 1435):
        atoms = db.get_atoms(id=entry_id)
        structure = AseAtomsAdaptor.get_structure(atoms)
        for target in RADIATION:
            wavelength = RADIATION[target][0]
            want = XRDCalculator(wavelength=wavelength).get_pattern(structure, two_theta_range=(10, 90))
            peaks = sim.peaks(atoms, (wavelength,), (10, 90))
            np.testing.assert_allclose(peaks[0], want.x, atol=3e-5, rtol=0)
            np.testing.assert_allclose(100 * peaks[1] / peaks[1].max(), want.y, atol=0.004, rtol=0.0003)
    peaks = np.array([[35, 60], [10, 10], [1.5406, 1.5406]])
    profile = sim.broaden(peaks[:, :1], [45, 45], [[1], [1]], [0, 0], [0, 1.5]).cpu().numpy()
    np.testing.assert_allclose(np.diff(GRID[profile.argmax(axis=1)]), 1.5, atol=0.024)
    many_peaks = np.tile(peaks, 16)
    with patch.object(sim, "broaden", wraps=sim.broaden) as wrapped:
        y, valid, shifts = sim.views(many_peaks, "Cu", 1, 5, 42)
        grain, orientation, vibration, _ = wrapped.call_args.args[1:]
        assert np.all((grain >= 1) & (grain <= 19)) and np.all((vibration >= 0) & (vibration <= 3))
        assert np.all(np.ptp(orientation, axis=1) > 0)
        assert np.all((orientation >= 0.4) & (orientation <= 1.6))
    assert np.isfinite(y).all() and valid.all() and np.all((shifts >= 0) & (shifts <= 4))
    with patch.object(sim, "broaden", wraps=sim.broaden) as wrapped:
        y, valid, shifts = sim.views(peaks, "Co", 1, 8, 42)
        grain, orientation, vibration, _ = wrapped.call_args.args[1:]
        b_factor = 8 * np.pi**2 * vibration**2 / 3
        assert np.all((grain >= 22.36) & (grain <= 559.02)) and orientation == 1.0
        assert np.all((b_factor >= 0) & (b_factor <= 1.8))
        assert np.all((shifts >= -0.05) & (shifts <= 0.05))
        assert shifts.min() < 0 < shifts.max() and valid.all() and np.isfinite(y).all()
    y, valid, _ = sim.views(peaks[:, :0], "Cu", 1, 5, 42)
    assert not valid.any() and not y.any()
    print("Bragg peaks match pymatgen on 18 crystal/target pairs; Cu and non-Cu parameter ranges passed")


def check_resume():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        args = Namespace(output=out, views=2, block_entries=2, gpus=[1, 2, 3], seed=42)
        catalog = db_catalog(DATABASE, 4)
        common_spec = {"schema": SCHEMA, "db_sha256": "test", "views": args.views,
                       "entries": 4, "seed": args.seed, "block_entries": args.block_entries}
        spec = {**common_spec, "target": "Cu"}
        encoded = json.dumps(spec, sort_keys=True).encode()
        signature = hashlib.sha256(encoded).hexdigest()
        stage = out / "Cu/.patterns.blocks"
        stage.mkdir(parents=True)
        (stage / "spec.json").write_bytes(encoded)
        ids = catalog["entry_id"]
        pattern = np.full((2, 2, POINTS), 100, dtype=np.float32)
        valid = np.ones((2, 2), dtype=bool)
        shifts = np.zeros((2, 2), dtype=np.float32)
        write_block(stage / "block-0000000.npz", signature, ids[:2], pattern, valid, shifts)
        (stage / "block-0000002.npz").write_bytes(b"simulated truncated block")

        with patch("generate.generate_blocks") as start_workers:
            try:
                run_target(args, None, catalog, {**common_spec, "seed": 0}, "Cu")
            except ValueError as error:
                assert "Resume settings differ" in str(error)
            else:
                raise AssertionError("resume accepted mismatched settings")
            start_workers.assert_not_called()

        def fill_missing(_pool, output, digest, _target, jobs, *_):
            assert len(jobs) == 1 and jobs[0][0] == 2
            arrays = read_block(output / "block-0000000.npz", digest, ids[:2], 2)
            np.testing.assert_array_equal(arrays[0], pattern)
            write_block(output / "block-0000002.npz", digest, jobs[0][1], pattern, valid, shifts)

        with patch("generate.generate_blocks", side_effect=fill_missing) as workers:
            run_target(args, None, catalog, common_spec, "Cu")
            workers.assert_called_once()
        final = out / "Cu/patterns.h5"
        assert verify_final(final, spec, catalog) == 8 and not stage.exists()
        with patch("generate.generate_blocks") as workers:
            run_target(args, None, catalog, common_spec, "Cu")
            workers.assert_not_called()
        reader = GanjiangPatterns(final)
        np.testing.assert_array_equal(reader.pattern(reader.row_for_entry(ids[-1]), 1), pattern[0, 0])
        reader.close()
    print("Resume reused an intact block, regenerated a damaged block, rejected changed settings and verified the final HDF5")


if __name__ == "__main__":
    check_physics()
    check_resume()
