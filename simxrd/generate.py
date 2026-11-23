"""Generate one resumable Ganjiang HDF5 file per X-ray target."""

import argparse
import fcntl
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import torch
from ase.db import connect

from patterns import file_sha256
from physics import GRID, LINE_WEIGHTS, POINTS, RADIATION, Simulator, scattering_coefficients


HERE = Path(__file__).resolve().parent
DATABASE = HERE.parent / "data/ganjiang_db_v1.db"
OUTPUT = HERE.parent / "data/simulated"
SCHEMA = "ganjiang-pxrd-1"
TARGETS = tuple(RADIATION)
SIMULATOR = DB = None


def db_catalog(path, limit):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id, unique_id, json_extract(key_value_pairs, '$.label') "
                          "FROM systems ORDER BY id LIMIT ?", (limit or -1,)).fetchall()
        if not rows or any(not r[2] or not r[2].startswith("gj-") for r in rows):
            raise ValueError("Expected a nonempty ASE database with Ganjiang labels")
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        labels = [r[2] for r in rows]
        if len(set(labels)) != len(ids):
            raise ValueError("Ganjiang labels must be unique")
        species = np.array(db.execute("SELECT id, Z FROM species WHERE id <= ?", (int(ids[-1]),)).fetchall())
    atomic_numbers = np.unique(species[:, 1])
    scattering_coefficients(atomic_numbers)
    elements = np.zeros((len(ids), int(atomic_numbers[-1])), dtype=np.uint8)
    elements[np.searchsorted(ids, species[:, 0]), species[:, 1] - 1] = 1
    return {"entry_id": ids, "label": np.asarray(labels, dtype="S"),
            "unique_id": np.asarray([r[1] for r in rows], dtype="S"), "elements": elements}


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, write):
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def block_path(stage, start):
    return stage / f"block-{start:07d}.npz"


def write_block(path, signature, ids, values, valid, shifts):
    write_atomic(path, lambda handle: np.savez(handle, signature=signature, entry_id=ids,
                                               intensity=values, valid=valid, zero_shift_deg=shifts))


def read_block(path, signature, ids, views):
    with np.load(path, allow_pickle=False) as block:
        if block["signature"].item() != signature or not np.array_equal(block["entry_id"], ids):
            raise ValueError("block provenance or ASE IDs do not match")
        patterns, valid, shift = block["intensity"], block["valid"], block["zero_shift_deg"]
    if (patterns.shape != (len(ids), views, POINTS) or patterns.dtype != np.float32
            or valid.shape != (len(ids), views) or valid.dtype != np.bool_
            or shift.shape != valid.shape or not np.isfinite(shift).all()
            or not np.isfinite(patterns).all() or np.any(patterns < 0)
            or not np.allclose(patterns.max(axis=-1), 100 * valid, atol=1e-4, rtol=0)):
        raise ValueError("invalid block spectra, flags or dimensions")
    return patterns, valid, shift


def init_worker(path, devices):
    global SIMULATOR, DB
    rank, physical = devices.get()
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    DB = connect(path, use_lock_file=False)
    DB.__enter__()
    SIMULATOR = Simulator(f"cuda:{rank}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("GPU worker PID %s physical=%s CUDA=%s", os.getpid(), physical, rank)


def simulate_block(stage, signature, target, start, ids, views, seed):
    patterns = np.empty((len(ids), views, POINTS), dtype=np.float32)
    valid = np.empty((len(ids), views), dtype=bool)
    shifts = np.empty((len(ids), views), dtype=np.float32)
    for i, entry_id in enumerate(ids):
        try:
            atoms = DB.get_atoms(id=int(entry_id))
            patterns[i], valid[i], shifts[i] = SIMULATOR.simulate(atoms, target, int(entry_id), views, seed)
        except Exception as error:
            raise RuntimeError(f"{target} ASE entry {entry_id} in block {start}: {error}") from error
    write_block(block_path(stage, start), signature, ids, patterns, valid, shifts)
    return len(ids), int(valid.sum())


def generate_blocks(pool, stage, signature, target, blocks, views, seed, workers, total):
    jobs = iter(blocks)
    pending = set()
    done = total - sum(len(ids) for _, ids in blocks)
    last_report = 0.0
    while True:
        while len(pending) < 2 * workers:
            job = next(jobs, None)
            if job is None:
                break
            start, ids = job
            pending.add(pool.submit(simulate_block, stage, signature, target, start, ids, views, seed))
        if not pending:
            return
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for task in finished:
            nrows, valid = task.result()
            done += nrows
            if valid < nrows * views:
                logging.info("%s block has %d views with no positive diffraction", target, nrows * views - valid)
        if time.monotonic() - last_report >= 30 or done == total:
            logging.info("%s simulated %d/%d crystals (%.1f%%)", target, done, total, 100 * done / total)
            last_report = time.monotonic()


def verify_final(path, spec, catalog):
    with h5py.File(path, "r") as data:
        expected = (len(catalog["entry_id"]), spec["views"], POINTS)
        if (data.attrs.get("spec") != json.dumps(spec, sort_keys=True) or not data.attrs.get("complete")
                or data["intensity"].shape != expected or data["valid"].shape != expected[:2]
                or not np.array_equal(data["two_theta_deg"][:], GRID)
                or not np.array_equal(data["entry_id"][:], catalog["entry_id"])):
            raise ValueError(f"Existing {path} does not match this run")
        return int(data.attrs["valid_spectra"])


def pack_target(stage, final, spec, catalog, signature, blocks):
    temporary = final.with_name(final.name + ".partial")
    views = spec["views"]
    count = len(catalog["entry_id"])
    valid_spectra = 0
    with h5py.File(temporary, "w", libver="v110") as data:
        data.attrs.update(schema=SCHEMA, target=spec["target"], spec=json.dumps(spec, sort_keys=True))
        data.attrs["wavelength_angstrom"] = RADIATION[spec["target"]][:2]
        data.attrs["line_weights"] = LINE_WEIGHTS
        data.attrs["db_sha256"] = spec["db_sha256"]
        for name, values in catalog.items():
            data.create_dataset(name, data=values)
        data.create_dataset("two_theta_deg", data=GRID)
        spectra = data.create_dataset("intensity", (count, views, POINTS), dtype="f4",
                                      chunks=(1, min(views, 5), POINTS), fletcher32=True)
        validity = data.create_dataset("valid", (count, views), dtype="?", chunks=True, fletcher32=True)
        offsets = data.create_dataset("zero_shift_deg", (count, views), dtype="f4")
        for start, ids in blocks:
            y, valid, shifts = read_block(block_path(stage, start), signature, ids, views)
            rows = slice(start, start + len(ids))
            spectra[rows] = y
            validity[rows] = valid
            offsets[rows] = shifts
            valid_spectra += int(valid.sum())
        data.attrs["valid_spectra"] = valid_spectra
        data.attrs["complete"] = True
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    verify_final(temporary, spec, catalog)
    # Link the finished file into place without replacing an existing file.
    os.link(temporary, final)
    sync_directory(final.parent)
    temporary.unlink()
    return valid_spectra


def run_target(args, pool, catalog, common_spec, target):
    directory = args.output / target
    directory.mkdir(exist_ok=True)
    final = directory / "patterns.h5"
    stage = directory / ".patterns.blocks"
    spec = {**common_spec, "target": target}
    encoded = json.dumps(spec, sort_keys=True).encode()
    signature = hashlib.sha256(encoded).hexdigest()
    ids = catalog["entry_id"]
    blocks = [(i, ids[i:i + args.block_entries]) for i in range(0, len(ids), args.block_entries)]
    manifest = stage / "spec.json"
    if final.exists():
        valid = verify_final(final, spec, catalog)
        logging.info("%s verified existing %s: %d/%d valid spectra", target, final, valid, len(ids) * args.views)
    else:
        stage.mkdir(exist_ok=True)
        if not manifest.exists():
            if any(stage.glob("block-*.npz")):
                raise ValueError(f"Found blocks without a run spec in {stage}")
            write_atomic(manifest, lambda handle: handle.write(encoded))
        elif manifest.read_bytes() != encoded:
            raise ValueError(f"Resume settings differ from {manifest}")
        missing = []
        for start, block_ids in blocks:
            path = block_path(stage, start)
            if path.exists():
                try:
                    read_block(path, signature, block_ids, args.views)
                    continue
                except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile) as error:
                    logging.warning("%s will regenerate %s: %s", target, path.name, error)
            missing.append((start, block_ids))
        logging.info("%s start: %d/%d blocks pending, %d views per crystal", target,
                     len(missing), len(blocks), args.views)
        generate_blocks(pool, stage, signature, target, missing, args.views, args.seed, len(args.gpus), len(ids))
        logging.info("%s assembling and checking HDF5", target)
        valid = pack_target(stage, final, spec, catalog, signature, blocks)
        logging.info("%s complete: %s, %d/%d valid spectra", target, final, valid, len(ids) * args.views)
    if manifest.is_file() and manifest.read_bytes() == encoded:
        for start, _ in blocks:
            block_path(stage, start).unlink(missing_ok=True)
        for leftover in stage.glob("block-*.npz.*.tmp"):
            leftover.unlink()
        manifest.unlink()
        if not any(stage.iterdir()):
            stage.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--views", type=int, default=20, help="simulations per ASE entry and target (default: 20)")
    parser.add_argument("--block-entries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="physical GPU indices from nvidia-smi")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--max-entries", type=int, help="limit structures for a test output directory")
    args = parser.parse_args()
    if (min(args.views, args.block_entries) < 1 or args.seed < 0 or min(args.gpus) < 0
            or len(set(args.gpus)) != len(args.gpus) or len(set(args.targets)) != len(args.targets)
            or (args.max_entries is not None and args.max_entries < 1)):
        parser.error("positive sizes, a nonnegative seed, and distinct GPU indices and targets are required")
    args.db = args.db.resolve(strict=True)
    args.output = args.output.resolve()
    if args.max_entries is not None and args.output == OUTPUT:
        parser.error("use a separate --output for a limited run")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    gpu_map = {}
    command = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    for line in subprocess.check_output(command, text=True).splitlines():
        index, uuid = (value.strip() for value in line.split(","))
        gpu_map[int(index)] = uuid
    if not set(args.gpus).issubset(gpu_map):
        parser.error(f"available physical GPU indices: {sorted(gpu_map)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_map[index] for index in args.gpus)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    with (args.output / ".simxrd.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = db_catalog(args.db, args.max_entries)
        spec = {"schema": SCHEMA, "db_sha256": file_sha256(args.db), "views": args.views,
                "entries": len(catalog["entry_id"]), "seed": args.seed, "points": POINTS,
                "block_entries": args.block_entries, "physics_sha256": file_sha256(HERE / "physics.py"),
                "versions": {name: version(name) for name in ("pymatgen", "ase", "torch", "numpy")}}
        logging.info("DB %s SHA256=%s, %d crystals; physical GPUs %s; target order %s",
                     args.db, spec["db_sha256"], spec["entries"], args.gpus, ",".join(args.targets))
        context = mp.get_context("spawn")
        devices = context.Queue()
        for rank, physical in enumerate(args.gpus):
            devices.put((rank, physical))
        pool = ProcessPoolExecutor(max_workers=len(args.gpus), initializer=init_worker,
                                   initargs=(str(args.db), devices), mp_context=context)
        try:
            for target in args.targets:
                run_target(args, pool, catalog, spec, target)
        finally:
            pool.shutdown(cancel_futures=True)
        logging.info("All requested targets completed")


if __name__ == "__main__":
    main()
