"""Inference entry point for XQueryer Lightweight.

Folder: project root
Purpose: load a trained checkpoint, interpolate experimental XRD CSV files,
and report top-k matching MP entries.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from ase.db import connect
from torch.utils.data import DataLoader

from src.model.XQueryer import Xmodel
from src.model.dataset import DEFAULT_NUM_CLASSES, ExperimentalXRDDataset


def load_model(checkpoint_path: str, device: torch.device, args: argparse.Namespace) -> Xmodel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    model = Xmodel(
        input_length=int(saved_args.get("input_length", args.input_length)),
        num_classes=int(saved_args.get("num_classes", args.num_classes)),
        base_channels=int(saved_args.get("base_channels", args.base_channels)),
        attn_dim=int(saved_args.get("attn_dim", args.attn_dim)),
        num_heads=int(saved_args.get("num_heads", args.num_heads)),
        num_tokens=int(saved_args.get("num_tokens", args.num_tokens)),
        num_queries=int(saved_args.get("num_queries", args.num_queries)),
        num_attn_layers=int(saved_args.get("num_attn_layers", args.num_attn_layers)),
        dropout=float(saved_args.get("dropout", 0.0)),
        classifier=str(saved_args.get("classifier", args.classifier)),
    )
    state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def expand_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for value in inputs:
        matches = sorted(glob.glob(value))
        paths.extend(matches if matches else [value])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XQueryer Lightweight inference on CSV XRD files.")
    parser.add_argument("--checkpoint", default="outputs/lightweight/checkpoints/best.pth")
    parser.add_argument("--db_path", default="data/MP500.db")
    parser.add_argument("--inputs", nargs="+", default=["exp_data/*.csv"])
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--num_classes", default=DEFAULT_NUM_CLASSES, type=int)
    parser.add_argument("--input_length", default=3500, type=int)
    parser.add_argument("--base_channels", default=64, type=int)
    parser.add_argument("--attn_dim", default=192, type=int)
    parser.add_argument("--num_heads", default=6, type=int)
    parser.add_argument("--num_tokens", default=96, type=int)
    parser.add_argument("--num_queries", default=4, type=int)
    parser.add_argument("--num_attn_layers", default=2, type=int)
    parser.add_argument("--classifier", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError("No input XRD CSV files found.")

    model = load_model(args.checkpoint, device, args)
    dataset = ExperimentalXRDDataset(paths, num_points=args.input_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    database = connect(args.db_path)

    with torch.no_grad():
        for batch in loader:
            xrd = batch["intensity"].to(device)
            element = batch["element"].to(device)
            logits = model(xrd, element)
            probs = torch.softmax(logits, dim=1)
            values, indices = probs.topk(k=min(args.topk, probs.shape[1]), dim=1)
            for path, score_row, class_row in zip(batch["path"], values.cpu(), indices.cpu()):
                print(f"\n{path}")
                for score, class_id in zip(score_row.tolist(), class_row.tolist()):
                    entry_id = int(class_id) + 1
                    row = database.get(entry_id)
                    mpid = row.key_value_pairs.get("mpid", "unknown")
                    print(f"  class={class_id} entry_id={entry_id} mpid={mpid} score={score:.6f}")


if __name__ == "__main__":
    main()
