"""Training entry point for XQueryer Lightweight.

Folder: project root
Purpose: single-GPU, CPU, or torchrun/DDP training with dynamic XRD simulation.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.model.XQueryer import Xmodel, count_parameters
from src.model.dataset import DEFAULT_NUM_CLASSES, SimulatedXRDDataset


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_device() -> tuple[torch.device, int, int, bool]:
    if is_dist():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
        return device, rank, local_rank, True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 0, False


def cleanup(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    result = {}
    for k in topk:
        k = min(k, logits.shape[1])
        result[k] = correct[:k].reshape(-1).float().sum().item() / labels.numel()
    return result


def topk_correct_counts(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    result = {}
    for k in topk:
        k = min(k, logits.shape[1])
        result[k] = float(correct[:k].reshape(-1).float().sum().item())
    return result


def soft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)


def mixup_batch(
    xrd: torch.Tensor,
    elements: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return xrd, elements, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(xrd.shape[0], device=xrd.device)
    xrd = lam * xrd + (1.0 - lam) * xrd[index]
    elements = torch.maximum(elements, elements[index])
    return xrd, elements, labels[index], lam


class ModelEma:
    """Exponential moving average of trainable model weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        raw = model.module if isinstance(model, DDP) else model
        self.shadow = {name: param.detach().clone() for name, param in raw.state_dict().items()}
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        raw = model.module if isinstance(model, DDP) else model
        with torch.no_grad():
            for name, value in raw.state_dict().items():
                if value.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(value)

    def apply_to(self, model: nn.Module) -> None:
        raw = model.module if isinstance(model, DDP) else model
        self.backup = {name: value.detach().clone() for name, value in raw.state_dict().items()}
        raw.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(self.backup, strict=True)
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler,
    ema: ModelEma | None,
    device: torch.device,
    rank: int,
    epoch: int,
    train: bool,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    total_samples = 0.0
    iterator = loader
    if rank == 0:
        iterator = tqdm(loader, desc=f"{'train' if train else 'eval'} epoch {epoch}", leave=False)

    for batch in iterator:
        xrd = batch["intensity"].to(device, non_blocking=True)
        labels = batch["id"].to(device, non_blocking=True)
        elements = batch["element"].to(device, non_blocking=True)
        labels_b = labels
        lam = 1.0
        if train and args.mixup_alpha > 0:
            xrd, elements, labels_b, lam = mixup_batch(xrd, elements, labels, args.mixup_alpha)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xrd, elements)
                if train and args.mixup_alpha > 0:
                    loss = lam * soft_cross_entropy(logits, labels, args.label_smoothing)
                    loss = loss + (1.0 - lam) * soft_cross_entropy(logits, labels_b, args.label_smoothing)
                else:
                    loss = criterion(logits, labels)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

        batch_size = float(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        corrects = topk_correct_counts(logits.detach(), labels)
        total_top1 += corrects[1]
        total_top5 += corrects[5]
        total_top10 += corrects[10]
        total_samples += batch_size

    totals = torch.tensor(
        [total_loss, total_top1, total_top5, total_top10, total_samples],
        dtype=torch.float64,
        device=device,
    )
    if is_dist():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_loss, total_top1, total_top5, total_top10, total_samples = totals.tolist()
    total_samples = max(total_samples, 1.0)

    return (
        total_loss / total_samples,
        total_top1 / total_samples,
        total_top5 / total_samples,
        total_top10 / total_samples,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    ema: ModelEma | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "ema_model": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XQueryer Lightweight with dynamic Pysimxrd simulation.")
    parser.add_argument("--db_path", default="data/MP500.db", help="ASE database path with 100315 structures.")
    parser.add_argument("--output_dir", default="outputs/lightweight", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--label_smoothing", default=0.05, type=float)
    parser.add_argument("--mixup_alpha", default=0.0, type=float, help="Weak XRD mixup; 0 disables it.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="0 disables EMA.")
    parser.add_argument("--simulations_per_entry", default=20, type=int)
    parser.add_argument("--val_simulations_per_entry", default=1, type=int)
    parser.add_argument("--test_simulations_per_entry", default=2, type=int)
    parser.add_argument("--test_interval", default=10, type=int, help="Evaluate test split every N epochs; 0 disables it.")
    parser.add_argument("--num_classes", default=DEFAULT_NUM_CLASSES, type=int)
    parser.add_argument("--input_length", default=3500, type=int)
    parser.add_argument("--base_channels", default=64, type=int)
    parser.add_argument("--attn_dim", default=192, type=int)
    parser.add_argument("--num_heads", default=6, type=int)
    parser.add_argument("--num_tokens", default=96, type=int)
    parser.add_argument("--num_queries", default=4, type=int)
    parser.add_argument("--num_attn_layers", default=2, type=int)
    parser.add_argument("--classifier", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_train_entries", default=None, type=int, help="Debug limit for train structures.")
    parser.add_argument("--max_val_entries", default=None, type=int, help="Debug limit for val structures.")
    parser.add_argument("--max_test_entries", default=None, type=int, help="Debug limit for test structures.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device, rank, _local_rank, distributed = setup_device()
    output_dir = Path(args.output_dir)

    train_set = SimulatedXRDDataset(
        args.db_path,
        split="train",
        simulations_per_entry=args.simulations_per_entry,
        seed=args.seed,
        num_points=args.input_length,
        max_entries=args.max_train_entries,
    )
    val_set = SimulatedXRDDataset(
        args.db_path,
        split="val",
        simulations_per_entry=args.val_simulations_per_entry,
        seed=args.seed,
        num_points=args.input_length,
        max_entries=args.max_val_entries,
    )
    test_set = SimulatedXRDDataset(
        args.db_path,
        split="test",
        simulations_per_entry=args.test_simulations_per_entry,
        seed=args.seed,
        num_points=args.input_length,
        max_entries=args.max_test_entries,
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Xmodel(
        input_length=args.input_length,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        attn_dim=args.attn_dim,
        num_heads=args.num_heads,
        num_tokens=args.num_tokens,
        num_queries=args.num_queries,
        num_attn_layers=args.num_attn_layers,
        dropout=args.dropout,
        classifier=args.classifier,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = max(len(train_loader) * max(args.epochs, 1), 1)
    warmup_steps = min(len(train_loader) * args.warmup_epochs, total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)
        print(f"Device: {device}")
        print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)} | Test samples: {len(test_set)}")
        print(f"Trainable parameters: {count_parameters(model.module if isinstance(model, DDP) else model):,}")
        save_checkpoint(output_dir / "checkpoints" / "checkpoint_0000.pth", model, optimizer, 0, args, ema)
        save_checkpoint(output_dir / "checkpoints" / "latest.pth", model, optimizer, 0, args, ema)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        if distributed:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5, train_top10 = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, ema, device, rank, epoch, True, args
        )
        if ema is not None:
            ema.apply_to(model)
        val_loss, val_top1, val_top5, val_top10 = run_epoch(
            model, val_loader, criterion, None, None, scaler, None, device, rank, epoch, False, args
        )
        test_metrics = None
        if args.test_interval > 0 and epoch % args.test_interval == 0:
            test_metrics = run_epoch(
                model, test_loader, criterion, None, None, scaler, None, device, rank, epoch, False, args
            )
        if ema is not None:
            ema.restore(model)

        if rank == 0:
            message = (
                f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6e} "
                f"train_loss={train_loss:.4f} top1={train_top1:.4f} top5={train_top5:.4f} top10={train_top10:.4f} "
                f"val_loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f} top10={val_top10:.4f}"
            )
            if test_metrics is not None:
                test_loss, test_top1, test_top5, test_top10 = test_metrics
                message += (
                    f" test_loss={test_loss:.4f} top1={test_top1:.4f} "
                    f"top5={test_top5:.4f} top10={test_top10:.4f}"
                )
            print(message)
            save_checkpoint(output_dir / "checkpoints" / f"checkpoint_{epoch:04d}.pth", model, optimizer, epoch, args, ema)
            save_checkpoint(output_dir / "checkpoints" / "latest.pth", model, optimizer, epoch, args, ema)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(output_dir / "checkpoints" / "best.pth", model, optimizer, epoch, args, ema)

    cleanup(distributed)


if __name__ == "__main__":
    main()
