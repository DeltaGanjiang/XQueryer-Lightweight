<div align="center">

# XQueryer Lightweight

Lightweight crystal structure identification from powder XRD.

[![GitHub stars](https://img.shields.io/github/stars/Asterbin/Asterbin-XQueryer-Lightweight?style=social)](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/stargazers)
[![Release](https://img.shields.io/github/v/release/Asterbin/Asterbin-XQueryer-Lightweight?include_prereleases)](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/releases)
[![Documentation](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/actions/workflows/pages.yml/badge.svg)](https://asterbin.github.io/Asterbin-XQueryer-Lightweight/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [Español](README.es.md)

</div>

Lightweight improved version of XQueryer for powder X-ray diffraction crystal
structure identification. The original project is available at
https://github.com/Bin-Cao/XQueryer.

Author: Dr. Bin Cao, https://bin-cao.github.io/

## What changed

- Dynamic XRD simulation from `data/MP500.db` with `Pysimxrd.generator.parser`.
- All simulated and experimental XRD patterns are aligned to 3500 points over
  10-90 degrees.
- XRD-level train/validation/test split: all splits cover all 100315
  structures, but use different simulated parameter combinations.
- 20 train XRD patterns, 1 validation XRD pattern, and 2 test XRD patterns
  per structure by default.
- XQueryer-compatible model framework with FFT filtering, CNN peak encoding,
  element-guided Cross-Attention, and a classification head. The heavy
  element-to-sequence expansion is replaced by compact element queries, giving
  about 28.0M trainable parameters while keeping the original feature path.
- Enhanced training logic: warmup-cosine learning rate, gradient clipping, EMA
  checkpoints, label smoothing, top-k metrics, and optional weak XRD mixup.
- Root-level `trainer.py` for CPU, single GPU, or multi-GPU `torchrun` training.
- Root-level `inference.py` for experimental CSV XRD inference with linear
  interpolation to the model grid.

## Important files

- `trainer.py`: train the lightweight model from `MP500.db`.
- `inference.py`: run top-k inference on experimental XRD CSV files.
- `src/model/XQueryer.py`: FFT + CNN + Cross-Attention neural network.
- `src/model/dataset.py`: dynamic simulation dataset and interpolation helpers.
- `data/MP500.db`: ASE crystal structure database. If not present locally,
  download it from the project GitHub release and place it here.
- `exp_data/*.csv`: example experimental XRD files with `angle,intensity`.
- `docs/algorithm_en.html`: English algorithm manual.
- `docs/algorithm_zh.html`: Chinese algorithm manual.

## Data download

`MP500.db` is not tracked by Git because it is a large ASE database. Download it
from the Releases page of this repository, then place it at:

```text
data/MP500.db
```

The training and inference scripts use this path by default. Keep the database
out of normal commits; `.gitignore` is configured to ignore local data files.

## Install

```bash
pip install torch ase scipy tqdm Pysimxrd
```

## Quick smoke test

```bash
python trainer.py \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --simulations_per_entry 1 \
  --max_train_entries 2 \
  --max_val_entries 2 \
  --output_dir outputs/smoke
```

This writes `checkpoint_0000.pth`, `latest.pth`, and `best.pth`.

## Full training

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --val_simulations_per_entry 1 \
  --test_simulations_per_entry 2 \
  --test_interval 10 \
  --output_dir outputs/lightweight
```

Useful architecture knobs:

- `--base_channels`: CNN width, default `64`.
- `--attn_dim`: Cross-Attention hidden dimension, default `192`.
- `--num_heads`: attention heads, default `6`.
- `--num_tokens`: pooled XRD tokens sent to attention, default `96`.
- `--num_queries`: element-conditioned structure queries, default `4`.
- `--num_attn_layers`: residual Cross-Attention refinement depth, default `2`.
- `--classifier`: `cosine` normalized classifier or `linear`, default `cosine`.

Useful training knobs:

- `--warmup_epochs`: warmup before cosine decay, default `5`.
- `--grad_clip`: gradient clipping norm, default `1.0`.
- `--ema_decay`: EMA checkpoint decay, default `0.999`.
- `--label_smoothing`: default `0.05`.
- `--mixup_alpha`: weak XRD mixup, disabled by default.

## Inference

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

Input CSV files must contain two columns:

```csv
angle,intensity
10.0,0.42
10.02,0.98
```

The inference script uses EMA weights when a checkpoint contains `ema_model`.

## Citation

```bibtex
@article{cao2025xqueryer,
  title={XQueryer: an intelligent crystal structure identifier for powder X-ray diffraction},
  author={Cao, Bin and Zheng, Zinan and Liu, Yang and Zhang, Longhan and Wong, Lawrence WY and Weng, Lu-Tao and Li, Jia and Li, Haoxiang and Zhang, Tong-Yi},
  journal={National Science Review},
  volume={12},
  number={12},
  pages={nwaf421},
  year={2025},
  publisher={Oxford University Press}
}
```
