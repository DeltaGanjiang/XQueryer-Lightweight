<div align="center">

# XQueryer Lightweight

Identificación ligera de estructuras cristalinas a partir de XRD en polvo.

[![GitHub stars](https://img.shields.io/github/stars/Asterbin/Asterbin-XQueryer-Lightweight?style=social)](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/stargazers)
[![Release](https://img.shields.io/github/v/release/Asterbin/Asterbin-XQueryer-Lightweight?include_prereleases)](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/releases)
[![Documentation](https://github.com/Asterbin/Asterbin-XQueryer-Lightweight/actions/workflows/pages.yml/badge.svg)](https://asterbin.github.io/Asterbin-XQueryer-Lightweight/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [Español](README.es.md)

</div>

XQueryer Lightweight es una versión mejorada y más ligera de XQueryer para la
identificación de estructuras cristalinas mediante difracción de rayos X en
polvo. Proyecto original: https://github.com/Bin-Cao/XQueryer

Autor: Dr. Bin Cao, https://bin-cao.github.io/

## Descargar datos

`MP500.db` es una base de datos ASE grande y no se guarda en Git. Descárguela
desde la página de GitHub Releases de este repositorio y colóquela en:

```text
data/MP500.db
```

Los scripts de entrenamiento e inferencia usan esta ruta por defecto.

## Características principales

- Simulación dinámica de XRD desde `MP500.db`.
- Todos los patrones se interpolan a 3500 puntos entre 10 y 90 grados.
- División train/val/test a nivel de XRD: todos los splits contienen las
  100315 estructuras, pero con parámetros de simulación diferentes.
- Por defecto, 20 patrones XRD de entrenamiento, 1 de validación y 2 de prueba
  por estructura.
- Se conserva el marco de XQueryer: FFT, CNN, Cross-Attention y cabeza de clasificación.
- La gran expansión de elementos a secuencia se sustituye por element queries compactas.
- Entrenamiento con warmup-cosine, EMA, label smoothing, métricas top-k y mixup débil opcional.

## Prueba rápida

```bash
python trainer.py --epochs 1 --batch_size 2 --num_workers 0 \
  --simulations_per_entry 1 --max_train_entries 2 --max_val_entries 2 \
  --output_dir outputs/smoke
```

## Entrenamiento

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --output_dir outputs/lightweight
```

## Inferencia

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

Los archivos CSV deben contener las columnas `angle,intensity`.
