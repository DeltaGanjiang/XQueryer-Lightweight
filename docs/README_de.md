# XQueryer Lightweight

XQueryer Lightweight ist eine verbesserte, leichtere Version von XQueryer zur
Identifikation von Kristallstrukturen aus Pulver-XRD-Daten. Originalprojekt:
https://github.com/Bin-Cao/XQueryer

Autor: Dr. Bin Cao, https://bin-cao.github.io/

## Daten herunterladen

`MP500.db` ist eine große ASE-Datenbank und wird deshalb nicht im Git-Repository
gespeichert. Laden Sie die Datei von der GitHub-Releases-Seite dieses
Repositories herunter und legen Sie sie hier ab:

```text
data/MP500.db
```

Training und Inferenz verwenden diesen Pfad standardmäßig.

## Hauptmerkmale

- Dynamische XRD-Simulation aus `MP500.db`.
- Alle Muster werden auf 3500 Punkte im Bereich 10-90 Grad interpoliert.
- Strukturweiser Split train/val/test = 8:1:1.
- Standardmäßig 20 simulierte XRD-Muster pro Struktur.
- Das XQueryer-Framework aus FFT, CNN, Cross-Attention und Klassifikationskopf bleibt erhalten.
- Die große Element-zu-Sequenz-Erweiterung wird durch kompakte Element-Queries ersetzt.
- Training mit warmup-cosine, EMA, label smoothing, top-k Metriken und optionalem schwachem mixup.

## Schnelltest

```bash
python trainer.py --epochs 1 --batch_size 2 --num_workers 0 \
  --simulations_per_entry 1 --max_train_entries 2 --max_val_entries 2 \
  --output_dir outputs/smoke
```

## Training

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --output_dir outputs/lightweight
```

## Inferenz

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

CSV-Dateien müssen die Spalten `angle,intensity` enthalten.
