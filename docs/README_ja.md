# XQueryer Lightweight

XQueryer Lightweight は、粉末 X 線回折による結晶構造識別のための
XQueryer 改良版です。元プロジェクト：
https://github.com/Bin-Cao/XQueryer

Author: Dr. Bin Cao, https://bin-cao.github.io/

## データのダウンロード

`MP500.db` は大きな ASE データベースなので Git には含めていません。
このリポジトリの GitHub Releases からダウンロードし、次の場所に配置してください。

```text
data/MP500.db
```

学習と推論スクリプトはこのパスをデフォルトで使用します。

## 主な特徴

- `MP500.db` から XRD を動的にシミュレーション。
- すべてのパターンを 10-90 度、3500 点に補間。
- 構造単位で train/val/test = 8:1:1 に分割。
- 各構造につきデフォルトで 20 本の XRD を生成。
- FFT、CNN、Cross-Attention、分類ヘッドという XQueryer の枠組みを維持。
- 大きな元素系列展開を、軽量な element query に置き換え。
- warmup-cosine、EMA、label smoothing、top-k 指標、弱い mixup に対応。

## クイックテスト

```bash
python trainer.py --epochs 1 --batch_size 2 --num_workers 0 \
  --simulations_per_entry 1 --max_train_entries 2 --max_val_entries 2 \
  --output_dir outputs/smoke
```

## 学習

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --output_dir outputs/lightweight
```

## 推論

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

入力 CSV には `angle,intensity` の 2 列が必要です。
