# GanJiang 六种靶材训练

GanJiang 使用 ASE 结构库 `ganjiang_db_v1.db`。正式训练针对 Cu、Co、Mo、Cr、Fe、Ag
分别训练一个模型。分类索引对应 ASE 的 `entry_id - 1`，正式结构库有 207,146 个类别。

## Conda 环境

在仓库根目录执行以下命令。GPU 模拟需要 Linux 和可用的 NVIDIA CUDA 环境。

```bash
conda create -n ganjiang-xqueryer python=3.11 pip
conda activate ganjiang-xqueryer
python -m pip install -r requirements-ganjiang.txt
```

## 准备训练、验证和测试谱

将数据库放到 `data/ganjiang_db_v1.db`，或在命令里指定数据库位置。以下示例用两组独立
种子生成每个结构的 20 条训练谱和 3 条留出谱。将 `spectra` 改为有足够空间的目录，
将 `--gpus 0` 改为当前机器上可用的物理 GPU 编号。

```bash
db=data/ganjiang_db_v1.db
spectra=/path/to/spectra

python simxrd/generate.py --db "$db" --output "$spectra/train" --gpus 0 --views 20 --seed 42
python simxrd/generate.py --db "$db" --output "$spectra/holdout" --gpus 0 --views 3 --seed 43
```

每个目录下都有 `<靶材>/patterns.h5`。训练读取独立的 20 条训练谱。留出文件中第 1 条
用于验证，第 2、3 条用于测试。已有文件可以直接使用。

### 模拟参数

每条谱使用 10–90° 的 3,500 点网格。HDF5 文件记录 ASE 行号、GanJiang 结构编号、
谱图有效标记和每条谱的零点偏移。

| 参数 | Cu | Co、Mo、Cr、Fe、Ag |
| --- | --- | --- |
| 晶粒尺寸 | 10–190 Å（1–19 nm），均匀采样 | 22.36–559.02 nm，对数均匀采样 |
| 热振动 | 位移 0–3 Å | 各向同性 B 因子 0–1.8 Å² |
| 2θ 零点偏移 | 0–4° | −0.05° 至 +0.05° |
| 择优取向 | 峰强扰动上限 ±0.6 | 无 |
| 微应变 | 无 | 0–1.5×10⁻³ |
| 高斯峰宽系数 U | 0.0036 deg² | 0–0.020 deg² |
| 高斯峰宽系数 W | 0.0036 deg² | 0–0.015 deg² |
| 六阶多项式背景幅度 | 2–20% | 2–20% |
| 均匀噪声幅度 | 0–18% | 0–18% |

微应变并入高斯展宽，非 Cu 靶材从 B 因子换算各向同性位移。两条 Kα 线按 2:1 加权。

运行 `python simxrd/generate.py --help` 查看参数。默认输出到 `data/simulated/`，
使用物理 GPU 0。用 `--gpus 0 1` 指定多张卡，用 `--targets Cu` 只生成 Cu。
重新运行相同命令会核对数据库和模拟配置，并复用校验通过的分块。

### 导出单条谱图

以下命令从 Cu 的 HDF5 文件取出 ASE 行号 1 的首条谱，导出 `angle,intensity` CSV。

```bash
python simxrd/patterns.py /path/to/Cu/patterns.h5 1 --view 0 > spectrum.csv
```

## 训练

以下命令以 Cu 为例，使用正式实验的 512 维余弦分类头、batch size 256 和 100 轮配置。
可以在 `--targets` 和 `--output_dir` 中替换靶材。类别数从 HDF5 结构目录读取。

```bash
CUDA_VISIBLE_DEVICES=0 python trainer.py \
  --db_path "$db" \
  --hdf5_root "$spectra/train" --hdf5_holdout_root "$spectra/holdout" \
  --targets Cu --classifier_dim 512 --batch_size 256 --epochs 100 \
  --simulations_per_entry 20 --val_simulations_per_entry 1 --test_simulations_per_entry 2 \
  --output_dir outputs/ganjiang-cu
```

对于使用单份 20 条谱 HDF5 的旧实验，只提供 `--hdf5_root` 时默认按 17/1/2 划分。
MP500 的在线模拟仍可通过 `trainer.py` 的默认数据入口运行。

## 模型和推理

六份原始最佳训练检查点和对应的轻量推理权重位于 Hugging Face。 [GanJiang 仓库](https://huggingface.co/Licht0812/GanJiang/tree/main)
使用原始 `.pth` 推理时，`inference.py` 会优先加载 EMA 权重，并检查数据库与训练所用版本一致。

```bash
python inference.py --db_path "$db" --checkpoint /path/to/Cu_best.pth \
  --inputs 'exp_data/*.csv' --topk 5
```

输入 CSV 包含 `angle,intensity` 两列。原始训练检查点还保留了模型参数、优化器状态、
保存轮次和训练参数。
