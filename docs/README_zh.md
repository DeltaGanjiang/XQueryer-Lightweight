# XQueryer Lightweight

XQueryer Lightweight 是 XQueryer 的轻量化改良版本，用于粉末 X 射线衍射
晶体结构识别。原始项目：https://github.com/Bin-Cao/XQueryer

作者：Dr. Bin Cao，https://bin-cao.github.io/

## 数据下载

`MP500.db` 文件较大，不随 Git 仓库提交。请从本仓库的 GitHub Releases 页面下载，
然后放到：

```text
data/MP500.db
```

训练和推理脚本默认读取这个路径。`.gitignore` 已经配置为忽略本地数据文件。

## 主要特点

- 从 `MP500.db` 动态模拟 XRD，而不是预先导入固定 XRD。
- 所有谱图统一插值到 10-90 度、3500 个点。
- 按结构固定划分 train/val/test = 8:1:1。
- 默认每个结构模拟 20 条 XRD。
- 模型保留 XQueryer 的 FFT、CNN、Cross-Attention、分类头框架。
- 使用紧凑元素 query 替代巨大元素序列展开。
- 训练支持 warmup-cosine、EMA、label smoothing、top-k 指标和可选弱 mixup。

## 快速训练测试

```bash
python trainer.py --epochs 1 --batch_size 2 --num_workers 0 \
  --simulations_per_entry 1 --max_train_entries 2 --max_val_entries 2 \
  --output_dir outputs/smoke
```

## 完整训练

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --output_dir outputs/lightweight
```

## 推理

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

实验 CSV 需要包含 `angle,intensity` 两列。
