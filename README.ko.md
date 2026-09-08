<div align="center">

# XQueryer Lightweight

분말 XRD 기반 경량 결정 구조 식별 도구.

[![GitHub stars](https://img.shields.io/github/stars/DeltaGanjiang/XQueryer-Lightweight?style=social)](https://github.com/DeltaGanjiang/XQueryer-Lightweight/stargazers)
[![Release](https://img.shields.io/github/v/release/DeltaGanjiang/XQueryer-Lightweight?include_prereleases)](https://github.com/DeltaGanjiang/XQueryer-Lightweight/releases)
[![Documentation](https://github.com/DeltaGanjiang/XQueryer-Lightweight/actions/workflows/pages.yml/badge.svg)](https://deltaganjiang.github.io/XQueryer-Lightweight/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [Español](README.es.md)

</div>

XQueryer Lightweight는 분말 X선 회절 기반 결정 구조 식별을 위한
XQueryer의 경량 개선 버전입니다. 원본 프로젝트:
https://github.com/Bin-Cao/XQueryer

Author: Dr. Bin Cao, https://bin-cao.github.io/

## 데이터 다운로드

`MP500.db`는 큰 ASE 데이터베이스이므로 Git 저장소에 포함하지 않습니다.
이 저장소의 GitHub Releases 페이지에서 다운로드한 뒤 다음 위치에 두십시오.

```text
data/MP500.db
```

학습과 추론 스크립트는 이 경로를 기본값으로 사용합니다.

## 사전 학습 체크포인트

검증 세트 최적 가중치 `best.pth`는 이 저장소의
[GitHub Releases 페이지](https://github.com/DeltaGanjiang/XQueryer-Lightweight/releases)에서
다운로드할 수 있습니다. 기본 추론 명령을 사용하려면 다운로드한 파일을 다음 위치에 두십시오.

```text
outputs/lightweight/checkpoints/best.pth
```

다른 위치에 둘 경우 `--checkpoint`로 해당 경로를 지정하십시오.

## 주요 특징

- `MP500.db`에서 XRD를 동적으로 시뮬레이션합니다.
- 모든 패턴을 10-90도, 3500개 포인트로 보간합니다.
- XRD 단위로 train/val/test를 분할합니다. 모든 split은 100315개 구조를
  모두 포함하지만 서로 다른 시뮬레이션 파라미터를 사용합니다.
- 기본적으로 각 구조마다 train 20개, val 1개, test 2개의 XRD 패턴을 생성합니다.
- FFT, CNN, Cross-Attention, classification head라는 XQueryer 프레임워크를 유지합니다.
- 큰 element-to-sequence 확장을 compact element query로 대체합니다.
- warmup-cosine, EMA, label smoothing, top-k metric, 선택적 약한 mixup을 지원합니다.

## 빠른 테스트

```bash
python trainer.py --epochs 1 --batch_size 2 --num_workers 0 \
  --simulations_per_entry 1 --max_train_entries 2 --max_val_entries 2 \
  --output_dir outputs/smoke
```

## 학습

```bash
torchrun --nproc_per_node=4 trainer.py \
  --db_path data/MP500.db \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --simulations_per_entry 20 \
  --output_dir outputs/lightweight
```

## 추론

```bash
python inference.py \
  --checkpoint outputs/lightweight/checkpoints/best.pth \
  --inputs "exp_data/*.csv" \
  --topk 5
```

입력 CSV에는 `angle,intensity` 두 열이 필요합니다.
