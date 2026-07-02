# src/model

Neural-network and data-processing code for XQueryer Lightweight.

Author: Dr. Bin Cao, https://bin-cao.github.io/

- `XQueryer.py`: XQueryer-compatible FFT, CNN, Cross-Attention, and
  classification-head model. It keeps the original framework but uses compact
  element queries, residual attention refinement, query self-attention, CNN/FFT
  feature fusion, and a cosine classifier to improve large-class retrieval.
- `dataset.py`: dynamic Pysimxrd simulation dataset, XRD-level split seed
  separation, XRD interpolation, and experimental CSV loading.
- `focal_loss.py`: legacy focal loss utility retained for experiments.
