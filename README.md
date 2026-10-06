# Fine-Grained Image Classification with Explainability

Transfer learning with a pretrained **ResNet-50** on the **Oxford 102 Flowers** benchmark —
8,189 images across 102 visually similar flower species, with only 10 training images per
class — plus **Grad-CAM** explanations (implemented from scratch) showing *where* the model
looks when it classifies.

## Problem statement

Fine-grained classification is hard: the visual difference between two daisy species can be
smaller than the variation within one species (lighting, pose, growth stage). The questions:

1. How far does ImageNet transfer learning get us with just 10 images per class?
2. Which species get confused, and are the confusions botanically sensible?
3. Does the model attend to the flower or to background shortcuts? (Grad-CAM)
4. How much do augmentation and full fine-tuning matter vs. a frozen backbone? (ablations)

## Methodology

- **Data.** Oxford 102 Flowers, official splits (train 1,020 / val 1,020 / test 6,149).
  Augmentation: random resized crops, horizontal flips, color jitter.
- **Model.** ResNet-50 pretrained on ImageNet (`IMAGENET1K_V2`), classification head replaced
  with Dropout + Linear(2048 → 102). Full fine-tuning with **AdamW** (decoupled weight decay),
  **cosine LR schedule**, early stopping on validation loss. Why ResNet-50: a strong,
  well-understood baseline whose residual connections keep gradients stable during
  fine-tuning, and whose pretrained texture/shape filters transfer well to petals.
- **Evaluation.** Top-1 / top-5 accuracy, per-class precision/recall/F1, confusion-pair
  analysis on the held-out test set.
- **Explainability.** Grad-CAM from scratch (gradient-weighted class activation maps on the
  last ResNet block) for correctly classified and misclassified examples.

## Results

> All numbers below are from actual runs — see `results/metrics.json`.

| Model | Val acc | Test top-1 | Test top-5 | Macro F1 | Train time |
|---|---|---|---|---|---|
| ResNet-50, full fine-tune + augmentation (main, 12 epochs) | 93.2% | 90.4% | 98.1% | 0.900 | ~79 min |
| Ablation: no augmentation (3 epochs) | 90.3% | (val only) | — | — | ~20 min |
| Ablation: frozen backbone / linear probe (3 epochs) | 59.3% | (val only) | — | — | ~10 min |

Ablations ran 3 epochs each (CPU runtime trade-off) with validation metrics only; the main
run's best validation loss was at epoch 10 of 12. Full fine-tuning beats a frozen backbone
by ~34 points of validation accuracy — transfer features alone are not enough for
fine-grained distinctions.

**Error analysis.** Test-set macro F1 is 0.900 (6,149 test images). Hardest classes
(1-based class indices, with F1 and test support): #3 (F1 0.47, n=20), #16 (0.63, n=21),
#4 (0.65, n=36), #1 (0.66, n=20), #40 (0.72, n=47) — rare, visually similar species where
10 training images are thinnest. Most frequent confusions (true → predicted, count):
50 → 12 (16), 51 → 1 (15), 84 → 40 (13), 74 → 88 (11), 51 → 86 (11) — confusions cluster
between morphologically similar species, not random errors. Confusion matrix of the 8
hardest classes: `results/figures/confusion_worst8.png`; full 102×102 matrix saved as
`results/confusion_matrix.npy`.

**Grad-CAM.** See `results/figures/gradcam_panels.png` — the model attends to petal
structure on correct predictions; misclassifications correlate with attention drifting to
background foliage.

## Quick start

```bash
python data/download.py                                   # fetch Oxford 102 Flowers (~330 MB)
pip install -r requirements.txt                           # CPU torch; see notes for CUDA
PYTHONPATH=src python scripts/run_analysis.py             # train + evaluate + explain
```

## Project structure

```
├── data/               # download.py + README (images git-ignored)
├── src/flowers/        # data, models, train, evaluate, explain, visualization
├── scripts/            # run_analysis.py — full pipeline
├── tests/              # pytest suite (13 tests)
├── docs/               # math_notes.md — derivations
├── results/            # figures/ + metrics.json (real run outputs)
└── .github/workflows/  # CI
```

## Reproducibility

- Fixed seeds everywhere (`--seed`, default 42); pinned dependencies in `requirements.txt`.
- Hardware used for the reference run: CPU-only, 2 threads, PyTorch 2.14.1+cpu, seed 42.
  Main training (12 epochs) ≈ 79 min; full pipeline (training + 2 ablations + test
  evaluation + Grad-CAM) ≈ 121 min wall-clock.
- Checkpoints saved to `checkpoints/` (git-ignored); best model by validation loss.

## Limitations & future work

- CPU-only training limited the epoch budget; longer schedules and larger backbones
  (EfficientNet, ViT) would likely improve accuracy.
- Only 10 training images per class — few-shot/meta-learning approaches are a natural
  extension; test-time augmentation was not explored.
- Grad-CAM is coarse (7×7); Grad-CAM++ or Score-CAM could sharpen localization.

## References

- Nilsback & Zisserman (2008). Automated flower classification over a large number of classes.
- He et al. (2016). Deep residual learning for image recognition.
- Loshchilov & Hutter (2019). Decoupled weight decay regularization.
- Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks.

## License

MIT — see [LICENSE](LICENSE).
