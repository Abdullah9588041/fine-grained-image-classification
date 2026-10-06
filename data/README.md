# Data

## Source (real dataset)

**Oxford 102 Flowers** — Nilsback & Zisserman, "Automated Flower Classification over a
Large Number of Classes", Proc. Indian Conference on Computer Vision, Graphics and Image
Processing, Dec 2008.

- URL: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Accessed: 2026-10-06
- Contents: 8,189 images across **102 fine-grained flower categories** (40–258 images per class)
- Files: `102flowers.tgz` (images), `imagelabels.mat` (labels 1–102), `setid.mat` (official splits)

## Splits (official, from `setid.mat`)

| Split | Images |
|---|---|
| train | 1,020 (10 per class) |
| val | 1,020 (10 per class) |
| test | 6,149 |

The train/val sets are deliberately tiny per class — this is what makes fine-grained
classification hard and why transfer learning + augmentation matter.

## Reproducing

```bash
python data/download.py   # downloads + extracts into data/raw/
```

`data/raw/` is git-ignored (images are ~330 MB). The analysis scripts assume it exists.
