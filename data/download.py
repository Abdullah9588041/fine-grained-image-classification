"""Download the Oxford 102 Flowers dataset.

Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
(Nilsback & Zisserman, 2008 — "Automated Flower Classification over a Large
Number of Classes")
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FILES = ["102flowers.tgz", "imagelabels.mat", "setid.mat"]


def download(raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = raw_dir / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {name} ...", flush=True)
        urlretrieve(BASE_URL + name, dest)
    tgz = raw_dir / "102flowers.tgz"
    jpg_dir = raw_dir / "jpg"
    if not jpg_dir.exists():
        print("extracting 102flowers.tgz ...", flush=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(raw_dir)
    print(f"done. images in {jpg_dir}: {len(list(jpg_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    download(Path(__file__).resolve().parent / "raw")
