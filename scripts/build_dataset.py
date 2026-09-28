"""Reorganiza el dataset de huevos a formato YOLO con 2 clases (bueno/malo).

Original : 6 clases A, AA, B, C, D, E (ids 0-5) con recortes + etiquetas YOLO.
Nuevo    : 0 = bueno (AA, A, B) | 1 = malo (C, D, E)
Split    : 80% train / 20% val (deterministico).

Uso:
    python build_dataset.py \
        --src "ruta/a/Self-Collected Dataset/Yolo Dataset" \
        --dst "ruta/destino/dataset" \
        --val 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

BUENO = {0, 1, 2}   # A, AA, B
MALO = {3, 4, 5}    # C, D, E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"Self-Collected Dataset/Yolo Dataset", type=Path)
    parser.add_argument("--dst", default=r"dataset", type=Path)
    parser.add_argument("--val", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    src_images = args.src / "images"
    src_labels = args.src / "labels"
    dst = args.dst

    if not src_images.exists() or not src_labels.exists():
        raise SystemExit(f"No se encontraron carpetas images/labels en {args.src}")

    for sub in ("train", "val"):
        (dst / "images" / sub).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / sub).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    imgs = sorted(src_images.glob("*.jpg"))
    stats = {"bueno": 0, "malo": 0}
    for img in imgs:
        lbl = src_labels / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = lbl.read_text().strip().splitlines()
        if not lines:
            continue
        orig_cls = int(lines[0].split()[0])
        new_cls = 0 if orig_cls in BUENO else 1
        stats["bueno" if new_cls == 0 else "malo"] += 1
        split = "val" if random.random() < args.val else "train"
        shutil.copy2(img, dst / "images" / split / img.name)
        with open(dst / "labels" / split / (img.stem + ".txt"), "w") as f:
            for line in lines:
                parts = line.split()
                if parts and orig_cls == int(parts[0]):
                    parts[0] = str(new_cls)
                f.write(" ".join(parts) + "\n")

    for split in ("train", "val"):
        ni = len(list((dst / "images" / split).glob("*.jpg")))
        nl = len(list((dst / "labels" / split).glob("*.txt")))
        print(f"{split}: {ni} imagenes, {nl} labels")

    print("Distribucion:", stats)

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: bueno\n"
        "  1: malo\n"
    )
    print("data.yaml creado en", dst / "data.yaml")


if __name__ == "__main__":
    main()