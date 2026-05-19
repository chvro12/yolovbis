"""Extrait le sous-ensemble véhicules de DOTAv1 et convertit OBB → HBB single-class.

DOTAv1 fournit 15 classes en OBB. On garde :
- 9 = large vehicle
- 10 = small vehicle
On les fusionne en classe 0 = "vehicle" et on convertit le format polygone (4 points)
en bbox horizontale axis-aligned (format YOLO centre+wh normalisé).
"""

from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("/Users/mac/Yolo/datasets/DOTAv1")
DST = Path("/Users/mac/Yolo/datasets/DOTAv1_vehicles")
KEEP_CLASSES = {9, 10}  # large vehicle, small vehicle


def obb_to_hbb_yolo(coords: list[float]) -> tuple[float, float, float, float]:
    """8 coords (x1,y1,...,x4,y4) normalisées → (cx, cy, w, h) normalisé."""
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return cx, cy, w, h


def process_split(split: str) -> tuple[int, int, int]:
    """Traite train ou val. Retourne (images_with_vehicles, total_vehicles, images_skipped)."""
    img_src = SRC / "images" / split
    lbl_src = SRC / "labels" / split
    img_dst = DST / "images" / split
    lbl_dst = DST / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_imgs = 0
    n_veh = 0
    n_skip = 0
    for lbl_file in sorted(lbl_src.glob("*.txt")):
        out_lines = []
        with lbl_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls not in KEEP_CLASSES:
                    continue
                try:
                    coords = [float(x) for x in parts[1:9]]
                except ValueError:
                    continue
                cx, cy, w, h = obb_to_hbb_yolo(coords)
                # Clamp dans [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))
                # On filtre les bboxes minuscules (< 5 px à 1000px image typique)
                if w < 0.003 or h < 0.003:
                    continue
                out_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not out_lines:
            n_skip += 1
            continue

        # Trouver et copier l'image (extension .jpg, .png, .tif)
        stem = lbl_file.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = img_src / f"{stem}{ext}"
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            continue

        # Lien symbolique (économise disque) au lieu de copie
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())

        # Écrire le label converti
        (lbl_dst / f"{stem}.txt").write_text("\n".join(out_lines) + "\n")
        n_imgs += 1
        n_veh += len(out_lines)

    return n_imgs, n_veh, n_skip


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        if not (SRC / "labels" / split).is_dir():
            print(f"split {split} skip (pas de labels)")
            continue
        n_imgs, n_veh, n_skip = process_split(split)
        print(f"  {split}: {n_imgs} images avec véhicules ({n_veh} bboxes), "
              f"{n_skip} images sans véhicules ignorées")

    # YAML
    yaml_path = DST / "dataset.yaml"
    yaml_path.write_text(f"""# DOTAv1 vehicles subset (small + large vehicle → single class)
path: {DST}
train: images/train
val: images/val
names:
  0: vehicle
""")
    print(f"\n  YAML : {yaml_path}")


if __name__ == "__main__":
    main()
