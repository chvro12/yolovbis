"""Compare VisDrone brut vs Fine-tuned français sur les 3 sites connus.

Pour chaque site, compte :
- Nombre véhicules détectés (post-filtre taille)
- Nombre détections aberrantes (bboxes > 35 m²)
- Confiance moyenne des détections gardées
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SITES = [
    ("Neufchâtel (clinique vétérinaire)", "/tmp/audit2_neuf/chip.png", "~25 places, ~9 voitures"),
    ("Vénissieux (parking public)", "/tmp/venissieux_v3/chip.png", "~40-50 places, ~25 voitures"),
    ("Bouzonville (clinique rurale)", "/tmp/bouzonville/chip.png", "~5-10 places, 1 voiture"),
]

VISDRONE = "/Users/mac/Yolo/data/aerial_weights/models--mshamrai--yolov8s-visdrone/snapshots/996db9c3fbdf92ecdcf58411449a39232bfbcff6/best.pt"
DOTA_FINETUNED = "/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1/run1/weights/best.pt"
FINETUNED = DOTA_FINETUNED  # alias historique

M_PER_PX = 160.0 / 768  # ~0.208 m/px


def run_yolo(image: Image.Image, weights: str, conf_th: float = 0.10):
    """Inférence directe + filtrage taille véhicule plausible."""
    from ultralytics import YOLO
    m = YOLO(weights)
    arr = np.asarray(image.convert("RGB"))
    r = m.predict(arr, conf=conf_th, verbose=False, imgsz=768)[0]
    vehicle_names = {"car", "van", "truck", "bus", "vehicle"}
    plausible = 0
    aberrant = 0
    confs = []
    if r.boxes is None:
        return {"raw": 0, "plausible": 0, "aberrant": 0, "mean_conf": 0.0}
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].cpu().numpy())
        cname = m.names[cls_id].lower()
        if cname not in vehicle_names:
            continue
        box = r.boxes.xyxy[i].cpu().numpy()
        w_m = (box[2] - box[0]) * M_PER_PX
        h_m = (box[3] - box[1]) * M_PER_PX
        area = w_m * h_m
        longe = max(w_m, h_m)
        shorte = min(w_m, h_m)
        plausible_size = (
            2.0 <= longe <= 8.5
            and 1.0 <= shorte <= 4.0
            and 2.5 <= area <= 35.0
        )
        conf = float(r.boxes.conf[i].cpu().numpy())
        if plausible_size:
            plausible += 1
            confs.append(conf)
        else:
            aberrant += 1
    raw = len([i for i in range(len(r.boxes))
               if m.names[int(r.boxes.cls[i].cpu().numpy())].lower() in vehicle_names])
    return {
        "raw": raw,
        "plausible": plausible,
        "aberrant": aberrant,
        "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 3),
    }


def main():
    if not Path(FINETUNED).is_file():
        print(f"⚠ Fine-tuned weights not found yet at: {FINETUNED}")
        print("  Wait for training to complete then re-run.")
        sys.exit(1)

    print(f"\nComparing VisDrone vs Fine-tuned French on 3 sites:\n")
    print(f"{'Site':<40} | {'Model':<14} | {'raw':>4} | {'plaus':>6} | {'aberr':>6} | {'conf':>5}")
    print("-" * 95)
    for label, chip_path, truth in SITES:
        if not Path(chip_path).is_file():
            print(f"{label:<40} | NO CHIP at {chip_path}")
            continue
        img = Image.open(chip_path).convert("RGB")
        v = run_yolo(img, VISDRONE)
        f = run_yolo(img, FINETUNED)
        print(f"{label:<40} | {'VisDrone':<14} | {v['raw']:>4} | {v['plausible']:>6} | {v['aberrant']:>6} | {v['mean_conf']:>5}")
        print(f"{'  → vérité ' + truth:<40} | {'Fine-tuned FR':<14} | {f['raw']:>4} | {f['plausible']:>6} | {f['aberrant']:>6} | {f['mean_conf']:>5}")
        print()


if __name__ == "__main__":
    main()
