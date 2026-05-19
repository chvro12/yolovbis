"""Teste rapidement le DOTA fine-tuning en cours (utilise last.pt à chaque epoch).

Compare directement vs VisDrone sur les 3 sites tests pour décider d'arrêter ou continuer.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

VISDRONE = "/Users/mac/Yolo/data/aerial_weights/models--mshamrai--yolov8s-visdrone/snapshots/996db9c3fbdf92ecdcf58411449a39232bfbcff6/best.pt"

DOTA_LAST = Path("/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1/run2/weights/last.pt")
DOTA_BEST = Path("/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1/run2/weights/best.pt")

SITES = [
    ("Neufchâtel (~25 places, ~9 voitures)", "/tmp/audit2_neuf/chip.png"),
    ("Vénissieux (~40-50 places, ~25 voitures)", "/tmp/venissieux_v3/chip.png"),
    ("Bouzonville (~5-10 places, 1 voiture)", "/tmp/bouzonville/chip.png"),
]

M_PER_PX = 160.0 / 768


def run(weights: str, img: Image.Image, conf: float = 0.15) -> dict:
    """Détection avec filtre taille véhicule plausible."""
    from ultralytics import YOLO
    m = YOLO(weights)
    arr = np.asarray(img.convert("RGB"))
    r = m.predict(arr, conf=conf, verbose=False, imgsz=640)[0]
    veh_classes = {"car", "van", "truck", "bus", "vehicle", "motor"}
    plausible = 0
    aberrant = 0
    confs = []
    if r.boxes is None or len(r.boxes) == 0:
        return {"raw": 0, "plausible": 0, "aberrant": 0, "mean_conf": 0.0}
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].cpu().numpy())
        cname = m.names[cls_id].lower()
        if cname not in veh_classes:
            continue
        box = r.boxes.xyxy[i].cpu().numpy()
        w_m = (box[2] - box[0]) * M_PER_PX
        h_m = (box[3] - box[1]) * M_PER_PX
        area = w_m * h_m
        longe = max(w_m, h_m)
        shorte = min(w_m, h_m)
        c = float(r.boxes.conf[i].cpu().numpy())
        if (2.0 <= longe <= 8.5 and 1.0 <= shorte <= 4.0 and 2.5 <= area <= 35.0):
            plausible += 1
            confs.append(c)
        else:
            aberrant += 1
    raw = len([i for i in range(len(r.boxes))
               if m.names[int(r.boxes.cls[i].cpu().numpy())].lower() in veh_classes])
    return {"raw": raw, "plausible": plausible, "aberrant": aberrant,
            "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 3)}


def main():
    dota_path = DOTA_BEST if DOTA_BEST.is_file() else (DOTA_LAST if DOTA_LAST.is_file() else None)
    if dota_path is None:
        print(f"⚠ DOTA weights not yet available at:\n  {DOTA_BEST}\n  or {DOTA_LAST}")
        print("Wait for at least epoch 1 to complete.")
        return
    print(f"DOTA weights: {dota_path.name} ({dota_path.stat().st_size/1024/1024:.1f} MB)")
    print()
    print(f"{'Site':<45} | {'Model':<14} | raw | plaus | aberr | conf")
    print("-" * 100)
    for label, chip_path in SITES:
        if not Path(chip_path).is_file():
            print(f"{label:<45} | NO CHIP")
            continue
        img = Image.open(chip_path).convert("RGB")
        v = run(VISDRONE, img)
        d = run(str(dota_path), img)
        print(f"{label:<45} | {'VisDrone':<14} | {v['raw']:>3} | {v['plausible']:>5} | {v['aberrant']:>5} | {v['mean_conf']}")
        print(f"{'':<45} | {'DOTA-finetune':<14} | {d['raw']:>3} | {d['plausible']:>5} | {d['aberrant']:>5} | {d['mean_conf']}")
        print()


if __name__ == "__main__":
    main()
