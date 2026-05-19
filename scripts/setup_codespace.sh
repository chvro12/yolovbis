#!/usr/bin/env bash
# Installation dépendances + poids YOLO manquants (Codespaces / devcontainer).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> parking-capacity : installation Python"
python -m pip install --upgrade pip
pip install -e ".[dev,ui,train_satellite]"

echo "==> Téléchargement yolov8s.pt (Ultralytics COCO, secours véhicules)"
python - <<'PY'
from pathlib import Path

root = Path(".").resolve()
target = root / "yolov8s.pt"
if target.is_file():
    print(f"  déjà présent : {target}")
else:
    from ultralytics import YOLO

    YOLO("yolov8s.pt")
    print(f"  OK : {target}")
PY

echo "==> Téléchargement YOLO aérien VisDrone (Hugging Face)"
python - <<'PY'
from pathlib import Path

from parking_capacity.vehicle_detection import _resolve_aerial_yolo_weights

path = _resolve_aerial_yolo_weights(cache_dir=Path("data/.cache_http"))
print(f"  VisDrone : {path or 'échec (vérifier réseau / huggingface_hub)'}")
PY

echo "==> Vérification des checkpoints versionnés"
python - <<'PY'
from parking_capacity.repo_paths import (
    dota_finetuned_weight_candidates,
    finetuned_french_weights,
    parking_seg_weights,
    resolve_existing,
)

for label, path in (
    ("DOTA véhicules", resolve_existing(dota_finetuned_weight_candidates())),
    ("FR fine-tuné", finetuned_french_weights() if finetuned_french_weights().is_file() else None),
    ("YOLO parking-seg", parking_seg_weights() if parking_seg_weights().is_file() else None),
):
    status = path if path else "absent (optionnel)"
    print(f"  {label}: {status}")
PY

echo ""
echo "Prêt. Exemples :"
echo "  pytest -q"
echo '  parking-capacity diagnose-address "2 Bd Industriel, 76270 Neufchâtel-en-Bray" \'
echo "    --dota-yolo --cache-dir data/.cache_http --out /tmp/diag"
echo "  parking-capacity ui --host 0.0.0.0 --port 7860"
