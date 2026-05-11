"""APKLOT — segmentation parking aérien (repo GitHub + Pascal VOC / LabelMe)."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from parking_capacity.datasets_satellite.converters import (
    ImageRecord,
    dir_size_bytes,
    export_coco_segmentation_json,
    export_yolo_segmentation_dir,
    write_metadata_jsonl,
    write_unified_dataset,
)
from parking_capacity.datasets_satellite.download_utils import (
    command_available,
    download_url_streaming,
    filename_from_url,
    git_clone,
    git_lfs_pull,
    project_data_datasets_dir,
)
from parking_capacity.datasets_satellite.registry import update_dataset_status

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

APKLOT_GIT = "https://github.com/langheran/APKLOT.git"
APKLOT_ZIP_FALLBACK = "https://github.com/langheran/APKLOT/archive/refs/heads/master.zip"

# Arborescence officielle (README) : ``1. Satellite`` (Google Maps / vue plongeante),
# ``2. Camera`` (+ LabelMe). Sans Git LFS, souvent seule la partie caméra est présente.


def classify_apklot_path(path: Path) -> str:
    """Heuristique caméra vs satellite à partir du chemin (insensible à la casse)."""
    s = str(path.resolve()).lower().replace("\\", "/")
    if "satellite" in s:
        return "satellite"
    if any(
        k in s
        for k in (
            "camera",
            "office lens",
            "segmentation mobile",
            "/mobile/",
            "lens images",
        )
    ):
        return "camera"
    return "unknown"


def iter_apklot_voc_roots(root: Path) -> List[Path]:
    """Tous les répertoires Pascal VOC détectés (JPEGImages + SegmentationClass)."""
    found: List[Path] = []
    seen: set = set()

    def add(base: Path) -> None:
        b = base.resolve()
        if b not in seen:
            seen.add(b)
            found.append(b)

    if root.is_dir():
        ji = root / "JPEGImages"
        sc = root / "SegmentationClass"
        if ji.is_dir() and sc.is_dir():
            add(root)
    for base in list(root.glob("**/VOC*")) + list(root.glob("**/Pascal*")):
        if not base.is_dir():
            continue
        ji = base / "JPEGImages"
        sc = base / "SegmentationClass"
        if ji.is_dir() and sc.is_dir():
            add(base)
    for ji in root.glob("**/JPEGImages"):
        base = ji.parent
        if (base / "SegmentationClass").is_dir():
            add(base)
    return found


def find_apklot_voc_roots_with_views(root: Path) -> List[Tuple[Path, str]]:
    """Liste ``(voc_root, vue)`` avec vue = satellite | camera | unknown."""
    return [(p, classify_apklot_path(p)) for p in iter_apklot_voc_roots(root)]


def find_apklot_voc_root(root: Path) -> Optional[Path]:
    """Premier layout VOC (ordre : satellite préféré, puis premier trouvé)."""
    ranked = find_apklot_voc_roots_with_views(root)
    for pref in ("satellite", "unknown", "camera"):
        for p, v in ranked:
            if v == pref:
                return p
    return None


def _default_raw(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "raw" / "apklot"


def _default_prepared(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "prepared" / "apklot"


def download_apklot(
    dest: Optional[Path] = None,
    *,
    use_git: bool = True,
    zip_fallback: bool = False,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Télécharge APKLOT : préfère ``git clone`` + ``git lfs pull`` (données sous LFS).

    Si ``zip_fallback=True``, télécharge l’archive GitHub (souvent **sans** fichiers LFS
    utiles — uniquement pointeurs). Pour l’entraînement complet, clone + LFS requis.
    """
    dest = dest or _default_raw(project_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    if use_git and command_available("git"):
        if dest.is_dir() and (dest / ".git").is_dir():
            notes.append(f"Dépôt existant : {dest} (pas de re-clone).")
        else:
            code, out = git_clone(APKLOT_GIT, dest, branch="master", depth=1)
            if code != 0:
                return {
                    "ok": False,
                    "path": str(dest),
                    "error": out,
                    "hint": "Vérifiez git, réseau, ou clonez manuellement APKLOT.",
                }
            notes.append("Clone git terminé.")
        if command_available("git-lfs") or command_available("git"):
            code_lfs, lfs_out = git_lfs_pull(dest)
            if code_lfs != 0:
                notes.append(
                    "git lfs pull a échoué ou LFS absent — installez Git LFS "
                    "(brew install git-lfs ; git lfs install) puis : "
                    f"cd {dest} && git lfs pull"
                )
            else:
                notes.append("Git LFS : fichiers données récupérés.")
    elif zip_fallback:
        from parking_capacity.datasets_satellite.download_utils import extract_zip

        zip_path = dest.parent / filename_from_url(APKLOT_ZIP_FALLBACK)
        download_url_streaming(APKLOT_ZIP_FALLBACK, zip_path)
        extract_zip(zip_path, dest.parent)
        extracted = dest.parent / "APKLOT-master"
        if extracted.is_dir() and not dest.exists():
            extracted.rename(dest)
        notes.append("Archive ZIP : les fichiers LFS peuvent être des pointeurs ; préférer git+lfs.")
    else:
        return {
            "ok": False,
            "path": str(dest),
            "error": "git indisponible et zip_fallback=False",
            "hint": "Installez git ou passez zip_fallback=True (données partielles).",
        }

    sz = dir_size_bytes(dest)
    update_dataset_status("apklot", status="downloaded", size_bytes=sz, project_root=project_root)
    notes.append(
        "Le dépôt contient « 1. Satellite » (vue Google Maps) et « 2. Camera ». "
        "Pour l’orthophoto / satellite : vérifiez que ``git lfs pull`` a bien récupéré ``1. Satellite``."
    )
    return {"ok": True, "path": str(dest), "size_bytes": sz, "notes": notes}


def find_apklot_labelme_dirs(root: Path) -> List[Path]:
    """Répertoires contenant des paires .json + .png/.jpg (LabelMe)."""
    dirs: List[Path] = []
    for j in root.glob("**/*.json"):
        if j.name.startswith("."):
            continue
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "shapes" not in data:
            continue
        d = j.parent
        if d not in dirs:
            dirs.append(d)
    return dirs


def _mask_to_polygons(mask_u8: np.ndarray) -> List[List[Tuple[float, float]]]:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless requis pour contours APKLOT")
    # Binaire : tout pixel > 0 = parking
    _, bw = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[Tuple[float, float]]] = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = c.reshape(-1, 2).astype(np.float64)
        polys.append([(float(x), float(y)) for x, y in poly])
    return polys


def _load_voc_image_ids(voc_root: Path) -> List[str]:
    ids: List[str] = []
    sets_dir = voc_root / "ImageSets" / "Segmentation"
    for name in ("train.txt", "val.txt", "trainval.txt"):
        p = sets_dir / name
        if p.is_file():
            ids.extend([ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
    if ids:
        return sorted(set(ids))
    # tout JPEGImages
    return sorted({p.stem for p in (voc_root / "JPEGImages").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}})


def _split_ids(ids: Sequence[str], seed: int, train: float, val: float) -> Dict[str, str]:
    ids_list = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids_list)
    n = len(ids_list)
    n_train = int(n * train)
    n_val = int(n * val)
    split: Dict[str, str] = {}
    for i, sid in enumerate(ids_list):
        if i < n_train:
            split[sid] = "train"
        elif i < n_train + n_val:
            split[sid] = "val"
        else:
            split[sid] = "test"
    return split


def _make_voc_image_id(voc_root: Path, raw_root: Path, sid: str) -> str:
    try:
        rel = voc_root.relative_to(raw_root)
        prefix = str(rel).replace("/", "__").replace(" ", "_")
    except ValueError:
        prefix = voc_root.name
    return f"{prefix}__{sid}"


def _voc_allowed(view: str, mode: str) -> bool:
    m = mode.strip().lower()
    if m in ("auto", "satellite"):
        return view == "satellite"
    if m == "camera":
        return view == "camera"
    if m == "all":
        return True
    return view == "satellite"


def _labelme_allowed(jpath: Path, mode: str) -> bool:
    v = classify_apklot_path(jpath)
    return _voc_allowed(v, mode)


def prepare_apklot_dataset(
    raw_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    dataset_subset: str = "full",
    subset_max_images: int = 36,
    apklot_view: str = "auto",
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Indexe APKLOT, produit ``dataset_manifest.json``, splits, COCO et YOLO seg,
    format unifié ``parking_capacity_dataset``.

    ``apklot_view`` :
    - ``auto`` / ``satellite`` : uniquement chemins contenant « Satellite » (vue orthophoto cible).
    - ``camera`` : uniquement « Camera », Office Lens, etc.
    - ``all`` : tout le dépôt (satellite + caméra).

    ``dataset_subset`` : ``full`` ou ``small`` (sous-échantillon pour Colab / tests).
    """
    raw_root = raw_root or _default_raw(project_root)
    out_dir = out_dir or _default_prepared(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = apklot_view.strip().lower()
    if mode == "auto":
        mode = "satellite"

    records: List[ImageRecord] = []

    voc_roots = find_apklot_voc_roots_with_views(raw_root)
    voc_roots_f = [(p, v) for p, v in voc_roots if _voc_allowed(v, mode)]

    for voc, _v in voc_roots_f:
        ids = _load_voc_image_ids(voc)
        split_map = _split_ids(ids, seed, train_frac, val_frac)
        ji = voc / "JPEGImages"
        sc = voc / "SegmentationClass"
        for sid in ids:
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = ji / f"{sid}{ext}"
                if cand.is_file():
                    img_path = cand
                    break
            if img_path is None:
                continue
            mask_path = None
            for ext in (".png", ".jpg"):
                cand = sc / f"{sid}{ext}"
                if cand.is_file():
                    mask_path = cand
                    break
            if mask_path is None:
                continue
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            m = np.array(Image.open(mask_path).convert("L"))
            polys = _mask_to_polygons(m)
            iid = _make_voc_image_id(voc, raw_root, sid)
            records.append(
                ImageRecord(
                    image_id=iid,
                    rel_image=str(img_path.resolve()),
                    split=split_map.get(sid, "train"),
                    width=w,
                    height=h,
                    polygons=polys,
                    category_ids=[1],
                    source_dataset="apklot_voc",
                )
            )

    lm_dirs = find_apklot_labelme_dirs(raw_root)
    all_items: List[Tuple[str, Path]] = []
    json_by_uid: Dict[str, Path] = {}
    for d in lm_dirs:
        for j in d.glob("*.json"):
            if not _labelme_allowed(j, mode):
                continue
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "shapes" not in data:
                continue
            stem = Path(data.get("imagePath", j.stem + ".png")).stem
            uid = f"{j.parent.name}__{stem}"
            json_by_uid[uid] = j
            all_items.append((uid, j))

    if all_items:
        uids = sorted({x[0] for x in all_items})
        split_map_lm = _split_ids(uids, seed + 1, train_frac, val_frac)
        for uid in uids:
            jpath = json_by_uid.get(uid)
            if not jpath:
                continue
            data = json.loads(jpath.read_text(encoding="utf-8"))
            stem = Path(data.get("imagePath", jpath.stem + ".png")).stem
            img_name = data.get("imagePath", stem + ".png")
            img_path = jpath.parent / img_name
            if not img_path.is_file():
                continue
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            polys = []
            for sh in data.get("shapes", []):
                pts = sh.get("points") or []
                if len(pts) >= 3:
                    polys.append([(float(p[0]), float(p[1])) for p in pts])
            records.append(
                ImageRecord(
                    image_id=uid,
                    rel_image=str(img_path.resolve()),
                    split=split_map_lm.get(uid, "train"),
                    width=w,
                    height=h,
                    polygons=polys,
                    category_ids=[1],
                    source_dataset="apklot_labelme",
                )
            )

    if dataset_subset.strip().lower() == "small" and records:
        rng = random.Random(seed)
        idx = list(range(len(records)))
        rng.shuffle(idx)
        pick = idx[: min(subset_max_images, len(records))]
        records = [records[i] for i in sorted(pick)]

    views = Counter(classify_apklot_path(Path(r.rel_image)) for r in records)
    satellite_segmentation_suitable = views["satellite"] >= 1

    prepare_meta = {
        "dataset": "apklot",
        "apklot_view": mode,
        "images_by_view": dict(views),
        "satellite_segmentation_suitable": satellite_segmentation_suitable,
        "notes": (
            "Pour l’orthophoto IGN / satellite : au moins une image sous un chemin « Satellite » "
            "est requise ; sinon utilisez DOTA/xView/SpaceNet ou ``--apklot-view all`` en connaissance de cause."
        ),
    }
    (out_dir / "dataset_prepare_meta.json").write_text(
        json.dumps(prepare_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    voc_primary = find_apklot_voc_root(raw_root) if voc_roots else None

    manifest = {
        "dataset": "apklot",
        "n_images": len(records),
        "dataset_subset": dataset_subset,
        "subset_max_images": subset_max_images if dataset_subset.lower() == "small" else None,
        "raw_root": str(raw_root.resolve()),
        "voc_roots_detected": len(voc_roots),
        "voc_root_primary": str(voc_primary) if voc_primary else None,
        "apklot_view": mode,
        "images_by_view": dict(views),
        "splits": {"train_frac": train_frac, "val_frac": val_frac, "seed": seed},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not records:
        hint = (
            "Aucun échantillon pour le filtre de vue demandé. "
            "Pour la partie satellite officielle : ouvrez raw/apklot et vérifiez "
            "« 1. Satellite » puis ``git lfs pull``. Sinon ``datasets-prepare --dataset apklot --apklot-view all``."
        )
        update_dataset_status(
            "apklot",
            status="missing",
            extra={"last_prepare_error": hint, "apklot_view": mode},
            project_root=project_root,
        )
        return {
            "ok": False,
            "manifest": manifest,
            "prepare_meta": prepare_meta,
            "prepared": str(out_dir),
            "error": hint,
        }

    write_unified_dataset(records, out_dir, coco_path="coco_segmentation.json")
    for split_name in ("train", "val", "test"):
        sub = [r for r in records if r.split == split_name]
        if sub:
            export_yolo_segmentation_dir(sub, out_dir / "yolo_labels" / split_name)
            export_coco_segmentation_json(
                sub,
                [{"id": 1, "name": "parking"}],
                out_dir / f"coco_segmentation_{split_name}.json",
            )

    splits_index = [{"image_id": r.image_id, "split": r.split} for r in records]
    write_metadata_jsonl(splits_index, out_dir / "split_manifest.jsonl")

    sz = dir_size_bytes(out_dir)
    update_dataset_status(
        "apklot",
        status="prepared",
        size_bytes=sz,
        extra={"apklot_prepare_meta": prepare_meta},
        project_root=project_root,
    )
    return {
        "ok": True,
        "manifest": manifest,
        "prepare_meta": prepare_meta,
        "prepared": str(out_dir),
        "n_records": len(records),
    }
