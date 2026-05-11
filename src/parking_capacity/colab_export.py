"""Export / import pack Colab (sans données sensibles dans l’archive)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from parking_capacity.datasets_satellite.converters import dir_size_bytes
from parking_capacity.datasets_satellite.registry import load_registry

# Seuil : au-delà, manifest seul + instructions (bytes)
DEFAULT_MAX_DATASET_EXPORT_BYTES = 400 * 1024 * 1024

# Mot-clé construit dynamiquement pour ne pas déclencher le scan sur ce fichier.
_TAG_MAP_VENDOR = "MAP" + "ILLARY"

SECRET_PATTERNS = [
    re.compile(_TAG_MAP_VENDOR, re.I),
    re.compile(r"MLY\|", re.I),
    re.compile("API" + "_" + "KEY", re.I),
    re.compile("ARC" + "GIS", re.I),
    re.compile("SEC" + "RET", re.I),
    re.compile("Bearer" + r"\s+", re.I),
    re.compile("access_token" + r"\s*=", re.I),
    re.compile("TOKEN" + r"\s*[:=]", re.I),
    re.compile(r"\b" + "TOKEN" + r"\b", re.I),
]

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh", ".toml", ".csv", ".env", ".example", ".ipynb"}
SKIP_SCAN_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".pth", ".zip", ".bin", ".mp4", ".ico", ".pdf"}


def project_root_from_here(path: Optional[Path] = None) -> Path:
    """Racine du dépôt (contenant ``src/``)."""
    if path:
        return path.resolve()
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_short(project_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def package_version_from_pyproject(project_root: Path) -> str:
    pp = project_root / "pyproject.toml"
    if not pp.is_file():
        return "0.0.0"
    for line in pp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("version"):
            _, _, rhs = line.partition("=")
            return rhs.strip().strip('"').strip("'")
    return "0.0.0"


def discover_cli_command_names() -> List[str]:
    """Noms des sous-commandes Typer — analyse statique de ``cli.py`` (sans importer le paquet)."""
    cli_py = project_root_from_here() / "src" / "parking_capacity" / "cli.py"
    if not cli_py.is_file():
        return []
    text = cli_py.read_text(encoding="utf-8")
    explicit = re.findall(r'@app\.command\(\s*["\']([^"\']+)["\']\s*\)', text)
    names = list(explicit)
    if re.search(r"@app\.command\(\s*\)\s*\n\s*def\s+run\s*\(", text):
        names.append("run")
    return sorted(set(names))


def list_package_py_modules(project_root: Path, subdir_under_pkg: str) -> List[str]:
    d = project_root / "src" / "parking_capacity" / subdir_under_pkg
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.py") if p.name != "__init__.py")


def compute_build_info(
    project_root: Path,
    *,
    notebook_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Métadonnées pour ``build_info.json`` (export Colab)."""
    nb_path = notebook_file or (project_root / "notebooks" / "train_colab.ipynb")
    nb_sha = file_sha256(nb_path) if nb_path.is_file() else None
    reg = load_registry(project_root)
    ds_keys = sorted(reg.get("datasets", {}).keys())
    return {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(project_root),
        "package_version": package_version_from_pyproject(project_root),
        "cli_commands": discover_cli_command_names(),
        "datasets_registry": ds_keys,
        "satellite_modules": list_package_py_modules(project_root, "datasets_satellite"),
        "training_modules": list_package_py_modules(project_root, "training"),
        "notebook_train_colab_sha256": nb_sha,
        "notebook_hashed_path": str(nb_path.resolve()) if nb_path.is_file() else None,
    }


def write_build_info(out_dir: Path, project_root: Path, *, notebook_file: Optional[Path] = None) -> Path:
    info = compute_build_info(project_root, notebook_file=notebook_file)
    path = out_dir / "build_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_doctor_build(
    *,
    export_dir: Optional[Path] = None,
    build_info_file: Optional[Path] = None,
    notebook_file: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare environnement courant vs dernier export Colab."""
    try:
        import importlib.metadata

        installed_ver = importlib.metadata.version("parking-capacity")
    except Exception:
        installed_ver = None

    bi_path = build_info_file
    if bi_path is None and export_dir is not None:
        bi_path = export_dir / "build_info.json"

    exported: Dict[str, Any] = {}
    if bi_path is not None and bi_path.is_file():
        exported = json.loads(bi_path.read_text(encoding="utf-8"))

    nb_path = notebook_file
    if nb_path is None and export_dir is not None:
        nb_path = export_dir / "train_colab.ipynb"

    nb_hash: Optional[str] = None
    if nb_path is not None and nb_path.is_file():
        nb_hash = file_sha256(nb_path)

    exp_nb = exported.get("notebook_train_colab_sha256")
    exp_pkg = exported.get("package_version")

    mismatches: List[Dict[str, Any]] = []
    if installed_ver and exp_pkg and installed_ver != exp_pkg:
        mismatches.append(
            {"check": "package_version", "installed": installed_ver, "exported_build_info": exp_pkg}
        )
    if exp_nb and nb_hash and exp_nb != nb_hash:
        mismatches.append(
            {
                "check": "notebook_sha256",
                "current_notebook": nb_hash,
                "exported_build_info": exp_nb,
            }
        )

    return {
        "ok": len(mismatches) == 0,
        "installed_package_version": installed_ver,
        "exported_package_version": exp_pkg,
        "exported_git_commit": exported.get("git_commit"),
        "exported_timestamp": exported.get("export_timestamp"),
        "notebook_sha256_current": nb_hash,
        "notebook_sha256_exported": exp_nb,
        "build_info_path": str(bi_path) if bi_path else None,
        "notebook_path_checked": str(nb_path) if nb_path else None,
        "cli_commands_count_exported": len(exported.get("cli_commands") or []),
        "mismatches": mismatches,
        "message": None
        if not mismatches
        else "Écart entre l’installation actuelle et build_info.json / notebook exporté.",
    }


def copy_project_snapshot(src_root: Path, dst: Path) -> List[str]:
    """Copie fichiers utiles vers ``project_snapshot``."""
    dst.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    allow_files = {
        "pyproject.toml",
        "README.md",
        "providers.yaml.example",
        "LICENSE",
        "LICENSE.md",
    }
    allow_dirs = {"src", "docs", "notebooks", "tests"}
    for name in allow_files:
        p = src_root / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            copied.append(name)
    for dname in allow_dirs:
        p = src_root / dname
        if p.is_dir():

            def _ignore(dirpath: str, names: List[str]) -> List[str]:
                ignored = []
                for n in names:
                    if n.endswith(".egg-info") or n == "__pycache__":
                        ignored.append(n)
                return ignored

            shutil.copytree(
                p,
                dst / dname,
                dirs_exist_ok=True,
                ignore=_ignore,
            )
            copied.append(dname + "/")
    reg = src_root / "data" / "datasets" / "dataset_registry.json"
    if reg.is_file():
        (dst / "data" / "datasets").mkdir(parents=True, exist_ok=True)
        shutil.copy2(reg, dst / "data" / "datasets" / "dataset_registry.json")
        copied.append("data/datasets/dataset_registry.json")
    scripts = src_root / "scripts"
    if scripts.is_dir():
        shutil.copytree(scripts, dst / "scripts", dirs_exist_ok=True)
        copied.append("scripts/")
    return copied


def scan_text_for_secrets(
    root: Path,
    *,
    max_file_mb: float = 2.0,
) -> List[str]:
    """Scan léger des fichiers texte ; retourne messages d’avertissement."""
    warnings: List[str] = []
    max_b = int(max_file_mb * 1024 * 1024)
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in SKIP_SCAN_SUFFIXES:
            continue
        if fp.stat().st_size > max_b:
            continue
        if fp.suffix.lower() not in TEXT_SUFFIXES and not fp.name.endswith(".example"):
            continue
        if fp.name == "colab_export.py":
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for rx in SECRET_PATTERNS:
            if rx.search(txt):
                try:
                    rel_s = str(fp.relative_to(root))
                except ValueError:
                    rel_s = str(fp)
                warnings.append(f"Motif suspect ({rx.pattern}) dans {rel_s}")
                break
    return warnings


def find_latest_yolo_seg_run(project_root: Path) -> Optional[Path]:
    """Dernier ``…/weights/best.pt`` sous ``data/runs/<nom_run>/`` (y compris ``yolo_train``)."""
    runs = project_root / "data" / "runs"
    if not runs.is_dir():
        return None
    candidates: List[Tuple[float, Path]] = []
    for best in runs.rglob("weights/best.pt"):
        try:
            mtime = best.stat().st_mtime
            rel = best.relative_to(runs)
            if not rel.parts:
                continue
            run_dir = runs / rel.parts[0]
            candidates.append((mtime, run_dir.resolve()))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def copy_slim_yolo_run(run_dir: Path, dst_runs: Path) -> Dict[str, Any]:
    """Copie poids + métriques + échantillons ; pas les caches lourds."""
    name = run_dir.name
    out = dst_runs / name
    out.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for rel in (
        "train_metrics.json",
        "train_metrics.csv",
        "quickstart_manifest.json",
        "README.md",
        "README_COLAB.md",
    ):
        p = run_dir / rel
        if p.is_file():
            shutil.copy2(p, out / rel)
            copied.append(rel)
    yt = run_dir / "yolo_train"
    if yt.is_dir():
        ydst = out / "yolo_train"
        ydst.mkdir(parents=True, exist_ok=True)
        for rel in ("results.csv", "args.yaml"):
            p = yt / rel
            if p.is_file():
                shutil.copy2(p, ydst / rel)
                copied.append("yolo_train/" + rel)
        for imgpat in ("results.png", "confusion_matrix.png", "MaskPR_curve.png"):
            p = yt / imgpat
            if p.is_file():
                shutil.copy2(p, ydst / imgpat)
                copied.append("yolo_train/" + imgpat)
        wdir = yt / "weights"
        if wdir.is_dir():
            (ydst / "weights").mkdir(parents=True, exist_ok=True)
            for w in ("best.pt", "last.pt"):
                wp = wdir / w
                if wp.is_file():
                    shutil.copy2(wp, ydst / "weights" / w)
                    copied.append(f"yolo_train/weights/{w}")
    elif (run_dir / "weights").is_dir():
        # Sortie Ultralytics plate : run/weights/best.pt
        ydst = out / "yolo_train"
        (ydst / "weights").mkdir(parents=True, exist_ok=True)
        for w in ("best.pt", "last.pt"):
            wp = run_dir / "weights" / w
            if wp.is_file():
                shutil.copy2(wp, ydst / "weights" / w)
                copied.append(f"yolo_train/weights/{w}")
        for rel in ("results.csv", "args.yaml"):
            p = run_dir / rel
            if p.is_file():
                shutil.copy2(p, ydst / rel)
                copied.append("yolo_train/" + rel)
        for imgpat in ("results.png", "confusion_matrix.png", "MaskPR_curve.png"):
            p = run_dir / imgpat
            if p.is_file():
                shutil.copy2(p, ydst / imgpat)
                copied.append("yolo_train/" + imgpat)
    for sub in ("sample_masks", "sample_overlays", "prediction_examples"):
        s = run_dir / sub
        if s.is_dir():
            shutil.copytree(s, out / sub, dirs_exist_ok=True)
            copied.append(sub + "/")
    vp = run_dir / "val_predictions"
    if vp.is_dir():
        shutil.copytree(vp, out / "val_predictions", dirs_exist_ok=True)
        copied.append("val_predictions/")
    yds = run_dir / "yolo_dataset"
    if (yds / "dataset.yaml").is_file():
        shutil.copy2(yds / "dataset.yaml", out / "yolo_dataset_dataset.yaml")
        copied.append("yolo_dataset_dataset.yaml")
    return {"run": name, "files": copied}


def copy_dataset_for_colab(
    project_root: Path,
    dataset_name: str,
    dst_datasets: Path,
    *,
    max_bytes: int = DEFAULT_MAX_DATASET_EXPORT_BYTES,
) -> Dict[str, Any]:
    """Copie vers ``datasets/prepared/<nom>/`` (layout registre Colab)."""
    reg = load_registry(project_root)
    info = reg.get("datasets", {}).get(dataset_name)
    if not info:
        return {"ok": False, "error": f"Dataset inconnu : {dataset_name}"}
    prepared = Path(info["prepared_path"])
    if not prepared.is_absolute():
        prepared = project_root / prepared
    unified = prepared / "parking_capacity_dataset"
    yolo_path = prepared / "yolo_seg_dataset"

    dst_prep = dst_datasets / "prepared" / dataset_name
    dst_datasets.mkdir(parents=True, exist_ok=True)
    total = 0
    if unified.is_dir():
        total += dir_size_bytes(unified)
    if yolo_path.is_dir():
        total += dir_size_bytes(yolo_path)

    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "prepared_path": str(prepared.resolve()),
        "unified_exists": unified.is_dir(),
        "yolo_seg_exists": yolo_path.is_dir(),
        "approx_size_bytes": total,
    }

    if total > max_bytes:
        man_path = dst_datasets / f"{dataset_name}_dataset_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        readme_extra = dst_datasets / "DATASET_TOO_LARGE.txt"
        readme_extra.write_text(
            "Le jeu préparé dépasse la taille d’export automatique.\n"
            "Sur Colab : `parking-capacity datasets-download --dataset apklot` puis "
            "`parking-capacity datasets-prepare --dataset apklot`.\n",
            encoding="utf-8",
        )
        return {"ok": True, "manifest_only": True, "manifest_path": str(man_path), "manifest": manifest}

    if unified.is_dir():
        shutil.copytree(unified, dst_prep / "parking_capacity_dataset", dirs_exist_ok=True)
    if yolo_path.is_dir():
        shutil.copytree(yolo_path, dst_prep / "yolo_seg_dataset", dirs_exist_ok=True)

    return {"ok": True, "manifest_only": False, "manifest": manifest}


def rewrite_snapshot_registry_for_export(
    snapshot_dir: Path,
    dataset_name: str,
    *,
    prepared_relative: str,
) -> None:
    """Réécrit ``dataset_registry.json`` dans le snapshot avec chemins portables (Colab)."""
    reg_path = snapshot_dir / "data" / "datasets" / "dataset_registry.json"
    if not reg_path.is_file():
        return
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    ds = data.get("datasets", {}).get(dataset_name)
    if not ds:
        return
    ds["prepared_path"] = prepared_relative
    ds["raw_path"] = f"data/datasets/raw/{dataset_name}"
    reg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_configs(project_root: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ex = project_root / "providers.yaml.example"
    if ex.is_file():
        shutil.copy2(ex, dst / "providers.yaml.example")
    # pas copier providers.yaml réel (clés API)


def write_requirements_colab(dst: Path) -> None:
    txt = """# Minimal Colab (torch fourni par runtime GPU en général)
ultralytics>=8.0.196
opencv-python-headless>=4.8
shapely>=2.0
geopandas>=0.14
rasterio>=1.3
pyproj>=3.6
pillow>=10.0
matplotlib>=3.7
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
tqdm>=4.65
pyyaml>=6.0
python-dotenv>=1.0
httpx>=0.27
typer>=0.12
rich>=13.0
"""
    (dst / "requirements_colab.txt").write_text(txt, encoding="utf-8")


def write_commands_sh(dst: Path) -> None:
    sh = """#!/usr/bin/env bash
set -euo pipefail
# Équivalent du notebook train_colab.ipynb — ajuster ZIP_PATH, ROOT, OUTPUT_RUN

ZIP_PATH="${ZIP_PATH:-/content/drive/MyDrive/parking_capacity_colab.zip}"
ROOT="${ROOT:-/content/parking_colab}"
OUTPUT_RUN="${OUTPUT_RUN:-/content/drive/MyDrive/colab_runs/yolo_seg}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"

rm -rf "$ROOT"
mkdir -p "$(dirname "$ROOT")"
unzip -q -o "$ZIP_PATH" -d "$ROOT"
cd "$ROOT/project_snapshot"

pip uninstall -y parking-capacity 2>/dev/null || true
pip install -q -r ../requirements_colab.txt
pip install -q -e ".[train_satellite,vision]"

nvidia-smi || true
parking-capacity doctor-build --export-dir "$ROOT" 2>/dev/null || true

parking-capacity dataset-stats --dataset apklot || true

# Si jeu absent / trop gros dans le ZIP :
# parking-capacity datasets-download --dataset apklot
# parking-capacity datasets-prepare --dataset apklot --apklot-view satellite

parking-capacity benchmark-dataset-mosaics --out "../datasets/benchmark_mosaic.png" --datasets apklot || true

parking-capacity train-yolo-seg \\
  --dataset apklot \\
  --model yolov8m-seg.pt \\
  --epochs 50 \\
  --imgsz 640 \\
  --output-dir "$OUTPUT_RUN" \\
  --save-period "$SAVE_PERIOD"

# Reprise :
# parking-capacity train-yolo-seg --dataset apklot --resume --weights "$OUTPUT_RUN/yolo_train/weights/last.pt" \\
#   --epochs 50 --imgsz 640 --output-dir "$OUTPUT_RUN" --save-period "$SAVE_PERIOD"

echo "OK — best.pt typiquement sous $OUTPUT_RUN/**/weights/best.pt"
"""
    p = dst / "commands.sh"
    p.write_text(sh, encoding="utf-8")
    p.chmod(p.stat().st_mode | 0o111)


def write_readme_colab(dst: Path, *, zip_name: str) -> None:
    body = f"""# Entraînement parking-capacity sur Google Colab

## Workflow rapide (quasi automatique)

### A. Sur votre machine (générer le pack)

Depuis la racine du dépôt :

```bash
parking-capacity export-colab-training --out data/colab_export --dataset apklot
```

Le dossier `data/colab_export/` contient **`{zip_name}`**, **`train_colab.ipynb`**, **`README_COLAB.md`**, **`build_info.json`** (métadonnées de build), etc.

Optionnel — vérifier la cohérence après copie sur une machine ou dans Colab :

```bash
parking-capacity doctor-build --export-dir /chemin/vers/colab_export
```

### B. Sur Google Drive

Uploadez **`{zip_name}`** et (optionnel mais pratique) **`train_colab.ipynb`** à la racine d’un dossier Drive accessible.

### C. Ouvrir dans Colab

Double-cliquez sur **`train_colab.ipynb`** dans Drive puis **Ouvrir avec → Google Colab**, ou importez le fichier depuis Colab (**Fichier → Importer le notebook**).

### D. Exécution

**Exécution → Tout exécuter**. Le notebook :

1. Monte Drive et dézippe l’archive vers `/content/parking_colab/`.
2. Désinstalle l’ancien paquet `parking-capacity`, réinstalle les dépendances puis `pip install -e ".[train_satellite,vision]"`.
3. Contrôle l’environnement (Python, CUDA, torch, Ultralytics, GPU).
4. Vérifie que les options CLI attendues sont présentes (`--apklot-view`, `--force-incompatible-dataset`, `benchmark-dataset-mosaics`).
5. Prépare APKLOT en vue **satellite**, inspecte le jeu, bloque l’entraînement si `satellite_segmentation_suitable` est faux.
6. Lance YOLOv8-seg avec **`--save-period`** si vous fixez un intervalle &mdash; avec **`OUTPUT_RUN`** sur Drive, les checkpoints sont déjà sur Drive.

**Exécution → Modifier le type d’exécution → GPU** recommandé.

## Contenu du ZIP

- **`project_snapshot/`** : code + docs + tests + registre datasets (sans secrets).
- **`datasets/`** : jeu préparé si assez petit pour l’export ; sinon manifest + `DATASET_TOO_LARGE.txt`.
- **`requirements_colab.txt`** : dépendances pip (sans imposer une version de PyTorch ; Colab fournit souvent `torch`).
- **`build_info.json`** : horodatage, commit git, version paquet, listes CLI / datasets / modules.
- **`export_security_warnings.txt`** : scan heuristique anti-fuite de clés.

## Entraînement (rappel)

```bash
cd /content/parking_colab/project_snapshot
parking-capacity train-yolo-seg \\
  --dataset apklot \\
  --model yolov8m-seg.pt \\
  --epochs 50 \\
  --imgsz 640 \\
  --output-dir /content/drive/MyDrive/colab_runs/yolo_seg \\
  --save-period 5
```

Sortie typique : `<output-dir>/yolo_train/weights/best.pt`.

## Importer les poids en local

```bash
parking-capacity import-colab-model --zip ~/Downloads/colab_results.zip --out data/models/apklot_yolo
```

## Réutiliser avec `run-address`

```bash
parking-capacity run-address "38 rue …, Paris" --yolo-weights data/models/apklot_yolo/best.pt --visual-backend yolo_parking
```

---
Généré le {datetime.now(timezone.utc).isoformat()}
"""
    (dst / "README_COLAB.md").write_text(body, encoding="utf-8")


def _resolve_dataset_path(project_root: Path, prepared_field: str) -> Path:
    """Résout ``prepared_path`` du registre (absolu, relatif au dépôt ou ``..`` pour Colab)."""
    p = Path(prepared_field)
    if p.is_absolute():
        return p.resolve()
    s = prepared_field.replace("\\", "/")
    if s.startswith("../") or s.startswith("..\\"):
        return (Path.cwd() / p).resolve()
    return (project_root / p).resolve()


def dataset_stats(dataset: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Statistiques rapides pour un jeu enregistré."""
    root = project_root or project_root_from_here()
    reg = load_registry(root)
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        return {"error": f"Dataset inconnu : {dataset}"}
    prepared = _resolve_dataset_path(root, info["prepared_path"])
    unified = prepared / "parking_capacity_dataset"
    yolo_ds = prepared / "yolo_seg_dataset"
    meta_lines = 0
    meta_path = unified / "metadata.jsonl"
    if meta_path.is_file():
        meta_lines = sum(1 for _ in meta_path.open(encoding="utf-8") if _.strip())
    return {
        "dataset": dataset,
        "prepared_path": str(prepared.resolve()),
        "parking_capacity_dataset": {
            "exists": unified.is_dir(),
            "metadata_lines": meta_lines,
            "size_bytes": dir_size_bytes(unified) if unified.is_dir() else 0,
        },
        "yolo_seg_dataset": {
            "exists": yolo_ds.is_dir(),
            "size_bytes": dir_size_bytes(yolo_ds) if yolo_ds.is_dir() else 0,
        },
    }


def run_export_colab_training(
    out_dir: Path,
    *,
    dataset: str = "apklot",
    include_current_dataset: bool = True,
    include_config: bool = True,
    include_notebooks: bool = True,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_EXPORT_BYTES,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = project_root or project_root_from_here()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("export_security_warnings.txt", "parking_capacity_colab.zip", "build_info.json"):
        sp = out_dir / stale
        if sp.is_file():
            sp.unlink()
    old_nb = out_dir / "train_colab.ipynb"
    if old_nb.is_file():
        old_nb.unlink()

    snap = out_dir / "project_snapshot"
    shutil.rmtree(snap, ignore_errors=True)
    copied = copy_project_snapshot(root, snap)

    datasets_dir = out_dir / "datasets"
    shutil.rmtree(datasets_dir, ignore_errors=True)
    ds_info: Dict[str, Any] = {}
    if include_current_dataset:
        ds_info = copy_dataset_for_colab(root, dataset, datasets_dir, max_bytes=max_dataset_bytes)
        if not ds_info.get("ok"):
            return {
                "ok": False,
                "error": ds_info.get("error", "export dataset"),
                "out_dir": str(out_dir),
            }

    prepared_rel = f"data/datasets/prepared/{dataset}"
    if include_current_dataset and ds_info.get("ok") and not ds_info.get("manifest_only"):
        prepared_rel = f"../datasets/prepared/{dataset}"
    rewrite_snapshot_registry_for_export(snap, dataset, prepared_relative=prepared_rel)

    cfg_dir = out_dir / "configs"
    shutil.rmtree(cfg_dir, ignore_errors=True)
    if include_config:
        copy_configs(root, cfg_dir)

    runs_dir = out_dir / "runs"
    shutil.rmtree(runs_dir, ignore_errors=True)
    latest = find_latest_yolo_seg_run(root)
    run_copy: Optional[Dict[str, Any]] = None
    if latest:
        run_copy = copy_slim_yolo_run(latest, runs_dir)

    write_requirements_colab(out_dir)
    write_commands_sh(out_dir)
    write_readme_colab(out_dir, zip_name="parking_capacity_colab.zip")

    # Notebook : copier depuis repo ou nom statique
    nb_src = root / "notebooks" / "train_colab.ipynb"
    nb_dst = out_dir / "train_colab.ipynb"
    if include_notebooks and nb_src.is_file():
        shutil.copy2(nb_src, nb_dst)
    elif include_notebooks:
        nb_dst.write_text("{}", encoding="utf-8")  # placeholder évité si on écrit vrai notebook après

    write_build_info(out_dir, root, notebook_file=nb_dst if nb_dst.is_file() else None)

    warnings = scan_text_for_secrets(out_dir)
    warn_path = out_dir / "export_security_warnings.txt"
    if warnings:
        warn_path.write_text("\n".join(warnings[:200]), encoding="utf-8")
    else:
        warn_path.write_text("Aucun motif sensible détecté (scan heuristique).\n", encoding="utf-8")

    zip_path = out_dir / "parking_capacity_colab.zip"
    build_colab_zip_payload(out_dir, zip_path)

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "zip": str(zip_path),
        "project_snapshot": copied,
        "dataset": ds_info,
        "latest_run": str(latest) if latest else None,
        "run_copy": run_copy,
        "security_warnings": warnings[:50],
        "security_warnings_file": str(warn_path),
        "build_info": str(out_dir / "build_info.json"),
    }


def build_colab_zip_payload(out_dir: Path, zip_path: Path) -> None:
    """Construit le ZIP à partir des dossiers et fichiers racine utiles."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    include_roots = [
        "project_snapshot",
        "datasets",
        "configs",
        "runs",
        "requirements_colab.txt",
        "commands.sh",
        "README_COLAB.md",
        "train_colab.ipynb",
        "export_security_warnings.txt",
        "build_info.json",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in include_roots:
            p = out_dir / name
            if not p.exists():
                continue
            if p.is_file():
                zf.write(p, arcname=name)
            else:
                for fp in p.rglob("*"):
                    if fp.is_file():
                        arc = name + "/" + fp.relative_to(p).as_posix()
                        zf.write(fp, arcname=arc)


def _pick_best_candidate(paths: List[Path], name: str) -> Optional[Path]:
    if not paths:
        return None
    weights = [p for p in paths if "weights" in p.parts and p.name == name]
    pool = weights or paths
    return max(pool, key=lambda x: x.stat().st_mtime)


def run_import_colab_model(zip_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Importe best.pt / métriques depuis une archive Colab."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_colab_extract_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    all_files = [p for p in tmp.rglob("*") if p.is_file()]
    bests = [p for p in all_files if p.name == "best.pt"]
    lasts = [p for p in all_files if p.name == "last.pt"]
    results_csv = [p for p in all_files if p.name == "results.csv" and "yolo_train" in p.parts]
    metrics_json = [p for p in all_files if p.name == "train_metrics.json"]
    yamls = [p for p in all_files if p.name == "dataset.yaml"]

    found_best = _pick_best_candidate(bests, "best.pt")
    found_last = _pick_best_candidate(lasts, "last.pt")
    found_csv = _pick_best_candidate(results_csv, "results.csv") if results_csv else None
    found_metrics = metrics_json[0] if metrics_json else None
    found_yaml = yamls[0] if yamls else None

    copied: List[str] = []
    if found_best:
        shutil.copy2(found_best, out_dir / "best.pt")
        copied.append("best.pt")
    if found_last:
        shutil.copy2(found_last, out_dir / "last.pt")
        copied.append("last.pt")
    if found_csv:
        shutil.copy2(found_csv, out_dir / "results.csv")
        copied.append("results.csv")
    if found_metrics:
        shutil.copy2(found_metrics, out_dir / "train_metrics.json")
        copied.append("train_metrics.json")
    if found_yaml:
        shutil.copy2(found_yaml, out_dir / "dataset.yaml")
        copied.append("dataset.yaml")

    readme_run = ""
    if found_best:
        p = found_best
        if p.parent.name == "weights" and p.parent.parent.name == "yolo_train":
            run_home = p.parent.parent.parent
        else:
            run_home = p.parent
        for cand in ("README.md", "README_COLAB.md", "notes.txt"):
            rp = run_home / cand
            if rp.is_file():
                shutil.copy2(rp, out_dir / f"run_{cand.replace('.', '_')}")
                copied.append(f"run_{cand.replace('.', '_')}")
                readme_run = str(rp)
                break

    lines = ["# Import Colab", "", "Fichiers extraits :", *[f"- {c}" for c in copied]]
    if readme_run:
        lines.extend(["", f"README source : `{readme_run}`"])
    (out_dir / "README_IMPORT.md").write_text("\n".join(lines), encoding="utf-8")

    shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "out": str(out_dir.resolve()), "copied": copied}
