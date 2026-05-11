"""Mosaïque d’aperçus pour comparer visuellement plusieurs jeux (orthophoto vs autres)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir
from parking_capacity.datasets_satellite.registry import load_registry


def _repo_root(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root).parent.parent


def _resolve_prepared_images(project_root: Optional[Path], dataset: str) -> Optional[Path]:
    reg = load_registry(project_root)
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        return None
    p = Path(info["prepared_path"])
    if not p.is_absolute():
        p = _repo_root(project_root) / p
    uni = p / "parking_capacity_dataset" / "images"
    if uni.is_dir():
        return uni
    return None


def _resolve_raw_glob(project_root: Optional[Path], dataset: str) -> List[Path]:
    reg = load_registry(project_root)
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        return []
    p = Path(info["raw_path"])
    if not p.is_absolute():
        p = _repo_root(project_root) / p
    if not p.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    out: List[Path] = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            out.append(fp)
            if len(out) >= 120:
                break
    return out


def _thumb(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (32, 32, 32))
    ox = (size[0] - im.width) // 2
    oy = (size[1] - im.height) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def build_dataset_benchmark_mosaic(
    datasets: Sequence[str],
    out_path: Path,
    *,
    project_root: Optional[Path] = None,
    thumb_size: Tuple[int, int] = (200, 200),
    samples_per_dataset: int = 4,
) -> Dict[str, Any]:
    """
    Grille : une ligne par jeu, ``samples_per_dataset`` vignettes.
    Priorité aux images préparées ; sinon échantillon brut.
    """
    rows: List[List[Image.Image]] = []
    report: Dict[str, Any] = {"datasets": {}, "out": None}

    for ds in datasets:
        ds = ds.strip().lower()
        imgs_dir = _resolve_prepared_images(project_root, ds)
        paths: List[Path] = []
        source = "prepared"
        if imgs_dir:
            paths = sorted([p for p in imgs_dir.iterdir() if p.is_file()])[:samples_per_dataset]
        if not paths:
            source = "raw"
            paths = _resolve_raw_glob(project_root, ds)[:samples_per_dataset]

        thumbs: List[Image.Image] = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    thumbs.append(_thumb(im, thumb_size))
            except OSError:
                continue

        while len(thumbs) < samples_per_dataset:
            thumbs.append(Image.new("RGB", thumb_size, (48, 48, 48)))

        rows.append(thumbs[:samples_per_dataset])
        report["datasets"][ds] = {"n_thumbs": len(paths), "source": source}

    if not rows:
        raise ValueError("Aucune image trouvée pour les jeux demandés.")

    ncols = samples_per_dataset
    nrows = len(rows)
    tw, th = thumb_size
    gap = 8
    title_h = 28
    W = gap + ncols * (tw + gap)
    H = gap + nrows * (th + title_h + gap)
    mosaic = Image.new("RGB", (W, H), (245, 245, 245))

    try:
        from PIL import ImageDraw, ImageFont

        font = ImageFont.load_default()
        draw = ImageDraw.Draw(mosaic)
    except ImportError:
        draw = None
        font = None

    for ri, row in enumerate(rows):
        name = datasets[ri].strip().lower()
        y0 = gap + ri * (th + title_h + gap)
        if draw is not None:
            draw.text((gap, y0), name, fill=(20, 20, 20), font=font)
        for ci, thumb in enumerate(row):
            x = gap + ci * (tw + gap)
            mosaic.paste(thumb, (x, y0 + title_h))

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path, format="PNG", optimize=True)
    report["out"] = str(out_path)
    report["size_px"] = {"width": W, "height": H}
    return report
