"""Interface en ligne de commande."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

from parking_capacity.pipeline import process_address, row_to_json_serializable

app = typer.Typer(no_args_is_help=True, help="Capacité de stationnement par adresse (France).")


def _detect_address_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred:
        if preferred not in df.columns:
            raise typer.BadParameter(f"Colonne absente : {preferred}")
        return preferred
    for c in ("adresse", "address", "Adresse", "Address"):
        if c in df.columns:
            return c
    raise typer.BadParameter(
        "Aucune colonne 'adresse' ou 'address'. Utilisez --address-column."
    )


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", exists=True, readable=True, help="CSV d'entrée"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Fichier de sortie"),
    fmt: str = typer.Option("csv", "--format", "-f", help="csv ou jsonl"),
    address_column: Optional[str] = typer.Option(
        None, "--address-column", help="Nom de colonne adresse (sinon détection auto)"
    ),
    radius_m: int = typer.Option(50, "--radius-m", help="Rayon Overpass et analyse autour du point (m)"),
    buffer_m: float = typer.Option(40.0, "--buffer-m", help="Buffer autour du point (m)"),
    min_intersection_m2: float = typer.Option(
        25.0, "--min-intersection-m2", help="Seuil intersection parking / parcelle"
    ),
    overpass_delay: float = typer.Option(
        1.0, "--overpass-delay", help="Pause entre requêtes Overpass (s)"
    ),
    no_vision: bool = typer.Option(False, "--no-vision", help="Désactiver orthophoto + SegFormer"),
    vision_compare: bool = typer.Option(
        False,
        "--vision-compare",
        help="Lancer aussi la vision si OSM a déjà une capacité (comparaison)",
    ),
    vision_device: Optional[str] = typer.Option(
        None, "--vision-device", help="cpu ou cuda (défaut : auto)"
    ),
    chip_half_side_m: float = typer.Option(
        55.0, "--chip-half-side-m", help="Demi-côté de la puce orthophoto (m)"
    ),
    chip_pixels: int = typer.Option(512, "--chip-pixels", help="Taille image WMS (px)"),
    wms_base: Optional[str] = typer.Option(None, "--wms-base", help="URL de base WMS Géoplateforme"),
    wms_layer: Optional[str] = typer.Option(None, "--wms-layer", help="Nom de couche WMS"),
    m2_per_space: float = typer.Option(
        26.0, "--m2-per-space", help="Heuristique m² par place (vision)"
    ),
    ml_checkpoint: Optional[Path] = typer.Option(
        None,
        "--ml-checkpoint",
        exists=True,
        readable=True,
        help="model.pt entraîné (train-model) : prédit ml_estimated_capacity sur la même puce WMS",
    ),
    ml_mode: str = typer.Option(
        "fallback",
        "--ml-mode",
        help="aux=colonne ML seulement ; fallback=OSM puis vision puis ML ; before_vision=OSM puis ML puis vision",
    ),
    ml_device: Optional[str] = typer.Option(None, "--ml-device", help="cpu ou cuda pour le modèle ML"),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Répertoire cache Overpass/WMS (réduit la charge sur les serveurs)",
    ),
    aerial_first: bool = typer.Option(
        True,
        "--aerial-first/--no-aerial-first",
        help="Sans capacity OSM : prioriser orthophoto (vision/ML) avant surface/heuristiques",
    ),
    source_priority: str = typer.Option(
        "hybrid",
        "--source-priority",
        help="hybrid | aerial | osm : ordre des sources pour la capacité principale",
    ),
    refresh_imagery: bool = typer.Option(
        False,
        "--refresh-imagery",
        help="Forcer le retéléchargement WMS (ignore le cache puces)",
    ),
    force_ml: bool = typer.Option(False, "--force-ml", help="Forcer l’inférence ML si model_meta déconseille le modèle"),
    visual_backend: str = typer.Option(
        "auto",
        "--visual-backend",
        help="none | auto | geometry_only | segformer_generic | yolo_parking | groundingdino_sam | future_custom",
    ),
    visual_model_specialized: bool = typer.Option(
        False,
        "--visual-model-specialized/--no-visual-model-specialized",
        help="SegFormer traité comme modèle orthophoto parking spécialisé (comptage places depuis masque).",
    ),
    yolo_weights: Optional[Path] = typer.Option(
        None,
        "--yolo-weights",
        exists=True,
        readable=True,
        help="Poids YOLO (ultralytics) si visual-backend=yolo_parking",
    ),
    providers_yaml: Optional[Path] = typer.Option(
        None,
        "--providers-yaml",
        help="Fichier providers.yaml pour sources GIS (IGN WFS, Overpass, …)",
    ),
    include_raw: bool = typer.Option(False, "--include-raw", help="Colonne raw_debug (json)"),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reprend un run interrompu en relisant le ledger .progress.jsonl à côté de --output",
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows",
        help="Limite le nombre d'adresses traitées (utile pour tester sur un petit échantillon)",
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        min=1,
        help="Nombre de threads parallèles. Le rate-limiter Overpass borne automatiquement le débit ;"
        " mettre 1 pour rester en séquentiel (utile pour debug).",
    ),
    geocode_batch: bool = typer.Option(
        True,
        "--geocode-batch/--no-geocode-batch",
        help="Pré-géocode toutes les adresses via BAN /search/csv avant la boucle (1 appel pour 500 adresses).",
    ),
) -> None:
    """Traite chaque ligne du CSV et écrit les résultats.

    Bulk friendly :
      - le client HTTP est partagé sur toutes les adresses (connexions BAN/APICarto/Overpass/SIRENE)
      - un ledger JSONL est écrit en append à côté de --output : reprise après interruption
      - si --cache-dir n'est pas fourni, on bascule sur ``data/.cache_http`` pour éviter de retaper les API.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import httpx
    from tqdm import tqdm

    from parking_capacity.geocode import batch_geocode_csv, prime_geocode_cache

    df = pd.read_csv(input_path)
    col = _detect_address_column(df, address_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = output_path.with_suffix(output_path.suffix + ".progress.jsonl")

    # Cache disque par défaut en bulk : évite de retaper BAN/Overpass/WMS sur des reruns.
    effective_cache_dir = cache_dir
    if effective_cache_dir is None:
        effective_cache_dir = Path("data/.cache_http")
        typer.echo(
            f"[bulk] --cache-dir non fourni → {effective_cache_dir} (réutilise les réponses HTTP en cache disque).",
            err=True,
        )
    effective_cache_dir.mkdir(parents=True, exist_ok=True)

    # Reprise : on relit les indices déjà traités dans le ledger.
    done_indices: set[int] = set()
    if resume and ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = rec.get("source_row_index")
                if isinstance(idx, int):
                    done_indices.add(idx)
        if done_indices:
            typer.echo(f"[bulk] Reprise : {len(done_indices)} adresses déjà dans {ledger_path.name}.", err=True)
    else:
        # Sans reprise on remet à zéro le ledger pour ne pas mélanger d'anciens runs.
        if ledger_path.exists():
            ledger_path.unlink()

    rows_out: List[Dict[str, Any]] = []
    addresses = df[col].astype(str).tolist()

    # On présélectionne les (index, adresse) à traiter pour avoir une barre de progression cohérente.
    todo: list[tuple[int, str]] = []
    for idx, val in enumerate(addresses):
        addr = val.strip()
        if not addr or idx in done_indices:
            continue
        todo.append((idx, addr))
        if max_rows is not None and len(todo) >= max_rows:
            break

    # Pré-géocodage en masse via BAN /search/csv (1 POST = jusqu'à 500 adresses).
    # Why: évite N round-trips HTTP au début du pipeline et donne un cache mémoire partagé entre workers.
    if geocode_batch and todo:
        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as ban_client:
                prewarm = batch_geocode_csv((addr for _, addr in todo), client=ban_client)
            if prewarm:
                prime_geocode_cache(prewarm)
                typer.echo(
                    f"[bulk] Pré-géocodage BAN : {len(prewarm)}/{len(todo)} adresses résolues en batch.",
                    err=True,
                )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[bulk] Pré-géocodage BAN ignoré ({type(e).__name__}: {e}).", err=True)

    # httpx est thread-safe au niveau d'un Client (selon les docs) ; chaque worker partage le pool.
    ledger_lock = threading.Lock()
    pbar = tqdm(total=len(todo), desc="Adresses", unit="addr")

    def _worker(client: httpx.Client, idx: int, addr: str) -> Dict[str, Any]:
        try:
            r = process_address(
                addr,
                client=client,
                overpass_delay_s=overpass_delay,
                search_radius_m=radius_m,
                point_buffer_m=buffer_m,
                min_intersection_m2=min_intersection_m2,
                use_vision=not no_vision,
                vision_device=vision_device,
                chip_half_side_m=chip_half_side_m,
                chip_pixels=chip_pixels,
                wms_base=wms_base,
                wms_layer=wms_layer,
                m2_per_space=m2_per_space,
                vision_compare=vision_compare,
                include_raw=include_raw,
                ml_checkpoint=ml_checkpoint,
                ml_mode=ml_mode,
                ml_device=ml_device,
                cache_dir=effective_cache_dir,
                aerial_first=aerial_first,
                source_priority=source_priority,
                refresh_imagery=refresh_imagery,
                force_ml=force_ml,
                visual_backend=visual_backend,
                visual_model_specialized_for_parking=visual_model_specialized,
                yolo_weights=yolo_weights,
                providers_yaml=providers_yaml,
            )
            base = row_to_json_serializable(r)
        except Exception as e:  # noqa: BLE001
            base = {"input_address": addr, "error": f"{type(e).__name__}: {e}"}
        base["source_row_index"] = idx
        return base

    with httpx.Client(timeout=120.0, follow_redirects=True) as client, ledger_path.open("a", encoding="utf-8") as ledger:
        if max_workers == 1:
            for idx, addr in todo:
                rec = _worker(client, idx, addr)
                ledger.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                ledger.flush()
                rows_out.append(rec)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_worker, client, idx, addr): idx for idx, addr in todo}
                for fut in as_completed(futures):
                    rec = fut.result()
                    with ledger_lock:
                        ledger.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        ledger.flush()
                    rows_out.append(rec)
                    pbar.update(1)
    pbar.close()

    # Sortie finale : on relit le ledger complet (= runs précédents + nouveau) pour produire le fichier
    # demandé. Garantit que --resume produit le même CSV/JSONL que sans interruption.
    all_records: List[Dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    all_records.sort(key=lambda d: d.get("source_row_index", 0))

    if fmt == "csv":
        pd.DataFrame(all_records).to_csv(output_path, index=False)
    elif fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    else:
        raise typer.BadParameter("--format doit être csv ou jsonl")

    typer.echo(
        f"Écrit : {output_path} ({len(all_records)} lignes ; "
        f"+{len(rows_out)} ce run ; ledger : {ledger_path.name})"
    )


@app.command("run-address")
def run_address_cmd(
    address: str = typer.Argument(..., help="Adresse (ex. « 38 rue du Moulin à Vent, Paris »)"),
    radius_m: int = typer.Option(50, "--radius-m"),
    buffer_m: float = typer.Option(40.0, "--buffer-m"),
    no_vision: bool = typer.Option(False, "--no-vision"),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Cache Overpass + orthophoto WMS sur disque",
    ),
    refresh_imagery: bool = typer.Option(False, "--refresh-imagery"),
    source_priority: str = typer.Option("hybrid", "--source-priority", help="hybrid | aerial | osm"),
    aerial_first: bool = typer.Option(True, "--aerial-first/--no-aerial-first"),
    ml_checkpoint: Optional[Path] = typer.Option(
        None,
        "--ml-checkpoint",
        exists=True,
        readable=True,
    ),
    ml_mode: str = typer.Option("fallback", "--ml-mode"),
    ml_device: Optional[str] = typer.Option(None, "--ml-device"),
    fmt: str = typer.Option("pretty", "--format", "-f", help="pretty | json | json-pretty"),
    force_ml: bool = typer.Option(False, "--force-ml"),
    visual_backend: str = typer.Option(
        "auto",
        "--visual-backend",
        help="none | auto | geometry_only | segformer_generic | yolo_parking | …",
    ),
    visual_model_specialized: bool = typer.Option(
        False,
        "--visual-model-specialized/--no-visual-model-specialized",
        help="SegFormer spécialisé parking (comptage depuis masque autorisé).",
    ),
    yolo_weights: Optional[Path] = typer.Option(
        None,
        "--yolo-weights",
        exists=True,
        readable=True,
    ),
    providers_yaml: Optional[Path] = typer.Option(
        None,
        "--providers-yaml",
        help="Fichier providers.yaml (sinon env PARKING_PROVIDERS_YAML ou ./providers.yaml)",
    ),
) -> None:
    """Analyse une seule adresse et affiche le résultat sur stdout."""
    from parking_capacity.human_output import format_run_address_pretty

    r = process_address(
        address,
        search_radius_m=radius_m,
        point_buffer_m=buffer_m,
        use_vision=not no_vision,
        ml_checkpoint=ml_checkpoint,
        ml_mode=ml_mode,
        ml_device=ml_device,
        cache_dir=cache_dir,
        refresh_imagery=refresh_imagery,
        source_priority=source_priority,
        aerial_first=aerial_first,
        force_ml=force_ml,
        visual_backend=visual_backend,
        visual_model_specialized_for_parking=visual_model_specialized,
        yolo_weights=yolo_weights,
        providers_yaml=providers_yaml,
    )
    d = row_to_json_serializable(r)
    if fmt == "json-pretty":
        typer.echo(json.dumps(d, ensure_ascii=False, indent=2, default=str))
    elif fmt == "json":
        typer.echo(json.dumps(d, ensure_ascii=False, default=str))
    else:
        typer.echo(format_run_address_pretty(r))


@app.command("harvest-real-dataset")
def harvest_real_dataset_cmd(
    out_dir: Path = typer.Option(..., "--out", "-o", help="Répertoire : manifest.csv + images/"),
    bbox: str = typer.Option(..., "--bbox", help="min_lon,min_lat,max_lon,max_lat"),
    country: str = typer.Option("FR", "--country"),
    half_side_m: float = typer.Option(55.0, "--half-side-m"),
    chip_pixels: int = typer.Option(512, "--chip-pixels"),
    delay_s: float = typer.Option(0.75, "--delay"),
    max_features: int = typer.Option(5000, "--max-features"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
) -> None:
    """Moissonne OSM (capacity) dans une bbox + puces orthophoto IGN."""
    from parking_capacity.harvest_real_dataset import harvest_real_dataset

    path = harvest_real_dataset(
        out_dir,
        bbox=bbox,
        country=country,
        half_side_m=half_side_m,
        chip_pixels=chip_pixels,
        delay_s=delay_s,
        max_features=max_features,
        cache_dir=cache_dir,
    )
    typer.echo(f"Manifest : {path}")


def _default_ui_host() -> str:
    if os.environ.get("CODESPACES") or os.environ.get("CODESPACE_NAME"):
        return "0.0.0.0"
    return "127.0.0.1"


@app.command("ui")
def ui_command(
    host: str = typer.Option(_default_ui_host(), "--host", help="Adresse d’écoute"),
    port: int = typer.Option(7860, "--port", help="Port HTTP"),
    share: bool = typer.Option(
        False,
        "--share",
        help="Lien public Gradio (les données quittent votre machine ; à éviter pour des données sensibles)",
    ),
) -> None:
    """Ouvre une page web pour saisir une adresse et lancer l’analyse."""
    try:
        from parking_capacity.ui import launch_ui
    except ImportError as e:
        typer.echo(
            "Gradio n’est pas installé. Exécutez : pip install -e \".[ui]\""
        )
        raise typer.Exit(1) from e
    typer.echo(f"Interface : http://{host}:{port}/")
    launch_ui(host=host, port=port, share=share)


@app.command("catalog")
def catalog_cmd(
    output: Path = typer.Option(
        Path("data/catalog_stationnement.csv"),
        "--output",
        "-o",
        help="Fichier de sortie (CSV ou JSONL)",
    ),
    fmt: str = typer.Option("csv", "--format", "-f", help="csv ou jsonl"),
    pan_keywords: str = typer.Option(
        "",
        "--pan-keywords",
        help="Mots-clés PAN séparés par des virgules (défaut : liste intégrée)",
    ),
    datagouv_queries: str = typer.Option(
        "",
        "--datagouv-queries",
        help="Requêtes data.gouv séparées par des virgules (défaut : liste intégrée)",
    ),
    max_datagouv_pages: int = typer.Option(
        30,
        "--max-datagouv-pages",
        help="Pages max par requête data.gouv (page_size=100)",
    ),
    no_pan: bool = typer.Option(False, "--no-pan", help="Ne pas inclure le PAN"),
    no_datagouv: bool = typer.Option(False, "--no-datagouv", help="Ne pas inclure data.gouv.fr"),
) -> None:
    """Construit un catalogue de ressources (fichiers) liées au stationnement / parking."""
    from parking_capacity.data_sources.catalog import (
        build_merged_catalog,
        default_datagouv_queries,
        default_pan_keywords,
        write_catalog,
    )

    pkw = [x.strip() for x in pan_keywords.split(",") if x.strip()]
    if not pkw:
        pkw = default_pan_keywords()
    dq = [x.strip() for x in datagouv_queries.split(",") if x.strip()]
    if not dq:
        dq = default_datagouv_queries()

    rows = build_merged_catalog(
        pan_keywords=pkw,
        datagouv_queries=dq,
        include_pan=not no_pan,
        include_datagouv=not no_datagouv,
        datagouv_max_pages=max_datagouv_pages,
    )
    write_catalog(rows, output, fmt=fmt)
    typer.echo(f"Écrit {len(rows)} ressources → {output}")


@app.command("fetch-resource")
def fetch_resource_cmd(
    url: str = typer.Argument(..., help="URL directe de la ressource"),
    output: Path = typer.Option(..., "--output", "-o", help="Fichier de destination"),
    max_mb: int = typer.Option(500, "--max-mb", help="Taille max (Mo)"),
) -> None:
    """Télécharge une ressource HTTP (ex. lien du catalogue) vers un fichier local."""
    from parking_capacity.data_sources.download import download_url_to_file

    n = download_url_to_file(url, output, max_bytes=max_mb * 1024 * 1024)
    typer.echo(f"Téléchargé {n} octets → {output}")


@app.command("harvest-labels")
def harvest_labels_cmd(
    catalog_path: Path = typer.Option(
        ...,
        "--catalog",
        "-c",
        exists=True,
        readable=True,
        help="CSV produit par `parking-capacity catalog`",
    ),
    output_path: Path = typer.Option(
        Path("data/harvested_labels.csv"),
        "--output",
        "-o",
        help="CSV des lignes extraites (capacité + coordonnées si détectées)",
    ),
    max_files: int = typer.Option(200, "--max-files", help="Nombre max de ressources à télécharger"),
    max_mb_per_file: int = typer.Option(40, "--max-mb-per-file", help="Plafond par fichier (Mo)"),
    delay_s: float = typer.Option(0.75, "--delay", help="Pause entre téléchargements (s)"),
) -> None:
    """Télécharge des ressources du catalogue et extrait capacité / positions (heuristique)."""
    from parking_capacity.data_sources.harvest_labels import harvest_from_catalog

    n = harvest_from_catalog(
        catalog_path,
        output_path,
        max_files=max_files,
        max_mb_per_file=max_mb_per_file,
        delay_s=delay_s,
    )
    typer.echo(f"Écrit {n} lignes → {output_path}")


@app.command("build-chips")
def build_chips_cmd(
    labels_csv: Path = typer.Option(
        ...,
        "--input",
        "-i",
        "--manifest",
        exists=True,
        readable=True,
        help="CSV (lon/lat/capacity) ou sortie harvest-real-dataset",
    ),
    output_dir: Path = typer.Option(
        Path("data/chip_dataset"),
        "--output-dir",
        "-d",
        "--out",
        help="Répertoire : images/*.png + manifest.csv + manifest.jsonl",
    ),
    lon_column: Optional[str] = typer.Option(None, "--lon-column"),
    lat_column: Optional[str] = typer.Option(None, "--lat-column"),
    capacity_column: Optional[str] = typer.Option(None, "--capacity-column"),
    max_rows: int = typer.Option(2000, "--max-rows", help="Nombre max de puces"),
    delay_s: float = typer.Option(0.6, "--delay", help="Pause entre requêtes WMS (s)"),
    half_side_m: float = typer.Option(55.0, "--half-side-m"),
    chip_pixels: int = typer.Option(512, "--chip-pixels"),
    wms_base: Optional[str] = typer.Option(None, "--wms-base"),
    wms_layer: Optional[str] = typer.Option(None, "--wms-layer"),
    no_require_capacity: bool = typer.Option(
        False,
        "--no-require-capacity",
        help="Inclure les lignes sans capacité numérique (label null)",
    ),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache WMS / HTTP"),
    refresh_imagery: bool = typer.Option(False, "--refresh-imagery"),
) -> None:
    """Télécharge des puces orthophoto BD ORTHO centrées sur chaque point (pour ML)."""
    from parking_capacity.chip_dataset import build_chip_dataset

    path = build_chip_dataset(
        labels_csv,
        output_dir,
        lon_column=lon_column,
        lat_column=lat_column,
        capacity_column=capacity_column,
        max_rows=max_rows,
        delay_s=delay_s,
        half_side_m=half_side_m,
        chip_pixels=chip_pixels,
        wms_base=wms_base,
        wms_layer=wms_layer,
        require_capacity=not no_require_capacity,
        cache_dir=cache_dir,
        refresh_imagery=refresh_imagery,
    )
    typer.echo(f"Manifest : {path}")


@app.command("train-model")
def train_model_cmd(
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-d",
        help="Répertoire : model.pt, summary.json, metrics_history.jsonl",
    ),
    synthetic_n: int = typer.Option(
        0,
        "--synthetic-n",
        help="DÉMO/CI : génère N images couleur unie fictives (pas d’orthophoto). "
        "Pour du réel : 0 et fournir --chip-dir (ex. sortie de build-chips).",
    ),
    chip_dir: Optional[Path] = typer.Option(
        None,
        "--chip-dir",
        help="Jeu réel : manifest.csv + images/ (obligatoire si --synthetic-n=0)",
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Manifest CSV (défaut: <chip-dir>/manifest.csv)",
    ),
    architecture: str = typer.Option("tiny", "--architecture", "--arch", help="tiny | resnet18 | resnet50 | efficientnet_b0"),
    epochs: int = typer.Option(15, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    seed: int = typer.Option(42, "--seed"),
    no_pretrained: bool = typer.Option(False, "--no-pretrained"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu ou cuda"),
    img_size: int = typer.Option(128, "--img-size", help="Taille d'entrée du modèle (redimensionnement)"),
    half_side_m: Optional[float] = typer.Option(
        None,
        "--half-side-m",
        help="Demi-côté WMS aligné avec build-chips (défaut : 1re ligne du manifest ou 55)",
    ),
    chip_px: Optional[int] = typer.Option(
        None,
        "--chip-pixels-train",
        help="Résolution WMS enregistrée dans le checkpoint (défaut : manifest ou 512)",
    ),
    loss: str = typer.Option("mse", "--loss", help="mse | huber"),
    target_transform: str = typer.Option("none", "--target-transform", help="none | log1p"),
    split_mode: str = typer.Option("random", "--split", help="random | geo (split géographique grossier)"),
    no_augment: bool = typer.Option(False, "--no-augment", help="Désactiver flips / jitter léger"),
) -> None:
    """Entraîne une régression capacité sur des puces (données synthétiques ou réelles)."""
    from parking_capacity.ml.train import run_training

    if synthetic_n <= 0 and chip_dir is None:
        raise typer.BadParameter("Indiquez --synthetic-n > 0 ou --chip-dir vers un jeu manifest+images.")
    cd = chip_dir if chip_dir is not None else (output_dir / "chips")
    summary = run_training(
        chip_dir=cd,
        manifest_csv=manifest,
        output_dir=output_dir,
        synthetic_n=synthetic_n,
        architecture=architecture,
        pretrained=not no_pretrained,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device_str=device,
        img_size=img_size,
        half_side_m=half_side_m,
        chip_pixels=chip_px,
        loss=loss,
        target_transform=target_transform,
        split_mode=split_mode,
        augment=not no_augment,
    )
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("eval-model")
def eval_model_cmd(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        exists=True,
        readable=True,
        help="Fichier model.pt",
    ),
    chip_dir: Path = typer.Option(
        ...,
        "--chip-dir",
        "-d",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Même répertoire racine que lors de l'entraînement (images + manifest)",
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Manifest (défaut : lu depuis le checkpoint ou chip-dir/manifest.csv)",
    ),
    batch_size: int = typer.Option(32, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device"),
    metrics_json: Optional[Path] = typer.Option(
        None,
        "--metrics-json",
        help="Écrit les métriques détaillées dans ce fichier JSON",
    ),
) -> None:
    """Évalue un checkpoint sur tout le manifest."""
    from parking_capacity.ml.train import run_eval_from_checkpoint

    metrics = run_eval_from_checkpoint(
        checkpoint,
        chip_dir,
        manifest,
        batch_size=batch_size,
        device_str=device,
    )
    typer.echo(json.dumps(metrics, indent=2, default=float))
    if metrics_json is not None:
        metrics_json.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")


@app.command("diagnose-address")
def diagnose_address_cmd(
    address: str = typer.Argument(..., help="Adresse à analyser"),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire : chip.png, result.json, sources.json, …"),
    radius_m: int = typer.Option(50, "--radius-m"),
    buffer_m: float = typer.Option(40.0, "--buffer-m"),
    chip_half_side_m: float = typer.Option(55.0, "--chip-half-side-m"),
    chip_pixels: int = typer.Option(512, "--chip-pixels"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    refresh_imagery: bool = typer.Option(False, "--refresh-imagery"),
    source_priority: str = typer.Option("hybrid", "--source-priority", help="hybrid | aerial | osm"),
    no_vision: bool = typer.Option(False, "--no-vision"),
    mock: bool = typer.Option(False, "--mock", help="HTTP simulé (CI / sans réseau)"),
    ml_checkpoint: Optional[Path] = typer.Option(
        None, "--ml-checkpoint", exists=True, readable=True
    ),
    ml_mode: str = typer.Option("fallback", "--ml-mode"),
    force_ml: bool = typer.Option(False, "--force-ml"),
    visual_backend: str = typer.Option("auto", "--visual-backend"),
    visual_model_specialized: bool = typer.Option(
        False,
        "--visual-model-specialized/--no-visual-model-specialized",
    ),
    yolo_weights: Optional[Path] = typer.Option(
        None, "--yolo-weights", exists=True, readable=True
    ),
    providers_yaml: Optional[Path] = typer.Option(
        None,
        "--providers-yaml",
        help="Fichier providers.yaml (GIS : IGN WFS, Overpass transport, …)",
    ),
    vehicle_yolo_weights: Optional[Path] = typer.Option(
        None,
        "--vehicle-yolo-weights",
        exists=True, readable=True,
        help="Poids YOLOv8 véhicules aériens (CARPK/DOTA fine-tuné).",
    ),
    auto_yolo: bool = typer.Option(
        False,
        "--auto-yolo",
        help="Télécharge automatiquement yolov8s.pt (COCO) pour détecter véhicules via SAHI.",
    ),
    aerial_yolo: bool = typer.Option(
        False,
        "--aerial-yolo",
        help="Télécharge automatiquement YOLOv8 VisDrone (mshamrai/yolov8s-visdrone) "
             "depuis HuggingFace Hub — modèle aérien spécialisé, recommandé sur orthophoto.",
    ),
    finetuned_yolo: bool = typer.Option(
        False,
        "--finetuned-yolo",
        help="Utilise le modèle YOLO fine-tuné sur chips BD ORTHO françaises "
             "(self-pseudo-labeling). Fallback sur VisDrone si fichier absent.",
    ),
    dota_yolo: bool = typer.Option(
        False,
        "--dota-yolo",
        help="Utilise le modèle YOLO fine-tuné sur DOTAv1 vehicles "
             "(51k bboxes humaines, Wuhan University). Fallback VisDrone si absent.",
    ),
    slot_yolo_weights: Optional[Path] = typer.Option(
        None,
        "--slot-yolo-weights",
        exists=True, readable=True,
        help="Poids YOLO places de parking (fine-tuné aérien parking_space_empty/filled).",
    ),
    roboflow_api_key: Optional[str] = typer.Option(
        None, "--roboflow-api-key",
        help="Clé API Roboflow Universe pour modèle slot detection hébergé.",
    ),
    roboflow_model_id: Optional[str] = typer.Option(
        None, "--roboflow-model-id",
        help="Identifiant modèle Roboflow (ex. 'workspace/project/3').",
    ),
) -> None:
    """Exporte orthophoto + JSON + carte de debug pour vérifier visuellement l'analyse."""
    from parking_capacity.diagnose import run_diagnose_address

    run_diagnose_address(
        address,
        out,
        radius_m=radius_m,
        buffer_m=buffer_m,
        chip_half_side_m=chip_half_side_m,
        chip_pixels=chip_pixels,
        cache_dir=cache_dir,
        refresh_imagery=refresh_imagery,
        source_priority=source_priority,
        no_vision=no_vision,
        mock=mock,
        ml_checkpoint=ml_checkpoint,
        ml_mode=ml_mode,
        force_ml=force_ml,
        visual_backend=visual_backend,
        visual_model_specialized_for_parking=visual_model_specialized,
        yolo_weights=yolo_weights,
        providers_yaml=providers_yaml,
        vehicle_yolo_weights=vehicle_yolo_weights,
        auto_download_vehicle_yolo=auto_yolo,
        auto_download_aerial_yolo=aerial_yolo,
        use_finetuned_french_yolo=finetuned_yolo,
        use_dota_finetuned_yolo=dota_yolo,
        slot_yolo_weights=slot_yolo_weights,
        roboflow_api_key=roboflow_api_key,
        roboflow_model_id=roboflow_model_id,
    )
    typer.echo(f"Artefacts : {out.resolve()}")


@app.command("check-gis-providers")
def check_gis_providers_cmd(
    lat: float = typer.Option(..., "--lat", help="Latitude WGS84"),
    lon: float = typer.Option(..., "--lon", help="Longitude WGS84"),
    radius_m: int = typer.Option(80, "--radius-m", help="Rayon Overpass transport (m)"),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire : providers_report.md, providers_raw.json, debug_gis_layers.png"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    providers_yaml: Optional[Path] = typer.Option(None, "--providers-yaml"),
) -> None:
    """Teste IGN WFS, Overpass transport, Microsoft path, Mapillary ; écrit un rapport."""
    from parking_capacity.gis_providers_check import run_gis_providers_check
    from parking_capacity.providers_config import load_gis_providers_config

    cfg = load_gis_providers_config(yaml_path=providers_yaml)
    run_gis_providers_check(lat, lon, radius_m=radius_m, out_dir=out, cfg=cfg, cache_dir=cache_dir)
    typer.echo(f"Rapport GIS : {out.resolve()}")


@app.command("make-training-run")
def make_training_run_cmd(
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire : harvest/, chips/, model.pt, real_run_report.md"),
    bbox: Optional[str] = typer.Option(None, "--bbox", help="min_lon,min_lat,max_lon,max_lat"),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="BBox prédéfinie : paris_small, lyon_small, nantes_small, rennes_small, bordeaux_small",
    ),
    max_samples: int = typer.Option(500, "--max-samples"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    mock: bool = typer.Option(False, "--mock", help="Sans réseau : jeu synthétique + tiny"),
    epochs: int = typer.Option(8, "--epochs"),
) -> None:
    """Enchaîne harvest → build-chips → train (resnet18) → eval → rapport markdown."""
    if not mock and bbox is None and preset is None:
        raise typer.BadParameter("Indiquez --bbox ou --preset (ou --mock).")
    from parking_capacity.make_training_run import make_training_run

    report = make_training_run(
        out,
        bbox=bbox,
        preset=preset,
        max_samples=max_samples,
        cache_dir=cache_dir,
        mock=mock,
        epochs=epochs,
    )
    typer.echo(f"Rapport : {report}")


@app.command("benchmark-addresses")
def benchmark_addresses_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        readable=True,
        help="CSV : address, radius_m optionnel, expected_capacity optionnel, notes optionnel",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire résultats + sous-dossiers par adresse"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    refresh_imagery: bool = typer.Option(False, "--refresh-imagery"),
    source_priority: str = typer.Option("hybrid", "--source-priority"),
    mock: bool = typer.Option(False, "--mock"),
    overpass_delay: float = typer.Option(1.0, "--overpass-delay"),
    force_ml: bool = typer.Option(False, "--force-ml"),
    visual_backend: str = typer.Option("auto", "--visual-backend"),
    visual_model_specialized: bool = typer.Option(
        False,
        "--visual-model-specialized/--no-visual-model-specialized",
    ),
    yolo_weights: Optional[Path] = typer.Option(
        None, "--yolo-weights", exists=True, readable=True
    ),
) -> None:
    """Benchmark terrain : métriques si expected_capacity, sinon diagnostic seul."""
    from parking_capacity.benchmark_addresses import run_benchmark_addresses

    run_benchmark_addresses(
        input_path,
        out,
        cache_dir=cache_dir,
        refresh_imagery=refresh_imagery,
        source_priority=source_priority,
        mock=mock,
        overpass_delay_s=overpass_delay,
        force_ml=force_ml,
        visual_backend=visual_backend,
        visual_model_specialized_for_parking=visual_model_specialized,
        yolo_weights=yolo_weights,
    )
    typer.echo(f"Benchmark écrit sous {out.resolve()}")


@app.command("export-vision-dataset")
def export_vision_dataset_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Répertoire résultats benchmark-addresses (sous-dossiers avec chip.png)",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Répertoire : images/, overlays/, metadata.jsonl, coco_minimal.json",
    ),
) -> None:
    """Exporte puces + COCO minimal pour entraînement YOLO / Detectron."""
    from parking_capacity.vision.export_dataset import export_vision_dataset

    p = export_vision_dataset(input_path, out)
    typer.echo(f"Dataset vision : {out.resolve()} — COCO : {p}")


@app.command("benchmark-vision")
def benchmark_vision_cmd(
    address: str = typer.Argument(..., help="Adresse à comparer selon plusieurs backends vision"),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire : vision_benchmark.json, vision_benchmark.md"),
    radius_m: int = typer.Option(50, "--radius-m"),
    ml_checkpoint: Optional[Path] = typer.Option(
        None,
        "--ml-checkpoint",
        exists=True,
        readable=True,
        help="Optionnel : ajoute un mode régression ML si le fichier existe",
    ),
    yolo_weights: Optional[Path] = typer.Option(
        None,
        "--yolo-weights",
        exists=True,
        readable=True,
        help="Optionnel : ajoute un mode YOLO parking si le fichier existe",
    ),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
) -> None:
    """Compare surface OSM seule, géométrie, SegFormer et éventuellement ML sur une même adresse."""
    from parking_capacity.vision_benchmark import run_benchmark_vision_modes

    p = run_benchmark_vision_modes(
        address,
        out,
        radius_m=radius_m,
        ml_checkpoint=ml_checkpoint,
        yolo_weights=yolo_weights,
        cache_dir=cache_dir,
    )
    typer.echo(f"Benchmark vision : {p.resolve()}")


@app.command("evaluate-manual-review")
def evaluate_manual_review_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        readable=True,
        help="manual_review.csv (colonnes human_count, …)",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire : manual_eval_report.md, summary.json"),
) -> None:
    """Compare estimated_capacity aux comptages humains."""
    from parking_capacity.evaluate_manual import run_evaluate_manual_review

    p = run_evaluate_manual_review(input_path, out)
    typer.echo(f"Rapport : {p}")


@app.command("go-no-go-report")
def go_no_go_report_cmd(
    benchmark: Path = typer.Option(
        ...,
        "--benchmark",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Répertoire sortie benchmark-addresses",
    ),
    model: Path = typer.Option(
        ...,
        "--model",
        exists=True,
        readable=True,
        help="Chemin model.pt",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Fichier markdown go/no-go"),
    manual_eval: Optional[Path] = typer.Option(
        None,
        "--manual-eval",
        file_okay=False,
        dir_okay=True,
        help="Répertoire evaluate-manual-review (défaut : parent du benchmark / manual_eval)",
    ),
) -> None:
    """Synthèse fiabilité produit (benchmark + métadonnées modèle + optionnel inspection humaine)."""
    from parking_capacity.go_no_go import write_go_no_go_report

    write_go_no_go_report(benchmark, model, out, manual_eval_dir=manual_eval)
    typer.echo(f"Rapport : {out.resolve()}")


@app.command("datasets-download")
def datasets_download_cmd(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="apklot | dota | xview | spacenet",
    ),
    raw_dir: Optional[Path] = typer.Option(
        None,
        "--raw-dir",
        help="Répertoire brut (défaut : data/datasets/raw/<dataset>)",
    ),
) -> None:
    """Télécharge ou prépare l’auto-téléchargement des jeux satellite (voir docs/datasets.md)."""
    from parking_capacity.datasets_satellite import apklot, dota, spacenet, xview

    dataset = dataset.lower().strip()
    if dataset == "apklot":
        r = apklot.download_apklot(dest=raw_dir)
    elif dataset == "dota":
        r = dota.download_dota(dest=raw_dir)
    elif dataset == "xview":
        r = xview.download_xview(dest=raw_dir)
    elif dataset == "spacenet":
        r = spacenet.download_spacenet(dest=raw_dir)
    else:
        raise typer.BadParameter("dataset inconnu")
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("datasets-prepare")
def datasets_prepare_cmd(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="apklot | dota | xview | spacenet",
    ),
    raw_dir: Optional[Path] = typer.Option(None, "--raw-dir"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    dataset_subset: str = typer.Option(
        "full",
        "--dataset-subset",
        help="APKLOT : full | small",
    ),
    subset_max_images: int = typer.Option(36, "--subset-max-images"),
    apklot_view: str = typer.Option(
        "auto",
        "--apklot-view",
        help="APKLOT : auto|satellite|camera|all (auto=satellite seul)",
    ),
) -> None:
    """Convertit les données brutes vers parking_capacity_dataset + manifests."""
    from parking_capacity.datasets_satellite import apklot, dota, spacenet, xview

    dataset = dataset.lower().strip()
    if dataset == "apklot":
        r = apklot.prepare_apklot_dataset(
            raw_root=raw_dir,
            out_dir=out_dir,
            dataset_subset=dataset_subset,
            subset_max_images=subset_max_images,
            apklot_view=apklot_view,
        )
    elif dataset == "dota":
        r = dota.prepare_dota_dataset(raw_root=raw_dir, out_dir=out_dir)
    elif dataset == "xview":
        r = xview.prepare_xview_dataset(raw_root=raw_dir, out_dir=out_dir)
    elif dataset == "spacenet":
        r = spacenet.prepare_spacenet_dataset(raw_root=raw_dir, out_dir=out_dir)
    else:
        raise typer.BadParameter("dataset inconnu")
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("train-segformer")
def train_segformer_cmd(
    dataset: str = typer.Option("apklot", "--dataset", help="Nom jeu préparé sous data/datasets/prepared"),
    epochs: int = typer.Option(50, "--epochs"),
    img_size: int = typer.Option(1024, "--img-size"),
    output_dir: Path = typer.Option(Path("runs/segformer"), "--output-dir", "-o"),
    batch_size: int = typer.Option(2, "--batch-size"),
    force_incompatible_dataset: bool = typer.Option(False, "--force-incompatible-dataset"),
) -> None:
    """Fine-tuning SegFormer sur parking_capacity_dataset."""
    from pathlib import Path as P

    from parking_capacity.datasets_satellite.dataset_types import (
        assert_satellite_segmentation_training_allowed,
    )
    from parking_capacity.datasets_satellite.registry import load_registry

    reg = load_registry()
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        raise typer.BadParameter(f"Dataset inconnu dans le registre : {dataset}")
    try:
        assert_satellite_segmentation_training_allowed(
            dataset,
            force=force_incompatible_dataset,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))
    root = P(info["prepared_path"]) / "parking_capacity_dataset"
    if not root.is_dir():
        raise typer.BadParameter(f"Chemin préparé introuvable : {root}")
    cmd = [
        sys.executable,
        "-m",
        "parking_capacity.training.train_segformer",
        "--dataset-root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--img-size",
        str(img_size),
    ]
    rc = subprocess.call(cmd)
    raise typer.Exit(rc)


@app.command("train-yolo-seg")
def train_yolo_seg_cmd(
    dataset: str = typer.Option("apklot", "--dataset"),
    model: str = typer.Option("yolov8m-seg.pt", "--model", "-m"),
    epochs: int = typer.Option(20, "--epochs"),
    imgsz: int = typer.Option(640, "--imgsz", "--img-size"),
    patience: int = typer.Option(15, "--patience"),
    batch: int = typer.Option(-1, "--batch"),
    output_dir: Path = typer.Option(Path("data/runs/yolo_seg"), "--output-dir", "-o"),
    resume: bool = typer.Option(False, "--resume"),
    weights_resume: Optional[Path] = typer.Option(None, "--weights"),
    save_period: int = typer.Option(
        0,
        "--save-period",
        help="Checkpoint tous les N epochs (0=désactivé). Mettre OUTPUT_RUN sur Drive pour sauvegarder là.",
    ),
    force_incompatible_dataset: bool = typer.Option(
        False,
        "--force-incompatible-dataset",
        help="Contourner le contrôle dataset_type / APKLOT satellite.",
    ),
) -> None:
    """YOLOv8 segmentation — défauts : 20 epochs, imgsz 640, AMP, batch auto."""
    from pathlib import Path as P

    from parking_capacity.datasets_satellite.dataset_types import (
        assert_satellite_segmentation_training_allowed,
    )
    from parking_capacity.datasets_satellite.registry import load_registry

    reg = load_registry()
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        raise typer.BadParameter(f"Dataset inconnu : {dataset}")
    try:
        assert_satellite_segmentation_training_allowed(
            dataset,
            force=force_incompatible_dataset,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))
    root = P(info["prepared_path"]) / "parking_capacity_dataset"
    cmd = [
        sys.executable,
        "-m",
        "parking_capacity.training.train_yolov8_seg",
        "--dataset-root",
        str(root),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--patience",
        str(patience),
        "--output-dir",
        str(output_dir),
        "--save-period",
        str(save_period),
    ]
    if resume:
        cmd.append("--resume")
    if weights_resume is not None:
        cmd.extend(["--weights", str(weights_resume)])
    rc = subprocess.call(cmd)
    raise typer.Exit(rc)


@app.command("quickstart-apklot-yolo")
def quickstart_apklot_yolo_cmd(
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    dataset_subset: str = typer.Option("small", "--dataset-subset"),
    subset_max_images: int = typer.Option(36, "--subset-max-images"),
    epochs: int = typer.Option(20, "--epochs"),
    imgsz: int = typer.Option(640, "--imgsz"),
    model: str = typer.Option("yolov8m-seg.pt", "--model", "-m"),
    patience: int = typer.Option(15, "--patience"),
    resume: bool = typer.Option(False, "--resume"),
    weights_resume: Optional[Path] = typer.Option(None, "--weights"),
    skip_train: bool = typer.Option(False, "--skip-train"),
    apklot_view: str = typer.Option(
        "satellite",
        "--apklot-view",
        help="auto|satellite|camera|all — défaut satellite (vue Maps)",
    ),
    force_incompatible_dataset: bool = typer.Option(
        False,
        "--force-incompatible-dataset",
        help="Ignore contrôle satellite / préparation caméra-only",
    ),
) -> None:
    """Workflow APKLOT → YOLOv8-seg → data/runs/<nom> (métriques, masks, overlays)."""
    from parking_capacity.quickstart_apklot_yolo import run_quickstart_apklot_yolo

    r = run_quickstart_apklot_yolo(
        run_name=run_name,
        dataset_subset=dataset_subset,
        subset_max_images=subset_max_images,
        epochs=epochs,
        imgsz=imgsz,
        model=model,
        patience=patience,
        resume=resume,
        weights_resume=weights_resume,
        skip_train=skip_train,
        apklot_view=apklot_view,
        force_incompatible_dataset=force_incompatible_dataset,
    )
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("test-segmentation-real")
def test_segmentation_real_cmd(
    address: str = typer.Argument(..., help="Adresse (BAN)"),
    weights: Path = typer.Option(
        ...,
        "--weights",
        "-w",
        exists=True,
        readable=True,
        help="Poids YOLOv8-seg (best.pt)",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire résultats"),
    radius_m: int = typer.Option(80, "--radius-m"),
    chip_pixels: int = typer.Option(640, "--chip-pixels"),
    half_side_m: Optional[float] = typer.Option(None, "--half-side-m"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    multiscale: bool = typer.Option(False, "--multiscale"),
    providers_yaml: Optional[Path] = typer.Option(None, "--providers-yaml"),
    m2_per_space: float = typer.Option(26.0, "--m2-per-space"),
) -> None:
    """Orthophoto IGN + segmentation YOLO + fusion GIS + GeoJSON."""
    from parking_capacity.test_segmentation_real import run_test_segmentation_real

    summary = run_test_segmentation_real(
        address,
        weights,
        out,
        radius_m=radius_m,
        chip_pixels=chip_pixels,
        half_side_m=half_side_m,
        cache_dir=cache_dir,
        multiscale=multiscale,
        providers_yaml=providers_yaml,
        m2_per_space=m2_per_space,
    )
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("train-mask2former")
def train_mask2former_cmd(
    dataset: str = typer.Option("apklot", "--dataset"),
    epochs: int = typer.Option(50, "--epochs"),
    output_dir: Path = typer.Option(Path("runs/mask2former"), "--output-dir", "-o"),
    force_incompatible_dataset: bool = typer.Option(False, "--force-incompatible-dataset"),
) -> None:
    """Point d’entrée Mask2Former (documentation + hook Hugging Face)."""
    from pathlib import Path as P

    from parking_capacity.datasets_satellite.dataset_types import (
        assert_satellite_segmentation_training_allowed,
    )
    from parking_capacity.datasets_satellite.registry import load_registry

    reg = load_registry()
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        raise typer.BadParameter(f"Dataset inconnu : {dataset}")
    try:
        assert_satellite_segmentation_training_allowed(
            dataset,
            force=force_incompatible_dataset,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))
    root = P(info["prepared_path"]) / "parking_capacity_dataset"
    cmd = [
        sys.executable,
        "-m",
        "parking_capacity.training.train_mask2former",
        "--dataset-root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
    ]
    rc = subprocess.call(cmd)
    raise typer.Exit(rc)


@app.command("train-vehicle-detector")
def train_vehicle_detector_cmd(
    dataset: str = typer.Option("xview", "--dataset"),
    model: str = typer.Option("yolov8m.pt", "--model", "-m"),
    epochs: int = typer.Option(50, "--epochs"),
    output_dir: Path = typer.Option(Path("runs/vehicle_det"), "--output-dir", "-o"),
) -> None:
    """Détecteur véhicules (YOLO) — jeu au format Ultralytics (dataset.yaml)."""
    from pathlib import Path as P

    from parking_capacity.datasets_satellite.registry import load_registry

    reg = load_registry()
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        raise typer.BadParameter(f"Dataset inconnu : {dataset}")
    root = P(info["prepared_path"]) / "parking_capacity_dataset"
    cmd = [
        sys.executable,
        "-m",
        "parking_capacity.training.train_vehicle_detector",
        "--dataset-root",
        str(root),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--output-dir",
        str(output_dir),
    ]
    rc = subprocess.call(cmd)
    raise typer.Exit(rc)


@app.command("benchmark-satellite-modes")
def benchmark_satellite_modes_cmd(
    benchmark_dir: Path = typer.Option(
        ...,
        "--benchmark-dir",
        "-b",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Répertoire benchmark-addresses",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="benchmark_satellite_summary.md"),
    extended: bool = typer.Option(
        False,
        "--extended",
        help="Inclure segmentation_benchmark.json (modes A–D) si présents",
    ),
) -> None:
    """Agrège les result.json ; option ``--extended`` pour les modes segmentation."""
    from parking_capacity.training.satellite_benchmark import (
        write_extended_satellite_benchmark_report,
        write_satellite_benchmark_report,
    )

    if extended:
        summ = write_extended_satellite_benchmark_report(benchmark_dir, out)
    else:
        summ = write_satellite_benchmark_report(benchmark_dir, out)
    typer.echo(json.dumps(summ, indent=2, default=str))


@app.command("inspect-dataset")
def inspect_dataset_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="apklot | dota | xview | spacenet"),
    project_root: Optional[Path] = typer.Option(
        None,
        "--project-root",
        help="Racine du dépôt (défaut : détection automatique)",
    ),
) -> None:
    """Résumé caméra/satellite, masques, résolution, adéquation orthophoto."""
    from parking_capacity.datasets_satellite.dataset_inspect import inspect_dataset

    r = inspect_dataset(dataset.strip().lower(), project_root)
    typer.echo(json.dumps(r, indent=2, ensure_ascii=False, default=str))


@app.command("benchmark-dataset-mosaics")
def benchmark_dataset_mosaics_cmd(
    out: Path = typer.Option(
        Path("data/datasets/benchmark_mosaic.png"),
        "--out",
        "-o",
        help="PNG sortie",
    ),
    datasets: str = typer.Option(
        "apklot,dota,xview,spacenet",
        "--datasets",
        help="Liste séparée par virgules",
    ),
    samples: int = typer.Option(4, "--samples", "-n", help="Vignettes par jeu"),
    project_root: Optional[Path] = typer.Option(None, "--project-root"),
) -> None:
    """Mosaïque visuelle pour comparer APKLOT vs DOTA/xView/SpaceNet."""
    from parking_capacity.datasets_satellite.benchmark_mosaic import (
        build_dataset_benchmark_mosaic,
    )

    names = [x.strip().lower() for x in datasets.split(",") if x.strip()]
    if not names:
        raise typer.BadParameter("--datasets vide")
    r = build_dataset_benchmark_mosaic(
        names,
        out,
        project_root=project_root,
        samples_per_dataset=samples,
    )
    typer.echo(json.dumps(r, indent=2, ensure_ascii=False, default=str))


@app.command("dataset-stats")
def dataset_stats_cmd(
    dataset: str = typer.Option("apklot", "--dataset", "-d"),
    project_root: Optional[Path] = typer.Option(
        None,
        "--project-root",
        help="Racine du clone (défaut : racine du paquet installé)",
    ),
) -> None:
    """Résumé taille / présence du jeu préparé et du layout YOLO."""
    from parking_capacity.colab_export import dataset_stats

    r = dataset_stats(dataset, project_root)
    if r.get("error"):
        raise typer.BadParameter(str(r["error"]))
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("doctor-build")
def doctor_build_cmd(
    export_dir: Optional[Path] = typer.Option(
        None,
        "--export-dir",
        "-e",
        help="Dossier export Colab (contient build_info.json et train_colab.ipynb)",
    ),
    build_info: Optional[Path] = typer.Option(
        None,
        "--build-info",
        "-b",
        help="Chemin explicite vers build_info.json",
    ),
    notebook: Optional[Path] = typer.Option(
        None,
        "--notebook",
        "-n",
        help="Notebook à comparer au hash enregistré",
    ),
) -> None:
    """Compare version pip, build_info exporté et hash du notebook (cohérence Colab)."""
    from parking_capacity.colab_export import run_doctor_build

    r = run_doctor_build(
        export_dir=export_dir,
        build_info_file=build_info,
        notebook_file=notebook,
    )
    typer.echo(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    if not r.get("ok"):
        raise typer.Exit(1)


@app.command("export-colab-training")
def export_colab_training_cmd(
    out: Path = typer.Option(Path("data/colab_export"), "--out", "-o", help="Répertoire sortie"),
    dataset: str = typer.Option("apklot", "--dataset", "-d"),
    include_current_dataset: bool = typer.Option(
        True,
        "--include-current-dataset/--no-include-current-dataset",
        help="Copier parking_capacity_dataset + yolo_seg_dataset si présents",
    ),
    include_config: bool = typer.Option(
        True,
        "--include-config/--no-include-config",
        help="Copier providers.yaml.example dans configs/",
    ),
    include_notebooks: bool = typer.Option(
        True,
        "--include-notebooks/--no-include-notebooks",
        help="Copier train_colab.ipynb à la racine du pack",
    ),
    max_dataset_mb: Optional[float] = typer.Option(
        None,
        "--max-dataset-mb",
        help="Taille max copie dataset (Mo), défaut 400 ; au-delà manifest seul",
    ),
) -> None:
    """Export ZIP + README + snapshot projet pour entraînement Google Colab."""
    from parking_capacity.colab_export import DEFAULT_MAX_DATASET_EXPORT_BYTES, run_export_colab_training

    max_b = (
        int(max_dataset_mb * 1024 * 1024)
        if max_dataset_mb is not None
        else DEFAULT_MAX_DATASET_EXPORT_BYTES
    )
    r = run_export_colab_training(
        out,
        dataset=dataset,
        include_current_dataset=include_current_dataset,
        include_config=include_config,
        include_notebooks=include_notebooks,
        max_dataset_bytes=max_b,
    )
    if not r.get("ok"):
        typer.echo(r.get("error", "export failed"), err=True)
        raise typer.Exit(1)
    for w in r.get("security_warnings") or []:
        typer.echo(f"[export warning] {w}", err=True)
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("import-colab-model")
def import_colab_model_cmd(
    zip_path: Path = typer.Option(..., "--zip", exists=True, readable=True, help="Archive résultats Colab"),
    out: Path = typer.Option(
        Path("data/models/apklot_yolo"),
        "--out",
        "-o",
        help="Dossier local pour best.pt et métriques",
    ),
) -> None:
    """Importe best.pt, métriques et dataset.yaml depuis une archive Colab."""
    from parking_capacity.colab_export import run_import_colab_model

    r = run_import_colab_model(zip_path, out)
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("validate-benchmark")
def validate_benchmark_cmd(
    input_csv: Path = typer.Option(
        ...,
        "--input", "-i",
        exists=True, readable=True,
        help="CSV avec colonnes au minimum 'address' et 'human_count' "
             "(estimated_capacity sera ajoutée si absente).",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Répertoire de sortie."),
    skip_inference: bool = typer.Option(
        False, "--skip-inference",
        help="Si la colonne estimated_capacity est déjà présente, ne pas relancer le pipeline.",
    ),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    overpass_delay: float = typer.Option(1.0, "--overpass-delay"),
    source_priority: str = typer.Option("hybrid", "--source-priority"),
    radius_m: int = typer.Option(80, "--radius-m"),
    chip_half_side_m: float = typer.Option(80.0, "--chip-half-side-m"),
    visual_backend: str = typer.Option("auto", "--visual-backend"),
    providers_yaml: Optional[Path] = typer.Option(None, "--providers-yaml"),
    auto_yolo: bool = typer.Option(False, "--auto-yolo"),
    aerial_yolo: bool = typer.Option(False, "--aerial-yolo", help="VisDrone aérien via HF Hub."),
    dota_yolo: bool = typer.Option(False, "--dota-yolo", help="YOLO fine-tuné sur DOTAv1 vehicles."),
    vehicle_yolo_weights: Optional[Path] = typer.Option(None, "--vehicle-yolo-weights", exists=True, readable=True),
    slot_yolo_weights: Optional[Path] = typer.Option(None, "--slot-yolo-weights", exists=True, readable=True),
    roboflow_api_key: Optional[str] = typer.Option(None, "--roboflow-api-key"),
    roboflow_model_id: Optional[str] = typer.Option(None, "--roboflow-model-id"),
) -> None:
    """Validation terrain avec MAE/MAPE/R²/bootstrap CI + segmentation par typologie.

    Format CSV : address, human_count (obligatoires) ; site_type, notes (optionnels).
    """
    import pandas as pd

    from parking_capacity.benchmark_validation import write_validation_report
    from parking_capacity.pipeline import process_address, row_to_json_serializable

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    if "address" not in df.columns:
        for c in ("Adresse", "adresse", "Address"):
            if c in df.columns:
                df = df.rename(columns={c: "address"})
                break
    if "human_count" not in df.columns:
        raise typer.BadParameter("CSV doit contenir une colonne 'human_count'.")

    if not skip_inference:
        rows_out = []
        for idx, row in df.iterrows():
            addr = str(row.get("address", "")).strip()
            if not addr:
                continue
            try:
                r = process_address(
                    addr,
                    search_radius_m=radius_m,
                    chip_half_side_m=chip_half_side_m,
                    use_vision=True,
                    cache_dir=cache_dir,
                    source_priority=source_priority,
                    visual_backend=visual_backend,
                    providers_yaml=providers_yaml,
                    overpass_delay_s=overpass_delay,
                    auto_download_vehicle_yolo=auto_yolo,
                    auto_download_aerial_yolo=aerial_yolo,
                    use_dota_finetuned_yolo=dota_yolo,
                    vehicle_yolo_weights=vehicle_yolo_weights,
                    slot_yolo_weights=slot_yolo_weights,
                    roboflow_api_key=roboflow_api_key,
                    roboflow_model_id=roboflow_model_id,
                )
                d = row_to_json_serializable(r)
            except Exception as e:  # noqa: BLE001
                d = {"address": addr, "error": str(e)}
            d["human_count"] = row.get("human_count")
            if "site_type" in row and not pd.isna(row["site_type"]):
                d["site_type"] = row["site_type"]
            rows_out.append(d)
            typer.echo(f"[{idx+1}/{len(df)}] {addr} → "
                       f"est={d.get('estimated_capacity')}, hum={d.get('human_count')}")
        df = pd.DataFrame(rows_out)
        df.to_csv(out / "predictions.csv", index=False)

    md = write_validation_report(df, out)
    typer.echo(f"Rapport : {md}")


@app.command("evaluate-front-validations")
def evaluate_front_validations_cmd(
    input_csv: Path = typer.Option(
        Path("data/benchmark/manual_front_validations.csv"),
        "--input", "-i",
        exists=True, readable=True,
        help="CSV produit par le panneau de validation de l'UI.",
    ),
    out: Path = typer.Option(
        Path("data/benchmark/manual_front_validation_report"),
        "--out", "-o",
        help="Répertoire de sortie du rapport.",
    ),
) -> None:
    """Évalue les annotations enregistrées depuis le front sans relancer le modèle."""
    from parking_capacity.benchmark_validation import write_front_validation_report

    md = write_front_validation_report(input_csv, out)
    typer.echo(f"Rapport : {md}")


def main() -> None:
    try:
        from dotenv import load_dotenv

        roots = (Path.cwd(), Path(__file__).resolve().parent.parent.parent)
        for root in roots:
            env_path = root / ".env"
            if env_path.is_file():
                load_dotenv(env_path)
                break
    except ImportError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    app()


if __name__ == "__main__":
    main()
