"""Interface web locale (Gradio) pour une adresse à la fois."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from parking_capacity.pipeline import process_address, row_to_json_serializable


def _analyze(
    address: str,
    no_vision: bool,
    vision_compare: bool,
    aerial_first: bool,
    radius_m: float,
    buffer_m: float,
    min_intersection_m2: float,
    overpass_delay: float,
    chip_half_side_m: float,
    chip_pixels: float,
    m2_per_space: float,
    ml_checkpoint: str,
    ml_mode: str,
) -> Tuple[str, Optional[np.ndarray], pd.DataFrame]:
    addr = (address or "").strip()
    if not addr:
        empty = pd.DataFrame()
        return "**Entrez une adresse.**", None, empty

    ml_ckpt = (ml_checkpoint or "").strip() or None
    want_chip = (not bool(no_vision)) or bool(ml_ckpt)
    chip_tmp: Optional[Path] = None
    if want_chip:
        fd, name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        chip_tmp = Path(name)

    try:
        r = process_address(
            addr,
            overpass_delay_s=float(overpass_delay),
            search_radius_m=int(radius_m),
            point_buffer_m=float(buffer_m),
            min_intersection_m2=float(min_intersection_m2),
            use_vision=not bool(no_vision),
            vision_device=None,
            chip_half_side_m=float(chip_half_side_m),
            chip_pixels=int(chip_pixels),
            wms_base=None,
            wms_layer=None,
            m2_per_space=float(m2_per_space),
            vision_compare=bool(vision_compare),
            include_raw=True,
            ml_checkpoint=ml_ckpt,
            ml_mode=(ml_mode or "fallback").strip(),
            ml_device=None,
            save_chip_path=chip_tmp,
            aerial_first=bool(aerial_first),
        )
    finally:
        pass

    d: dict[str, Any] = row_to_json_serializable(r)
    raw_dbg = d.pop("raw_debug", None)
    ortho: Optional[np.ndarray] = None
    if chip_tmp is not None and chip_tmp.is_file():
        try:
            ortho = np.array(Image.open(chip_tmp).convert("RGB"))
        except OSError:
            ortho = None
        try:
            chip_tmp.unlink(missing_ok=True)
        except OSError:
            pass

    lines = [
        "### Résultat",
        "",
        f"- **Coordonnées** : `{d.get('lat')}`, `{d.get('lon')}` (rayon analyse : `{d.get('radius_m_used')}` m)",
        f"- **Capacité estimée** : `{d.get('estimated_capacity')}` (min `{d.get('min_capacity')}` — max `{d.get('max_capacity')}`)",
        f"- **Méthode** : `{d.get('method_used')}`",
        f"- **Sources** : `{d.get('sources_used')}`",
        f"- **Parkings OSM à proximité** : `{d.get('nearby_osm_parkings_count')}`",
        f"- **Places parking_space OSM** : `{d.get('osm_parking_space_count')}`",
        f"- **Surface sans tag (m²)** : `{d.get('area_total_m2')}`",
        f"- **Baseline (hors ordre fusion)** : `{d.get('baseline_estimate')}` (`{d.get('baseline_method')}`)",
        f"- **Note ML vs baseline** : `{d.get('ml_vs_baseline_note') or '—'}`",
        f"- **Vision** : `{d.get('vision_estimated_spaces')}`",
        f"- **ML** : `{d.get('ml_estimated_capacity')}` (brut `{d.get('ml_estimated_raw')}`)",
        f"- **Erreur** : `{d.get('error')}`",
        "",
        "**Avertissements**",
        "",
        d.get("warnings") or d.get("caveats") or "—",
        "",
        "<details><summary>JSON complet</summary>",
        "",
        "```json",
        json.dumps(d, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "</details>",
    ]
    if raw_dbg:
        lines.extend(
            [
                "",
                "<details><summary>raw_debug</summary>",
                "",
                "```json",
                json.dumps(raw_dbg, ensure_ascii=False, indent=2, default=str),
                "```",
                "",
                "</details>",
            ]
        )

    df = pd.DataFrame([d])
    return "\n".join(lines), ortho, df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Capacité parking — adresse") as demo:
        gr.Markdown(
            "# Capacité de stationnement (France)\n\n"
            "Saisissez une adresse, puis **Analyser**. Pipeline : **BAN** → **cadastre** → **OSM (Overpass)** → "
            "**orthophoto IGN** + **SegFormer** + éventuellement **ML** (`model.pt`). "
            "Sans tag `capacity` OSM, l’orthophoto est priorisée (voir option *Priorité orthophoto*).\n\n"
            "Respectez les CGU des services publics ; évitez les requêtes massives."
        )
        address = gr.Textbox(
            label="Adresse",
            placeholder="Ex. 10 Rue de la Santé, 75014 Paris",
            lines=2,
        )
        with gr.Row():
            no_vision = gr.Checkbox(label="Sans vision (plus rapide)", value=False)
            vision_compare = gr.Checkbox(
                label="Vision même si OSM a une capacité (comparaison)", value=False
            )
            aerial_first = gr.Checkbox(
                label="Priorité orthophoto si pas de capacity OSM",
                value=True,
            )
        with gr.Row():
            radius_m = gr.Number(label="Rayon analyse (m)", value=50, minimum=30, maximum=500, step=10)
            buffer_m = gr.Number(label="Buffer point (m)", value=40, minimum=10, maximum=200, step=5)
        with gr.Row():
            min_intersection_m2 = gr.Number(
                label="Seuil intersection parking / parcelle (m²)",
                value=25,
                minimum=1,
                maximum=500,
                step=1,
            )
            overpass_delay = gr.Number(
                label="Pause Overpass (s)",
                value=1.0,
                minimum=0,
                maximum=10,
                step=0.5,
            )
        with gr.Accordion("Modèle ML (train-model)", open=False):
            ml_checkpoint = gr.Textbox(
                label="Chemin vers model.pt (vide = désactivé)",
                placeholder="/chemin/vers/run/model.pt",
                lines=1,
            )
            ml_mode = gr.Dropdown(
                label="Mode ML",
                choices=["fallback", "before_vision", "aux"],
                value="fallback",
            )
        with gr.Accordion("Orthophoto / vision", open=False):
            chip_half_side_m = gr.Number(
                label="Demi-côté puce orthophoto (m)", value=55, minimum=20, maximum=200, step=5
            )
            chip_pixels = gr.Number(label="Taille image (px)", value=512, minimum=256, maximum=1024, step=64)
            m2_per_space = gr.Number(
                label="m² par place (heuristique vision)", value=26, minimum=15, maximum=40, step=1
            )

        btn = gr.Button("Analyser", variant="primary")
        summary_md = gr.Markdown()
        ortho_img = gr.Image(label="Puce orthophoto (aperçu)", type="numpy")
        table = gr.Dataframe(label="Tableau (une ligne)", interactive=False)

        btn.click(
            fn=_analyze,
            inputs=[
                address,
                no_vision,
                vision_compare,
                aerial_first,
                radius_m,
                buffer_m,
                min_intersection_m2,
                overpass_delay,
                chip_half_side_m,
                chip_pixels,
                m2_per_space,
                ml_checkpoint,
                ml_mode,
            ],
            outputs=[summary_md, ortho_img, table],
        )
    return demo


def launch_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    demo = build_app()
    demo.launch(server_name=host, server_port=port, share=share, show_error=True)
