"""Interface web locale (Gradio) pour une adresse à la fois."""

from __future__ import annotations

import json
import os
import tempfile
from csv import DictWriter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parking_capacity.pipeline import process_address, row_to_json_serializable

VALIDATION_CSV = Path("data/benchmark/manual_front_validations.csv")


def _format_consistency_flags(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "_aucun flag — la prédiction est cohérente avec les signaux observés._"
    icons = {"high": "🔴", "medium": "🟠", "info": "🔵"}
    lines = []
    for f in flags:
        sev = str(f.get("severity") or "info")
        name = str(f.get("name") or "")
        reason = str(f.get("reason") or "")
        lines.append(f"- {icons.get(sev, '•')} **{sev}** `{name}` — {reason}")
    return "\n".join(lines)


def _draw_detection_overlay(image: Image.Image, data: dict[str, Any]) -> np.ndarray:
    im = image.convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    def poly_points(item: dict[str, Any]) -> list[tuple[float, float]]:
        pts = item.get("polygon")
        if isinstance(pts, list) and len(pts) >= 3:
            out = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    out.append((float(p[0]), float(p[1])))
            if len(out) >= 3:
                return out
        cx = float(item.get("cx", 0.0))
        cy = float(item.get("cy", 0.0))
        w = float(item.get("width_px", 1.0))
        h = float(item.get("height_px", 1.0))
        return [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]

    slot_evidence = data.get("slot_evidence") if isinstance(data.get("slot_evidence"), dict) else {}
    slot_method = str(slot_evidence.get("method") or data.get("slot_detection_method") or "none")
    heuristic_slots_hidden = False
    for item in slot_evidence.get("detections", []) or []:
        if not isinstance(item, dict):
            continue
        # Les slots heuristiques sont une extrapolation de capacité, pas une vraie détection
        # place par place. On ne les dessine donc pas comme Roboflow.
        if str(item.get("source") or "") == "heuristic" or slot_method.startswith("heuristic"):
            heuristic_slots_hidden = True
            continue
        status = str(item.get("status") or "unknown")
        if status == "filled":
            stroke = (255, 95, 65, 255)
            fill = (255, 95, 65, 50)
        elif status == "empty":
            stroke = (45, 210, 120, 255)
            fill = (45, 210, 120, 45)
        else:
            stroke = (250, 210, 60, 255)
            fill = (250, 210, 60, 40)
        pts = poly_points(item)
        draw.polygon(pts, fill=fill, outline=stroke)
        draw.line(pts + [pts[0]], fill=stroke, width=2)

    vehicles = data.get("vehicles") if isinstance(data.get("vehicles"), dict) else {}
    vehicle_method = str(data.get("vehicle_detection_method") or "")
    for item in vehicles.get("detections", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            conf = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        # Le fallback OpenCV produit des candidats bruités ; on ne montre que les plus crédibles.
        if vehicle_method.startswith("opencv") and conf < 0.55:
            continue
        pts = poly_points(item)
        stroke = (70, 150, 255, 255)
        fill = (70, 150, 255, 50)
        draw.polygon(pts, fill=fill, outline=stroke)
        draw.line(pts + [pts[0]], fill=stroke, width=2)

    legend = [
        ("Vehicule candidat", (70, 150, 255, 255)),
        ("Place occupee detectee", (255, 95, 65, 255)),
        ("Place vide detectee", (45, 210, 120, 255)),
    ]
    if heuristic_slots_hidden:
        legend.append(("Slots heuristiques masques", (250, 180, 45, 255)))
    x0, y0 = 12, 12
    for i, (txt, color) in enumerate(legend):
        y = y0 + i * 18
        draw.rectangle([x0, y + 3, x0 + 11, y + 14], fill=color)
        draw.text((x0 + 17, y), txt, fill=(255, 255, 255, 255), font=font)

    return np.array(Image.alpha_composite(im, overlay).convert("RGB"))


def _analyze(
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
    slot_yolo_weights,
    roboflow_api_key,
    roboflow_model_id,
):
    addr = (address or "").strip()
    if not addr:
        return "**Entrez une adresse.**", None, None, {}

    ml_ckpt = (ml_checkpoint or "").strip() or None
    slot_weights = (slot_yolo_weights or "").strip() or None
    rf_key = (roboflow_api_key or "").strip() or None
    rf_model = (roboflow_model_id or "").strip() or None
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
            slot_yolo_weights=Path(slot_weights) if slot_weights else None,
            roboflow_api_key=rf_key,
            roboflow_model_id=rf_model,
        )
    finally:
        pass

    d: dict[str, Any] = row_to_json_serializable(r)
    raw_dbg = d.pop("raw_debug", None)
    ortho: Optional[Image.Image] = None
    overlay: Optional[Image.Image] = None
    if chip_tmp is not None and chip_tmp.is_file():
        try:
            chip_img = Image.open(chip_tmp).convert("RGB")
            ortho = chip_img
            overlay = Image.fromarray(_draw_detection_overlay(chip_img, d))
        except OSError:
            ortho = None
            overlay = None
        try:
            chip_tmp.unlink(missing_ok=True)
        except OSError:
            pass

    lines = [
        "### Résultat",
        "",
        f"- **Coordonnées** : `{d.get('lat')}`, `{d.get('lon')}` (rayon analyse : `{d.get('radius_m_used')}` m)",
        f"- **Capacité estimée (privée)** : `{d.get('estimated_capacity')}` (min `{d.get('min_capacity')}` — max `{d.get('max_capacity')}`)",
        f"- **Capacité parking public OSM à côté** : `{d.get('nearby_public_capacity_estimate')}` "
        f"(min `{d.get('nearby_public_capacity_min')}` — max `{d.get('nearby_public_capacity_max')}`, "
        f"source `{d.get('nearby_public_capacity_source')}`)",
        f"- **Méthode** : `{d.get('method_used')}`",
        f"- **Sources** : `{d.get('sources_used')}`",
        f"- **Parkings OSM à proximité** : `{d.get('nearby_osm_parkings_count')}`",
        f"- **Places parking_space OSM** : `{d.get('osm_parking_space_count')}`",
        f"- **Surface sans tag (m²)** : `{d.get('area_total_m2')}`",
        f"- **Baseline (hors ordre fusion)** : `{d.get('baseline_estimate')}` (`{d.get('baseline_method')}`)",
        f"- **Note ML vs baseline** : `{d.get('ml_vs_baseline_note') or '—'}`",
        f"- **Vision** : `{d.get('vision_estimated_spaces')}`",
        f"- **Slots détectés** : `{d.get('slots_total_count')}` dont vides `{d.get('slots_empty_count')}` / occupés `{d.get('slots_filled_count')}` (`{d.get('slot_detection_method')}`)",
        f"- **Véhicules détectés** : `{d.get('vehicle_count')}` (`{d.get('vehicle_detection_method')}`)",
        f"- **ML** : `{d.get('ml_estimated_capacity')}` (brut `{d.get('ml_estimated_raw')}`)",
        f"- **Cohérence** : sévérité `{d.get('consistency_max_severity')}`, "
        f"{d.get('consistency_high_count') or 0} flag(s) `high`, "
        f"review {'**OUI**' if d.get('consistency_needs_review') else 'non'}",
        f"- **Erreur** : `{d.get('error')}`",
        "",
        "**Flags de cohérence**",
        "",
        _format_consistency_flags(d.get("consistency_flags") or []),
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

    return "\n".join(lines), ortho, overlay, d


def _save_validation(
    last_prediction,
    true_capacity,
    verdict,
    parcel_ok,
    private_boundary_ok,
    correction_reason,
    reviewer_notes,
):
    if not isinstance(last_prediction, dict) or not last_prediction.get("input_address"):
        return "Aucune prédiction en mémoire. Lance d'abord une analyse."

    try:
        true_cap = int(true_capacity)
    except (TypeError, ValueError):
        return "Renseigne une capacité réelle entière, par exemple 0, 4, 12."

    pred = last_prediction
    pred_cap = pred.get("estimated_capacity")
    try:
        pred_cap_int = int(pred_cap) if pred_cap is not None else None
    except (TypeError, ValueError):
        pred_cap_int = None

    abs_error = abs(pred_cap_int - true_cap) if pred_cap_int is not None else None
    row = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "input_address": pred.get("input_address"),
        "ban_label": pred.get("ban_label"),
        "lat": pred.get("lat"),
        "lon": pred.get("lon"),
        "predicted_capacity": pred_cap_int,
        "predicted_min": pred.get("min_capacity"),
        "predicted_max": pred.get("max_capacity"),
        "true_capacity": true_cap,
        "abs_error": abs_error,
        "verdict": verdict or "",
        "parcel_ok": bool(parcel_ok),
        "private_boundary_ok": bool(private_boundary_ok),
        "correction_reason": correction_reason or "",
        "reviewer_notes": reviewer_notes or "",
        "primary_source": pred.get("primary_source"),
        "primary_confidence": pred.get("primary_confidence"),
        "method_used": pred.get("method_used"),
        "sources_used": pred.get("sources_used"),
        "private_garage_capacity": json.dumps(pred.get("private_garage_capacity"), ensure_ascii=False, default=str),
        "nearby_public_or_shared_parking": json.dumps(pred.get("nearby_public_or_shared_parking"), ensure_ascii=False, default=str),
        "nearby_public_capacity_estimate": pred.get("nearby_public_capacity_estimate"),
        "nearby_public_capacity_min": pred.get("nearby_public_capacity_min"),
        "nearby_public_capacity_max": pred.get("nearby_public_capacity_max"),
        "nearby_public_capacity_source": pred.get("nearby_public_capacity_source"),
        "consistency_max_severity": pred.get("consistency_max_severity"),
        "consistency_high_count": pred.get("consistency_high_count"),
        "consistency_needs_review": pred.get("consistency_needs_review"),
        "consistency_flags": json.dumps(pred.get("consistency_flags") or [], ensure_ascii=False, default=str),
        "vehicle_count": pred.get("vehicle_count"),
        "vehicle_detection_method": pred.get("vehicle_detection_method"),
        "slot_detection_method": pred.get("slot_detection_method"),
        "warnings": pred.get("warnings") or pred.get("caveats") or "",
    }

    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not VALIDATION_CSV.exists() or VALIDATION_CSV.stat().st_size == 0

    # Si le CSV existe déjà avec un schéma antérieur (avant l'ajout des colonnes
    # nearby_public_capacity_*), on aligne sur l'entête existante pour ne pas
    # désaligner les colonnes des anciennes lignes. Les nouveaux champs sont ignorés
    # silencieusement jusqu'à régénération du fichier.
    fieldnames: list[str]
    if write_header:
        fieldnames = list(row.keys())
    else:
        with VALIDATION_CSV.open("r", encoding="utf-8") as fh:
            header_line = fh.readline().strip()
        fieldnames = [c.strip() for c in header_line.split(",")] if header_line else list(row.keys())

    with VALIDATION_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return (
        f"Validation enregistrée dans `{VALIDATION_CSV}` : "
        f"prédiction={pred_cap_int}, vérité={true_cap}, erreur={abs_error}."
    )


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
        with gr.Accordion("Détection places façon Roboflow", open=False):
            slot_yolo_weights = gr.Textbox(
                label="Poids YOLO slots local (.pt)",
                placeholder="/chemin/vers/best.pt",
                lines=1,
            )
            roboflow_api_key = gr.Textbox(
                label="Roboflow API key",
                type="password",
                lines=1,
            )
            roboflow_model_id = gr.Textbox(
                label="Roboflow model id",
                placeholder="workspace/project/version",
                lines=1,
            )

        last_prediction = gr.State({})
        btn = gr.Button("Analyser", variant="primary")
        summary_md = gr.Markdown()
        ortho_img = gr.Image(label="Puce orthophoto (aperçu)", type="pil")
        overlay_img = gr.Image(label="Overlay voitures / places vides", type="pil")

        with gr.Accordion("Validation manuelle / vérité terrain", open=True):
            with gr.Row():
                true_capacity = gr.Number(
                    label="Vraie capacité privée",
                    value=0,
                    precision=0,
                    minimum=0,
                    step=1,
                )
                verdict = gr.Dropdown(
                    label="Verdict",
                    choices=[
                        "correct",
                        "acceptable",
                        "sur-estime",
                        "sous-estime",
                        "mauvaise parcelle",
                        "parking public/voisin compté",
                        "impossible à vérifier",
                    ],
                    value="acceptable",
                )
            with gr.Row():
                parcel_ok = gr.Checkbox(label="Parcelle correcte", value=True)
                private_boundary_ok = gr.Checkbox(label="Frontière privé/public correcte", value=True)
            correction_reason = gr.Dropdown(
                label="Cause principale si erreur",
                choices=[
                    "",
                    "géocodage",
                    "cadastre",
                    "parking public ou voisin",
                    "véhicules mal détectés",
                    "zone bitume surestimée",
                    "places marquées non détectées",
                    "parking couvert/souterrain invisible",
                    "ombre/végétation",
                    "autre",
                ],
                value="",
            )
            reviewer_notes = gr.Textbox(label="Notes", lines=3)
            save_validation = gr.Button("Enregistrer validation")
            validation_status = gr.Markdown()

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
                slot_yolo_weights,
                roboflow_api_key,
                roboflow_model_id,
            ],
            outputs=[summary_md, ortho_img, overlay_img, last_prediction],
        )
        save_validation.click(
            fn=_save_validation,
            inputs=[
                last_prediction,
                true_capacity,
                verdict,
                parcel_ok,
                private_boundary_ok,
                correction_reason,
                reviewer_notes,
            ],
            outputs=[validation_status],
        )
    return demo


def _patch_gradio_client_schema_bug() -> None:
    """Gradio 4.44 + gradio-client 1.3 : schémas JSON invalides (bool au lieu de dict)."""
    try:
        import gradio_client.utils as gc_utils

        _orig_get_type = gc_utils.get_type
        _orig_json_schema = gc_utils._json_schema_to_python_type

        def _safe_get_type(schema: Any) -> Any:
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)

        def _safe_json_schema(schema: Any, defs: Any) -> str:
            if not isinstance(schema, dict):
                return "Any"
            return _orig_json_schema(schema, defs)

        gc_utils.get_type = _safe_get_type
        gc_utils._json_schema_to_python_type = _safe_json_schema
    except Exception:
        pass


def launch_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    _patch_gradio_client_schema_bug()
    demo = build_app()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        show_api=False,
        prevent_thread_lock=False,
    )
