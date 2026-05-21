"""UI Gradio : suivi en direct d'un run bulk parking-capacity.

Lit en continu le ledger ``<output>.progress.jsonl`` produit par ``parking-capacity run``
et affiche : progression, ETA, throughput, qualité des prédictions, dernières adresses.

Usage :
    python scripts/bulk_monitor_ui.py \
      --input data/bulk/adresses_clean.csv \
      --ledger data/bulk/run_output.csv.progress.jsonl \
      --port 7861
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

REFRESH_S = 5.0  # autorefresh interval


def _safe_int(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _load_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    records: list[dict] = []
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "—"
    td = timedelta(seconds=int(seconds))
    if td.days > 0:
        return f"{td.days} j {td.seconds // 3600} h"
    h = td.seconds // 3600
    m = (td.seconds % 3600) // 60
    if h > 0:
        return f"{h} h {m} min"
    return f"{m} min {td.seconds % 60} s"


def _build_snapshot(input_path: Path, ledger_path: Path) -> tuple[str, pd.DataFrame, str]:
    """Retourne (markdown stats, dataframe dernières adresses, état brut)."""
    total = 0
    try:
        total = sum(1 for _ in input_path.open("r", encoding="utf-8")) - 1
    except FileNotFoundError:
        pass

    records = _load_ledger(ledger_path)
    n = len(records)

    if n == 0:
        md = (
            f"## Suivi du run bulk\n\n"
            f"Ledger : `{ledger_path}` — **vide**\n\n"
            f"Total attendu : **{total}** adresses\n\n"
            f"En attente du démarrage…"
        )
        return md, pd.DataFrame(), "0 / 0"

    # Throughput : lignes / temps écoulé depuis création du ledger.
    try:
        ledger_age = time.time() - ledger_path.stat().st_ctime
    except FileNotFoundError:
        ledger_age = 0.0
    rate = n / ledger_age if ledger_age > 0 else 0.0
    remaining = max(0, total - n)
    eta_s = remaining / rate if rate > 0 else math.inf

    with_cap = sum(1 for r in records if _safe_int(r.get("estimated_capacity")) is not None)
    with_nearby = sum(1 for r in records if (r.get("nearby_osm_parkings_count") or 0) > 0)
    errors = sum(1 for r in records if r.get("error"))

    pct = (n / total * 100) if total else 0.0
    md = (
        f"## Suivi du run bulk\n\n"
        f"**Progression** : `{n} / {total}` ({pct:.1f} %)\n\n"
        f"**Throughput** : {rate * 60:.1f} adresses/min — ETA **{_format_eta(eta_s)}**\n\n"
        f"**Qualité** : capacité estimée sur **{with_cap}** ({with_cap / n:.0%}) · "
        f"parking à côté sur **{with_nearby}** ({with_nearby / n:.0%}) · "
        f"erreurs **{errors}**\n\n"
        f"Ledger : `{ledger_path.name}` — dernière maj : "
        f"{time.strftime('%H:%M:%S', time.localtime(ledger_path.stat().st_mtime))}"
    )

    # Dernières 20 adresses (par ordre d'écriture dans le ledger = ordre de complétion).
    tail = records[-20:][::-1]
    df = pd.DataFrame(
        {
            "#": [r.get("source_row_index") for r in tail],
            "adresse": [r.get("input_address") for r in tail],
            "capacité": [_safe_int(r.get("estimated_capacity")) for r in tail],
            "intervalle": [
                f"{_safe_int(r.get('min_capacity'))}-{_safe_int(r.get('max_capacity'))}"
                if r.get("min_capacity") is not None and r.get("max_capacity") is not None
                else ""
                for r in tail
            ],
            "parking à côté": [
                "oui" if (r.get("nearby_osm_parkings_count") or 0) > 0 else "non" for r in tail
            ],
            "méthode": [r.get("method_used") or "" for r in tail],
            "erreur": [r.get("error") or "" for r in tail],
        }
    )

    return md, df, f"{n} / {total}"


def build_app(input_path: Path, ledger_path: Path) -> gr.Blocks:
    with gr.Blocks(title=f"Bulk run · {ledger_path.name}") as demo:
        gr.Markdown(f"# Suivi run `parking-capacity` — bulk\n\nCSV d'entrée : `{input_path}`")
        stats_md = gr.Markdown()
        tail_df = gr.Dataframe(
            label="Dernières adresses traitées (les plus récentes en haut)",
            wrap=True,
            interactive=False,
        )
        with gr.Row():
            counter = gr.Textbox(label="Compteur brut", interactive=False)
            refresh_btn = gr.Button("Rafraîchir maintenant", variant="primary")

        def _refresh():
            return _build_snapshot(input_path, ledger_path)

        # Premier rendu + autorefresh.
        demo.load(_refresh, inputs=None, outputs=[stats_md, tail_df, counter])
        refresh_btn.click(_refresh, inputs=None, outputs=[stats_md, tail_df, counter])

        # Timer Gradio (gr.Timer disponible depuis 4.40).
        timer = gr.Timer(REFRESH_S)
        timer.tick(_refresh, inputs=None, outputs=[stats_md, tail_df, counter])

    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/bulk/adresses_clean.csv"))
    p.add_argument("--ledger", type=Path, default=Path("data/bulk/run_output.csv.progress.jsonl"))
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args(argv)

    app = build_app(args.input, args.ledger)
    app.launch(server_name=args.host, server_port=args.port, show_error=True, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
