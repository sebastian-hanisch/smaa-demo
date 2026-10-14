"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import sma_constants as C
from sma_evaluation import Analysis, de, index_rank

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "α": "alpha", "±": "+-", "≥": ">=", "≤": "<=", "≠": "!=", "−": "-", "€": "EUR", "·": "*",
            "★": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Unsichere Gewichte: Rangakzeptanz mit SMAA"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Alternativen, {s.m} Kriterien, {s.ns} Ziehungen aus Dirichlet({de(s.alpha, 1)}); Vorwissen: {C.PRIOR_LABELS[s.prior]}; Skala: {C.SCALE_LABELS[s.scale]}. "
                            f"Entscheider: {C.DECIDER_LABELS[s.decider]}, verborgene Gewichte aus Dirichlet({de(s.alpha_true, 1)}). Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Index-Sieger: {names[a.pick]} ({de(100 * a.b[a.pick, 0], 1)} %). Bester des Entscheiders: {names[a.best]} (Index {de(100 * a.b[a.best, 0], 1)} %, Rang {index_rank(a, a.best)} im Index). "
                            f"Mögliche Beste (LP): {int(a.possible_lp.sum())} von {s.n}, in der Stichprobe: {int(a.possible_sample.sum())}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Alternativen (Index Rang 1, erwarteter Rang, Rang beim Entscheider)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i in np.lexsort((np.arange(s.n), -a.b[:, 0])):
        pdf.cell(0, 5, _s(f"{names[i]}: Index {de(100 * a.b[i, 0], 1)} %, erwarteter Rang {de(a.er[i], 2)}, Rang beim Entscheider {int(a.dm_rank[i])}"
                          f"{' (Bester des Entscheiders)' if i == a.best else ''}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Index ist nur unter der gezogenen Verteilung eine Wahrscheinlichkeit. Der Entscheider ist ein simuliertes Orakel, kein Mensch. "
                            "Quelle: smaa-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
