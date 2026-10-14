"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sma_constants as C
from sma_evaluation import Analysis, de


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def row_order(a: Analysis) -> np.ndarray:
    """Alternativen nach erwartetem Rang aufsteigend (beste oben); Gleichstand: kleinere Nummer zuerst."""
    return np.lexsort((np.arange(len(a.er)), a.er))


def build_heatmap(a: Analysis) -> go.Figure:
    """Rangakzeptanz: Zeilen Alternativen (nach erwartetem Rang), Spalten Ränge, Zelle = Anteil der Ziehungen in Prozent. ★ = Bester des Entscheiders."""
    n = a.settings.n
    names = alt_names(n)
    order = row_order(a)
    labels = [f"{names[i]} ★" if i == a.best else names[i] for i in order]
    z = 100 * a.b[order]
    text = [[("" if v < 0.05 else de(v, 0 if v >= 9.5 else 1)) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[str(r + 1) for r in range(n)], y=labels, text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
                               colorbar=dict(title="%"), hovertemplate="%{y}, Rang %{x}: %{z:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Alternative (nach erwartetem Rang)")
    fig.update_xaxes(title="Rang", side="top")
    return _fixed(fig, max(300, 34 * n + 120))


def results_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    order = np.lexsort((np.arange(a.settings.n), -a.b[:, 0]))
    rows = []
    for i in order:
        rows.append({"Alternative": names[i] + (" ★" if i == a.best else ""), "Index Rang 1": a.b[i, 0], "erwarteter Rang": a.er[i], "Rang beim Entscheider": int(a.dm_rank[i]),
                     "für ein Gewicht beste (LP)": "ja" if a.possible_lp[i] else "nein", "Pareto-effizient": "ja" if a.efficient[i] else "nein"})
    return pd.DataFrame(rows)


def style_results(df: pd.DataFrame):
    return df.style.format({"Index Rang 1": lambda v: f"{de(100 * v, 1)} %", "erwarteter Rang": lambda v: de(v, 2)})


def build_central(a: Analysis, top: int = 3) -> go.Figure:
    """Zentrale Gewichte der drei Alternativen mit dem größten Index (Mittel der Gewichte, bei denen sie Rang 1 haben) neben den verborgenen und den gleichen Gewichten."""
    s = a.settings
    names = alt_names(s.n)
    crit = [f"K{j + 1}" for j in range(s.m)]
    fig = go.Figure()
    order = np.lexsort((np.arange(s.n), -a.b[:, 0]))
    shown = [i for i in order if a.b[i, 0] > 0][:top]
    for rank, i in enumerate(shown):
        fig.add_trace(go.Bar(x=crit, y=a.cw[i], name=f"{names[i]} (Index {de(100 * a.b[i, 0], 1)} %)", marker_color=C.COLOR_CRIT[rank]))
    fig.add_trace(go.Scatter(x=crit, y=a.w, mode="markers", name="verborgene Gewichte des Entscheiders", marker=dict(symbol="diamond", size=11, color="black")))
    fig.add_trace(go.Scatter(x=crit, y=np.full(s.m, 1.0 / s.m), mode="lines", name="gleiche Gewichte", line=dict(dash="dot", color=C.COLOR_LINE)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Gewicht", range=[0, 1])
    return _fixed(fig, 340)


def build_ternary(a: Analysis) -> go.Figure:
    """Nur m = 3: die gezogenen Gewichte im Dreieck, gefärbt nach der Alternative, die bei ihnen Rang 1 hat."""
    names = alt_names(a.settings.n)
    fig = go.Figure()
    win = a.winner_sub
    counts = np.bincount(win, minlength=a.settings.n)
    for rank, i in enumerate(np.argsort(-counts, kind="stable")):
        if counts[i] == 0:
            continue
        sel = win == i
        fig.add_trace(go.Scatterternary(a=a.W_sub[sel, 0], b=a.W_sub[sel, 1], c=a.W_sub[sel, 2], mode="markers", name=names[i], marker=dict(size=4, color=C.COLOR_CRIT[rank % len(C.COLOR_CRIT)]),
                                        hoverinfo="skip"))
    fig.add_trace(go.Scatterternary(a=[a.w[0]], b=[a.w[1]], c=[a.w[2]], mode="markers", name="verborgene Gewichte", marker=dict(symbol="diamond", size=13, color="black")))
    fig.update_layout(ternary=dict(sum=1, aaxis=dict(title="K1"), baxis=dict(title="K2"), caxis=dict(title="K3")))
    return _fixed(fig, 420)


# ----------------------------------------------------------------------------- Experimente

def accuracy_table(acc: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Entscheider": C.DECIDER_SHORT[r["decider"]], "Index-Sieger trifft": f"{de(100 * r['argmax'], 1)} %", "gleiche Gewichte": f"{de(100 * r['equal'], 1)} %",
                          "kleinster erwarteter Rang": f"{de(100 * r['er'], 1)} %", "Bester unter den ersten drei": f"{de(100 * r['top3'], 1)} %",
                          "Rang des Siegers beim Entscheider": de(r["rank"], 2)} for r in acc["deciders"]])


def prior_table(acc: dict) -> pd.DataFrame:
    rows = []
    for r in acc["priors"]:
        rows.append({"Vorwissen": C.PRIOR_LABELS[r["prior"]], "mögliche Beste (Index > 0)": de(r["nposs"], 2), "Index-Sieger trifft": f"{de(100 * r['argmax'], 1)} %",
                     "größter Index (Mittel)": f"{de(100 * r['maxb'], 1)} %",
                     "ROC-Gewichte": "-" if r["roc"] is None else f"{de(100 * r['roc'], 1)} %", "Rangsumme": "-" if r["ranksum"] is None else f"{de(100 * r['ranksum'], 1)} %"})
    return pd.DataFrame(rows)


def build_prior(acc: dict) -> go.Figure:
    pri = acc["priors"]
    x = [C.PRIOR_SHORT[r["prior"]] for r in pri]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[100 * r["argmax"] for r in pri], name="Index-Sieger trifft (%)", marker_color=C.COLOR_CRIT[0]))
    fig.add_trace(go.Scatter(x=x, y=[r["nposs"] for r in pri], name="mögliche Beste (Zahl)", yaxis="y2", mode="lines+markers", line=dict(color=C.COLOR_CRIT[1], width=3)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="mögliche Beste", rangemode="tozero", fixedrange=True))
    fig.update_yaxes(title="Index-Sieger trifft (%)", range=[0, 102])
    fig.update_xaxes(title="Vorwissen über die Gewichte")
    return _fixed(fig, 360)


def calibration_table(cal: dict) -> pd.DataFrame:
    rows = []
    for c in cal["cells"]:
        rows.append({"α wahr": de(c["alpha_true"], 1), "α Ziehung": de(c["alpha_smaa"], 1), "Fälle mit Index ≥ 0,5": f"{c['n_claims']} von {c['cases']}",
                     "behauptet (mittlerer Index)": "-" if c["claimed"] is None else de(c["claimed"], 2),
                     "richtig": "-" if c["right"] is None else f"{de(100 * c['right'], 1)} %", "Sieger trifft (alle Fälle)": f"{de(100 * c['hit'], 1)} %"})
    return pd.DataFrame(rows)


def build_calibration(cal: dict) -> go.Figure:
    cells = [c for c in cal["cells"] if c["claimed"] is not None]
    x = [f"wahr {de(c['alpha_true'], 1)} / Ziehung {de(c['alpha_smaa'], 1)}" for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[100 * c["claimed"] for c in cells], name="behauptet (Index)", marker_color=C.COLOR_CRIT[4]))
    fig.add_trace(go.Bar(x=x, y=[100 * c["right"] for c in cells], name="richtig (Anteil)", marker_color=C.COLOR_CRIT[0]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="% (nur Fälle mit Index ≥ 0,5)", range=[0, 102])
    fig.update_xaxes(tickangle=-30)
    return _fixed(fig, 400)


def sampling_table(sam: dict) -> pd.DataFrame:
    poss = {r["ns"]: r for r in sam["possible"]}
    return pd.DataFrame([{"Ziehungen N": f"{r['ns']:,}".replace(",", " "), "Fehler des Index (Mittel / Maximum)": f"{de(r['mean'], 3)} / {de(r['max'], 3)}",
                          "mögliche Beste in der Stichprobe": de(poss[r["ns"]]["found"], 2), "davon übersehen (gegen LP)": de(poss[r["ns"]]["missed"], 2),
                          f"Matrizen mit Übersehenem (von {sam['n_lp']})": poss[r["ns"]]["mats_missed"]} for r in sam["error"]])


def scale_table(sc_: dict) -> pd.DataFrame:
    rows = [{"Kriterium 1 streut": f"{de(100 * r['share'], 0)} %", "Sieger wechselt": f"{de(100 * r['change'], 1)} %", "Abstand der Indizes (TV)": de(r["tv"], 3),
             "Index des Besten in K1 (Nenn / Min-Max)": f"{de(r['b_lead_nom'], 3)} / {de(r['b_lead_mm'], 3)}",
             "Trefferquote (Nenn / Min-Max)": f"{de(100 * r['hit_nom'], 1)} % / {de(100 * r['hit_mm'], 1)} %"} for r in sc_["rows"]]
    nv = sc_["naive"]
    rows.append({"Kriterium 1 streut": "naive Ziehung statt Dirichlet(1)", "Sieger wechselt": f"{de(100 * nv['change'], 1)} %", "Abstand der Indizes (TV)": de(nv["tv"], 3),
                 "Index des Besten in K1 (Nenn / Min-Max)": f"Varianz je Gewicht {de(nv['var_ratio'], 2)}-fach", "Trefferquote (Nenn / Min-Max)": "-"})
    return pd.DataFrame(rows)
