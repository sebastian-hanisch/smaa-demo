"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte (reine Funktionen). Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sma_constants as C
import sma_sampling as sp
import sma_score as sc
import sma_smaa as sm
from entscheidungsmatrix_generator import SplitMix64, make_matrix, to_raw


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    ns: int = C.DEFAULT_NS
    alpha: float = C.DEFAULT_ALPHA
    prior: str = C.DEFAULT_PRIOR
    scale: str = C.DEFAULT_SCALE
    narrow: float = C.DEFAULT_NARROW
    decider: str = C.DEFAULT_DECIDER
    alpha_true: float = C.DEFAULT_TRUE_ALPHA
    seed: int = C.DEFAULT_SEED


def build_case(n: int, m: int, seed: int, narrow: float = 1.0, alpha_true: float = 1.0):
    """(Güte, signierte Rohwerte, verborgene Gewichte) einer Instanz. Kriterium 1 kann gestaucht sein (Anteil `narrow` der Streuung); die Gewichte des Entscheiders
    sind die des Generators (gleichverteilt) oder Dirichlet(alpha_true) aus einem eigenen Strom."""
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    g = np.asarray(x.goodness, dtype=float)
    if narrow != 1.0:
        g = sc.narrow_criterion(g, narrow)
    S = to_raw(g, m) * np.asarray(x.directions, dtype=float)
    w = np.asarray(x.weights, dtype=float) if alpha_true == 1.0 else sp.draw_weights(m, alpha_true, SplitMix64(seed + C.OFFSET_WEIGHTS))
    return g, S, w


def prior_bounds(prior: str, w: np.ndarray):
    """(lo, hi, order) des zulässigen Gewichtsraums unter dem Vorwissen, für den LP."""
    m = len(w)
    if prior in C.PRIOR_DELTA:
        d = C.PRIOR_DELTA[prior]
        return np.maximum(0.0, w - d), np.minimum(1.0, w + d), None
    if prior == "rang":
        return None, None, sc.importance_order(w)
    return None, None, None


@dataclass
class Analysis:
    settings: Settings
    S: np.ndarray
    G: np.ndarray
    w: np.ndarray               # verborgene Gewichte des Entscheiders
    utility: np.ndarray
    V: np.ndarray               # normierte Werte (Skala der Einstellung)
    b: np.ndarray               # n x n Rangakzeptanz
    cw: np.ndarray              # n x m zentrale Gewichte
    er: np.ndarray              # erwarteter Rang
    best: int                   # bester des Entscheiders
    dm_rank: np.ndarray         # Rang beim Entscheider (1 = bester)
    pick: int                   # argmax b1 (Index-Sieger)
    pick_equal: int             # bester bei gleichen Gewichten
    pick_er: int                # kleinster erwarteter Rang
    best_in_top3: bool
    possible_lp: np.ndarray     # exakt (LP) mögliche Beste unter dem Vorwissen
    possible_sample: np.ndarray  # b1 > 0
    efficient: np.ndarray
    W_sub: np.ndarray           # bis zu 1500 gezogene Gewichte (Dreiecksdiagramm)
    winner_sub: np.ndarray


def dm_ranks(u: np.ndarray) -> np.ndarray:
    """Rang 1..n beim Entscheider (Nutzen absteigend; Gleichstand: kleinere Nummer zuerst)."""
    order = np.lexsort((np.arange(len(u)), -u))
    r = np.empty(len(u), dtype=int)
    r[order] = np.arange(1, len(u) + 1)
    return r


def analyse(settings: Settings) -> Analysis:
    s = settings
    g, S, w = build_case(s.n, s.m, s.seed, s.narrow, s.alpha_true)
    u = sc.decider_utility(s.decider, g, S, w)
    V = sc.normalize(S, s.scale)
    W = sp.prior_samples(s.prior, s.ns, s.m, s.alpha, w, SplitMix64(s.seed + C.OFFSET_DRAW))
    R = sm.rank_matrix(V, W)
    b = sm.acceptability(R)
    cw = sm.central_weights(R, W)
    er = sm.expected_rank(b)
    best = int(np.argmax(u))
    pick = int(np.argmax(b[:, 0]))
    lo, hi, order = prior_bounds(s.prior, w)
    top3 = set(np.lexsort((np.arange(s.n), -b[:, 0]))[:3].tolist())
    sub = slice(0, min(1500, s.ns))
    return Analysis(s, S, g, w, u, V, b, cw, er, best, dm_ranks(u), pick, int(np.argmax(V.mean(axis=1))), int(np.argmin(er)), best in top3,
                    sm.potentially_best(V, lo, hi, order), b[:, 0] > 0, sm.pareto_efficient(V), W[sub], np.argmin(R[:, sub], axis=0))


def index_rank(a: Analysis, i: int) -> int:
    """Rang (1 = größter Index Rang 1) der Alternative i; Gleichstand: kleinere Nummer zuerst."""
    order = np.lexsort((np.arange(len(a.b)), -a.b[:, 0]))
    return int(np.flatnonzero(order == i)[0]) + 1


# ----------------------------------------------------------------------------- Experimente

def _top3(b1: np.ndarray) -> set:
    return set(np.lexsort((np.arange(len(b1)), -b1))[:3].tolist())


def experiment_accuracy(deciders=C.DECIDERS, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, ns=C.EXP_NS) -> dict:
    """Experiment 1: Trifft der Index den Besten des Entscheiders (je Entscheider), und was das Vorwissen über die Gewichte bringt (Entscheider mit Rohwerten)."""
    dec = {d: {"argmax": [], "equal": [], "er": [], "top3": [], "rank": []} for d in deciders}
    pri = {p: {"nposs": [], "argmax": [], "maxb": [], "roc": [], "ranksum": [], "equal": []} for p in C.PRIORS}
    for seed in seeds:
        g, S, w = build_case(n, m, seed)
        V = sc.normalize_nominal(S)
        res = sm.smaa(V, sp.dirichlet(1.0, ns, m, SplitMix64(seed + C.OFFSET_DRAW)))
        pick, pick_er, eq = int(np.argmax(res.b[:, 0])), int(np.argmin(res.er)), int(np.argmax(V.mean(axis=1)))
        top3 = _top3(res.b[:, 0])
        for d in deciders:
            u = sc.decider_utility(d, g, S, w)
            best = int(np.argmax(u))
            dec[d]["argmax"].append(pick == best)
            dec[d]["equal"].append(eq == best)
            dec[d]["er"].append(pick_er == best)
            dec[d]["top3"].append(best in top3)
            dec[d]["rank"].append(int(dm_ranks(u)[pick]))
        u_roh = V @ w
        best_roh = int(np.argmax(u_roh))
        order = sc.importance_order(w)
        for p in C.PRIORS:
            r = res if p == "keines" else sm.smaa(V, sp.prior_samples(p, ns, m, 1.0, w, SplitMix64(seed + C.OFFSET_DRAW + 1)))
            pri[p]["nposs"].append(int((r.b[:, 0] > 0).sum()))
            pri[p]["argmax"].append(int(np.argmax(r.b[:, 0])) == best_roh)
            pri[p]["maxb"].append(float(r.b[:, 0].max()))
            if p == "rang":
                pri[p]["roc"].append(int(np.argmax(V @ sc.roc_weights(order))) == best_roh)
                pri[p]["ranksum"].append(int(np.argmax(V @ sc.rank_sum_weights(order))) == best_roh)
                pri[p]["equal"].append(eq == best_roh)
    return {
        "deciders": [{"decider": d, **{k: float(np.mean(v)) for k, v in dec[d].items()}} for d in deciders],
        "priors": [{"prior": p, **{k: (float(np.mean(v)) if v else None) for k, v in pri[p].items()}} for p in C.PRIORS],
    }


def experiment_calibration(alphas=C.ALPHA_CHOICES, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, ns=C.EXP_NS, draws=5) -> dict:
    """Experiment 2: Ist der Index eine Wahrscheinlichkeit? Gewichte des Entscheiders ~ Dirichlet(alpha_wahr), Ziehung ~ Dirichlet(alpha_smaa); Entscheider mit Rohwerten.
    Je Zelle: Fälle, in denen der Index des Siegers mindestens CLAIM_MIN ist, dort der mittlere Index (behauptet) und der Anteil richtiger Aussagen."""
    cells = {(a_t, a_s): {"claimed": [], "right": [], "hit": [], "top3": [], "cases": 0} for a_t in alphas for a_s in alphas}
    same = []
    for seed in seeds:
        g, S, _ = build_case(n, m, seed)
        V = sc.normalize_nominal(S)
        res = {a_s: sm.smaa(V, sp.dirichlet(a_s, ns, m, SplitMix64(seed + C.OFFSET_DRAW))) for a_s in alphas}
        picks = {a_s: int(np.argmax(res[a_s].b[:, 0])) for a_s in alphas}
        same.append(len(set(picks.values())) == 1)
        for a_t in alphas:
            rng = SplitMix64(seed + C.OFFSET_WEIGHTS)
            for _ in range(draws):
                w = sp.draw_weights(m, a_t, rng)
                best = int(np.argmax(V @ w))
                for a_s in alphas:
                    cell = cells[(a_t, a_s)]
                    pick = picks[a_s]
                    top = float(res[a_s].b[pick, 0])
                    cell["cases"] += 1
                    cell["hit"].append(pick == best)
                    cell["top3"].append(best in _top3(res[a_s].b[:, 0]))
                    if top >= C.CLAIM_MIN:
                        cell["claimed"].append(top)
                        cell["right"].append(pick == best)
    out = []
    for (a_t, a_s), c in cells.items():
        out.append({"alpha_true": a_t, "alpha_smaa": a_s, "cases": c["cases"], "n_claims": len(c["claimed"]),
                    "claimed": float(np.mean(c["claimed"])) if c["claimed"] else None, "right": float(np.mean(c["right"])) if c["right"] else None,
                    "hit": float(np.mean(c["hit"])), "top3": float(np.mean(c["top3"]))})
    return {"cells": out, "same": float(np.mean(same))}


def experiment_sampling(seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, sizes=(100, 1000, 10000), n_error=20, n_lp=50, ref_ns=40000) -> dict:
    """Experiment 3: Monte-Carlo-Fehler des Index gegen einen Referenzlauf (die ersten n_error Matrizen) und mögliche Beste per LP gegen die Stichprobe (die ersten n_lp)."""
    err = {k: [] for k in sizes}
    for seed in seeds[:n_error]:
        _, S, _ = build_case(n, m, seed)
        V = sc.normalize_nominal(S)
        ref = sm.smaa(V, sp.dirichlet(1.0, ref_ns, m, SplitMix64(seed + C.OFFSET_REF))).b[:, 0]
        for k in sizes:
            b = sm.smaa(V, sp.dirichlet(1.0, k, m, SplitMix64(seed + C.OFFSET_DRAW))).b[:, 0]
            err[k].append(float(np.abs(b - ref).max()))
    exact, eff, unsup, found = [], [], [], {k: [] for k in sizes}
    missed = {k: [] for k in sizes}
    for seed in seeds[:n_lp]:
        _, S, _ = build_case(n, m, seed)
        V = sc.normalize_nominal(S)
        lp = sm.potentially_best(V)
        ef = sm.pareto_efficient(V)
        exact.append(int(lp.sum()))
        eff.append(int(ef.sum()))
        unsup.append(int((ef & ~lp).sum()))
        for k in sizes:
            got = sm.smaa(V, sp.dirichlet(1.0, k, m, SplitMix64(seed + C.OFFSET_DRAW))).b[:, 0] > 0
            found[k].append(int(got.sum()))
            missed[k].append(int((lp & ~got).sum()))
    return {
        "error": [{"ns": k, "mean": float(np.mean(err[k])), "max": float(np.max(err[k]))} for k in sizes],
        "possible": [{"ns": k, "found": float(np.mean(found[k])), "missed": float(np.mean(missed[k])), "mats_missed": int(sum(v > 0 for v in missed[k]))} for k in sizes],
        "exact": float(np.mean(exact)), "efficient": float(np.mean(eff)), "unsupported": float(np.mean(unsup)), "unsupported_mats": int(sum(v > 0 for v in unsup)),
        "n_error": min(n_error, len(seeds)), "n_lp": min(n_lp, len(seeds)),
    }


def experiment_scale(shares=(1.0, 0.25), seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, ns=C.EXP_NS) -> dict:
    """Experiment 4: Nenn-Spannweite gegen Min-Max, mit und ohne gestauchtes Kriterium 1 (gleiche Ziehungen); dazu die naive Ziehung gegen Dirichlet(1)."""
    rows = []
    for share in shares:
        change, tv, b_nom, b_mm, hit_nom, hit_mm = [], [], [], [], [], []
        for seed in seeds:
            g, S, w = build_case(n, m, seed, share)
            W = sp.dirichlet(1.0, ns, m, SplitMix64(seed + C.OFFSET_DRAW))
            Vn, Vm = sc.normalize_nominal(S), sc.normalize_minmax(S)
            bn, bm = sm.smaa(Vn, W).b[:, 0], sm.smaa(Vm, W).b[:, 0]
            change.append(int(np.argmax(bn)) != int(np.argmax(bm)))
            tv.append(sc.tv_distance(bn, bm))
            lead = int(np.argmax(S[:, 0]))
            b_nom.append(float(bn[lead]))
            b_mm.append(float(bm[lead]))
            best = int(np.argmax(Vn @ w))
            hit_nom.append(int(np.argmax(bn)) == best)
            hit_mm.append(int(np.argmax(bm)) == best)
        rows.append({"share": share, "change": float(np.mean(change)), "tv": float(np.mean(tv)), "b_lead_nom": float(np.mean(b_nom)), "b_lead_mm": float(np.mean(b_mm)),
                     "hit_nom": float(np.mean(hit_nom)), "hit_mm": float(np.mean(hit_mm))})
    ch, tvs, var = [], [], []
    for seed in seeds:
        _, S, _ = build_case(n, m, seed)
        V = sc.normalize_nominal(S)
        bd = sm.smaa(V, sp.dirichlet(1.0, ns, m, SplitMix64(seed + C.OFFSET_DRAW))).b[:, 0]
        Wn = sp.naive(ns, m, SplitMix64(seed + C.OFFSET_NAIVE))
        bn = sm.smaa(V, Wn).b[:, 0]
        ch.append(int(np.argmax(bd)) != int(np.argmax(bn)))
        tvs.append(sc.tv_distance(bd, bn))
    rng = SplitMix64(1)
    var_d = float(sp.dirichlet(1.0, 20000, m, rng)[:, 0].var())
    var_n = float(sp.naive(20000, m, rng)[:, 0].var())
    return {"rows": rows, "naive": {"change": float(np.mean(ch)), "tv": float(np.mean(tvs)), "var_ratio": var_n / var_d}}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_accuracy(acc: dict) -> tuple[str, str]:
    d = acc["deciders"]
    pri = {p["prior"]: p for p in acc["priors"]}
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    lo, hi = min(r["argmax"] for r in d), max(r["argmax"] for r in d)
    elo, ehi = min(r["equal"] for r in d), max(r["equal"] for r in d)
    tlo, thi = min(r["top3"] for r in d), max(r["top3"] for r in d)
    rang, fine = pri["rang"], pri["int05"]
    gain = max(r["argmax"] - r["equal"] for r in d)
    text = (f"Ohne Wissen über die Gewichte trifft der Index-Sieger den Besten des Entscheiders in {pct(lo)} bis {pct(hi)} % der Matrizen, gleiche Gewichte in {pct(elo)} bis {pct(ehi)} %: "
            f"höchstens {de(100 * gain, 1)} Punkte mehr. Der Beste steht in {pct(tlo)} bis {pct(thi)} % unter den ersten drei nach Index. Ist die Rangfolge der Gewichte bekannt, sinkt die Zahl der möglichen Besten "
            f"von {de(pri['keines']['nposs'], 1)} auf {de(rang['nposs'], 1)} und der Sieger trifft in {pct(rang['argmax'])} % (ROC-Gewichte {pct(rang['roc'])} %, Rangsumme {pct(rang['ranksum'])} %); "
            f"bei Gewichten auf ±0,05 sind es {de(fine['nposs'], 1)} mögliche Beste und {pct(fine['argmax'])} %.")
    return "warning", text


def _cell(cells: list[dict], a_true: float, a_smaa: float) -> dict:
    return next(c for c in cells if c["alpha_true"] == a_true and c["alpha_smaa"] == a_smaa)


def verdict_calibration(cal: dict) -> tuple[str, str]:
    cells = cal["cells"]
    ok, tight, wide = _cell(cells, 1.0, 1.0), _cell(cells, 1.0, 5.0), _cell(cells, 5.0, 0.5)
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    text = (f"Bei einem Index des Siegers von mindestens 0,5 steht im Mittel {de(ok['claimed'], 2)} und die Aussage stimmt in {pct(ok['right'])} % der Fälle, wenn die Ziehung zur Verteilung der "
            f"Gewichte passt (α = 1 gegen 1). Zieht man zu eng um gleiche Gewichte (α = 5 gegen 1), steht {de(tight['claimed'], 2)} und es stimmt in {pct(tight['right'])} %; zieht man zu breit "
            f"(α = 0,5 gegen 5), steht {de(wide['claimed'], 2)} und es stimmt in {pct(wide['right'])} %. Der Sieger selbst bleibt bei den drei Verteilungen in {pct(cal['same'])} % der Matrizen derselbe: "
            f"Die Wahrscheinlichkeit hängt an der Verteilung, die Rangfolge kaum.")
    return "warning", text


def verdict_sampling(sam: dict) -> tuple[str, str]:
    big = next(r for r in sam["error"] if r["ns"] == 10000)
    small = next(r for r in sam["error"] if r["ns"] == 100)
    poss = next(r for r in sam["possible"] if r["ns"] == 10000)
    text = (f"Mit 10 000 Ziehungen weicht der Index des Siegers im Mittel um {de(big['mean'], 3)} (höchstens {de(big['max'], 3)}) vom Referenzlauf ab, mit 100 um {de(small['mean'], 3)}. "
            f"Der LP findet im Mittel {de(sam['exact'], 2)} mögliche Beste je Matrix, 10 000 Ziehungen {de(poss['found'], 2)}: Index null heißt „unter 1/N“, nicht „unmöglich“. "
            f"Von {de(sam['efficient'], 2)} effizienten Alternativen sind {de(sam['unsupported'], 2)} für kein Gewicht beste (in {sam['unsupported_mats']} von {sam['n_lp']} Matrizen mindestens eine).")
    return "info", text


def verdict_scale(sc_: dict) -> tuple[str, str]:
    rows = {r["share"]: r for r in sc_["rows"]}
    plain, narrow = rows[1.0], rows[0.25]
    nv = sc_["naive"]
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    text = (f"Streut ein Kriterium nur zu einem Viertel, wechselt der Sieger zwischen Nenn-Spannweite und Min-Max in {pct(narrow['change'])} % der Matrizen (ohne Stauchung {pct(plain['change'])} %); "
            f"der Index des Besten in diesem Kriterium steigt von {de(narrow['b_lead_nom'], 3)} auf {de(narrow['b_lead_mm'], 3)}. Die Trefferquote bleibt ähnlich ({pct(narrow['hit_nom'])} "
            f"gegen {pct(narrow['hit_mm'])} %): Die Skala ändert die Antwort, nicht ihre Güte. Die naive Ziehung (m Gleichverteilte normiert) hat das {de(nv['var_ratio'], 2)}-fache der Varianz "
            f"und ändert den Sieger in {pct(nv['change'])} %.")
    return "info", text
