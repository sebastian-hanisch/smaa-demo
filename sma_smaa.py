"""SMAA: Rangmatrix der Ziehungen, Rangakzeptanz, zentrale Gewichte, erwarteter Rang, mögliche Beste (LP) und Pareto-Effizienz. Kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


def rank_matrix(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rang 0 .. n-1 (0 = bester) jeder Alternative (Zeile) in jeder Ziehung (Spalte) nach der gewichteten Summe V @ w. Gleicher Nutzen: kleinere Nummer zuerst."""
    V = np.asarray(V, dtype=float)
    U = V @ np.asarray(W, dtype=float).T
    order = np.argsort(-U, axis=0, kind="stable")
    ranks = np.empty_like(order)
    ranks[order, np.arange(U.shape[1])[None, :]] = np.arange(V.shape[0])[:, None]
    return ranks


def acceptability(R: np.ndarray) -> np.ndarray:
    """b[i, r] = Anteil der Ziehungen, in denen Alternative i den Rang r (0 = bester) hat."""
    n, ns = R.shape
    counts = np.bincount((np.arange(n)[:, None] * n + R).ravel(), minlength=n * n)
    return counts.reshape(n, n) / ns


def central_weights(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Mittel der Gewichte, bei denen die Alternative Rang 1 hat; NaN, wenn sie nie Rang 1 hat."""
    n = R.shape[0]
    W = np.asarray(W, dtype=float)
    best = np.argmin(R, axis=0)
    sums = np.zeros((n, W.shape[1]))
    np.add.at(sums, best, W)
    cnt = np.bincount(best, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / cnt[:, None]                                    # 0 / 0 = NaN: nie Erster


def expected_rank(b: np.ndarray) -> np.ndarray:
    """Erwarteter Rang (1 = bester) aus den Indizes."""
    return b @ np.arange(1, b.shape[1] + 1)


@dataclass
class Smaa:
    b: np.ndarray          # n x n Rangakzeptanz
    cw: np.ndarray         # n x m zentrale Gewichte (NaN bei b1 = 0)
    er: np.ndarray         # n erwarteter Rang
    ns: int


def smaa(V: np.ndarray, W: np.ndarray) -> Smaa:
    R = rank_matrix(V, W)
    b = acceptability(R)
    return Smaa(b, central_weights(R, W), expected_rank(b), R.shape[1])


def pareto_efficient(V: np.ndarray) -> np.ndarray:
    """Alternativen, die von keiner anderen dominiert werden (mindestens so gut in allen, in einem besser)."""
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        dom = ((V >= V[i]).all(axis=1)) & ((V > V[i]).any(axis=1))
        eff[i] = not dom.any()
    return eff


def potentially_best(V: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None, order: np.ndarray | None = None, tol: float = 1e-9) -> np.ndarray:
    """Exakt per LP: Alternative i ist für mindestens ein zulässiges Gewicht streng bester, wenn max t mit (v_i - v_j) w >= t für alle j, w im Simplex, t > tol.
    Zulässig: w >= lo (Standard 0), w <= hi (Standard 1), optional w[order[0]] >= w[order[1]] >= ... (Rangfolge bekannt)."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    if n == 1:
        return np.ones(1, dtype=bool)
    lo_ = np.zeros(m) if lo is None else np.asarray(lo, dtype=float)
    hi_ = np.ones(m) if hi is None else np.asarray(hi, dtype=float)
    bounds = [(float(lo_[j]), float(hi_[j])) for j in range(m)] + [(None, None)]
    ord_rows = []
    if order is not None:
        for a, c in zip(order[:-1], order[1:]):
            row = np.zeros(m + 1)
            row[a], row[c] = -1.0, 1.0                       # w[c] - w[a] <= 0
            ord_rows.append(row)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        D = V[i] - np.delete(V, i, axis=0)
        A = np.hstack([-D, np.ones((n - 1, 1))])
        b_ub = np.zeros(n - 1)
        if ord_rows:
            A = np.vstack([A, np.array(ord_rows)])
            b_ub = np.concatenate([b_ub, np.zeros(len(ord_rows))])
        c = np.zeros(m + 1)
        c[-1] = -1.0
        res = linprog(c, A_ub=A, b_ub=b_ub, A_eq=np.hstack([np.ones((1, m)), np.zeros((1, 1))]), b_eq=[1.0], bounds=bounds, method="highs")
        out[i] = res.status == 0 and -res.fun > tol
    return out
