"""Skalen, Nutzen des Entscheiders, Rangformel-Gewichte und Kendalls tau-b. Nur numpy; eigene Kopie der kleinen Helfer (kein Import über Repos).
Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

import sma_constants as C
from entscheidungsmatrix_generator import CRITERIA, utility


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def normalize_nominal(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    lo, hi = nominal_bounds(S.shape[1])
    return (S - lo) / (hi - lo)


def normalize_minmax(S: np.ndarray) -> np.ndarray:
    """Min-Max der Stichprobe je Kriterium; konstante Spalte -> lauter Nullen."""
    S = np.asarray(S, dtype=float)
    lo, hi = S.min(axis=0), S.max(axis=0)
    return (S - lo) / np.where(hi > lo, hi - lo, 1.0)


def normalize(S: np.ndarray, scale: str) -> np.ndarray:
    if scale == "nenn":
        return normalize_nominal(S)
    if scale == "minmax":
        return normalize_minmax(S)
    raise ValueError(f"unbekannte Skala: {scale}")


def narrow_criterion(goodness: np.ndarray, share: float, column: int = 0) -> np.ndarray:
    """Staucht die Güte eines Kriteriums um 0,5 auf den Anteil `share` der Streuung: g' = 0,5 + share (g - 0,5). share = 1 ändert nichts."""
    g = np.array(goodness, dtype=float, copy=True)
    g[:, column] = 0.5 + share * (g[:, column] - 0.5)
    return g


def decider_utility(kind: str, G: np.ndarray, S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Nutzen je Alternative für die fünf Entscheider (G Güte, S signierte Rohwerte). Untergrenze: ein Kriterium unter Güte THETA gibt UNDER_FLOOR."""
    if kind == "roh_linear":
        return normalize_nominal(S) @ w
    forms = {"guete_linear": "linear", "guete_konkav": "konkav", "guete_cobb": "cobb", "untergrenze": "linear"}
    if kind not in forms:
        raise ValueError(f"unbekannter Entscheider: {kind}")
    u = utility(np.asarray(G, dtype=float), w, forms[kind])
    if kind == "untergrenze":
        u = np.where((np.asarray(G) >= C.THETA).all(axis=1), u, C.UNDER_FLOOR)
    return u


def roc_weights(order: np.ndarray) -> np.ndarray:
    """Rangzentroid: das r-wichtigste Kriterium bekommt (1/m) Summe_{j=r..m} 1/j. `order[r]` ist das Kriterium mit dem Rang r + 1."""
    m = len(order)
    tail = np.array([sum(1.0 / j for j in range(i, m + 1)) for i in range(1, m + 1)]) / m
    w = np.empty(m)
    w[np.asarray(order)] = tail
    return w


def rank_sum_weights(order: np.ndarray) -> np.ndarray:
    """Rangsumme: das r-wichtigste Kriterium bekommt (m + 1 - r) / (m (m + 1) / 2)."""
    m = len(order)
    x = (m - np.arange(m)).astype(float)
    w = np.empty(m)
    w[np.asarray(order)] = x / x.sum()
    return w


def importance_order(w: np.ndarray) -> np.ndarray:
    """Kriterien nach absteigendem Gewicht; Gleichstand: kleinere Nummer zuerst."""
    return np.lexsort((np.arange(len(w)), -np.asarray(w, dtype=float)))


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Totalvariationsabstand zweier Verteilungen (halbe L1-Norm)."""
    return 0.5 * float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())
