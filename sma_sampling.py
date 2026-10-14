"""Ziehung des Gewichtsraums: Dirichlet(alpha) über dem Simplex, geordnet (Rangfolge bekannt), Intervall (Box um die Gewichte), naive Normierung.

Zufall kommt aus SplitMix64 (plattformunabhängig). Gamma(0,5) = Z^2 / 2, Gamma(1) = Exponential, Gamma(5) = Summe von fünf Exponentialverteilten."""

from __future__ import annotations

import numpy as np

import sma_constants as C
from entscheidungsmatrix_generator import SplitMix64


def gamma_draws(alpha: float, size: int, rng: SplitMix64) -> np.ndarray:
    if alpha == 0.5:
        return rng.normal(size) ** 2 / 2.0
    if alpha == 1.0:
        return -np.log1p(-rng.uniform(size))
    if alpha == 5.0:
        return (-np.log1p(-rng.uniform(5 * size))).reshape(5, size).sum(axis=0)
    raise ValueError(f"alpha muss 0,5, 1 oder 5 sein, nicht {alpha}")


def dirichlet(alpha: float, ns: int, m: int, rng: SplitMix64) -> np.ndarray:
    """ns Gewichtsvektoren der Länge m aus Dirichlet(alpha) (alpha = 1: gleichverteilt über dem Simplex)."""
    g = gamma_draws(alpha, ns * m, rng).reshape(ns, m)
    return g / g.sum(axis=1, keepdims=True)


def draw_weights(m: int, alpha: float, rng: SplitMix64) -> np.ndarray:
    """Ein Gewichtsvektor (die verborgenen Gewichte des Entscheiders)."""
    return dirichlet(alpha, 1, m, rng)[0]


def ordered(alpha: float, ns: int, m: int, order: np.ndarray, rng: SplitMix64) -> np.ndarray:
    """Gewichte, deren Rangfolge die des Entscheiders ist: Dirichlet-Ziehungen absteigend sortiert und dem Kriterium mit Rang r zugeordnet; exakt gleichverteilt
    (für alpha = 1) über dem Teilbereich w[order[0]] >= w[order[1]] >= ... (1/m! des Simplex)."""
    W = -np.sort(-dirichlet(alpha, ns, m, rng), axis=1)
    out = np.empty_like(W)
    out[:, np.asarray(order)] = W
    return out


def box(center: np.ndarray, delta: float, ns: int, rng: SplitMix64, max_batches: int = 2000) -> np.ndarray:
    """Gleichverteilt über dem Simplex innerhalb der Box |w_j - center_j| <= delta: die ersten m - 1 Gewichte gleichverteilt in der Box, das letzte ergibt sich;
    zurückgewiesen wird, was das letzte außerhalb seiner Grenzen oder unter null legt."""
    c = np.asarray(center, dtype=float)
    m = len(c)
    lo = np.maximum(0.0, c - delta)
    hi = np.minimum(1.0, c + delta)
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(max_batches):
        batch = max(4 * ns, 1000)
        u = lo[: m - 1] + (hi - lo)[: m - 1] * rng.uniform(batch * (m - 1)).reshape(batch, m - 1)
        last = 1.0 - u.sum(axis=1)
        ok = (last >= lo[m - 1]) & (last <= hi[m - 1])
        if ok.any():
            kept.append(np.hstack([u[ok], last[ok, None]]))
            total += int(ok.sum())
        if total >= ns:
            return np.vstack(kept)[:ns]
    raise RuntimeError("Intervall trifft den Simplex kaum")


def naive(ns: int, m: int, rng: SplitMix64) -> np.ndarray:
    """Der häufige Fehler: m Gleichverteilte normieren. Das ist nicht gleichverteilt über dem Simplex (zu nahe an gleichen Gewichten)."""
    u = rng.uniform(ns * m).reshape(ns, m)
    return u / u.sum(axis=1, keepdims=True)


def prior_samples(prior: str, ns: int, m: int, alpha: float, true_w: np.ndarray, rng: SplitMix64) -> np.ndarray:
    """Gezogene Gewichte unter dem Vorwissen `prior`. Bei einem Intervall sind die Gewichte gleichverteilt in der Box (alpha wirkt dort nicht)."""
    if prior == "keines":
        return dirichlet(alpha, ns, m, rng)
    if prior == "rang":
        order = np.lexsort((np.arange(m), -np.asarray(true_w, dtype=float)))
        return ordered(alpha, ns, m, order, rng)
    if prior in C.PRIOR_DELTA:
        return box(true_w, C.PRIOR_DELTA[prior], ns, rng)
    raise ValueError(f"unbekanntes Vorwissen: {prior}")
