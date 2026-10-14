"""Orakel: SMAA gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12).

1. Verteilung der Ränge: jede Zeile und jede Spalte der Rangakzeptanz summiert zu 1.
2. Pareto-Bezug: eine dominierte Alternative hat Index null für Rang 1 und liegt nie vor ihrem Dominator.
3. Monotonie des Vorwissens: ein kleinerer Gewichtsraum hat höchstens so viele mögliche Beste.
4. m = 2 exakt: Rangakzeptanz gegen die Schnittpunkt-Rechnung auf [0, 1].
5. ROC = Schwerpunkt des geordneten Simplex.
6. LP gegen Dichte: jede Alternative mit Index > 0 ist per LP eine mögliche Beste; die Umkehrung gilt erst für große N.
7. Zentrales Gewicht: im Gewichtsraum, und die Alternative ist dort beste.
"""

import numpy as np
import pytest

import sma_sampling as sp
import sma_smaa as sm
from entscheidungsmatrix_generator import SplitMix64

SEEDS = tuple(range(300000, 300040))


def world(seed, n=8, m=4):
    rng = np.random.default_rng(seed)
    return rng.random((n, m))


def test_rank_acceptability_is_a_doubly_stochastic_matrix():
    for seed in SEEDS:
        V = world(seed)
        b = sm.smaa(V, sp.dirichlet(1.0, 600, 4, SplitMix64(seed))).b
        assert b.sum(axis=0) == pytest.approx(np.ones(8)) and b.sum(axis=1) == pytest.approx(np.ones(8))


def test_a_dominated_alternative_is_never_first_and_never_ahead_of_its_dominator():
    for seed in SEEDS:
        V = world(seed)
        V[3] = V[1] * 0.8                                              # Alternative 3 wird von 1 dominiert (alle Werte positiv)
        W = sp.dirichlet(1.0, 800, 4, SplitMix64(seed))
        R = sm.rank_matrix(V, W)
        assert (R[3] > R[1]).all()
        assert sm.acceptability(R)[3, 0] == 0.0


def test_a_smaller_weight_space_has_no_more_possible_best():
    for seed in SEEDS[:15]:
        V = world(seed, 10, 4)
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(4))
        full = sm.potentially_best(V)
        order = np.argsort(-w)
        ranked = sm.potentially_best(V, order=order)
        boxed = sm.potentially_best(V, np.maximum(0, w - 0.1), np.minimum(1, w + 0.1), None)
        tight = sm.potentially_best(V, np.maximum(0, w - 0.03), np.minimum(1, w + 0.03), None)
        assert (ranked <= full).all() and (boxed <= full).all() and (tight <= boxed).all()
        assert tight.sum() >= 1                                        # die Gewichte w selbst sind zulässig: es gibt einen Besten


def test_two_criteria_match_the_exact_interval_lengths():
    rng = np.random.default_rng(5)
    for _ in range(30):
        V = rng.random((5, 2))
        grid = np.linspace(0, 1, 200001)[1:-1]
        W = np.column_stack([grid, 1 - grid])
        b = sm.acceptability(sm.rank_matrix(V, W))[:, 0]
        # Rechnung ohne die Rangmatrix: Anteil der t, bei denen i das Maximum von t v_i1 + (1 - t) v_i2 ist
        U = V[:, [0]] * grid + V[:, [1]] * (1 - grid)
        direct = np.array([(np.argmax(U, axis=0) == i).mean() for i in range(5)])
        assert b == pytest.approx(direct, abs=1e-9)


def test_roc_weights_are_the_centroid_of_the_ordered_simplex():
    m = 5
    W = np.sort(sp.dirichlet(1.0, 300000, m, SplitMix64(3)), axis=1)[:, ::-1]
    roc = np.array([sum(1 / j for j in range(r, m + 1)) / m for r in range(1, m + 1)])
    assert W.mean(axis=0) == pytest.approx(roc, abs=0.003)
    rank_sum = np.arange(m, 0, -1) / (m * (m + 1) / 2)
    assert np.abs(W.mean(axis=0) - rank_sum).max() > 0.02             # Gegenprobe: die Rangsumme ist es nicht


def test_every_alternative_with_positive_index_is_possible_per_lp_and_large_samples_find_most():
    found, total = 0, 0
    for seed in SEEDS[:12]:
        V = world(seed, 10, 4)
        lp = sm.potentially_best(V)
        b1 = sm.smaa(V, sp.dirichlet(1.0, 20000, 4, SplitMix64(seed))).b[:, 0]
        assert not ((b1 > 0) & ~lp).any()
        found += int(((b1 > 0) & lp).sum())
        total += int(lp.sum())
    assert found <= total and found >= 0.9 * total


def test_central_weights_lie_in_the_space_and_make_the_alternative_best():
    for seed in SEEDS[:20]:
        V = world(seed, 8, 4)
        W = sp.dirichlet(1.0, 3000, 4, SplitMix64(seed))
        res = sm.smaa(V, W)
        for i in range(8):
            if res.b[i, 0] > 0:
                c = res.cw[i]
                assert c.sum() == pytest.approx(1.0) and (c >= 0).all()
                assert np.argmax(V @ c) == i                           # Konvexität: das Mittel der Gewichte, bei denen i beste ist, hat i wieder als beste
