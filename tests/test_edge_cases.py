"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 1, n = 2, m = 1, eine Ziehung, Gleichstände, Intervall am Rand, identische Alternativen."""

import numpy as np
import pytest

import sma_evaluation as ev
import sma_sampling as sp
import sma_score as sc
import sma_smaa as sm
from entscheidungsmatrix_generator import SplitMix64


def test_single_alternative_is_always_first():
    res = sm.smaa(np.array([[0.3, 0.7]]), sp.dirichlet(1.0, 50, 2, SplitMix64(1)))
    assert res.b.tolist() == [[1.0]] and res.er.tolist() == [1.0] and res.cw[0].sum() == pytest.approx(1.0)
    assert sm.potentially_best(np.array([[0.3, 0.7]])).tolist() == [True]
    assert sm.pareto_efficient(np.array([[0.3, 0.7]])).tolist() == [True]


def test_two_alternatives_cross_at_the_exact_point():
    V = np.array([[1.0, 0.0], [0.0, 1.0]])                            # gleicher Nutzen bei t = 0,5
    W = np.column_stack([np.array([0.25, 0.75, 0.5]), np.array([0.75, 0.25, 0.5])])
    R = sm.rank_matrix(V, W)
    assert R.tolist() == [[1, 0, 0], [0, 1, 1]]                      # beim Gleichstand die kleinere Nummer zuerst
    assert sm.potentially_best(V).tolist() == [True, True]


def test_one_criterion_every_weight_is_one():
    V = np.array([[0.2], [0.9], [0.5]])
    res = sm.smaa(V, np.ones((10, 1)))
    assert res.b[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert sm.potentially_best(V).tolist() == [False, True, False]


def test_a_single_draw():
    res = sm.smaa(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.9, 0.1]]))
    assert res.b.tolist() == [[1.0, 0.0], [0.0, 1.0]] and res.ns == 1


def test_identical_alternatives_split_by_index_and_are_never_strictly_best():
    V = np.tile([0.5, 0.5], (3, 1))
    res = sm.smaa(V, sp.dirichlet(1.0, 100, 2, SplitMix64(2)))
    assert res.b[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert sm.potentially_best(V).tolist() == [False, False, False]  # streng beste gibt es nicht (Marge null)


def test_constant_criterion_under_minmax_has_no_effect():
    S = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    N = sc.normalize_minmax(S)
    assert N[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert sm.smaa(N, np.array([[0.4, 0.6]])).b[:, 0].tolist() == [0.0, 0.0, 1.0]


def test_interval_at_the_boundary_of_the_simplex():
    c = np.array([1.0, 0.0, 0.0])                                     # Eckpunkt: die Box schneidet den Simplex nur nahe der Ecke
    W = sp.box(c, 0.05, 100, SplitMix64(1))
    assert W.shape == (100, 3) and (W[:, 0] >= 0.95 - 1e-12).all() and W.sum(axis=1) == pytest.approx(np.ones(100))
    assert W.min() >= 0.0


def test_analysis_with_the_smallest_matrix_and_the_most_criteria():
    a = ev.analyse(ev.Settings(n=4, m=3, ns=100, prior="int05", seed=2))
    assert a.b.shape == (4, 4) and a.W_sub.shape == (100, 3)
    b = ev.analyse(ev.Settings(n=20, m=8, ns=1000, prior="rang", alpha=0.5, alpha_true=5.0, narrow=0.25, scale="minmax", decider="untergrenze", seed=9))
    assert b.b.shape == (20, 20) and b.b.sum(axis=1) == pytest.approx(np.ones(20))


def test_interval_around_a_point_outside_the_simplex_raises_instead_of_looping():
    with pytest.raises(RuntimeError):
        sp.box(np.array([0.5, 0.2]), 0.0, 10, SplitMix64(1), max_batches=3)       # Mitte summiert nicht zu 1: keine Ziehung wird je angenommen


def test_zero_width_interval_on_a_point_of_the_simplex_returns_that_point():
    W = sp.box(np.array([0.5, 0.5]), 0.0, 4, SplitMix64(1))
    assert W.tolist() == [[0.5, 0.5]] * 4


def test_one_batch_is_enough_when_the_interval_is_wide():
    W = sp.box(np.array([0.4, 0.3, 0.3]), 0.2, 10, SplitMix64(1), max_batches=1)
    assert W.shape == (10, 3)
