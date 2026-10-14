"""Skalen, Nutzen des Entscheiders, Rangformeln, tau-b und TV-Abstand."""

import numpy as np
import pytest

import sma_score as sc


def test_nominal_normalisation_uses_the_fixed_ranges():
    lo, hi = sc.nominal_bounds(3)
    assert sc.normalize_nominal(np.array([lo, hi, (lo + hi) / 2])).ravel() == pytest.approx(np.repeat([0.0, 1.0, 0.5], 3))


def test_minmax_normalisation_uses_the_sample_and_a_constant_column_is_zero():
    S = np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
    assert sc.normalize_minmax(S).tolist() == [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]
    assert sc.normalize(S, "minmax").tolist() == sc.normalize_minmax(S).tolist()
    with pytest.raises(ValueError):
        sc.normalize(S, "zscore")


def test_narrow_criterion_contracts_one_column_around_one_half():
    g = np.array([[0.0, 0.2], [1.0, 0.9]])
    out = sc.narrow_criterion(g, 0.25)
    assert out[:, 0].tolist() == pytest.approx([0.375, 0.625]) and out[:, 1].tolist() == [0.2, 0.9]
    assert sc.narrow_criterion(g, 1.0).tolist() == g.tolist()


def test_decider_utilities_by_hand():
    G = np.array([[0.04, 0.9], [0.5, 0.5]])
    S = np.array([[2000.0, 0.0], [0.0, 0.0]])
    w = np.array([0.5, 0.5])
    assert sc.decider_utility("guete_linear", G, S, w).tolist() == pytest.approx([0.47, 0.5])
    assert sc.decider_utility("guete_konkav", G, S, w).tolist() == pytest.approx([0.5 * (0.2 + np.sqrt(0.9)), 0.5 * np.sqrt(0.5) * 2])
    u = sc.decider_utility("untergrenze", G, S, w)
    assert u[0] < -1e8 and u[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        sc.decider_utility("orakel", G, S, w)


def test_rank_formulas_by_hand():
    order = np.array([2, 0, 1])                                       # Kriterium 2 am wichtigsten
    assert sc.roc_weights(order).tolist() == pytest.approx([5 / 18, 2 / 18, 11 / 18])
    assert sc.rank_sum_weights(order).tolist() == pytest.approx([2 / 6, 1 / 6, 3 / 6])
    assert sc.importance_order(np.array([0.2, 0.5, 0.2, 0.1])).tolist() == [1, 0, 2, 3]       # Gleichstand: kleinere Nummer zuerst


def test_kendall_tau_b_and_tv_distance():
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
    assert sc.tv_distance(np.array([0.5, 0.5]), np.array([0.2, 0.8])) == pytest.approx(0.3)
