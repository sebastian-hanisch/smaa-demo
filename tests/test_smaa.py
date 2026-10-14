"""SMAA-Einheiten gegen die Handrechnung der Mini-Instanz (tests/conftest.py) und kleine Festwerte."""

import numpy as np
import pytest

import sma_smaa as sm
from conftest import MINI_B, MINI_CENTRAL_T, MINI_V


def test_rank_matrix_on_three_weights_by_hand():
    W = np.array([[0.05, 0.95], [0.5, 0.5], [0.8, 0.2]])             # t = 0,05: c b d a; t = 0,5: b c a d; t = 0,8: a b c d
    R = sm.rank_matrix(MINI_V, W)
    assert R.tolist() == [[3, 2, 0], [1, 0, 1], [0, 1, 2], [2, 3, 3]]


def test_acceptability_on_the_mini_instance_matches_the_exact_areas(mini_V, mini_grid):
    R = sm.rank_matrix(mini_V, mini_grid)
    b = sm.acceptability(R)
    assert b == pytest.approx(MINI_B, abs=2e-5)
    assert b.sum(axis=0) == pytest.approx(np.ones(4)) and b.sum(axis=1) == pytest.approx(np.ones(4))


def test_central_weights_on_the_mini_instance(mini_V, mini_grid):
    R = sm.rank_matrix(mini_V, mini_grid)
    cw = sm.central_weights(R, mini_grid)
    assert cw[:3, 0] == pytest.approx(MINI_CENTRAL_T, abs=1e-4)
    assert cw[:3, 1] == pytest.approx(1.0 - MINI_CENTRAL_T, abs=1e-4)
    assert np.isnan(cw[3]).all()                                      # d ist nie Erster


def test_expected_rank_is_one_based_and_matches_the_hand_value(mini_V, mini_grid):
    res = sm.smaa(mini_V, mini_grid)
    assert res.er == pytest.approx([0.4 + 2 * (0.6 - 9 / 17) + 3 * (9 / 17 - 0.1) + 4 * 0.1, 1.0 * (0.6 - 3 / 7) + 2 * (3 / 7 + 0.4),
                                    3 / 7 + 2 * (9 / 17 - 3 / 7) + 3 * (1 - 9 / 17), 3 * 0.1 + 4 * 0.9], abs=1e-4)
    assert res.ns == len(mini_grid)


def test_equal_utilities_rank_by_index():
    V = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    R = sm.rank_matrix(V, np.array([[0.3, 0.7], [0.9, 0.1]]))
    assert R.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert sm.acceptability(R)[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_potentially_best_on_the_mini_instance(mini_V):
    assert sm.potentially_best(mini_V).tolist() == [True, True, True, False]
    # Vorwissen: Gewicht auf Kriterium 1 mindestens 0,7: nur a bleibt (a ist für t > 0,6 beste); b und c nicht mehr
    assert sm.potentially_best(mini_V, lo=np.array([0.7, 0.0]), hi=np.array([1.0, 1.0])).tolist() == [True, False, False, False]
    # Rangfolge bekannt: Kriterium 2 wichtiger als Kriterium 1 (t <= 0,5): a ist nie beste, b und c schon
    assert sm.potentially_best(mini_V, order=np.array([1, 0])).tolist() == [False, True, True, False]


def test_pareto_efficient_on_the_mini_instance(mini_V):
    assert sm.pareto_efficient(mini_V).tolist() == [True, True, True, False]       # d wird von b und c dominiert
    assert sm.pareto_efficient(np.array([[1.0, 1.0], [1.0, 1.0]])).tolist() == [True, True]       # gleiche Alternativen dominieren sich nicht


def test_non_supported_efficient_alternative_is_never_best():
    """Effizient, aber im Inneren der konvexen Hülle: für kein Gewicht beste (nicht unterstützt)."""
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]])               # der Mittelpunkt 0,4 liegt unter der Geraden 0,5
    assert sm.pareto_efficient(V).tolist() == [True, True, True]
    assert sm.potentially_best(V).tolist() == [True, True, False]


def test_weak_dominance_counts_as_dominated():
    V = np.array([[0.5, 0.5], [0.5, 0.6]])                            # gleich im ersten, besser im zweiten Kriterium
    assert sm.pareto_efficient(V).tolist() == [False, True]
