"""Ziehung des Gewichtsraums: Momente, Ordnung, Box, naive Ziehung, Determinismus."""

import numpy as np
import pytest

import sma_constants as C
import sma_sampling as sp
from entscheidungsmatrix_generator import SplitMix64


def test_dirichlet_rows_sum_to_one_and_are_positive():
    for alpha in C.ALPHA_CHOICES:
        W = sp.dirichlet(alpha, 500, 5, SplitMix64(3))
        assert W.shape == (500, 5) and (W > 0).all() and W.sum(axis=1) == pytest.approx(np.ones(500))


def test_dirichlet_moments_match_theory():
    m = 5
    for alpha in C.ALPHA_CHOICES:
        W = sp.dirichlet(alpha, 200000, m, SplitMix64(11))
        var = (m - 1) / (m * m * (m * alpha + 1))                      # Var(w_j) = (m - 1) / (m^2 (m alpha + 1))
        assert W.mean(axis=0) == pytest.approx(np.full(m, 1 / m), abs=0.004)
        assert W[:, 0].var() == pytest.approx(var, rel=0.04)


def test_alpha_must_be_one_of_the_documented_values():
    with pytest.raises(ValueError):
        sp.dirichlet(2.0, 10, 3, SplitMix64(1))


def test_the_same_seed_gives_the_same_draws_and_other_seeds_differ():
    a = sp.dirichlet(1.0, 50, 4, SplitMix64(5))
    assert (a == sp.dirichlet(1.0, 50, 4, SplitMix64(5))).all()
    assert not (a == sp.dirichlet(1.0, 50, 4, SplitMix64(6))).all()


def test_ordered_draws_follow_the_given_order_and_the_roc_centroid():
    order = np.array([2, 0, 3, 1, 4])
    W = sp.ordered(1.0, 100000, 5, order, SplitMix64(7))
    assert (W[:, 2] >= W[:, 0]).all() and (W[:, 0] >= W[:, 3]).all() and (W[:, 3] >= W[:, 1]).all() and (W[:, 1] >= W[:, 4]).all()
    roc = np.array([sum(1 / j for j in range(r, 6)) / 5 for r in range(1, 6)])
    assert W.mean(axis=0)[order] == pytest.approx(roc, abs=0.004)


def test_box_stays_inside_the_interval_and_the_simplex():
    c = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    W = sp.box(c, 0.05, 2000, SplitMix64(9))
    assert W.shape == (2000, 5)
    assert W.sum(axis=1) == pytest.approx(np.ones(2000))
    assert (W >= np.maximum(0, c - 0.05) - 1e-12).all() and (W <= np.minimum(1, c + 0.05) + 1e-12).all()
    assert (W >= 0).all()


def test_box_is_centered_on_the_center_when_it_is_not_clipped():
    c = np.array([0.4, 0.3, 0.3])
    W = sp.box(c, 0.1, 40000, SplitMix64(2))
    assert W.mean(axis=0) == pytest.approx(c, abs=0.004)


def test_box_with_one_criterion_is_the_unit_vector():
    assert sp.box(np.array([1.0]), 0.1, 5, SplitMix64(1)).tolist() == [[1.0]] * 5


def test_naive_normalisation_is_too_close_to_equal_weights():
    d = sp.dirichlet(1.0, 100000, 5, SplitMix64(4))[:, 0].var()
    n = sp.naive(100000, 5, SplitMix64(4))[:, 0].var()
    assert n < 0.55 * d


def test_prior_samples_dispatch_and_unknown_prior():
    w = np.array([0.5, 0.3, 0.2])
    rng = SplitMix64(1)
    assert sp.prior_samples("keines", 10, 3, 1.0, w, rng).shape == (10, 3)
    ranked = sp.prior_samples("rang", 200, 3, 1.0, w, SplitMix64(1))
    assert (ranked[:, 0] >= ranked[:, 1]).all() and (ranked[:, 1] >= ranked[:, 2]).all()
    boxed = sp.prior_samples("int10", 200, 3, 1.0, w, SplitMix64(1))
    assert (np.abs(boxed - w) <= 0.10 + 1e-12).all()
    with pytest.raises(ValueError):
        sp.prior_samples("unbekannt", 10, 3, 1.0, w, rng)
