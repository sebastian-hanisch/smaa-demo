"""Auswertung: Fall der App, Grenzfälle und Experimente auf wenigen Seeds (Aussagen über alle 200 Matrizen stehen in test_claims.py)."""

import numpy as np
import pytest

import sma_constants as C
import sma_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    s = a.settings
    assert a.b.shape == (s.n, s.n) and a.cw.shape == (s.n, s.m) and a.er.shape == (s.n,)
    assert a.b.sum(axis=0) == pytest.approx(np.ones(s.n)) and a.b.sum(axis=1) == pytest.approx(np.ones(s.n))
    assert a.pick == int(np.argmax(a.b[:, 0])) and a.best == int(np.argmax(a.utility))
    assert sorted(a.dm_rank.tolist()) == list(range(1, s.n + 1)) and a.dm_rank[a.best] == 1
    assert (a.possible_sample <= a.possible_lp).all()                    # was die Stichprobe findet, ist per LP möglich
    assert (a.possible_lp <= a.efficient).all()                           # mögliche Beste sind effizient
    assert a.W_sub.shape == (min(1500, s.ns), s.m) and len(a.winner_sub) == len(a.W_sub)


def test_the_same_settings_give_the_same_analysis():
    a, b = ev.analyse(ev.Settings(seed=5)), ev.analyse(ev.Settings(seed=5))
    assert (a.b == b.b).all() and a.pick == b.pick and a.best == b.best


def test_index_rank_orders_by_index_with_ties_by_number():
    a = ev.analyse(ev.Settings(seed=2))
    ranks = sorted(ev.index_rank(a, i) for i in range(a.settings.n))
    assert ranks == list(range(1, a.settings.n + 1)) and ev.index_rank(a, a.pick) == 1


def test_rank_prior_shrinks_the_possible_best_and_contains_the_hidden_weights_best():
    for seed in range(1, 8):
        free = ev.analyse(ev.Settings(seed=seed))
        ranked = ev.analyse(ev.Settings(prior="rang", seed=seed))
        assert ranked.possible_lp.sum() <= free.possible_lp.sum()
        assert ranked.possible_lp[ranked.best]                            # die verborgenen Gewichte sind zulässig, der Beste ist also möglich (Rohwert-Entscheider)


def test_interval_prior_is_centered_on_the_hidden_weights():
    a = ev.analyse(ev.Settings(prior="int05", ns=10000, seed=3))
    assert a.W_sub.shape[0] == 1500
    assert (np.abs(a.W_sub - a.w) <= 0.05 + 1e-12).all()


def test_untergrenze_decider_marks_alternatives_under_the_floor():
    found = False
    for seed in range(1, 40):
        a = ev.analyse(ev.Settings(decider="untergrenze", seed=seed))
        if (a.utility < -1e8).any():
            found = True
            assert a.dm_rank[int(np.argmin(a.utility))] == a.settings.n or (a.utility < -1e8).sum() > 1
            break
    assert found


def test_narrowing_keeps_the_hidden_weights_and_changes_the_scale():
    plain, narrow = ev.analyse(ev.Settings(seed=4)), ev.analyse(ev.Settings(seed=4, narrow=0.25))
    assert (plain.w == narrow.w).all()
    assert narrow.S[:, 0].std() < plain.S[:, 0].std()


def test_alpha_true_other_than_one_uses_its_own_stream():
    a, b = ev.analyse(ev.Settings(seed=4, alpha_true=1.0)), ev.analyse(ev.Settings(seed=4, alpha_true=5.0))
    assert not np.allclose(a.w, b.w) and b.w.sum() == pytest.approx(1.0)


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:6]
    acc = ev.experiment_accuracy(seeds=seeds, ns=500)
    assert len(acc["deciders"]) == len(C.DECIDERS) and len(acc["priors"]) == len(C.PRIORS)
    assert acc["priors"][1]["roc"] is not None and acc["priors"][0]["roc"] is None
    cal = ev.experiment_calibration(seeds=seeds, ns=500, draws=2)
    assert len(cal["cells"]) == 9 and 0.0 <= cal["same"] <= 1.0
    sam = ev.experiment_sampling(seeds=seeds, sizes=(100, 1000, 10000), n_error=3, n_lp=3, ref_ns=2000)
    assert len(sam["error"]) == 3 and len(sam["possible"]) == 3 and sam["n_error"] == 3
    sca = ev.experiment_scale(shares=(1.0, 0.25), seeds=seeds, ns=500)
    assert len(sca["rows"]) == 2 and "naive" in sca


def test_sampling_experiment_error_shrinks_with_the_number_of_draws():
    sam = ev.experiment_sampling(seeds=C.SWEEP_SEEDS[:8], n_error=8, n_lp=2, ref_ns=20000)
    means = [r["mean"] for r in sam["error"]]
    assert means[0] > means[1] > means[2]
