"""Jede Zahl der README, gemessen mit den eigenen Experimenten der Demo (200 feste Matrizen, SWEEP_SEEDS). Die Werte sind deterministisch (SplitMix64)."""

import pytest

import sma_constants as C
import sma_evaluation as ev


@pytest.fixture(scope="module")
def exp():
    return {"accuracy": ev.experiment_accuracy(), "calibration": ev.experiment_calibration(), "sampling": ev.experiment_sampling(), "scale": ev.experiment_scale()}


def test_hit_rates_by_decider(exp):
    rows = {r["decider"]: r for r in exp["accuracy"]["deciders"]}
    assert [rows[d]["argmax"] for d in C.DECIDERS] == pytest.approx([0.47, 0.525, 0.55, 0.59, 0.515])
    assert [rows[d]["equal"] for d in C.DECIDERS] == pytest.approx([0.455, 0.52, 0.565, 0.56, 0.52])
    assert [rows[d]["er"] for d in C.DECIDERS] == pytest.approx([0.45, 0.515, 0.555, 0.565, 0.51])
    assert [rows[d]["top3"] for d in C.DECIDERS] == pytest.approx([0.79, 0.83, 0.835, 0.91, 0.775])
    assert rows["roh_linear"]["rank"] == pytest.approx(2.445)
    assert max(rows[d]["argmax"] - rows[d]["equal"] for d in C.DECIDERS) == pytest.approx(0.03)


def test_prior_knowledge_shrinks_the_possible_best_and_raises_the_hit_rate(exp):
    rows = {r["prior"]: r for r in exp["accuracy"]["priors"]}
    assert [rows[p]["nposs"] for p in C.PRIORS] == pytest.approx([8.91, 2.91, 3.58, 2.605, 1.705])
    assert [rows[p]["argmax"] for p in C.PRIORS] == pytest.approx([0.59, 0.835, 0.875, 0.93, 0.99])
    assert rows["rang"]["roc"] == pytest.approx(0.845) and rows["rang"]["ranksum"] == pytest.approx(0.76) and rows["rang"]["equal"] == pytest.approx(0.56)
    assert rows["rang"]["maxb"] == pytest.approx(0.825, abs=0.002)


def test_calibration_depends_on_the_distribution_not_the_winner(exp):
    cells = {(c["alpha_true"], c["alpha_smaa"]): c for c in exp["calibration"]["cells"]}
    ok, tight, wide = cells[(1.0, 1.0)], cells[(1.0, 5.0)], cells[(5.0, 0.5)]
    assert (ok["claimed"], ok["right"]) == pytest.approx((0.674, 0.679), abs=0.0006)
    assert (tight["claimed"], tight["right"]) == pytest.approx((0.786, 0.545), abs=0.0006)
    assert (wide["claimed"], wide["right"]) == pytest.approx((0.602, 0.923), abs=0.0006)
    assert (cells[(0.5, 5.0)]["claimed"], cells[(0.5, 5.0)]["right"]) == pytest.approx((0.786, 0.436), abs=0.0006)
    assert [cells[(a, a)]["right"] - cells[(a, a)]["claimed"] for a in C.ALPHA_CHOICES] == pytest.approx([-0.002, 0.005, -0.017], abs=0.0015)
    assert exp["calibration"]["same"] == pytest.approx(0.78)
    assert (ok["n_claims"], tight["n_claims"], wide["n_claims"]) == (495, 870, 310)


def test_sampling_error_and_what_the_sample_misses(exp):
    sam = exp["sampling"]
    assert [r["mean"] for r in sam["error"]] == pytest.approx([0.0547, 0.0173, 0.0060], abs=0.0001)
    assert [r["max"] for r in sam["error"]] == pytest.approx([0.1107, 0.0407, 0.0130], abs=0.0001)
    assert sam["exact"] == pytest.approx(11.06) and sam["efficient"] == pytest.approx(14.06) and sam["unsupported"] == pytest.approx(3.0)
    assert sam["unsupported_mats"] == 50 and sam["n_lp"] == 50 and sam["n_error"] == 20
    assert [r["found"] for r in sam["possible"]] == pytest.approx([6.48, 8.38, 9.44])
    assert [r["missed"] for r in sam["possible"]] == pytest.approx([4.58, 2.68, 1.62])
    assert [r["mats_missed"] for r in sam["possible"]] == [50, 47, 41]


def test_scale_changes_the_winner_for_a_narrow_criterion(exp):
    rows = {r["share"]: r for r in exp["scale"]["rows"]}
    assert (rows[1.0]["change"], rows[0.25]["change"]) == pytest.approx((0.065, 0.27))
    assert (rows[1.0]["tv"], rows[0.25]["tv"]) == pytest.approx((0.037, 0.221), abs=0.0005)
    assert (rows[0.25]["b_lead_nom"], rows[0.25]["b_lead_mm"]) == pytest.approx((0.086, 0.165), abs=0.0006)
    assert (rows[0.25]["hit_nom"], rows[0.25]["hit_mm"]) == pytest.approx((0.595, 0.58))
    nv = exp["scale"]["naive"]
    assert (nv["change"], nv["var_ratio"]) == pytest.approx((0.075, 0.475), abs=0.001)
    assert nv["tv"] == pytest.approx(0.123, abs=0.0006)


def test_standard_error_of_an_index_follows_the_binomial(exp):
    """Der größte mittlere Fehler bei N = 10 000 liegt unter dem Doppelten des Standardfehlers eines Anteils 0,5."""
    big = next(r for r in exp["sampling"]["error"] if r["ns"] == 10000)
    assert big["mean"] < 2 * (0.25 / 10000) ** 0.5
