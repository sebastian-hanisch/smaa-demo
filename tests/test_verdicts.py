"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Auswahl der Zellen steht hier."""

import sma_constants as C
import sma_evaluation as ev


def _acc(**over):
    deciders = []
    for i, d in enumerate(C.DECIDERS):
        deciders.append({"decider": d, "argmax": 0.5 + 0.01 * i, "equal": 0.45 + 0.01 * i, "er": 0.5, "top3": 0.8 + 0.01 * i, "rank": 2.5})
    priors = [{"prior": p, "nposs": 9.0, "argmax": 0.6, "maxb": 0.5, "roc": None, "ranksum": None, "equal": None} for p in C.PRIORS]
    priors[1].update({"nposs": 3.0, "argmax": 0.835, "roc": 0.845, "ranksum": 0.76, "equal": 0.56})
    priors[4].update({"nposs": 1.7, "argmax": 0.99})
    acc = {"deciders": deciders, "priors": priors}
    acc.update(over)
    return acc


def test_verdict_accuracy_names_range_and_prior_effect():
    kind, text = ev.verdict_accuracy(_acc())
    assert kind == "warning"
    assert "in 50,0 bis 54,0 % der Matrizen, gleiche Gewichte in 45,0 bis 49,0 %" in text
    assert "höchstens 5,0 Punkte mehr" in text
    assert "in 80,0 bis 84,0 % unter den ersten drei" in text
    assert "von 9,0 auf 3,0 und der Sieger trifft in 83,5 % (ROC-Gewichte 84,5 %, Rangsumme 76,0 %)" in text
    assert "bei Gewichten auf ±0,05 sind es 1,7 mögliche Beste und 99,0 %" in text


def _cal(over=None):
    over = over or {}
    cells = []
    for a_t in C.ALPHA_CHOICES:
        for a_s in C.ALPHA_CHOICES:
            cells.append({"alpha_true": a_t, "alpha_smaa": a_s, "cases": 1000, "n_claims": 400, "claimed": 0.5, "right": 0.5, "hit": 0.5, "top3": 0.8})
    for c in cells:
        key = (c["alpha_true"], c["alpha_smaa"])
        if key in over:
            c.update(over[key])
    return {"cells": cells, "same": 0.78}


def test_verdict_calibration_picks_the_matched_tight_and_wide_cells():
    cal = _cal({(1.0, 1.0): {"claimed": 0.677, "right": 0.691}, (1.0, 5.0): {"claimed": 0.786, "right": 0.557}, (5.0, 0.5): {"claimed": 0.602, "right": 0.884}})
    kind, text = ev.verdict_calibration(cal)
    assert kind == "warning"
    assert "steht im Mittel 0,68 und die Aussage stimmt in 69,1 % der Fälle" in text
    assert "(α = 5 gegen 1), steht 0,79 und es stimmt in 55,7 %" in text
    assert "(α = 0,5 gegen 5), steht 0,60 und es stimmt in 88,4 %" in text
    assert "in 78,0 % der Matrizen derselbe" in text


def test_verdict_sampling_names_error_exact_count_and_unsupported():
    sam = {"error": [{"ns": 100, "mean": 0.055, "max": 0.11}, {"ns": 1000, "mean": 0.017, "max": 0.04}, {"ns": 10000, "mean": 0.006, "max": 0.013}],
           "possible": [{"ns": 100, "found": 6.5, "missed": 4.6, "mats_missed": 50}, {"ns": 1000, "found": 8.4, "missed": 2.7, "mats_missed": 47},
                        {"ns": 10000, "found": 9.44, "missed": 1.62, "mats_missed": 41}],
           "exact": 11.06, "efficient": 14.06, "unsupported": 3.0, "unsupported_mats": 50, "n_error": 20, "n_lp": 50}
    kind, text = ev.verdict_sampling(sam)
    assert kind == "info"
    assert "im Mittel um 0,006 (höchstens 0,013) vom Referenzlauf ab, mit 100 um 0,055" in text
    assert "11,06 mögliche Beste je Matrix, 10 000 Ziehungen 9,44" in text
    assert "Von 14,06 effizienten Alternativen sind 3,00 für kein Gewicht beste (in 50 von 50 Matrizen mindestens eine)" in text


def test_verdict_scale_names_both_shares_and_the_naive_draw():
    sc_ = {"rows": [{"share": 1.0, "change": 0.065, "tv": 0.037, "b_lead_nom": 0.169, "b_lead_mm": 0.166, "hit_nom": 0.59, "hit_mm": 0.595},
                    {"share": 0.25, "change": 0.27, "tv": 0.221, "b_lead_nom": 0.086, "b_lead_mm": 0.165, "hit_nom": 0.595, "hit_mm": 0.58}],
           "naive": {"change": 0.075, "tv": 0.123, "var_ratio": 0.4752}}
    kind, text = ev.verdict_scale(sc_)
    assert kind == "info"
    assert "in 27,0 % der Matrizen (ohne Stauchung 6,5 %)" in text
    assert "steigt von 0,086 auf 0,165" in text and "(59,5 gegen 58,0 %)" in text
    assert "das 0,48-fache der Varianz" in text
    assert "ändert den Sieger in 7,5 %" in text
