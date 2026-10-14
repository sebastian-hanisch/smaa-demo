"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import numpy as np

import sma_constants as C
import sma_evaluation as ev
import sma_presets as P

STANDARD = "Standardfall (kein Vorwissen, α = 1)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["ns"] in C.NS_CHOICES and p["alpha"] in C.ALPHA_CHOICES and p["alpha_true"] in C.ALPHA_CHOICES, name
        assert p["prior"] in C.PRIORS and p["scale"] in C.SCALES and p["narrow"] in C.NARROW_CHOICES and p["decider"] in C.DECIDERS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.PRIOR_LABELS) == set(C.PRIORS) == set(C.PRIOR_SHORT)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT)
    assert set(C.SCALE_LABELS) == set(C.SCALES)
    assert set(C.PRIOR_DELTA) == {p for p in C.PRIORS if p.startswith("int")}


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis(STANDARD)
    assert a.pick != a.best and a.possible_lp.sum() >= 8 and a.b[a.pick, 0] < 0.4         # keine klare Spitze, viele mögliche Beste

    a = _analysis("Rangfolge bekannt")
    free = _analysis("Rangfolge bekannt", prior="keines")
    assert a.pick == a.best and free.pick != free.best
    assert a.possible_lp.sum() <= 3 and free.possible_lp.sum() >= 8 and a.b[a.pick, 0] > 0.7

    a = _analysis("Falsche Verteilung")
    assert a.b[a.pick, 0] >= C.CLAIM_MIN and a.pick != a.best

    a = _analysis("Enges Kriterium")
    nominal = _analysis("Enges Kriterium", scale="nenn")
    assert a.pick != nominal.pick and nominal.pick == nominal.best

    a = _analysis("Zu wenig Ziehungen")
    big = _analysis("Zu wenig Ziehungen", ns=10000)
    assert int(a.possible_lp.sum()) - int(a.possible_sample.sum()) >= int(big.possible_lp.sum()) - int(big.possible_sample.sum()) + 2
    assert np.abs(a.b[:, 0] - big.b[:, 0]).max() > 0.03
