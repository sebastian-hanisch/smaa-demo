"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler werden immer gezeichnet (α der Ziehung ist beim Intervall-Vorwissen deaktiviert); es gibt deshalb kein KEPT-Muster."""

import random
from dataclasses import dataclass

import streamlit as st

import sma_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "ns_select": SettingSpec("ns", int, C.DEFAULT_NS, choices=C.NS_CHOICES),
    "alpha_select": SettingSpec("al", float, C.DEFAULT_ALPHA, choices=C.ALPHA_CHOICES),
    "prior_select": SettingSpec("pr", str, C.DEFAULT_PRIOR, choices=C.PRIORS),
    "scale_select": SettingSpec("sc", str, C.DEFAULT_SCALE, choices=C.SCALES),
    "narrow_select": SettingSpec("nw", float, C.DEFAULT_NARROW, choices=C.NARROW_CHOICES),
    "decider_select": SettingSpec("dc", str, C.DEFAULT_DECIDER, choices=C.DECIDERS),
    "true_alpha_select": SettingSpec("at", float, C.DEFAULT_TRUE_ALPHA, choices=C.ALPHA_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "m": "m_slider", "ns": "ns_select", "alpha": "alpha_select", "prior": "prior_select", "scale": "scale_select", "narrow": "narrow_select",
               "decider": "decider_select", "alpha_true": "true_alpha_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
