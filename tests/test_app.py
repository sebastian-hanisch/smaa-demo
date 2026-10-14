"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    import sma_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"] and at.slider(key="m_slider").value == preset["m"]
        for key, field in (("ns_select", "ns"), ("alpha_select", "alpha"), ("prior_select", "prior"), ("scale_select", "scale"), ("narrow_select", "narrow"),
                           ("decider_select", "decider"), ("true_alpha_select", "alpha_true")):
            assert at.selectbox(key=key).value == preset[field], (name, key)
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_every_choice_of_every_selectbox_renders_without_exception():
    import sma_constants as C

    for key, choices in (("ns_select", C.NS_CHOICES), ("alpha_select", C.ALPHA_CHOICES), ("prior_select", C.PRIORS), ("scale_select", C.SCALES),
                         ("narrow_select", C.NARROW_CHOICES), ("decider_select", C.DECIDERS), ("true_alpha_select", C.ALPHA_CHOICES)):
        for choice in choices:
            at = _fresh_app()
            at.selectbox(key=key).select(choice).run(timeout=TIMEOUT)
            assert not at.exception, (key, choice, [str(e) for e in at.exception])


def test_alpha_of_the_draw_is_disabled_for_an_interval_prior():
    at = _fresh_app()
    assert not at.selectbox(key="alpha_select").disabled
    at.selectbox(key="prior_select").select("int10").run(timeout=TIMEOUT)
    assert at.selectbox(key="alpha_select").disabled and not at.exception
    at.selectbox(key="prior_select").select("rang").run(timeout=TIMEOUT)
    assert not at.selectbox(key="alpha_select").disabled


def test_three_criteria_show_the_ternary_chart_and_others_do_not():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(3).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Gewichtsraum für m = 3" in md.value for md in at.markdown)
    at.slider(key="m_slider").set_value(4).run(timeout=TIMEOUT)
    assert not any("Gewichtsraum für m = 3" in md.value for md in at.markdown)


def test_extreme_settings_render_without_exception():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(8).run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(20).run(timeout=TIMEOUT)
    at.selectbox(key="prior_select").select("int05").run(timeout=TIMEOUT)
    at.selectbox(key="decider_select").select("untergrenze").run(timeout=TIMEOUT)
    at.selectbox(key="narrow_select").select(0.25).run(timeout=TIMEOUT)
    at.selectbox(key="scale_select").select("minmax").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_smallest_matrix_and_fewest_draws_render_without_exception():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(4).run(timeout=TIMEOUT)
    at.selectbox(key="ns_select").select(100).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_experiments_run_only_after_the_click():
    at = _fresh_app()
    assert "exp_rows" not in at.session_state
    at.button(key="exp_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert set(at.session_state["exp_rows"]) == {"accuracy", "calibration", "sampling", "scale"}


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "9999"
    at.query_params["m"] = "1"
    at.query_params["ns"] = "123"
    at.query_params["pr"] = "orakel"
    at.query_params["al"] = "0.5"
    at.query_params["at"] = "5.0"
    at.query_params["nw"] = "0.25"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 20 and at.slider(key="m_slider").value == 3
    assert at.selectbox(key="ns_select").value == 10000 and at.selectbox(key="prior_select").value == "keines"
    assert at.selectbox(key="alpha_select").value == 0.5 and at.selectbox(key="true_alpha_select").value == 5.0 and at.selectbox(key="narrow_select").value == 0.25
