"""Unsichere Gewichte: Rangakzeptanz mit SMAA - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Linie Mehrkriterien-Ranking (Kind von „Gewichte erheben“). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte, das Verfahren nicht.
Statt eines Gewichtsvektors wird der Gewichtsraum gezogen: je Alternative der Anteil der Gewichte, bei denen sie Rang r erreicht (Rangakzeptanz), und ein zentraler Gewichtsvektor.
Kern: sma_smaa.py, sma_sampling.py, sma_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import sma_constants as C
from sma_evaluation import (
    Settings,
    analyse,
    de,
    experiment_accuracy,
    experiment_calibration,
    experiment_sampling,
    experiment_scale,
    index_rank,
    verdict_accuracy,
    verdict_calibration,
    verdict_sampling,
    verdict_scale,
)
from sma_pdf_export import build_pdf
from sma_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from sma_visualization import (
    accuracy_table,
    alt_names,
    build_calibration,
    build_central,
    build_heatmap,
    build_prior,
    build_ternary,
    calibration_table,
    prior_table,
    results_table,
    sampling_table,
    scale_table,
    style_results,
)

st.set_page_config(page_title="Unsichere Gewichte: SMAA – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {"accuracy": experiment_accuracy(), "calibration": experiment_calibration(), "sampling": experiment_sampling(), "scale": experiment_scale()}


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎲 Unsichere Gewichte: Rangakzeptanz mit SMAA")
st.markdown(
    """
**Der Index sagt, wie oft eine Alternative bei den gezogenen Gewichten vorne liegt, aber er ist nur unter der gezogenen Verteilung eine Wahrscheinlichkeit.** Dies ist das elfte Stück der
Linie **Mehrkriterien-Ranking**: Statt eines Gewichtsvektors (Stück 3) wird der ganze **Gewichtsraum** gezogen. Je Alternative entsteht die **Rangakzeptanz** (Anteil der Ziehungen, in
denen sie Rang r erreicht) und ein **zentraler Gewichtsvektor** (Mittel der Gewichte, bei denen sie vorne liegt). Die Demo misst, wie gut der Index den Besten eines simulierten
Entscheiders trifft, was Vorwissen über die Gewichte bringt und woran die Zahlen hängen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert SMAA", expanded=True):
    st.markdown(
        """
1. **Gewichtsraum:** Alle Gewichtsvektoren, die sich zu 1 summieren. Ohne Wissen sind alle gleich plausibel (Dirichlet α = 1); α = 0,5 zieht ungleichere, α = 5 gleichere Gewichte.
2. **Ziehen:** N Gewichtsvektoren; bei jedem wird die gewichtete Summe der normierten Werte gerechnet und die Alternativen werden geordnet.
3. **Rangakzeptanz** b(i, r): Anteil der Ziehungen, in denen Alternative i den Rang r hat; **erwarteter Rang** = Mittel der Ränge. **Zentraler Gewichtsvektor:** Mittel der Gewichte,
   bei denen i Rang 1 hat – das Gewicht, das i am ehesten zum Besten macht.
4. **Vorwissen** verengt den Raum: die Rangfolge der Gewichte (1/m! des Raums) oder ein Intervall um jedes Gewicht. **Mögliche Beste** (LP): für mindestens ein zulässiges Gewicht
   beste; ein Index null in der Stichprobe heißt nur „unter 1/N“.
5. **Annahmen:** Die Kriterienwerte sind gegeben; unsicher sind nur die Gewichte. Der Index ist unter der gezogenen Verteilung eine Wahrscheinlichkeit, sonst ein Anteil.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen geordnet werden.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist; bei m = 3 gibt es ein Dreiecksdiagramm der Ziehungen.")
    narrow = st.selectbox("Streuung von Kriterium 1", C.NARROW_CHOICES, key="narrow_select", format_func=lambda v: f"{de(100 * v, 0)} % der Streuung",
                          help="Wie weit die Alternativen in Kriterium 1 auseinanderliegen; 25 % heißt: fast gleich. Das ändert die Skala, nicht die Alternativen.")
    st.markdown("**SMAA**")
    ns = st.selectbox("Ziehungen N", C.NS_CHOICES, key="ns_select", format_func=lambda v: f"{v:,}".replace(",", " "), help="Wie viele Gewichtsvektoren gezogen werden.")
    prior = st.selectbox("Vorwissen über die Gewichte", C.PRIORS, key="prior_select", format_func=lambda x: C.PRIOR_LABELS[x],
                         help="Was über die verborgenen Gewichte bekannt ist: nichts, nur ihre Rangfolge oder ein Intervall um jedes Gewicht.")
    interval = prior in C.PRIOR_DELTA
    alpha = st.selectbox("Verteilung der Ziehung α", C.ALPHA_CHOICES, key="alpha_select", format_func=lambda v: de(v, 1), disabled=interval,
                         help="Dirichlet-Parameter der Ziehung: 1 gleichverteilt, 0,5 ungleicher, 5 näher an gleichen Gewichten. Bei einem Intervall wirkt er nicht: dort ist die Ziehung gleichverteilt.")
    scale = st.selectbox("Skala der Werte", C.SCALES, key="scale_select", format_func=lambda x: C.SCALE_LABELS[x],
                         help="Worauf die Werte je Kriterium bezogen werden, bevor die Gewichte wirken.")
    st.markdown("**Entscheider (simuliert)**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda x: C.DECIDER_LABELS[x],
                           help="Wie der verborgene Entscheider bewertet. Nur der mit Rohwerten passt zum Modell von SMAA; die anderen sind Modellrisiko.")
    alpha_true = st.selectbox("Verteilung seiner Gewichte α", C.ALPHA_CHOICES, key="true_alpha_select", format_func=lambda v: de(v, 1),
                              help="Aus welcher Dirichlet-Verteilung die verborgenen Gewichte kommen; 1 = gleichverteilt (die Annahme des Generators).")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "ns_select": int(ns), "alpha_select": float(alpha), "prior_select": prior, "scale_select": scale, "narrow_select": float(narrow),
                   "decider_select": decider, "true_alpha_select": float(alpha_true), "seed_input": int(seed)})

settings = Settings(int(n), int(m), int(ns), float(alpha), prior, scale, float(narrow), decider, float(alpha_true), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
b1 = a.b[:, 0]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Index-Sieger", names[a.pick], delta=f"Rang 1 in {de(100 * b1[a.pick], 1)} % der Ziehungen", delta_color="off",
          help="Die Alternative mit dem größten Anteil an Rang 1 unter den gezogenen Gewichten.")
c2.metric("Bester des Entscheiders", names[a.best], delta=f"Index {de(100 * b1[a.best], 1)} %, Rang {index_rank(a, a.best)} im Index", delta_color="off",
          help="Die Alternative mit dem größten Nutzen für die verborgenen Gewichte; das Verfahren kennt sie nicht.")
c3.metric("Mögliche Beste", f"{int(a.possible_lp.sum())} von {settings.n}", delta=f"in der Stichprobe: {int(a.possible_sample.sum())}", delta_color="off",
          help="Alternativen, die für mindestens ein zulässiges Gewicht beste sind (LP, exakt); darunter, wie viele die Stichprobe gefunden hat.")
c4.metric("Genauigkeit der Ziehung", f"{settings.ns:,}".replace(",", " "), delta=f"Standardfehler des Index höchstens {de(50 / np.sqrt(settings.ns), 1)} Punkte", delta_color="off",
          help="Der Standardfehler eines Anteils p aus N Ziehungen ist √(p (1 − p) / N), höchstens 0,5 / √N.")

left, right_col = st.columns([3, 2])
with left:
    st.markdown("**Rangakzeptanz** (Zeilen: Alternativen nach erwartetem Rang; Spalten: Rang; ★ Bester des Entscheiders)")
    st.plotly_chart(build_heatmap(a), width="stretch", key="heatmap_chart")
with right_col:
    st.markdown("**Zentrale Gewichte** der drei Alternativen mit dem größten Index")
    st.plotly_chart(build_central(a), width="stretch", key="central_chart")
    st.caption("Die Balken zeigen, welche Gewichte die Alternative im Mittel zur besten machen; die Rauten sind die verborgenen Gewichte des Entscheiders.")

if settings.m == 3:
    st.markdown("**Gewichtsraum für m = 3:** die gezogenen Gewichte im Dreieck, nach dem Sieger gefärbt (höchstens 1 500 Punkte)")
    st.plotly_chart(build_ternary(a), width="stretch", key="ternary_chart")

st.dataframe(style_results(results_table(a)), hide_index=True, width="stretch")
st.caption("Sortiert nach dem Index für Rang 1. Rang beim Entscheider: 1 = bester nach seinem Nutzen. „Für ein Gewicht beste (LP)“ gilt unter dem gewählten Vorwissen.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="smaa.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel sagt der Index?")
if a.pick == a.best:
    st.info(f"Der Index-Sieger {names[a.pick]} ist hier der Beste des Entscheiders (Index {de(100 * b1[a.pick], 1)} %). Das Verfahren kannte die Gewichte nicht; "
            f"{int(a.possible_lp.sum())} von {settings.n} Alternativen wären für ein Gewicht beste gewesen.")
else:
    st.warning(f"Der Index-Sieger {names[a.pick]} ({de(100 * b1[a.pick], 1)} %) ist nicht der Beste des Entscheiders: Der Beste {names[a.best]} hat einen Index von {de(100 * b1[a.best], 1)} % "
               f"und steht im Index auf Rang {index_rank(a, a.best)}. Ein Index von {de(100 * b1[a.pick], 1)} % heißt: bei so vielen der gezogenen Gewichte liegt {names[a.pick]} vorne, "
               "nicht, dass es für den Entscheider so wahrscheinlich ist.")
if b1[a.pick] >= C.CLAIM_MIN and (settings.alpha != settings.alpha_true) and not interval and settings.prior == "keines":
    st.caption(f"Die Ziehung (α = {de(settings.alpha, 1)}) und die Gewichte des Entscheiders (α = {de(settings.alpha_true, 1)}) stammen aus verschiedenen Verteilungen: "
               "Der Index ist hier keine Wahrscheinlichkeit für den Entscheider.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), 20 Alternativen, 5 Kriterien, 4 000 Ziehungen je Matrix, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 15 Sekunden "
           "und starten erst auf Klick. Standardfehler der Anteile 1,5 bis 3,5 Punkte: Unterschiede unter etwa 5 Punkten sind nicht belastbar.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Trifft der Index den Besten, und was bringt Vorwissen?")
    st.caption("Index-Sieger gegen gleiche Gewichte, je Entscheider; darunter das Vorwissen über die Gewichte (Entscheider mit Rohwerten, Gewichte gleichverteilt).")
    verdict(verdict_accuracy(exp["accuracy"]))
    st.dataframe(accuracy_table(exp["accuracy"]), hide_index=True, width="stretch")
    st.plotly_chart(build_prior(exp["accuracy"]), width="stretch", key="prior_chart")
    st.dataframe(prior_table(exp["accuracy"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Ist der Index eine Wahrscheinlichkeit?")
    st.caption("Die verborgenen Gewichte kommen aus Dirichlet(α wahr), die Ziehung aus Dirichlet(α Ziehung); gezählt werden die Fälle, in denen der Index des Siegers mindestens 0,5 ist: "
               "Was er behauptet gegen den Anteil, in dem der Sieger der Beste ist (Entscheider mit Rohwerten, 5 Gewichtsvektoren je Matrix).")
    st.plotly_chart(build_calibration(exp["calibration"]), width="stretch", key="calibration_chart")
    verdict(verdict_calibration(exp["calibration"]))
    st.dataframe(calibration_table(exp["calibration"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Was die Stichprobe übersieht")
    st.caption(f"Fehler des Index gegen einen Referenzlauf mit 40 000 Ziehungen ({exp['sampling']['n_error']} Matrizen); mögliche Beste per LP gegen die Stichprobe "
               f"({exp['sampling']['n_lp']} Matrizen).")
    verdict(verdict_sampling(exp["sampling"]))
    st.dataframe(sampling_table(exp["sampling"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 4: Hängt es an der Skala?")
    st.caption("Dieselben Ziehungen auf der Nenn-Spannweite und auf Min-Max der Stichprobe, mit und ohne gestauchtes Kriterium 1; dazu die naive Ziehung (m Gleichverteilte normiert).")
    verdict(verdict_scale(exp["scale"]))
    st.dataframe(scale_table(exp["scale"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Verteilung der Ziehung ist die Annahme** | Der Index ist nur unter der Verteilung, aus der die Gewichte kommen, eine Wahrscheinlichkeit; Experiment 2 zeigt die Lücke. Der Anwender kennt diese Verteilung nicht. | Gewichte erheben |
| **Die Kriterienwerte sind gegeben** | Unsichere Werte (SMAA-2 mit Verteilungen, Konfidenzfaktor) sind nicht Teil; die Aussagen gelten für unsichere Gewichte allein. | – |
| **Das Modell ist die gewichtete Summe** | Der Entscheider mit Güte-Nutzen ist ein Modellrisiko und senkt die Trefferquote (Experiment 1). | Kriterien-Wechselwirkung |
| **Das Vorwissen ist unverzerrt** | Das Intervall liegt symmetrisch um die wahren Gewichte; ein Erheber ohne Fehler. Was Urteile kosten, zeigt „Gewichte erheben“. | Gewichte erheben |
| **Eine Stichprobe ist kein Beweis** | Index null heißt „unter 1/N“; der LP findet mögliche Beste, die 10 000 Ziehungen übersehen (Experiment 3). | Aus Beispielen lernen |
| **Der Entscheider ist ein Orakel** | Kein Mensch, der zögert oder seine Meinung ändert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gewichtsraum.** $\mathcal{W} = \{w \in \mathbb{R}^m_{\ge 0} : \sum_j w_j = 1\}$; ohne Wissen gleichverteilt (Dirichlet(1)), allgemein Dirichlet($\alpha$). Vorwissen: eine Teilmenge
$\mathcal{W}' \subseteq \mathcal{W}$ (Rangfolge: $w_{\pi(1)} \ge \dots \ge w_{\pi(m)}$; Intervall: $|w_j - \hat w_j| \le \delta$).

**Rangakzeptanz.** Mit $u(a_i, w) = \sum_j w_j\, n_j(a_i)$ ist $\mathrm{rank}(i, w) = 1 + |\{k : u(a_k, w) > u(a_i, w)\}|$ (Gleichstand: kleinere Nummer zuerst) und
$b_i^r = P_{w \sim \mathcal{W}'}[\mathrm{rank}(i, w) = r] \approx \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}[\mathrm{rank}(i, w^{(t)}) = r]$; Standardfehler $\sqrt{b (1 - b) / N}$.

**Zentraler Gewichtsvektor.** $w_i^c = \mathbb{E}[w \mid \mathrm{rank}(i, w) = 1]$; er liegt im Bereich, in dem $i$ beste ist (der Bereich ist konvex).

**Erwarteter Rang.** $\bar r_i = \sum_r r\, b_i^r$.

**Mögliche Beste.** $i$ ist für ein $w \in \mathcal{W}'$ streng beste genau dann, wenn $\max\{t : (n(a_i) - n(a_k))^\top w \ge t\ \forall k \ne i,\ w \in \mathcal{W}'\} > 0$ (lineares Programm).

**Rangfolge bekannt.** Der Schwerpunkt von $w_{\pi(1)} \ge \dots \ge w_{\pi(m)}$ ist der Rangzentroid $w_{\pi(r)} = \frac{1}{m} \sum_{j=r}^{m} \frac{1}{j}$ (ROC); die Rangsumme
$w_{\pi(r)} \propto m + 1 - r$ ist es nicht.

**Literatur:** SMAA (Lahdelma, Hokkanen, Salminen) ist hier nicht im Original geprüft und deshalb nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
