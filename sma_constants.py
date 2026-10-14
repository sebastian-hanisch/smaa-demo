"""Konstanten: Regler-Grenzen, Verteilungen, Vorwissen, Skalen, Entscheider, feste Annahmen der Experimente und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 20, 10
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 1           # je Preset geprüft (tests/test_presets.py)

NS_CHOICES = (100, 1000, 10000)               # Zahl der gezogenen Gewichtsvektoren
DEFAULT_NS = 10000
ALPHA_CHOICES = (0.5, 1.0, 5.0)               # Dirichlet(alpha): 1 gleichverteilt über dem Simplex, 0,5 ungleicher, 5 näher an gleichen Gewichten
DEFAULT_ALPHA = 1.0
DEFAULT_TRUE_ALPHA = 1.0                      # Verteilung der verborgenen Gewichte des Entscheiders (1 = die des Generators)

PRIORS = ("keines", "rang", "int15", "int10", "int05")
PRIOR_LABELS = {
    "keines": "kein Vorwissen (ganzer Gewichtsraum)",
    "rang": "Rangfolge der Gewichte bekannt",
    "int15": "Gewichte auf ±0,15 bekannt",
    "int10": "Gewichte auf ±0,10 bekannt",
    "int05": "Gewichte auf ±0,05 bekannt",
}
PRIOR_SHORT = {"keines": "keines", "rang": "Rangfolge", "int15": "±0,15", "int10": "±0,10", "int05": "±0,05"}
PRIOR_DELTA = {"int15": 0.15, "int10": 0.10, "int05": 0.05}
DEFAULT_PRIOR = "keines"

SCALES = ("nenn", "minmax")
SCALE_LABELS = {"nenn": "feste Nenn-Spannweite (unabhängig von der Menge)", "minmax": "Min-Max der Stichprobe (hängt von der Menge ab)"}
DEFAULT_SCALE = "nenn"
NARROW_CHOICES = (1.0, 0.5, 0.25)             # Anteil der Streuung, den Kriterium 1 behält (1 = unverändert)
DEFAULT_NARROW = 1.0

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite): passt zum Modell von SMAA",
    "untergrenze": "Linear in der Güte, mit Untergrenze (kein Kriterium unter 0,1)",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear", "untergrenze": "Güte/Untergrenze"}
DEFAULT_DECIDER = "roh_linear"
THETA = 0.1                                   # Untergrenze der Güte des Entscheiders mit Untergrenze
UNDER_FLOOR = -1e9                            # Nutzen einer Alternative unter der Untergrenze

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20
EXP_M = 5
EXP_NS = 4000                                 # Ziehungen je Matrix in den Experimenten
CLAIM_MIN = 0.5                               # Schwelle des Index, ab der die Aussage "der Beste" als behauptet gilt
OFFSET_DRAW = 104729                          # Strom der Ziehungen: SplitMix64(seed + OFFSET_DRAW)
OFFSET_WEIGHTS = 7919                         # Strom der verborgenen Gewichte (nur bei alpha_wahr != 1)
OFFSET_REF = 15485863                         # Strom des Referenzlaufs im Stichprobenexperiment
OFFSET_NAIVE = 32452843                       # Strom der naiven Ziehung

COLOR_BEST = "#2F6B65"
COLOR_WRONG = "#c0392b"
COLOR_LINE = "rgba(80,80,80,0.7)"
COLOR_CRIT = ("#4c78a8", "#D68A2E", "#2F6B65", "#7a5195", "#8A96A6", "#c0392b", "#59a14f", "#b07aa1")

PRESETS = {
    "Standardfall (kein Vorwissen, α = 1)": {"n": DEFAULT_N, "m": DEFAULT_M, "ns": 10000, "alpha": 1.0, "prior": "keines", "scale": "nenn", "narrow": 1.0, "decider": "roh_linear",
                                            "alpha_true": 1.0, "seed": DEFAULT_SEED},
    "Rangfolge bekannt": {"n": 10, "m": 5, "ns": 10000, "alpha": 1.0, "prior": "rang", "scale": "nenn", "narrow": 1.0, "decider": "roh_linear", "alpha_true": 1.0, "seed": 1},
    "Falsche Verteilung": {"n": 10, "m": 5, "ns": 10000, "alpha": 5.0, "prior": "keines", "scale": "nenn", "narrow": 1.0, "decider": "roh_linear", "alpha_true": 0.5, "seed": 1},
    "Enges Kriterium": {"n": 10, "m": 5, "ns": 10000, "alpha": 1.0, "prior": "keines", "scale": "minmax", "narrow": 0.25, "decider": "roh_linear", "alpha_true": 1.0, "seed": 1},
    "Zu wenig Ziehungen": {"n": 10, "m": 5, "ns": 100, "alpha": 1.0, "prior": "keines", "scale": "nenn", "narrow": 1.0, "decider": "roh_linear", "alpha_true": 1.0, "seed": 8},
}
PRESET_HELP = {
    "Standardfall (kein Vorwissen, α = 1)": "10 Alternativen, 5 Kriterien, Gewichte gleichverteilt gezogen: mehrere Alternativen sind für ein Gewicht beste, keine hat eine klare Spitze.",
    "Rangfolge bekannt": "Die Rangfolge der Gewichte ist bekannt: der Gewichtsraum schrumpft auf 1/120, die Zahl der möglichen Besten sinkt und der Index-Sieger trifft deutlich öfter.",
    "Falsche Verteilung": "Die Ziehung ist zu eng um gleiche Gewichte (α = 5), die wahren Gewichte sind ungleich (α = 0,5): der Index behauptet mehr, als stimmt.",
    "Enges Kriterium": "Ein Kriterium streut nur zu einem Viertel: Nenn-Spannweite und Min-Max geben ihm ein sehr verschiedenes Gewicht in der Ziehung, der Sieger wechselt.",
    "Zu wenig Ziehungen": "Nur 100 Ziehungen: Die Indizes wackeln um mehrere Punkte, ein kleiner Index kann null werden, obwohl die Alternative möglich ist.",
}
