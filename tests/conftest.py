"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

m = 2, das Gewicht t auf Kriterium 1 ist gleichverteilt in [0, 1] (w = (t, 1 - t)); normierte Werte a = (1; 0), b = (0,6; 0,6), c = (0,2; 0,9), d = (0,1; 0,1).
Nutzen: a = t, b = 0,6, c = 0,9 - 0,7 t, d = 0,1. Schnittpunkte: a = d bei t = 0,1; b = c bei t = 3/7; a = c bei t = 9/17; a = b bei t = 0,6.
Reihenfolge je Bereich: t < 0,1: c b d a; 0,1 < t < 3/7: c b a d; 3/7 < t < 9/17: b c a d; 9/17 < t < 0,6: b a c d; t > 0,6: a b c d.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MINI_V = np.array([[1.0, 0.0], [0.6, 0.6], [0.2, 0.9], [0.1, 0.1]])

# Rangakzeptanz b[i, r] (Zeilen a b c d; Spalten Rang 1 bis 4), exakt aus den Längen der Bereiche
MINI_B = np.array([
    [0.4, 0.6 - 9 / 17, 9 / 17 - 0.1, 0.1],                      # a: Rang 1 für t > 0,6; Rang 2 für 9/17 < t < 0,6; Rang 3 für 0,1 < t < 9/17; Rang 4 für t < 0,1
    [0.6 - 3 / 7, 3 / 7 + 0.4, 0.0, 0.0],                        # b: Rang 1 für 3/7 < t < 0,6; sonst Rang 2
    [3 / 7, 9 / 17 - 3 / 7, 1.0 - 9 / 17, 0.0],                  # c: Rang 1 für t < 3/7; Rang 2 für 3/7 < t < 9/17; sonst Rang 3
    [0.0, 0.0, 0.1, 0.9],                                        # d: Rang 3 für t < 0,1; sonst Rang 4
])
MINI_CENTRAL_T = np.array([0.8, (3 / 7 + 0.6) / 2, 3 / 14])      # mittleres Gewicht auf Kriterium 1, bei dem a, b, c Rang 1 haben


@pytest.fixture
def mini_V():
    return MINI_V.copy()


@pytest.fixture
def mini_grid():
    """Gewichte (t, 1 - t) auf einem feinen Mittenraster: Mittelwerte über das Raster sind die Gleichverteilung auf [0, 1] bis auf 1/N."""
    n = 200000
    t = (np.arange(n) + 0.5) / n
    return np.column_stack([t, 1.0 - t])
