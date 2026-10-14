# Unsichere Gewichte: Rangakzeptanz mit SMAA – Streamlit-Demo

<!-- Nach dem Deploy hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-smaa-demo.streamlit.app/)** -->

Elftes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[Gewichte erheben](https://github.com/sebastian-hanisch/gewichte-erheben-demo). Statt eines Gewichtsvektors wird der ganze **Gewichtsraum** gezogen: Je Alternative entsteht die
**Rangakzeptanz** (Anteil der gezogenen Gewichte, bei denen sie Rang r erreicht) und ein **zentraler Gewichtsvektor** (das Mittel der Gewichte, bei denen sie vorne liegt).

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seine
Gewichte und seinen Nutzen (linear, konkav oder Cobb-Douglas in der Güte, linear in den Rohwerten, oder mit Untergrenze); das Verfahren kennt nur die Matrix und, wenn man es dazugibt,
Vorwissen über die Gewichte. Der Entscheider ist ein Orakel, kein Mensch.

## Warum dieses Problem

**Der Index sagt, wie oft eine Alternative bei den gezogenen Gewichten vorne liegt, aber er ist nur unter der gezogenen Verteilung eine Wahrscheinlichkeit.** Die Kernfrage der Demo: Wie
viel sagt der Index über den Besten eines Entscheiders, was bringt Vorwissen über die Gewichte, und woran hängen die Zahlen?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Hypothese „das Ergebnis hängt an der Gleichverteilung“ ist halb richtig.** Der Sieger bleibt bei den drei Verteilungen α = 0,5 / 1 / 5 in 78,0 % der Matrizen derselbe; die
  Wahrscheinlichkeit hängt an der Verteilung: Bei Index ≥ 0,5 und passender Ziehung stimmt die Aussage in 67,9 % der Fälle (behauptet 0,67), bei zu enger Ziehung in 54,5 % (behauptet
  0,79), bei zu breiter in 92,3 % (behauptet 0,60).
- **Ohne Wissen über die Gewichte ist der Index-Sieger nicht besser als gleiche Gewichte** (höchstens 3,0 Punkte mehr). Der Beste steht aber in 77,5 bis 91,0 % der Matrizen unter den
  ersten drei nach Index: Die Methode leistet die Menge der möglichen Besten, nicht den Besten.
- **Wissen über die Gewichte leistet mehr als die Ziehung:** Die Rangfolge der Gewichte senkt die möglichen Beste von 8,9 auf 2,9 und hebt die Trefferquote von 59,0 auf 83,5 %; die
  ROC-Gewichte (der Schwerpunkt dieses Bereichs) treffen 84,5 %, die Rangsumme 76,0 %.
- **Index null heißt „unter 1/N“, nicht „unmöglich“:** Der LP findet 11,06 mögliche Beste je Matrix, 10 000 Ziehungen 9,44. 3,00 von 14,06 effizienten Alternativen sind für kein Gewicht
  beste.
- **Die Skala ändert die Antwort, nicht ihre Güte:** Streut ein Kriterium nur zu einem Viertel, wechselt der Sieger zwischen Nenn-Spannweite und Min-Max in 27,0 % der Matrizen.
- **Die Experimente starten auf Klick** (rund 15 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (4 bis 20, Standard 10), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; Werte auf die feste Nenn-Spannweite oder auf Min-Max der Stichprobe
  normiert. Die Kriterienwerte sind gegeben, unsicher sind nur die Gewichte.
- **Ziehung:** N = 100 / 1 000 / 10 000 Gewichtsvektoren aus Dirichlet(α), α = 0,5 / 1 / 5 (1 = gleichverteilt über dem Simplex); Vorwissen: die Rangfolge der Gewichte (sortierte
  Ziehung, 1/m! des Raums) oder ein Intervall ±0,15 / ±0,10 / ±0,05 um jedes Gewicht (gleichverteilt in der Box).
- **Ausgaben:** Rangakzeptanz b(i, r), erwarteter Rang, zentraler Gewichtsvektor; dazu per LP die für ein Gewicht besten Alternativen.
- **Entscheider:** verborgene Gewichte aus Dirichlet(α wahr); fünf Nutzenformen. Nur Rohwerte/linear passt zum Modell von SMAA.
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), 20 Alternativen, 5 Kriterien, 4 000 Ziehungen je Matrix, unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Verfahren:** Vergleiche sind gepaart (Index-Sieger, gleiche Gewichte, kleinster erwarteter Rang, ROC, Rangsumme).
2. **Messgröße ist die Entscheidung:** Trifft der Sieger den Besten des Entscheiders, steht er unter den ersten drei, wie groß ist der behauptete Index gegen den Anteil richtiger Aussagen.
3. **Kalibrierung:** Nur Fälle mit Index ≥ 0,5; gezählt wird, wie oft der Sieger der Beste ist, gegen den Index, den er trägt.
4. **Stichprobe gegen LP:** Fehler des Index gegen einen Referenzlauf und die übersehenen möglichen Beste.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, 20 Alternativen, 5 Kriterien, 4 000 Ziehungen. **Der Standardfehler eines Anteils ist 1,5 bis 3,5 Punkte; Unterschiede unter etwa
5 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Trifft der Index-Sieger den Besten? (Güte/linear, konkav, Cobb-Douglas, Rohwerte, Untergrenze; Gewichte gleichverteilt) | **Index-Sieger 47,0 / 52,5 / 55,0 / 59,0 / 51,5 %**; gleiche Gewichte 45,5 / 52,0 / 56,5 / 56,0 / 52,0 %; kleinster erwarteter Rang 45,0 / 51,5 / 55,5 / 56,5 / 51,0 %; **Bester unter den ersten drei 79,0 / 83,0 / 83,5 / 91,0 / 77,5 %**; Rang des Siegers beim Entscheider im Mittel 2,4 bis 3,0. | `test_hit_rates_by_decider` |
| Was Vorwissen bringt (Rohwerte): mögliche Beste / Index-Sieger trifft; keines, Rangfolge, ±0,15, ±0,10, ±0,05 | **8,9 / 59,0 %; 2,9 / 83,5 %; 3,6 / 87,5 %; 2,6 / 93,0 %; 1,7 / 99,0 %**; bei bekannter Rangfolge: ROC-Gewichte 84,5 %, Rangsumme 76,0 %, gleiche Gewichte 56,0 %. | `test_prior_knowledge_shrinks_the_possible_best_and_raises_the_hit_rate` |
| Ist der Index eine Wahrscheinlichkeit? Index des Siegers ≥ 0,5: behauptet / richtig; (α wahr, α Ziehung) | (1; 1) **0,67 / 67,9 %**; (1; 5) **0,79 / 54,5 %**; (5; 0,5) **0,60 / 92,3 %**; (0,5; 5) 0,79 / 43,6 %; (0,5; 0,5) 0,60 / 60,0 %; (1; 0,5) 0,60 / 72,9 %; (5; 5) 0,79 / 76,9 %. Fälle 310 / 495 / 870 von 1 000 je Ziehungsverteilung. Der Sieger selbst ist bei α = 0,5 / 1 / 5 in **78,0 %** der Matrizen derselbe. | `test_calibration_depends_on_the_distribution_not_the_winner` |
| Fehler des Index und was die Stichprobe übersieht (N = 100 / 1 000 / 10 000) | Fehler Mittel **0,055 / 0,017 / 0,006**, Maximum 0,111 / 0,041 / 0,013 (20 Matrizen, Referenz 40 000 Ziehungen); mögliche Beste per LP **11,06**, in der Stichprobe 6,48 / 8,38 / 9,44, übersehen 4,58 / 2,68 / 1,62 (50 Matrizen, in 50 / 47 / 41 mindestens eines); effizient 14,06, **für kein Gewicht beste 3,00** (in 50 von 50 Matrizen mindestens eine). | `test_sampling_error_and_what_the_sample_misses` |
| Hängt es an der Skala? Kriterium 1 streut zu 100 % / 25 %: Sieger wechselt (Nenn gegen Min-Max) | **6,5 % / 27,0 %**; Abstand der Indizes (TV) 0,037 / 0,221; Index des Besten in Kriterium 1 bei 25 %: 0,086 gegen 0,165; Trefferquote 59,5 / 58,0 %. Naive Ziehung (m Gleichverteilte normiert): Varianz je Gewicht 0,48-fach, Sieger wechselt in 7,5 %. | `test_scale_changes_the_winner_for_a_narrow_criterion` |

## Ehrliche Grenzen

- **Die Verteilung der Ziehung ist die Annahme.** Der Index ist nur unter der Verteilung, aus der die Gewichte kommen, eine Wahrscheinlichkeit; der Anwender kennt sie nicht. Die
  Lücke (−35 bis +32 Punkte bei Index ≥ 0,5) ist an drei Verteilungen gemessen, nicht an Menschen.
- **Kriterienwerte sind gegeben.** SMAA-2 mit verteilten Werten und der Konfidenzfaktor sind nicht Teil; bei gegebenen Werten ist der Konfidenzfaktor trivial 1.
- **Lineares Modell.** Der Entscheider mit Güte-Nutzen ist ein Modellrisiko (47,0 gegen 59,0 %), kein Mangel der Ziehung.
- **Stichprobe, nicht Beweis.** Index null heißt „unter 1/N“; der LP ist hier der Gegenpol und kostet eine Berechnung je Alternative.
- **Das Vorwissen ist unverzerrt.** Das Intervall liegt symmetrisch um die wahren Gewichte, ein Erheber ohne Fehler; bei einem Intervall ist die Ziehung gleichverteilt (α wirkt nicht).
- **Kleine Unterschiede sind Rauschen.** Index-Sieger gegen gleiche Gewichte (höchstens 3,0 Punkte) und die Spalten der Güte-Entscheider liegen innerhalb von zwei Standardfehlern.
  SMAA (Lahdelma, Hokkanen, Salminen) ist hier nicht im Original geprüft.

## Verifikation

- **Kern:** Rangmatrix, Rangakzeptanz, zentrale Gewichte, erwarteter Rang, mögliche Beste (LP) und Pareto-Effizienz gegen die Handrechnung der Mini-Instanz (`tests/conftest.py`: m = 2,
  vier Alternativen, exakte Schnittpunkte 0,1 / 3/7 / 9/17 / 0,6, Index 0,4 / 0,171 / 0,429 / 0).
- **Orakel** (`tests/test_oracle_smaa.py`): Zeilen- und Spaltensummen 1; dominierte Alternativen nie vorne; kleinerer Gewichtsraum hat höchstens so viele mögliche Beste; m = 2 gegen
  eine unabhängige Rechnung; ROC gleich Schwerpunkt des geordneten Simplex (Gegenprobe: Rangsumme nicht); LP gegen Dichte; zentrales Gewicht im Raum und dort beste.
- **Randfälle** (`tests/test_edge_cases.py`): n = 1 und 2, m = 1, eine Ziehung, Gleichstände, Intervall an der Ecke des Simplex, identische Alternativen, konstantes Kriterium bei Min-Max.
- **Ziehung** (`tests/test_sampling.py`): Dirichlet-Momente gegen die Theorie, Ordnung, Box, naive Ziehung, Determinismus.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an von Hand gesetzten Zeilen in `test_verdicts.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jede Auswahl), alle Presets, Intervall-Vorwissen, Dreiecksdiagramm bei m = 3, Permalink, PDF
  und Experimente erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `sma_constants.py` | Regler-Grenzen, Verteilungen, Vorwissen, Skalen, Entscheider, feste Annahmen der Experimente, Presets |
| `sma_score.py` | Skalen, Nutzen der Entscheider, Rangformel-Gewichte, Kendalls τ-b, TV-Abstand |
| `sma_sampling.py` | Ziehung des Gewichtsraums: Dirichlet, geordnet, Intervall, naive Normierung |
| `sma_smaa.py` | Rangmatrix, Rangakzeptanz, zentrale Gewichte, erwarteter Rang, mögliche Beste (LP), Pareto-Effizienz |
| `sma_evaluation.py` | Fall der App, vier Experimente, Verdikte (reine Funktionen) |
| `sma_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `sma_visualization.py` | Heatmap der Rangakzeptanz, zentrale Gewichte, Dreiecksdiagramm, Tabellen und Abbildungen der Experimente (Achsen gesperrt) |
| `sma_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, SMAA, Ziehung, Score, Orakel, Randfälle, Generator, Auswertung, Verdikte, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Unsichere Kriterienwerte und der Konfidenzfaktor (SMAA-2):** gegeben sind die Werte, unsicher sind die Gewichte.
- **Hit-and-Run und andere Ziehungsverfahren für beliebige Gewichtsräume:** Rangfolge und Intervall um die Gewichte genügen für die Aussagen; eine Box wird per Rückweisung gezogen.
- **Teilinformation über Paare von Gewichten (w₁ ≥ 2 w₂) und Intervalle um Verhältnisse:** ein Vorwissen je Auswahl.
- **Verteilungen über α ∈ {0,5; 1; 5} hinaus:** drei Werte reichen für die Aussage über die Kalibrierung.
- **Das Lernen der Gewichte aus Beispielurteilen:** steht in einem eigenen Stück der Linie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
