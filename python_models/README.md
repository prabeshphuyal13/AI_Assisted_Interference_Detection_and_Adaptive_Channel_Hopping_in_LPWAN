# LoRa Cognitive Radio — Random Forest Classifier

Offline training pipeline for the ESP32 interference-detection / channel-hopping
node. Classifies each 5 s / 20-packet telemetry window into an action:

| Label | Condition | Firmware action |
|---|---|---|
| 0 | Clear / Nominal | No action — stay on channel |
| 1 | Jammed / Primary User active | Hop channel (beacon over 868 MHz control plane) |
| 2 | Severe link fading / path loss | Keep frequency, SF+1, CR=4/8 |

## Run it

```bash
bash run_all.sh          # cleaning -> EDA -> train -> export -> verify
```
Deterministic (`seed=42`). Needs `pandas numpy scikit-learn matplotlib joblib scipy`,
plus `gcc` for the compiled-C verification step.

| Script | Workflow steps | Purpose |
|---|---|---|
| `prepare_data.py` | 1–3 | Cleaning, sentinel handling, feature selection |
| `eda.py` | 2 | Plots (run **after** cleaning — see below) |
| `train_model.py` | 4–10 | Split, tuning, evaluation, CV, importance |
| `export_firmware.py` | 11 | Emit `rf_model.h`, prove it matches sklearn |
| `test_rf_model.c` | 11 | Replay every row through the **compiled** C |

## Results

| Metric | Value |
|---|---|
| Test accuracy | **98.66 %** |
| Test macro-F1 | **98.66 %** |
| 10-fold CV macro-F1 | 0.9902 ± 0.0052 |
| Train→test gap | +0.011 (no overfitting) |
| Critical (class 1↔2) error | **0.87 %** — PASS |
| Forest | 30 trees, depth 8, 3030 nodes |
| Compiled size | ~73 KB (`-Os`, x86-64 proxy) |
| sklearn ↔ compiled C agreement | **5234 / 5234 (100 %)** |

Final config — 30 trees, `max_depth=8`, `max_features='sqrt'` — was selected by
constrained search and landed **exactly on the spec in the proposal**
(10–50 trees, depth ≤ 8, ~30 typical), arrived at independently from the data.

## Findings that shaped the design

**1. EDA must run after cleaning, not before.**
The raw file encodes "no packet received" as `RSSI = SNR = -999`. Those 49 rows
drag the RSSI/SNR Pearson correlation to ~1.00. Real value after cleaning: 0.54.
Doing EDA on the raw CSV leads to the wrong conclusion about feature redundancy.

**2. `meanToA == 0` is a second, hidden sentinel.**
It means "not measurable", not "zero airtime", and appears in 400 class-1 rows,
10 class-2 rows, and **zero** class-0 rows — i.e. it leaks the label. It's
imputed from the (SF, CR) lookup. Removing the shortcut changed accuracy by
0.00 %, so the headline number never depended on it. `meanToA` is then dropped
entirely: it takes exactly one value per (SF, CR) pair, so it is redundant.

**3. Feature engineering was measured and rejected.**
A tree splits on thresholds, so any strictly *monotonic* transform yields
identical splits and adds nothing. Verified by Spearman = 1.0000 for
`sdRSSI`/`varRSSI`, `sdSNR`/`varSNR`, `crcRate`/`CRC`, `snrMargin`/`meanSNR`.
This invalidates 3 of the 4 derived features usually proposed here.

| Feature set | 5-fold macro-F1 |
|---|---|
| base only | **0.9902** |
| base + RSSI/SNR ratio | 0.9884 (−0.0017) |
| base + 4 monotonic transforms | 0.9861 (−0.0041) |

They *hurt*: `max_features` samples columns at random, so redundant twins waste
picks. Same reason scaling/normalisation is unnecessary for RF.

**4. Tuning is a constrained optimisation, not an accuracy contest.**
The forest becomes nested `if/else` in firmware, so node count is a real
constraint. Selection rule: minimise nodes, subject to CV macro-F1 ≥ peak−0.005
**and** critical error < 1 %. The safety gate is measured by CV on the training
set — scoring candidates on the test set would leak selection into the final
number. Without the gate the search picks a 640-node forest with a **1.34 %**
critical error; the gate correctly rejects it.

**5. The class 1↔2 error has a floor of ~0.62 %.**
Measured at 100/300/500 trees with unlimited depth (~46 k nodes, far past any
ESP32 budget) — it does not improve. The residual is **feature-limited, not
hyperparameter-limited**, so the gate is set to an achievable 1 % rather than an
aspirational 0.5 % that could only be met by overfitting the selection.

The error is also **asymmetric**, which matters more than the aggregate:
- jammed → predicted fading: **0 / 348 (0.00 %)** ← the dangerous direction
  (raising SF on a jammed channel keeps transmitting into the blockage)
- fading → predicted jammed: 6 / 341 (1.76 %) ← merely wasteful (hops away from
  a channel that only needed SF+1)

**6. Total link loss is physically undecidable — firmware needs a fallback.**
The 49 "zero packets received" rows collapse to **2 unique feature vectors**, and
one carries **both label 1 and label 2** (28 rows). When nothing arrives,
jamming and fading are indistinguishable — the information is not in the window.
No model fixes this. `link_lost` consequently has ~0 permutation importance.
**Recommendation:** when `link_lost = 1`, do not consult the RF. Use a
deterministic rule (e.g. hop first — cheap and reversible — then try SF+1 if the
new channel also fails), and keep the 868 MHz control plane authoritative.

**7. The `.pkl` is not the deliverable, and the export can silently lie.**
sklearn's `.predict()` does **not** take a hard majority vote — it averages each
tree's leaf *probability* vector and argmaxes the mean. With `max_depth=8` some
leaves stay impure, so a hard-vote C port disagreed with sklearn on 1 of 5234
rows. Caught only because every row is replayed through the compiled binary.
The exporter now emits a leaf-probability table matching sklearn exactly:
**5234 / 5234**.

## Feature importance

Permutation importance (held-out) is the trustworthy column; Gini is biased
toward high-cardinality and correlated features.

| Feature | Gini | Permutation |
|---|---|---|
| meanRSSI / meanSNR | ~0.36 each | ~0.33 each |
| CRC | 0.11 | 0.08 |
| PLR | 0.03 | 0.06 |
| CFO | 0.01 | 0.03 |
| varRSSI / varSNR | ~0.01 | ~0.02 |
| SF / CR / link_lost | ~0.00 | ~0.00 |

`SF`, `CR` and `link_lost` can be dropped from firmware to shrink the forest,
**but** keep `link_lost` wired as the fallback trigger in finding 6 — its low
importance is exactly *because* it marks the undecidable rows.

## Known limitations (state these in the thesis)

- **Synthetic data.** ~99 % accuracy reflects a generator with clean class
  boundaries. Real hardware will be lower. Treat this as an upper bound and
  re-validate on captured telemetry before claiming field performance.
- **10 of 14 features.** The proposal's deployed vector adds BW, payload length,
  PDR, latency, cumulative CRC and channel ID. The model is only valid for the
  10 it was trained on — firmware must not feed it a different vector.
- **3 classes, not 5.** The proposal describes up to 5 actions (incl. SF−1 to
  maximise throughput). Confirm this scope cut is intentional.
- **Static evaluation.** Rows are shuffled i.i.d.; the real system sees a
  *time series*. A temporal split would be a stricter and more honest test.
- **Footprint is an x86-64 proxy.** Xtensa codegen differs; measure on target.
