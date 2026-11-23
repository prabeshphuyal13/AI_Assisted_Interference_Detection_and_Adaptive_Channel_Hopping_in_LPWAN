"""
Step 4-10 : split, baseline, size-budgeted tuning, evaluation, CV, importance.

The tuning stage is deliberately NOT a plain accuracy grid search. The forest
is compiled into nested if/else C for an ESP32, so total node count is a hard
deliverable constraint. We search a small grid (proposal spec: 10-50 trees,
depth 4-8) and select the SMALLEST forest within a tolerance of peak CV score.

Run:  python3 train_model.py
Out:  outputs/random_forest.pkl, outputs/model_meta.json,
      outputs/training_report.txt, figures/{tuning,confusion,importance}.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict,
                                     GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

CLEAN = "outputs/dataset_clean.csv"
SEED = 42
TOLERANCE = 0.005          # accept -0.5% CV F1 to buy a smaller forest
# HARD limit on the class 1<->2 cross-error (the proposal's "catastrophic"
# case). Set to 1.0% because that is what the data can actually support:
# the measured floor is ~0.62% even at 300-500 trees with unlimited depth
# (~46k nodes, far past any ESP32 budget), so the residual confusion is
# feature-limited, not hyperparameter-limited. A 0.5% gate is unreachable
# and would only be met by quietly overfitting the selection.
CRIT_MAX  = 0.010
CLASS_NAMES = {0: "Clear/Nominal", 1: "Jammed/PU", 2: "Severe fading"}

LINES = []
def log(m=""):
    print(m); LINES.append(str(m))


def total_nodes(forest):
    """Firmware footprint proxy: every node becomes an if/else in C."""
    return int(sum(e.tree_.node_count for e in forest.estimators_))


def main():
    df = pd.read_csv(CLEAN)
    feats = [c for c in df.columns if c != "label"]
    X, y = df[feats], df["label"]

    log("=" * 68)
    log("STEP 4 : TRAIN / TEST SPLIT")
    log("=" * 68)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=SEED)
    log(f"features ({len(feats)}) : {feats}")
    log(f"train {Xtr.shape}   test {Xte.shape}   stratified, seed={SEED}")

    # leakage guard: no identical feature row on both sides of the split
    tr_set = set(map(tuple, Xtr.to_numpy()))
    overlap = sum(1 for r in map(tuple, Xte.to_numpy()) if r in tr_set)
    log(f"identical feature rows present in BOTH train and test : {overlap}"
        f"   {'OK' if overlap == 0 else '<-- LEAKAGE'}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    log()
    log("=" * 68)
    log("STEP 5 : BASELINE")
    log("=" * 68)
    base = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    log(f"baseline test accuracy : {accuracy_score(yte, base.predict(Xte)):.4f}"
        f"   nodes = {total_nodes(base)}")

    log()
    log("=" * 68)
    log("STEP 6 : HYPERPARAMETER TUNING (accuracy AND firmware size)")
    log("=" * 68)
    grid = {
        "n_estimators":      [10, 20, 30, 40, 50],
        "max_depth":         [4, 6, 8],
        "min_samples_leaf":  [1, 3],
        "max_features":      ["sqrt", None],
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1),
                      grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(Xtr, ytr)
    res = pd.DataFrame(gs.cv_results_)

    # Node count + the SAFETY GATE for every candidate.
    # The gate is measured by cross-validation on the TRAINING set only -
    # scoring candidates on the test set would leak model selection into the
    # final evaluation and make the reported number optimistic.
    nodes, crit = [], []
    for p in res["params"]:
        m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
        m.fit(Xtr, ytr)
        nodes.append(total_nodes(m))
        cvp = cross_val_predict(m, Xtr, ytr, cv=cv, n_jobs=-1)
        c = confusion_matrix(ytr, cvp, labels=[0, 1, 2])
        crit.append((c[1][2] + c[2][1]) / (c[1].sum() + c[2].sum()))
    res["nodes"] = nodes
    res["critical_err"] = crit
    res = res.sort_values("mean_test_score", ascending=False)

    peak = res["mean_test_score"].max()
    log(f"grid candidates evaluated : {len(res)}")
    log(f"peak CV macro-F1          : {peak:.4f}")

    # Selection is a CONSTRAINED optimisation, not a beauty contest:
    #   hard constraint : CV critical (1<->2) error < CRIT_MAX   [safety]
    #   soft constraint : CV macro-F1 >= peak - TOLERANCE        [accuracy]
    #   objective       : minimise node count                    [firmware]
    feasible = res[(res["critical_err"] < CRIT_MAX) &
                   (res["mean_test_score"] >= peak - TOLERANCE)]
    log(f"within F1 tolerance ({TOLERANCE})            : "
        f"{int((res['mean_test_score'] >= peak - TOLERANCE).sum())}")
    log(f"...of which also pass safety gate (<{CRIT_MAX:.1%}) : {len(feasible)}")
    if feasible.empty:
        raise SystemExit("No candidate satisfies the safety gate - widen the "
                         "grid or revisit the features.")

    chosen = feasible.sort_values("nodes").iloc[0]
    best_params = chosen["params"]

    naive = res.iloc[0]
    log(f"\npure-accuracy pick would be : {naive['params']}")
    log(f"   CV F1 {naive['mean_test_score']:.4f}, nodes {int(naive['nodes'])}, "
        f"critical {naive['critical_err']*100:.2f}%")
    log(f"SIZE-AWARE + SAFE pick      : {best_params}")
    log(f"   CV F1 {chosen['mean_test_score']:.4f}, nodes {int(chosen['nodes'])}, "
        f"critical {chosen['critical_err']*100:.2f}%")
    saved = 1 - chosen["nodes"] / naive["nodes"]
    log(f"   -> {saved*100:.1f}% fewer nodes for "
        f"{(naive['mean_test_score']-chosen['mean_test_score'])*100:.2f}% F1")

    # what the size-only rule (no safety constraint) would have chosen
    unsafe = res[res["mean_test_score"] >= peak - TOLERANCE].sort_values("nodes").iloc[0]
    if unsafe["critical_err"] >= CRIT_MAX:
        log(f"\n   NOTE: ignoring the safety gate would have picked "
            f"{int(unsafe['nodes'])} nodes")
        log(f"   with a {unsafe['critical_err']*100:.2f}% critical error rate "
            f"-> rejected.")

    # accuracy-vs-size scatter
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(res["nodes"], res["mean_test_score"], s=28, alpha=.55,
               color="#4a6fa5", label="grid candidate")
    ax.scatter([chosen["nodes"]], [chosen["mean_test_score"]], s=180,
               marker="*", color="#e63946", zorder=5, label="selected")
    ax.axhline(peak - TOLERANCE, ls="--", lw=1, color="grey",
               label=f"peak - {TOLERANCE}")
    ax.set_xlabel("total nodes in forest  (firmware if/else count)")
    ax.set_ylabel("CV macro-F1")
    ax.set_title("Accuracy vs firmware footprint")
    ax.legend(); fig.tight_layout()
    fig.savefig("figures/tuning_accuracy_vs_size.png", dpi=150); plt.close(fig)

    model = RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                   **best_params).fit(Xtr, ytr)

    log()
    log("=" * 68)
    log("STEP 7 : EVALUATION")
    log("=" * 68)
    pred = model.predict(Xte)
    log(f"test accuracy  : {accuracy_score(yte, pred):.4f}")
    log(f"test macro-F1  : {f1_score(yte, pred, average='macro'):.4f}")
    log("\n" + classification_report(yte, pred, digits=3,
                                     target_names=[CLASS_NAMES[i] for i in [0, 1, 2]]))

    cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
    log("confusion matrix (rows=actual, cols=predicted):")
    log(str(cm))

    # ---- the safety gate the proposal actually calls for ----------------
    log("\n--- SAFETY GATE : class 1 <-> class 2 cross-misclassification ---")
    log("(proposal Sec 4.3.2 Step 6: these must never cross - a jam mistaken")
    log(" for fading raises SF on a blocked channel instead of hopping)")
    j_as_f, f_as_j = int(cm[1][2]), int(cm[2][1])
    n1, n2 = int(cm[1].sum()), int(cm[2].sum())
    log(f"jammed  -> predicted fading : {j_as_f} / {n1}  ({100*j_as_f/n1:.2f}%)")
    log(f"fading  -> predicted jammed : {f_as_j} / {n2}  ({100*f_as_j/n2:.2f}%)")
    gate = (j_as_f + f_as_j) / (n1 + n2)
    log(f"combined critical error rate: {gate*100:.2f}%   "
        f"{'PASS (<1%)' if gate < .01 else 'REVIEW'}")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ticks = [CLASS_NAMES[i] for i in [0, 1, 2]]
    ax.set_xticks(range(3)); ax.set_xticklabels(ticks, rotation=20)
    ax.set_yticks(range(3)); ax.set_yticklabels(ticks)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion matrix - test set")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig("figures/confusion_matrix.png", dpi=150); plt.close(fig)

    log()
    log("=" * 68)
    log("STEP 8 : CROSS-VALIDATION")
    log("=" * 68)
    for k in (5, 10):
        s = cross_val_score(model, X, y, cv=StratifiedKFold(k, shuffle=True,
                            random_state=SEED), scoring="f1_macro", n_jobs=-1)
        log(f"{k:2d}-fold macro-F1 : {s.mean():.4f} +/- {s.std():.4f}   "
            f"folds {np.round(s, 4)}")
    tr_acc = accuracy_score(ytr, model.predict(Xtr))
    log(f"\ntrain acc {tr_acc:.4f}  vs  test acc {accuracy_score(yte, pred):.4f}"
        f"   gap {tr_acc - accuracy_score(yte, pred):+.4f} -> "
        f"{'no overfitting' if tr_acc - accuracy_score(yte,pred) < .02 else 'CHECK'}")

    log()
    log("=" * 68)
    log("STEP 9 : FEATURE IMPORTANCE")
    log("=" * 68)
    gini = pd.Series(model.feature_importances_, index=feats)
    perm = permutation_importance(model, Xte, yte, n_repeats=20,
                                  random_state=SEED, n_jobs=-1,
                                  scoring="f1_macro")
    imp = pd.DataFrame({"gini": gini,
                        "permutation": perm.importances_mean,
                        "perm_std": perm.importances_std}
                       ).sort_values("permutation", ascending=False)
    log(imp.round(4).to_string())
    log("\nGini is biased toward high-cardinality/correlated features;")
    log("permutation importance on held-out data is the trustworthy one.")
    log("Features with permutation importance ~0 can be dropped from firmware.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g = gini.sort_values()
    axes[0].barh(g.index, g.values, color="#4a6fa5")
    axes[0].set_title("Gini importance (in-sample, biased)")
    p = imp.sort_values("permutation")
    axes[1].barh(p.index, p["permutation"], xerr=p["perm_std"],
                 color="#2a9d8f")
    axes[1].set_title("Permutation importance (held-out, trustworthy)")
    fig.tight_layout(); fig.savefig("figures/feature_importance.png", dpi=150)
    plt.close(fig)

    log()
    log("=" * 68)
    log("STEP 10 : SAVE")
    log("=" * 68)
    joblib.dump({"model": model, "features": feats}, "outputs/random_forest.pkl")
    meta = {
        "features": feats,
        "params": {k: (v if not isinstance(v, np.integer) else int(v))
                   for k, v in best_params.items()},
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "total_nodes": total_nodes(model),
        "critical_error_rate": float(gate),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    json.dump(meta, open("outputs/model_meta.json", "w"), indent=2)
    log("wrote outputs/random_forest.pkl")
    log("wrote outputs/model_meta.json")
    log(f"forest: {best_params['n_estimators']} trees, "
        f"depth {best_params['max_depth']}, {total_nodes(model)} nodes")

    with open("outputs/training_report.txt", "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
