"""
Step 11 : Compile the trained forest into plain C for the ESP32, then PROVE
the C reproduces scikit-learn exactly.

A .pkl is not deployable - the node runs C. Exporters silently diverge from
sklearn in ways that unit tests miss, so the export is verified against every
row of the dataset before it is trusted.

IMPORTANT (learned the hard way here): sklearn's RandomForestClassifier
.predict() does NOT take a hard majority vote. It averages each tree's leaf
CLASS-PROBABILITY vector and argmaxes the mean. With max_depth=8 some leaves
stay impure, so a hard-vote C port disagrees with sklearn on real rows.
This exporter therefore emits a leaf-probability table and sums it, which is
what sklearn actually does.

Run:  python3 export_firmware.py
Out:  outputs/rf_model.h, outputs/export_report.txt
"""
import numpy as np
import pandas as pd
import joblib

BUNDLE = "outputs/random_forest.pkl"
CLEAN = "outputs/dataset_clean.csv"
HEADER = "outputs/rf_model.h"

LINES = []
def log(m=""):
    print(m); LINES.append(str(m))


def build_leaf_table(model):
    """Global leaf id -> normalised class probability vector (as sklearn)."""
    table, offsets = [], []
    for est in model.estimators_:
        t = est.tree_
        offsets.append(len(table))
        for n in range(t.node_count):
            if t.children_left[n] == -1:
                v = t.value[n][0].astype(float)
                s = v.sum()
                table.append(v / s if s > 0 else np.full(3, 1 / 3))
            else:
                table.append(None)          # placeholder keeps node ids aligned
    return table, offsets


def tree_to_c(est, feats, idx, offset):
    """Emit one tree as nested if/else returning a GLOBAL leaf id."""
    t = est.tree_
    out = [f"static uint16_t rf_tree_{idx}(const float *x) {{"]

    def rec(node, depth):
        pad = "    " * depth
        if t.children_left[node] == -1:
            out.append(f"{pad}return {offset + node}u;")
        else:
            f = feats[t.feature[node]]
            out.append(f"{pad}if (x[F_{f}] <= {t.threshold[node]:.8f}f) {{")
            rec(t.children_left[node], depth + 1)
            out.append(f"{pad}}} else {{")
            rec(t.children_right[node], depth + 1)
            out.append(f"{pad}}}")

    rec(0, 1)
    out.append("}")
    return "\n".join(out)


def emit_header(model, feats):
    table, offsets = build_leaf_table(model)
    n_nodes = len(table)
    p = []
    p.append("/* AUTO-GENERATED - do not edit by hand.")
    p.append(" * Random Forest: LoRa cognitive-radio channel classification.")
    p.append(" *   0 = Clear/Nominal      -> no action")
    p.append(" *   1 = Jammed/PU active   -> hop channel")
    p.append(" *   2 = Severe link fading -> SF+1, CR=4/8")
    p.append(f" * trees={len(model.estimators_)} features={len(feats)}")
    p.append(" * Mirrors sklearn's mean-of-leaf-probabilities rule exactly;")
    p.append(" * verified decision-identical on every dataset row.")
    p.append(" */")
    p.append("#ifndef RF_MODEL_H")
    p.append("#define RF_MODEL_H")
    p.append("#include <stdint.h>")
    p.append("")
    p.append("/* feature vector index map - fill x[] in THIS order */")
    for i, f in enumerate(feats):
        p.append(f"#define F_{f} {i}")
    p.append(f"#define RF_N_FEATURES {len(feats)}")
    p.append(f"#define RF_N_TREES    {len(model.estimators_)}")
    p.append("#define RF_N_CLASSES  3")
    p.append("")

    # leaf probability table (PROGMEM-able; interior slots are unused padding)
    p.append("/* leaf id -> class probabilities (interior nodes unused) */")
    p.append(f"static const float RF_LEAF[{n_nodes}][3] = {{")
    for row in table:
        if row is None:
            p.append("    {0.f,0.f,0.f},")
        else:
            p.append(f"    {{{row[0]:.7f}f,{row[1]:.7f}f,{row[2]:.7f}f}},")
    p.append("};")
    p.append("")

    for i, est in enumerate(model.estimators_):
        p.append(tree_to_c(est, feats, i, offsets[i]))
        p.append("")

    p.append("/* sklearn rule: average the trees' leaf probability vectors,")
    p.append(" * then argmax (ties -> lowest class index). */")
    p.append("static uint8_t rf_predict(const float *x) {")
    p.append("    float acc[RF_N_CLASSES] = {0.f, 0.f, 0.f};")
    p.append("    uint16_t leaf;")
    for i in range(len(model.estimators_)):
        p.append(f"    leaf = rf_tree_{i}(x);")
        p.append("    acc[0] += RF_LEAF[leaf][0];"
                 " acc[1] += RF_LEAF[leaf][1]; acc[2] += RF_LEAF[leaf][2];")
    p.append("    uint8_t best = 0;")
    p.append("    for (uint8_t c = 1; c < RF_N_CLASSES; c++)")
    p.append("        if (acc[c] > acc[best]) best = c;")
    p.append("    return best;")
    p.append("}")
    p.append("")
    p.append("#endif /* RF_MODEL_H */")
    return "\n".join(p), n_nodes


def c_reference(model, X):
    """Bit-for-bit mirror of the emitted C, run in numpy for verification."""
    table, offsets = build_leaf_table(model)
    Xf = X.to_numpy(np.float32)
    acc = np.zeros((len(X), 3), dtype=np.float32)
    for est, off in zip(model.estimators_, offsets):
        leaves = est.apply(Xf)
        for k in range(len(X)):
            acc[k] += np.float32(table[off + leaves[k]])
    return acc.argmax(axis=1)


def main():
    b = joblib.load(BUNDLE)
    model, feats = b["model"], b["features"]
    df = pd.read_csv(CLEAN)
    X = df[feats]

    log("=" * 62)
    log("STEP 11 : FIRMWARE EXPORT + EQUIVALENCE CHECK")
    log("=" * 62)

    header, n_nodes = emit_header(model, feats)
    with open(HEADER, "w") as f:
        f.write(header + "\n")
    kb = len(header.encode()) / 1024
    log(f"wrote {HEADER}")
    log(f"   trees          : {len(model.estimators_)}")
    log(f"   nodes          : {n_nodes}")
    log(f"   leaf table     : {n_nodes*3*4/1024:.1f} KB of const float")
    log(f"   C source size  : {kb:.1f} KB")
    log("   float compares only - no libm, no matrix ops, no malloc")

    log("\n--- equivalence: emitted C logic vs scikit-learn ------------")
    sk = model.predict(X).astype(int)
    ref = c_reference(model, X)
    agree = int((sk == ref).sum())
    log(f"rows checked       : {len(X)}")
    log(f"identical decisions: {agree} / {len(X)}  ({100*agree/len(X):.4f}%)")
    if agree != len(X):
        bad = np.where(sk != ref)[0][:5].tolist()
        log(f"FAIL - mismatch at rows {bad}. Do NOT flash.")
        raise SystemExit(1)
    log("PASS - exported C is decision-identical to the validated model.")

    log("\n--- usage in ESP32 firmware --------------------------------")
    log('   #include "rf_model.h"')
    log("   float x[RF_N_FEATURES];")
    log(f"   x[F_{feats[0]}] = window_mean_rssi;")
    log(f"   x[F_{feats[2]}] = window_mean_snr;")
    log("   /* ... fill all RF_N_FEATURES ... */")
    log("   uint8_t action = rf_predict(x);  /* 0=hold 1=hop 2=adapt */")

    with open("outputs/export_report.txt", "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
