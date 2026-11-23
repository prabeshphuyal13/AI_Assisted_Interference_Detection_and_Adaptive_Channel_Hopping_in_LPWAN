#!/usr/bin/env bash
# Full pipeline, start to finish. Deterministic (seed=42).
set -e
mkdir -p outputs figures

echo "### 1/4  cleaning + feature selection"
python3 prepare_data.py

echo; echo "### 2/4  exploratory data analysis"
python3 eda.py

echo; echo "### 3/4  train / tune / evaluate"
python3 train_model.py

echo; echo "### 4/4  firmware export + equivalence proof"
python3 export_firmware.py

# The check that actually matters: replay every row through the COMPILED C.
if command -v gcc >/dev/null 2>&1; then
  echo; echo "### verifying compiled C against sklearn"
  python3 - <<'PY'
import joblib, pandas as pd, numpy as np
b = joblib.load('outputs/random_forest.pkl')
df = pd.read_csv('outputs/dataset_clean.csv')
X = df[b['features']]
np.savetxt('outputs/_vectors.csv',
           np.column_stack([X.to_numpy(), b['model'].predict(X)]),
           delimiter=',', fmt='%.8g')
PY
  gcc -O2 -std=c99 -I outputs test_rf_model.c -o /tmp/test_rf
  /tmp/test_rf outputs/_vectors.csv
  rm -f outputs/_vectors.csv
else
  echo "gcc not found - skipping compiled-C verification"
fi

echo; echo "### done. deliverables in outputs/ and figures/"
