#!/usr/bin/env bash
# Sequential per-fold LightGBM preview run.
#
# Each fold is a separate process: starts a fresh Python interpreter, runs
# its Optuna study, fits, evaluates, writes its own report, and exits — so
# memory from prior folds (Optuna trials, LightGBM trees, Prefect server,
# MLflow client) is fully released before the next fold begins. This
# sidesteps the cumulative memory pressure that triggered a jetsam kill
# during the first attempt to run all folds in one process.
#
# Per-fold report: reports/lgbm_v1_preview_20d_fold_{N}.json
# Per-fold log:    /tmp/lgbm_20d_fold_{N}.log
# Combined aggregation is done in a Python step after all 3 finish.

set -euo pipefail

HORIZON=20
N_TRIALS=50
FOLDS=(0 7 14)

cd "$(dirname "$0")/.."

for fold in "${FOLDS[@]}"; do
    echo "=== Starting fold ${fold} (horizon ${HORIZON}d, ${N_TRIALS} trials) ==="
    report_path="reports/lgbm_v1_preview_20d_fold_${fold}.json"
    log_path="/tmp/lgbm_20d_fold_${fold}.log"
    uv run kuri models train-lgbm \
        --folds "${fold}" \
        --horizon "${HORIZON}" \
        --n-trials "${N_TRIALS}" \
        --n-shuffles 1000 \
        --report-path "${report_path}" 2>&1 | tee "${log_path}"
    echo "=== Fold ${fold} complete ==="
done

echo "All folds done. Per-fold reports:"
for fold in "${FOLDS[@]}"; do
    echo "  reports/lgbm_v1_preview_20d_fold_${fold}.json"
done
