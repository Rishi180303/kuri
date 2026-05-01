#!/usr/bin/env bash
# Full 15-fold walk-forward LightGBM run on the 20d target.
#
# Each fold runs in its own process (fresh Python interpreter), same pattern
# as scripts/run_lgbm_preview.sh — sidesteps the cumulative memory pressure
# that triggered a jetsam kill when all folds ran in one process.
#
# Folds 0..14 (fold 15 doesn't exist at the 20d horizon: the last 20 trading
# days lack forward labels, so walk-forward generates one fewer fold than at
# 5d).
#
# `set -e` is intentionally NOT used: if a fold dies (jetsam, transient
# failure), the loop continues with the remaining folds and the summary at
# the end lists which folds succeeded vs failed. Re-run a failed fold by
# invoking `kuri models train-lgbm --folds {N} --horizon 20 --n-trials 50`
# directly.
#
# Per-fold report: reports/lgbm_v1_full_20d_fold_{N}.json
# Per-fold log:    /tmp/lgbm_20d_full_fold_{N}.log
# Final aggregate: produced separately by scripts/aggregate_lgbm_full.py

set -uo pipefail

HORIZON=20
N_TRIALS=50
FEATURE_VERSION="${FEATURE_VERSION:-1}"
FOLDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

cd "$(dirname "$0")/.."

succeeded=()
failed=()

for fold in "${FOLDS[@]}"; do
    echo
    echo "=== [$(date '+%H:%M:%S')] Starting fold ${fold} (horizon ${HORIZON}d, ${N_TRIALS} trials, features v${FEATURE_VERSION}) ==="
    report_path="reports/lgbm_v${FEATURE_VERSION}_full_20d_fold_${fold}.json"
    log_path="/tmp/lgbm_v${FEATURE_VERSION}_20d_full_fold_${fold}.log"
    if uv run kuri models train-lgbm \
        --folds "${fold}" \
        --horizon "${HORIZON}" \
        --n-trials "${N_TRIALS}" \
        --n-shuffles 1000 \
        --feature-version "${FEATURE_VERSION}" \
        --report-path "${report_path}" 2>&1 | tee "${log_path}"; then
        succeeded+=("$fold")
        echo "=== [$(date '+%H:%M:%S')] Fold ${fold} complete ==="
    else
        failed+=("$fold")
        echo "=== [$(date '+%H:%M:%S')] Fold ${fold} FAILED — continuing ==="
    fi
done

echo
echo "================================================================"
echo "Full run complete at $(date '+%H:%M:%S')."
echo "  Features:  v${FEATURE_VERSION}"
echo "  Succeeded: ${succeeded[*]:-(none)}"
echo "  Failed:    ${failed[*]:-(none)}"
echo "  Reports:   reports/lgbm_v${FEATURE_VERSION}_full_20d_fold_{0..14}.json"
echo "================================================================"

if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
