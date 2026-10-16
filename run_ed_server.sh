#!/usr/bin/env bash
# where-teacher-acts: the equal-dose comparison and the anchor on A (declared 2026-10-14 before running in
# the dated plan, docs/PLAN_AND_LOG.md), meant to run inside tmux:
#
#   tmux new -c <repo> -s qj_ed -d \
#       "bash <repo>/run_ed_server.sh"
#
# Trains the 126 runs of plan "ed" (seed by seed, student by student: FP32 phase ed_ft and ed_kd from A,
# then ed_none, ed_before, ed_during, then qanc_a), then measures them on the validation fold and, once every
# run is done, on the test fold, each fold once, under the tag "ed" (reports_qj/part2/ed/), so no file of
# the grid is written. Each step is resumable; a failed step stops the script. Log: logs/qj_ed.log.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_ed.log
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "start in $(pwd) (repo $(git rev-parse --short HEAD))"
$PY pipeline/21_train_part2.py --plan ed --ft-epochs 40 --wait >> logs/qj_ed_train.log 2>&1
rc=$?
say "train: exit $rc"
if [ "$rc" -ne 0 ]; then say "not every run is done; stopping before any measurement"; say "EXIT"; exit 1; fi

for fold in val test; do
    $PY pipeline/22_eval_part2.py --fold "$fold" --plan ed --ft-epochs 40 --tag ed --require-all \
        >> logs/qj_ed_eval.log 2>&1
    rc=$?
    say "eval $fold: exit $rc"
    if [ "$rc" -ne 0 ]; then say "stopping"; say "EXIT"; exit 1; fi
done
say "EXIT"
