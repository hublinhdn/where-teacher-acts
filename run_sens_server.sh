#!/usr/bin/env bash
# where-teacher-acts: the sensitivity analyses declared on 2026-10-16 before running (the dated plan,
# docs/PLAN_AND_LOG.md), meant to run inside tmux, in two sessions that can run side by side:
#
#   tmux new -c <repo> -s qj_sens -d \
#       "bash <repo>/run_sens_server.sh train"
#   tmux new -c <repo> -s qj_ptqsens -d \
#       "bash <repo>/run_sens_server.sh ptq"
#
# train: the 35 runs of plan "sens" (ed_anc, three seeds; qat_a_fs, two seeds), then their measurement on the
#        validation and the test fold under the tag "sens" (reports_qj/part2/sens/). Log: logs/qj_sens.log.
# ptq:   39_ptq_sensitivity.py, part "strong" (test, then validation), then part "calib" (test), into
#        reports_qj/bench/sens/. Log: logs/qj_ptqsens.log.
# Each step is resumable; a failed step stops its script.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
mkdir -p logs
what=${1:?train or ptq}
if [ "$what" = train ]; then
    LOG=logs/qj_sens.log
    say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
    say "start train in $(pwd) (repo $(git rev-parse --short HEAD))"
    $PY pipeline/21_train_part2.py --plan sens --ft-epochs 40 --wait >> logs/qj_sens_train.log 2>&1
    rc=$?
    say "train: exit $rc"
    if [ "$rc" -ne 0 ]; then say "not every run is done; stopping before any measurement"; say "EXIT"; exit 1; fi
    for fold in val test; do
        $PY pipeline/22_eval_part2.py --fold "$fold" --plan sens --ft-epochs 40 --tag sens --require-all \
            >> logs/qj_sens_eval.log 2>&1
        rc=$?
        say "eval $fold: exit $rc"
        if [ "$rc" -ne 0 ]; then say "stopping"; say "EXIT"; exit 1; fi
    done
    say "EXIT"
elif [ "$what" = ptq ]; then
    LOG=logs/qj_ptqsens.log
    say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
    say "start ptq in $(pwd) (repo $(git rev-parse --short HEAD))"
    for part in strong calib; do
        $PY pipeline/39_ptq_sensitivity.py --part "$part" --tag sens >> logs/qj_ptqsens_run.log 2>&1
        rc=$?
        say "$part: exit $rc"
        if [ "$rc" -ne 0 ]; then say "stopping"; say "EXIT"; exit 1; fi
    done
    say "EXIT"
else
    echo "usage: $0 train|ptq" >&2; exit 2
fi
