#!/usr/bin/env bash
# where-teacher-acts: the exploratory anchor control (declared 2026-10-13 before running in the dated plan,
# docs/PLAN_AND_LOG.md), meant to run inside tmux:
#
#   tmux new -c <repo> -s qj_anchor -d \
#       "bash <repo>/run_anchor_server.sh"
#
# Trains the seven qanc_b runs (B distilled from ConvNeXt-S, fine-tuned 40 epochs with B's own frozen FP32
# model in the cosine term), then measures them on the validation fold and, once every run is done, on the
# test fold, each fold once, under the tag "anchor" (reports_qj/part2/anchor/), so no file of the grid is
# written. Each step is resumable; a step that fails stops the script. Log: logs/qj_anchor.log.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_anchor.log
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "start in $(pwd) (repo $(git rev-parse --short HEAD))"
$PY pipeline/21_train_part2.py --plan anchor --ft-epochs 40 --wait >> logs/qj_anchor_train.log 2>&1
rc=$?
say "train: exit $rc"
if [ "$rc" -ne 0 ]; then say "not every anchor run is done; stopping before any measurement"; say "EXIT"; exit 1; fi

for fold in val test; do
    $PY pipeline/22_eval_part2.py --fold "$fold" --plan anchor --ft-epochs 40 --tag anchor --require-all \
        >> logs/qj_anchor_eval.log 2>&1
    rc=$?
    say "eval $fold: exit $rc"
    if [ "$rc" -ne 0 ]; then say "stopping"; say "EXIT"; exit 1; fi
done
say "EXIT"
