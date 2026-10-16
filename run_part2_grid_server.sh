#!/usr/bin/env bash
# where-teacher-acts: the rest of Part II after gate G1, meant to run inside tmux, and only once G1 has
# been reviewed:
#
#   tmux new -c <repo> -s qj_grid -d \
#       "bash <repo>/run_part2_grid_server.sh"
#   tmux new -c <repo> -s qj_valeval -d \
#       "bash <repo>/run_part2_valeval_server.sh"
#
# Priority phases (decided 2026-10-10), so that a stop part way leaves complete questions:
#   core    QAT(A), QAT(B) and QAKD(B) with every teacher, one student at a time (the pilot student's
#           cells were done for G1 and are skipped);
#   qakd_a  QAKD(A) with every teacher, one student at a time;
#   seeds   QAT(A) and QAKD(B) with ConvNeXt-S at seeds 123 and 223, for the seed-repeat students.
# QJ_PHASES chooses the phases, in this order (default "core qakd_a seeds"); on 2026-10-10 the grid
# was run with "core qakd_a" and the seeds phase was held for a later decision. Each call of
# 21_train_part2.py is resumable and waits for room on the shared card. A stop for disk,
# by a signal, or a dead scheduler stops this script; a run that is not done (exit 1) is listed and
# the grid goes on. The validation follower (run_part2_valeval_server.sh) measures finished runs on
# the CPU beside it. Nothing here reads the test fold. Log: logs/qj_part2_grid.log.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_part2_grid.log
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

N=$($PY -c "import json; print(json.load(open('reports_qj/part2/g0.json'))['chosen_N'])") \
    || { say "no G0 result in reports_qj/part2/g0.json; stopping"; say "EXIT"; exit 1; }
STUDENTS=$($PY -c "import sys; sys.path.insert(0, 'pipeline'); import _qj_common as QJ; \
print(' '.join([QJ.PILOT_STUDENT] + [s for s in QJ.STUDENTS if s != QJ.PILOT_STUDENT]))") \
    || { say "cannot read the student roster; stopping"; say "EXIT"; exit 1; }
PHASES=${QJ_PHASES:-core qakd_a seeds}
for phase in $PHASES; do
    case "$phase" in core|qakd_a|seeds) ;; *) say "unknown phase $phase in QJ_PHASES; stopping"; say "EXIT"; exit 1;; esac
done
say "start in $(pwd) (repo $(git rev-parse --short HEAD)); N=$N; phases: $PHASES; students: $STUDENTS"

for phase in $PHASES; do
    for s in $STUDENTS; do
        $PY pipeline/21_train_part2.py --plan grid --ft-epochs "$N" --phase "$phase" --students "$s" --wait \
            >> logs/qj_part2_train.log 2>&1
        rc=$?
        say "phase $phase, $s: exit $rc"
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
            say "the grid stopped (2: disk below its floor, 3: stopped by a signal, other: the scheduler died); stopping here"
            say "EXIT"; exit 0
        fi
    done
done
for phase in $PHASES; do
    $PY pipeline/21_train_part2.py --plan grid --ft-epochs "$N" --phase "$phase" --dry-run >> "$LOG" 2>&1
done
say "grid finished for phases: $PHASES; any run not done is listed above"
say "EXIT"
