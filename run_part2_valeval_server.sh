#!/usr/bin/env bash
# where-teacher-acts: validation measurement beside the Part II grid, on the CPU, inside tmux (started with
# run_part2_grid_server.sh, see there):
#
#   tmux new -c <repo> -s qj_valeval -d \
#       "bash <repo>/run_part2_valeval_server.sh"
#
# Every 15 minutes: measures on the validation fold every Part II run (all seeds) that finished since
# the last pass, with the FP32 references it started from (22_eval_part2.py --device cpu, niced, so the
# card stays with training; resumable, and it builds nothing when nothing is new), then lists the runs
# whose real INT8 is below 0.7 of the simulated INT8 that selected them (25_check_sim_real.py). The list
# is for monitoring only; nothing is stopped by it. Ends after the grid supervisor logs EXIT and one
# last pass is done. Nothing here reads the test fold. Log: logs/qj_part2_valeval.log.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_part2_valeval.log
GRID_LOG=logs/qj_part2_grid.log
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

N=$($PY -c "import json; print(json.load(open('reports_qj/part2/g0.json'))['chosen_N'])") \
    || { say "no G0 result; stopping"; say "EXIT"; exit 1; }
START=$( (wc -l < "$GRID_LOG") 2>/dev/null || echo 0)
say "start (repo $(git rev-parse --short HEAD)); N=$N; the grid log has $START lines before this run"
while true; do
    last=0
    if tail -n +"$((START + 1))" "$GRID_LOG" 2>/dev/null | grep -a -q "\] EXIT"; then last=1; fi
    nice -n 10 $PY pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs "$N" --fp32-refs --device cpu \
        >> logs/qj_part2_eval_val.log 2>&1
    say "validation pass: exit $? (1: some target failed, missing or stale; see logs/qj_part2_eval_val.log)"
    $PY pipeline/25_check_sim_real.py --ft-epochs "$N" --seeds 42 123 223 --flags-only >> "$LOG" 2>&1
    say "simulated vs real check: exit $? (1: some run below 0.7, listed above)"
    [ "$last" -eq 1 ] && break
    sleep 900
done
say "EXIT"
