#!/usr/bin/env bash
# where-teacher-acts: Part II on the server, up to gate G1, meant to run inside tmux:
#
#   tmux new -c <repo> -s qj_part2 -d \
#       "bash <repo>/run_part2_server.sh"
#
# 0. waits until the FP32 supervisor has finished the test-fold benchmark B1 (its FP32 evaluation
#    uses the card; B2 that follows is CPU work);
# 1. gate F2 on VALIDATION: INT8 floors of the seven A (22_eval_part2.py --kinds fp32_A, then
#    13_check_fp32.py); stops for discussion if the floors no longer spread;
# 2. pilot G0: EfficientNet-B1 with ConvNeXt-S, QAT(A) and QAKD(B) at N = 20 and 40;
# 3. measures the pilot and its FP32 references on validation; 4. gate G0 (23_gate_g0.py), stops
#    for discussion if it fails;
# 5. the first student's core cells at the chosen N (G1: QAT(A), QAT(B) and QAKD(B) with every
#    teacher; QAKD(A) comes later with the other students, decided 2026-10-10);
# 6. measures it on validation; then stops, so G1 can be reviewed before the other six students.
# Nothing here reads the test fold. Logs: logs/qj_part2_main.log plus one log per step.
# Every step is resumable (finished runs and measurements are skipped), so after a stop the script is
# simply started again. Training waits for room on the shared card using the thresholds measured for
# Part II (21_train_part2.py defaults) and retries a run that failed for want of memory.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_part2_main.log
S1=efficientnet_b1_timm
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "start in $(pwd) (repo $(git rev-parse --short HEAD)); waiting for the FP32 benchmark B1"
until grep -q "benchmark exit" logs/qj_server_main.log 2>/dev/null; do sleep 300; done

say "step 1: gate F2 on validation (INT8 floors of the seven A)"
$PY pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs 20 --fp32-refs --kinds fp32_A --seeds 42 >> logs/qj_part2_eval_val.log 2>&1
$PY pipeline/13_check_fp32.py >> "$LOG" 2>&1 || { say "F2 did not pass or is incomplete; stopping for discussion"; say "EXIT"; exit 0; }

say "step 2: pilot"
$PY pipeline/21_train_part2.py --plan pilot --wait >> logs/qj_part2_train.log 2>&1
rc=$?
say "pilot exit $rc"
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    say "the pilot stopped (2: disk below its floor, 3: stopped by a signal, other: the scheduler died); stopping here"
    say "EXIT"; exit 0
fi

say "step 3: pilot on validation"
$PY pipeline/22_eval_part2.py --fold val --plan pilot --fp32-refs >> logs/qj_part2_eval_val.log 2>&1
say "pilot evaluation exit $?"

say "step 4: gate G0"
$PY pipeline/23_gate_g0.py >> "$LOG" 2>&1 || { say "G0 did not pass; stopping for discussion"; say "EXIT"; exit 0; }
N=$($PY -c "import json; print(json.load(open('reports_qj/part2/g0.json'))['chosen_N'])")

say "step 5: first student's core cells at N=$N (G1)"
$PY pipeline/21_train_part2.py --plan grid --ft-epochs "$N" --phase core --first-student-only --wait >> logs/qj_part2_train.log 2>&1
rc=$?
say "G1 grid exit $rc"
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    say "the grid stopped (2: disk below its floor, 3: stopped by a signal, other: the scheduler died); stopping here"
    say "EXIT"; exit 0
fi

say "step 6: first student on validation"
$PY pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs "$N" --fp32-refs --students "$S1" --seeds 42 >> logs/qj_part2_eval_val.log 2>&1
say "G1 evaluation exit $?; G1 is ready for review, stopping here"
say "EXIT"
