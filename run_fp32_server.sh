#!/usr/bin/env bash
# where-teacher-acts: the FP32 phase on the server, meant to run inside tmux (absolute paths, so tmux's own
# start directory does not matter):
#
#   tmux new -c <repo> -s qj_main -d \
#       "bash <repo>/run_fp32_server.sh"
#
# 1. trains the FP32 grid: 6 teachers, 7 A, 42 B, 12 seed repeats (pipeline/10_train_fp32.py,
#    sharing the card with other tenants (decided 2026-10-09): a run starts when 4 GB are free,
#    enough for any run of this grid, which peaks at about 3 GB);
# 2. benchmarks every FP32 model on the test fold (pipeline/11_bench_fp32_ptq.py, tag "main");
# 3. measures calibration noise on the test fold (pipeline/12_bench_calib_noise.py, tag "main").
#
# Every step appends to its own log under logs/; this script's own log is logs/qj_server_main.log.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python3}
LOG=logs/qj_server_main.log
mkdir -p logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "start in $(pwd) (repo $(git rev-parse --short HEAD))"

say "step 1: FP32 grid"
$PY pipeline/10_train_fp32.py --wait --min-free-gb 4 >> logs/qj_fp32_grid.log 2>&1
say "FP32 grid exit $?"

say "step 2: test-fold benchmark"
$PY pipeline/11_bench_fp32_ptq.py --tag main >> logs/qj_bench_main.log 2>&1
say "benchmark exit $?"

say "step 3: calibration noise"
$PY pipeline/12_bench_calib_noise.py --tag main >> logs/qj_noise_main.log 2>&1
say "noise exit $?"
say "EXIT"
