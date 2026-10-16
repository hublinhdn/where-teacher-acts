"""
where-teacher-acts: trains this study's FP32 grid, one process per model, in dependency order: the 6
teachers, the 7 undistilled students (A), the 42 distilled students (B), each B from this study's
own teacher, then the training-noise repeats (A and B with ConvNeXt-S for three students, seeds 123
and 223; 12 runs). Every run goes through pipeline/_qj_run.py and owns its directory under
weights/qj/fp32 (pipeline/_qj_common.fp32_dir).

Finished runs are skipped. Before starting a run the script reads the card's free memory; when it is
below --min-free-gb (another job holds the card) it stops instead of starting, unless --wait is
given, in which case it re-checks every --poll minutes. A failed run is recorded and the grid moves
on; its kd pairs are skipped if the failure was a teacher.

    python pipeline/10_train_fp32.py --dry-run          # the plan and what is already done
    python pipeline/10_train_fp32.py                    # everything still missing
    python pipeline/10_train_fp32.py --roles teacher base

Per-run logs go to logs/qj_fp32/; a summary of every run's status is rewritten to
reports_qj/fp32/train_summary.csv after each run.
"""
import os
import sys
import csv
import json
import time
import argparse
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import _qj_common as QJ  # noqa: E402

LOG_DIR = os.path.join("logs", "qj_fp32")
SUMMARY = os.path.join(QJ.REPORTS, "fp32", "train_summary.csv")


def plan(roles, seeds=True):
    """(role, model, teacher, seed) in dependency order; the seed repeats come last."""
    S42 = QJ.FP32_SEED
    jobs = []
    if "teacher" in roles:
        jobs += [("teacher", t, None, S42) for t in QJ.TEACHERS]
    if "base" in roles:
        jobs += [("base", s, None, S42) for s in QJ.STUDENTS]
    if "kd" in roles:
        jobs += [("kd", s, t, S42) for s in QJ.STUDENTS for t in QJ.TEACHERS]
    if seeds:
        for seed in QJ.EXTRA_SEEDS:
            for s in QJ.SEED_REPEAT_STUDENTS:
                if "base" in roles:
                    jobs.append(("base", s, None, seed))
                if "kd" in roles:
                    jobs.append(("kd", s, QJ.SEED_REPEAT_TEACHER, seed))
    return jobs


def status(role, model, teacher, seed=QJ.FP32_SEED):
    try:
        with open(os.path.join(QJ.fp32_dir(role, model, teacher, seed), "run.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def queue_active(patterns):
    """Pids of other processes whose command line contains one of `patterns` (for example another
    tenant's job queue), so no run starts while that queue may launch its next job. Skips this
    process and any of our orchestrators (their own command lines carry --yield-to)."""
    hits = []
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "--yield-to" in cmd:
            continue
        if any(p in cmd for p in patterns):
            hits.append(int(pid))
    return hits


def card_stays_free(min_free, checks=3, gap=60):
    """True if the card has min_free GB free on `checks` readings `gap` seconds apart. Used after a
    wait, so a job queue that frees the card for a few seconds between its jobs does not let a run
    slip in and starve the queue's next job (or be starved by it)."""
    for k in range(checks):
        if free_gb() < min_free:
            return False
        if k < checks - 1:
            time.sleep(gap)
    return True


def free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()
        return float(out[0]) / 1024
    except Exception:  # noqa: BLE001  no card visible or driver broken: treat as no room, never as room
        return 0.0


def write_summary(jobs):
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    rows = []
    for role, model, teacher, seed in jobs:
        st = status(role, model, teacher, seed)
        rows.append({"role": role, "model": model, "teacher": teacher or "", "seed": seed,
                     "status": st.get("status", "missing"),
                     "val_mAP": st.get("val_mAP", ""), "best_epoch": st.get("best_epoch", ""),
                     "minutes": st.get("minutes", ""), "error": st.get("error", "")})
    tmp = f"{SUMMARY}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=list(QJ.ROLES), choices=QJ.ROLES)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--yield-to", nargs="*", default=[],
                    help="wait while a process whose command line contains one of these is alive")
    ap.add_argument("--wait", action="store_true", help="wait for the card instead of stopping")
    ap.add_argument("--poll", type=float, default=10.0, help="minutes between checks with --wait")
    ap.add_argument("--no-seeds", action="store_true", help="skip the training-noise seed repeats")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = plan(a.roles, seeds=not a.no_seeds)
    todo = [j for j in jobs if status(*j).get("status") != "done"]   # j = (role, model, teacher, seed)
    print(f"[fp32] {len(jobs)} runs planned, {len(jobs) - len(todo)} done, {len(todo)} to train", flush=True)
    if a.dry_run:
        for j in todo:
            print("   ", *[x or "-" for x in j])
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    failed_teachers, failures = set(), 0
    for i, (role, model, teacher, seed) in enumerate(todo, 1):
        if role == "kd" and (teacher in failed_teachers or status("teacher", teacher, None).get("status") != "done"):
            print(f"[{i}/{len(todo)}] skip kd {model} <- {teacher}: teacher run not finished", flush=True)
            continue
        waited = False
        while (queue_active(a.yield_to) or free_gb() < a.min_free_gb
               or (waited and not card_stays_free(a.min_free_gb))):
            q = queue_active(a.yield_to)
            msg = (f"another job queue is active (pids {q})" if q else f"GPU has {free_gb():.1f} GB free, below {a.min_free_gb} GB: another job is using the card")
            if not a.wait:
                print(f"[fp32] stop before {role} {model}: {msg}. Rerun when the card is free.", flush=True)
                write_summary(jobs)
                sys.exit(2)
            print(f"[fp32] waiting {a.poll:g} min: {msg}", flush=True)
            waited = True
            time.sleep(a.poll * 60)
        name = f"{role}__{model}__{teacher or 'none'}__seed{seed}"
        cmd = [sys.executable, os.path.join("pipeline", "_qj_run.py"), "--role", role, "--model", model,
               "--seed", str(seed)]
        if teacher:
            cmd += ["--teacher", teacher]
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {role} {model}{' <- ' + teacher if teacher else ''} seed {seed} ...", flush=True)
        with open(os.path.join(LOG_DIR, f"{name}.log"), "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        st = status(role, model, teacher, seed)
        if rc != 0 or st.get("status") != "done":
            failures += 1
            if role == "teacher":
                failed_teachers.add(model)
            print(f"      FAILED (exit {rc}): {st.get('error', 'see ' + name + '.log')}", flush=True)
        else:
            print(f"      done in {(time.time() - t0) / 60:.0f} min, val mAP {st.get('val_mAP')}"
                  f" at epoch {st.get('best_epoch')}", flush=True)
        write_summary(jobs)
    print(f"[fp32] finished: {failures} failed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
