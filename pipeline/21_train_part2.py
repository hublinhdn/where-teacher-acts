"""
where-teacher-acts: trains Part II, one process per run (pipeline/_qj_run_qat.py), resumable.

    python pipeline/21_train_part2.py --plan pilot                                  # G0: 4 runs (dated plan)
    python pipeline/21_train_part2.py --plan grid --ft-epochs 40 --phase core --first-student-only   # G1
    python pipeline/21_train_part2.py --plan grid --ft-epochs 40 --phase all       # core, then qakd_a, then seeds
    python pipeline/21_train_part2.py --plan grid --ft-epochs 40 --dry-run

pilot: EfficientNet-B1 with ConvNeXt-S, QAT(A) and QAKD(B), at N = 20 and N = 40.
anchor: the exploratory anchor control (_qj_common.EXTRA_ARMS), one run per student, seed 42, from B
distilled from ConvNeXt-S, with B's own FP32 model in the cosine term; its summary goes to
train_summary_anchor.csv so the grid's summary is never rewritten by it.
ed: the equal-dose comparison and the anchor on A (declared 2026-10-14), seed by seed (42, 123, 223) and
student by student: ed_ft, ed_kd (FP32 phase from A), then ed_none, ed_before, ed_during (quantization-aware
phase), then qanc_a; its summary goes to train_summary_ed.csv.
grid (the dated plan, docs/PLAN_AND_LOG.md), in priority phases (decided 2026-10-10), so that if the
budget runs out the core questions are complete on every student rather than all questions on a few:
  core    per student (pilot student first): QAT(A), then QAT(B) and QAKD(B) with every teacher (91 runs)
  qakd_a  QAKD(A) with every teacher, per student (42 runs, RQ4)
  seeds   QAT(A) and QAKD(B) with ConvNeXt-S, seeds 123 and 223, for QJ.SEED_REPEAT_STUDENTS (12 runs)
--phase all runs them in that order. Runs whose directory already holds a finished run are skipped,
so the pilot's runs count toward the grid when N matches.

Before each run the card must have room for it: the peak measured for its student (PEAK_GIB, with the
heaviest teacher on) plus HEADROOM_GIB for the CUDA context and a margin, or a fixed --min-free-gb for
every run; otherwise the script stops, or waits with --wait. The disk holding
weights/qj must keep --min-free-disk-gb free, or the script stops. A run that fails for want of memory
(CUDA or host out of memory, a DataLoader worker killed) is retried from scratch, after waiting for
room again, up to --attempts times; any other failure is recorded and the grid moves on; a run stopped
by a signal stops the grid. SIGTERM or SIGHUP to this script is passed on to the running run (which
marks itself failed and frees its directory) and stops the grid with exit 3. Each attempt carries a
token (environment QJ_ATTEMPT, recorded in run.json), so an attempt is judged on its own record and
log only. Per-run logs go to logs/qj_part2/ (every attempt appended); reports_qj/part2/train_summary.csv
is rewritten after each run. Exit codes: 0 all done, 1 some runs not done, 2 stopped for disk or (without
--wait) for the card, 3 stopped by a signal.
"""
import os
import sys
import csv
import json
import time
import uuid
import shutil
import signal
import argparse
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import _qj_common as QJ  # noqa: E402

LOG_DIR = os.path.join("logs", "qj_part2")
SUMMARY = os.path.join(QJ.REPORTS, "part2", "train_summary.csv")
SUMMARY_ANCHOR = os.path.join(QJ.REPORTS, "part2", "train_summary_anchor.csv")
SUMMARY_ED = os.path.join(QJ.REPORTS, "part2", "train_summary_ed.csv")
SUMMARY_SENS = os.path.join(QJ.REPORTS, "part2", "train_summary_sens.csv")
S42 = QJ.FP32_SEED
# Peak GPU memory of one Part II run per student (torch.cuda.max_memory_reserved, GiB), measured on the
# server 2026-10-10 over a short run that crosses both phases (approximate fold with AMP, then the exact
# fold in FP32) and validation, with the heaviest teacher on (QAKD(B) with ConvNeXt-B); a run without a
# teacher needs slightly less. The card must have the peak plus HEADROOM_GIB free (nvidia-smi free
# memory): about 0.3 GiB of CUDA context and the rest as margin against another tenant's growth.
PEAK_GIB = {
    "efficientnet_b1_timm": 5.66,   # 5.34 without a teacher, 5.44 with EfficientNet-B5 or DenseNet-161
    "efficientnet_b0_timm": 4.29,
    "ghostnet_100_timm": 2.86,
    "mobilenet_v3_large_tv": 2.81,
    "repvgg_a0_timm": 2.53,         # 2.37 without a teacher
    "squeezenet1_0_tv": 2.33,
    "resnet18_tv": 2.32,
}
HEADROOM_GIB = 1.0


def need_gb(student, fixed=None):
    """GiB the card must have free before a run of `student` starts."""
    if fixed:
        return fixed
    return round(PEAK_GIB.get(student, max(PEAK_GIB.values())) + HEADROOM_GIB, 2)


PHASES = ("core", "qakd_a", "seeds")


def plan(kind, ft_epochs, first_student_only=False, phase="all", only=None):
    """(arm, student, teacher, N, seed), in priority order; `only` keeps these students."""
    if kind == "pilot":
        s, t = QJ.PILOT_STUDENT, QJ.PILOT_TEACHER
        return [(arm, s, (t if arm != "qat_a" else None), n, S42)
                for n in QJ.PILOT_EPOCHS for arm in ("qat_a", "qakd_b")]
    students = [QJ.PILOT_STUDENT] + [s for s in QJ.STUDENTS if s != QJ.PILOT_STUDENT]
    if kind == "anchor":
        if only:
            students = [s for s in students if s in only]
        return [("qanc_b", s, QJ.ANCHOR_TEACHER, ft_epochs, S42) for s in students]
    if kind == "ed":
        if only:
            students = [s for s in students if s in only]
        t = QJ.ED_TEACHER
        return [job for seed in QJ.ED_SEEDS for s in students
                for job in (("ed_ft", s, None, ft_epochs, seed), ("ed_kd", s, t, ft_epochs, seed),
                            ("ed_none", s, None, ft_epochs, seed), ("ed_before", s, t, ft_epochs, seed),
                            ("ed_during", s, t, ft_epochs, seed), ("qanc_a", s, None, ft_epochs, seed))]
    if kind == "sens":
        # sensitivity analyses declared 2026-10-16: the self-anchored FP32 phase (three seeds), then two more
        # fine-tuning seeds of QAT(A), both from A of seed ED_START_SEED
        if only:
            students = [s for s in students if s in only]
        return ([("ed_anc", s, None, ft_epochs, seed) for seed in QJ.ED_SEEDS for s in students]
                + [("qat_a_fs", s, None, ft_epochs, seed) for seed in QJ.FS_SEEDS for s in students])
    if first_student_only:
        students = students[:1]
    if only:
        unknown = [s for s in only if s not in QJ.STUDENTS]
        if unknown:
            raise ValueError(f"not Part II students: {unknown}")
        students = [s for s in students if s in only]
    phases = PHASES if phase == "all" else (phase,)
    jobs = []
    if "core" in phases:
        for s in students:
            jobs.append(("qat_a", s, None, ft_epochs, S42))
            for t in QJ.TEACHERS:
                jobs.append(("qat_b", s, t, ft_epochs, S42))
                jobs.append(("qakd_b", s, t, ft_epochs, S42))
    if "qakd_a" in phases:
        for s in students:
            for t in QJ.TEACHERS:
                jobs.append(("qakd_a", s, t, ft_epochs, S42))
    if "seeds" in phases and not first_student_only:
        for seed in QJ.EXTRA_SEEDS:
            for s in [x for x in QJ.SEED_REPEAT_STUDENTS if x in students]:
                jobs.append(("qat_a", s, None, ft_epochs, seed))
                jobs.append(("qakd_b", s, QJ.SEED_REPEAT_TEACHER, ft_epochs, seed))
    return jobs


def status(arm, student, teacher, n, seed):
    try:
        with open(os.path.join(QJ.run_dir(arm, student, teacher, n, seed), "run.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def prerequisites(arm, student, teacher, seed=S42, n=None):
    """The runs this run reads must be finished: its start checkpoint (_qj_run_qat.py: for the grid the FP32
    run of the same seed, for the exploratory arms A of seed ED_START_SEED or the FP32 phase of the same
    seed) and, with a real teacher on, the teacher (seed 42)."""
    if arm in ("qat_b", "qakd_b", "qanc_b"):
        need = [QJ.fp32_dir("kd", student, teacher, seed)]
    elif arm in ("qat_a", "qakd_a"):
        need = [QJ.fp32_dir("base", student, None, seed)]
    elif arm in ("qanc_a", "ed_ft", "ed_kd", "ed_anc", "qat_a_fs"):
        need = [QJ.fp32_dir("base", student, None, QJ.ED_START_SEED)]
    elif arm in ("ed_none", "ed_during"):
        need = [QJ.run_dir("ed_ft", student, None, n, seed)]
    else:   # ed_before
        need = [QJ.run_dir("ed_kd", student, teacher, n, seed)]
    if arm in ("qakd_a", "qakd_b", "ed_kd", "ed_during"):
        need.append(QJ.fp32_dir("teacher", teacher, None, S42))
    missing = []
    for d in need:
        try:
            with open(os.path.join(d, "run.json")) as f:
                if json.load(f).get("status") != "done":
                    missing.append(d)
        except (OSError, ValueError):
            missing.append(d)
    return missing


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
    except Exception:  # noqa: BLE001  no card or broken driver: no room
        return 0.0


def free_disk_gb(path=QJ.WEIGHT_DIR):
    try:
        return shutil.disk_usage(os.path.realpath(path)).free / 2 ** 30
    except OSError:
        return 0.0


# what a run that died for want of memory leaves in run.json or at the end of its log. Not the bare
# "DataLoader worker": PyTorch re-raises ANY exception of a dataset worker as "Caught <Error> in
# DataLoader worker process N", which is deterministic; only a worker killed (host out of memory, shared
# memory exhausted) is transient.
TRANSIENT = ("out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED",
             "Unable to find a valid cuDNN algorithm", "MemoryError", "Cannot allocate memory",
             "is killed by signal", "exited unexpectedly")
SIGNAL_RC = (-2, -15, -1, 130, 143)


def classify(rc, st, log_path, offset=0, token=None):
    """'done', 'stopped' (an operator's signal), 'transient' (memory) or 'failed'. Trusts run.json only
    when it carries this attempt's token (an attempt that dies before claiming its directory leaves the
    previous attempt's record there), and reads only what this attempt appended to the log (from
    `offset`), its last 8 KB."""
    own = token is None or st.get("attempt") == token
    st = st if own else {}
    if st.get("status") == "done" and (rc == 0 or token is not None):
        return "done"
    err = str(st.get("error", "")) if st.get("status") == "failed" else ""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(offset, f.tell() - 8192))
            tail = f.read().decode(errors="ignore")
    except OSError:
        tail = ""
    if "stopped by signal" in err or "stopped by signal" in tail or rc in SIGNAL_RC:
        return "stopped"
    if any(m in err or m in tail for m in TRANSIENT) or rc == -9:
        return "transient"
    return "failed"


_CHILD = []       # the running attempt, so a signal to the scheduler reaches it
_STOP = []        # signals received


def _on_signal(signum, _frame):
    _STOP.append(signum)
    if _CHILD and _CHILD[0].poll() is None:
        _CHILD[0].send_signal(signum)      # the runner turns it into a failed run and frees its directory
    else:
        raise SystemExit(3)                # between runs (waiting): stop at once


def run_attempt(cmd, log, token):
    """Run one attempt as a child process; its return code."""
    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, QJ_ATTEMPT=token))
    _CHILD[:] = [child]
    try:
        return child.wait()
    finally:
        _CHILD[:] = []


def write_summary(jobs, path=SUMMARY):
    """Status of every run of `jobs` (the whole plan, whatever subset this call trains)."""
    if not jobs:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for job in jobs:
        st = status(*job)
        arm, s, t, n, seed = job
        rows.append({"arm": arm, "student": s, "teacher": t or "", "ft_epochs": n, "seed": seed,
                     "status": st.get("status", "missing"), "val_mAP_sim": st.get("val_mAP_sim", ""),
                     "best_epoch": st.get("best_epoch", ""), "minutes": st.get("minutes", ""),
                     "error": st.get("error", "")})
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=("pilot", "grid", "anchor", "ed", "sens"))
    ap.add_argument("--ft-epochs", type=int, default=None, help="N for the grid (chosen by G0)")
    ap.add_argument("--first-student-only", action="store_true", help="G1: only the pilot student")
    ap.add_argument("--phase", default="all", choices=PHASES + ("all",))
    ap.add_argument("--students", nargs="*", default=None, help="only these students (priority order kept)")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="a fixed GiB threshold for every run instead of the per-student one (need_gb)")
    ap.add_argument("--min-free-disk-gb", type=float, default=5.0)
    ap.add_argument("--attempts", type=int, default=3, help="attempts per run when it fails for memory")
    ap.add_argument("--yield-to", nargs="*", default=[],
                    help="wait while a process whose command line contains one of these is alive")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.plan in ("grid", "anchor", "ed", "sens") and not a.ft_epochs:
        sys.exit(f"--plan {a.plan} needs --ft-epochs (the N chosen by G0)")
    if a.attempts < 1:
        sys.exit("--attempts must be at least 1")

    jobs = plan(a.plan, a.ft_epochs, a.first_student_only, a.phase, a.students)
    whole = plan(a.plan, a.ft_epochs)          # the summary always covers the whole plan
    todo = [j for j in jobs if status(*j).get("status") != "done"]
    print(f"[part2] {a.plan}: {len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to train", flush=True)
    if a.dry_run:
        for j in todo:
            need = need_gb(j[1], a.min_free_gb)
            print("   ", *[x if x is not None else "-" for x in j], f"needs {need:g} GB", "missing:",
                  prerequisites(j[0], j[1], j[2], j[4], j[3]) or "none")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _on_signal)
    try:
        failed = train(a, whole, todo)
    finally:
        write_summary(whole, {"anchor": SUMMARY_ANCHOR, "ed": SUMMARY_ED, "sens": SUMMARY_SENS}.get(a.plan, SUMMARY))
    print(f"[part2] finished: {len(failed)} not done", flush=True)
    for f in failed:
        print("   ", *[x if x is not None else "-" for x in f], flush=True)
    sys.exit(1 if failed else 0)


def train(a, jobs, todo):
    """The training loop of main(); returns the runs left not done. Exits 2 or 3 to stop the grid."""
    failed = []
    for i, (arm, s, t, n, seed) in enumerate(todo, 1):
        missing = prerequisites(arm, s, t, seed, n)
        if missing:
            print(f"[{i}/{len(todo)}] skip {arm} {s} {t or ''} seed {seed}: FP32 prerequisites not finished: "
                  f"{missing}", flush=True)
            failed.append((arm, s, t, n, seed, "prerequisites"))
            continue
        need = need_gb(s, a.min_free_gb)
        name = f"{arm}__{s}__{t or 'none'}__ft{n}__seed{seed}"
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        cmd = [sys.executable, os.path.join("pipeline", "_qj_run_qat.py"), "--arm", arm, "--student", s,
               "--ft-epochs", str(n), "--seed", str(seed)] + (["--teacher", t] if t else [])
        for attempt in range(1, a.attempts + 1):
            disk = free_disk_gb()
            if disk < a.min_free_disk_gb:
                print(f"[part2] stop before {arm} {s}: {disk:.1f} GB free on the disk of {QJ.WEIGHT_DIR}, "
                      f"below {a.min_free_disk_gb} GB", flush=True)
                write_summary(jobs)
                sys.exit(2)
            waited = False
            while (queue_active(a.yield_to) or free_gb() < need
                   or (waited and not card_stays_free(need))):
                q = queue_active(a.yield_to)
                msg = (f"another job queue is active (pids {q})" if q
                       else f"GPU has {free_gb():.1f} GB free, below the {need:g} GB this run needs")
                if not a.wait:
                    print(f"[part2] stop before {arm} {s}: {msg}. Rerun when the card is free.", flush=True)
                    write_summary(jobs)
                    sys.exit(2)
                print(f"[part2] waiting {a.poll:g} min: {msg}", flush=True)
                waited = True
                time.sleep(a.poll * 60)
            t0 = time.time()
            print(f"[{i}/{len(todo)}] {arm} {s}{' <- ' + t if t else ''} N={n} seed {seed}"
                  f"{f' (attempt {attempt})' if attempt > 1 else ''} ...", flush=True)
            token = uuid.uuid4().hex
            with open(log_path, "a") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {attempt} ({token}): "
                          f"{' '.join(cmd)}\n")
                log.flush()
                offset = log.tell()
                rc = run_attempt(cmd, log, token)
            st = status(arm, s, t, n, seed)
            kind = classify(rc, st, log_path, offset, token)
            if _STOP:
                kind = "stopped"
            if kind == "done":
                print(f"      done in {(time.time() - t0) / 60:.0f} min, val mAP (sim INT8) "
                      f"{st.get('val_mAP_sim')} at epoch {st.get('best_epoch')}", flush=True)
                break
            err = st.get("error") if st.get("attempt") == token else None
            print(f"      FAILED ({kind}, exit {rc}) after {(time.time() - t0) / 60:.0f} min: "
                  f"{err or 'see ' + log_path}", flush=True)
            if kind == "stopped":
                print("[part2] the run was stopped by a signal: stopping the grid", flush=True)
                sys.exit(3)
            if kind != "transient" or attempt == a.attempts:
                failed.append((arm, s, t, n, seed, kind))
                break
            print(f"      out of memory: waiting {a.poll:g} min, then attempt {attempt + 1} of {a.attempts}",
                  flush=True)
            time.sleep(a.poll * 60)
        write_summary(jobs)
    return failed


if __name__ == "__main__":
    main()
