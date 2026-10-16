"""
where-teacher-acts: deployment measurement of Part II runs, on the validation fold (decisions) or the
test fold (results).

For every finished Part II run: export the network with its learned clipping ranges as Clip nodes
(_qakd_export.export_clipped_onnx, the inherited exporter), measure that FP32 graph, quantize it
with ONNX Runtime static PTQ (MinMax primary, percentile 99.9 secondary; per-channel int8 weights,
per-tensor uint8 activations) and measure INT8 on the CPU execution provider, all at this study's
image size (224). With --fp32-refs the plain FP32 A and B the runs started from are measured on the
same fold through the same exporter path as the FP32 benchmark, for the G0 thresholds.

    --fold test  queries fold 4; calibration = 256 images (seed 123) from every image except fold-4
                 queries, the inherited rule (identical to the FP32 benchmark).
    --fold val   queries fold 0; calibration excludes fold-0 and fold-4 queries, so a decision never
                 calibrates on the queries it scores and never touches a test image.

    python pipeline/22_eval_part2.py --fold val --plan pilot --fp32-refs
    python pipeline/22_eval_part2.py --fold test --plan grid --ft-epochs 20
    python pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs 20 --fp32-refs --kinds fp32_A   # gate F2
    python pipeline/22_eval_part2.py --fold val --plan anchor --ft-epochs 40 --tag anchor              # anchor control
    python pipeline/22_eval_part2.py --fold val --plan ed --ft-epochs 40 --tag ed                      # equal dose

    --device cpu  measures FP32 on the CPU too (INT8 always runs on the CPU), so an evaluation can run
                  while a training run holds the card.
    --require-all exit non-zero unless every target was finished and measured (the gates and the
                  result tables use it; by default unfinished runs are listed and skipped).

The FP32 references of a run are the checkpoints it started from: A and B of the run's own seed.
Resumable by checkpoint fingerprint; reports_qj/part2/<tag>/eval_<fold>.csv; one process per tag and
fold (a lock left by a process that no longer exists on this host is removed). Every row records
whether ONNX Runtime's pre-processing ran on the graph it quantized ("preprocess").
"""
import os
import sys
import csv
import gc
import json
import time
import signal
import socket
import argparse
import tempfile
import importlib

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import resource  # noqa: E402
import torch  # noqa: E402
torch.multiprocessing.set_sharing_strategy("file_system")
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(_soft, 65536), _hard), _hard))

import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from journal2_pipeline._provenance import fingerprint  # noqa: E402
from journal2_pipeline import _quant_utils as QU  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402
from _qj_io import take_lock  # noqa: E402
from journal2_pipeline._eval_utils import (build_eval_loader_epillid, extract_features_onnx,  # noqa: E402
                                           calculate_both_sides_map)

PCT = 99.9
FIELDS = ["kind", "arm", "student", "teacher", "ft_epochs", "seed", "path", "sha", "graph", "mAP_FP32",
          "mAP_INT8_minmax", "mAP_INT8_pct99.9", "preprocess", "fp32_provider", "seconds", "status", "error",
          "timestamp"]


def _exit_on_signal(signum, _frame):
    raise SystemExit(f"stopped by signal {signum}")


def measure(onnx_path, loader, provider, device):
    feats, labels, subs, is_ref = extract_features_onnx(onnx_path, loader, provider=provider,
                                                        desc=os.path.basename(onnx_path))
    m, _ = calculate_both_sides_map(feats, labels, subs, is_ref, device=device)
    return round(float(m), 4)


def save_rows(path, rows):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows.values())
    os.replace(tmp, path)


def targets(plan_kind, ft_epochs, fp32_refs, root):
    part2 = importlib.import_module("21_train_part2")
    jobs = part2.plan(plan_kind, ft_epochs)
    out = []
    if fp32_refs:
        seen = set()
        for arm, s, t, n, seed in jobs:
            for kind, key in (("fp32_A", ("base", s, None, seed)), ("fp32_B", ("kd", s, t, seed))):
                if kind == "fp32_B" and (arm not in ("qat_b", "qakd_b")):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                d = QJ.fp32_dir(*key, root=os.path.join(QJ.WEIGHT_DIR, "smoke") if root else None)
                out.append((kind, "", s, key[2] or "", "", seed, d))
    for arm, s, t, n, seed in jobs:
        # the FP32 phase of the equal-dose comparison has no quantizer: measured like an FP32 reference
        kind = "ed_fp32" if arm in QJ.ED_FP32_ARMS else "part2"
        out.append((kind, arm, s, t or "", n, seed, QJ.run_dir(arm, s, t, n, seed, root=root)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=("val", "test"))
    ap.add_argument("--plan", required=True, choices=("pilot", "grid", "anchor", "ed", "sens"))
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--fp32-refs", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--smoke", action="store_true", help="measure smoke runs under weights/qj/smoke")
    ap.add_argument("--students", nargs="*", default=None, help="only these students")
    ap.add_argument("--kinds", nargs="*", default=None, choices=("fp32_A", "fp32_B", "part2", "ed_fp32"),
                    help="only these kinds of target (fp32_A, fp32_B: FP32 references; part2: Part II runs)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="only targets of these seeds (the grid is seed 42; 123 and 223 are the repeats)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu"),
                    help="cpu: FP32 inference and the mAP on the CPU (leaves the card to training)")
    ap.add_argument("--require-all", action="store_true",
                    help="exit non-zero unless every target was finished and measured")
    a = ap.parse_args()
    if a.plan in ("grid", "anchor", "ed", "sens") and not a.ft_epochs:
        sys.exit(f"--plan {a.plan} needs --ft-epochs")
    if a.plan in ("anchor", "ed", "sens") and a.tag == "main":
        sys.exit(f"the {a.plan} runs are measured under their own tag (--tag {a.plan}), never into the grid's files")
    sys.path.insert(0, THIS)
    fold = S.VAL_FOLD if a.fold == "val" else S.TEST_FOLD
    exclude = {S.VAL_FOLD, S.TEST_FOLD} if a.fold == "val" else {S.TEST_FOLD}
    root = os.path.join(QJ.WEIGHT_DIR, "smoke", "runs") if a.smoke else None
    tag = f"smoke-{a.tag}" if a.smoke else a.tag
    out_dir = os.path.join(QJ.REPORTS, "part2", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"eval_{a.fold}.csv")
    lock = os.path.join(out_dir, f".lock_{a.fold}")
    try:
        fd = take_lock(lock, f"{out_csv} is being written by another process (or one on another host); "
                             f"check, then delete {lock}")
    except FileExistsError as exc:
        sys.exit(str(exc))
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _exit_on_signal)
    done = {}
    unfinished, missing = [], []
    try:
        S.load_split()
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                done = {(r["kind"], r["arm"], r["student"], r["teacher"], r["ft_epochs"], r["seed"]): r
                        for r in csv.DictReader(f)}
        todo = targets(a.plan, a.ft_epochs, a.fp32_refs, root)
        if a.students:
            todo = [x for x in todo if x[2] in a.students]
        if a.kinds:
            todo = [x for x in todo if x[0] in a.kinds]
        if a.seeds:
            todo = [x for x in todo if int(x[5]) in a.seeds]
        # first pass, cheap: which targets need a measurement now
        pending = []
        for kind, arm, s, t, n, seed, rdir in todo:
            key = (kind, arm, s, t, str(n), str(seed))
            path = os.path.join(rdir, "model.pth")
            try:
                with open(os.path.join(rdir, "run.json")) as f:
                    run_meta = json.load(f)
            except (OSError, ValueError):
                run_meta = {}
            finished = run_meta.get("status") == "done"
            # the graph this target is deployed as: a Part II run as it was trained (its run.json; runs
            # recorded before deploy_graph existed were plain), an FP32 reference by this study's rule
            graph = run_meta.get("deploy_graph", "plain") if kind == "part2" else QJ.deploy_graph(s)
            prev = done.get(key)
            if not finished:
                unfinished.append(key)
                if prev and prev["status"] == "SUCCESS":
                    # measured before, but the run was claimed again since (redone on purpose), and the
                    # claim deleted the checkpoint that row measured: keep the numbers, stop trusting them
                    prev["status"] = "STALE"
                    prev["error"] = "the run was claimed again after this measurement"
                    save_rows(out_csv, done)
                    print(f"STALE {kind} {arm} {s} {t} N{n} s{seed}: the run is no longer done; its old row is "
                          f"marked STALE", flush=True)
                continue
            if not os.path.exists(path) or (kind == "part2" and not os.path.exists(path + ".clip.pt")):
                print(f"MISSING {kind} {arm} {s} {t} N{n} s{seed}: the run is marked done but its checkpoint is "
                      f"gone; the previous row (if any) is kept", flush=True)
                missing.append(key)
                continue
            sha = fingerprint(path)["sha"]
            # rows written before the graph column existed were plain graphs
            if prev and prev["status"] == "SUCCESS" and prev["sha"] == sha and (prev.get("graph") or "plain") == graph:
                continue
            pending.append((kind, arm, s, t, n, seed, path, sha, graph))
        print(f"[part2-eval] {a.fold} (fold {fold}, calibration excludes queries of folds {sorted(exclude)}): "
              f"{len(todo)} targets, {len(pending)} to measure -> {out_csv}", flush=True)
        if pending:
            gpu = torch.cuda.is_available() and a.device != "cpu"
            device = torch.device("cuda" if gpu else "cpu")
            fp32_ep = "CUDAExecutionProvider" if gpu else "CPUExecutionProvider"
            eval_loader = build_eval_loader_epillid(img_size=QJ.IMG_SIZE, batch_size=32, num_workers=4,
                                                    test_fold=fold)
            calib_loader = S.calibration_loader(QJ.IMG_SIZE, exclude, QJ.CALIB_SEED, QJ.CALIB_N)
            from _qakd_export import export_clipped_onnx
        for i, (kind, arm, s, t, n, seed, path, sha, graph) in enumerate(pending, 1):
            key = (kind, arm, s, t, str(n), str(seed))
            row = {"kind": kind, "arm": arm, "student": s, "teacher": t, "ft_epochs": n, "seed": seed, "path": path,
                   "sha": sha, "graph": graph, "mAP_FP32": "", "mAP_INT8_minmax": "",
                   "mAP_INT8_pct99.9": "", "preprocess": "",
                   "fp32_provider": fp32_ep, "seconds": "", "status": "FAILED", "error": "",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t0 = time.time()
            try:
                with tempfile.TemporaryDirectory(prefix="qj_p2_") as tmp:
                    f32 = os.path.join(tmp, "fp32.onnx")
                    if kind == "part2":
                        export_clipped_onnx(s, path, path + ".clip.pt", f32, num_classes=QJ.NUM_CLASSES,
                                            img_size=QJ.IMG_SIZE)
                    else:
                        export_fp32_onnx(s, path, f32, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                                         pooling="gem", device=torch.device("cpu"),
                                         fuse_branches=graph == "fused")
                    row["mAP_FP32"] = measure(f32, eval_loader, fp32_ep, device)
                    pp = []
                    for rule, col in (("minmax", "mAP_INT8_minmax"), ("pct", "mAP_INT8_pct99.9")):
                        i8 = os.path.join(tmp, f"int8_{rule}.onnx")
                        kw = (dict(calibration="minmax") if rule == "minmax"
                              else dict(calibration="percentile", extra_options={"CalibPercentile": PCT}))
                        run_ptq(f32, i8, calib_loader, per_channel=True, **kw)
                        pp.append("ok" if QU.LAST_PREPROCESS.get("ok") else f"raw ({QU.LAST_PREPROCESS.get('error')})")
                        row[col] = measure(i8, eval_loader, "CPUExecutionProvider", device)
                    row["preprocess"] = pp[0] if pp[0] == pp[1] else " / ".join(pp)
                row["status"] = "SUCCESS"
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - t0)
            done[key] = row
            save_rows(out_csv, done)
            print(f"[{i:3d}/{len(pending)}] {row['status']:7s} {kind:6s} {arm:7s} {s:24s} {t:24s} N{n} s{seed}  "
                  f"FP32 {row['mAP_FP32']}  INT8 {row['mAP_INT8_minmax']}  pct {row['mAP_INT8_pct99.9']}  "
                  f"{row['seconds']}s {'' if row['preprocess'] in ('ok', '') else 'PREPROCESS ' + row['preprocess']} "
                  f"{row['error']}", flush=True)
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} measured, {len(failed)} failed; {len(unfinished)} targets not finished yet, "
          f"{len(missing)} finished but without a checkpoint")
    for k in missing + unfinished[:10]:
        print("    not measured:", *k)
    if len(unfinished) > 10:
        print(f"    ... and {len(unfinished) - 10} more not finished yet")
    ok = not failed and not missing and (not a.require_all or not unfinished)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
