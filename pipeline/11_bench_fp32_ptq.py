"""
where-teacher-acts: the deployment benchmark of this study's FP32 models on the test fold.

For every finished FP32 run (the 7 undistilled students A, the 42 distilled students B and the 12
seed repeats; the 6 teachers in FP32 only, for reference) this measures, on test fold 4 of _qj_split: FP32 mAP, and INT8
mAP after ONNX Runtime static PTQ with MinMax (primary) and with percentile 99.9 (secondary). It uses
the very functions of the inherited pipeline, with the test fold as their argument, at this study's one
image size (_qj_common.IMG_SIZE, 224 x 224): the evaluation loader (fold-4
consumer queries against all 9,804 references, consumer-side both-sides mAP) and the calibration
builder (256 images drawn with seed 123 from every image except the test fold's consumer queries;
per-channel int8 weights, per-tensor uint8 activations). FP32 runs on CUDA, INT8 on the CPU
execution provider.

Each row carries the fingerprint of the checkpoint it measured and the run's validation mAP (fold 0,
from run.json). Resumable: a checkpoint already measured with the same fingerprint is skipped.
Results go to reports_qj/bench/<tag>/fp32_ptq.csv; the tag defaults to today, one process per tag.

    python pipeline/11_bench_fp32_ptq.py
    python pipeline/11_bench_fp32_ptq.py --smoke      # the smoke-test runs under weights/qj/smoke
"""
import os
import sys
import csv
import gc
import json
import time
import socket
import argparse
import tempfile
import subprocess

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
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq, build_calibration_loader  # noqa: E402
from journal2_pipeline._eval_utils import (build_eval_loader_epillid, extract_features_onnx,  # noqa: E402
                                           calculate_both_sides_map)

PCT = 99.9
FIELDS = ["role", "model", "teacher", "seed", "path", "sha", "graph", "val_mAP_fold0", "best_epoch",
          "test_mAP_FP32", "test_mAP_INT8_minmax", "test_mAP_INT8_pct99.9", "seconds", "status", "error",
          "timestamp"]


def runs(roles):
    S42 = QJ.FP32_SEED
    out = []
    if "teacher" in roles:
        out += [("teacher", t, None, S42) for t in QJ.TEACHERS]
    if "base" in roles:
        out += [("base", s, None, S42) for s in QJ.STUDENTS]
    if "kd" in roles:
        out += [("kd", s, t, S42) for s in QJ.STUDENTS for t in QJ.TEACHERS]
    for seed in QJ.EXTRA_SEEDS:
        for s in QJ.SEED_REPEAT_STUDENTS:
            if "base" in roles:
                out.append(("base", s, None, seed))
            if "kd" in roles:
                out.append(("kd", s, QJ.SEED_REPEAT_TEACHER, seed))
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=list(QJ.ROLES), choices=QJ.ROLES)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="measure the smoke-test runs under weights/qj/smoke")
    a = ap.parse_args()
    root = os.path.join(QJ.WEIGHT_DIR, "smoke") if a.smoke else None
    tag = a.tag or (f"smoke-{time.strftime('%Y%m%d')}" if (a.smoke or a.limit) else time.strftime("%Y%m%d"))
    out_dir = os.path.join(QJ.REPORTS, "bench", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "fp32_ptq.csv")
    lock = os.path.join(out_dir, ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{out_dir} is in use by another process, or by one that was killed; check, then delete {lock}")
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)

    try:
        S.load_split()                             # refuse to measure under anything but the design's split
        import onnxruntime as ort
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        meta_path = os.path.join(out_dir, "meta.json")
        meta = {"tag": tag, "sessions": [], "test_fold": S.TEST_FOLD,
                "calibration": {"rule": "minmax", "N": QJ.CALIB_N, "seed": QJ.CALIB_SEED,
                                "pool": f"every image except fold-{S.TEST_FOLD} consumer queries",
                                "secondary": f"percentile {PCT:g}"}}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        meta["sessions"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(),
                                 "repo_commit": commit, "onnxruntime": ort.__version__, "torch": torch.__version__})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        done = {}
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                done = {(r["role"], r["model"], r["teacher"], r["seed"]): r for r in csv.DictReader(f)}
        todo = runs(a.roles)
        if a.limit:
            todo = todo[:a.limit]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fp32_ep = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        loaders = {}

        def get_loaders(size):
            if size not in loaders:
                loaders[size] = (
                    build_eval_loader_epillid(img_size=size, batch_size=32, num_workers=4, test_fold=S.TEST_FOLD),
                    build_calibration_loader(img_size=size, batch_size=8, fold_excluded=S.TEST_FOLD,
                                             seed=QJ.CALIB_SEED, num_samples=QJ.CALIB_N))
            return loaders[size]

        print(f"[bench] tag {tag}: {len(todo)} checkpoints, test fold {S.TEST_FOLD} -> {out_csv}", flush=True)
        for i, (role, model, teacher, seed) in enumerate(todo, 1):
            key = (role, model, teacher or "", str(seed))
            rdir = QJ.fp32_dir(role, model, teacher, seed, root=root)
            path = os.path.join(rdir, "model.pth")
            try:
                with open(os.path.join(rdir, "run.json")) as f:
                    run = json.load(f)
            except (OSError, ValueError):
                run = {}
            if run.get("status") != "done":
                if not a.smoke:
                    print(f"[{i:2d}/{len(todo)}] not trained yet: {role} {model} {teacher or ''} seed {seed}", flush=True)
                continue
            fp = fingerprint(path)
            prev = done.get(key)
            graph = QJ.deploy_graph(model)        # rows written before the column existed were plain graphs
            if (prev and prev["status"] == "SUCCESS" and prev["sha"] == fp["sha"]
                    and (prev.get("graph") or "plain") == graph):
                continue
            row = {"role": role, "model": model, "teacher": teacher or "", "seed": seed, "path": path, "sha": fp["sha"],
                   "graph": graph,
                   "val_mAP_fold0": run.get("val_mAP", ""), "best_epoch": run.get("best_epoch", ""),
                   "test_mAP_FP32": "", "test_mAP_INT8_minmax": "", "test_mAP_INT8_pct99.9": "", "seconds": "",
                   "status": "FAILED", "error": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t0 = time.time()
            try:
                size = QJ.IMG_SIZE
                eval_loader, calib_loader = get_loaders(size)
                with tempfile.TemporaryDirectory(prefix="qj_bench_") as tmp:
                    f32 = os.path.join(tmp, "fp32.onnx")
                    export_fp32_onnx(model, path, f32, num_classes=QJ.NUM_CLASSES, img_size=size, pooling="gem",
                                     device=torch.device("cpu"), fuse_branches=graph == "fused")
                    row["test_mAP_FP32"] = measure(f32, eval_loader, fp32_ep, device)
                    if role != "teacher":
                        i8 = os.path.join(tmp, "int8_minmax.onnx")
                        run_ptq(f32, i8, calib_loader, per_channel=True, calibration="minmax")
                        row["test_mAP_INT8_minmax"] = measure(i8, eval_loader, "CPUExecutionProvider", device)
                        ip = os.path.join(tmp, "int8_pct.onnx")
                        run_ptq(f32, ip, calib_loader, per_channel=True, calibration="percentile",
                                extra_options={"CalibPercentile": PCT})
                        row["test_mAP_INT8_pct99.9"] = measure(ip, eval_loader, "CPUExecutionProvider", device)
                row["status"] = "SUCCESS"
            except Exception as exc:  # noqa: BLE001  one bad checkpoint must not stop the others
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - t0)
            done[key] = row
            save_rows(out_csv, done)
            print(f"[{i:2d}/{len(todo)}] {row['status']:7s} {role:7s} {model:24s} {teacher or '':26s} s{seed} "
                  f"val {row['val_mAP_fold0']}  test FP32 {row['test_mAP_FP32']}  INT8 {row['test_mAP_INT8_minmax']}"
                  f"  pct {row['test_mAP_INT8_pct99.9']}  {row['seconds']}s {row['error']}", flush=True)
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} measured, {len(failed)} failed\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
