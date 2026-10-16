"""
where-teacher-acts: calibration noise of the INT8 benchmark on the test fold.

INT8 accuracy after static PTQ depends on which 256 images calibrate it. For each of the 7 students,
the undistilled A and the distilled B with the pilot teacher (ConvNeXt-S) are quantized with each of
the calibration seeds in QJ.CALIB_SEEDS (123 is the primary seed, so its rows must equal the main
benchmark's), with MinMax and with percentile 99.9, and measured on test fold 4. FP32 does not depend
on calibration and is not repeated. Per student, the spread of INT8 mAP and of the seed-matched INT8
distillation effect (B minus A) is the noise floor the analysis uses to decide which INT8
differences are real.

Same functions and calibration pool as 11_bench_fp32_ptq.py; only the seed varies. Resumable by
checkpoint fingerprint; reports_qj/bench/<tag>/calib_noise.csv; one process per tag.

    python pipeline/12_bench_calib_noise.py
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
FIELDS = ["student", "checkpoint", "teacher", "calib_seed", "calibration", "test_mAP_INT8", "sha", "graph",
          "seconds", "status", "error", "timestamp"]


def save_rows(path, rows):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows.values())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or time.strftime("%Y%m%d")
    out_dir = os.path.join(QJ.REPORTS, "bench", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "calib_noise.csv")
    lock = os.path.join(out_dir, ".lock_noise")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{out_dir} noise run in use by another process, or by one that was killed; check, then delete {lock}")
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    done = {}
    try:
        S.load_split()
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                done = {(r["student"], r["checkpoint"], r["calib_seed"], r["calibration"]): r for r in csv.DictReader(f)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_loaders, calib_loaders = {}, {}
        jobs = [(s, ck) for s in QJ.STUDENTS for ck in ("A", "B")]
        print(f"[noise] tag {tag}: {len(jobs)} checkpoints x {len(QJ.CALIB_SEEDS)} seeds x 2 rules -> {out_csv}",
              flush=True)
        for student, ck in jobs:
            teacher = QJ.SEED_REPEAT_TEACHER if ck == "B" else None
            rdir = QJ.fp32_dir("kd" if ck == "B" else "base", student, teacher)
            path = os.path.join(rdir, "model.pth")
            try:
                with open(os.path.join(rdir, "run.json")) as f:
                    finished = json.load(f).get("status") == "done"
            except (OSError, ValueError):
                finished = False
            if not finished:
                print(f"   not trained yet: {student} {ck}", flush=True)
                continue
            sha = fingerprint(path)["sha"]
            graph = QJ.deploy_graph(student)      # rows written before the column existed were plain graphs
            cells = [(seed, rule) for seed in QJ.CALIB_SEEDS for rule in ("minmax", f"pct{PCT:g}")]
            cells = [c for c in cells if not (done.get((student, ck, str(c[0]), c[1]), {}).get("status") == "SUCCESS"
                                              and done[(student, ck, str(c[0]), c[1])]["sha"] == sha
                                              and (done[(student, ck, str(c[0]), c[1])].get("graph") or "plain")
                                              == graph)]
            if not cells:
                continue
            size = QJ.IMG_SIZE
            if size not in eval_loaders:
                eval_loaders[size] = build_eval_loader_epillid(img_size=size, batch_size=32, num_workers=4,
                                                               test_fold=S.TEST_FOLD)
            with tempfile.TemporaryDirectory(prefix="qj_noise_") as tmp:
                f32 = os.path.join(tmp, "fp32.onnx")
                export_fp32_onnx(student, path, f32, num_classes=QJ.NUM_CLASSES, img_size=size, pooling="gem",
                                 device=torch.device("cpu"), fuse_branches=graph == "fused")
                for seed, rule in cells:
                    if (size, seed) not in calib_loaders:
                        calib_loaders[(size, seed)] = build_calibration_loader(
                            img_size=size, batch_size=8, fold_excluded=S.TEST_FOLD, seed=seed, num_samples=QJ.CALIB_N)
                    row = {"student": student, "checkpoint": ck, "teacher": teacher or "", "calib_seed": seed,
                           "calibration": rule, "test_mAP_INT8": "", "sha": sha, "graph": graph, "seconds": "",
                           "status": "FAILED",
                           "error": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    t0 = time.time()
                    try:
                        i8 = os.path.join(tmp, f"int8_{seed}_{rule}.onnx")
                        kw = (dict(calibration="minmax") if rule == "minmax" else
                              dict(calibration="percentile", extra_options={"CalibPercentile": PCT}))
                        run_ptq(f32, i8, calib_loaders[(size, seed)], per_channel=True, **kw)
                        feats, labels, subs, is_ref = extract_features_onnx(i8, eval_loaders[size],
                                                                            provider="CPUExecutionProvider",
                                                                            desc=f"{student}_{ck}_{seed}_{rule}")
                        m, _ = calculate_both_sides_map(feats, labels, subs, is_ref, device=device)
                        row["test_mAP_INT8"] = round(float(m), 4)
                        row["status"] = "SUCCESS"
                        os.remove(i8)
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row["seconds"] = round(time.time() - t0)
                    done[(student, ck, str(seed), rule)] = row
                    save_rows(out_csv, done)
                    print(f"   {row['status']:7s} {student:24s} {ck} seed {seed} {rule:8s} {row['test_mAP_INT8']} "
                          f"{row['seconds']}s {row['error']}", flush=True)
            gc.collect()
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} measurements, {len(failed)} failed\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
