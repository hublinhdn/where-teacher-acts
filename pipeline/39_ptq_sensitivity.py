"""
where-teacher-acts: sensitivity of the INT8 results to the quantizer (declared 2026-10-16 before running, in
the dated plan, docs/PLAN_AND_LOG.md; exploratory). Two parts, both through the same run_ptq path,
calibration pool and evaluation as the main benchmark (22_eval_part2.py):

part "strong"  A and B (ConvNeXt-S) of seed 42 for the seven students, recalibrated with settings stronger than,
               or different from, the two rules of the main benchmark, with the main calibration draw (seed 123):
                 entropy2048     ONNX Runtime's Entropy calibrator with 2048 histogram bins (128 quantized bins);
                                 its default of 128 bins leaves one threshold candidate, the full range, and
                                 quantize_static passes no bin count, so the calibrator is created with it here
                 pct99.99, pct99.999   Percentile calibration at those percentiles
                 minmax_headfp, pct99.9_headfp   the two main rules with the retrieval head, from the GeM clamp
                                 onward, kept in FP32 (nodes_to_exclude)
                 minmax          the main rule itself, on the test fold only, as a reproduction check
               measured on the test fold, then on the validation fold.
part "calib"   calibration noise of the fine-tuned models: QAT(A), QAT(B) and QAKD(B) with ConvNeXt-S, seed 42,
               recalibrated with the seeds 223, 323, 423 and 523 under MinMax and Percentile 99.9 (seed 123 is the
               main measurement), measured on the test fold.

Resumable by checkpoint fingerprint; one process per tag and part; reports_qj/bench/<tag>/ptq_sensitivity.csv and
calib_noise_ft.csv. Every row records the number of QuantizeLinear nodes of the INT8 graph.

    python pipeline/39_ptq_sensitivity.py --part strong [--tag sens] [--students resnet18_tv] [--folds test val]
    python pipeline/39_ptq_sensitivity.py --part calib [--tag sens]
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

import onnx  # noqa: E402
import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from journal2_pipeline._provenance import fingerprint  # noqa: E402
from journal2_pipeline import _quant_utils as QU  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402
from journal2_pipeline._eval_utils import (build_eval_loader_epillid, extract_features_onnx,  # noqa: E402
                                           calculate_both_sides_map)

ENTROPY_BINS = 2048
STRONG = {   # name: (run_ptq calibration, extra_options, head kept in FP32, folds)
    "entropy2048": ("entropy", None, False, ("test", "val")),
    "pct99.99": ("percentile", {"CalibPercentile": 99.99}, False, ("test", "val")),
    "pct99.999": ("percentile", {"CalibPercentile": 99.999}, False, ("test", "val")),
    "minmax_headfp": ("minmax", None, True, ("test", "val")),
    "pct99.9_headfp": ("percentile", {"CalibPercentile": 99.9}, True, ("test", "val")),
    "minmax": ("minmax", None, False, ("test",)),
}
NOISE_SEEDS = (223, 323, 423, 523)
FIELDS = ["part", "fold", "student", "checkpoint", "teacher", "config", "calib_seed", "mAP_INT8", "n_qnodes",
          "n_excluded", "sha", "graph", "seconds", "status", "error", "timestamp"]


def _patch_entropy_bins():
    """quantize_static creates its calibrator without a bin count; give the Entropy calibrator ENTROPY_BINS
    histogram bins (the quantized-bin count stays at its default of 128)."""
    import importlib
    from onnxruntime.quantization.calibrate import CalibrationMethod
    # the package re-exports a function named quantize, which hides the module of the same name
    ortq = importlib.import_module("onnxruntime.quantization.quantize")
    original = ortq.create_calibrator

    def create_calibrator(*args, **kwargs):
        if kwargs.get("calibrate_method") == CalibrationMethod.Entropy:
            eo = dict(kwargs.get("extra_options") or {})
            eo["num_bins"] = ENTROPY_BINS
            kwargs["extra_options"] = eo
        return original(*args, **kwargs)
    ortq.create_calibrator = create_calibrator


def head_nodes(onnx_path):
    """Names of the retrieval-head nodes of ONNX Runtime's pre-processed graph: the GeM clamp feeding the first
    Pow, that Pow and every node downstream of it."""
    import fcntl
    quant_pre_process = QU._import_ort_quant()["quant_pre_process"]     # the function run_ptq uses
    # the pre-processing writes sym_shape_infer_temp.onnx into the working directory, as in run_ptq, so it
    # takes run_ptq's lock and cleans up the same way
    with open(os.path.join(os.getcwd(), ".ptq.lock"), "a") as lockf, \
            tempfile.TemporaryDirectory(prefix="qj_head_") as tmp:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            pp = os.path.join(tmp, "pp.onnx")
            quant_pre_process(onnx_path, pp)
            g = onnx.load(pp).graph
        finally:
            if os.path.exists("sym_shape_infer_temp.onnx"):
                os.remove("sym_shape_infer_temp.onnx")
            fcntl.flock(lockf, fcntl.LOCK_UN)
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    first_pow = next((n for n in g.node if n.op_type == "Pow"), None)
    if first_pow is None:
        raise RuntimeError("no Pow node: the GeM head was not found")
    names, stack = {first_pow.name}, [first_pow]
    clamp = producer.get(first_pow.input[0])
    if clamp is not None and clamp.op_type in ("Clip", "Max"):
        names.add(clamp.name)
    while stack:
        n = stack.pop()
        for o in n.output:
            for c in consumers.get(o, []):
                if c.name not in names:
                    names.add(c.name)
                    stack.append(c)
    return sorted(names)


def count_q(path):
    return sum(1 for n in onnx.load(path).graph.node if n.op_type == "QuantizeLinear")


def save_rows(path, rows):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows.values())
    os.replace(tmp, path)


def checkpoints(part, students):
    t = QJ.SEED_REPEAT_TEACHER
    out = []
    for s in students:
        if part == "strong":
            out.append((s, "A", None, os.path.join(QJ.fp32_dir("base", s), "model.pth"), "fp32"))
            out.append((s, "B", t, os.path.join(QJ.fp32_dir("kd", s, t), "model.pth"), "fp32"))
        else:
            for ck, arm, tt in (("QAT_A", "qat_a", None), ("QAT_B", "qat_b", t), ("QAKD_B", "qakd_b", t)):
                out.append((s, ck, tt, os.path.join(QJ.run_dir(arm, s, tt, 40, QJ.FP32_SEED), "model.pth"), "part2"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("strong", "calib"))
    ap.add_argument("--tag", default="sens")
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", default=("test", "val"), choices=("test", "val"))
    a = ap.parse_args()
    out_dir = os.path.join(QJ.REPORTS, "bench", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ptq_sensitivity.csv" if a.part == "strong" else "calib_noise_ft.csv")
    lock = os.path.join(out_dir, f".lock_{a.part}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{out_csv} is being written by another process, or by one that was killed; check, then delete {lock}")
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    _patch_entropy_bins()
    done = {}
    try:
        S.load_split()
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                done = {(r["fold"], r["student"], r["checkpoint"], r["config"], r["calib_seed"]): r
                        for r in csv.DictReader(f)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from _qakd_export import export_clipped_onnx
        folds = [f for f in ("test", "val") if f in a.folds] if a.part == "strong" else ["test"]
        for fold_name in folds:     # the whole test fold first, then validation
            fold = S.TEST_FOLD if fold_name == "test" else S.VAL_FOLD
            exclude = {S.TEST_FOLD} if fold_name == "test" else {S.VAL_FOLD, S.TEST_FOLD}
            eval_loader, calib = None, {}
            for s, ck, t, path, kind in checkpoints(a.part, a.students or QJ.STUDENTS):
                rdir = os.path.dirname(path)
                try:
                    with open(os.path.join(rdir, "run.json")) as f:
                        finished = json.load(f).get("status") == "done"
                except (OSError, ValueError):
                    finished = False
                if not finished or not os.path.exists(path):
                    print(f"   not available: {s} {ck}", flush=True)
                    continue
                sha = fingerprint(path)["sha"]
                graph = QJ.deploy_graph(s) if kind == "fp32" else json.load(open(os.path.join(rdir, "run.json"))).get(
                    "deploy_graph", "plain")
                if a.part == "strong":
                    cells = [(cfg, QJ.CALIB_SEED) for cfg, v in STRONG.items() if fold_name in v[3]]
                else:
                    cells = [(f"{rule}_seed{seed}", seed) for seed in NOISE_SEEDS for rule in ("minmax", "pct99.9")]
                cells = [c for c in cells if not (done.get((fold_name, s, ck, c[0], str(c[1])), {}).get("status")
                                                  == "SUCCESS" and done[(fold_name, s, ck, c[0], str(c[1]))]["sha"] == sha)]
                if not cells:
                    continue
                if eval_loader is None:
                    eval_loader = build_eval_loader_epillid(img_size=QJ.IMG_SIZE, batch_size=32, num_workers=4,
                                                            test_fold=fold)
                with tempfile.TemporaryDirectory(prefix="qj_sens_") as tmp:
                    f32 = os.path.join(tmp, "fp32.onnx")
                    if kind == "part2":
                        export_clipped_onnx(s, path, path + ".clip.pt", f32, num_classes=QJ.NUM_CLASSES,
                                            img_size=QJ.IMG_SIZE)
                    else:
                        export_fp32_onnx(s, path, f32, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                                         pooling="gem", device=torch.device("cpu"), fuse_branches=graph == "fused")
                    excl = None
                    for cfg, seed in cells:
                        if seed not in calib:
                            calib[seed] = S.calibration_loader(QJ.IMG_SIZE, exclude, seed, QJ.CALIB_N)
                        row = {"part": a.part, "fold": fold_name, "student": s, "checkpoint": ck, "teacher": t or "",
                               "config": cfg, "calib_seed": seed, "mAP_INT8": "", "n_qnodes": "", "n_excluded": 0,
                               "sha": sha, "graph": graph, "seconds": "", "status": "FAILED", "error": "",
                               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                        t0 = time.time()
                        try:
                            if a.part == "strong":
                                rule, extra, head_fp, _ = STRONG[cfg]
                            else:
                                rule, extra, head_fp = (("minmax", None, False) if cfg.startswith("minmax")
                                                        else ("percentile", {"CalibPercentile": 99.9}, False))
                            if head_fp and excl is None:
                                excl = head_nodes(f32)
                            i8 = os.path.join(tmp, f"int8_{cfg}_{seed}.onnx")
                            run_ptq(f32, i8, calib[seed], per_channel=True, calibration=rule, extra_options=extra,
                                    nodes_to_exclude=excl if head_fp else None)
                            if not QU.LAST_PREPROCESS.get("ok"):
                                raise RuntimeError(f"pre-processing failed: {QU.LAST_PREPROCESS.get('error')}")
                            row["n_qnodes"] = count_q(i8)
                            row["n_excluded"] = len(excl) if head_fp else 0
                            feats, labels, subs, is_ref = extract_features_onnx(i8, eval_loader,
                                                                                provider="CPUExecutionProvider",
                                                                                desc=f"{s}_{ck}_{cfg}_{seed}")
                            m, _ = calculate_both_sides_map(feats, labels, subs, is_ref, device=device)
                            row["mAP_INT8"] = round(float(m), 4)
                            row["status"] = "SUCCESS"
                            os.remove(i8)
                        except Exception as exc:  # noqa: BLE001
                            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                        row["seconds"] = round(time.time() - t0)
                        done[(fold_name, s, ck, cfg, str(seed))] = row
                        save_rows(out_csv, done)
                        print(f"   {row['status']:7s} {fold_name:4s} {s:24s} {ck:6s} {cfg:16s} seed {seed} "
                              f"{row['mAP_INT8']} q {row['n_qnodes']} excl {row['n_excluded']} {row['seconds']}s "
                              f"{row['error']}", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} measurements, {len(failed)} failed -> {out_csv}\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
