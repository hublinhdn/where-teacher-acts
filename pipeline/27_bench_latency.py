"""
where-teacher-acts: deployment cost of the seven students on the CPU, FP32 against INT8, read-only.

For every student, two deployed networks: A after post-training quantization (the PTQ route) and the
QAKD(B) model distilled from ConvNeXt-S (the quantization-aware route, learned ranges exported as Clip
nodes), each as FP32 ONNX and as INT8 (ONNX Runtime static PTQ, QDQ, MinMax, per-channel int8 weights,
per-tensor uint8 activations, the test protocol's calibration set). RepVGG-A0 is measured fused, as it
is deployed. Per network: parameters of the deployed embedding network (backbone, pooling, projection),
ONNX file size, latency for one 224 x 224 image with ONNX Runtime's CPU provider at 1 and 4 intra-op
threads. Memory is not reported: sessions measured one after another in one process reuse the
allocator's pages, so a per-session resident-memory difference reads zero.

The server is shared, so one pass per graph is not trustworthy: a graph timed while a neighbour is
busy looks slow. Every graph is exported once, then all graphs are timed round robin for --rounds
rounds (30 warm-up and 200 timed runs per graph, thread setting and round); the reported latency is
the median over rounds of the per-round medians, with the per-round minimum and maximum and the
1-minute load average at the start of every round. Pin the process to fixed physical cores
(taskset) so runs are comparable. Refuses to overwrite an existing output unless --force.

    taskset -c 0,2,4,6 python pipeline/27_bench_latency.py [--rounds 5] [--students resnet18_tv] [--tag main]
"""
import os
import sys
import csv
import time
import argparse
import tempfile
import statistics as st

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from models.pill_retrieval_model import PillRetrievalModel  # noqa: E402
from models.reparam import fuse_repvgg_branches  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402
from journal2_pipeline._benchmark_utils import measure_onnx_latency_ms, get_model_size_mb  # noqa: E402

THREADS = (1, 4)
FIELDS = ["student", "network", "precision", "graph", "params_backbone_M", "params_deployed_M", "size_MB"]
for _t in THREADS:
    FIELDS += [f"lat_t{_t}_median_ms", f"lat_t{_t}_min_ms", f"lat_t{_t}_max_ms", f"lat_t{_t}_p95_ms"]
FIELDS += ["rounds", "iters", "load_min", "load_max", "status", "error"]
WARMUP, ITERS = 30, 200


def deployed_params(student):
    """Parameters of the network the deployed graph runs (no training heads), in millions."""
    m = PillRetrievalModel(num_classes=QJ.NUM_CLASSES, backbone_type=student, pooling_type="gem").eval()
    if QJ.deploy_graph(student) == "fused":
        fuse_repvgg_branches(m)
    backbone = sum(p.numel() for p in m.features.parameters())
    rest = sum(p.numel() for name in ("pool", "fc_projection", "bn_head")
               if hasattr(m, name) for p in getattr(m, name).parameters())
    return backbone / 1e6, (backbone + rest) / 1e6


def load_avg():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def search_ms(g, q):
    t0 = time.perf_counter()
    for _ in range(ITERS):
        (q @ g.T).argsort()
    return (time.perf_counter() - t0) / ITERS * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args()
    out_dir = os.path.join(QJ.REPORTS, "bench", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"latency_rr{a.rounds}.csv")
    if os.path.exists(out_csv) and not a.force:
        sys.exit(f"{out_csv} exists; pass --force to overwrite it")
    S.load_split()
    calib = S.calibration_loader(QJ.IMG_SIZE, {S.TEST_FOLD}, QJ.CALIB_SEED, QJ.CALIB_N)
    from _qakd_export import export_clipped_onnx
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    print(f"[latency] cores {affinity}; {a.rounds} rounds, {WARMUP} warm-up and {ITERS} timed runs per "
          f"graph, thread setting and round", flush=True)
    rows = {}
    with tempfile.TemporaryDirectory(prefix="qj_lat_") as tmp:
        # 1. export and quantize every graph once
        for s in a.students or QJ.STUDENTS:
            pb, pd = deployed_params(s)
            graph = QJ.deploy_graph(s)
            a_path = os.path.join(QJ.fp32_dir("base", s), "model.pth")
            k_dir = QJ.run_dir("qakd_b", s, QJ.SEED_REPEAT_TEACHER, 40, QJ.FP32_SEED)
            nets = {"A": os.path.join(tmp, f"{s}_a_fp32.onnx"), "QAKD(B)": os.path.join(tmp, f"{s}_k_fp32.onnx")}
            try:
                export_fp32_onnx(s, a_path, nets["A"], num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                                 pooling="gem", device=torch.device("cpu"), fuse_branches=graph == "fused")
                export_clipped_onnx(s, os.path.join(k_dir, "model.pth"), os.path.join(k_dir, "model.pth.clip.pt"),
                                    nets["QAKD(B)"], num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE)
            except Exception as exc:  # noqa: BLE001
                rows[(s, "-", "-")] = {"student": s, "status": "FAILED", "error": f"export: {type(exc).__name__}: {exc}"}
                continue
            for net, f32 in nets.items():
                i8 = f32.replace("_fp32.onnx", "_int8.onnx")
                for prec, path in (("FP32", f32), ("INT8", i8)):
                    row = {"student": s, "network": net, "precision": prec, "graph": graph,
                           "params_backbone_M": round(pb, 3), "params_deployed_M": round(pd, 3),
                           "status": "FAILED", "error": "", "_path": path, "_t": {t: [] for t in THREADS},
                           "_p95": {t: [] for t in THREADS}}
                    try:
                        if prec == "INT8":
                            run_ptq(f32, i8, calib, per_channel=True, calibration="minmax")
                        row["size_MB"] = round(get_model_size_mb(path), 2)
                        row["status"] = "OK"
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows[(s, net, prec)] = row
            print(f"   exported {s}", flush=True)

        # 2. time every graph, round robin
        timed = [r for r in rows.values() if r.get("status") == "OK"]
        g = np.random.default_rng(0).standard_normal((9804, 512)).astype(np.float32)
        q = np.random.default_rng(1).standard_normal((1, 512)).astype(np.float32)
        loads, search = [], []
        for rnd in range(a.rounds):
            loads.append(load_avg())
            for r in timed:
                for t in THREADS:
                    try:
                        m = measure_onnx_latency_ms(r["_path"], img_size=QJ.IMG_SIZE, warmup=WARMUP,
                                                    iterations=ITERS, intra_threads=t)
                        r["_t"][t].append(m["median"])
                        r["_p95"][t].append(m["p95"])
                    except Exception as exc:  # noqa: BLE001
                        r["status"], r["error"] = "FAILED", f"timing: {type(exc).__name__}: {exc}"
            for _ in range(20):
                (q @ g.T).argsort()
            search.append(search_ms(g, q))
            print(f"   round {rnd + 1}/{a.rounds} done, load at start {loads[-1]}", flush=True)
        loads.append(load_avg())

    out = []
    for r in rows.values():
        if r.get("status") == "OK" and all(len(r["_t"][t]) == a.rounds for t in THREADS):
            for t in THREADS:
                v = r["_t"][t]
                r[f"lat_t{t}_median_ms"] = round(st.median(v), 2)
                r[f"lat_t{t}_min_ms"] = round(min(v), 2)
                r[f"lat_t{t}_max_ms"] = round(max(v), 2)
                r[f"lat_t{t}_p95_ms"] = round(st.median(r["_p95"][t]), 2)
            r["status"] = "SUCCESS"
        elif r.get("status") == "OK":
            r["status"], r["error"] = "FAILED", "incomplete rounds"
        r.update(rounds=a.rounds, iters=ITERS, load_min=min(loads), load_max=max(loads))
        out.append(r)
        if r.get("network"):
            print(f"   {r['status']:7s} {r['student']:24s} {r['network']:8s} {r['precision']}  {r.get('size_MB', '')} MB  "
                  f"t1 {r.get('lat_t1_median_ms', '')} [{r.get('lat_t1_min_ms', '')}, {r.get('lat_t1_max_ms', '')}] ms  "
                  f"t4 {r.get('lat_t4_median_ms', '')} [{r.get('lat_t4_min_ms', '')}, {r.get('lat_t4_max_ms', '')}] ms "
                  f"{r['error']}", flush=True)
    out.append({"student": "-", "network": "gallery search, 9804 x 512, numpy", "precision": "FP32",
                "lat_t1_median_ms": round(st.median(search), 3), "lat_t1_min_ms": round(min(search), 3),
                "lat_t1_max_ms": round(max(search), 3), "rounds": a.rounds, "iters": ITERS,
                "load_min": min(loads), "load_max": max(loads), "status": "SUCCESS"})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    failed = [r for r in out if r.get("status") != "SUCCESS"]
    print(f"\nload average during the run: min {min(loads)} max {max(loads)}")
    print(f"{len(out)} rows, {len(failed)} failed -> {out_csv}\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
