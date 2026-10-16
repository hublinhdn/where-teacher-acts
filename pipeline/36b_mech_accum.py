"""
where-teacher-acts: where along the network the quantization noise drowns the signal (mechanism, read-only).

36_mech_measure.py measures every quantized activation tensor in isolation: the FP32 tensor passed
through its own deployed scale and zero point. That isolated noise turned out to be modest everywhere,
even in students whose INT8 floor collapses. This script measures the noise each tensor actually carries
in the deployed graph, which includes the error inherited from every quantized tensor upstream: the FP32
graph that ONNX Runtime quantized and the QDQ graph it produced run on the same 64 analysis images, and
for every quantized tensor x the accumulated signal-to-noise ratio is

    SQNR_acc = 10 log10( E[x^2] / E[(x_int8 - x)^2] ),

where x_int8 is the dequantized value of x in the QDQ graph. The QDQ graph is run with graph
optimizations off, so the quantize and dequantize nodes execute as written (the reference QDQ
arithmetic, not the fused integer kernels); the final embedding of that run is checked against the
deployed INT8 embedding stored by 36_mech_measure.py.

Models: the 42 of the SQNR subset (A and B with ConvNeXt-S under MinMax and percentile 99.9, QAT(A) and
QAKD(B) with ConvNeXt-S under MinMax). Outputs: reports_qj/mech/<tag>/accum/<model>.csv (one row per
quantized tensor: producer, depth, isolated and accumulated SQNR, effective bits over the 0.1 to 99.9
percentile span) and accum_models.csv. Resumable; nothing is overwritten. Uses the lock of 36, so the two
never run together.

    python pipeline/36b_mech_accum.py [--students resnet18_tv] [--tag main]
"""
import os
import sys
import csv
import gc
import time
import math
import signal
import socket
import argparse
import tempfile
import importlib.util

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
_spec = importlib.util.spec_from_file_location("mech36", os.path.join(THIS, "36_mech_measure.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)       # moves to the project root and sets up the imports of 36

import numpy as np  # noqa: E402
import torch  # noqa: E402
import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from _qj_io import take_lock  # noqa: E402
from journal2_pipeline._provenance import fingerprint  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402

FIELDS = ["tensor", "producer", "depth", "scale", "zero_point", "sqnr_isolated_db", "sqnr_accum_db",
          "bulk_bits", "clip_frac"]
MODEL_FIELDS = ["model", "student", "cell", "teacher", "rule", "sha", "n_tensors", "final_tensor",
                "final_sqnr_accum_db", "embed_cos_vs_deployed", "seconds", "status", "error", "timestamp"]


def dq_outputs(int8_path, names):
    """{fp32 tensor name: output name of the DequantizeLinear that carries its quantized value}."""
    import onnx
    g = onnx.load(int8_path).graph
    q_out = {n.input[0]: n.output[0] for n in g.node if n.op_type == "QuantizeLinear"}
    dq_of = {n.input[0]: n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
    return {t: dq_of[q_out[t]] for t in names if t in q_out and q_out[t] in dq_of}


def run_outputs(path, outputs, imgs, batch=8):
    """Run `path` (optimizations off) on imgs and return {output name: array over all images}."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto
    m = onnx.load(path)
    have = {o.name for o in m.graph.output}
    inputs = {i.name for i in m.graph.input}
    for t in outputs:
        if t not in have and t not in inputs:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    want = [t for t in outputs if t not in inputs]
    out = {t: [] for t in outputs}
    for b0 in range(0, len(imgs), batch):
        x = imgs[b0:b0 + batch]
        res = dict(zip(want, sess.run(want, {in_name: x}))) if want else {}
        for t in outputs:
            out[t].append(x if t in inputs else res[t])
    del sess
    return {t: np.concatenate(v) for t, v in out.items()}


def accum_pass(pp_path, int8_path, points, imgs, chunk=40):
    import onnx
    base = onnx.load(pp_path)
    nodes = list(base.graph.node)
    producer, depth = {}, {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o], depth[o] = n.op_type, i / max(1, len(nodes) - 1)
    inputs = {i.name for i in base.graph.input}
    names = [t for t in points if t in producer or t in inputs]
    dq = dq_outputs(int8_path, names)
    if len(dq) < len(names):
        raise RuntimeError(f"{len(names) - len(dq)} quantized tensors have no dequantized output in the QDQ graph")
    rows = []
    for c0 in range(0, len(names), chunk):
        part = names[c0:c0 + chunk]
        f = run_outputs(pp_path, part, imgs)
        q = run_outputs(int8_path, [dq[t] for t in part], imgs)
        for t in part:
            x = f[t].astype(np.float64).ravel()
            xq_acc = q[dq[t]].astype(np.float64).ravel()
            s_, z_, dt = points[t]
            qmin, qmax = (0, 255) if dt == "uint8" else (-128, 127)
            xq_iso = (np.clip(np.round(x / s_) + z_, qmin, qmax) - z_) * s_
            sig = float(np.dot(x, x))
            n_iso = float(np.dot(x - xq_iso, x - xq_iso))
            n_acc = float(np.dot(x - xq_acc, x - xq_acc))
            p001, p999 = np.percentile(x[np.random.default_rng(0).integers(0, x.size, size=min(x.size, 200000))],
                                       [0.1, 99.9])
            lo, hi = (qmin - z_) * s_, (qmax - z_) * s_
            rows.append({"tensor": t, "producer": producer.get(t, "input"), "depth": round(depth.get(t, 0.0), 4),
                         "scale": s_, "zero_point": z_,
                         "sqnr_isolated_db": 10 * math.log10(sig / n_iso) if n_iso > 0 else float("inf"),
                         "sqnr_accum_db": 10 * math.log10(sig / n_acc) if n_acc > 0 else float("inf"),
                         "bulk_bits": math.log2(max(0.0, (p999 - p001) / s_) + 1),
                         "clip_frac": float(np.mean((x < lo) | (x > hi)))})
        del f, q
        gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    out = os.path.join(QJ.REPORTS, "mech", a.tag)
    os.makedirs(os.path.join(out, "accum"), exist_ok=True)
    lock = os.path.join(out, ".lock")
    try:
        fd = take_lock(lock, f"{out} is being written by another process; check, then delete {lock}")
    except FileExistsError as exc:
        sys.exit(str(exc))
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, M._exit_on_signal)
    models_csv = os.path.join(out, "accum_models.csv")
    done = {r["model"]: r for r in csv.DictReader(open(models_csv))} if os.path.exists(models_csv) else {}
    try:
        df = M.analysis_set(os.path.join(out, "analysis_set.csv"))
        sub = df[df.sqnr]
        imgs = np.concatenate([b[0].numpy() for b in M.loader(sub, batch_size=16, workers=2)]).astype(np.float32)
        pos = np.flatnonzero(df.sqnr.values)          # rows of the stored embeddings for these images
        jobs = [j for j in M.plan(a.students or QJ.STUDENTS, only_sqnr=True) if j[4]]
        todo = [j for j in jobs if not os.path.exists(os.path.join(out, "accum", M.model_id(*j[:4]) + ".csv"))]
        print(f"[accum] {len(imgs)} images; {len(jobs)} models, {len(todo)} to measure", flush=True)
        calib = S.calibration_loader(QJ.IMG_SIZE, {S.TEST_FOLD}, QJ.CALIB_SEED, QJ.CALIB_N)
        from _qakd_export import export_clipped_onnx
        for i, (cell, s, t, rule, _) in enumerate(todo, 1):
            mid = M.model_id(cell, s, t, rule)
            path, clipped = M.checkpoint(cell, s, t)
            row = {"model": mid, "student": s, "cell": cell, "teacher": t or "", "rule": rule, "sha": "",
                   "status": "FAILED", "error": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t0 = time.time()
            try:
                row["sha"] = fingerprint(path)["sha"]
                with tempfile.TemporaryDirectory(prefix="qj_acc_") as tmp:
                    f32, i8, pp = (os.path.join(tmp, n) for n in ("fp32.onnx", "int8.onnx", "pre.onnx"))
                    if clipped:
                        export_clipped_onnx(s, path, path + ".clip.pt", f32, num_classes=QJ.NUM_CLASSES,
                                            img_size=QJ.IMG_SIZE)
                    else:
                        export_fp32_onnx(s, path, f32, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                                         pooling="gem", device=torch.device("cpu"),
                                         fuse_branches=QJ.deploy_graph(s) == "fused")
                    run_ptq(f32, i8, calib, per_channel=True, keep_preprocessed=pp, **M.RULE_KW[rule])
                    points = M.activation_quant_points(i8)
                    rows = accum_pass(pp, i8, points, imgs)
                    # the QDQ reference run against the deployed INT8 embedding stored by 36
                    out_name = "output_embedding"
                    ref = run_outputs(i8, [out_name], imgs)[out_name]
                    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
                    dep = np.load(os.path.join(out, "emb", mid + ".npy")).astype(np.float32)[pos]
                    dep = dep / np.linalg.norm(dep, axis=1, keepdims=True)
                    row["embed_cos_vs_deployed"] = round(float(np.mean(np.sum(ref * dep, axis=1))), 5)
                last = max(rows, key=lambda r: r["depth"])
                row.update(n_tensors=len(rows), final_tensor=last["tensor"],
                           final_sqnr_accum_db=round(last["sqnr_accum_db"], 3))
                M.write_csv(os.path.join(out, "accum", mid + ".csv"), FIELDS, rows)
                row["status"] = "SUCCESS"
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                gc.collect()
            row["seconds"] = round(time.time() - t0)
            done[mid] = row
            M.write_csv(models_csv, MODEL_FIELDS, list(done.values()))
            print(f"[{i:2d}/{len(todo)}] {row['status']:7s} {mid:62s} {row['seconds']:4d}s "
                  f"final SQNR {row.get('final_sqnr_accum_db', '')} dB, cos vs deployed "
                  f"{row.get('embed_cos_vs_deployed', '')} {row['error']}", flush=True)
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} models recorded, {len(failed)} failed")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
