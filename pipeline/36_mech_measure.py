"""
where-teacher-acts: measurements for the mechanism analysis (no training; checkpoints are only read).

The analysis set is drawn from the validation fold, so the test fold stays reserved for the results:
the fold-0 consumer queries and 2,000 reference images (every reference of the 192 fold-0 pill types,
plus a fixed random sample of the others), minus any image of the deployment calibration set. For
every model below the script stores its L2-normalized embedding of the analysis set, as deployed:

  FP32, CUDA execution provider:   A (7), B (42), the six teachers T
  INT8 MinMax, CPU:                PTQ(A) 7, PTQ(B) 42, QAT(A) 7, QAT(B) 42, QAKD(A) 42, QAKD(B) 42
  INT8 percentile 99.9, CPU:       PTQ(A) 7, PTQ(B) with ConvNeXt-S 7

The INT8 models are built exactly as in the test benchmark (same exporters; ONNX Runtime static PTQ in
QDQ form, per-channel int8 weights, per-tensor uint8 activations; 256 calibration images with seed 123
from every image except the test queries), so they are the models whose test mAP the paper reports.

For A and for B with ConvNeXt-S under both rules, and for QAT(A) and QAKD(B) with ConvNeXt-S under
MinMax, the script also measures every activation tensor that the deployed graph quantizes: on 64
analysis images, the FP32 tensor of the graph ONNX Runtime quantized is passed through that tensor's
deployed scale and zero point, giving its isolated signal-to-quantization-noise ratio
SQNR = 10 log10(E[x^2] / E[(x - q(x))^2]), the share of clipped values, and how far the deployed range
reaches beyond the bulk of the values (range over the 0.1 to 99.9 percentile span).

Outputs, reports_qj/mech/<tag>/: analysis_set.csv, emb/<model>.npy (float16), sqnr/<model>.csv,
models.csv. Resumable: a model whose outputs exist is skipped, and nothing is overwritten. One process
per tag (lock file). FP32 and SQNR-subset models run first, so the first results come early.

    python pipeline/36_mech_measure.py [--students resnet18_tv] [--only-sqnr] [--tag main]
"""
import os
import sys
import csv
import gc
import time
import signal
import socket
import argparse
import tempfile

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import resource  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
torch.multiprocessing.set_sharing_strategy("file_system")
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(_soft, 65536), _hard), _hard))

import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from _qj_io import take_lock  # noqa: E402
from journal2_pipeline._provenance import fingerprint  # noqa: E402
from journal2_pipeline import _quant_utils as QU  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402
from journal2_pipeline._eval_utils import extract_features_onnx  # noqa: E402

N_GALLERY, N_SQNR, SET_SEED, SQNR_SEED = 2000, 64, 42, 7
PCT = 99.9
CNX_S = QJ.SEED_REPEAT_TEACHER
RULE_KW = {"mm": dict(calibration="minmax"),
           "pc": dict(calibration="percentile", extra_options={"CalibPercentile": PCT})}
MODEL_FIELDS = ["model", "cell", "student", "teacher", "rule", "graph", "path", "sha", "preprocess", "n_images",
                "n_tensors", "seconds", "status", "error", "timestamp"]
SQNR_FIELDS = ["tensor", "producer", "depth", "scale", "zero_point", "lo", "hi", "n_values", "signal_power",
               "noise_power", "sqnr_db", "clip_frac", "clip_noise_share", "p001", "p999", "vmin", "vmax",
               "range_ratio"]


def _exit_on_signal(signum, _frame):
    raise SystemExit(f"stopped by signal {signum}")


def model_id(cell, student, teacher, rule):
    return f"{cell}__{student or '-'}__{teacher or '-'}__{rule}"


def checkpoint(cell, student, teacher):
    """The checkpoint a cell is built from (seed 42), and whether it carries learned clipping ranges."""
    if cell in ("A_fp32", "PTQ_A"):
        return os.path.join(QJ.fp32_dir("base", student), "model.pth"), False
    if cell in ("B_fp32", "PTQ_B"):
        return os.path.join(QJ.fp32_dir("kd", student, teacher), "model.pth"), False
    if cell == "T_fp32":
        return os.path.join(QJ.fp32_dir("teacher", teacher), "model.pth"), False
    arm = cell.lower()
    return os.path.join(QJ.run_dir(arm, student, None if arm == "qat_a" else teacher, 40, QJ.FP32_SEED),
                        "model.pth"), True


def plan(students, only_sqnr):
    """(cell, student, teacher, rule, sqnr) in run order: FP32, then the SQNR subset, then the rest."""
    fp32 = [("T_fp32", None, t, "fp32", False) for t in QJ.TEACHERS]
    first, rest = [], []
    for s in students:
        fp32.append(("A_fp32", s, None, "fp32", False))
        fp32 += [("B_fp32", s, t, "fp32", False) for t in QJ.TEACHERS]
        for rule in ("mm", "pc"):
            first += [("PTQ_A", s, None, rule, True), ("PTQ_B", s, CNX_S, rule, True)]
        first += [("QAT_A", s, None, "mm", True), ("QAKD_B", s, CNX_S, "mm", True)]
        for t in QJ.TEACHERS:
            rest += [("PTQ_B", s, t, "mm", False)] if t != CNX_S else []
            rest += [("QAT_B", s, t, "mm", False), ("QAKD_A", s, t, "mm", False)]
            rest += [("QAKD_B", s, t, "mm", False)] if t != CNX_S else []
    return fp32 + first + ([] if only_sqnr else rest)


def analysis_set(out_csv):
    """Validation queries and 2,000 references, calibration images excluded; fixed once and reused."""
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    split = S.load_split()
    calib = S.calibration_loader(QJ.IMG_SIZE, {S.TEST_FOLD}, QJ.CALIB_SEED, QJ.CALIB_N).dataset.df
    used = set(calib["image_path"])
    q = split.val[~split.val["image_path"].isin(used)]
    g = split.gallery[~split.gallery["image_path"].isin(used)]
    val_types = set(split.val["label_idx"])
    g_val = g[g["label_idx"].isin(val_types)]
    g_rest = g[~g["label_idx"].isin(val_types)].sample(n=N_GALLERY - len(g_val), random_state=SET_SEED)
    df = pd.concat([q.assign(role="query"), g_val.assign(role="gallery"), g_rest.assign(role="gallery")])
    df = df.reset_index(drop=True)
    df["sqnr"] = False
    df.loc[df.sample(n=N_SQNR, random_state=SQNR_SEED).index, "sqnr"] = True
    df.to_csv(out_csv, index=False)
    return df


def loader(df, batch_size=32, workers=4):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from utils.dataset_loader import PillDataset
    tf = transforms.Compose([transforms.Resize((QJ.IMG_SIZE, QJ.IMG_SIZE)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return DataLoader(PillDataset(df.reset_index(drop=True), transform=tf), batch_size=batch_size,
                      shuffle=False, num_workers=workers)


def activation_quant_points(int8_path):
    """{fp32 tensor name: (scale, zero_point)} for every activation the QDQ graph quantizes."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(int8_path)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    points = {}
    for n in m.graph.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in init:
            continue
        scale, zp = init.get(n.input[1]), init.get(n.input[2]) if len(n.input) > 2 else None
        if scale is None or np.asarray(scale).size != 1:
            continue                            # per-channel quantization of an activation: not used here
        points[n.input[0]] = (float(np.asarray(scale).reshape(-1)[0]),
                              int(np.asarray(zp).reshape(-1)[0]) if zp is not None else 0,
                              str(np.asarray(zp).dtype) if zp is not None else "uint8")
    return points


def sqnr_pass(pp_path, points, imgs, chunk=48, keep=4096):
    """Per quantized tensor: deployed range, isolated SQNR, clipped share and value percentiles, from
    the pre-processed FP32 graph run on `imgs` with every quantized tensor exposed as an output."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto
    base = onnx.load(pp_path)
    producer, depth = {}, {}
    nodes = list(base.graph.node)
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o], depth[o] = n.op_type, i / max(1, len(nodes) - 1)
    inputs = {i.name for i in base.graph.input}
    names = [t for t in points if t in producer or t in inputs]
    if len(names) < len(points):
        missing = [t for t in points if t not in names]
        raise RuntimeError(f"{len(missing)} of {len(points)} quantized tensors are not in the pre-processed graph, "
                           f"for example {missing[:3]}")
    rng = np.random.default_rng(0)
    rows = []
    for c0 in range(0, len(names), chunk):
        part = names[c0:c0 + chunk]
        m = onnx.load(pp_path)
        have = {o.name for o in m.graph.output}
        for t in part:
            if t not in have and t not in inputs:
                m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        want = [t for t in part if t not in inputs]
        acc = {t: {"sig": 0.0, "noise": 0.0, "noise_clip": 0.0, "clip": 0, "n": 0, "vmin": np.inf, "vmax": -np.inf,
                   "sample": []} for t in part}
        for b0 in range(0, len(imgs), 8):
            x = imgs[b0:b0 + 8]
            outs = dict(zip(want, sess.run(want, {in_name: x}))) if want else {}
            for t in part:
                v = (x if t in inputs else outs[t]).astype(np.float64).ravel()
                s_, z_, dt = points[t]
                qmin, qmax = (0, 255) if dt == "uint8" else (-128, 127)
                q = np.clip(np.round(v / s_) + z_, qmin, qmax)
                vq = (q - z_) * s_
                a = acc[t]
                a["sig"] += float(np.dot(v, v))
                a["noise"] += float(np.dot(v - vq, v - vq))
                lo, hi = (qmin - z_) * s_, (qmax - z_) * s_
                out_ = (v < lo) | (v > hi)
                a["clip"] += int(np.count_nonzero(out_))
                a["noise_clip"] += float(np.dot((v - vq)[out_], (v - vq)[out_]))
                a["n"] += v.size
                a["vmin"], a["vmax"] = min(a["vmin"], float(v.min())), max(a["vmax"], float(v.max()))
                a["sample"].append(v[rng.integers(0, v.size, size=min(keep, v.size))])
        del sess
        for t in part:
            a = acc[t]
            s_, z_, dt = points[t]
            qmin, qmax = (0, 255) if dt == "uint8" else (-128, 127)
            lo, hi = (qmin - z_) * s_, (qmax - z_) * s_
            smp = np.concatenate(a["sample"])
            p001, p999 = (float(v) for v in np.percentile(smp, [0.1, 99.9]))
            sig, noise = a["sig"] / a["n"], a["noise"] / a["n"]
            rows.append({"tensor": t, "producer": producer.get(t, "input"), "depth": round(depth.get(t, 0.0), 4),
                         "scale": s_, "zero_point": z_, "lo": lo, "hi": hi, "n_values": a["n"],
                         "signal_power": sig, "noise_power": noise,
                         "sqnr_db": 10 * np.log10(sig / noise) if noise > 0 else float("inf"),
                         "clip_frac": a["clip"] / a["n"],
                         "clip_noise_share": a["noise_clip"] / a["noise"] if a["noise"] > 0 else 0.0, "p001": p001, "p999": p999, "vmin": a["vmin"],
                         "vmax": a["vmax"], "range_ratio": (hi - lo) / (p999 - p001) if p999 > p001 else float("nan")})
        gc.collect()
    return rows


def write_csv(path, fields, rows):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--only-sqnr", action="store_true", help="FP32 models and the SQNR subset only")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    out = os.path.join(QJ.REPORTS, "mech", a.tag)
    for d in ("emb", "sqnr"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    lock = os.path.join(out, ".lock")
    try:
        fd = take_lock(lock, f"{out} is being written by another process; check, then delete {lock}")
    except FileExistsError as exc:
        sys.exit(str(exc))
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _exit_on_signal)
    models_csv = os.path.join(out, "models.csv")
    done = {r["model"]: r for r in csv.DictReader(open(models_csv))} if os.path.exists(models_csv) else {}
    try:
        df = analysis_set(os.path.join(out, "analysis_set.csv"))
        students = a.students or QJ.STUDENTS
        jobs = plan(students, a.only_sqnr)
        todo = []
        for cell, s, t, rule, sq in jobs:
            mid = model_id(cell, s, t, rule)
            emb = os.path.join(out, "emb", mid + ".npy")
            sqf = os.path.join(out, "sqnr", mid + ".csv")
            if os.path.exists(emb) and (not sq or os.path.exists(sqf)):
                continue
            todo.append((cell, s, t, rule, sq, mid, emb, sqf))
        print(f"[mech] {len(df)} analysis images ({(df.role == 'query').sum()} queries, "
              f"{(df.role == 'gallery').sum()} references, {int(df.sqnr.sum())} for SQNR); {len(jobs)} models, "
              f"{len(todo)} to measure -> {out}", flush=True)
        if not todo:
            return
        full = loader(df)
        sq_imgs = np.concatenate([b[0].numpy() for b in loader(df[df.sqnr], batch_size=16, workers=2)]).astype(np.float32)
        calib = S.calibration_loader(QJ.IMG_SIZE, {S.TEST_FOLD}, QJ.CALIB_SEED, QJ.CALIB_N)
        from _qakd_export import export_clipped_onnx
        gpu = torch.cuda.is_available()
        for i, (cell, s, t, rule, sq, mid, emb, sqf) in enumerate(todo, 1):
            path, clipped = checkpoint(cell, s, t)
            graph = QJ.deploy_graph(s) if s else "plain"
            row = {"model": mid, "cell": cell, "student": s or "", "teacher": t or "", "rule": rule, "graph": graph,
                   "path": path, "sha": "", "preprocess": "", "n_images": len(df), "n_tensors": "", "seconds": "",
                   "status": "FAILED", "error": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t0 = time.time()
            try:
                if not os.path.exists(path) or (clipped and not os.path.exists(path + ".clip.pt")):
                    raise FileNotFoundError(path)
                row["sha"] = fingerprint(path)["sha"]
                with tempfile.TemporaryDirectory(prefix="qj_mech_") as tmp:
                    f32 = os.path.join(tmp, "fp32.onnx")
                    if clipped:
                        export_clipped_onnx(s, path, path + ".clip.pt", f32, num_classes=QJ.NUM_CLASSES,
                                            img_size=QJ.IMG_SIZE)
                    else:
                        export_fp32_onnx(s or t, path, f32, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                                         pooling="gem", device=torch.device("cpu"), fuse_branches=graph == "fused")
                    if rule == "fp32":
                        feats = extract_features_onnx(f32, full, provider="CUDAExecutionProvider" if gpu
                                                      else "CPUExecutionProvider", desc=mid)[0]
                    else:
                        i8, pp = os.path.join(tmp, "int8.onnx"), os.path.join(tmp, "pre.onnx")
                        run_ptq(f32, i8, calib, per_channel=True, keep_preprocessed=pp if sq else None,
                                **RULE_KW[rule])
                        row["preprocess"] = "ok" if QU.LAST_PREPROCESS.get("ok") else \
                            f"raw ({QU.LAST_PREPROCESS.get('error')})"
                        feats = extract_features_onnx(i8, full, provider="CPUExecutionProvider", desc=mid)[0]
                        if sq:
                            rows = sqnr_pass(pp, activation_quant_points(i8), sq_imgs)
                            row["n_tensors"] = len(rows)
                            write_csv(sqf, SQNR_FIELDS, rows)
                    if feats.shape[0] != len(df):
                        raise RuntimeError(f"{feats.shape[0]} embeddings for {len(df)} images")
                    tmp_npy = emb + f".tmp-{os.getpid()}.npy"
                    np.save(tmp_npy, feats.numpy().astype(np.float16))
                    os.replace(tmp_npy, emb)
                row["status"] = "SUCCESS"
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - t0)
            done[mid] = row
            write_csv(models_csv, MODEL_FIELDS, list(done.values()))
            print(f"[{i:3d}/{len(todo)}] {row['status']:7s} {mid:70s} {row['seconds']:4d}s "
                  f"{'tensors ' + str(row['n_tensors']) if row['n_tensors'] != '' else ''} "
                  f"{'' if row['preprocess'] in ('ok', '') else 'PREPROCESS ' + row['preprocess']} {row['error']}",
                  flush=True)
    finally:
        os.remove(lock)
    failed = [r for r in done.values() if r["status"] != "SUCCESS"]
    print(f"\n{len(done)} models recorded, {len(failed)} failed")
    for r in failed[:20]:
        print("   FAILED", r["model"], r["error"])
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
