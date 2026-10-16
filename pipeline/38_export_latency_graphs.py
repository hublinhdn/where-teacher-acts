"""
where-teacher-acts: write the graphs of the deployment-cost benchmark to disk, for timing on other devices.

pipeline/27_bench_latency.py exports and quantizes its graphs in a temporary folder and times them on the
desktop CPU. This script produces the same graphs, with the same calls and the same calibration set, and
keeps them, with a manifest, so that android/run_android_bench.py can time them on a phone: for every
student, A (the PTQ route) and QAKD(B) distilled from ConvNeXt-S (learned ranges as Clip nodes), each as
FP32 ONNX and as INT8 (ONNX Runtime static PTQ, QDQ, MinMax, per-channel int8 weights, per-tensor uint8
activations). RepVGG-A0 is exported fused, as it is deployed. Refuses to overwrite unless --force.

    python pipeline/38_export_latency_graphs.py [--tag main] [--students resnet18_tv]

Output: reports_qj/bench/<tag>/latency_graphs/{*.onnx, manifest.csv}; the manifest records the size and
SHA-256 of every file, and the sizes can be checked against latency_rr5.csv.
"""
import os
import sys
import csv
import hashlib
import argparse
import importlib.util

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
_spec = importlib.util.spec_from_file_location("qj_bench_latency", os.path.join(THIS, "27_bench_latency.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)        # imports, the shared code path and the move to the project root

import torch  # noqa: E402
import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from journal2_pipeline._quant_utils import export_fp32_onnx, run_ptq  # noqa: E402
from journal2_pipeline._benchmark_utils import get_model_size_mb  # noqa: E402

FIELDS = ["student", "network", "precision", "graph", "params_backbone_M", "params_deployed_M", "size_MB",
          "file", "sha256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(QJ.REPORTS, "bench", a.tag, "latency_graphs")
    if os.path.exists(os.path.join(out_dir, "manifest.csv")) and not a.force:
        sys.exit(f"{out_dir} holds a manifest; pass --force to overwrite it")
    os.makedirs(out_dir, exist_ok=True)
    S.load_split()
    calib = S.calibration_loader(QJ.IMG_SIZE, {S.TEST_FOLD}, QJ.CALIB_SEED, QJ.CALIB_N)
    from _qakd_export import export_clipped_onnx
    rows, failed = [], []
    for s in a.students or QJ.STUDENTS:
        pb, pd = B.deployed_params(s)
        graph = QJ.deploy_graph(s)
        a_path = os.path.join(QJ.fp32_dir("base", s), "model.pth")
        k_dir = QJ.run_dir("qakd_b", s, QJ.SEED_REPEAT_TEACHER, 40, QJ.FP32_SEED)
        nets = {"A": f"{s}_a_fp32.onnx", "QAKD(B)": f"{s}_k_fp32.onnx"}
        try:
            export_fp32_onnx(s, a_path, os.path.join(out_dir, nets["A"]), num_classes=QJ.NUM_CLASSES,
                             img_size=QJ.IMG_SIZE, pooling="gem", device=torch.device("cpu"),
                             fuse_branches=graph == "fused")
            export_clipped_onnx(s, os.path.join(k_dir, "model.pth"), os.path.join(k_dir, "model.pth.clip.pt"),
                                os.path.join(out_dir, nets["QAKD(B)"]), num_classes=QJ.NUM_CLASSES,
                                img_size=QJ.IMG_SIZE)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{s}: export: {type(exc).__name__}: {exc}")
            continue
        for net, f32 in nets.items():
            i8 = f32.replace("_fp32.onnx", "_int8.onnx")
            try:
                run_ptq(os.path.join(out_dir, f32), os.path.join(out_dir, i8), calib, per_channel=True,
                        calibration="minmax")
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{s} {net}: quantization: {type(exc).__name__}: {exc}")
                continue
            for prec, fname in (("FP32", f32), ("INT8", i8)):
                p = os.path.join(out_dir, fname)
                rows.append({"student": s, "network": net, "precision": prec, "graph": graph,
                             "params_backbone_M": round(pb, 3), "params_deployed_M": round(pd, 3),
                             "size_MB": round(get_model_size_mb(p), 2), "file": fname, "sha256": sha256(p)})
        print(f"   exported {s}", flush=True)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for msg in failed:
        print(f"   FAILED {msg}")
    print(f"{len(rows)} graphs -> {out_dir}\nRESULT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
