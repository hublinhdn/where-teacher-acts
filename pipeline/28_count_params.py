"""
where-teacher-acts: parameter counts of the thirteen models of this study, read-only.

For every student and teacher, the network the deployed graph runs (backbone, pooling, projection,
embedding BatchNorm; no training heads), with RepVGG-A0 fused as it is deployed, and the training
model with its two classification heads, in millions. The architecture is built by the study's
model wrapper and the study's own FP32 checkpoint (seed 42) is loaded strictly into it, so the
count is that of the trained network. Writes reports_qj/bench/<tag>/params.csv.

    python pipeline/28_count_params.py [--tag main]
"""
import os
import sys
import csv
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import torch  # noqa: E402
import _qj_common as QJ  # noqa: E402
from models.pill_retrieval_model import PillRetrievalModel  # noqa: E402
from models.reparam import fuse_repvgg_branches  # noqa: E402

FIELDS = ["model", "role", "graph", "params_backbone_M", "params_deployed_M", "params_training_M", "checkpoint"]
DEPLOYED = ("features", "pool", "fc_projection", "bn_head")


def count(model, role):
    m = PillRetrievalModel(num_classes=QJ.NUM_CLASSES, backbone_type=model, pooling_type="gem").eval()
    path = os.path.join(QJ.fp32_dir("teacher" if role == "teacher" else "base", model), "model.pth")
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    training = sum(p.numel() for p in m.parameters())
    graph = QJ.deploy_graph(model) if role == "student" else "plain"
    if graph == "fused":
        fuse_repvgg_branches(m)
    part = {name: sum(p.numel() for p in getattr(m, name).parameters()) for name in DEPLOYED if hasattr(m, name)}
    return {"model": model, "role": role, "graph": graph, "params_backbone_M": round(part["features"] / 1e6, 3),
            "params_deployed_M": round(sum(part.values()) / 1e6, 3), "params_training_M": round(training / 1e6, 3),
            "checkpoint": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    out = os.path.join(QJ.REPORTS, "bench", a.tag, "params.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = [count(s, "student") for s in QJ.STUDENTS] + [count(t, "teacher") for t in QJ.TEACHERS]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"   {r['role']:8s} {r['model']:26s} {r['graph']:6s} backbone {r['params_backbone_M']:8.3f} M  "
              f"deployed {r['params_deployed_M']:8.3f} M  training {r['params_training_M']:8.3f} M")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
