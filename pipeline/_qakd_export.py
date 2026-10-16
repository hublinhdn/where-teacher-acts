"""
QAKD: ONNX export of a learnable-clip (LSQ) checkpoint with its learned Clip nodes in the graph.

The clean state dict loads into an unmodified PillRetrievalModel; the sidecar restores every
observer's learned range; `set_deploy_clip` turns the wrapped model into "original network +
Clip at the trained positions" (no rounding, BN as in the original graph). Exported with the
same opset / input / output names as the inherited FP32 exporter (export_fp32_onnx) so the
downstream PTQ + evaluation code is identical. ORT then folds BN, calibrates MinMax on the clipped
tensors and quantizes: the deployed ranges are the trained ranges by construction.
"""
import os
import sys
import gc
import inspect

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.abspath(os.path.join(THIS, "..")))
from models.pill_retrieval_model import PillRetrievalModel  # noqa: E402
from _qakd_fakequant import prepare_fakequant, load_observer_state, set_deploy_clip, ActObserver  # noqa: E402

INPUT_NAMES = ["input_image"]
OUTPUT_NAMES = ["output_embedding"]


def build_deploy_model(backbone, weight_path, sidecar_path, num_classes, margin=0.2):
    m = PillRetrievalModel(num_classes=num_classes, backbone_type=backbone, pooling_type="gem", m=margin)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    # where-teacher-acts: a checkpoint saved from a fused RepVGG (models/reparam.py) loads into a fused model
    from models.reparam import is_fused_state, fuse_repvgg_branches
    if is_fused_state(state) and fuse_repvgg_branches(m) == 0:
        raise RuntimeError(f"{weight_path} is a fused RepVGG checkpoint but {backbone} has no RepVGG block")
    m.load_state_dict(state, strict=True)
    prepare_fakequant(m, learnable_clip=True)
    st = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    n = load_observer_state(m, st)
    # where-teacher-acts: every observer of the model must get its trained range and every range in the
    # sidecar must find its observer; a partial load would export untrained ranges silently
    names = {k for k, o in m.named_modules() if isinstance(o, ActObserver)}
    if n == 0 or n != len(names) or set(st) != names:
        raise RuntimeError(f"{sidecar_path}: loaded {n} of {len(names)} observers; "
                           f"{len(set(st) - names)} sidecar entries without an observer, "
                           f"{len(names - set(st))} observers without an entry")
    set_deploy_clip(m, True)
    return m.eval()


def export_clipped_onnx(backbone, weight_path, sidecar_path, onnx_out_path, num_classes, img_size=384):
    os.makedirs(os.path.dirname(onnx_out_path), exist_ok=True)
    m = build_deploy_model(backbone, weight_path, sidecar_path, num_classes)
    dummy = torch.randn(1, 3, img_size, img_size)
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(m, dummy, onnx_out_path, export_params=True, opset_version=14,
                      do_constant_folding=True, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                      dynamic_axes={INPUT_NAMES[0]: {0: "batch_size"}, OUTPUT_NAMES[0]: {0: "batch_size"}},
                      **extra)
    del m
    gc.collect()
    return onnx_out_path


def count_clip_nodes(onnx_path):
    import onnx
    g = onnx.load(onnx_path).graph
    return sum(1 for n in g.node if n.op_type in ("Clip", "Min", "Max"))
