"""
where-teacher-acts: proves, by running it on the CPU, that RepVGG-A0 is deployed in fused form without
changing what the network computes, and that every other student is untouched.

  R1  fusion folds all 22 multi-branch blocks into single convolutions and the FP32 embedding does
      not change (a fresh pretrained network, or a trained checkpoint with --checkpoint)
  R2  a fused state dict is recognized and loads strictly into a freshly fused network
  R3  fake quantization wraps the 22 fused convolutions and no BatchNorm stays inside the backbone;
      the clean state dict of the wrapped network loads into a fused network
  R4  the FP32 ONNX export of the fused network gives the same embedding as the multi-branch export,
      with fewer nodes (no identity branch, no branch sum)
  R5  a fused Part II checkpoint with its clip ranges exports through build_deploy_model and runs
  R6  nothing changes for the other students: no RepVGG block to fuse, plain state dicts, and
      fuse_branches=True on them is refused

Writes only under weights/qj/_selftest_fused and removes it. Exit 1 on any failure.

    python pipeline/07_sanity_fused.py [--checkpoint weights/qj/fp32/base/repvgg_a0_timm/none/seed42/model.pth]
"""
import os
import sys
import shutil
import argparse
import tempfile

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import _qj_common as QJ  # noqa: E402
from _qj_io import atomic_save, WRITE_ROOT  # noqa: E402
from models.pill_retrieval_model import PillRetrievalModel  # noqa: E402
from models.reparam import fuse_repvgg_branches, count_unfused, is_fused_state  # noqa: E402

STUDENT = "repvgg_a0_timm"
SELFTEST = os.path.join(WRITE_ROOT, "_selftest_fused")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"   {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def build(student=STUDENT, state=None):
    m = PillRetrievalModel(num_classes=QJ.NUM_CLASSES, backbone_type=student, pooling_type="gem")
    if state is not None:
        m.load_state_dict(state, strict=True)
    return m.eval()


def ort_embed(path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="a trained multi-branch RepVGG-A0 state dict (FP32 A or B)")
    a = ap.parse_args()
    torch.manual_seed(0)
    x = torch.randn(2, 3, QJ.IMG_SIZE, QJ.IMG_SIZE)
    shutil.rmtree(SELFTEST, ignore_errors=True)
    try:
        print("R1  fusion is exact")
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=True) if a.checkpoint else None
        m = build(state=state)
        with torch.no_grad():
            e0 = m(x)
        n = fuse_repvgg_branches(m)
        with torch.no_grad():
            e1 = m(x)
        diff = float((e0 - e1).abs().max())
        cos = float(torch.nn.functional.cosine_similarity(e0, e1).min())
        check("22 multi-branch blocks fused, none left", n == 22 and count_unfused(m) == 0, f"fused {n}")
        check("the FP32 embedding is unchanged by fusion", diff < 1e-4 and cos > 0.99999,
              f"max |diff| {diff:.2e}, min cosine {cos:.7f}" + (", trained checkpoint" if state else ", pretrained"))

        print("R2  fused state dicts")
        fused_state = m.state_dict()
        check("a fused state dict is recognized; a multi-branch one is not",
              is_fused_state(fused_state) and not is_fused_state(build().state_dict()))
        m2 = build()
        fuse_repvgg_branches(m2)
        m2.load_state_dict(fused_state, strict=True)
        with torch.no_grad():
            e2 = m2(x)
        check("it loads strictly into a freshly fused network, same embedding", float((e2 - e1).abs().max()) < 1e-6)

        print("R3  fake quantization on the fused network")
        from _qakd_fakequant import prepare_fakequant, clean_state_dict, observer_state, BNShim, QWrap
        mq = build(state=state)
        fuse_repvgg_branches(mq)
        stats = prepare_fakequant(mq, learnable_clip=True)
        wrapped_feat = sum(1 for _ in mq.features.modules() if isinstance(_, QWrap))
        bn_feat = sum(1 for _ in mq.features.modules() if isinstance(_, (torch.nn.BatchNorm2d, BNShim)))
        check("the 22 fused convolutions are wrapped, no BatchNorm left in the backbone",
              wrapped_feat == 22 and bn_feat == 0, f"wrapped {wrapped_feat} in features, {bn_feat} BN; {stats}")
        cs = clean_state_dict(mq)
        m3 = build()
        fuse_repvgg_branches(m3)
        try:
            m3.load_state_dict(cs, strict=True)
            ok = True
        except RuntimeError as exc:
            ok, cs_err = False, str(exc)[:200]
        check("its clean state dict loads strictly into a fused network", ok, "" if ok else cs_err)

        print("R4  FP32 ONNX export, multi-branch against fused")
        from journal2_pipeline._quant_utils import export_fp32_onnx
        os.makedirs(SELFTEST, exist_ok=True)
        pth = os.path.join(SELFTEST, "a.pth")
        atomic_save(build(state=state).state_dict(), pth)
        with tempfile.TemporaryDirectory() as tmp:
            f_multi, f_fused = os.path.join(tmp, "multi.onnx"), os.path.join(tmp, "fused.onnx")
            export_fp32_onnx(STUDENT, pth, f_multi, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE)
            export_fp32_onnx(STUDENT, pth, f_fused, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE,
                             fuse_branches=True)
            o_multi, o_fused = ort_embed(f_multi, x), ort_embed(f_fused, x)
            import onnx
            ops = lambda p: [nd.op_type for nd in onnx.load(p).graph.node]  # noqa: E731
            nm, nf = ops(f_multi), ops(f_fused)
        d = float(np.abs(o_multi - o_fused).max())
        check("same embedding from both graphs", d < 1e-4, f"max |diff| {d:.2e}")
        check("the fused graph has fewer convolutions and no branch sums",
              nf.count("Conv") < nm.count("Conv") and nf.count("Add") < nm.count("Add"),
              f"Conv {nm.count('Conv')} -> {nf.count('Conv')}, Add {nm.count('Add')} -> {nf.count('Add')}")

        print("R5  a fused Part II checkpoint deploys")
        from _qakd_export import export_clipped_onnx
        with torch.no_grad():                     # give the observers ranges, as training would
            mq.train()
            from _qakd_fakequant import enable_quant
            enable_quant(mq, enabled=True)
            mq(x)
            mq.eval()
        w, sc = os.path.join(SELFTEST, "model.pth"), os.path.join(SELFTEST, "model.pth.clip.pt")
        atomic_save(clean_state_dict(mq), w)
        atomic_save(observer_state(mq), sc)
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "p2.onnx")
            try:
                export_clipped_onnx(STUDENT, w, sc, f, num_classes=QJ.NUM_CLASSES, img_size=QJ.IMG_SIZE)
                o = ort_embed(f, x)
                ok, detail = bool(np.isfinite(o).all()) and o.shape == (2, 512), f"output {o.shape}"
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        check("export with clip ranges (every observer loaded) and ONNX Runtime runs it", ok, detail)

        print("R6  the other students are untouched")
        others = [s for s in QJ.STUDENTS if s not in QJ.FUSED_STUDENTS]
        check("only RepVGG-A0 is deployed fused", QJ.FUSED_STUDENTS == (STUDENT,)
              and all(QJ.deploy_graph(s) == "plain" for s in others))
        r18 = build("resnet18_tv")
        check("a ResNet-18 has nothing to fuse and a plain state dict",
              fuse_repvgg_branches(r18) == 0 and not is_fused_state(r18.state_dict()))
        p18 = os.path.join(SELFTEST, "r18.pth")
        atomic_save(r18.state_dict(), p18)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                export_fp32_onnx("resnet18_tv", p18, os.path.join(tmp, "r.onnx"), num_classes=QJ.NUM_CLASSES,
                                 img_size=QJ.IMG_SIZE, fuse_branches=True)
                refused = False
            except ValueError:
                refused = True
        check("fuse_branches=True is refused for a network without RepVGG blocks", refused)
    finally:
        shutil.rmtree(SELFTEST, ignore_errors=True)
    ok = all(results)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
