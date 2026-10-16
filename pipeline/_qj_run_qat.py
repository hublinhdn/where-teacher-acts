"""
where-teacher-acts: trains one Part II run, in its own process, and records it in the run directory it owns.

    python pipeline/_qj_run_qat.py --arm qat_a  --student efficientnet_b1_timm --ft-epochs 20
    python pipeline/_qj_run_qat.py --arm qakd_b --student efficientnet_b1_timm --teacher convnext_small_timm --ft-epochs 20

The four cells of the factorial (the dated plan, docs/PLAN_AND_LOG.md): where the run starts and whether
the teacher acts during quantization-aware fine-tuning.

    qat_a   starts from A, teacher off         qakd_a  starts from A, teacher on
    qat_b   starts from B(teacher), teacher off qakd_b  starts from B(teacher), teacher on
    qanc_b  starts from B(teacher); the frozen FP32 B itself takes the teacher's place in the cosine term
    qanc_a  starts from A (seed ED_START_SEED); the frozen FP32 A takes the teacher's place
    ed_ft, ed_kd               FP32 phase of the equal-dose comparison from A (seed ED_START_SEED), teacher off/on
    ed_none, ed_before, ed_during   quantization-aware phase: QAT from ed_ft, QAT from ed_kd, QAKD from ed_ft
    ed_anc     FP32 phase from A (seed ED_START_SEED) with the frozen FP32 A in the cosine term (self-anchored)
    qat_a_fs   QAT from A of seed ED_START_SEED with another fine-tuning seed (the fine-tuning noise of QAT(A))
    (the exploratory arms of _qj_common.EXTRA_ARMS)

A and B are this study's own FP32 checkpoints of the SAME seed as the run (seed 42 for the grid; the
seed repeats 123 and 223 start from the FP32 seed repeats, so a repeat replicates the whole chain,
FP32 training included); the teacher is this study's own teacher, seed 42 for every run.
Every arm uses the KD trainer (_qakd_train_kd.py) with fake quantization and learned clipping from
the first epoch (_qj_common.PART2), this study's split and image size, and the same student reseed,
so arms differ only by their start checkpoint and by the teacher term. Selection is the inherited
one: best validation mAP on fold 0 (simulated INT8 forward), checked every 10 epochs and in the last
5. --smoke caps training at one epoch under weights/qj/smoke/runs and may point at smoke checkpoints.
"""
import os
import sys
import json
import math
import time
import random
import hashlib
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _qj_common as QJ  # noqa: E402
import _qj_split as S  # noqa: E402
from _qj_io import RunClaim  # noqa: E402
from journal2_pipeline._provenance import fingerprint  # noqa: E402

TRAINER = "pipeline/_qakd_train_kd.py"
LOSS_WEIGHTS = dict(w_sce=1.0, w_csce=0.2, w_triplet=1.0, w_cont=1.0)


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def done(rdir):
    try:
        with open(os.path.join(rdir, "run.json")) as f:
            return json.load(f).get("status") == "done"
    except (OSError, ValueError):
        return False


def _exit_on_signal(signum, _frame):
    raise SystemExit(f"stopped by signal {signum}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=QJ.ARMS + QJ.EXTRA_ARMS)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--ft-epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=QJ.FP32_SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start-weight", default=None, help="smoke only: start checkpoint to use")
    ap.add_argument("--teacher-weight", default=None, help="smoke only: teacher file to use")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if (a.start_weight or a.teacher_weight) and not a.smoke:
        sys.exit("--start-weight and --teacher-weight are for smoke tests only")
    if a.arm in QJ.NO_TEACHER_ARMS and a.teacher:
        sys.exit(f"{a.arm} has no teacher")
    if a.arm not in QJ.NO_TEACHER_ARMS and not a.teacher:
        sys.exit(f"{a.arm} needs --teacher")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    smoke_root = os.path.join(QJ.WEIGHT_DIR, "smoke")
    rdir = QJ.run_dir(a.arm, a.student, a.teacher, a.ft_epochs, a.seed,
                      root=os.path.join(smoke_root, "runs") if a.smoke else None)
    split = S.load_split()

    fp32_only = a.arm in QJ.ED_FP32_ARMS             # the FP32 phase of the equal-dose comparison
    anchor = a.arm in ("qanc_b", "qanc_a", "ed_anc")  # the frozen FP32 start, not a teacher, in the cosine term
    real_teacher = a.arm in ("qakd_a", "qakd_b", "ed_kd", "ed_during")
    teacher_on = real_teacher or anchor
    fp32_root = smoke_root if a.smoke else None
    runs_root = os.path.join(smoke_root, "runs") if a.smoke else None
    # where the run starts: the grid's arms from the FP32 checkpoint of their own seed; the exploratory
    # arms from A of seed ED_START_SEED, or from the FP32 phase of the same seed
    if a.arm in ("qat_b", "qakd_b", "qanc_b"):
        start_dir, start_seed = QJ.fp32_dir("kd", a.student, a.teacher, a.seed, root=fp32_root), a.seed
    elif a.arm in ("qat_a", "qakd_a"):
        start_dir, start_seed = QJ.fp32_dir("base", a.student, None, a.seed, root=fp32_root), a.seed
    elif a.arm in ("qanc_a", "ed_ft", "ed_kd", "ed_anc", "qat_a_fs"):
        start_dir = QJ.fp32_dir("base", a.student, None, QJ.ED_START_SEED, root=fp32_root)
        start_seed = QJ.ED_START_SEED
    elif a.arm in ("ed_none", "ed_during"):
        start_dir, start_seed = QJ.run_dir("ed_ft", a.student, None, a.ft_epochs, a.seed, root=runs_root), a.seed
    else:   # ed_before
        start_dir = QJ.run_dir("ed_kd", a.student, a.teacher, a.ft_epochs, a.seed, root=runs_root)
        start_seed = a.seed
    start = a.start_weight or os.path.join(start_dir, "model.pth")
    if not a.start_weight and not done(start_dir):
        sys.exit(f"start checkpoint run {start_dir} is not finished")
    teacher_weight = ""
    if anchor:
        teacher_weight = start                     # the start anchors itself; no teacher network is loaded
    elif real_teacher:
        tdir = QJ.fp32_dir("teacher", a.teacher, None, QJ.FP32_SEED, root=fp32_root)   # teachers: seed 42 only
        teacher_weight = a.teacher_weight or os.path.join(tdir, "model.pth")
        if not a.teacher_weight and not done(tdir):
            sys.exit(f"teacher run {tdir} is not finished")

    meta = {"arm": a.arm, "student": a.student, "teacher": a.teacher, "ft_epochs": a.ft_epochs, "seed": a.seed,
            "smoke": a.smoke, "start": start, "start_seed": start_seed, "start_fingerprint": fingerprint(start),
            "teacher_on": teacher_on, "anchor": anchor, "fp32_only": fp32_only,
            "teacher_weight": teacher_weight or None,
            "teacher_fingerprint": fingerprint(teacher_weight) if teacher_weight else None,
            "split": {"train_folds": list(S.TRAIN_FOLDS), "val_fold": S.VAL_FOLD, "test_fold": S.TEST_FOLD},
            "trainer": TRAINER, "trainer_sha": file_sha(TRAINER), "deploy_graph": QJ.deploy_graph(a.student),
            "recipe": dict(LOSS_WEIGHTS, img_size=QJ.IMG_SIZE, kd_type=QJ.KD_TYPE, alpha=QJ.KD_ALPHA,
                           temperature=QJ.KD_TEMPERATURE, **QJ.PART2),
            "torch": torch.__version__}

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
    elif a.smoke:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        sys.exit("no CUDA device: refusing to train a real run on the CPU (check the driver)")
    import signal
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _exit_on_signal)
    meta["device"] = str(device)

    t0 = time.time()
    with RunClaim(rdir, meta, overwrite=a.overwrite) as claim:
        from _qakd_train_kd import train_kd_fold
        args = argparse.Namespace(
            student=a.student, epochs=60, ft_epochs=a.ft_epochs, init_from=start, pipeline_name="qj_part2",
            seed=a.seed, img_size=QJ.IMG_SIZE, run_claim=claim, val_curve=[], reseed_student=a.seed,
            kd_type=QJ.KD_TYPE, alpha=QJ.KD_ALPHA, temperature=QJ.KD_TEMPERATURE,
            lambda_topo=0.0, topo_method="mst_h0", topo_warmup_epochs=0, qakd=not fp32_only,
            **QJ.PART2, **LOSS_WEIGHTS)
        # the FP32 phase trains the multi-branch form, as A and B were; quantization-aware runs fold it
        args.fuse_branches = QJ.deploy_graph(a.student) == "fused" and not fp32_only
        if anchor:
            # the teacher name keeps the batch layout of every arm (all six teachers share it); the network
            # in the cosine term is the student's own architecture, loaded from its start
            args.teacher, args.teacher_weight, args.no_kd = (a.teacher or QJ.A_RECIPE_TEACHER), teacher_weight, False
            args.anchor_backbone = a.student
        elif real_teacher:
            args.teacher, args.teacher_weight, args.no_kd = a.teacher, teacher_weight, False
        else:
            args.teacher, args.teacher_weight, args.no_kd = (a.teacher or QJ.A_RECIPE_TEACHER), "", True
        if a.smoke:
            args.max_epochs_cap = 1
        best, r1, _ = train_kd_fold(args, S.VAL_FOLD, split.num_sub_classes, split.train, split.val,
                                    split.gallery, device)
        model_path = os.path.join(rdir, "model.pth")
        if not (math.isfinite(float(best)) and float(best) > 0.0):
            raise RuntimeError(f"best validation mAP {best} is not a positive number: no usable checkpoint")
        need = {"model.pth"} if fp32_only else {"model.pth", "model.pth.clip.pt"}
        if not (need <= claim.saved and all(os.path.exists(os.path.join(rdir, f)) for f in need)):
            raise RuntimeError("the trainer finished without saving the checkpoint"
                               + ("" if fp32_only else " and its clip ranges") + " in this attempt")
        curve = args.val_curve
        best_i = max(range(len(curve)), key=lambda i: (curve[i]["mAP"], -i)) if curve else None
        claim.finish(val_mAP_sim=round(float(best), 6), val_rank1_sim=round(float(r1), 6),
                     best_epoch=curve[best_i]["epoch"] if curve else None, val_curve=curve,
                     minutes=round((time.time() - t0) / 60, 1), model_fingerprint=fingerprint(model_path))
    print(f"[qj_run_qat] {a.arm} {a.student}{' <- ' + a.teacher if a.teacher else ''} N={a.ft_epochs} "
          f"seed {a.seed}: val mAP (simulated INT8) {best:.4f} -> {rdir}", flush=True)


if __name__ == "__main__":
    main()
