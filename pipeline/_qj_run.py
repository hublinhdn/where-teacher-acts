"""
where-teacher-acts: trains one model of this study's FP32 grid, in its own process, under this study's
split, and records it in the run directory it owns.

    python pipeline/_qj_run.py --role teacher --model convnext_small_timm
    python pipeline/_qj_run.py --role base    --model resnet18_tv
    python pipeline/_qj_run.py --role kd      --model resnet18_tv --teacher convnext_small_timm

Teachers use the inherited baseline trainer (_qj_train_base.py). Students use the KD trainer
(_qakd_train_kd.py with fake quantization off, whose arithmetic is the inherited KD trainer's): kd runs
with their teacher, base runs (A) with the teacher switched off (no_kd), so A and B share one recipe and
differ only by the teacher signal (see _qj_common.A_RECIPE_TEACHER). The arguments are the ones the earlier
study passed: loss weights 1.0 / 0.2 / 1.0 / 1.0, 60 epochs (raised by the trainers' own group rules),
cosine KD with alpha 10 and temperature 4, seed 42 set the way the inherited KD trainer sets it, and one
image size for every model (_qj_common.IMG_SIZE, 224 x 224, the size the images are stored at). The
earlier study left its baseline trainer unseeded; here every role is seeded, and cuDNN keeps the mode each
inherited trainer ran in (deterministic for the KD trainer, default for teachers).

The split is _qj_split.load_split(): train folds 1 to 3, validation fold 0. The test fold is never
loaded by this script. A kd run reads its teacher from this study's own teacher run, never from a
file of the earlier study. run.json records the split, the trainer source, the teacher's fingerprint, every
validation result and the selected epoch. --smoke caps training at one epoch and writes under
weights/qj/smoke; it checks the code path and produces no result.
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

LOSS_WEIGHTS = dict(w_sce=1.0, w_csce=0.2, w_triplet=1.0, w_cont=1.0)
EPOCHS = 60
TRAINER = {"teacher": "pipeline/_qj_train_base.py", "base": "pipeline/_qakd_train_kd.py",
           "kd": "pipeline/_qakd_train_kd.py"}


def seed_everything(seed, deterministic):
    """What the inherited KD trainer does in main(). The inherited baseline trainer set no seed at all; here
    it gets the same seeds, and cuDNN keeps the mode each trainer ran in (deterministic for the KD trainer)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def pick_device(smoke):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not smoke:
        sys.exit("no CUDA device: refusing to train a real run on the CPU (check the driver)")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _exit_on_signal(signum, _frame):
    """SIGTERM or SIGHUP (tmux killed, shutdown): unwind normally so RunClaim marks the run failed
    and removes its lock, instead of dying with the lock left behind."""
    raise SystemExit(f"stopped by signal {signum}")


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def done(rdir):
    try:
        with open(os.path.join(rdir, "run.json")) as f:
            return json.load(f).get("status") == "done"
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=QJ.ROLES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--seed", type=int, default=QJ.FP32_SEED)
    ap.add_argument("--smoke", action="store_true", help="one epoch under weights/qj/smoke, no result")
    ap.add_argument("--teacher-weight", default=None,
                    help="smoke only: a teacher file to use before this study's teacher run exists")
    ap.add_argument("--overwrite", action="store_true", help="redo a finished run on purpose")
    a = ap.parse_args()
    if a.teacher_weight and not a.smoke:
        sys.exit("--teacher-weight is for smoke tests; real kd runs read this study's own teacher")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    root = os.path.join(QJ.WEIGHT_DIR, "smoke") if a.smoke else None
    rdir = QJ.fp32_dir(a.role, a.model, a.teacher, a.seed, root=root)
    split = S.load_split()                      # raises unless the split is exactly the design's

    meta = {"role": a.role, "model": a.model, "teacher": a.teacher, "seed": a.seed, "smoke": a.smoke,
            "split": {"train_folds": list(S.TRAIN_FOLDS), "val_fold": S.VAL_FOLD, "test_fold": S.TEST_FOLD,
                      "counts": split.counts()},
            "trainer": TRAINER[a.role], "trainer_sha": file_sha(TRAINER[a.role]),
            "recipe": dict(LOSS_WEIGHTS, epochs=EPOCHS, pooling="gem", img_size=QJ.IMG_SIZE,
                           **({"kd_type": QJ.KD_TYPE, "alpha": QJ.KD_ALPHA, "temperature": QJ.KD_TEMPERATURE}
                              if a.role == "kd" else {})),
            "torch": torch.__version__}
    teacher_weight = None
    if a.role == "kd":
        trun = QJ.fp32_dir("teacher", a.teacher, None, QJ.FP32_SEED, root=root)   # teachers: seed 42 only
        teacher_weight = a.teacher_weight or os.path.join(trun, "model.pth")
        if not a.teacher_weight and not done(trun):
            sys.exit(f"teacher run {trun} is not finished; train the teacher first")
        meta["teacher_weight"] = teacher_weight
        meta["teacher_fingerprint"] = fingerprint(teacher_weight)

    if a.role == "base":
        meta["recipe_teacher"] = QJ.A_RECIPE_TEACHER       # layout only; the teacher is not loaded
    seed_everything(a.seed, deterministic=(a.role != "teacher"))
    device = pick_device(a.smoke)
    import signal
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _exit_on_signal)
    meta["device"] = str(device)
    t0 = time.time()
    with RunClaim(rdir, meta, overwrite=a.overwrite) as claim:
        args = argparse.Namespace(pooling="gem", epochs=EPOCHS, pipeline_name="qj_fp32", seed=a.seed,
                                  img_size=QJ.IMG_SIZE, run_claim=claim, val_curve=[], **LOSS_WEIGHTS)
        if a.smoke:
            args.max_epochs_cap = 1
        if a.role != "teacher":
            args.reseed_student = a.seed      # A and B share the student's random stream
        if a.role == "teacher":
            from _qj_train_base import train_one_fold
            args.backbone = a.model
            best, r1, _ = train_one_fold(args, S.VAL_FOLD, split.num_sub_classes, split.train, split.val,
                                         split.gallery, device)
        else:
            from _qakd_train_kd import train_kd_fold
            args.student = a.model
            args.kd_type, args.alpha, args.temperature = QJ.KD_TYPE, QJ.KD_ALPHA, QJ.KD_TEMPERATURE
            args.lambda_topo, args.topo_method, args.topo_warmup_epochs = 0.0, "mst_h0", 0
            if a.role == "kd":
                args.teacher, args.teacher_weight = a.teacher, teacher_weight
            else:
                args.teacher, args.teacher_weight, args.no_kd = QJ.A_RECIPE_TEACHER, "", True
            best, r1, _ = train_kd_fold(args, S.VAL_FOLD, split.num_sub_classes, split.train, split.val,
                                        split.gallery, device)
        model_path = os.path.join(rdir, "model.pth")
        if not (math.isfinite(float(best)) and float(best) > 0.0):
            raise RuntimeError(f"best validation mAP {best} is not a positive number: no usable checkpoint")
        if "model.pth" not in claim.saved or not os.path.exists(model_path):
            raise RuntimeError("the trainer finished without saving a checkpoint in this attempt")
        curve = args.val_curve
        best_i = max(range(len(curve)), key=lambda i: (curve[i]["mAP"], -i)) if curve else None
        claim.finish(val_mAP=round(float(best), 6), val_rank1=round(float(r1), 6),
                     best_epoch=curve[best_i]["epoch"] if curve else None, val_curve=curve,
                     minutes=round((time.time() - t0) / 60, 1), model_fingerprint=fingerprint(model_path))
    print(f"[qj_run] {a.role} {a.model}{' <- ' + a.teacher if a.teacher else ''}: val mAP {best:.4f} "
          f"-> {rdir}", flush=True)


if __name__ == "__main__":
    main()
