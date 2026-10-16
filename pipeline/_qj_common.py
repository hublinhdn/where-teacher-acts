"""
where-teacher-acts: shared constants, the single source of truth.

Students are seven convolutional students chosen with preliminary experiments on another split of ePillID
to span a wide range of INT8 floors, listed by their floor under MinMax post-training quantization in those
experiments, highest first. Teachers are six that improved the students in FP32 there and fit beside a
quantization-aware student on a 10 GB card.

This study trains its own models on its own split (_qj_split: train folds 1 to 3, validation
fold 0, test fold 4): six teachers, the undistilled students A and the distilled students B, all
with the same trainers and recipe. Part II starts from these A and B.
"""
import os

STUDENTS = [
    "resnet18_tv",
    "squeezenet1_0_tv",
    "mobilenet_v3_large_tv",
    "ghostnet_100_timm",
    "repvgg_a0_timm",
    "efficientnet_b0_timm",
    "efficientnet_b1_timm",
]
TEACHERS = [
    "efficientnet_b5_timm",
    "seresnext101_32x4d_timm",
    "convnext_base_timm",
    "convnext_small_timm",
    "resnet101_tv",
    "densenet161_tv",
]

# The inherited distillation recipe (cosine objective), reused for the teacher-on arms.
KD_TYPE, KD_ALPHA, KD_TEMPERATURE = "cosine", 10.0, 4.0

REPORTS = "reports_qj"
WEIGHT_DIR = os.path.join("weights", "qj")

# Deployment protocol, identical to the inherited one: ONNX Runtime static PTQ, 256 calibration images drawn
# with seed 123 from the training folds of both domains, per-channel int8 weights, per-tensor uint8
# activations, MinMax as the primary rule.
# One image size for every model of this study and every step (train, validation, ONNX export,
# calibration, test): 224 x 224, the size ePillID images are stored at (every image on disk is
# 224 x 224, so training at 384 only upsampled them). None of this study's backbones needs another
# size.
IMG_SIZE = 224
NUM_CLASSES = 9804
CALIB_N = 256
CALIB_SEED = 123

# Students deployed in fused form (models/reparam.py): RepVGG-A0's training branches are folded into
# single convolutions before export and before Part II, the way RepVGG is meant to be deployed, so the
# simulated quantization of Part II is the one ONNX Runtime applies (decided 2026-10-11, after
# validation showed real INT8 far below the simulation for the training form). FP32 training is
# unchanged: A and B are trained in the multi-branch form and fused afterwards.
FUSED_STUDENTS = ("repvgg_a0_timm",)


def deploy_graph(model):
    """The graph a model is deployed and measured as: 'fused' or 'plain'."""
    return "fused" if model in FUSED_STUDENTS else "plain"


# The four cells of the factorial: where the run starts (A undistilled, B distilled) and whether
# the teacher acts during quantization (QAT: off, QAKD: on). QAT(A) has no teacher anywhere.
ARMS = ("qat_a", "qat_b", "qakd_a", "qakd_b")

# Exploratory arms added after the test results (internal reviews, 2026-10-13 and 14; the dated plan,
# docs/PLAN_AND_LOG.md). None of them is in ARMS, so the grid, its completeness checks and the planned
# analysis are unchanged.
#   qanc_b     QAT of B with B's own frozen FP32 model in place of the teacher in the cosine term (anchor),
#              from B distilled from ANCHOR_TEACHER (the teacher label names that start, as for qat_b).
#   qanc_a     the same anchor on A: QAT of A with A's own frozen FP32 model in the cosine term.
#   ed_*       the equal-dose comparison (declared 2026-10-14): from A, an FP32 phase of N epochs without
#              (ed_ft) or with (ed_kd) the teacher ED_TEACHER, then a quantization-aware phase of N epochs:
#              ed_none (QAT from ed_ft), ed_before (QAT from ed_kd), ed_during (QAKD from ed_ft).
#   ed_anc     sensitivity analysis (declared 2026-10-16): the FP32 phase of N epochs from A with A's own frozen
#              FP32 model in the cosine term (a self-anchored ed_kd), measured in FP32 and after PTQ.
#   qat_a_fs   sensitivity analysis (declared 2026-10-16): QAT(A) of the grid repeated with other fine-tuning
#              seeds from the same A (seed ED_START_SEED), for the fine-tuning noise of QAT(A) in every student.
# qanc_a, every ed_* run and qat_a_fs start from A of seed ED_START_SEED; the run seed varies the fine-tuning only.
EXTRA_ARMS = ("qanc_b", "qanc_a", "ed_ft", "ed_kd", "ed_none", "ed_before", "ed_during", "ed_anc", "qat_a_fs")
ANCHOR_TEACHER = "convnext_small_timm"
ED_TEACHER = "convnext_small_timm"
ED_START_SEED = 42
ED_SEEDS = (42, 123, 223)
ED_FP32_ARMS = ("ed_ft", "ed_kd", "ed_anc")            # trained in FP32, without the simulated quantizer
NO_TEACHER_ARMS = ("qat_a", "qanc_a", "ed_ft", "ed_none", "ed_anc", "qat_a_fs")   # run directories without a teacher label
FS_SEEDS = (123, 223)                                  # the extra fine-tuning seeds of qat_a_fs (seed 42 is the grid's)


def run_dir(arm, student, teacher, ft_epochs, seed, root=None):
    """The one directory a Part II run owns. Every factor of the design is a path component, so
    two different runs can never share a file: arm / student / teacher / fine-tune epochs / seed.
    `root` (smoke tests only) replaces weights/qj/runs."""
    if arm not in ARMS + EXTRA_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS + EXTRA_ARMS}")
    if student not in STUDENTS:
        raise ValueError(f"{student!r} is not a Part II student")
    if arm in NO_TEACHER_ARMS:
        if teacher is not None:
            raise ValueError(f"{arm} has no teacher")
    elif teacher not in TEACHERS:
        raise ValueError(f"{arm} needs one of the Part II teachers, got {teacher!r}")
    return os.path.join(root or os.path.join(WEIGHT_DIR, "runs"), arm, student, teacher or "none",
                        f"ft{int(ft_epochs)}", f"seed{int(seed)}")


# This study's FP32 models: teachers, undistilled students (A, role "base") and distilled
# students (B, role "kd"). Trained once, seed 42, under _qj_split.
ROLES = ("teacher", "base", "kd")
FP32_SEED = 42

# Training-noise repeats of the FP32 baselines: A and B (with the pilot teacher) of three students
# spanning the INT8 floor, retrained with two more seeds. Teachers are trained once, with seed 42.
SEED_REPEAT_STUDENTS = ["efficientnet_b1_timm", "mobilenet_v3_large_tv", "resnet18_tv"]
SEED_REPEAT_TEACHER = "convnext_small_timm"
EXTRA_SEEDS = (123, 223)

# The undistilled students A are trained by the KD trainer with the teacher switched off (no_kd), so
# that A and B share one recipe and differ only by the teacher signal. The KD trainer derives image
# size and batch layout from the teacher's name; all six teachers give the same (384 px, 8 x 2 images
# per step, 8 accumulation steps), so the name below only fixes that layout. The earlier study trained A
# with its baseline trainer instead (224 px, 16 x 2, augmented references), which differs from B in more
# than the teacher.
A_RECIPE_TEACHER = "convnext_small_timm"

# Part II fine-tune recipe (the dated plan, docs/PLAN_AND_LOG.md): quantization and learned clipping from
# the first epoch, BN folded exactly and frozen for the last 5, LR starting where the recipe stood at
# epoch 41 of 80.
PART2 = dict(q_start_epoch=1, q_freeze_last=5, q_clip=True, lr_clip=5e-3, q_clip_warmup=20,
             q_quant_output=True, q_clip_penalty=0.0, ft_lr_from_epoch=41)
PILOT_STUDENT, PILOT_TEACHER, PILOT_EPOCHS = "efficientnet_b1_timm", "convnext_small_timm", (20, 40)

# Calibration-noise seeds for the INT8 benchmark (the first is the primary calibration seed).
CALIB_SEEDS = (123, 223, 323, 423, 523)


def fp32_dir(role, model, teacher=None, seed=FP32_SEED, root=None):
    """The one directory an FP32 run owns: role / model / teacher / seed."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    if role == "teacher":
        if model not in TEACHERS or teacher is not None:
            raise ValueError(f"teacher runs train one of {TEACHERS} and have no teacher")
    elif role == "base":
        if model not in STUDENTS or teacher is not None:
            raise ValueError(f"base runs train one of the students and have no teacher, got {model!r}, {teacher!r}")
    elif model not in STUDENTS or teacher not in TEACHERS:
        raise ValueError(f"kd runs pair a student with a Part II teacher, got {model!r} <- {teacher!r}")
    return os.path.join(root or os.path.join(WEIGHT_DIR, "fp32"), role, model, teacher or "none", f"seed{int(seed)}")


def start_checkpoint(student, teacher=None, seed=FP32_SEED):
    """This study's own checkpoint a Part II run of `seed` starts from: A (base), or B (kd) with
    `teacher`, both of the same seed, so a seed repeat replicates the whole chain (FP32 included)."""
    if teacher is None:
        return os.path.join(fp32_dir("base", student, None, seed), "model.pth")
    return os.path.join(fp32_dir("kd", student, teacher, seed), "model.pth")


def teacher_checkpoint(teacher):
    return os.path.join(fp32_dir("teacher", teacher), "model.pth")


def row_is_current(row):
    """Whether an evaluation row (22_eval_part2.py) still describes the checkpoint its run directory
    holds, deployed as this study deploys it: the run is done, its recorded fingerprint equals the
    row's, and the row measured the right graph (an FP32 reference: deploy_graph of its student; a
    Part II run: the graph its run.json records). A run redone on purpose invalidates its old rows.
    Where the run directory is absent (results synced to a machine without the checkpoints) the row
    is trusted as written apart from the graph rule; the evaluator marks such rows STALE itself."""
    import json
    kind, graph = row.get("kind", ""), row.get("graph") or "plain"   # rows before the column: plain
    if kind.startswith("fp32") and graph != deploy_graph(row.get("student", "")):
        return False
    meta = os.path.join(os.path.dirname(row.get("path", "")), "run.json")
    if not row.get("path") or not os.path.exists(meta):
        return True
    try:
        with open(meta) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return False
    if kind == "part2" and graph != st.get("deploy_graph", "plain"):
        return False
    return st.get("status") == "done" and (st.get("model_fingerprint") or {}).get("sha") == row.get("sha")

