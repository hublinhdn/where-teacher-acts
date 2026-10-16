import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import sys
import traceback
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from models.model_category_config import (
    super_large_backbones, large_backbones, medium_backbones,
    small_backbones, vit_backbones, swin_backbones, lightweight_vit_backbones,
    ghostnet_like_backbones, dw_moderate_backbones,
    fragile_medium_backbones, medium_experimental_backbones,
    dinov2_backbones, efficientnet_backbones, high_wd_backbones
)
from utils.dataset_loader import PillDataset, BalancedBatchSampler
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

# optional structural distillation terms of an earlier version of this code (stubbed in this release)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _j3_topo_losses import TopoKDLoss  # noqa: E402
# ============================================================================
# QAKD FORK of the inherited KD trainer (copied verbatim from an earlier version of this code).
# Additions are gated on two args and default OFF (default path == inherited trainer):
#   args.qakd (bool)          -> student wrapped with simulated INT8 (_qakd_fakequant); phase B
#                                starts at args.q_start_epoch: BN frozen+folded, fake-quant on,
#                                best-checkpoint tracking reset; saved state_dict is CLEAN.
#   args.no_kd (bool)         -> quantization-aware training WITHOUT a teacher (control arm);
#                                args.teacher is still used for the batch/accum configuration so
#                                the control shares the exact optimisation protocol of the KD arms.
# ============================================================================
from _qakd_fakequant import (prepare_fakequant, enable_quant, freeze_bn_fold,  # noqa: E402
                             clean_state_dict, quant_summary, clip_params, clip_penalty,
                             observer_state, clip_summary)


def _build_dinov2_llrd_param_groups(backbone, lr_backbone, decay_factor=0.75):
    """Layer-wise LR decay for DINOv2: the last block gets lr_backbone,
    each block further back is multiplied by decay_factor once more. Embedding/norm get the lowest LR.
    (Kept in sync with train_teacher_cv.py)"""
    if not hasattr(backbone, 'blocks'):
        return [{'params': list(backbone.parameters()), 'lr': lr_backbone}]

    n_blocks = len(backbone.blocks)
    param_groups = []
    assigned_ids = set()

    for i, block in enumerate(backbone.blocks):
        block_lr = lr_backbone * (decay_factor ** (n_blocks - i - 1))
        params = list(block.parameters())
        param_groups.append({'params': params, 'lr': block_lr})
        assigned_ids.update(id(p) for p in params)

    remaining = [p for p in backbone.parameters() if id(p) not in assigned_ids]
    if remaining:
        param_groups.append({'params': remaining, 'lr': lr_backbone * (decay_factor ** n_blocks)})

    return param_groups


def _recipe_lrs_at(base_lrs, total, warmup, at_epoch):
    """where-teacher-acts Part II: the learning rate each group had at the start of `at_epoch` (1-based)
    under this trainer's own schedule (linear warm-up from 1%, then cosine to 1e-6, one step per
    epoch), so a fine-tune can start exactly where an earlier version of this code switched quantization on."""
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in base_lrs]
    opt = torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, base_lrs)], lr=0.1)
    w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup)
    c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(total - warmup), eta_min=1e-6)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[w, c], milestones=[warmup])
    for _ in range(int(at_epoch) - 1):
        opt.step()
        sch.step()
    return [g['lr'] for g in opt.param_groups]


def train_kd_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    L_SCE = args.w_sce
    L_CSCE = args.w_csce
    L_TRIPLET = args.w_triplet
    L_CONTRASTIVE = args.w_cont
    ALPHA_KD = args.alpha 
    
    student_name = args.student.lower()
    teacher_name = args.teacher.lower()
    
    # Clean the names by removing the suffix, to match the explicit mapping
    clean_student_name = student_name.replace('_tv', '').replace('_timm', '')
    clean_teacher_name = teacher_name.replace('_tv', '').replace('_timm', '')
    
    
    # ==========================================
    # 🧩 0. CLASSIFY THE STUDENT FOR ITS SAFEGUARDS (EXPLICIT)
    # ==========================================
    is_swin_backbone = clean_student_name in swin_backbones
    is_vit_backbone = clean_student_name in vit_backbones

    is_lightweight_vit = clean_student_name in lightweight_vit_backbones
    is_small_backbones = clean_student_name in small_backbones
    is_massive = clean_student_name in super_large_backbones
    is_large_backbones = clean_student_name in large_backbones
    is_medium_backbones = clean_student_name in medium_backbones

    is_pure_vit = is_swin_backbone or is_vit_backbone
    is_fragile = is_small_backbones or is_lightweight_vit
    is_dinov2 = clean_student_name in dinov2_backbones   # explicit, no substring matching

    # Edge CNNs with DW convs: LR tiers (kept in sync with train_teacher_cv.py)
    is_ghostnet_like = clean_student_name in ghostnet_like_backbones
    is_dw_moderate = clean_student_name in dw_moderate_backbones

    # Medium fragile/experimental
    is_fragile_medium = clean_student_name in fragile_medium_backbones
    is_medium_experimental = clean_student_name in medium_experimental_backbones
    
    # ==========================================
    # 🧠 1. BUILD THE TEACHER FIRST (to get its size and dimension)
    # ==========================================
    _no_kd = bool(getattr(args, 'no_kd', False))
    if _no_kd:
        # QAKD control arm: no teacher in memory, no KD term. Batch/accum config below still
        # derives from clean_teacher_name so the control matches the KD arms' protocol exactly.
        teacher = None
        print(f"🚫 [QAKD] no_kd control: teacher [{args.teacher}] NOT loaded; KD term disabled (alpha ignored)")
    else:
        _anchor = getattr(args, 'anchor_backbone', None)
        teacher = PillRetrievalModel(num_classes=num_classes, backbone_type=(_anchor or args.teacher), pooling_type='gem').to(device)
        if args.teacher_weight != "pretrained_online":
            # Load the weights with strict=False to tolerate a mismatched MLP head (an anchor is the student's own
            # FP32 checkpoint, same architecture and classes, so it is loaded strictly)
            teacher.load_state_dict(torch.load(args.teacher_weight, map_location=device), strict=bool(_anchor))
        if _anchor:
            print(f"🧲 [qj] anchor: the frozen FP32 start ({_anchor}, {args.teacher_weight}) takes the teacher's "
                  f"place in the cosine term; batch layout from [{args.teacher}]")
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
    teacher_dim = 512
    if '_224' in clean_teacher_name:
        teacher_target_size = 224
    elif '_384' in clean_teacher_name:
        teacher_target_size = 384
    elif 'patch14' in clean_teacher_name:
        teacher_target_size = 392
    else:
        teacher_target_size = 384
        

    # ==========================================
    # ⚙️ 2. RESOURCES & RESOLUTION
    # ==========================================
    # Batch size / accumulation allocated flexibly from the teacher's size
    # In sync with train_teacher_cv.py: super_large uses accum=20 (eff_batch=160),
    # except DINOv2 and EfficientNet, which need accum=16 (low LR + LLRD = more updates needed)
    is_teacher_dinov2 = clean_teacher_name in dinov2_backbones
    is_teacher_efficientnet = clean_teacher_name in efficientnet_backbones

    if clean_teacher_name in super_large_backbones:
        n_classes_batch, n_samples = 4, 2
        if is_teacher_dinov2 or is_teacher_efficientnet:
            accumulation_steps = 16   # Effective Batch = 128
        else:
            accumulation_steps = 20   # Effective Batch = 160
    elif clean_teacher_name in large_backbones:
        n_classes_batch, n_samples = 8, 2
        accumulation_steps = 8        # Effective Batch = 128
    elif clean_teacher_name in medium_backbones:
        n_classes_batch, n_samples = 8, 2
        accumulation_steps = 8
    else:
        n_classes_batch, n_samples = 16, 2
        accumulation_steps = 4

    # Learning rate for the student
    if is_massive:
        lr_backbone, lr_head = 2e-5, 2e-4
    elif is_large_backbones:
        lr_backbone, lr_head = 3e-5, 3e-4
    elif is_medium_backbones:
        lr_backbone, lr_head = 3e-5, 3e-4
    elif is_lightweight_vit:
        lr_backbone, lr_head = 3e-5, 2e-4
    elif is_small_backbones:
        lr_backbone, lr_head = 5e-5, 5e-4
    else:
        lr_backbone, lr_head = 5e-5, 5e-4

    # 🩹 LR OVERRIDE for sensitive edge CNNs (kept in sync with train_teacher_cv.py)
    if is_ghostnet_like:
        lr_backbone, lr_head = 2e-5, 2e-4   # Prevents collapse
        print(f"🩹 Ghostnet-like Student [{clean_student_name}]: LR = 2e-5/2e-4")
    elif is_dw_moderate:
        lr_backbone, lr_head = 4e-5, 4e-4   # Intermediate
        print(f"🩹 DW-moderate Student [{clean_student_name}]: LR = 4e-5/4e-4")

    if is_pure_vit:
        # DINOv2 needs a much lower LR (its features are already very strong)
        multiplier = 0.1 if is_dinov2 else 0.3
        lr_backbone = lr_backbone * multiplier
        print(f"📉 Backbone LR lowered automatically for {'DINOv2' if is_dinov2 else 'Pure Transformer'} to: {lr_backbone}")

    # ==========================================
    # 📐 2.5. ARCFACE MARGIN + LOSS WEIGHTS + EPOCHS (kept in sync with train_teacher_cv.py)
    # ==========================================
    # ArcFace margin: small model → light margin (low capacity + large margin = underfit)
    if is_lightweight_vit:
        arcface_margin = 0.25
    elif is_small_backbones:
        arcface_margin = 0.20
    elif is_fragile_medium:
        arcface_margin = 0.25
    elif is_medium_experimental:
        arcface_margin = 0.30
    else:
        arcface_margin = 0.35

    # CSCE weight: lightweight ViTs need it lowered to balance metric against classification
    if is_lightweight_vit:
        L_CSCE = min(L_CSCE, 0.10)
        print(f"📉 Lightweight ViT Student: reduced L_CSCE={L_CSCE}")

    # More warmup + epochs for small models
    TOTAL_EPOCHS = args.epochs
    WARMUP_EPOCHS = 5
    if is_lightweight_vit:
        WARMUP_EPOCHS = 10
        TOTAL_EPOCHS = max(args.epochs, 80)
    elif is_small_backbones or is_fragile_medium:
        TOTAL_EPOCHS = max(args.epochs, 80)
    elif is_medium_experimental:
        TOTAL_EPOCHS = max(args.epochs, 70)
    # where-teacher-acts Part II: fine-tune from an FP32 checkpoint of this study for args.ft_epochs epochs,
    # no warm-up and no head freeze; the LR starts where the recipe's own schedule stood at epoch
    # args.ft_lr_from_epoch (41 of 80, where an earlier version of this code switched quantization on).
    _ft = bool(getattr(args, 'init_from', None))
    _recipe_total, _recipe_warmup = TOTAL_EPOCHS, WARMUP_EPOCHS
    if _ft:
        TOTAL_EPOCHS = int(args.ft_epochs)
        WARMUP_EPOCHS = 0
        print(f"🧷 [qj] fine-tune: {TOTAL_EPOCHS} epochs from {args.init_from}, LR from epoch "
              f"{int(getattr(args, 'ft_lr_from_epoch', 41))} of the recipe's {_recipe_total}-epoch schedule")
    # optional hard cap on epochs (calibration sweeps of an earlier version only). getattr-guarded
    # so the callers of earlier versions (bare Namespace) are unaffected. None/0 => no cap (full schedule).
    _cap = getattr(args, 'max_epochs_cap', None)
    if _cap:
        TOTAL_EPOCHS = min(TOTAL_EPOCHS, int(_cap))
        print(f"⏩ [cap] epochs capped at {TOTAL_EPOCHS} (calibration mode, NOT a final number)")
    print(f"📅 Schedule: TOTAL_EPOCHS={TOTAL_EPOCHS} | WARMUP={WARMUP_EPOCHS} | ArcFace margin={arcface_margin}")

    # DYNAMIC IMAGE SIZE: the DataLoader loads large images at the teacher's standard size.
    img_size = teacher_target_size
    # where-teacher-acts: one image size for every model of this study (args.img_size, _qj_common.IMG_SIZE);
    # teacher and student see the same images, so the teacher runs at that size too
    _fixed = getattr(args, 'img_size', None)
    if _fixed:
        img_size = int(_fixed)
        print(f"📏 [qj] image size fixed at {img_size}x{img_size} for student and teacher")
    resize_scale = int(img_size * 1.15)

    print(f"⚙️ KD Config: Teacher [{args.teacher}] -> Student [{args.student}] | Teacher size={teacher_target_size} | Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")

    # ==========================================
    # 🖼️ 3. DATA AUGMENTATION & LOADER
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((resize_scale, resize_scale)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PillDataset(df_train, transform=train_transform, transform_ref=val_transform), 
        batch_sampler=BalancedBatchSampler(df_train['sub_label_idx'].values, n_classes_batch, n_samples),
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref]).reset_index(drop=True), transform=val_transform), 
        batch_size=n_classes_batch * n_samples, shuffle=False, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 🧠 4. STUDENT & PROJECTOR SETUP
    # ==========================================
    # where-teacher-acts: reseed right before the student is built, so a run with the teacher off (A) and
    # one with it on (B) share the student's initialization, batch order and augmentation stream;
    # building the teacher above would otherwise consume random draws in B only.
    _rs = getattr(args, 'reseed_student', None)
    if _rs is not None:
        random.seed(int(_rs))
        np.random.seed(int(_rs))
        torch.manual_seed(int(_rs))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(_rs))
    student = PillRetrievalModel(
        num_classes=num_classes,
        backbone_type=args.student,
        pooling_type='gem',
        m=arcface_margin   # 🆕 margin set per student group
    ).to(device)

    # where-teacher-acts Part II: the start checkpoint (A or B), loaded before fake quantization wraps it
    if _ft:
        student.load_state_dict(torch.load(args.init_from, map_location=device), strict=True)
        # a student deployed in fused form (RepVGG, _qj_common.FUSED_STUDENTS) is fused here, after its
        # FP32 checkpoint is loaded and before fake quantization wraps it, so the quantization simulated
        # in training is the one applied at deployment
        if getattr(args, 'fuse_branches', False):
            from models.reparam import fuse_repvgg_branches
            _nf = fuse_repvgg_branches(student)
            if _nf == 0:
                raise RuntimeError("fuse_branches is set but the student has no multi-branch RepVGG block")
            student.to(device)
            print(f"🔗 [qj] fused {_nf} RepVGG blocks into single convolutions before quantization")

    # --- QAKD fork: wrap the student with simulated INT8 (inactive until phase B) ---
    _qakd = bool(getattr(args, 'qakd', False))
    _q_start = max(1, int(getattr(args, 'q_start_epoch', 1) or 1))
    _q_clip = _qakd and bool(getattr(args, 'q_clip', False))
    _lam_clip = float(getattr(args, 'q_clip_penalty', 0.0) or 0.0)
    if _qakd:
        _qinfo = prepare_fakequant(student, w_bits=8, a_bits=8,
                                   quant_output=bool(getattr(args, 'q_quant_output', True)),
                                   learnable_clip=_q_clip,
                                   warmup_steps=int(getattr(args, 'q_clip_warmup', 20)))
        print(f"🧊 [QAKD] fake-quant prepared on student: {_qinfo} | phase B (approximate fold, "
              f"fake-quant ON, best-tracking reset) starts at epoch {_q_start}"
              + (f" | learnable clip: lr {getattr(args, 'lr_clip', 5e-3)}, warm-up "
                 f"{getattr(args, 'q_clip_warmup', 20)} batches, penalty {_lam_clip}" if _q_clip else ""))

    projector = None
    if teacher_dim != 512:
        projector = nn.Linear(512, teacher_dim, bias=False).to(device)
        nn.init.xavier_uniform_(projector.weight)
        print(f"🔗 KD projector created for the student: 512 -> {teacher_dim}")

    # ==========================================
    # 🎯 5. LOSS & OPTIMIZER 
    # ==========================================
    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    
    criterion_mse = nn.MSELoss()
    criterion_cosine = nn.CosineEmbeddingLoss()
    criterion_kld = nn.KLDivLoss(reduction='batchmean')

    # optional structural criterion, built once per fold (None for the pointwise kd_types)
    # method default 'mst_h0' (direct differentiable H0 alignment, no pretrained approximator).
    # Read every knob of this fork via getattr so the callers of earlier versions (bare Namespace) are unaffected.
    criterion_topo = None
    if getattr(args, 'kd_type', 'cosine') in ('topo', 'topo_only', 'hybrid_topo', 'rkd'):
        # Guard: a structural kd_type with lambda_topo <= 0 would train without its structural signal:
        # fail before spending GPU time.
        assert float(getattr(args, 'lambda_topo', 0.0)) > 0.0, \
            f"kd_type={args.kd_type} requires --lambda_topo > 0 (got {getattr(args,'lambda_topo',0.0)})"
        # kd_type='rkd' is the relational KD of Park et al. (2019), on the same schedule as the other
        # structural terms.
        _tm = 'rkd' if args.kd_type == 'rkd' else getattr(args, 'topo_method', 'mst_h0')
        criterion_topo = TopoKDLoss(method=_tm).to(device)
    _topo_nonfinite = 0  # counter: how often L_topo was skipped for being NaN/inf (logged at fold end)
    _topo_sum, _topo_max, _topo_n = 0.0, 0.0, 0  # raw L_topo magnitude stats (for λ calibration logging)

    # QAKD fork: learnable-clip parameters (theta of every observer) get their OWN group below;
    # they must not ride the backbone/head LR (5e-5 would never move a log-range in 40 epochs).
    _is_obs = lambda n: ('.qin.' in n) or ('.qout.' in n)  # noqa: E731
    head_params = [p for n, p in student.named_parameters() if 'features' not in n and not _is_obs(n)]
    if projector is not None:
        head_params.extend(list(projector.parameters()))

    # Dynamic weight decay (DINOv2 needs a light wd of 0.05 to preserve its pretrained features)
    wd_value = 0.1 if (is_massive or is_large_backbones or clean_student_name in high_wd_backbones) else 5e-2
    if is_dinov2:
        wd_value = 0.05

    if is_dinov2:
        # LLRD for a DINOv2 student: small/base use decay 0.85, large uses 0.75
        llrd_decay = 0.85 if (is_fragile_medium or is_medium_experimental or is_medium_backbones) else 0.75
        llrd_groups = _build_dinov2_llrd_param_groups(student.features, lr_backbone, decay_factor=llrd_decay)
        optimizer = torch.optim.AdamW(
            llrd_groups + [{'params': head_params, 'lr': lr_head}],
            weight_decay=wd_value
        )
        n_blocks_est = max(len(llrd_groups) - 1, 1)
        print(f"🔬 DINOv2 Student LLRD: {len(llrd_groups)} groups | decay={llrd_decay} | lr range [{lr_backbone * llrd_decay**n_blocks_est:.2e} ~ {lr_backbone:.2e}]")
    else:
        backbone_params = [p for n, p in student.named_parameters() if 'features' in n and not _is_obs(n)]
        _groups = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ]
        _cp = clip_params(student) if _q_clip else []
        if _cp:
            _groups.append({'params': _cp, 'lr': float(getattr(args, 'lr_clip', 5e-3)), 'weight_decay': 0.0})
            print(f"🧊 [QAKD] {len(_cp)} learnable-clip parameters in their own group (lr {getattr(args, 'lr_clip', 5e-3)}, wd 0)")
        optimizer = torch.optim.AdamW(_groups, weight_decay=wd_value)

    # Do NOT re-declare TOTAL_EPOCHS/WARMUP_EPOCHS: they are set in section 2.5 (by student type)
    if _ft:
        _start = _recipe_lrs_at([g['lr'] for g in optimizer.param_groups], _recipe_total, _recipe_warmup,
                                int(getattr(args, 'ft_lr_from_epoch', 41)))
        for g, lr in zip(optimizer.param_groups, _start):
            g['lr'] = lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
        print(f"🧷 [qj] fine-tune LR per group at start: {[f'{lr:.2e}' for lr in _start]} -> cosine to 1e-6")
    else:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    scaler = torch.amp.GradScaler('cuda')
    best_val_map, r1 = 0.0, 0.0
    best_metrics = {
        'Rank-1': 0.0, 'mAP': 0.0, 'mAP(Cons)': 0.0, 
        'mAP(Ref)': 0.0, 'mAP(Delta)': 0.0, 'Rank-1(Ref)': 0.0
    }

    # optional structural term: kept off until the student has passed the LR warm-up and any
    # backbone freeze, then ramped in (not used in this study)
    _freeze_until = 0
    if ((is_fragile or is_massive) and not is_pure_vit) or is_dinov2:
        if is_dinov2 and (is_massive or is_large_backbones):
            _freeze_until = 10
        elif is_massive or is_dinov2:
            _freeze_until = 4
        else:
            _freeze_until = 2
    _topo_start = getattr(args, 'topo_start_epoch', None)
    _topo_start = int(_topo_start) if _topo_start is not None else max(WARMUP_EPOCHS, _freeze_until)
    if criterion_topo is not None:
        print(f"🌀 [topo] schedule: OFF until epoch {_topo_start} (LR-warmup={WARMUP_EPOCHS}, freeze={_freeze_until}), "
              f"then ramp 0->λ={getattr(args, 'lambda_topo', 0.0)} over {max(1, int(getattr(args, 'topo_warmup_epochs', 5)))} epochs.")

    # ==========================================
    # 🚀 6. DISTILLATION TRAINING LOOP
    # ==========================================
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Head warm-up: freeze the backbone for small/very large models + DINOv2
        # Do NOT freeze ordinary pure ViTs (gradient shock), except DINOv2, which has its own logic
        # where-teacher-acts: no head warm-up when fine-tuning an already trained checkpoint
        if not _ft and (((is_fragile or is_massive) and not is_pure_vit) or is_dinov2):
            # DINOv2 large/base: freeze 10 epochs so the head settles first
            if is_dinov2 and (is_massive or is_large_backbones):
                freeze_epochs = 10
            elif is_massive or is_dinov2:
                freeze_epochs = 4
            else:
                freeze_epochs = 2
            if epoch <= freeze_epochs:
                for param in student.features.parameters():
                    param.requires_grad = False
            elif epoch == freeze_epochs + 1:
                for param in student.features.parameters():
                    param.requires_grad = True

        # --- QAKD fork: phase B switch. From here every forward (train AND val) is simulated INT8;
        # BN statistics are frozen so conv->bn folding is exact; the best-checkpoint tracker is
        # reset so the saved model is a quantization-aware one, not the FP32 model of phase A.
        if _qakd and epoch == _q_start:
            enable_quant(student, enabled=True)          # approximate fold: BN still batch-normalizes
            best_val_map, r1 = 0.0, 0.0
            print(f"🧊 [QAKD] epoch {epoch}: phase B ON (approximate fold) -> {quant_summary(student)} | best-tracking reset")
        # final epochs: freeze BN statistics and fold exactly (the deployed arithmetic)
        _q_freeze_at = TOTAL_EPOCHS - int(getattr(args, 'q_freeze_last', 5)) + 1
        if _qakd and epoch == max(_q_freeze_at, _q_start + 1):
            freeze_bn_fold(student)
            print(f"🧊 [QAKD] epoch {epoch}: BN statistics FROZEN, exact fold for the last "
                  f"{TOTAL_EPOCHS - epoch + 1} epochs -> {quant_summary(student)}")

        student.train()
        if projector is not None:
            projector.train()

        pbar = tqdm(train_loader, desc=f"KD Fold {f_idx} | Ep {epoch} | Best {best_val_map:.4f}")
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()

            with torch.amp.autocast('cuda'):
                
                # ----------------------------------------------------
                # 🔵 TEACHER FEATURES (the original large images)
                # ----------------------------------------------------
                # The teacher is always a fine-tuned PillRetrievalModel → it has both logits & emb
                with torch.no_grad():
                    if teacher is not None:
                        t_logits_sce, _, t_norm_emb = teacher(imgs, labels=sub_labels)
                    else:
                        t_logits_sce, t_norm_emb = None, None   # QAKD no_kd control
                
                # ----------------------------------------------------
                # 🟢 STUDENT FEATURES (size handling for edge models)
                # ----------------------------------------------------
                if is_fragile or '_224' in clean_student_name:
                    if imgs.shape[-1] != 224:
                        s_imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                    else:
                        s_imgs = imgs
                elif '_384' in clean_student_name and imgs.shape[-1] != 384:
                    s_imgs = F.interpolate(imgs, size=(384, 384), mode='bilinear', align_corners=False)
                else:
                    s_imgs = imgs

                # Feed s_imgs to the student
                s_logits_sce, s_logits_csce, s_norm_emb = student(s_imgs, labels=sub_labels)

                # ----------------------------------------------------
                # 🟠 TASK LOSS (the student's own learning)
                # ----------------------------------------------------
                loss_sce = criterion_sce(s_logits_sce, sub_labels)
                loss_csce = criterion_sce(s_logits_csce, sub_labels)
                
                if L_TRIPLET > 0 or L_CONTRASTIVE > 0:
                    hard_pairs = miner(s_norm_emb, sub_labels)
                    loss_triplet = criterion_triplet(s_norm_emb, sub_labels, hard_pairs)
                    loss_contrastive = criterion_contrastive(s_norm_emb, sub_labels)
                else:
                    loss_triplet = torch.tensor(0.0, device=device)
                    loss_contrastive = torch.tensor(0.0, device=device)

                student_loss = L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive

                # ----------------------------------------------------
                # 🟣 DISTILLATION LOSS (learning from the teacher)
                # ----------------------------------------------------
                kd_loss = torch.tensor(0.0, device=device)
                
                s_kd_feats = projector(s_norm_emb) if projector is not None else s_norm_emb
                s_kd_feats = F.normalize(s_kd_feats, p=2, dim=1)
                
                if not _no_kd and args.kd_type in ['mse', 'hybrid', 'hybrid_topo']:
                    kd_loss += criterion_mse(s_kd_feats, t_norm_emb.detach())

                if not _no_kd and args.kd_type in ['cosine', 'hybrid', 'topo', 'hybrid_topo']:
                    target_cos = torch.ones(s_kd_feats.size(0), device=device)
                    kd_loss += criterion_cosine(s_kd_feats, t_norm_emb.detach(), target_cos)

                if not _no_kd and args.kd_type in ['kl', 'hybrid', 'hybrid_topo']:
                    T = args.temperature
                    s_log_probs = F.log_softmax(s_logits_sce.float() / T, dim=1)
                    t_probs = F.softmax(t_logits_sce.float() / T, dim=1).detach()
                    kl_val = criterion_kld(s_log_probs, t_probs) * (T * T)
                    kd_loss += kl_val if args.kd_type == 'kl' else (kl_val * 0.1)

                # split KD into pointwise vs structural for the progress bar (see set_postfix below)
                _disp_kd_pt = float(kd_loss)     # pointwise KD so far (mse/cosine/kl)
                _disp_kd_topo = 0.0              # topo contribution (scaled), filled in below if active

                # optional structural distillation term, added on top of the pointwise KD for the structural
                # kd_types only (not used in this study, which uses kd_type 'cosine')
                # AMP: persistence/cdist/MST are fp16-fragile -> compute with autocast OFF, in fp32.
                if (not _no_kd) and args.kd_type in ['topo', 'topo_only', 'hybrid_topo', 'rkd'] and criterion_topo is not None:
                    _lam = float(getattr(args, 'lambda_topo', 0.0))
                    _ramp = max(1, int(getattr(args, 'topo_warmup_epochs', 5)))
                    _scale = _lam * min(1.0, max(0.0, (epoch - _topo_start) / _ramp))
                    if _scale > 0.0:
                        with torch.amp.autocast('cuda', enabled=False):
                            topo_val = criterion_topo(s_kd_feats.float(), t_norm_emb.detach().float())
                        if torch.isfinite(topo_val):
                            kd_loss = kd_loss + _scale * topo_val.to(kd_loss.dtype)
                            _disp_kd_topo = float(_scale * topo_val)
                            _tv = float(topo_val); _topo_sum += _tv; _topo_n += 1
                            if _tv > _topo_max:
                                _topo_max = _tv
                        else:
                            _topo_nonfinite += 1

                loss = (student_loss + ALPHA_KD * kd_loss) / accumulation_steps
                if _q_clip and _lam_clip > 0.0 and epoch >= _q_start:
                    loss = loss + (_lam_clip * clip_penalty(student)) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                clip_val = 3.0 if (is_fragile or is_pure_vit) else 5.0
                
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=clip_val)
                if projector is not None:
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=clip_val)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # progress: task loss + KD split into pointwise vs structural contribution (ALPHA_KD scales both in total)
            _post = {'S_Loss': f"{student_loss.item():.2f}", 'KD_pt': f"{_disp_kd_pt:.4f}"}
            if criterion_topo is not None:
                _post['KD_topo'] = f"{_disp_kd_topo:.4f}"
            pbar.set_postfix(_post)

        scheduler.step()

        if (epoch % 10 == 0) or (epoch > TOTAL_EPOCHS - 5):
            # Additive guard of an earlier version: default True keeps the original behavior;
            # OGYEIv2 passes both_side=False (single-side selection signal)
            val_metrics = evaluate_retrieval(student, val_loader, device, flag_both_side=getattr(args, 'both_side', True), flag_eval_delta=True)
            val_map = val_metrics.get('mAP(Cons)') or val_metrics['mAP']
            val_r1 = val_metrics.get('Rank-1(Cons)') or val_metrics['Rank-1']
            
            val_map_ref = val_metrics.get('mAP(Ref)', 0.0)
            val_map_delta = val_metrics.get('mAP(Delta)', 0.0)

            print(f"📊 Epoch {epoch} mAP(Cons): {val_map:.4f} | mAP(Ref): {val_map_ref:.4f} | Delta: {val_map_delta:.4f}")
            _curve = getattr(args, 'val_curve', None)   # where-teacher-acts: the caller records the curve
            if _curve is not None:
                _curve.append({'epoch': epoch, 'mAP': round(float(val_map), 6), 'rank1': round(float(val_r1), 6)})
            
            if val_map > best_val_map:
                best_val_map = val_map
                r1 = val_r1
                best_metrics = val_metrics
                
                # where-teacher-acts: the only write path is the run's claimed directory
                # (pipeline/_qj_io.RunClaim, one directory per arm/student/teacher/epochs/seed).
                # Deliberately no fallback: the inherited default was weights/phase3_kd, which in
                # this repository is a symlink into the checkpoints of an earlier study.
                _claim = getattr(args, 'run_claim', None)
                if _claim is None:
                    raise RuntimeError("where-teacher-acts: train_kd_fold needs args.run_claim "
                                       "(pipeline/_qj_io.RunClaim); refusing to fall back to "
                                       "weights/phase3_kd")
                # state dict saved CLEAN (original PillRetrievalModel keys) for strict loading and
                # ONNX export; learned ranges travel in a sidecar next to it
                _sd = clean_state_dict(student) if _qakd else student.state_dict()
                _claim.save(_sd, "model.pth")
                if _q_clip:
                    _claim.save(observer_state(student), "model.pth.clip.pt")
                    print(f"🧊 [QAKD] clip ranges: {clip_summary(student)}")

            # Additive guard of an earlier version: ceiling early-stop (default None => original behavior
            # unchanged). Stop once the student hits the mAP ceiling: no headroom for KD gain/harm to show.
            _esm = getattr(args, 'early_stop_map', None)
            if _esm is not None and val_map >= _esm:
                print(f"⛔ [early-stop] {teacher_name}->{student_name}: val mAP {val_map:.4f} >= {_esm} (ceiling) @ epoch {epoch}, stopping")
                break

    if criterion_topo is not None and _topo_n > 0:
        _mean = _topo_sum / _topo_n
        # Persist the structural-loss magnitude (optional terms, not used in this study)
        # (the selection-via-dynamics analysis of an earlier study) without re-parsing logs. Additive; best_metrics is a dict.
        if isinstance(best_metrics, dict):
            best_metrics['topo_mag_mean'] = round(_mean, 6)
            best_metrics['topo_mag_max'] = round(_topo_max, 6)
        _lam = float(getattr(args, 'lambda_topo', 0.0))
        print(f"📐 [topo] L_topo magnitude over {_topo_n} batches: mean={_mean:.5f} max={_topo_max:.5f} | "
              f"λ={_lam} -> mean topo contribution to kd_loss ≈ {_lam * _mean:.5f} "
              f"(cf. cosine term ~1.0; if «mean topo contribution» << 0.1, λ is too small).")
        if _topo_nonfinite > 0:
            print(f"⚠️ [topo] non-finite & skipped on {_topo_nonfinite} batch(es); check batch size / stability.")
    elif criterion_topo is not None and _topo_n == 0:
        # Guard: a structural kd_type whose term never fired would
        # save as a valid baseline but actually be pure task-loss. Warn loudly so it isn't trusted as KD.
        print(f"⚠️ [topo] kd_type={getattr(args,'kd_type','?')} but the structural term fired on 0 batches "
              f"(start epoch {_topo_start} >= TOTAL_EPOCHS {TOTAL_EPOCHS}?): this run is NOT a valid KD result.")
    return best_val_map, r1, best_metrics

def main():
    # where-teacher-acts: runs start from pipeline/_qj_run.py, which builds the split (_qj_split), seeds
    # the run and owns its directory. This entry point of the inherited trainer would train on its fold-0 split.
    raise SystemExit("where-teacher-acts: start runs with pipeline/_qj_run.py --role kd --model <student> --teacher <teacher>")
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True)
    parser.add_argument('--teacher_weight', type=str, required=True)
    parser.add_argument('--student', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    
    parser.add_argument('--kd_type', type=str, default='cosine',
                        choices=['mse', 'cosine', 'kl', 'hybrid', 'rkd', 'topo', 'topo_only', 'hybrid_topo'])
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--temperature', type=float, default=4.0)
    # optional structural distillation terms (not used in this study; stubbed in this release)
    parser.add_argument('--lambda_topo', type=float, default=0.0,
                        help='weight of the optional structural term (0 = off; not part of this release)')
    parser.add_argument('--topo_method', type=str, default='mst_h0',
                        choices=['mst_h0', 'mst_h0_aligned', 'rkd', 'pi'],
                        help="the optional structural term (not part of this release)")
    parser.add_argument('--topo_warmup_epochs', type=int, default=0,
                        help='ramp of the optional structural term over the first N epochs (0 = no ramp)')
    
    parser.add_argument('--w_sce', type=float, default=1.0)
    parser.add_argument('--w_csce', type=float, default=0.2)
    parser.add_argument('--w_triplet', type=float, default=1.0)
    parser.add_argument('--w_cont', type=float, default=1.0)
    
    parser.add_argument('--pipeline_name', type=str, default='KD_Experiment')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
    else: 
        device = torch.device('cpu')

    os.makedirs(os.path.join('reports_qj', 'train_logs'), exist_ok=True)
    report_file_path = os.path.join('reports_qj', 'train_logs', f"{args.pipeline_name}_Summary.txt")
    loss_str_compact = f"{args.w_sce}/{args.w_csce}/{args.w_triplet}/{args.w_cont}"

    if not os.path.exists(report_file_path):
        with open(report_file_path, "w") as f:
            f.write(f"Knowledge Distillation Batch Experiment\n")
            f.write("="*135 + "\n")
            f.write(f"{'Teacher'.ljust(35)} | {'Student'.ljust(20)} | {'Alpha'.ljust(5)} | {'Temp'.ljust(4)} | {'Type'.ljust(7)} | {'Loss(S/CS/T/C)'.ljust(15)} | {'mAP(Cons)'.ljust(9)} | {'mAP(Ref)'.ljust(8)} | {'Delta'.ljust(6)}\n")
            f.write("-" * 135 + "\n")
    
    try:
        df_all = load_epill_full_data()
        total_sub_classes = df_all['sub_label_idx'].nunique()

        fold = 0
        df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
        df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
        
        cond_train = (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
        train_labels = df_all[cond_train]['label_idx'].unique()
        cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
        df_train = df_all[cond_train | cond_ref_train].reset_index(drop=True)
        
        map, r1, best_metrics = train_kd_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
        map_ref = best_metrics.get('mAP(Ref)', 0.0)
        map_delta = best_metrics.get('mAP(Delta)', 0.0)

        with open(report_file_path, "a") as f:
            f.write(f"{args.teacher.ljust(35)} | {args.student.ljust(20)} | {args.alpha:<5.1f} | {args.temperature:<4.1f} | {args.kd_type.ljust(7)} | {loss_str_compact.ljust(15)} | {map:.4f}    | {map_ref:.4f}   | {map_delta:.4f}\n")
            
    except Exception as e:
        with open(os.path.join('reports_qj', 'train_logs', 'KD_Crash_Traceback.txt'), 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"CRASH LOG: Teacher [{args.teacher}] -> Student [{args.student}] | Loss: {loss_str_compact}\n")
            f.write(traceback.format_exc())
        raise e 

if __name__ == '__main__':
    main()