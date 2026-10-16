import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import sys
import numpy as np
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import losses, miners
import torchvision.transforms.functional as TF
import traceback

# ============================================================================
# where-teacher-acts copy of the inherited baseline trainer (training/train_teacher_cv.py of an
# earlier version of this code), used for this study's teachers and undistilled students A.
# The training arithmetic and the validation (fold-0 consumer queries against every reference,
# consumer-side both-sides mAP every 10 epochs and in the last 5, checkpoint kept whenever it
# improves) are unchanged. Changes: the checkpoint is written only through the run's RunClaim
# (pipeline/_qj_io.py), never to weights/phase3; each validation result is appended to
# args.val_curve when the caller provides one; args.max_epochs_cap caps the schedule for smoke
# tests only. The data split comes from pipeline/_qj_split.py; runs are started by _qj_run.py.
# ============================================================================

# Import local modules
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

folder_weight = None  # where-teacher-acts: no default write location; see the save block

def _build_dinov2_llrd_param_groups(backbone, lr_backbone, decay_factor=0.75):
    """Layer-wise LR decay for DINOv2: the last block gets lr_backbone,
    each block further back is multiplied by decay_factor once more. Embedding/norm get the lowest LR.
    Decay 0.75 with 24 blocks → the first block gets only ~0.001 × lr_backbone."""
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

    # Embedding + norm + cls_token + pos_embed → lowest LR
    remaining = [p for p in backbone.parameters() if id(p) not in assigned_ids]
    if remaining:
        param_groups.append({'params': remaining, 'lr': lr_backbone * (decay_factor ** n_blocks)})

    return param_groups

def train_one_fold(args, f_idx, num_classes, df_train, df_val, df_ref, device):
    USE_SHAPE_LOSS, L_SHAPE = False, 1.0

    L_SCE = args.w_sce
    L_CSCE = args.w_csce
    L_TRIPLET = args.w_triplet
    L_CONTRASTIVE = args.w_cont

    TOTAL_EPOCHS = args.epochs
    WARMUP_EPOCHS = 5

    # =====================================================
    # 0. EXPLICIT ARCHITECTURE GROUPS (EXPLICIT MAPPING)
    # =====================================================
    backbone_name = args.backbone.lower()
    clean_name = backbone_name.replace('_tv', '').replace('_timm', '')

    is_swin_backbone = clean_name in swin_backbones
    is_vit_backbone = clean_name in vit_backbones

    is_lightweight_vit = clean_name in lightweight_vit_backbones
    is_small_backbones = clean_name in small_backbones
    is_massive = clean_name in super_large_backbones
    is_large_backbones = clean_name in large_backbones
    is_medium_backbones = clean_name in medium_backbones

    is_pure_vit = is_swin_backbone or is_vit_backbone
    is_fragile = is_small_backbones or is_lightweight_vit
    is_dinov2 = clean_name in dinov2_backbones        # explicit, no substring matching
    is_efficientnet = clean_name in efficientnet_backbones

    # Edge CNNs with heavy depth-wise convs: explicit groups by LR sensitivity
    # (validated in retrain v2, see model_category_config.py)
    is_ghostnet_like = clean_name in ghostnet_like_backbones
    is_dw_moderate = clean_name in dw_moderate_backbones

    # Medium fragile/experimental: applies the edge-CNN insight to the weak medium models
    is_fragile_medium = clean_name in fragile_medium_backbones
    is_medium_experimental = clean_name in medium_experimental_backbones

    # =====================================================
    # ⚙️ 1. DYNAMIC RESOURCE ALLOCATION (Batch Size & LR)
    # =====================================================
    if is_massive:
        n_classes_batch, n_samples = 4, 2  # Physical Batch = 8
        # DINOv2 super_large: accum rolled back to 16 because LLRD + low LR + high accum
        # = too few updates, gradient not strong enough (validated: 0.72 → 0.69 with accum=20)
        # The other models keep accum=20 because they are not affected (backbone LR 5-10× higher)
        if is_dinov2:
            accumulation_steps = 16   # rollback for DINOv2
        elif is_efficientnet:
            accumulation_steps = 16   # 🆕 rollback for EfficientNet (hypothesis: it needs more updates)
        else:
            accumulation_steps = 20
        lr_backbone, lr_head = 2e-5, 2e-4
    elif is_large_backbones:
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16
        accumulation_steps = 8             # Effective Batch = 128
        lr_backbone, lr_head = 3e-5, 3e-4
    elif is_medium_backbones:
        n_classes_batch, n_samples = 8, 2  # Physical Batch = 16
        accumulation_steps = 8             # Effective Batch = 128
        lr_backbone, lr_head = 3e-5, 3e-4
    elif is_lightweight_vit:
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32
        accumulation_steps = 4             # Effective Batch = 128
        lr_backbone, lr_head = 3e-5, 2e-4  # v3: back to 2e-4 (v2 used 1e-4 → mAP dropped for 5/7 models)
    elif is_small_backbones:
        n_classes_batch, n_samples = 16, 2 # Physical Batch = 32
        accumulation_steps = 4             # Effective Batch = 128
        lr_backbone, lr_head = 5e-5, 5e-4
    else:
        raise Exception(f"Error: please add [{clean_name}] to a group (super_large / large / medium / small / lightweight_vit) in model_category_config.py to continue.")

    # 🩹 LR OVERRIDE for sensitive edge CNNs: explicit tiers by sensitivity
    if is_ghostnet_like:
        lr_backbone, lr_head = 2e-5, 2e-4   # Prevents collapse (shown: 0.21 → 0.45)
        print(f"🩹 Ghostnet-like [{clean_name}]: LR = 2e-5/2e-4 (prevents collapse)")
    elif is_dw_moderate:
        lr_backbone, lr_head = 4e-5, 4e-4   # Intermediate: 2e-5 is too low → underfit, 5e-5 works but can do better
        print(f"🩹 DW-moderate [{clean_name}]: LR = 4e-5/4e-4 (balanced)")

    # =====================================================
    # 📐 1.5. LOSS WEIGHTS & ARCFACE MARGIN PER MODEL
    # =====================================================
    # ArcFace margin: small model → light margin (low capacity + large margin = underfit)
    # Lightweight ViT v3: 0.25 (v2 used 0.15, too light → embeddings not separated enough → 5/7 models dropped)
    if is_lightweight_vit:
        arcface_margin = 0.25
    elif is_small_backbones:
        arcface_margin = 0.20
    elif is_fragile_medium:
        arcface_margin = 0.25   # Weak medium models (swin_tiny/small, vit_small_dinov2): light margin
    elif is_medium_experimental:
        arcface_margin = 0.30   # Between 0.25 and 0.35
    else:
        arcface_margin = 0.35

    # CSCE weight: lightweight ViT v3 = 0.10 (v2 used 0.05, too low → the classification signal was lost)
    if is_lightweight_vit:
        L_CSCE = min(L_CSCE, 0.10)
        print(f"📉 Lightweight ViT: reduced L_CSCE={L_CSCE} (balance between metric and classification)")

    # More warmup + epochs for small models (they learn more slowly and need more time to converge)
    if is_lightweight_vit:
        WARMUP_EPOCHS = 10
        TOTAL_EPOCHS = max(args.epochs, 80)
    elif is_small_backbones or is_fragile_medium:
        TOTAL_EPOCHS = max(args.epochs, 80)   # Weak medium models also need 80 epochs
    elif is_medium_experimental:
        TOTAL_EPOCHS = max(args.epochs, 70)   # Intermediate, saves GPU time
    # where-teacher-acts: optional hard cap for smoke tests only (None => the full inherited schedule)
    _cap = getattr(args, 'max_epochs_cap', None)
    if _cap:
        TOTAL_EPOCHS = min(TOTAL_EPOCHS, int(_cap))
        print(f"⏩ [qj] epochs capped at {TOTAL_EPOCHS} (smoke test, NOT a result)")
    print(f"📅 Schedule: TOTAL_EPOCHS={TOTAL_EPOCHS} | WARMUP={WARMUP_EPOCHS} | ArcFace margin={arcface_margin}")

    # DINOv2-Large @ 392×392 with 12GB VRAM: KEEP batch=8 (4×2) + accum=16
    # The small batch is a forced tradeoff; LLRD + low drop_path + a long freeze compensate for it
    # With >=24GB VRAM, it can be overridden to 8×2=16 and accum=8 to reduce miner noise

    # Separate LR adjustment for pure Transformers, to avoid breaking the pretrained weights
    if is_pure_vit:
        # DINOv2 needs a much lower backbone LR: its features are already very strong, fine-tune lightly
        multiplier = 0.1 if is_dinov2 else 0.3
        lr_backbone = lr_backbone * multiplier
        print(f"📉 Backbone LR lowered automatically for {'DINOv2' if is_dinov2 else 'Pure Transformer'} to: {lr_backbone}")

    # =====================================================
    # 🛡️ 2. AUTOMATIC IMAGE SIZE ADJUSTMENT
    # =====================================================
    img_size = 384
    
    # Priority 1: read the size hard-coded in the model name (avoids tensor shape errors in Swin/MaxViT/DeiT)
    if '_224' in clean_name:
        img_size = 224
        print(f"📏 Image size forced to {img_size}x{img_size} by the size hard-coded in the network name.")
    elif '_384' in clean_name:
        img_size = 384
        print(f"📏 Image size forced to {img_size}x{img_size} by the size hard-coded in the network name.")
    elif is_lightweight_vit or is_small_backbones:
        img_size = 224 
        print(f"🧩 Edge/ultra-light architecture detected: image size fixed at {img_size}x{img_size}")
    elif 'patch14' in clean_name:
        img_size = 392 
        print(f"⚠️ ViT Patch14 (DINO/EVA) detected: image size raised automatically to {img_size}x{img_size}")
    elif 'patch16' in clean_name:
        img_size = 384
        
    resize_scale = int(img_size * 1.15) 
    # where-teacher-acts: one image size for every model of this study (args.img_size, _qj_common.IMG_SIZE)
    _fixed = getattr(args, 'img_size', None)
    if _fixed:
        img_size = int(_fixed)
        resize_scale = int(img_size * 1.15)
        print(f"📏 [qj] image size fixed at {img_size}x{img_size} for every model of this study")
    print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size} | Batch={n_classes_batch*n_samples} | Accum={accumulation_steps}")

    # =====================================================
    # 🖼️ 3. DUAL TRANSFORM & DATALOADER
    # =====================================================
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
    
    train_transform_ref = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PillDataset(df_train, transform=train_transform, transform_ref=train_transform_ref), 
        batch_sampler=BalancedBatchSampler(df_train['sub_label_idx'].values, n_classes_batch, n_samples),
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref]).reset_index(drop=True), transform=val_transform), 
        batch_size=n_classes_batch * n_samples, shuffle=False, num_workers=4, pin_memory=True
    )

    # =====================================================
    # 🧠 4. MODEL & OPTIMIZER SETUP
    # =====================================================
    model = PillRetrievalModel(
        num_classes=num_classes,
        backbone_type=args.backbone,
        pooling_type=args.pooling,
        m=arcface_margin   # 🆕 margin set per backbone group
    ).to(device)

    criterion_sce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    criterion_contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    criterion_shape = nn.MSELoss()

    head_params = [p for n, p in model.named_parameters() if 'features' not in n]

    wd_value = 0.1 if (is_massive or is_large_backbones or clean_name in high_wd_backbones) else 5e-2
    # DINOv2 fine-tuning is sensitive to a high weight decay: lowered to 0.05 for stability
    if is_dinov2:
        wd_value = 0.05

    if is_dinov2:
        # Layer-wise LR decay: the last block gets the full lr_backbone, the first blocks get very little
        # DINOv2-small/base: decay 0.85 (keeps more LR in the last layers; a smaller capacity needs to learn more)
        # DINOv2-large: decay 0.75 (standard for 24 blocks)
        llrd_decay = 0.85 if (is_fragile_medium or is_medium_experimental) else 0.75
        llrd_groups = _build_dinov2_llrd_param_groups(model.features, lr_backbone, decay_factor=llrd_decay)
        optimizer = torch.optim.AdamW(
            llrd_groups + [{'params': head_params, 'lr': lr_head}],
            weight_decay=wd_value
        )
        n_blocks_est = max(len(llrd_groups) - 1, 1)
        print(f"🔬 DINOv2 LLRD: {len(llrd_groups)} layer groups | decay={llrd_decay} | lr range [{lr_backbone * llrd_decay**n_blocks_est:.2e} ~ {lr_backbone:.2e}]")
    else:
        backbone_params = [p for n, p in model.named_parameters() if 'features' in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ], weight_decay=wd_value)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(TOTAL_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_map, r1 = 0.0, 0.0
    best_metrics = {
        'Rank-1': 0.0, 'mAP': 0.0, 'mAP(Cons)': 0.0, 
        'mAP(Ref)': 0.0, 'mAP(Delta)': 0.0, 'Rank-1(Ref)': 0.0
    }

    # =====================================================
    # 🏃 5. TRAINING LOOP
    # =====================================================
    for epoch in range(1, TOTAL_EPOCHS + 1):
        
        # 🛡️ HEAD WARM-UP (protects small CNNs and very large CNNs)
        # Pure ViTs are NOT frozen, to avoid gradient shock
        if ((is_fragile or is_massive) and not is_pure_vit) or is_dinov2:
            # DINOv2 large/base need a longer freeze: the head must settle before the
            # backbone is unfrozen, avoiding gradient shock with a low drop_path and LLRD
            if is_dinov2 and (is_massive or is_large_backbones):
                freeze_epochs = 10
            elif is_massive or is_dinov2:
                freeze_epochs = 4
            else:
                freeze_epochs = 2
            if epoch <= freeze_epochs:
                print(f"🔒 Epoch {epoch}: backbone {args.backbone} FROZEN, training the head only!")
                for param in model.features.parameters():
                    param.requires_grad = False
            elif epoch == freeze_epochs + 1:
                print(f"🔓 Epoch {epoch}: backbone {args.backbone} UNFROZEN, training the whole network!")
                for param in model.features.parameters():
                    param.requires_grad = True

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{args.backbone} | Fold {f_idx} | Ep {epoch}")
        
        optimizer.zero_grad()
        
        for i, (imgs, sub_labels, labels, _) in enumerate(pbar):
            imgs = imgs.to(device)
            sub_labels = sub_labels.to(device).long()
            imgs_gray = TF.rgb_to_grayscale(imgs, num_output_channels=3)

            with torch.amp.autocast('cuda'):
                logits_sce, logits_csce, norm_embedding_rgb = model(imgs, labels=sub_labels)
                loss_sce = criterion_sce(logits_sce, sub_labels)
                loss_csce = criterion_sce(logits_csce, sub_labels)

                if L_TRIPLET > 0 or L_CONTRASTIVE > 0:
                    hard_pairs = miner(norm_embedding_rgb, sub_labels)
                    loss_triplet = criterion_triplet(norm_embedding_rgb, sub_labels, hard_pairs)
                    loss_contrastive = criterion_contrastive(norm_embedding_rgb, sub_labels)
                else:
                    loss_triplet = torch.tensor(0.0, device=device)
                    loss_contrastive = torch.tensor(0.0, device=device)

                if USE_SHAPE_LOSS:
                    with torch.no_grad(): 
                        _, _, norm_embedding_gray = model(imgs_gray, labels=sub_labels) 
                    loss_shape = criterion_shape(norm_embedding_rgb, norm_embedding_gray.detach())
                else:
                    loss_shape = 0.0
                
                loss = (L_SCE * loss_sce + L_CSCE * loss_csce + L_TRIPLET * loss_triplet + L_CONTRASTIVE * loss_contrastive + L_SHAPE * loss_shape) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                clip_val = 3.0 if (is_fragile or is_pure_vit) else 5.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            current_avg_loss = total_loss / (i + 1)

            pbar.set_postfix({
                'Avg_L': f"{current_avg_loss:.4f}", 
                'mAp': f"{best_val_map:.4f}", 
                'R1': f"{r1:.4f}"
            })

        scheduler.step()

        # Evaluate once every 5 epochs or in the last 10 epochs
        if (epoch % 10 == 0) or (epoch > TOTAL_EPOCHS - 5):
            # Additive guard of an earlier version: default True keeps the original behavior (ePillID both-side);
            # OGYEIv2 passes both_side=False (single-side)
            val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=getattr(args, 'both_side', True), flag_eval_delta=True)
            val_map = val_metrics['mAP']
            val_r1 = val_metrics['Rank-1']
            print(f"📊 Epoch {epoch} mAP: {val_map:.4f} | (Rank-1: {val_r1:.4f}) | Extra: {val_metrics}")
            _curve = getattr(args, 'val_curve', None)   # where-teacher-acts: the caller records the curve
            if _curve is not None:
                _curve.append({'epoch': epoch, 'mAP': round(float(val_map), 6), 'rank1': round(float(val_r1), 6)})

            if val_map > best_val_map:
                best_val_map = val_map
                r1 = val_r1
                best_metrics = val_metrics
                # where-teacher-acts: the only write path is the run's claimed directory
                # (pipeline/_qj_io.RunClaim). The inherited default was weights/phase3, which in this
                # repository is a symlink into the checkpoints of an earlier study; there is no fallback.
                _claim = getattr(args, 'run_claim', None)
                if _claim is None:
                    raise RuntimeError("where-teacher-acts: train_one_fold needs args.run_claim "
                                       "(pipeline/_qj_io.RunClaim); refusing to write elsewhere")
                _claim.save(model.state_dict(), "model.pth")

            # Additive guard of an earlier version: ceiling early-stop. Default None => original behavior unchanged.
            # When set (e.g. 0.999), stop once val mAP hits the ceiling (no headroom: dataset/class trivially separable).
            _esm = getattr(args, 'early_stop_map', None)
            if _esm is not None and val_map >= _esm:
                print(f"⛔ [early-stop] {args.backbone}: val mAP {val_map:.4f} >= {_esm} (ceiling) @ epoch {epoch}, stopping")
                break

    return best_val_map, r1, best_metrics

def main():
    # where-teacher-acts: runs start from pipeline/_qj_run.py, which builds the split, seeds the run and
    # owns its directory. This batch entry point of the inherited trainer would train on its own fold-0 split.
    raise SystemExit("where-teacher-acts: start runs with pipeline/_qj_run.py --role teacher|base --model <name>")
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base,resnet50,mobilenet_v3_large')
    parser.add_argument('--pooling', type=str, default='gem', choices=['gem', 'mpncov'])
    parser.add_argument('--pipeline_name', type=str, default='baseline')

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--w_sce', type=float, default=1.0)
    parser.add_argument('--w_csce', type=float, default=0.2)
    parser.add_argument('--w_triplet', type=float, default=1.0)
    parser.add_argument('--w_cont', type=float, default=1.0)

    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
    else: 
        device = torch.device('cpu')
    
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()

    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    cond_train = (df_all['fold'] != fold) & (df_all['is_ref'] == 0)
    train_labels = df_all[cond_train]['label_idx'].unique()
    cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
    df_train = df_all[cond_train | cond_ref_train].reset_index(drop=True)
    
    backbone_list = [b.strip() for b in args.backbone.split(',')]
    print(f"🚀 STARTING THE CAMPAIGN WITH: {backbone_list}")

    L_SCE = args.w_sce
    L_CSCE = args.w_csce
    L_TRIPLET = args.w_triplet
    L_CONTRASTIVE = args.w_cont
    
    results_summary = {}
    pooling = args.pooling
    pipeline_name = args.pipeline_name

    # ==========================================
    # 📝 CREATE THE LOG FILES AND WRITE THEIR HEADERS FIRST
    # ==========================================
    os.makedirs("outputs/reports", exist_ok=True)
    log_file_path = f"outputs/reports/{pipeline_name}_batch_experiment_{pooling}_summary.txt"
    log_error_file_path = f"outputs/reports/{pipeline_name}_batch_experiment_{pooling}_error.txt"

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write(f"Batch Experiment {pipeline_name} Results for pooling: {pooling}\n")
            f.write("="*50 + "\n")
            f.write(f"{'Backbone'.ljust(35)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)} | {'SCE'.ljust(6)} | {'CSCE'.ljust(6)} | {'TRIP'.ljust(6)} | {'CONT'.ljust(6)} | EXTRA INFO\n")
            f.write("-"*50 + "\n")
    
    if not os.path.exists(log_error_file_path):
        with open(log_error_file_path, "w") as f:
            f.write(f"Batch Experiment {pipeline_name} Error for pooling: {pooling}\n")
            f.write("="*50 + "\n")
            f.write(f"{'Backbone'.ljust(35)} | {'Error reason'.ljust(6)}\n")
            f.write("-"*50 + "\n")

    for current_backbone in backbone_list:
        print(f"\n{'='*60}")
        print(f"🔥 TRAINING: {current_backbone.upper()}, pooling: {pooling}")
        print(f"{'='*60}")
        
        args.backbone = current_backbone
        try:
            best_map, r1, best_metrics = train_one_fold(args, fold, total_sub_classes, df_train, df_val, df_ref_gallery, device)
            mApCons = best_metrics.get('mAP(Cons)') or 0.0
            mApRef = best_metrics.get('mAP(Ref)') or 0.0
            mApDelta = best_metrics.get('mAP(Delta)') or 0.0
            r1Ref = best_metrics.get('Rank-1(Ref)') or 0.0
            results_summary[current_backbone] = (best_map, r1, best_metrics)
            status = "SUCCESS"
        except Exception as e:
            print(f"❌ A problem occurred while training {current_backbone}")
            print(f"Error details: {e}") 
            traceback.print_exc()     

            with open(log_error_file_path, "a") as f:
                f.write(f"{current_backbone.ljust(35)} | {e}\n")
            
            best_map, r1, mApCons, mApRef, mApDelta, r1Ref  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            results_summary[current_backbone] = (best_map, r1, {})
            status = "FAILED"
        
        # Log the result of each model
        with open(log_file_path, "a") as f:
            if status == "SUCCESS":
                f.write(f"{current_backbone.ljust(35)} | {best_map:.4f} | {r1:.4f} | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f}, mAP(Ref):{mApRef:.4f}, mAP(Delta):{mApDelta:.4f}, Rank-1(Ref):{r1Ref:.4f}\n")
            else:
                f.write(f"{current_backbone.ljust(35)} | 0.0000 | 0.0000 (ERROR) | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f}, mAP(Ref):{mApRef:.4f}, mAP(Delta):{mApDelta:.4f}, Rank-1(Ref):{r1Ref:.4f}\n")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================
    # 🏆 PRINT THE FINAL SUMMARY TABLE TO THE SCREEN
    # ==========================================
    print("\n" + "="*50)
    print("🏆 SUMMARY REPORT (FOLD 0)")
    print("="*70)
    print(f"{'Backbone'.ljust(35)} | {'mAP'.ljust(6)} | {'Rank-1'.ljust(6)} | {'SCE'.ljust(6)} | {'CSCE'.ljust(6)} | {'TRIP'.ljust(6)} | {'CONT'.ljust(6)} | Extra info")
    for bb, (mAP, r1, best_metrics) in results_summary.items():
        mApCons = best_metrics.get('mAP(Cons)') or 0.0
        mApRef = best_metrics.get('mAP(Ref)') or 0.0
        mApDelta = best_metrics.get('mAP(Delta)') or 0.0
        r1Ref = best_metrics.get('Rank-1(Ref)') or 0.0
        print(f"{bb.ljust(35)} | {mAP:.4f} | {r1:.4f} | {L_SCE} | {L_CSCE} | {L_TRIPLET} | {L_CONTRASTIVE} | mAP(Cons):{mApCons:.4f}, mAP(Ref):{mApRef:.4f}, mAP(Delta):{mApDelta:.4f}, Rank-1(Ref):{r1Ref:.4f}")
    print("="*70)
    print(f"📁 All results were saved safely to: {log_file_path}")

if __name__ == '__main__':
    main()