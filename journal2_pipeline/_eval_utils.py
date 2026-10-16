"""
Evaluation utilities (library).

Provides:
- extract_features_onnx(): batched ONNX inference (FP32 or INT8) over a dataloader
- calculate_both_sides_map(): Both-Sides Late Fusion mAP for ePillID
- evaluate_onnx_on_epillid(): convenience wrapper that returns (mAP, Rank-1)
- build_eval_loader_epillid(): query (fold 0 Consumer) + gallery (all Reference)

Extracted from an earlier version of this code, with 2 improvements:
1. Pure functions, reusable for cross-region evaluation (OGYEIv2)
2. Separate extract and eval steps, so features can be cached for a representation probe
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        raise ImportError("onnxruntime is required: pip install onnxruntime") from e
    return ort


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import load_epill_full_data  # noqa: E402
from utils.dataset_loader import PillDataset  # noqa: E402


# ============================================================
# 1. ONNX FEATURE EXTRACTION
# ============================================================

def extract_features_onnx(
    onnx_path: str,
    dataloader: DataLoader,
    desc: Optional[str] = None,
    intra_threads: int = 0,  # 0 = ORT default (multi-thread for batch infer)
    provider: str = 'CPUExecutionProvider',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the dataloader through the model and return (feats_L2, labels, sub_labels, is_ref).

    intra_threads=0: use ORT's default threading (multi-threaded, faster for batches).
    To measure single-image latency, use intra=1 instead (see _benchmark_utils).

    provider: the desired execution provider. Use 'CUDAExecutionProvider' to SPEED UP
      FP32 evaluation (mAP is value-correct whatever the provider). INT8 SHOULD stay on
      'CPUExecutionProvider' so it runs true integer arithmetic (fidelity to the edge device);
      the CUDA EP with QDQ is FP32 fake quantization, which differs from the real numbers
      and often falls back.
      ⚠️ The CUDA EP only works when `onnxruntime-gpu` is installed (the plain `onnxruntime`
      package has NO CUDA EP). CPU is always appended as a safe fallback, so without the GPU
      build it simply runs on CPU instead of failing.
    """
    ort = _import_ort()

    def _extract(providers_list):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads
        session = ort.InferenceSession(onnx_path, opts, providers=providers_list)
        input_name = session.get_inputs()[0].name

        feats, labels, sub_labels, is_refs = [], [], [], []
        iterator = tqdm(dataloader, desc=desc or os.path.basename(onnx_path), leave=False)
        for imgs, sub_lbl, lbl, is_ref in iterator:
            imgs_np = imgs.numpy().astype(np.float32)
            out = session.run(None, {input_name: imgs_np})[0]
            feats.append(torch.as_tensor(out))
            labels.append(lbl)
            sub_labels.append(sub_lbl)
            is_refs.append(is_ref)

        del session
        gc.collect()
        feats_t = torch.nn.functional.normalize(torch.cat(feats, dim=0), p=2, dim=1)
        return feats_t, torch.cat(labels), torch.cat(sub_labels), torch.cat(is_refs)

    if provider == 'CPUExecutionProvider':
        return _extract(['CPUExecutionProvider'])

    # GPU requested: [CUDA, CPU] lets ORT place the nodes CUDA does not support on the CPU
    # (within the graph). BUT the CUDA EP does NOT fall back to CPU on out-of-memory at RUN time
    # (BFCArena AllocateRawInternal failed): large models (eva02_large/maxvit/convnextv2 at batch 64,
    # attention O(tokens²)) exceed VRAM and raise an exception. Catch it and retry EVERYTHING on CPU
    # (slow but correct; 62GB of system RAM is plenty). mAP is value-correct on both EPs.
    try:
        return _extract([provider, 'CPUExecutionProvider'])
    except Exception as exc:  # noqa: BLE001
        print(f'   ⚠️  {provider} failed ({str(exc)[:70]}) → retrying on CPU')
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return _extract(['CPUExecutionProvider'])


# ============================================================
# 2. BOTH-SIDES LATE-FUSION mAP (ePillID protocol)
# ============================================================

def calculate_both_sides_map(
    feats: torch.Tensor,
    labels: torch.Tensor,
    sub_labels: torch.Tensor,
    is_ref: torch.Tensor,
    device: Optional[torch.device] = None,
    return_ranks: bool = False,
):
    """Cross-Max + Both-Sides fusion (same as the inherited evaluator in utils/evaluator.py).

    return_ranks=False (default) keeps the old behaviour UNCHANGED: returns (mAP, rank1). Set True to
    also get the per-query rank array, used for bootstrapping. The new parameter defaults to the old
    behaviour, so every existing call gives unchanged results.

    Each query has a front-back pair. Front query → max sim with the front gallery (sim_A),
    back query → max sim with the back gallery (sim_B). Final = max(fuse_1, fuse_2)
    with fuse_1 = sim_A[front] + sim_B[back], fuse_2 = sim_B[front] + sim_A[back].

    Returns (mAP, Rank-1).
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    feats = feats.to(device)
    labels = labels.to(device)
    sub_labels = sub_labels.to(device)
    is_ref = is_ref.to(device)

    g_mask = is_ref == 1
    q_mask = is_ref == 0
    g_f, g_l, g_sl = feats[g_mask], labels[g_mask], sub_labels[g_mask]
    q_f, q_l, q_sl = feats[q_mask], labels[q_mask], sub_labels[q_mask]

    if len(q_f) == 0 or len(g_f) == 0:
        return float('nan'), float('nan')

    unique_classes = torch.unique(g_l)
    n_classes = len(unique_classes)
    class_tensor = unique_classes.to(device)

    S = torch.mm(q_f, g_f.T)  # (Q, G)
    sim_A = torch.full((len(q_f), n_classes), -1.0, device=device)  # front gallery
    sim_B = torch.full((len(q_f), n_classes), -1.0, device=device)  # back gallery

    for c_idx, lbl in enumerate(unique_classes):
        lbl_val = lbl.item()
        mask_c = (g_l == lbl_val)
        mask_front = mask_c & (g_sl % 2 == 0)
        mask_back = mask_c & (g_sl % 2 != 0)
        if mask_front.any():
            sim_A[:, c_idx] = S[:, mask_front].max(dim=1).values
        if mask_back.any():
            sim_B[:, c_idx] = S[:, mask_back].max(dim=1).values

    ranks = []
    unique_q_labels = torch.unique(q_l)
    for lbl in unique_q_labels:
        lbl_val = lbl.item()
        q_indices = (q_l == lbl_val).nonzero(as_tuple=True)[0]
        q_subs = q_sl[q_indices]
        front_idx = q_indices[q_subs % 2 == 0]
        back_idx = q_indices[q_subs % 2 != 0]
        n_pairs = min(len(front_idx), len(back_idx))

        for i in range(n_pairs):
            idx1, idx2 = front_idx[i], back_idx[i]
            fuse_1 = sim_A[idx1] + sim_B[idx2]
            fuse_2 = sim_B[idx1] + sim_A[idx2]
            sim_scores = torch.max(fuse_1, fuse_2)
            sorted_classes = class_tensor[torch.argsort(sim_scores, descending=True)]
            rank = (sorted_classes == lbl_val).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        for idx in torch.cat([front_idx[n_pairs:], back_idx[n_pairs:]]):
            sim_scores = torch.max(sim_A[idx], sim_B[idx])
            sorted_classes = class_tensor[torch.argsort(sim_scores, descending=True)]
            rank = (sorted_classes == lbl_val).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    mAP = float((1.0 / ranks_arr).mean())
    rank1 = float((ranks_arr == 1).mean())

    del S, sim_A, sim_B, g_f, q_f
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if return_ranks:
        # The rank of EACH query. Both metrics above are means over this array, so keeping it keeps
        # enough information to bootstrap over the query set WITHOUT rerunning the model. The element
        # order is deterministic (it walks the sorted unique_q_labels), so two models measured on the
        # same loader give two arrays ALIGNED ELEMENT BY ELEMENT, the condition for a paired bootstrap.
        return mAP, rank1, ranks_arr
    return mAP, rank1


# ============================================================
# 3. ePillID EVAL LOADER (test split: fold 0 Consumer + all Reference)
# ============================================================

def build_eval_loader_epillid(
    img_size: int = 384,
    batch_size: int = 64,
    num_workers: int = 4,
    test_fold: int = 0,
) -> DataLoader:
    """Loader that follows the paper's protocol: query = fold 0 Consumer, gallery = all Reference."""
    df_all = load_epill_full_data()
    df_ref = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_query = (df_all[(df_all['fold'] == test_fold) & (df_all['is_ref'] == 0)]
                .reset_index(drop=True))
    df_eval = pd.concat([df_query, df_ref], ignore_index=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return DataLoader(PillDataset(df_eval, transform=transform),
                      batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


# ============================================================
# 4. END-TO-END WRAPPER
# ============================================================

def evaluate_onnx_on_epillid(
    onnx_path: str,
    img_size: int = 384,
    batch_size: int = 64,
    test_fold: int = 0,
    device: Optional[torch.device] = None,
    desc: Optional[str] = None,
) -> dict:
    """Run the ePillID test set through 1 ONNX model → return {mAP, rank1, n_query, n_gallery}.

    A loader could be reused if the caller passed one in, but this function builds its own for convenience.
    """
    loader = build_eval_loader_epillid(img_size=img_size,
                                       batch_size=batch_size,
                                       test_fold=test_fold)
    feats, labels, sub_labels, is_ref = extract_features_onnx(onnx_path, loader, desc=desc)
    mAP, rank1 = calculate_both_sides_map(feats, labels, sub_labels, is_ref, device=device)

    return {
        'mAP':       round(float(mAP), 4) if not np.isnan(mAP) else float('nan'),
        'rank1':     round(float(rank1), 4) if not np.isnan(rank1) else float('nan'),
        'n_query':   int((is_ref == 0).sum().item()),
        'n_gallery': int((is_ref == 1).sum().item()),
    }
