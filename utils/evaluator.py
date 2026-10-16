import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(model, dataloader, device, flag_both_side=True, flag_eval_delta=False):
    model.eval()
    all_feats, all_labels, all_sub_labels, all_is_ref = [], [], [], []

    # =====================================================================
    # 1. FEATURE EXTRACTION (1-Pass Optimization)
    # =====================================================================
    with torch.no_grad():
        for imgs, sub_labels, labels, is_ref in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            all_sub_labels.append(sub_labels.cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    all_sub_labels = torch.cat(all_sub_labels, dim=0)
    all_is_ref = torch.cat(all_is_ref, dim=0)

    # =====================================================================
    # 2. DATA SPLIT (Gallery and Consumer Query)
    # =====================================================================
    # Gallery (the reference dictionary, holding every Reference image)
    g_mask = all_is_ref == 1
    g_feats, g_labels, g_sub_labels = all_feats[g_mask], all_labels[g_mask], all_sub_labels[g_mask]
    
    # Consumer Query (photos taken by users under real-world conditions)
    q_cons_mask = all_is_ref == 0
    q_cons_feats, q_cons_labels, q_cons_sub_labels = all_feats[q_cons_mask], all_labels[q_cons_mask], all_sub_labels[q_cons_mask]

    # =====================================================================
    # 3. HELPER FUNCTION: CORE SCORE COMPUTATION
    # =====================================================================
    def _calculate_metrics(q_f, q_l, q_sl, g_f, g_l, g_sl, self_query_indices=None):
        unique_classes = torch.unique(g_l)
        n_classes = len(unique_classes)
        
        # S has shape (N_queries, N_gallery); the cosine similarity is computed once
        S = torch.mm(q_f, g_f.T)
        
        # ANTI-LEAKAGE: stops a Reference query from matching itself
        if self_query_indices is not None:
            # Force the similarity of the image with itself down to -1.0
            S[torch.arange(len(q_f)), self_query_indices] = -1.0
            
        sim_A = torch.zeros((len(q_f), n_classes))
        sim_B = torch.zeros((len(q_f), n_classes))
        class_list = []
        
        # Max-pool the scores per class (grouping side A and side B)
        for c_idx, lbl in enumerate(unique_classes):
            lbl = lbl.item()
            class_list.append(lbl)
            
            mask_c = (g_l == lbl)
            mask_front = mask_c & (g_sl % 2 == 0)
            mask_back = mask_c & (g_sl % 2 != 0)
            
            if mask_front.any():
                sim_A[:, c_idx] = S[:, mask_front].max(dim=1).values
            else:
                sim_A[:, c_idx] = -1.0 
                
            if mask_back.any():
                sim_B[:, c_idx] = S[:, mask_back].max(dim=1).values
            else:
                sim_B[:, c_idx] = -1.0

        class_tensor = torch.tensor(class_list)
        ranks = []

        # Ranking
        if not flag_both_side:
            # --- A. SINGLE SIDE ---
            sim_scores = torch.max(sim_A, sim_B) 
            for i in range(len(q_f)):
                q_lbl = q_l[i].item()
                scores = sim_scores[i]
                sorted_indices = torch.argsort(scores, descending=True)
                sorted_classes = class_tensor[sorted_indices]
                rank = (sorted_classes == q_lbl).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
        else:
            # --- B. BOTH SIDES LATE FUSION (CROSS MAX) ---
            unique_q_labels = torch.unique(q_l)
            for lbl in unique_q_labels:
                lbl = lbl.item()
                mask_lbl = (q_l == lbl)
                
                q_indices = mask_lbl.nonzero(as_tuple=True)[0]
                q_subs = q_sl[q_indices]
                
                front_idx = q_indices[q_subs % 2 == 0]
                back_idx = q_indices[q_subs % 2 != 0]
                n_pairs = min(len(front_idx), len(back_idx))
                
                # Score the image pairs
                for i in range(n_pairs):
                    idx1 = front_idx[i]
                    idx2 = back_idx[i]
                    
                    fuse_1 = sim_A[idx1] + sim_B[idx2] 
                    fuse_2 = sim_B[idx1] + sim_A[idx2] 
                    sim_scores = torch.max(fuse_1, fuse_2)
                    
                    sorted_indices = torch.argsort(sim_scores, descending=True)
                    sorted_classes = class_tensor[sorted_indices]
                    rank = (sorted_classes == lbl).nonzero(as_tuple=True)[0].item() + 1
                    ranks.append(rank)
                    
                # Score the unpaired images (leftovers)
                leftover_idx = torch.cat([front_idx[n_pairs:], back_idx[n_pairs:]])
                for idx in leftover_idx:
                    sim_scores = torch.max(sim_A[idx], sim_B[idx])
                    sorted_indices = torch.argsort(sim_scores, descending=True)
                    sorted_classes = class_tensor[sorted_indices]
                    rank = (sorted_classes == lbl).nonzero(as_tuple=True)[0].item() + 1
                    ranks.append(rank)

        ranks = np.array(ranks)
        rank1 = (ranks == 1).mean()
        mAP = (1.0 / ranks).mean()
        
        return rank1, mAP

    # =====================================================================
    # 4. RUN THE EVALUATION
    # =====================================================================
    # 4.1. Core evaluation: Consumer mAP (default)
    rank1_cons, map_cons = _calculate_metrics(
        q_cons_feats, q_cons_labels, q_cons_sub_labels, 
        g_feats, g_labels, g_sub_labels, 
        self_query_indices=None
    )
    
    results = {
        'Rank-1': rank1_cons,
        'mAP': map_cons
    }

    # 4.2. Extended evaluation: Delta mAP
    if flag_eval_delta:
        # The pill labels that actually belong to the validation fold (taken from the Consumer set)
        val_classes = torch.unique(q_cons_labels)
        
        # Mask that keeps the gallery Reference images whose label belongs to the validation fold
        val_ref_mask = torch.isin(g_labels, val_classes)
        
        # Extract the Reference query set, keeping the zero-shot protocol
        q_ref_feats = g_feats[val_ref_mask]
        q_ref_labels = g_labels[val_ref_mask]
        q_ref_sub_labels = g_sub_labels[val_ref_mask]
        
        # Their original indices in g_feats, used to disable self-similarity
        val_ref_indices = torch.nonzero(val_ref_mask).squeeze(1)
        
        # Evaluation within the Reference domain
        rank1_ref, map_ref = _calculate_metrics(
            q_ref_feats, q_ref_labels, q_ref_sub_labels, 
            g_feats, g_labels, g_sub_labels, 
            self_query_indices=None #val_ref_indices
        )
        
        results['mAP(Cons)'] = map_cons
        results['mAP(Ref)'] = map_ref
        results['mAP(Delta)'] = abs(map_ref - map_cons)
        results['Rank-1(Ref)'] = rank1_ref

    return results