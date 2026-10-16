all_backbones_to_train = [
    # --- 1. Super Large (6 Models) ---
        'vit_large_patch14_dinov2.lvd142m_timm', #0.7214
        'eva02_large_patch14_448_timm',  #0.7952
        'convnext_large_timm', #0.7846
        'swin_large_patch4_window12_384_timm', #0.7283
        'tf_efficientnetv2_l.in21k_ft_in1k_timm', #0.6679
        'convnextv2_large_timm', #0.7891

        # --- 2. Large (13 Models) ---
        'vit_base_patch14_dinov2.lvd142m_timm', #0.7043
        'eva02_base_patch14_448_timm', #0.7655
        'deit_base_patch16_224_timm', #0.6533
        'convnext_base_timm', #0.8181
        'convnextv2_base.fcmae_ft_in22k_in1k_384_timm', #0.7944
        'swin_base_patch4_window12_384_timm',#0.7439
        'efficientnet_b5_timm', #0.6668
        'resnest101e_timm', #0.7129
        'seresnext101_32x4d_timm', #0.7501
        'densenet161_tv', #0.7312
        'tresnet_l_timm', #0.7150
        'resnet101_tv', #0.7060
        'maxvit_base_tf_384_timm', #0.7032

        # --- 3. Medium (9 Models) ---
        'vit_small_patch14_dinov2.lvd142m_timm', #0.6668
        'deit_small_patch16_224_timm', #0.5862
        'swin_small_patch4_window7_224_timm', #0.6872
        'swin_tiny_patch4_window7_224_timm', #0.6154
        'convnext_small_timm', #0.8497
        'efficientnet_b3_timm', #0.7183
        'resnet50_tv', #0.7429
        'maxvit_small_tf_224_timm', #0.7228
        'maxvit_tiny_tf_224_timm', #0.6886

        # --- 4. Edge CNN (10 Models) ---
        'resnet18_tv', #0.5876
        'resnet34_tv', #0.7628
        'efficientnet_b0_timm', #0.6275
        'efficientnet_b1_timm', #0.6062
        'mobilenet_v3_large_tv', #0.6267
        'mobilenet_v2_tv',       #0.5463
        'ghostnet_100_timm', #0.2156
        'shufflenet_v2_x1_0_tv', #0.5015
        'squeezenet1_0_tv', #0.6236
        'repvgg_a0_timm', #0.7212

        # --- 5. LIGHTWEIGHT VIT (Edge Hybrid, 7 Models) ---
        'efficientformerv2_s0_timm', # 0.1737
        'efficientformerv2_s1_timm', #0.3591
        'mobilevit_s_timm', #0.3122
        'mobilevit_xs_timm', #0.3609
        'mobilevit_xxs_timm', #0.2888
        'fasternet_t0_timm', #0.2381
        'levit_128s_timm' #0.1193
]

super_large_backbones = [
    'vit_large_patch14_dinov2.lvd142m',
    'eva02_large_patch14_448',
    'convnext_large',
    'convnextv2_large',
    'swin_large_patch4_window12_384',
    'tf_efficientnetv2_l.in21k_ft_in1k',
    'internimage_l',
    'replknet_31b'
]

large_backbones = [
    'vit_base_patch14_dinov2.lvd142m',
    'eva02_base_patch14_448',
    'deit_base_patch16_224',  # <-- added to this size group
    'convnext_base',
    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    'swin_base_patch4_window12_384',
    'efficientnet_b5',
    'resnest101e',
    'seresnext101_32x4d',
    'densenet161',
    'tresnet_l',
    'resnet101',
    'maxvit_base_tf_384'
]

medium_backbones = [
    'vit_small_patch14_dinov2.lvd142m',
    'deit_small_patch16_224',         # <-- added to this size group (teacher for MobileViT)
    'swin_small_patch4_window7_224',
    'swin_tiny_patch4_window7_224',   # <-- added to this size group (teacher for S0)
    'convnext_small',
    'efficientnet_b3',
    'resnet50',
    'maxvit_small_tf_224',
    'maxvit_tiny_tf_224'             # <-- added to this size group (ablation test)
]

small_backbones = [
    'resnet18',
    'resnet34',
    'efficientnet_b0',
    'efficientnet_b1',
    'mobilenetv3_large_100',
    'mobilenetv2_100',
    'mobilenet_v3_large',
    'mobilenet_v2',
    'ghostnet_100',
    'shufflenet_v2_x1_0',
    'squeezenet1_0',
    'repvgg_a0'
]

vit_backbones = [
    'vit_base_patch14_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m',
    'vit_small_patch14_dinov2.lvd142m',
    'eva02_base_patch14_448',
    'eva02_large_patch14_448',
    'deit_base_patch16_224',
    'deit_small_patch16_224'
]

swin_backbones = [
    'swin_base_patch4_window12_384',
    'swin_large_patch4_window12_384',
    'swin_small_patch4_window7_224',
    'swin_tiny_patch4_window7_224'
]

lightweight_vit_backbones = [
    'efficientformerv2_s0',
    'efficientformerv2_s1',
    'mobilevit_s',
    'mobilevit_xs',
    'mobilevit_xxs',
    'fasternet_t0',
    'tiny_vit_11m_224',   # <-- added edge ViT (from the training script)
    'levit_128s'         # <-- added edge ViT (from the training script)
]

# ====================================================================
# Edge CNNs with heavy depth-wise convolutions need their own LR config
# (validated in retraining round 2)
# ====================================================================

# Ghostnet-like: collapsed completely under the default config
# (ghostnet_100: 0.21 mAP) and needs a low LR of 2e-5/2e-4 to recover (validated, reaching 0.45)
ghostnet_like_backbones = [
    'ghostnet_100',
]

# DW-moderate: depth-wise architectures that were already stable at a high LR
# Lowering the LR to 2e-5/2e-4 in round 2 made them underfit (mAP fell by 0.05 to 0.06)
# They need an intermediate LR of 4e-5/4e-4
dw_moderate_backbones = [
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenetv2_100',
    'mobilenetv3_large_100',
    'shufflenet_v2_x1_0',
]

# ====================================================================
# Medium fragile: small, low-capacity models in the medium group (mAP < 0.70)
# Applies the pattern that worked for edge CNNs: a light margin and more epochs
# ====================================================================
fragile_medium_backbones = [
    'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'vit_small_patch14_dinov2.lvd142m',
]

# Medium experimental: borderline models, lightly tuned to see whether they improve
medium_experimental_backbones = [
    'maxvit_tiny_tf_224',
    'vit_base_patch14_dinov2.lvd142m',
]

# ====================================================================
# Architecture-specific groups (explicit lists, no substring matching)
# ====================================================================

# DINOv2 family: needs LLRD, a low LR, a long freeze and drop_path 0.1
# A new DINOv2 model MUST be added here so that its special config is applied
dinov2_backbones = [
    'vit_small_patch14_dinov2.lvd142m',
    'vit_base_patch14_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m',
]

# EfficientNet family: super_large uses accum=16 (it needs more updates because its baseline is weak)
# Add any new EfficientNet here
efficientnet_backbones = [
    # super_large
    'tf_efficientnetv2_l.in21k_ft_in1k',
    # large
    'efficientnet_b5',
    # medium
    'efficientnet_b3',
    # small
    'efficientnet_b0',
    'efficientnet_b1',
]

# High weight decay backbones (wd=0.1): very large architectures need strong regularization
# Applies to: all super_large, all large, RepLKNet and the EVA family
# Plus: special architectures that need a high wd (RepLKNet with reparameterization)
high_wd_backbones = [
    'replknet_31b',
    'eva02_base_patch14_448',
    'eva02_large_patch14_448',
]

groupA = [
    # --- Lightweight ViT (7) ---
    'efficientformerv2_s0_timm',     # 0.1737 → expected 0.50+, result 0.1496
    'efficientformerv2_s1_timm',     # 0.3591 → expected 0.60+, result 0.3467
    'mobilevit_s_timm',              # 0.3122 → expected 0.50+, result 0.4033
    'mobilevit_xs_timm',             # 0.3609 → expected 0.50+, result 0.3293
    'mobilevit_xxs_timm',            # 0.2888 → expected 0.40+, result 0.2999
    'fasternet_t0_timm',             # 0.2381 → expected 0.40+, result 0.2241
    'levit_128s_timm',               # 0.1193 → expected 0.40+, result 0.1136

    # --- DW-heavy Edge CNN (4) ---
    'mobilenet_v3_large_tv',         # 0.6267 → expected 0.70+, result 0.5947
    'mobilenet_v2_tv',               # 0.5463 → expected 0.65+, result 0.4830
    'ghostnet_100_timm',             # 0.2156 → expected 0.60+ (fix collapse), result 0.4465
    'shufflenet_v2_x1_0_tv',         # 0.5015 → expected 0.60+, result 0.4401
]

groupB = [
    # --- Remaining edge CNNs (6) ---
    'resnet18_tv',                                       # 0.5876, result 0.7549
    'resnet34_tv',                                       # 0.7628 (hold or improve slightly), result 0.7856
    'efficientnet_b0_timm',                              # 0.6275, result 0.6846
    'efficientnet_b1_timm',                              # 0.6062, result 0.7668
    'squeezenet1_0_tv',                                  # 0.6236, result 0.6527
    'repvgg_a0_timm',                                    # 0.7212 (hold or improve slightly), result 0.7493

    # --- Super Large (6), drop_path and accumulation changed ---
    'vit_large_patch14_dinov2.lvd142m_timm',             # 0.7214 (accumulation 16→20), result 0.6905
    'eva02_large_patch14_448_timm',                      # 0.7952 (top, hold or improve slightly), result 0.8055
    'convnext_large_timm',                               # 0.7846 → expected 0.80+, result 0.7818
    'swin_large_patch4_window12_384_timm',               # 0.7283 → expected 0.76+, result 0.7237
    'tf_efficientnetv2_l.in21k_ft_in1k_timm',            # 0.6679 → expected 0.71+, result 0.6961
    'convnextv2_large_timm',                             # 0.7891 → expected 0.81+, result 0.7875

    # --- Large (12), drop_path changed (except DINOv2-base, which has its own drop_path setting) ---
    'eva02_base_patch14_448_timm',                       # 0.7655, result 0.7751
    'deit_base_patch16_224_timm',                        # 0.6533, result 0.6594
    'convnext_base_timm',                                # 0.8181 (top, consider skipping), result 0.8096
    'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',      # 0.7944, result 0.8006
    'swin_base_patch4_window12_384_timm',                # 0.7439, result 0.7528
    'efficientnet_b5_timm',                              # 0.6668, result 0.8578
    'resnest101e_timm',                                  # 0.7129, result 0.7593
    'seresnext101_32x4d_timm',                           # 0.7501, result 0.7844
    'densenet161_tv',                                    # 0.7312, result 0.7765
    'tresnet_l_timm',                                    # 0.7150, result 0.7730
    'resnet101_tv',                                      # 0.7060, result 0.7516
    'maxvit_base_tf_384_timm',                           # 0.7032, result 0.7762
]

# ====================================================================
# GROUP C: no retraining needed (10 models), the config is unchanged
#   Medium (9): outside the scope of the change
#   DINOv2-base: drop_path uses its own setting (0.1), accumulation unchanged
# Keep the old weights; re-evaluate only if needed
# ====================================================================
groupC = [
    # --- DINOv2-base (in the large group, but with its own drop_path setting) ---
    'vit_base_patch14_dinov2.lvd142m_timm',              # 0.7043 (already fixed, kept)

    # --- Medium (9) ---
    'vit_small_patch14_dinov2.lvd142m_timm',             # 0.6668
    'deit_small_patch16_224_timm',                       # 0.5862
    'swin_small_patch4_window7_224_timm',                # 0.6872
    'swin_tiny_patch4_window7_224_timm',                 # 0.6154
    'convnext_small_timm',                               # 0.8497 (top overall!)
    'efficientnet_b3_timm',                              # 0.7183
    'resnet50_tv',                                       # 0.7429
    'maxvit_small_tf_224_timm',                          # 0.7228
    'maxvit_tiny_tf_224_timm',                           # 0.6886
]

# ====================================================================
# GROUP A_v3: the models retrained in the third round (16 in total)
# Comment format: round 1 (original) → round 2 (already run) → expected in round 3, then the round 3 result
# ====================================================================
groupA_v3 = [
    # --- Lightweight ViT (7), round 3: margin=0.25, L_CSCE=0.10, lr_head=2e-4 ---
    # round 2 used margin=0.15, L_CSCE=0.05 and lr_head=1e-4, and 5 of the 7 models got worse
    'efficientformerv2_s0_timm',                # round 1: 0.1737 → round 2: 0.1496 → expected in round 3: 0.30+, result 0.2675
    'efficientformerv2_s1_timm',                # round 1: 0.3591 → round 2: 0.3467 → expected in round 3: 0.45+, result 0.4711
    'mobilevit_s_timm',                         # round 1: 0.3122 → round 2: 0.4033 → expected in round 3: 0.45+ (round 2 was already good), result 0.4148
    'mobilevit_xs_timm',                        # round 1: 0.3609 → round 2: 0.3293 → expected in round 3: 0.45+, result 0.3950
    'mobilevit_xxs_timm',                       # round 1: 0.2888 → round 2: 0.2999 → expected in round 3: 0.40+, result 0.3244
    'fasternet_t0_timm',                        # round 1: 0.2381 → round 2: 0.2241 → expected in round 3: 0.35+, result 0.3059
    'levit_128s_timm',                          # round 1: 0.1193 → round 2: 0.1136 → expected in round 3: 0.30+, result 0.1194

    # --- DW-moderate Edge CNN (3), round 3: lr=4e-5/4e-4 (intermediate) ---
    # round 2 used lr=2e-5/2e-4 and underfit, with mAP 0.05 to 0.06 lower than in round 1
    'mobilenet_v3_large_tv',                    # round 1: 0.6267 → round 2: 0.5947 → expected in round 3: 0.65+, result 0.6783
    'mobilenet_v2_tv',                          # round 1: 0.5463 → round 2: 0.4830 → expected in round 3: 0.60+, result 0.5783
    'shufflenet_v2_x1_0_tv',                    # round 1: 0.5015 → round 2: 0.4401 → expected in round 3: 0.55+, result 0.5301

    # --- DINOv2 super_large (1), round 3: accum 20→16 ---
    # round 2 used accum=20: LLRD with a low LR and a high accum gives too few updates, so mAP fell
    'vit_large_patch14_dinov2.lvd142m_timm',    # round 1: 0.7214 → round 2: 0.6905 → expected in round 3: 0.73+, result 0.7038

    # --- Fragile Medium (3), round 3: margin=0.25, epochs=80 (newly added after the Group C analysis) ---
    'swin_tiny_patch4_window7_224_timm',        # baseline: 0.6154 → expected in round 3: 0.68+, result 0.6616
    'swin_small_patch4_window7_224_timm',       # baseline: 0.6872 → expected in round 3: 0.72+, result 0.7019
    'vit_small_patch14_dinov2.lvd142m_timm',    # baseline: 0.6668 → expected in round 3: 0.71+ (LLRD decay 0.85), result 0.6993

    # --- Medium experimental (2), round 3: margin=0.30, epochs=70 (newly added after the Group C analysis) ---
    'maxvit_tiny_tf_224_timm',                  # baseline: 0.6886 → expected in round 3: 0.71+, result 0.6939
    'vit_base_patch14_dinov2.lvd142m_timm',     # baseline: 0.7043 → expected in round 3: 0.73+ (LLRD decay 0.85), result 0.7332
]  # 16 models in total (11 earlier ones plus 5 from Group C)