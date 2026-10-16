"""
Structural re-parameterization of RepVGG (where-teacher-acts).

RepVGG trains a multi-branch block (3x3 conv + BN, 1x1 conv + BN, and an identity BN) and is meant to
be deployed as the single 3x3 convolution those branches add up to. This study deploys RepVGG-A0 in
that fused form everywhere: the FP32 references are exported fused, and Part II fine-tunes the fused
network, so the quantization simulated during training is the quantization ONNX Runtime applies. In
the training form, the identity branch and the branch sum are quantized at deployment but never seen
by the simulator (validation 2026-10-11: real INT8 MinMax 0.28 to 0.43 against 0.68 to 0.73 simulated).

Fusion is exact up to float rounding: it folds each BatchNorm's running statistics into the
convolution weights and sums the branches into one kernel and bias (timm's RepVggBlock.reparameterize).
"""


def _repvgg_block_type():
    from timm.models.byobnet import RepVggBlock
    return RepVggBlock


def count_unfused(module):
    """RepVGG blocks inside `module` still in their multi-branch training form."""
    block = _repvgg_block_type()
    return sum(1 for m in module.modules() if isinstance(m, block) and m.reparam_conv is None)


def fuse_repvgg_branches(module):
    """Fuse every multi-branch RepVGG block inside `module`, in place, into its single deploy-time
    convolution; returns how many blocks were fused. Uses the BatchNorm running statistics, so call it
    on a trained network (the branches are deleted afterwards)."""
    block = _repvgg_block_type()
    blocks = [m for m in module.modules() if isinstance(m, block) and m.reparam_conv is None]
    for b in blocks:
        b.reparameterize()
    if count_unfused(module):
        raise RuntimeError("some RepVGG blocks are still multi-branch after fusion")
    return len(blocks)


def is_fused_state(state_dict):
    """True for a state dict saved from a fused RepVGG network (keys of the single convolutions)."""
    return any(".reparam_conv." in k for k in state_dict)
