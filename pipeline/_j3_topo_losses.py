"""
Placeholder for the optional structural distillation terms of an earlier version of this code, which the KD
trainer adds only for kd_type 'topo', 'topo_only', 'hybrid_topo' or 'rkd'. This study uses the pointwise cosine
term alone (kd_type 'cosine'), so those terms are not part of this release; the trainer still imports the name.
"""


class TopoKDLoss:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("where-teacher-acts: the structural distillation terms are not part of this "
                                  "release; this study uses kd_type 'cosine'")
