"""Silence known-benign third-party log/warning noise for cleaner pipeline logs.

Every message silenced here has been confirmed HARMLESS:
  - HF Hub 'unauthenticated requests': we load the fine-tuned .pth ON TOP of the pretrained
    weights, so nothing depends on the pretrained download.
  - torch.jit.TracerWarning ('if x.shape[-1] != safe_size'): the if branch is baked in correctly
    at ONNX export, because the pipeline always feeds the export size at inference.
  - ORT root-logger 'Axis 1 out-of-range for weight ... rank 1': per-channel quantization skips
    1-D weights (LayerNorm/BatchNorm) by itself and falls back to per-tensor (correct).
  - ORT 'Please consider pre-processing': quant_pre_process is now run explicitly.
"""
from __future__ import annotations

import contextlib
import logging
import warnings


def silence_benign_noise() -> None:
    """Silence third-party warnings confirmed to be harmless. Idempotent: calling it again is fine."""
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("timm").setLevel(logging.ERROR)
    try:
        import torch
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    except Exception:  # torch not installed, or no TracerWarning
        pass


@contextlib.contextmanager
def quiet_root_logs(level: int = logging.ERROR):
    """Temporarily raise the ROOT logger level (ORT quantization logs 'WARNING:root:...' via root).

    The pipeline scripts use print() for their own output, so raising root to ERROR does NOT hide it.
    """
    root = logging.getLogger()
    prev = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(prev)
