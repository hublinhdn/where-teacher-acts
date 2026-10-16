"""
PTQ utilities (library).

Provides:
- EpillCalibrationDataReader: feeds calibration batches into ONNX Runtime
- export_fp32_onnx(): load .pth → export FP32 ONNX (opset 14, QDQ-friendly)
- run_ptq(): quantize_static wrapper for both methods, min-max + entropy
- run_ptq_minmax(), run_ptq_entropy(): convenience wrappers

Extracted from an earlier version of this code, with 2 improvements:
1. Supports BOTH min-max AND entropy calibration (no single hard-coded method)
2. Library style (pure functions, no hard-coded lookups)
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import load_epill_full_data  # noqa: E402
from utils.dataset_loader import PillDataset  # noqa: E402
from models.pill_retrieval_model import PillRetrievalModel  # noqa: E402

from ._quiet import quiet_root_logs, silence_benign_noise  # noqa: E402

# onnxruntime imports are done lazily: they only raise when a function is called, so this
# file still imports in a development environment (laptop) without ORT installed.
def _import_ort_quant():
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "onnxruntime + onnxruntime-tools are required to run quantization. "
            "pip install onnxruntime onnx"
        ) from e
    # quant_pre_process (shape inference + graph opt): optional, best-effort.
    try:
        from onnxruntime.quantization import quant_pre_process
    except ImportError:
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process
        except ImportError:
            quant_pre_process = None
    return {
        'CalibrationDataReader': CalibrationDataReader,
        'CalibrationMethod':     CalibrationMethod,
        'QuantFormat':           QuantFormat,
        'QuantType':             QuantType,
        'quantize_static':       quantize_static,
        'quant_pre_process':     quant_pre_process,
    }


# ============================================================
# 1. CALIBRATION DATA READER (factory)
# ============================================================
#
# ONNX Runtime checks the reader by duck typing (get_next() returns a dict or None),
# but for maximum compatibility we still extend the proper base class when ORT is present.

def make_calibration_reader(
    dataloader: DataLoader,
    input_name: str,
    num_batches=None,
):
    """Create a CalibrationDataReader (extends the ORT base class if ORT is available).

    num_batches=None (default): consume the WHOLE loader, so the number of calibration images is
    controlled ONLY through build_calibration_loader(num_samples=...). This removes an old footgun
    (num_samples and num_batches disagreeing, which silently dropped the extra images).
    """
    ort_mod = _import_ort_quant()
    base = ort_mod['CalibrationDataReader']

    class EpillCalibrationDataReader(base):
        def __init__(self):
            self.dataloader = dataloader
            self.input_name = input_name
            self.num_batches = num_batches
            self.enum_data = iter(dataloader)
            self.current_batch = 0

        def get_next(self):
            if self.num_batches is not None and self.current_batch >= self.num_batches:
                return None
            try:
                imgs, *_ = next(self.enum_data)
                self.current_batch += 1
                return {self.input_name: imgs.numpy().astype(np.float32)}
            except StopIteration:
                return None

        def rewind(self):
            self.enum_data = iter(self.dataloader)
            self.current_batch = 0

    return EpillCalibrationDataReader()


# ============================================================
# 2. CALIBRATION LOADER (built from the ePillID training folds)
# ============================================================

def build_calibration_loader(
    img_size: int = 384,
    batch_size: int = 8,
    fold_excluded: int = 0,
    seed: int = 123,
    num_samples: int = 16,
) -> DataLoader:
    """16 random images from the ePillID training folds, MIXING both domains (Reference + Consumer).

    Calibration N = 16, SHARED by MinMax and Entropy, for a controlled comparison:
      - N=16 is FORCED by memory: ORT entropy collects the activations of EVERY tensor for EVERY
        image (O(N)); eva02 at 392px has O(tokens²) attention maps of about 2GB per image, so only
        16 images fit in 62GB of RAM (128 already ran out of memory on convnext_base). MinMax needs
        O(1) memory but keeps 16 for the controlled comparison.
      - 16 is still enough for range/histogram estimation: each CNN image gives hundreds of
        thousands of activations per channel, far more samples than a 128-bin histogram needs
        (class coverage does NOT matter for calibration, which needs only the range, not class semantics).
      - The SAME N for both methods isolates the calibration-method effect (the variable under study)
        without confounding it with the number of images. It also matches deployment practice
        (one calibration set for both methods).
      - Entropy needs O(N) memory, so 128 avoids running out of memory on 62GB (1280 needs about 64GB
        and runs out). MinMax needs O(1) memory but still uses 128 for the controlled comparison.
      - MIXES is_ref==1 (Reference) and is_ref==0 (Consumer): the encoder encodes BOTH domains
        at inference, so calibration must see the range of both.
      - seed=123 differs from the representation probe's seed (42), which avoids leakage.

    NOTE: N=1280 (kept consistent with an earlier study) is for the REPRESENTATION PROBE
    (topological similarity metrics, built by a separate probe loader in an earlier version of this
    code), where CKA/PWCCA need many samples for a stable geometry.
    Calibration is a different task (range estimation), so a smaller N is reasonable and necessary.

    fold_excluded = 0: use folds 1-4 (fold 0 is the test fold; never calibrate on test).
    """
    df_all = load_epill_full_data()
    df_train = df_all[df_all['fold'] != fold_excluded].reset_index(drop=True)
    if len(df_train) > num_samples:
        df_train = df_train.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PillDataset(df_train, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# ============================================================
# 3. FP32 ONNX EXPORT
# ============================================================

INPUT_NAMES = ['input_image']
OUTPUT_NAMES = ['output_embedding']


def _make_eva02_export_safe(model, backbone: str, img_size: int) -> None:
    """ONNX-export patch for EVA02 (eva02 ONLY). Rebuilds the backbone with
    `dynamic_img_size=False` + a static `img_size` + a resampled pos_embed.

    WHY: EVA02 with `dynamic_img_size=True` (the wrapper default) produces a DYNAMIC token reshape,
    so ONNX shape inference breaks at `/features/Reshape_1` and `/features/Concat_12`
    (Swin with the same flag is fine; the difference comes from EVA02's own rotary/patch reshape).
    Turning dynamic off and fixing img_size builds a STATIC pos_embed on the right grid, so the
    export is clean.

    SAFE FOR THE SHARED MODEL CODE: only `model.features` of a throwaway object is replaced IN
    MEMORY at export time. The file `models/pill_retrieval_model.py` (also imported by the training
    code) is NOT changed. This function is only called when 'eva02' is in the backbone name, so the
    other 19 backbones keep the old path.

    VERIFIED (with an export probe in an earlier version of this code): this configuration gives
    cosine = 1.0 against the original forward pass (large and base), so the features are EQUIVALENT
    and mAP is unchanged. (dynamic=True@392 in fact also interpolates pos_embed 32×32→28×28; we only
    make it static at construction.)
    """
    import math

    import timm  # local import: _quant_utils does not import timm at module level

    clean = backbone.lower().replace('_tv', '').replace('_timm', '')
    # global_pool MUST match _load_timm: DINOv2 → 'token' ([CLS]); eva02 (a different ViT) → 'avg'.
    # Wrong pool → different features → wrong mAP. (Tier-1: the fix was extended to vit_dinov2 patch14.)
    global_pool = 'token' if 'dinov2' in clean else 'avg'
    orig_sd = model.features.state_dict()
    new_bb = timm.create_model(clean, pretrained=False, num_classes=0,
                               global_pool=global_pool, dynamic_img_size=False,
                               img_size=img_size)
    new_sd = new_bb.state_dict()

    matched = {}
    for k, v in orig_sd.items():
        if k not in new_sd:
            continue
        if new_sd[k].shape == v.shape:
            matched[k] = v
        elif 'pos_embed' in k:
            # the grid changes with img_size → resample pos_embed (the probe confirmed the shapes match)
            try:
                from timm.layers import resample_abs_pos_embed
                tgt = new_sd[k]
                num_prefix = max(0, tgt.shape[1] - round(math.isqrt(tgt.shape[1])) ** 2)
                side = round(math.isqrt(tgt.shape[1] - num_prefix))
                pe = resample_abs_pos_embed(v, (side, side), num_prefix_tokens=num_prefix)
                if pe.shape == tgt.shape:
                    matched[k] = pe
            except Exception:  # noqa: BLE001
                pass  # leave this key missing → load_state_dict(strict=False) skips it
    new_bb.load_state_dict(matched, strict=False)
    new_bb.eval()
    model.features = new_bb


def export_fp32_onnx(
    backbone: str,
    weight_path: str,
    onnx_out_path: str,
    num_classes: int,
    img_size: int = 384,
    pooling: str = 'gem',
    device: Optional[torch.device] = None,
    fuse_branches: bool = False,
) -> str:
    """Load .pth → export ONNX FP32. Returns the path, to chain with quantize_static.
    where-teacher-acts: fuse_branches=True folds RepVGG's training branches into single convolutions
    (models/reparam.py) after loading, so the deployed graph is the fused one; default unchanged."""
    silence_benign_noise()  # silence the HF Hub warning (model creation) + TracerWarning (onnx export)
    device = device or torch.device('cpu')
    os.makedirs(os.path.dirname(onnx_out_path), exist_ok=True)

    model = PillRetrievalModel(num_classes=num_classes,
                               backbone_type=backbone,
                               pooling_type=pooling)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if fuse_branches:
        from models.reparam import fuse_repvgg_branches
        if fuse_repvgg_branches(model) == 0:
            raise ValueError(f'fuse_branches=True but {backbone} has no multi-branch RepVGG block')

    # ONNX-export fix for ViT dynamic_img_size (eva02 ×2 + vit_dinov2 ×3 = 5 backbones):
    # rebuild the backbone statically so it exports cleanly. Gated, so the other 32 backbones do NOT
    # take this branch and their export path is exactly as before. See _make_eva02_export_safe()
    # (probe config B_static, cos≈1.0).
    if 'eva02' in backbone.lower() or 'dinov2' in backbone.lower():
        _make_eva02_export_safe(model, backbone, img_size)

    dummy = torch.randn(1, 3, img_size, img_size)

    # EVA02/ViT workaround: timm resizes the positional embedding with
    # F.interpolate(mode='bicubic', antialias=True), which produces the op
    # aten::_upsample_bicubic2d_aa that CANNOT be exported at opset 14. Temporarily force
    # antialias=False ONLY during export (plain bicubic exports fine). The difference is only in
    # the pos-embed resize step (very small) and does not touch the main feature path. The model
    # wrapper's bilinear resize already uses antialias=False, so it is unchanged.
    import torch.nn.functional as _Fnn
    _orig_interpolate = _Fnn.interpolate

    def _interpolate_no_aa(*a, **k):
        if k.get('antialias', False):
            k = {**k, 'antialias': False}
        return _orig_interpolate(*a, **k)

    _Fnn.interpolate = _interpolate_no_aa
    # Pin the TorchScript exporter. torch >= 2.9 defaults torch.onnx.export to the
    # dynamo exporter (needs onnxscript, different graph); the server's torch 2.5 used TorchScript.
    # Passing dynamo=False only when the parameter exists keeps every torch version on ONE graph.
    import inspect as _inspect
    _extra = {'dynamo': False} if 'dynamo' in _inspect.signature(torch.onnx.export).parameters else {}
    try:
        torch.onnx.export(
            model, dummy, onnx_out_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes={INPUT_NAMES[0]: {0: 'batch_size'},
                          OUTPUT_NAMES[0]: {0: 'batch_size'}},
            **_extra,
        )
    finally:
        _Fnn.interpolate = _orig_interpolate

    del model
    gc.collect()
    return onnx_out_path


# ============================================================
# 4. INT8 PTQ WRAPPERS
# ============================================================

_CALIB_NAMES = ('minmax', 'entropy', 'percentile')


# where-teacher-acts: outcome of the ONNX Runtime pre-processing in the last run_ptq call
# (ok: the pre-processed graph was quantized; otherwise error says why the raw graph was)
LAST_PREPROCESS = {}


def run_ptq(
    onnx_fp32_path: str,
    onnx_int8_path: str,
    calibration_loader: DataLoader,
    calibration: str = 'minmax',
    num_batches=None,
    per_channel: bool = True,
    nodes_to_exclude=None,
    activation_symmetric: bool = False,
    extra_options=None,
    op_types_to_quantize=None,
    keep_preprocessed=None,
) -> str:
    """Static quantization of an FP32 ONNX model → INT8.

    Args:
        calibration: 'minmax', 'entropy', or 'percentile' (the study design requires minmax and
            entropy; 'percentile' was added for the activation-side experiment, which clips
            activation outliers).
        per_channel: True (default): per-channel WEIGHT quantization.
            Important for depthwise convolutions (GhostNet/MobileNet). (ORT static quantization does NOT
            support per-channel ACTIVATION quantization; activations are always per-tensor.)
        num_batches: None (default) = consume the WHOLE calibration_loader. The number of
            calibration images is controlled through build_calibration_loader(num_samples=...).
        nodes_to_exclude: list of node names kept in FP32 (mixed-precision PTQ, used for the
            mixed-precision experiment that leaves the sensitive layers unquantized).
            None/[] = quantize everything (default).
        activation_symmetric: True → symmetric activations (QUInt8 made symmetric around 0).
            Default False = asymmetric (the main setting). Activation-side control for the
            activation-side experiment.
        extra_options: dict of options passed straight to quantize_static (e.g.
            {'CalibPercentile': 99.999} for percentile). None = add nothing (the main setting).
        op_types_to_quantize: restricts the set of quantized op types (e.g. ['Conv','MatMul','Gemm']).
            None = the ORT default (also quantizes Add/Mul/Sigmoid/...). Used for the COVERAGE
            experiment: the training-time simulation covers only the Conv/Linear tensors, so this asks
            "what if only the part that was trained is compressed". NOT a deployment
            protocol (the model is then a mix of INT8 and FP32).
        keep_preprocessed: where-teacher-acts mechanism analysis. A path: the FP32 graph that was actually
            quantized (ONNX Runtime's pre-processed graph, or the raw graph when pre-processing failed)
            is copied there before the temporary files are removed, so its tensors can be compared
            with the quantized ones under the deployed scales. None (default) keeps the old behaviour.
    """
    if calibration not in _CALIB_NAMES:
        raise ValueError(f"Unknown calibration {calibration!r}; "
                         f"choose 'minmax', 'entropy' or 'percentile'.")
    os.makedirs(os.path.dirname(onnx_int8_path), exist_ok=True)

    ort_mod = _import_ort_quant()
    method_map = {
        'minmax':     ort_mod['CalibrationMethod'].MinMax,
        'entropy':    ort_mod['CalibrationMethod'].Entropy,
        'percentile': ort_mod['CalibrationMethod'].Percentile,
    }
    # Merge activation-symmetric flag into extra_options (backward-compatible: None→{}).
    eo = dict(extra_options) if extra_options else {}
    if activation_symmetric:
        eo['ActivationSymmetric'] = True

    # where-teacher-acts: ORT's pre-processing writes sym_shape_infer_temp.onnx into the working
    # directory and the cleanup below deletes it, so two PTQ processes sharing a directory can
    # delete or overwrite each other's temporary graph. Serialize them on a lock file there.
    import fcntl as _fcntl
    with open(os.path.join(os.getcwd(), '.ptq.lock'), 'a') as _lockf:
        _fcntl.flock(_lockf, _fcntl.LOCK_EX)
        try:
            # Pre-process (symbolic shape inference + graph opt) before quantizing.
            # ORT recommends this step: it slightly improves INT8 quality and silences the warning
            # "Please consider pre-processing". Best-effort: on error, fall back to the raw FP32 graph.
            model_for_quant = onnx_fp32_path
            pp_path = onnx_int8_path + '.pp.onnx'
            LAST_PREPROCESS.clear()
            LAST_PREPROCESS.update(ok=False, error='quant_pre_process not available')
            if ort_mod.get('quant_pre_process') is not None:
                try:
                    with quiet_root_logs():
                        ort_mod['quant_pre_process'](onnx_fp32_path, pp_path)
                    model_for_quant = pp_path
                    LAST_PREPROCESS.update(ok=True, error='')
                except Exception as _pp_exc:
                    model_for_quant = onnx_fp32_path  # fallback if pre_process fails
                    # where-teacher-acts: the fallback quantizes a different graph; say so and let the
                    # caller record it (LAST_PREPROCESS) instead of hiding it
                    LAST_PREPROCESS.update(ok=False, error=f'{type(_pp_exc).__name__}: {_pp_exc}'[:300])
                    print(f'[run_ptq] WARNING: quant_pre_process failed ({LAST_PREPROCESS["error"]}); '
                          f'quantizing the raw graph {os.path.basename(onnx_fp32_path)}', flush=True)

            reader = make_calibration_reader(calibration_loader,
                                             input_name=INPUT_NAMES[0],
                                             num_batches=num_batches)
            try:
                # quiet_root_logs: ORT logs 'Axis 1 out-of-range for ... rank 1' through the root
                # logger (per-channel skips 1-D weights, which is correct and harmless). Raise
                # root→ERROR for a clean log.
                with quiet_root_logs():
                    ort_mod['quantize_static'](
                        model_input=model_for_quant,
                        model_output=onnx_int8_path,
                        calibration_data_reader=reader,
                        quant_format=ort_mod['QuantFormat'].QDQ,
                        weight_type=ort_mod['QuantType'].QInt8,
                        activation_type=ort_mod['QuantType'].QUInt8,
                        calibrate_method=method_map[calibration],
                        per_channel=per_channel,
                        nodes_to_exclude=list(nodes_to_exclude) if nodes_to_exclude else None,
                        op_types_to_quantize=list(op_types_to_quantize) if op_types_to_quantize else None,
                        extra_options=eo if eo else None,
                    )
            finally:
                if keep_preprocessed:
                    import shutil as _shutil
                    if os.path.exists(model_for_quant + '.data'):
                        raise RuntimeError(f'{model_for_quant} keeps external data; copying the graph alone '
                                           f'would break its reference')
                    _shutil.copyfile(model_for_quant, keep_preprocessed)
                # Clean up the pre_process temporary files: <pp>.onnx, the external data <pp>.onnx.data,
                # and sym_shape_infer_temp.onnx (created in the CWD by ORT symbolic shape inference).
                for tmp in (pp_path, pp_path + '.data', 'sym_shape_infer_temp.onnx'):
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass
        finally:
            _fcntl.flock(_lockf, _fcntl.LOCK_UN)
    return onnx_int8_path


def run_ptq_minmax(*args, **kwargs):
    return run_ptq(*args, calibration='minmax', **kwargs)


def run_ptq_entropy(*args, **kwargs):
    return run_ptq(*args, calibration='entropy', **kwargs)


# ============================================================
# 4b. MIXED-PRECISION NODE SELECTION
# ============================================================

def categorize_quant_nodes(onnx_fp32_path: str) -> dict:
    """Classify the Conv/MatMul/Gemm nodes of an FP32 ONNX model, to choose the layers kept in FP32
    in the mixed-precision PTQ experiment.

    Returns dict:
        'first'      : [name of the first Conv/MatMul/Gemm node] (the input layer, sensitive to quantization)
        'last'       : [names of the last 1-2 nodes] (projection/embedding layers, sensitive to quantization)
        'depthwise'  : [names of all depthwise Convs] (weight dims[1]==1 → in_ch/group==1)
        'all'        : [names of all Conv/MatMul/Gemm nodes]
    Depthwise is detected from the weight initializer shape (dims[1]==1), which is more reliable
    than reading the 'group' attribute.
    """
    import onnx  # lazy
    model = onnx.load(onnx_fp32_path)
    g = model.graph
    init_dims = {i.name: list(i.dims) for i in g.initializer}
    quant_ops = {'Conv', 'MatMul', 'Gemm'}
    seq, depthwise = [], []
    for node in g.node:
        if node.op_type not in quant_ops:
            continue
        nm = node.name or f'{node.op_type}_{len(seq)}'
        seq.append(nm)
        if node.op_type == 'Conv':
            # the weight is input[1]; depthwise ⇔ dims = [out_C, in_C/group=1, kH, kW]
            for inp in node.input[1:2]:
                d = init_dims.get(inp)
                if d and len(d) == 4 and d[1] == 1:
                    depthwise.append(nm)
    return {
        'first':     seq[:1],
        'last':      seq[-2:] if len(seq) >= 2 else seq[-1:],
        'depthwise': depthwise,
        'all':       seq,
    }


# ============================================================
# 5. SIZE METRICS
# ============================================================

def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def compression_ratio(fp32_path: str, int8_path: str) -> float:
    return model_size_mb(fp32_path) / model_size_mb(int8_path)
