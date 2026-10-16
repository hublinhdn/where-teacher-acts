"""
QAKD: simulated INT8 for the STUDENT inside the distillation loop (library).

What is simulated, and why exactly this:
  * weights   : per-channel SYMMETRIC int8 on every Conv2d / Linear of `student.features` and
                `student.fc_projection`, computed on the BatchNorm-FOLDED weight whenever the conv
                is immediately followed by a BatchNorm (conv->bn adjacency in `named_children`
                order, which covers timm EfficientNet, torchvision MobileNetV3/ResNet, and the
                Linear->BatchNorm1d bottleneck head). ONNX Runtime folds BN into Conv before
                quantizing, so simulating the UNFOLDED weight would simulate the wrong tensor.
  * activations: per-tensor ASYMMETRIC uint8 at the INPUT of every wrapped op and (optionally) at
                its pre-activation OUTPUT, ranges tracked by an EMA of batch min/max (a smooth
                stand-in for ORT MinMax calibration; zero is always inside the range, as in ORT).
  * gradients : straight-through estimator (fake-quant is identity in the backward pass); weights
                and BN statistics stay FP32.

Phases (the trainer drives them): phase A = the inherited FP32 recipe untouched; phase B (from
`q_start_epoch`) = fake-quant ON in APPROXIMATE-fold mode (weights quantized as folded with
running statistics, i.e. the deployed tensor, while BatchNorm keeps normalizing with batch
statistics so training stays stable; PyTorch QAT ConvBn semantics), best-checkpoint tracking
RESET so the saved model is a quantization-aware one; final epochs (`q_freeze_last`) = BN
statistics frozen and folded EXACTLY. Why not freeze from the start: trained edge CNNs carry
dead BN channels (running_var ~1e-45, fold scale ~1e4); with frozen statistics any drift of such
a channel is amplified 1e4x and the run diverges to NaN (observed on EfficientNet-B1 in the
first launch). Deployment stays the inherited protocol: export FP32 ONNX -> ORT static PTQ (MinMax,
N=256, per-channel weights) -> INT8 mAP. Nothing about the deployment pipeline changes; only
training does. The saved checkpoint is a PLAIN PillRetrievalModel state_dict
(`clean_state_dict`), so it loads into the unmodified model class for export.

Not simulated (documented simplifications): residual Add / Mul quantization, GeM pooling, the
final BatchNorm1d head, SE Sigmoid/Mul. The paper reports REAL INT8 numbers from ORT, so these
only affect training-time fidelity, never the measurement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def _ste(x, xq):
    return x + (xq - x).detach()


class ActObserver(nn.Module):
    """Per-tensor asymmetric uint8 fake-quant with EMA min/max range tracking."""

    # NOTE: everything here is SYNC-FREE on purpose. A `bool(tensor)` / `if tensor:` forces a
    # GPU->CPU synchronisation; with ~230 observers per forward that made phase-B epochs 4x
    # slower (35 s vs 8 s on EfficientNet-B1). Range updates use torch.where, and the only
    # Python-side state is a bool set on the first call.

    def __init__(self, bits=8, momentum=0.99):
        super().__init__()
        self.bits = bits
        self.momentum = momentum
        self.enabled = False       # fake-quant on/off
        self.observe = True        # update EMA while training
        self._seen = False         # Python flag: at least one training batch observed
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    @torch.no_grad()
    def _update(self, x):
        if self.min_val.device != x.device:   # wrapped after .to(device): follow the data
            self.min_val = self.min_val.to(x.device)
            self.max_val = self.max_val.to(x.device)
        xd = x.detach()
        cur_min = xd.amin().float()
        cur_max = xd.amax().float()
        finite = torch.isfinite(cur_min) & torch.isfinite(cur_max)
        m = self.momentum
        # EMA when a range exists, plain copy when the stored range is still +-inf; a non-finite
        # batch leaves the stored range untouched (it must never poison the observer).
        new_min = torch.where(torch.isfinite(self.min_val), m * self.min_val + (1 - m) * cur_min, cur_min)
        new_max = torch.where(torch.isfinite(self.max_val), m * self.max_val + (1 - m) * cur_max, cur_max)
        self.min_val.copy_(torch.where(finite, new_min, self.min_val))
        self.max_val.copy_(torch.where(finite, new_max, self.max_val))
        self._seen = True

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training and self.observe:
            self._update(x)
        if not self._seen:                 # never observed -> pass-through
            return x
        qmax = 2 ** self.bits - 1
        lo = torch.clamp(self.min_val, max=0.0)   # ORT MinMax keeps 0 inside the range
        hi = torch.clamp(self.max_val, min=0.0)
        scale = torch.clamp((hi - lo) / qmax, min=1e-8)
        zp = torch.round(-lo / scale)
        xf = x.float()
        q = torch.clamp(torch.round(xf / scale) + zp, 0, qmax)
        xq = ((q - zp) * scale).to(x.dtype)
        return _ste(x, xq)


class LSQObserver(ActObserver):
    """Per-tensor asymmetric uint8 fake-quant with LEARNABLE clipping range (LSQ+ / PACT family).

    Warm-up (`warmup_steps` training batches): behaves exactly like ActObserver (EMA min/max),
    then freezes that range as (base_lo <= 0 <= base_hi). Afterwards the range is
        lo = base_lo * exp(theta_lo),   hi = base_hi * exp(theta_hi)
    with theta_* learnable (log-space so Adam can move a 1400-wide range with a sane LR).
    Quantization is written so autograd yields the LSQ gradients: x is clipped to [lo, hi]
    (gradient 1 inside, 0 outside; clipped elements push lo/hi outward), the rounding uses a
    straight-through estimator, and the step size s = (hi - lo)/qmax carries the rounding-error
    gradient onto theta (elements far from their grid point pull the range inward). The task
    loss therefore balances clipping error against resolution, which plain STE cannot do.

    `clip_only` = deployment mode: apply the learned clip, no rounding. Exporting the wrapped
    model in this mode puts Clip nodes in the FP32 ONNX graph, so ONNX Runtime's MinMax
    calibration sees exactly the ranges the student was trained for (no QDQ overrides needed).
    The clip MUST export as ONE `Clip` node, not a `Min`/`Max` pair: ONNX Runtime fuses
    Conv+Clip (the Relu6 pattern) and then places the quantization point AFTER the clip, which is
    the whole point. The Min/Max spelling measured 1.2x to 2.1x slower at INT8 (470 unfused nodes
    on EfficientNet-B1) and left an unclipped quantization point on every Conv output."""

    def __init__(self, bits=8, momentum=0.99, warmup_steps=20):
        super().__init__(bits=bits, momentum=momentum)
        self.warmup_steps = int(warmup_steps)
        self.steps = 0                       # Python counter (no sync)
        self.clip_only = False
        self.theta_lo = nn.Parameter(torch.zeros(()))
        self.theta_hi = nn.Parameter(torch.zeros(()))
        self.register_buffer("base_lo", torch.tensor(0.0))
        self.register_buffer("base_hi", torch.tensor(1.0))
        self.register_buffer("frozen_base", torch.tensor(0, dtype=torch.uint8))

    @property
    def learned(self):
        return self.steps >= self.warmup_steps

    def range(self):
        lo = self.base_lo * torch.exp(self.theta_lo)
        hi = self.base_hi * torch.exp(self.theta_hi)
        return lo, hi

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training and self.observe and not self.learned:
            self._update(x)
            self.steps += 1
            if self.steps >= self.warmup_steps:
                with torch.no_grad():
                    self.base_lo.copy_(torch.clamp(self.min_val, max=0.0))
                    self.base_hi.copy_(torch.clamp(self.max_val, min=0.0))
                    self.frozen_base.fill_(1)
        if not self._seen:
            return x
        if not self.learned:                 # still in warm-up: fixed EMA range, plain fake-quant
            return self._fixed_quant(x)
        lo, hi = self.range()
        xf = x.float()
        if self.clip_only:
            # deployment view: ONE Clip node (torch.clamp with 0-dim tensor bounds), never a
            # Min/Max pair, so ORT can fuse it into the preceding Conv. See the class docstring.
            return torch.clamp(xf, min=lo, max=hi).to(x.dtype)
        # Same integer-zero-point arithmetic and the same op order as the fixed-range observer
        # (round, then clamp in the integer domain), so with theta = 0 both paths coincide.
        # Straight-through estimators on the two roundings leave the LSQ gradients intact:
        # d xq / d s = round(r) - r for interior elements, and saturated elements pull (lo, hi).
        qmax = 2 ** self.bits - 1
        s = torch.clamp((hi - lo) / qmax, min=1e-8)
        zpf = -lo / s
        zp = zpf + (torch.round(zpf) - zpf).detach()
        k = xf / s
        kq = k + (torch.round(k) - k).detach()          # round FIRST (as the fixed path), then add zp:
        rq = torch.clamp(kq + zp, 0, qmax)              # adding ~1e2 before rounding flips float ties
        xq = (rq - zp) * s
        return xq.to(x.dtype)

    def _fixed_quant(self, x):
        qmax = 2 ** self.bits - 1
        lo = torch.clamp(self.min_val, max=0.0)
        hi = torch.clamp(self.max_val, min=0.0)
        scale = torch.clamp((hi - lo) / qmax, min=1e-8)
        zp = torch.round(-lo / scale)
        xf = x.float()
        q = torch.clamp(torch.round(xf / scale) + zp, 0, qmax)
        xq = ((q - zp) * scale).to(x.dtype)
        return _ste(x, xq)


def fake_quant_weight_per_channel(w, bits=8):
    """Symmetric per-output-channel int8 (range [-127,127], like ORT QInt8 per_channel)."""
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    flat = wf.reshape(wf.shape[0], -1)
    amax = flat.abs().max(dim=1).values.clamp(min=1e-8)
    scale = (amax / qmax).reshape(-1, *([1] * (wf.dim() - 1)))
    wq = torch.clamp(torch.round(wf / scale), -qmax, qmax) * scale
    return _ste(wf, wq).to(w.dtype)


class BNShim(nn.Module):
    """Holds the original BatchNorm that follows a wrapped Conv/Linear.

    Two phase-B modes (PyTorch QAT ConvBn semantics):
      approximate (frozen=False, bypass=False): the preceding QWrap fake-quantizes the weight
          folded with RUNNING statistics (the tensor ORT will deploy) but returns an UN-normalized
          conv output; this shim then normalizes with BATCH statistics in train mode (running
          stats keep tracking) and running stats in eval mode. Training dynamics stay
          BN-normalized, so dead channels (running_var ~1e-45, fold scale ~1e4) cannot explode.
      exact (frozen=True, bypass=True): statistics frozen, the QWrap already applied the exact
          fold; this shim only quantizes/activates. Used for the last epochs and for tests.
    The output observer sits here (post-normalization, pre-activation) for folded pairs, which
    is the tensor ORT quantizes after folding. Fused dropout/activation of timm BatchNormAct2d
    run after it."""

    def __init__(self, bn):
        super().__init__()
        self.bn = bn
        self.frozen = False
        self.bypass = False
        self.qout = None

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.bn.eval()
        return self

    def _norm(self, x):
        """BatchNorm normalization + affine only (no fused act), honoring train/eval/frozen."""
        bn = self.bn
        training = bn.training and not self.frozen
        momentum = bn.momentum if bn.momentum is not None else 0.1
        if training and bn.track_running_stats and bn.num_batches_tracked is not None:
            bn.num_batches_tracked.add_(1)
        use_stats = (not training) or bn.track_running_stats
        return F.batch_norm(x, bn.running_mean if use_stats else None,
                            bn.running_var if use_stats else None,
                            bn.weight, bn.bias, training, momentum, bn.eps)

    def forward(self, x):
        y = x if self.bypass else self._norm(x)
        if self.qout is not None:
            y = self.qout(y)
        if hasattr(self.bn, "drop"):
            y = self.bn.drop(y)
        if hasattr(self.bn, "act"):
            y = self.bn.act(y)
        return y


class QWrap(nn.Module):
    """Fake-quant wrapper around one Conv2d or Linear, with optional folded BatchNorm."""

    def __init__(self, m, bn_shim=None, w_bits=8, a_bits=8, quant_output=True, obs_factory=None):
        super().__init__()
        assert isinstance(m, (nn.Conv2d, nn.Linear)), type(m)
        self.m = m
        self.bn_shim = [bn_shim]          # plain list: reference, NOT a registered child
        self.w_bits = w_bits
        dev = m.weight.device             # observers live where the wrapped op lives
        mk = obs_factory or (lambda: ActObserver(a_bits))
        self.qin = mk().to(dev)
        # output observer: owned here for LONE ops; for folded pairs it lives in the BNShim
        # (post-normalization), so create it there instead.
        self.qout = None
        if quant_output:
            if bn_shim is None:
                self.qout = mk().to(dev)
            else:
                bn_shim.qout = mk().to(dev)
        self.enabled = False
        self.fold_only = False            # sanity mode: fold BN but skip rounding

    @property
    def bn(self):
        return self.bn_shim[0]

    def _op(self, x, w, b):
        if isinstance(self.m, nn.Conv2d):
            return F.conv2d(x, w, b, self.m.stride, self.m.padding, self.m.dilation, self.m.groups)
        return F.linear(x, w, b)

    def _bn_scale(self):
        bn = self.bn.bn
        inv = torch.rsqrt(bn.running_var.float() + bn.eps)
        gamma = bn.weight.float() if bn.weight is not None else torch.ones_like(inv)
        return gamma * inv                                     # gamma / sqrt(running_var + eps)

    def _folded(self):
        """Exact fold with running statistics (deployment tensor): (w_f, b_f)."""
        w, b = self.m.weight, self.m.bias
        bn = self.bn.bn
        scale = self._bn_scale()
        beta = bn.bias.float() if bn.bias is not None else torch.zeros_like(scale)
        w_f = w.float() * scale.reshape(-1, *([1] * (w.dim() - 1)))
        b0 = b.float() if b is not None else torch.zeros_like(bn.running_mean)
        b_f = beta + (b0 - bn.running_mean.float()) * scale
        return w_f, b_f

    def _fq(self, w):
        return w if self.fold_only else fake_quant_weight_per_channel(w, self.w_bits)

    def forward(self, x):
        if not self.enabled:
            return self.m(x)
        shim = self.bn
        if shim is None:
            # lone op (no BN): plain weight fake-quant, output observer here
            x = self.qin(x)
            y = self._op(x, self._fq(self.m.weight).to(x.dtype), self.m.bias)
            return self.qout(y) if self.qout is not None else y
        if not shim.bypass:
            # APPROXIMATE fold (training default): quantize the weight as ORT will see it (folded
            # with running stats), then undo the scale so the following BN normalizes with batch
            # statistics. Bias goes in before BN, as in the original graph.
            # By linearity conv(x, fq(w*s))/s == conv(x, fq(w*s)/s): quantize on the folded grid,
            # then unscale the WEIGHT (fp32, exact per channel) so the op itself sees normal
            # magnitudes and stays fp16-safe under autocast; the bias enters as in the original graph.
            x = self.qin(x)
            scale = self._bn_scale().reshape(-1, *([1] * (self.m.weight.dim() - 1)))
            w_s = self._fq(self.m.weight.float() * scale) / scale
            return self._op(x, w_s.to(x.dtype), self.m.bias)
        # EXACT fold (statistics frozen): FP32 with autocast OFF because folded weights of dead
        # channels (running_var ~1e-45) reach |w_f| ~ 1e3 and overflow fp16; ORT accumulates in
        # int32 and never overflows, so FP32 is the faithful simulation.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.qin(x.float())
            w_f, b_f = self._folded()
            y = self._op(x, self._fq(w_f), b_f)
        return y


# ----------------------------------------------------------------------------- preparation
def _wrap_container(container, w_bits, a_bits, quant_output, stats, obs_factory=None):
    """Wrap Conv2d/Linear children of `container` (recursively), folding an immediately
    following BatchNorm sibling. Returns nothing; mutates in place."""
    kids = list(container.named_children())
    names = [k for k, _ in kids]
    for i, (name, child) in enumerate(kids):
        if isinstance(child, (QWrap, BNShim)):
            continue
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            shim = None
            if i + 1 < len(kids) and isinstance(kids[i + 1][1], _BatchNorm):
                shim = BNShim(kids[i + 1][1])
                setattr(container, names[i + 1], shim)
                stats["folded"] += 1
            else:
                stats["lone"] += 1
            setattr(container, name, QWrap(child, shim, w_bits, a_bits, quant_output, obs_factory))
        elif isinstance(child, _BatchNorm):
            continue  # a BN not preceded by a wrapped op stays as is (normal FP32 module)
        else:
            _wrap_container(child, w_bits, a_bits, quant_output, stats, obs_factory)


def prepare_fakequant(student, w_bits=8, a_bits=8, quant_output=True, learnable_clip=False,
                      warmup_steps=20):
    """Wrap `student.features` and `student.fc_projection` in place. Idempotent-unsafe: once.
    learnable_clip=True -> LSQObserver (learned clipping range) instead of ActObserver."""
    stats = {"folded": 0, "lone": 0}
    factory = (lambda: LSQObserver(a_bits, warmup_steps=warmup_steps)) if learnable_clip else None
    _wrap_container(student.features, w_bits, a_bits, quant_output, stats, factory)
    fp = student.fc_projection
    if isinstance(fp, (nn.Conv2d, nn.Linear)):
        student.fc_projection = QWrap(fp, None, w_bits, a_bits, quant_output, factory)
        stats["lone"] += 1
    else:
        _wrap_container(fp, w_bits, a_bits, quant_output, stats, factory)
    stats["wrapped"] = stats["folded"] + stats["lone"]
    stats["learnable_clip"] = bool(learnable_clip)
    return stats


def _iter(model, cls):
    for m in model.modules():
        if isinstance(m, cls):
            yield m


def enable_quant(model, enabled=True, fold_only=False, exact=False):
    """Phase B switch. exact=False (training default): approximate fold, BN keeps normalizing
    with batch statistics. exact=True: statistics frozen and folded exactly (tests / final epochs;
    see freeze_bn_fold)."""
    for shim in _iter(model, BNShim):
        shim.frozen = enabled and exact
        shim.bypass = enabled and exact
        if shim.frozen:
            shim.bn.eval()
        elif not enabled:
            shim.bn.train(model.training)
    for q in _iter(model, QWrap):
        q.enabled = enabled
        q.fold_only = fold_only
    for obs in _iter(model, ActObserver):
        obs.enabled = enabled and not fold_only


def freeze_bn_fold(model):
    """Final-epochs switch: freeze BN statistics and fold exactly (the deployed arithmetic)."""
    for shim in _iter(model, BNShim):
        shim.frozen = True
        shim.bypass = True
        shim.bn.eval()


def set_observe(model, observe=True):
    for obs in _iter(model, ActObserver):
        obs.observe = observe


# ----------------------------------------------------------------- learnable-clip helpers
def clip_params(model):
    """The learnable range parameters (theta_lo/theta_hi of every LSQObserver)."""
    return [p for o in _iter(model, LSQObserver) for p in (o.theta_lo, o.theta_hi)]


def clip_penalty(model):
    """Optional relative-range penalty mean(exp(2*theta)) (1.0 = no shrink). Default weight 0."""
    ts = [torch.exp(2 * o.theta_hi) + torch.exp(2 * o.theta_lo) for o in _iter(model, LSQObserver)]
    return (torch.stack(ts).mean() / 2) if ts else torch.tensor(0.0)


def set_deploy_clip(model, on=True):
    """Deployment view for export: no weight rounding, BN as in the original graph (approximate
    fold in eval == original), every LSQ observer applies its learned clip and nothing else.
    Exporting the model in this state yields an FP32 graph with Clip nodes at the trained
    positions, which ORT then calibrates and quantizes with the standard MinMax protocol."""
    for shim in _iter(model, BNShim):
        shim.frozen = False
        shim.bypass = False
    for q in _iter(model, QWrap):
        q.enabled = on
        q.fold_only = on
    for obs in _iter(model, ActObserver):
        obs.enabled = on
        if isinstance(obs, LSQObserver):
            obs.clip_only = on


def observer_state(model):
    """Sidecar payload: learned ranges + EMA stats of every observer, keyed by module name."""
    st = {}
    for n, o in model.named_modules():
        if isinstance(o, ActObserver):
            d = {"min_val": o.min_val.detach().cpu(), "max_val": o.max_val.detach().cpu(),
                 "seen": bool(o._seen)}
            if isinstance(o, LSQObserver):
                d.update({"theta_lo": o.theta_lo.detach().cpu(), "theta_hi": o.theta_hi.detach().cpu(),
                          "base_lo": o.base_lo.detach().cpu(), "base_hi": o.base_hi.detach().cpu(),
                          "steps": int(o.steps), "warmup_steps": int(o.warmup_steps)})
            st[n] = d
    return st


@torch.no_grad()
def load_observer_state(model, st):
    n_loaded = 0
    for n, o in model.named_modules():
        if isinstance(o, ActObserver) and n in st:
            d = st[n]
            dev = o.min_val.device
            o.min_val = d["min_val"].to(dev)
            o.max_val = d["max_val"].to(dev)
            o._seen = bool(d.get("seen", True))
            if isinstance(o, LSQObserver) and "theta_hi" in d:
                o.theta_lo.copy_(d["theta_lo"].to(dev))
                o.theta_hi.copy_(d["theta_hi"].to(dev))
                o.base_lo.copy_(d["base_lo"].to(dev))
                o.base_hi.copy_(d["base_hi"].to(dev))
                o.steps = int(d["steps"])
                o.warmup_steps = int(d.get("warmup_steps", o.warmup_steps))
                o.frozen_base.fill_(1 if o.steps >= o.warmup_steps else 0)
            n_loaded += 1
    return n_loaded


def clip_summary(model):
    """Mean/min/max of the learned relative range exp(theta_hi) over LSQ observers."""
    hs = [float(torch.exp(o.theta_hi)) for o in _iter(model, LSQObserver) if o.learned]
    if not hs:
        return {"lsq_learned": 0}
    return {"lsq_learned": len(hs), "rel_hi_mean": round(sum(hs) / len(hs), 3),
            "rel_hi_min": round(min(hs), 3), "rel_hi_max": round(max(hs), 3)}


def quant_summary(model):
    n_q = sum(1 for _ in _iter(model, QWrap))
    n_on = sum(1 for q in _iter(model, QWrap) if q.enabled)
    n_bn = sum(1 for _ in _iter(model, BNShim))
    n_frozen = sum(1 for s in _iter(model, BNShim) if s.frozen)
    n_seen = sum(1 for o in _iter(model, ActObserver) if not torch.isinf(o.min_val))
    return {"wrapped": n_q, "enabled": n_on, "folded_bn": n_bn, "bn_frozen": n_frozen,
            "observers_seen": n_seen}


def clean_state_dict(model):
    """State dict with the ORIGINAL PillRetrievalModel key names (wrapper segments removed,
    observer buffers dropped) so it loads strict=True into an unmodified model for export."""
    qnames = {n for n, m in model.named_modules() if isinstance(m, QWrap)}
    bnames = {n for n, m in model.named_modules() if isinstance(m, BNShim)}
    out = {}
    for k, v in model.state_dict().items():
        if ".qin." in k or ".qout." in k or k.startswith("qin.") or k.startswith("qout."):
            continue                         # observer buffers/params live in the sidecar
        nk = k
        for qn in qnames:
            pre = qn + ".m."
            if k.startswith(pre):
                nk = qn + "." + k[len(pre):]
                break
        else:
            for bn in bnames:
                pre = bn + ".bn."
                if k.startswith(pre):
                    nk = bn + "." + k[len(pre):]
                    break
        out[nk] = v.detach().cpu().clone()
    return out


def export_ranges(model):
    """Learned activation ranges (for an optional QDQ export path / diagnostics)."""
    rng = {}
    for n, m in model.named_modules():
        if isinstance(m, ActObserver) and not torch.isinf(m.min_val):
            rng[n] = (float(m.min_val), float(m.max_val))
    return rng
