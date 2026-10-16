"""
Benchmark utilities (library).

Provides the deployment-metric measurements required by the study design:
- latency: median ms/inference, single-thread CPU
- peak memory: MB during inference (via resource/tracemalloc)
- model size: MB from the ONNX file
- throughput: FPS = 1000 / latency

Supports 2 platforms: desktop x86 (default) + ARM mobile (via a session option).
Extracted from an earlier version of this code.
"""
from __future__ import annotations

import gc
import os
import time
import tracemalloc
from typing import Optional

import numpy as np

def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required to run the benchmark. pip install onnxruntime"
        ) from e
    return ort


# ============================================================
# 1. ONNX SESSION BUILDER
# ============================================================

def make_session(
    onnx_path: str,
    intra_threads: int = 1,
    inter_threads: int = 1,
    enable_optimization: bool = True,
    provider: str = 'CPUExecutionProvider',
):
    """Create a single-threaded InferenceSession (simulates an edge device)."""
    ort = _import_ort()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    if enable_optimization:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, opts, providers=[provider])


# ============================================================
# 2. LATENCY
# ============================================================

def measure_onnx_latency_ms(
    onnx_path: str,
    img_size: int = 384,
    warmup: int = 50,
    iterations: int = 300,
    intra_threads: int = 1,
    provider: str = 'CPUExecutionProvider',
) -> dict:
    """Measure median/mean/p95 latency (ms) for a single input image.

    Returns: {median, mean, std, p95, min, max}
    """
    session = make_session(onnx_path,
                           intra_threads=intra_threads,
                           provider=provider)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    del session
    gc.collect()

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        'median': float(np.median(arr)),
        'mean':   float(arr.mean()),
        'std':    float(arr.std(ddof=1)),
        'p95':    float(np.percentile(arr, 95)),
        'min':    float(arr.min()),
        'max':    float(arr.max()),
    }


def fps_from_latency(median_ms: float) -> float:
    return 1000.0 / median_ms if median_ms > 0 else 0.0


# ============================================================
# 3. PEAK MEMORY
# ============================================================

def measure_peak_memory_mb(
    onnx_path: str,
    img_size: int = 384,
    iterations: int = 30,
    intra_threads: int = 1,
    provider: str = 'CPUExecutionProvider',
) -> float:
    """Measure peak Python-level memory (MB) during inference.

    Note: tracemalloc captures Python allocations; native (C++ ORT) memory
    is not included in this number. To measure native memory, use `psutil.Process().memory_info().rss`
    before and after (see measure_peak_rss_mb).
    """
    tracemalloc.start()
    session = make_session(onnx_path,
                           intra_threads=intra_threads,
                           provider=provider)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    for _ in range(iterations):
        _ = session.run(None, {input_name: dummy})

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del session
    gc.collect()
    return peak / (1024 * 1024)


def measure_peak_rss_mb(
    onnx_path: str,
    img_size: int = 384,
    iterations: int = 30,
    intra_threads: int = 1,
    provider: str = 'CPUExecutionProvider',
) -> Optional[float]:
    """Measure peak RSS (resident set size, MB) via psutil. Includes native ONNX memory.

    Returns None if psutil is not installed.
    """
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss

    session = make_session(onnx_path,
                           intra_threads=intra_threads,
                           provider=provider)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    peak_rss = baseline_rss
    for _ in range(iterations):
        _ = session.run(None, {input_name: dummy})
        cur = proc.memory_info().rss
        if cur > peak_rss:
            peak_rss = cur

    del session
    gc.collect()
    return (peak_rss - baseline_rss) / (1024 * 1024)


# ============================================================
# 4. MODEL SIZE
# ============================================================

def get_model_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return float('nan')
    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# 5. ALL-IN-ONE BENCHMARK
# ============================================================

def full_benchmark_row(
    label: str,
    onnx_path: str,
    img_size: int = 384,
    warmup: int = 50,
    iterations: int = 300,
    mem_iterations: int = 30,
) -> dict:
    """Measure all deployment metrics for 1 ONNX file → 1 CSV row.

    label: free-form identifier (e.g. 'resnet18_tv_int8_entropy').
    """
    size_mb = get_model_size_mb(onnx_path)
    if not os.path.exists(onnx_path):
        return {
            'label': label,
            'onnx_path': onnx_path,
            'size_mb': size_mb,
            'latency_median_ms': float('nan'),
            'latency_p95_ms': float('nan'),
            'fps': float('nan'),
            'peak_rss_mb': float('nan'),
            'status': 'MISSING',
        }

    try:
        lat = measure_onnx_latency_ms(onnx_path,
                                      img_size=img_size,
                                      warmup=warmup,
                                      iterations=iterations)
        rss = measure_peak_rss_mb(onnx_path,
                                  img_size=img_size,
                                  iterations=mem_iterations)
    except Exception as exc:  # ONNX session error etc.
        return {
            'label': label,
            'onnx_path': onnx_path,
            'size_mb': size_mb,
            'latency_median_ms': float('nan'),
            'latency_p95_ms': float('nan'),
            'fps': float('nan'),
            'peak_rss_mb': float('nan'),
            'status': f'ERROR: {exc!s}',
        }

    return {
        'label': label,
        'onnx_path': onnx_path,
        'size_mb': size_mb,
        'latency_median_ms': lat['median'],
        'latency_p95_ms': lat['p95'],
        'fps': fps_from_latency(lat['median']),
        'peak_rss_mb': rss if rss is not None else float('nan'),
        'status': 'OK',
    }
