# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention Benchmark.

Benchmarks implementations:
- naive: 3-pass attention (Warp baseline)
- flash_attn: official Flash Attention library (if installed)

Note: Tiled/SIMD kernels disabled due to wp.launch_tiled bug in Warp 1.12.0.dev0

Usage:
    python benchmark_flash_attention.py

To include official flash_attn comparison:
    pip install flash-attn --no-build-isolation
"""

import sys
import time
from statistics import mean

import numpy as np
import torch
import warp as wp

# Try to import official flash_attn library
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not installed. Install with: pip install flash-attn --no-build-isolation")

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from tile.example_tile_flash_attention import (
    get_naive_kernel,
    reference_attention,
)

IMPLEMENTATIONS = ["naive"]
if HAS_FLASH_ATTN:
    IMPLEMENTATIONS.append("flash_attn")

DISTRIBUTIONS = {
    "uniform": lambda shape, rng: rng.uniform(-1, 1, shape).astype(np.float32),
    "normal": lambda shape, rng: rng.standard_normal(shape).astype(np.float32),
    "extreme": lambda shape, rng: rng.uniform(10, 50, shape).astype(np.float32),
}

# Tolerance scales with reference magnitude: |err| < ATOL + RTOL * max(|ref|)
# Higher RTOL accounts for FP rounding differences in tree reduction vs sequential sum
RTOL = 5e-2
ATOL = 1e-5


def launch_kernel(impl: str, Q: wp.array, K: wp.array, V: wp.array, O: wp.array, sm_scale: float, head_dim: int):
    batch_heads, seq_len, _ = Q.shape
    if impl == "naive":
        kernel = get_naive_kernel(head_dim)
        wp.launch(kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])


def validate(result: np.ndarray, reference: np.ndarray) -> tuple[bool, float]:
    abs_diff = np.abs(result - reference)
    max_error = float(np.max(abs_diff))
    ref_scale = float(np.max(np.abs(reference)))
    tolerance = ATOL + RTOL * ref_scale
    passed = max_error < tolerance
    return passed, max_error


def benchmark(batch: int, heads: int, seq_len: int, impl: str, head_dim: int = 64, distribution: str = "uniform", warmup: int = 5, iterations: int = 20):
    batch_heads = batch * heads
    rng = np.random.default_rng(42)
    gen = DISTRIBUTIONS.get(distribution, DISTRIBUTIONS["uniform"])

    shape = (batch_heads, seq_len, head_dim)
    Q_np = gen(shape, rng)
    K_np = gen(shape, rng)
    V_np = gen(shape, rng)
    sm_scale = 1.0 / np.sqrt(head_dim)

    Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
    K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
    V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
    ref_out = reference_attention(Q_4d, K_4d, V_4d, sm_scale).reshape(shape)

    # Handle official flash_attn library separately (uses PyTorch)
    if impl == "flash_attn":
        # flash_attn_func expects (batch, seq_len, num_heads, head_dim) in FP16/BF16
        Q_torch = torch.from_numpy(Q_4d.transpose(0, 2, 1, 3)).cuda().half()  # (B, S, H, D)
        K_torch = torch.from_numpy(K_4d.transpose(0, 2, 1, 3)).cuda().half()
        V_torch = torch.from_numpy(V_4d.transpose(0, 2, 1, 3)).cuda().half()

        # Warmup
        for _ in range(warmup):
            _ = flash_attn_func(Q_torch, K_torch, V_torch, softmax_scale=sm_scale)
        torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            O_torch = flash_attn_func(Q_torch, K_torch, V_torch, softmax_scale=sm_scale)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        time_ms = mean(timings)
        # Convert back: (B, S, H, D) -> (B, H, S, D) -> (B*H, S, D)
        O_result = O_torch.float().cpu().numpy().transpose(0, 2, 1, 3).reshape(shape)
        passed, error = validate(O_result, ref_out)
        return time_ms, error, passed

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    for _ in range(warmup):
        launch_kernel(impl, Q, K, V, O, sm_scale, head_dim)
    wp.synchronize()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        launch_kernel(impl, Q, K, V, O, sm_scale, head_dim)
        wp.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    time_ms = mean(timings)
    passed, error = validate(O.numpy(), ref_out)

    return time_ms, error, passed


def run_correctness_benchmarks():
    print("\n" + "=" * 70)
    print("CORRECTNESS BENCHMARKS")
    print("=" * 70)

    # (batch, heads, seq_len, head_dim)
    configs = [
        (1, 1, 64, 64),
        (2, 4, 128, 64),
        (4, 8, 256, 64),
        (1, 4, 127, 64),   # Non-aligned seq_len
        (2, 4, 128, 32),   # Small head_dim
        (2, 4, 128, 128),  # Large head_dim
    ]
    all_passed = True

    for batch, heads, seq_len, head_dim in configs:
        for dist in ["uniform", "normal"]:
            for impl in IMPLEMENTATIONS:
                _, error, passed = benchmark(batch, heads, seq_len, impl, head_dim, dist, warmup=1, iterations=1)
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {impl:10} B={batch} H={heads} S={seq_len:4} D={head_dim:3} {dist:8} err={error:.2e}")
                if not passed:
                    all_passed = False

    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_scaling_benchmarks():
    base_width = 22  # naive only
    width = base_width + 14 if HAS_FLASH_ATTN else base_width
    print("\n" + "=" * width)
    print("SCALING BENCHMARKS (head_dim=64)")
    print("=" * width)

    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_heads = [(1, 8), (4, 8)]
    head_dim = 64

    for batch, heads in batch_heads:
        print(f"\nBatch={batch}, Heads={heads}, HeadDim={head_dim}")
        header = f"{'SeqLen':<8} {'naive':>12}"
        if HAS_FLASH_ATTN:
            header += f" {'flash_attn':>12}"
        print(header)
        print("-" * width)

        for seq_len in seq_lengths:
            times = {}
            for impl in IMPLEMENTATIONS:
                time_ms, _, _ = benchmark(batch, heads, seq_len, impl, head_dim)
                times[impl] = time_ms
            row = f"{seq_len:<8} {times['naive']:>10.3f}ms"
            if HAS_FLASH_ATTN:
                row += f" {times['flash_attn']:>10.3f}ms"
            print(row)


def run_distribution_benchmarks():
    print("\n" + "=" * 60)
    print("DISTRIBUTION BENCHMARKS (B=4, H=8, S=512, D=64)")
    print("=" * 60)

    head_dim = 64
    for dist in DISTRIBUTIONS:
        print(f"\n{dist}:")
        for impl in IMPLEMENTATIONS:
            time_ms, error, passed = benchmark(4, 8, 512, impl, head_dim, dist)
            status = "PASS" if passed else "FAIL"
            print(f"  {impl:10}: {time_ms:8.3f}ms  err={error:.2e}  [{status}]")

def main():
    wp.init()

    print("Flash Attention Benchmark")
    print(f"Device: {wp.get_device()}")

    run_correctness_benchmarks()
    run_scaling_benchmarks()
    run_distribution_benchmarks()


if __name__ == "__main__":
    main()
