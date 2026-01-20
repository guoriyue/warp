# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention Benchmark.

Benchmarks three implementations: naive, scalar (online softmax), simd (warp-parallel).

Usage:
    python benchmark_flash_attention.py
"""

import sys
import time
from statistics import mean

import numpy as np
import warp as wp

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from tile.example_tile_flash_attention import (
    flash_attention_kernel,
    flash_attention_simd_kernel,
    naive_attention_kernel,
    reference_attention,
)

IMPLEMENTATIONS = ["naive", "scalar", "simd"]

DISTRIBUTIONS = {
    "uniform": lambda shape, rng: rng.uniform(-1, 1, shape).astype(np.float32),
    "normal": lambda shape, rng: rng.standard_normal(shape).astype(np.float32),
    "extreme": lambda shape, rng: rng.uniform(10, 50, shape).astype(np.float32),
}

# Tolerance scales with reference magnitude: |err| < ATOL + RTOL * max(|ref|)
# Higher RTOL accounts for FP rounding differences in tree reduction vs sequential sum
RTOL = 5e-2
ATOL = 1e-5


def launch_kernel(impl: str, Q: wp.array, K: wp.array, V: wp.array, O: wp.array, sm_scale: float):
    batch_heads, seq_len, _ = Q.shape
    if impl == "naive":
        wp.launch(naive_attention_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    elif impl == "scalar":
        wp.launch(flash_attention_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    elif impl == "simd":
        wp.launch_tiled(flash_attention_simd_kernel, dim=seq_len * batch_heads, inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads], block_dim=32)


def validate(result: np.ndarray, reference: np.ndarray) -> tuple[bool, float]:
    abs_diff = np.abs(result - reference)
    max_error = float(np.max(abs_diff))
    ref_scale = float(np.max(np.abs(reference)))
    tolerance = ATOL + RTOL * ref_scale
    passed = max_error < tolerance
    return passed, max_error


def benchmark(batch: int, heads: int, seq_len: int, impl: str, distribution: str = "uniform", warmup: int = 5, iterations: int = 20):
    head_dim = 64
    rng = np.random.default_rng(42)
    gen = DISTRIBUTIONS.get(distribution, DISTRIBUTIONS["uniform"])

    shape = (batch * heads, seq_len, head_dim)
    Q_np = gen(shape, rng)
    K_np = gen(shape, rng)
    V_np = gen(shape, rng)
    sm_scale = 1.0 / np.sqrt(head_dim)

    Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
    K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
    V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
    ref_out = reference_attention(Q_4d, K_4d, V_4d, sm_scale).reshape(shape)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    for _ in range(warmup):
        launch_kernel(impl, Q, K, V, O, sm_scale)
    wp.synchronize()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        launch_kernel(impl, Q, K, V, O, sm_scale)
        wp.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    time_ms = mean(timings)
    passed, error = validate(O.numpy(), ref_out)

    return time_ms, error, passed


def run_correctness_benchmarks():
    print("\n" + "=" * 60)
    print("CORRECTNESS BENCHMARKS")
    print("=" * 60)

    configs = [(1, 1, 64), (2, 4, 128), (4, 8, 256), (1, 4, 127), (1, 4, 513)]
    all_passed = True

    for batch, heads, seq_len in configs:
        for dist in ["uniform", "normal"]:
            for impl in IMPLEMENTATIONS:
                _, error, passed = benchmark(batch, heads, seq_len, impl, dist, warmup=1, iterations=1)
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {impl:6} B={batch} H={heads} S={seq_len:4} {dist:8} err={error:.2e}")
                if not passed:
                    all_passed = False

    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_scaling_benchmarks():
    print("\n" + "=" * 60)
    print("SCALING BENCHMARKS")
    print("=" * 60)

    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_heads = [(1, 8), (4, 8)]

    for batch, heads in batch_heads:
        print(f"\nBatch={batch}, Heads={heads}")
        print(f"{'SeqLen':<8} {'naive':>12} {'scalar':>12} {'simd':>12}")
        print("-" * 48)

        for seq_len in seq_lengths:
            times = {}
            for impl in IMPLEMENTATIONS:
                time_ms, _, _ = benchmark(batch, heads, seq_len, impl)
                times[impl] = time_ms
            print(f"{seq_len:<8} {times['naive']:>10.3f}ms {times['scalar']:>10.3f}ms {times['simd']:>10.3f}ms")


def run_distribution_benchmarks():
    print("\n" + "=" * 60)
    print("DISTRIBUTION BENCHMARKS (B=4, H=8, S=512)")
    print("=" * 60)

    for dist in DISTRIBUTIONS:
        print(f"\n{dist}:")
        for impl in IMPLEMENTATIONS:
            time_ms, error, passed = benchmark(4, 8, 512, impl, dist)
            status = "PASS" if passed else "FAIL"
            print(f"  {impl:6}: {time_ms:8.3f}ms  err={error:.2e}  [{status}]")


def main():
    wp.init()

    print("Flash Attention Benchmark")
    print(f"Device: {wp.get_device()}")

    run_correctness_benchmarks()
    run_scaling_benchmarks()
    run_distribution_benchmarks()


if __name__ == "__main__":
    main()
