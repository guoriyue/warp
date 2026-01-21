# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention Benchmark.

Benchmarks four implementations:
- naive: 3-pass attention (baseline)
- scalar: online softmax (memory efficient)
- simd: warp-parallel with warp_reduce_sum
- tiled: tile_matmul-based FlashAttention (tensor cores + SRAM)

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
    get_naive_kernel,
    get_flash_kernel,
    get_simd_kernel,
    get_tiled_kernel,
    reference_attention,
    TILE_M,
    TILE_N,
    TILE_THREADS,
)

IMPLEMENTATIONS = ["naive", "scalar", "simd", "tiled"]

DISTRIBUTIONS = {
    "uniform": lambda shape, rng: rng.uniform(-1, 1, shape).astype(np.float32),
    "normal": lambda shape, rng: rng.standard_normal(shape).astype(np.float32),
    "extreme": lambda shape, rng: rng.uniform(10, 50, shape).astype(np.float32),
}

# Tolerance scales with reference magnitude: |err| < ATOL + RTOL * max(|ref|)
# Higher RTOL accounts for FP rounding differences in tree reduction vs sequential sum
RTOL = 5e-2
ATOL = 1e-5


def launch_kernel(impl: str, Q: wp.array, K: wp.array, V: wp.array, O: wp.array, sm_scale: float, head_dim: int,
                  Q_tiled: wp.array = None, K_tiled: wp.array = None, V_tiled: wp.array = None, O_tiled: wp.array = None):
    batch_heads, seq_len, _ = Q.shape
    if impl == "naive":
        kernel = get_naive_kernel(head_dim)
        wp.launch(kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    elif impl == "scalar":
        kernel = get_flash_kernel(head_dim)
        wp.launch(kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    elif impl == "simd":
        kernel = get_simd_kernel(head_dim)
        wp.launch_tiled(kernel, dim=seq_len * batch_heads, inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads], block_dim=32)
    elif impl == "tiled":
        kernel = get_tiled_kernel(head_dim)
        # Tiled kernel uses 3D arrays with batch_heads dimension
        padded_seq = ((seq_len + TILE_M - 1) // TILE_M) * TILE_M
        num_q_blocks = padded_seq // TILE_M
        num_k_blocks = padded_seq // TILE_N
        num_tiles = batch_heads * num_q_blocks
        wp.launch_tiled(
            kernel,
            dim=num_tiles,
            inputs=[Q_tiled, K_tiled, V_tiled, O_tiled, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
            block_dim=TILE_THREADS,
        )


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

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    # Prepare padded arrays for tiled implementation
    Q_tiled, K_tiled, V_tiled, O_tiled = None, None, None, None
    if impl == "tiled":
        padded_seq = ((seq_len + TILE_M - 1) // TILE_M) * TILE_M
        Q_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        K_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        V_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        Q_padded_np[:, :seq_len, :] = Q_np
        K_padded_np[:, :seq_len, :] = K_np
        V_padded_np[:, :seq_len, :] = V_np
        Q_tiled = wp.array(Q_padded_np, dtype=float)
        K_tiled = wp.array(K_padded_np, dtype=float)
        V_tiled = wp.array(V_padded_np, dtype=float)
        O_tiled = wp.zeros_like(Q_tiled)

    for _ in range(warmup):
        launch_kernel(impl, Q, K, V, O, sm_scale, head_dim, Q_tiled, K_tiled, V_tiled, O_tiled)
    wp.synchronize()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        launch_kernel(impl, Q, K, V, O, sm_scale, head_dim, Q_tiled, K_tiled, V_tiled, O_tiled)
        wp.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    time_ms = mean(timings)

    # For tiled, extract non-padded output from 3D array
    if impl == "tiled":
        O_result = O_tiled.numpy()[:, :seq_len, :]
        passed, error = validate(O_result, ref_out)
    else:
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
                print(f"[{status}] {impl:6} B={batch} H={heads} S={seq_len:4} D={head_dim:3} {dist:8} err={error:.2e}")
                if not passed:
                    all_passed = False

    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_scaling_benchmarks():
    print("\n" + "=" * 60)
    print("SCALING BENCHMARKS (head_dim=64)")
    print("=" * 60)

    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_heads = [(1, 8), (4, 8)]
    head_dim = 64

    for batch, heads in batch_heads:
        print(f"\nBatch={batch}, Heads={heads}, HeadDim={head_dim}")
        print(f"{'SeqLen':<8} {'naive':>12} {'scalar':>12} {'simd':>12} {'tiled':>12}")
        print("-" * 60)

        for seq_len in seq_lengths:
            times = {}
            for impl in IMPLEMENTATIONS:
                time_ms, _, _ = benchmark(batch, heads, seq_len, impl, head_dim)
                times[impl] = time_ms
            print(f"{seq_len:<8} {times['naive']:>10.3f}ms {times['scalar']:>10.3f}ms {times['simd']:>10.3f}ms {times['tiled']:>10.3f}ms")


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
