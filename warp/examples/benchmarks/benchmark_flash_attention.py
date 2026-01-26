# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention Benchmark.

Benchmarks implementations:
- naive: 3-pass attention (Warp baseline)
- tiled: tile_matmul based (Warp, tensor cores + SRAM)
- triton: Triton flash attention
- flash_attn: official Flash Attention library (if installed)

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

# Try to import Triton flash attention
try:
    sys.path.insert(0, "/home/mingfeiguo/Desktop/triton/python/tutorials")
    from importlib import util
    spec = util.spec_from_file_location("fused_attention", "/home/mingfeiguo/Desktop/triton/python/tutorials/06-fused-attention.py")
    triton_attn_module = util.module_from_spec(spec)
    spec.loader.exec_module(triton_attn_module)
    triton_attention = triton_attn_module.attention
    HAS_TRITON = True
except Exception as e:
    HAS_TRITON = False
    print(f"Warning: Triton flash attention not available: {e}")

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from tile.example_tile_flash_attention import (
    get_naive_kernel,
    get_tiled_kernel,
    get_tiled_kernel_f16,
    TILE_M,
    TILE_N,
    TILE_THREADS,
)

IMPLEMENTATIONS = ["naive", "tiled", "tiled_f16"]
if HAS_TRITON:
    IMPLEMENTATIONS.append("triton")
if HAS_FLASH_ATTN:
    IMPLEMENTATIONS.append("flash_attn")


def launch_kernel(impl: str, Q: wp.array, K: wp.array, V: wp.array, O: wp.array, sm_scale: float, head_dim: int):
    batch_heads, seq_len, _ = Q.shape
    if impl == "naive":
        kernel = get_naive_kernel(head_dim)
        wp.launch(kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    elif impl == "tiled":
        tile_m, tile_n = TILE_M, TILE_N
        kernel = get_tiled_kernel(head_dim, tile_m, tile_n)  # Pass tile sizes to kernel factory
        padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
        num_q_blocks = padded_seq // tile_m
        num_k_blocks = padded_seq // tile_n
        num_tiles = batch_heads * num_q_blocks
        wp.launch_tiled(
            kernel,
            dim=num_tiles,
            inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks],
            block_dim=TILE_THREADS,
        )


def benchmark(batch: int, heads: int, seq_len: int, impl: str, head_dim: int = 64,
              warmup: int = 5, iterations: int = 20, Q_np=None, K_np=None, V_np=None, ref_out=None):
    """Run benchmark and return (time_ms, max_error).

    If Q_np/K_np/V_np provided, use them. Otherwise generate random data.
    If ref_out provided, compute error against it.
    """
    batch_heads = batch * heads

    # Generate or use provided data
    if Q_np is None:
        rng = np.random.default_rng(42)
        shape = (batch_heads, seq_len, head_dim)
        Q_np = rng.uniform(-1, 1, shape).astype(np.float32)
        K_np = rng.uniform(-1, 1, shape).astype(np.float32)
        V_np = rng.uniform(-1, 1, shape).astype(np.float32)

    sm_scale = 1.0 / np.sqrt(head_dim)
    Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
    K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
    V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
    shape = (batch_heads, seq_len, head_dim)

    # Handle official flash_attn library separately (uses PyTorch)
    if impl == "flash_attn":
        Q_torch = torch.from_numpy(Q_4d.transpose(0, 2, 1, 3)).cuda().half()
        K_torch = torch.from_numpy(K_4d.transpose(0, 2, 1, 3)).cuda().half()
        V_torch = torch.from_numpy(V_4d.transpose(0, 2, 1, 3)).cuda().half()

        for _ in range(warmup):
            _ = flash_attn_func(Q_torch, K_torch, V_torch, softmax_scale=sm_scale)
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            O_torch = flash_attn_func(Q_torch, K_torch, V_torch, softmax_scale=sm_scale)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        O_out = O_torch.float().cpu().numpy().transpose(0, 2, 1, 3).reshape(shape)
        error = np.max(np.abs(O_out - ref_out)) if ref_out is not None else 0.0
        return mean(timings), error, O_out

    # Handle Triton flash attention (uses PyTorch)
    if impl == "triton":
        Q_torch = torch.from_numpy(Q_4d).cuda().half()
        K_torch = torch.from_numpy(K_4d).cuda().half()
        V_torch = torch.from_numpy(V_4d).cuda().half()

        for _ in range(warmup):
            _ = triton_attention(Q_torch, K_torch, V_torch, False, sm_scale)
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            O_torch = triton_attention(Q_torch, K_torch, V_torch, False, sm_scale)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        O_out = O_torch.float().cpu().numpy().reshape(shape)
        error = np.max(np.abs(O_out - ref_out)) if ref_out is not None else 0.0
        return mean(timings), error, O_out

    # Handle tiled kernel (needs padded arrays)
    if impl == "tiled":
        tile_m = TILE_M
        padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
        Q_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        K_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        V_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        Q_padded[:, :seq_len, :] = Q_np
        K_padded[:, :seq_len, :] = K_np
        V_padded[:, :seq_len, :] = V_np

        Q = wp.array(Q_padded, dtype=float)
        K = wp.array(K_padded, dtype=float)
        V = wp.array(V_padded, dtype=float)
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

        O_out = O.numpy()[:, :seq_len, :]
        error = np.max(np.abs(O_out - ref_out)) if ref_out is not None else 0.0
        return mean(timings), error, O_out

    # Handle tiled_f16 kernel (same tile sizes as FP32)
    if impl == "tiled_f16":
        tile_m = TILE_M
        padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
        Q_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float16)
        K_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float16)
        V_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float16)
        Q_padded[:, :seq_len, :] = Q_np.astype(np.float16)
        K_padded[:, :seq_len, :] = K_np.astype(np.float16)
        V_padded[:, :seq_len, :] = V_np.astype(np.float16)

        Q = wp.array(Q_padded, dtype=wp.float16)
        K = wp.array(K_padded, dtype=wp.float16)
        V = wp.array(V_padded, dtype=wp.float16)
        O = wp.zeros_like(Q)

        kernel = get_tiled_kernel_f16(head_dim, TILE_M, TILE_N)
        num_q_blocks = padded_seq // TILE_M
        num_k_blocks = padded_seq // TILE_N
        num_tiles = batch_heads * num_q_blocks
        sm_scale_f16 = wp.float16(sm_scale)

        for _ in range(warmup):
            wp.launch_tiled(
                kernel,
                dim=num_tiles,
                inputs=[Q, K, V, O, sm_scale_f16, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
                block_dim=TILE_THREADS,
            )
        wp.synchronize()

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            wp.launch_tiled(
                kernel,
                dim=num_tiles,
                inputs=[Q, K, V, O, sm_scale_f16, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
                block_dim=TILE_THREADS,
            )
            wp.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        O_out = O.numpy()[:, :seq_len, :].astype(np.float32)
        error = np.max(np.abs(O_out - ref_out)) if ref_out is not None else 0.0
        return mean(timings), error, O_out

    # Handle naive kernel
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

    O_out = O.numpy()
    error = np.max(np.abs(O_out - ref_out)) if ref_out is not None else 0.0
    return mean(timings), error, O_out


def run_scaling_benchmarks():
    print("\n" + "=" * 100)
    print("SCALING BENCHMARKS (head_dim=64) - comparing against flash_attn reference")
    print("=" * 100)

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    batch_heads_configs = [(1, 8), (2, 8), (4, 8)]
    head_dim = 64

    if not HAS_FLASH_ATTN:
        print("WARNING: flash_attn not available, skipping correctness checks")

    for batch, heads in batch_heads_configs:
        print(f"\nBatch={batch}, Heads={heads}, HeadDim={head_dim}")

        # Build header: impl (time, err)
        header = f"{'SeqLen':<8}"
        for impl in IMPLEMENTATIONS:
            header += f" {impl:>18}"
        print(header)
        print("-" * 100)

        for seq_len in seq_lengths:
            # Skip very large configs that might OOM
            if batch * heads * seq_len > 16 * 8192:
                continue

            # Generate shared test data
            batch_heads_total = batch * heads
            rng = np.random.default_rng(42)
            shape = (batch_heads_total, seq_len, head_dim)
            Q_np = rng.uniform(-1, 1, shape).astype(np.float32)
            K_np = rng.uniform(-1, 1, shape).astype(np.float32)
            V_np = rng.uniform(-1, 1, shape).astype(np.float32)

            # Get flash_attn reference output first
            ref_out = None
            if HAS_FLASH_ATTN:
                try:
                    _, _, ref_out = benchmark(batch, heads, seq_len, "flash_attn", head_dim,
                                              Q_np=Q_np, K_np=K_np, V_np=V_np, ref_out=None)
                except Exception:
                    pass

            results = {}
            for impl in IMPLEMENTATIONS:
                try:
                    time_ms, error, _ = benchmark(batch, heads, seq_len, impl, head_dim,
                                                   Q_np=Q_np, K_np=K_np, V_np=V_np, ref_out=ref_out)
                    results[impl] = (time_ms, error)
                except Exception as e:
                    results[impl] = (float('nan'), float('nan'))

            row = f"{seq_len:<8}"
            for impl in IMPLEMENTATIONS:
                if impl in results:
                    t, e = results[impl]
                    if impl == "flash_attn":
                        row += f" {t:>8.3f}ms (ref)"
                    else:
                        row += f" {t:>8.3f}ms e={e:.0e}"
                else:
                    row += f" {'N/A':>18}"
            print(row)


def main():
    wp.init()

    print("Flash Attention Benchmark")
    print(f"Device: {wp.get_device()}")
    print(f"Implementations: {IMPLEMENTATIONS}")

    run_scaling_benchmarks()


if __name__ == "__main__":
    main()
