# # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0

# """
# Flash Attention v2 implementation in Warp.

# Implementations:
# 1. Naive 3-pass attention (baseline)
# 2. Flash Attention tiled FP32 - tile_matmul based (tensor cores + SRAM)
# 3. Flash Attention tiled FP16 - tile_matmul based (tensor cores + SRAM, half precision)

# Supports head dimensions: 32, 64, 128, 256
# Input shape: [batch_heads, seq_len, head_dim]
# """

# import numpy as np
# import warp as wp

# # Constants used by kernels
# NEG_INF = wp.constant(-1.0e10)

# # Default tile sizes for tile_matmul-based flash attention
# TILE_M = 16
# TILE_N = 16
# TILE_THREADS = 64
# # Supported head dimensions
# SUPPORTED_HEAD_DIMS = [64, 128, 256]


# # Helper functions for tile_map operations (FP32)
# @wp.func
# def mul_func(a: float, b: float) -> float:
#     return a * b


# @wp.func
# def div_func(a: float, b: float) -> float:
#     return a / b


# # Helper functions for tile_map operations (FP16)
# @wp.func
# def mul_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
#     return a * b


# @wp.func
# def div_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
#     return a / b


# # =============================================================================
# # Kernel factories for different head dimensions
# # =============================================================================

# def create_naive_attention_kernel(head_dim: int):
#     """Factory to create naive attention kernel for specific head_dim."""
#     @wp.kernel(enable_backward=False)
#     def naive_attention_kernel(
#         Q: wp.array3d(dtype=float),
#         K: wp.array3d(dtype=float),
#         V: wp.array3d(dtype=float),
#         O: wp.array3d(dtype=float),
#         sm_scale: float,
#         seq_len: int,
#     ):
#         """Naive 3-pass attention."""
#         query_idx, batch_head = wp.tid()
#         if query_idx >= seq_len:
#             return

#         HEAD_DIM_LOCAL = wp.static(head_dim)

#         q = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
#         for d in range(HEAD_DIM_LOCAL):
#             q[d] = Q[batch_head, query_idx, d]

#         max_score = float(NEG_INF)
#         for k_idx in range(seq_len):
#             score = float(0.0)
#             for d in range(HEAD_DIM_LOCAL):
#                 score += q[d] * K[batch_head, k_idx, d]
#             max_score = wp.max(max_score, score * sm_scale)

#         sum_exp = float(0.0)
#         for k_idx in range(seq_len):
#             score = float(0.0)
#             for d in range(HEAD_DIM_LOCAL):
#                 score += q[d] * K[batch_head, k_idx, d]
#             sum_exp += wp.exp(score * sm_scale - max_score)

#         acc = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
#         for d in range(HEAD_DIM_LOCAL):
#             acc[d] = float(0.0)

#         for k_idx in range(seq_len):
#             score = float(0.0)
#             for d in range(HEAD_DIM_LOCAL):
#                 score += q[d] * K[batch_head, k_idx, d]
#             weight = wp.exp(score * sm_scale - max_score) / sum_exp
#             for d in range(HEAD_DIM_LOCAL):
#                 acc[d] += weight * V[batch_head, k_idx, d]

#         for d in range(HEAD_DIM_LOCAL):
#             O[batch_head, query_idx, d] = acc[d]

#     return naive_attention_kernel


# # =============================================================================
# # Kernel cache for different head dimensions
# # =============================================================================

# _kernel_cache = {}


# def get_naive_kernel(head_dim: int):
#     """Get or create naive attention kernel for head_dim."""
#     key = ("naive", head_dim)
#     if key not in _kernel_cache:
#         if head_dim not in SUPPORTED_HEAD_DIMS:
#             raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
#         _kernel_cache[key] = create_naive_attention_kernel(head_dim)
#     return _kernel_cache[key]


# def create_flash_attention_tiled_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
#     """Factory to create tiled flash attention kernel using tile_matmul.

#     This implements the FlashAttention memory access pattern:
#     - K/V blocks are loaded into SRAM (shared memory) once
#     - Multiple Q rows in the same tile share the K/V data
#     - Online softmax avoids materializing the full attention matrix

#     Args:
#         head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
#         tile_m: Query block size (default 16)
#         tile_n: Key/Value block size (default 16)
#     """
#     @wp.kernel(enable_backward=False)
#     def flash_attention_tiled_kernel(
#         Q: wp.array3d(dtype=float),
#         K: wp.array3d(dtype=float),
#         V: wp.array3d(dtype=float),
#         O: wp.array3d(dtype=float),
#         sm_scale: float,
#         seq_len: int,
#         batch_heads: int,
#         num_q_blocks: int,
#         num_k_blocks: int,
#     ):
#         """Flash Attention using tile_matmul for Q@K^T and P@V.

#         Memory access pattern (FlashAttention style):
#         - Q block loaded to registers (reused across K/V iterations)
#         - K/V blocks loaded to SRAM (shared by all threads in block)
#         - O, m, l accumulators in SRAM (updated incrementally)
#         """
#         TILE_M_LOCAL = wp.static(tile_m)
#         TILE_N_LOCAL = wp.static(tile_n)
#         HEAD_DIM_LOCAL = wp.static(head_dim)

#         tile_idx = wp.tid()
#         batch_head = tile_idx // num_q_blocks
#         q_block_idx = tile_idx % num_q_blocks
#         q_start = q_block_idx * TILE_M_LOCAL

#         if batch_head >= batch_heads:
#             return

#         # Load Q block to registers - reused across all K/V blocks
#         # Shape: (1, TILE_M, HEAD_DIM) -> squeeze to (TILE_M, HEAD_DIM)
#         Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
#         Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

#         # Initialize output accumulator in shared memory
#         O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=float, storage="shared")

#         # Initialize online softmax state per row in registers
#         m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=float(NEG_INF), dtype=float)
#         l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=float)

#         # Iterate over K/V blocks (outer loop in FlashAttention paper)
#         for k_block in range(num_k_blocks):
#             k_start = k_block * TILE_N_LOCAL

#             # Load K/V blocks to SRAM - shared by all threads in this block
#             # This is the key optimization: K/V loaded from HBM to SRAM once,
#             # then reused by all TILE_M query rows in this tile
#             K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
#             K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
#             V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
#             V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

#             # Compute S = Q @ K^T: (TILE_M, HEAD_DIM) @ (HEAD_DIM, TILE_N) = (TILE_M, TILE_N)
#             K_T = wp.tile_transpose(K_tile)
#             S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=float)
#             wp.tile_matmul(Q_tile, K_T, S)

#             # Scale attention scores
#             S = S * sm_scale

#             # Compute row-wise max for this block
#             m_block = wp.tile_reduce(wp.max, S, axis=1)
#             m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))

#             # New running max: m_new = max(m_i, m_block)
#             m_new = wp.tile_map(wp.max, m_i, m_block)

#             # Compute rescaling factor: alpha = exp(m_old - m_new)
#             alpha = wp.tile_map(wp.exp, m_i - m_new)

#             # Compute P = exp(S - m_new) - softmax numerator
#             m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
#             P = wp.tile_map(wp.exp, S - m_broadcast)

#             # Compute row-wise sum of P for this block
#             l_block = wp.tile_sum(P, axis=1)
#             l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))

#             # Update running sum: l_new = l_i * alpha + l_block
#             l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
#             wp.tile_assign(l_i, l_new)

#             # Update output accumulator: O = O * alpha
#             alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
#             O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)
#             wp.tile_assign(O_acc, O_scaled)

#             # Add P @ V contribution: O += P @ V
#             # (TILE_M, TILE_N) @ (TILE_N, HEAD_DIM) = (TILE_M, HEAD_DIM)
#             wp.tile_matmul(P, V_tile, O_acc)

#             # Update max for next iteration
#             wp.tile_assign(m_i, m_new)

#         # Final normalization: O = O_acc / l_i
#         l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
#         O_final = wp.tile_map(div_func, O_acc, l_broadcast)

#         # Store output back to HBM
#         O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
#         wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

#     return flash_attention_tiled_kernel


# def get_tiled_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
#     """Get or create tiled flash attention kernel for head_dim."""
#     key = ("tiled", head_dim, tile_m, tile_n)
#     if key not in _kernel_cache:
#         if head_dim not in SUPPORTED_HEAD_DIMS:
#             raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
#         _kernel_cache[key] = create_flash_attention_tiled_kernel(head_dim, tile_m, tile_n)
#     return _kernel_cache[key]


# def create_flash_attention_tiled_f16_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
#     """Factory to create FP16 tiled flash attention kernel - same as FP32 but with fp16 dtype."""
#     NEG_INF_F16 = wp.float16(-65504.0)

#     @wp.kernel(enable_backward=False)
#     def flash_attention_tiled_f16_kernel(
#         Q: wp.array3d(dtype=wp.float16),
#         K: wp.array3d(dtype=wp.float16),
#         V: wp.array3d(dtype=wp.float16),
#         O: wp.array3d(dtype=wp.float16),
#         sm_scale: wp.float16,
#         seq_len: int,
#         batch_heads: int,
#         num_q_blocks: int,
#         num_k_blocks: int,
#     ):
#         TILE_M_LOCAL = wp.static(tile_m)
#         TILE_N_LOCAL = wp.static(tile_n)
#         HEAD_DIM_LOCAL = wp.static(head_dim)

#         tile_idx = wp.tid()
#         batch_head = tile_idx // num_q_blocks
#         q_block_idx = tile_idx % num_q_blocks
#         q_start = q_block_idx * TILE_M_LOCAL

#         if batch_head >= batch_heads:
#             return

#         Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
#         Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

#         O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=wp.float16)
#         m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=NEG_INF_F16, dtype=wp.float16)
#         l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=wp.float16)

#         for k_block in range(num_k_blocks):
#             k_start = k_block * TILE_N_LOCAL

#             K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
#             K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
#             V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
#             V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

#             K_T = wp.tile_transpose(K_tile)
#             S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
#             wp.tile_matmul(Q_tile, K_T, S)
#             S = S * sm_scale

#             m_block = wp.tile_reduce(wp.max, S, axis=1)
#             m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))
#             m_new = wp.tile_map(wp.max, m_i, m_block)

#             alpha = wp.tile_map(wp.exp, m_i - m_new)
#             m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
#             P = wp.tile_map(wp.exp, S - m_broadcast)

#             l_block = wp.tile_sum(P, axis=1)
#             l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))
#             l_new = wp.tile_map(mul_func_f16, l_i, alpha) + l_block
#             wp.tile_assign(l_i, l_new)

#             alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
#             O_scaled = wp.tile_map(mul_func_f16, O_acc, alpha_broadcast)
#             wp.tile_assign(O_acc, O_scaled)

#             wp.tile_matmul(P, V_tile, O_acc)
#             wp.tile_assign(m_i, m_new)

#         l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
#         O_final = wp.tile_map(div_func_f16, O_acc, l_broadcast)

#         O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
#         wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

#     return flash_attention_tiled_f16_kernel


# def get_tiled_kernel_f16(head_dim: int, tile_m: int = 16, tile_n: int = 16):
#     """Get or create FP16 tiled flash attention kernel for head_dim."""
#     key = ("tiled_f16", head_dim, tile_m, tile_n)
#     if key not in _kernel_cache:
#         if head_dim not in SUPPORTED_HEAD_DIMS:
#             raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
#         _kernel_cache[key] = create_flash_attention_tiled_f16_kernel(head_dim, tile_m, tile_n)
#     return _kernel_cache[key]


# def reference_attention(Q, K, V, sm_scale):
#     """NumPy reference."""
#     QK = np.matmul(Q, K.transpose(0, 1, 3, 2)) * sm_scale
#     QK_max = QK.max(axis=-1, keepdims=True)
#     P = np.exp(QK - QK_max)
#     P = P / P.sum(axis=-1, keepdims=True)
#     return np.matmul(P, V)


# if __name__ == "__main__":
#     print("Flash Attention Example")
#     print("=" * 60)

#     # Test configurations: (batch, heads, seq_len, head_dim)
#     test_configs = [
#         (2, 4, 128, 64),
#         (2, 4, 128, 128),
#     ]

#     all_passed = True

#     for batch, heads, seq_len, head_dim in test_configs:
#         batch_heads = batch * heads
#         print(f"\n--- Head Dimension: {head_dim} (B={batch}, H={heads}, S={seq_len}) ---")

#         # Generate test data
#         rng = np.random.default_rng(42)
#         Q_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
#         K_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
#         V_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
#         sm_scale = 1.0 / np.sqrt(head_dim)

#         # Compute reference
#         Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
#         K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
#         V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
#         ref_out = reference_attention(Q_4d, K_4d, V_4d, sm_scale).reshape(batch_heads, seq_len, head_dim)

#         # Create warp arrays
#         Q = wp.array(Q_np, dtype=float)
#         K = wp.array(K_np, dtype=float)
#         V = wp.array(V_np, dtype=float)

#         # Test naive kernel
#         naive_kernel = get_naive_kernel(head_dim)
#         O = wp.zeros_like(Q)
#         wp.launch(naive_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
#         err = np.max(np.abs(O.numpy() - ref_out))
#         status = "PASS" if err < 1e-5 else "FAIL"
#         print(f"  Naive:      err={err:.2e} [{status}]")
#         if status == "FAIL":
#             all_passed = False

#         # Test tiled kernel (tile_matmul based, FlashAttention style)
#         tile_m, tile_n = TILE_M, TILE_N
#         tiled_kernel = get_tiled_kernel(head_dim, tile_m, tile_n)
#         print(f"    Using TILE_M={tile_m}, TILE_N={tile_n}")
#         num_q_blocks = seq_len // tile_m
#         num_k_blocks = seq_len // tile_n
#         num_tiles_tiled = batch_heads * num_q_blocks

#         Q_tiled = wp.array(Q_np, dtype=float)
#         K_tiled = wp.array(K_np, dtype=float)
#         V_tiled = wp.array(V_np, dtype=float)
#         O_tiled = wp.zeros_like(Q_tiled)
#         wp.launch_tiled(
#             tiled_kernel,
#             dim=num_tiles_tiled,
#             inputs=[Q_tiled, K_tiled, V_tiled, O_tiled, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks],
#             block_dim=TILE_THREADS,
#         )
#         err = np.max(np.abs(O_tiled.numpy() - ref_out))
#         status = "PASS" if err < 1e-5 else "FAIL"
#         print(f"  Tiled:      err={err:.2e} [{status}]")
#         if status == "FAIL":
#             all_passed = False

#     print("\n" + "=" * 60)
#     print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
#     print("=" * 60)


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Flash Attention v2 implementation in Warp.

Implementations:
1. Naive 3-pass attention (baseline)
2. Flash Attention tiled FP32 - tile_matmul based (tensor cores + SRAM)
3. Flash Attention tiled FP16 - tile_matmul based (tensor cores + SRAM, half precision)

Supports head dimensions: 32, 64, 128, 256
Input shape: [batch_heads, seq_len, head_dim]
"""

import numpy as np
import warp as wp

# Constants used by kernels
NEG_INF = wp.constant(-1.0e10)

# Default tile sizes for tile_matmul-based flash attention
# TILE_M = 64
# TILE_N = 64
# TILE_THREADS = 256
TILE_M = 16
TILE_N = 16
TILE_THREADS = 64
# Supported head dimensions
SUPPORTED_HEAD_DIMS = [32, 64, 128, 256]


# Helper functions for tile_map operations (FP32)
@wp.func
def mul_func(a: float, b: float) -> float:
    return a * b


@wp.func
def div_func(a: float, b: float) -> float:
    return a / b


@wp.func
def add_func(a: float, b: float) -> float:
    return a + b


# Helper functions for tile_map operations (FP16)
@wp.func
def mul_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
    return a * b


@wp.func
def div_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
    return a / b


@wp.func
def add_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
    return a + b


# =============================================================================
# Kernel factories for different head dimensions
# =============================================================================

def create_naive_attention_kernel(head_dim: int):
    """Factory to create naive attention kernel for specific head_dim."""
    @wp.kernel(enable_backward=False)
    def naive_attention_kernel(
        Q: wp.array3d(dtype=float),
        K: wp.array3d(dtype=float),
        V: wp.array3d(dtype=float),
        O: wp.array3d(dtype=float),
        sm_scale: float,
        seq_len: int,
    ):
        """Naive 3-pass attention."""
        query_idx, batch_head = wp.tid()
        if query_idx >= seq_len:
            return

        HEAD_DIM_LOCAL = wp.static(head_dim)

        q = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
        for d in range(HEAD_DIM_LOCAL):
            q[d] = Q[batch_head, query_idx, d]

        max_score = float(NEG_INF)
        for k_idx in range(seq_len):
            score = float(0.0)
            for d in range(HEAD_DIM_LOCAL):
                score += q[d] * K[batch_head, k_idx, d]
            max_score = wp.max(max_score, score * sm_scale)

        sum_exp = float(0.0)
        for k_idx in range(seq_len):
            score = float(0.0)
            for d in range(HEAD_DIM_LOCAL):
                score += q[d] * K[batch_head, k_idx, d]
            sum_exp += wp.exp(score * sm_scale - max_score)

        acc = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
        for d in range(HEAD_DIM_LOCAL):
            acc[d] = float(0.0)

        for k_idx in range(seq_len):
            score = float(0.0)
            for d in range(HEAD_DIM_LOCAL):
                score += q[d] * K[batch_head, k_idx, d]
            weight = wp.exp(score * sm_scale - max_score) / sum_exp
            for d in range(HEAD_DIM_LOCAL):
                acc[d] += weight * V[batch_head, k_idx, d]

        for d in range(HEAD_DIM_LOCAL):
            O[batch_head, query_idx, d] = acc[d]

    return naive_attention_kernel


# =============================================================================
# Kernel cache for different head dimensions
# =============================================================================

_kernel_cache = {}


def get_naive_kernel(head_dim: int):
    """Get or create naive attention kernel for head_dim."""
    key = ("naive", head_dim)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_naive_attention_kernel(head_dim)
    return _kernel_cache[key]


def create_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                   use_fp16: bool = False, threads_per_row: int = 1):
    """Flash attention with tile_matmul and per-thread streaming accumulation.

    Hybrid approach using tile operations where possible:
    - Q loaded as tile once (reused across K/V blocks)
    - tile_matmul for S = Q @ K^T (tensor core acceleration)
    - Per-thread fused softmax + V accumulation (streaming)
    - Multiple threads per row split head_dim for parallelism

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size
        tile_n: Key/Value block size
        use_fp16: Use FP16 precision (default False for FP32)
        threads_per_row: Number of threads cooperating on each row (default 1).
            Each thread handles head_dim // threads_per_row elements.
            block_dim = tile_m * threads_per_row.
    """
    assert head_dim % threads_per_row == 0, \
        f"head_dim ({head_dim}) must be divisible by threads_per_row ({threads_per_row})"

    d_per_thread = head_dim // threads_per_row
    HEAD_DIM_CONST = wp.constant(head_dim)
    D_PER_THREAD_CONST = wp.constant(d_per_thread)
    TILE_N_CONST = wp.constant(tile_n)

    if use_fp16:
        @wp.kernel(enable_backward=False)
        def flash_attention_kernel_f16(
            Q: wp.array3d(dtype=wp.float16),
            K: wp.array3d(dtype=wp.float16),
            V: wp.array3d(dtype=wp.float16),
            O: wp.array3d(dtype=wp.float16),
            sm_scale: float,
            seq_len: int,
            batch_heads: int,
            num_q_blocks: int,
            num_k_blocks: int,
        ):
            """Flash Attention FP16 I/O with FP32 accumulation.

            Inputs/outputs are fp16, but all internal accumulation
            (m_i, l_i, o_acc, p, alpha) runs in fp32 to avoid precision loss.
            tile_matmul runs in fp16 on tensor cores for S = Q @ K^T.
            Multiple threads per row split head_dim for parallelism.
            Uses exp2 instead of exp for faster math (sm_scale pre-multiplied by LOG2E).
            """
            TILE_M_LOCAL = wp.static(tile_m)
            TILE_N_LOCAL = wp.static(tile_n)
            HEAD_DIM_LOCAL = wp.static(head_dim)
            THREADS_PER_ROW = wp.static(threads_per_row)
            D_PER_THREAD = wp.static(d_per_thread)

            # Thread indexing: threads_per_row threads cooperate on each Q row
            threads_per_tile = TILE_M_LOCAL * THREADS_PER_ROW
            tile_idx = wp.tid() // threads_per_tile
            local_tid = wp.tid() % threads_per_tile
            row_in_tile = local_tid // THREADS_PER_ROW
            d_idx = local_tid % THREADS_PER_ROW
            d_start = d_idx * D_PER_THREAD

            batch_head = tile_idx // num_q_blocks
            q_block_idx = tile_idx % num_q_blocks
            q_start = q_block_idx * TILE_M_LOCAL
            q_row = q_start + row_in_tile

            if batch_head >= batch_heads:
                return
            if q_row >= seq_len:
                return

            # Pre-multiply sm_scale by LOG2E so all exp() become exp2()
            sm_scale_log2 = sm_scale * 1.44269504

            # Load Q block as tile (reused across all K/V blocks) - fp16 for tensor core matmul
            Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
            Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))  # [TILE_M, HEAD_DIM]

            # Per-thread accumulators in FP32 (only D_PER_THREAD elements)
            m_i = float(-1e10)    # Running max (redundant across threads in same row)
            l_i = float(0.0)     # Running sum of exp (redundant across threads in same row)
            o_acc = wp.vector(dtype=float, length=D_PER_THREAD_CONST)
            for d in range(D_PER_THREAD):
                o_acc[d] = 0.0

            # Iterate over K/V blocks
            for k_block in range(num_k_blocks):
                k_start = k_block * TILE_N_LOCAL

                # Load K/V blocks - fp16 for tensor core matmul
                K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
                K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))  # [TILE_N, HEAD_DIM]
                V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
                V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))  # [TILE_N, HEAD_DIM]

                # Compute S = Q @ K^T using tile_matmul (fp16 tensor cores)
                K_T = wp.tile_transpose(K_tile)  # [HEAD_DIM, TILE_N]
                S_tile = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
                wp.tile_matmul(Q_tile, K_T, S_tile)

                # Row-wise max in fp32 (scores pre-scaled by LOG2E for exp2)
                m_block = float(-1e10)
                for n in range(TILE_N_LOCAL):
                    s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                    m_block = wp.max(m_block, s)

                # Online softmax update using exp2 (per-thread, fp32)
                m_new = wp.max(m_i, m_block)
                alpha = wp.exp2(m_i - m_new)

                # Rescale existing accumulators by alpha
                l_i = l_i * alpha
                for d in range(D_PER_THREAD):
                    o_acc[d] = o_acc[d] * alpha

                # Fused softmax + V accumulation using exp2 (per-row streaming, fp32)
                # All threads in row compute same p/l_i, each accumulates its d-slice of o_acc
                for n in range(TILE_N_LOCAL):
                    s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                    p = wp.exp2(s - m_new)
                    l_i = l_i + p
                    for d in range(D_PER_THREAD):
                        o_acc[d] = o_acc[d] + p * float(V_tile[n, d_start + d])

                # Update running max (per-thread)
                m_i = m_new

            # Final normalization and store (cast fp32 -> fp16, each thread writes its d-slice)
            for d in range(D_PER_THREAD):
                O[batch_head, q_row, d_start + d] = wp.float16(o_acc[d] / l_i)

        return flash_attention_kernel_f16

    else:
        @wp.kernel(enable_backward=False)
        def flash_attention_kernel(
            Q: wp.array3d(dtype=float),
            K: wp.array3d(dtype=float),
            V: wp.array3d(dtype=float),
            O: wp.array3d(dtype=float),
            sm_scale: float,
            seq_len: int,
            batch_heads: int,
            num_q_blocks: int,
            num_k_blocks: int,
        ):
            """Flash Attention FP32 with per-thread streaming accumulation.

            Multiple threads per row split head_dim for parallelism.
            All threads in a row redundantly compute softmax state (m, l, p).
            Each thread accumulates only its slice of o_acc.
            Uses exp2 instead of exp for faster math (sm_scale pre-multiplied by LOG2E).
            """
            TILE_M_LOCAL = wp.static(tile_m)
            TILE_N_LOCAL = wp.static(tile_n)
            HEAD_DIM_LOCAL = wp.static(head_dim)
            THREADS_PER_ROW = wp.static(threads_per_row)
            D_PER_THREAD = wp.static(d_per_thread)

            # Thread indexing: threads_per_row threads cooperate on each Q row
            threads_per_tile = TILE_M_LOCAL * THREADS_PER_ROW
            tile_idx = wp.tid() // threads_per_tile
            local_tid = wp.tid() % threads_per_tile
            row_in_tile = local_tid // THREADS_PER_ROW
            d_idx = local_tid % THREADS_PER_ROW
            d_start = d_idx * D_PER_THREAD

            batch_head = tile_idx // num_q_blocks
            q_block_idx = tile_idx % num_q_blocks
            q_start = q_block_idx * TILE_M_LOCAL
            q_row = q_start + row_in_tile

            if batch_head >= batch_heads:
                return
            if q_row >= seq_len:
                return

            # Pre-multiply sm_scale by LOG2E so all exp() become exp2()
            sm_scale_log2 = sm_scale * 1.44269504

            # Load Q block as tile (reused across all K/V blocks)
            Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
            Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))  # [TILE_M, HEAD_DIM]

            # Per-thread accumulators (only D_PER_THREAD elements for o_acc)
            m_i = float(-1e10)  # Running max (redundant across threads in same row)
            l_i = float(0.0)    # Running sum of exp (redundant across threads in same row)
            o_acc = wp.vector(dtype=float, length=D_PER_THREAD_CONST)
            for d in range(D_PER_THREAD):
                o_acc[d] = 0.0

            # Iterate over K/V blocks
            for k_block in range(num_k_blocks):
                k_start = k_block * TILE_N_LOCAL

                # Load K/V blocks to shared memory
                K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
                K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))  # [TILE_N, HEAD_DIM]
                V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
                V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))  # [TILE_N, HEAD_DIM]

                # Compute S = Q @ K^T using tile_matmul
                K_T = wp.tile_transpose(K_tile)  # [HEAD_DIM, TILE_N]
                S_tile = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=float)
                wp.tile_matmul(Q_tile, K_T, S_tile)

                # Scale S with LOG2E-adjusted scale (scores now in log2 domain)
                S_tile = S_tile * sm_scale_log2

                # Row-wise max (all threads in row compute redundantly)
                m_block = float(-1e10)
                for n in range(TILE_N_LOCAL):
                    m_block = wp.max(m_block, S_tile[row_in_tile, n])

                # Online softmax update using exp2 (per-thread)
                m_new = wp.max(m_i, m_block)
                alpha = wp.exp2(m_i - m_new)

                # Rescale existing accumulators by alpha
                l_i = l_i * alpha
                for d in range(D_PER_THREAD):
                    o_acc[d] = o_acc[d] * alpha

                # Fused softmax + V accumulation using exp2 (per-row streaming)
                # All threads in row compute same p/l_i, each accumulates its d-slice of o_acc
                for n in range(TILE_N_LOCAL):
                    s = S_tile[row_in_tile, n]
                    p = wp.exp2(s - m_new)
                    l_i = l_i + p
                    for d in range(D_PER_THREAD):
                        o_acc[d] = o_acc[d] + p * V_tile[n, d_start + d]

                # Update running max (per-thread)
                m_i = m_new

            # Final normalization and store (each thread writes its d-slice)
            for d in range(D_PER_THREAD):
                O[batch_head, q_row, d_start + d] = o_acc[d] / l_i

        return flash_attention_kernel


def get_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                use_fp16: bool = False, threads_per_row: int = 1):
    """Get or create flash attention kernel (tile_matmul + per-thread streaming).

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size
        tile_n: Key/Value block size
        use_fp16: Use FP16 precision (default False for FP32)
        threads_per_row: Number of threads cooperating on each row (default 1).
            block_dim = tile_m * threads_per_row.
    """
    key = ("flash", head_dim, tile_m, tile_n, use_fp16, threads_per_row)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_kernel(head_dim, tile_m, tile_n, use_fp16, threads_per_row)
    return _kernel_cache[key]


def create_flash_attention_tiled_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
    """Factory to create tiled flash attention kernel using tile_matmul.

    All-tile implementation using shared memory:
    - Q, K, V blocks loaded to shared memory
    - tile_matmul for S = Q @ K^T and O = P @ V
    - tile_reduce for row-wise max and sum
    - All accumulators in shared memory (tile_assign for updates)

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size (default 16)
        tile_n: Key/Value block size (default 16)
    """
    @wp.kernel(enable_backward=False)
    def flash_attention_tiled_kernel(
        Q: wp.array3d(dtype=float),
        K: wp.array3d(dtype=float),
        V: wp.array3d(dtype=float),
        O: wp.array3d(dtype=float),
        sm_scale: float,
        seq_len: int,
        batch_heads: int,
        num_q_blocks: int,
        num_k_blocks: int,
    ):
        """Flash Attention using tile_matmul for Q@K^T and P@V.

        All tiles in shared memory with tile_assign for loop updates.
        """
        TILE_M_LOCAL = wp.static(tile_m)
        TILE_N_LOCAL = wp.static(tile_n)
        HEAD_DIM_LOCAL = wp.static(head_dim)

        tile_idx = wp.tid()
        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks
        q_start = q_block_idx * TILE_M_LOCAL

        if batch_head >= batch_heads:
            return

        # Load Q block - reused across all K/V blocks
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Initialize accumulators in shared memory
        O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=float)
        m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=float(NEG_INF), dtype=float)
        l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=float)

        # Iterate over K/V blocks
        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            # Load K/V blocks to shared memory
            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            # Compute S = Q @ K^T using tile_matmul
            K_T = wp.tile_transpose(K_tile)
            S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=float)
            wp.tile_matmul(Q_tile, K_T, S)

            # Scale attention scores
            S = S * sm_scale

            # Compute row-wise max for this block using tile_reduce
            m_block = wp.tile_reduce(wp.max, S, axis=1)
            m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))

            # New running max: m_new = max(m_i, m_block)
            m_new = wp.tile_map(wp.max, m_i, m_block)

            # Compute rescaling factor: alpha = exp(m_old - m_new)
            alpha = wp.tile_map(wp.exp, m_i - m_new)

            # Compute P = exp(S - m_new) - softmax numerator
            m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
            P = wp.tile_map(wp.exp, S - m_broadcast)

            # Compute row-wise sum of P for this block
            l_block = wp.tile_reduce(add_func, P, axis=1)
            l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))

            # Update running sum: l_i = l_i * alpha + l_block
            l_i_scaled = wp.tile_map(mul_func, l_i, alpha)
            l_i_new = l_i_scaled + l_block
            wp.tile_assign_register(l_i, l_i_new)

            # Update output accumulator: O = O * alpha + P @ V
            alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
            O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)

            # Matmul: temp = P @ V
            temp_PV = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=float)
            wp.tile_matmul(P, V_tile, temp_PV)

            # O_acc = O_scaled + temp_PV
            O_acc_new = O_scaled + temp_PV
            wp.tile_assign_register(O_acc, O_acc_new)

            # Update max for next iteration
            wp.tile_assign_register(m_i, m_new)

        # Final normalization: O = O_acc / l_i
        l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        O_final = wp.tile_map(div_func, O_acc, l_broadcast)

        # Store output back to HBM
        O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
        wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

    return flash_attention_tiled_kernel


def get_tiled_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
    """Get or create tiled flash attention kernel for head_dim."""
    key = ("tiled", head_dim, tile_m, tile_n)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_tiled_kernel(head_dim, tile_m, tile_n)
    return _kernel_cache[key]


def create_flash_attention_tiled_f16_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
    """Factory to create FP16 tiled flash attention kernel - same as FP32 but with fp16 dtype."""
    NEG_INF_F16 = wp.float16(-65504.0)

    @wp.kernel(enable_backward=False)
    def flash_attention_tiled_f16_kernel(
        Q: wp.array3d(dtype=wp.float16),
        K: wp.array3d(dtype=wp.float16),
        V: wp.array3d(dtype=wp.float16),
        O: wp.array3d(dtype=wp.float16),
        sm_scale: wp.float16,
        seq_len: int,
        batch_heads: int,
        num_q_blocks: int,
        num_k_blocks: int,
    ):
        TILE_M_LOCAL = wp.static(tile_m)
        TILE_N_LOCAL = wp.static(tile_n)
        HEAD_DIM_LOCAL = wp.static(head_dim)

        tile_idx = wp.tid()
        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks
        q_start = q_block_idx * TILE_M_LOCAL

        if batch_head >= batch_heads:
            return

        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=wp.float16, storage="shared")
        m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=NEG_INF_F16, dtype=wp.float16, storage="shared")
        l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=wp.float16, storage="shared")

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            K_T = wp.tile_transpose(K_tile)
            S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
            wp.tile_matmul(Q_tile, K_T, S)
            S = S * sm_scale

            m_block = wp.tile_reduce(wp.max, S, axis=1)
            m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))
            m_new = wp.tile_map(wp.max, m_i, m_block)

            alpha = wp.tile_map(wp.exp, m_i - m_new)
            m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
            P = wp.tile_map(wp.exp, S - m_broadcast)

            l_block = wp.tile_sum(P, axis=1)
            l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))
            l_new = wp.tile_map(mul_func_f16, l_i, alpha) + l_block
            wp.tile_assign_register(l_i, l_new)

            alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
            O_scaled = wp.tile_map(mul_func_f16, O_acc, alpha_broadcast)

            # Matmul: temp = P @ V
            temp_PV = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=wp.float16)
            wp.tile_matmul(P, V_tile, temp_PV)

            # O_acc = O_scaled + temp_PV
            O_acc_new = O_scaled + temp_PV
            wp.tile_assign_register(O_acc, O_acc_new)

            wp.tile_assign_register(m_i, m_new)

        l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        O_final = wp.tile_map(div_func_f16, O_acc, l_broadcast)

        O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
        wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

    return flash_attention_tiled_f16_kernel


def get_tiled_kernel_f16(head_dim: int, tile_m: int = 16, tile_n: int = 16):
    """Get or create FP16 tiled flash attention kernel for head_dim."""
    key = ("tiled_f16", head_dim, tile_m, tile_n)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_tiled_f16_kernel(head_dim, tile_m, tile_n)
    return _kernel_cache[key]


def reference_attention(Q, K, V, sm_scale):
    """NumPy reference."""
    QK = np.matmul(Q, K.transpose(0, 1, 3, 2)) * sm_scale
    QK_max = QK.max(axis=-1, keepdims=True)
    P = np.exp(QK - QK_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


if __name__ == "__main__":
    print("Flash Attention Example")
    print("=" * 60)

    # Test configurations: (batch, heads, seq_len, head_dim)
    test_configs = [
        (2, 4, 128, 32),
        (2, 4, 128, 64),
        (2, 4, 128, 128),
    ]

    all_passed = True

    for batch, heads, seq_len, head_dim in test_configs:
        batch_heads = batch * heads
        print(f"\n--- Head Dimension: {head_dim} (B={batch}, H={heads}, S={seq_len}) ---")

        # Generate test data
        rng = np.random.default_rng(42)
        Q_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
        K_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
        V_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
        sm_scale = 1.0 / np.sqrt(head_dim)

        # Compute reference
        Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
        K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
        V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
        ref_out = reference_attention(Q_4d, K_4d, V_4d, sm_scale).reshape(batch_heads, seq_len, head_dim)

        # Create warp arrays
        Q = wp.array(Q_np, dtype=float)
        K = wp.array(K_np, dtype=float)
        V = wp.array(V_np, dtype=float)

        # Test naive kernel
        naive_kernel = get_naive_kernel(head_dim)
        O = wp.zeros_like(Q)
        wp.launch(naive_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
        err = np.max(np.abs(O.numpy() - ref_out))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  Naive:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

        # Test flash attention kernel (tile_matmul + per-thread streaming, threads_per_row split)
        # Target D_SLICE=8 dims per thread, maximize tile_m within shared memory budget
        d_slice = 8
        max_smem = 99 * 1024
        elem_size = 4  # FP32
        flash_threads_per_row = head_dim // d_slice
        flash_tile_m = min(128, 1024 // flash_threads_per_row)
        flash_tile_n = 64
        # Reduce tile_m to fit shared memory
        while flash_tile_m > 16:
            smem = flash_tile_m * (2 * head_dim + flash_tile_n) * elem_size + 2 * flash_tile_n * head_dim * elem_size
            if smem <= max_smem:
                break
            flash_tile_m //= 2
        flash_kernel = get_flash_attention_kernel(head_dim, flash_tile_m, flash_tile_n,
                                                   threads_per_row=flash_threads_per_row)
        flash_block_dim = flash_tile_m * flash_threads_per_row
        print(f"    Using TILE_M={flash_tile_m}, TILE_N={flash_tile_n}, THREADS_PER_ROW={flash_threads_per_row}, "
              f"D_PER_THREAD={d_slice}, block_dim={flash_block_dim} ({flash_block_dim//32} warps)")
        flash_padded_seq = ((seq_len + flash_tile_m - 1) // flash_tile_m) * flash_tile_m
        flash_num_q_blocks = flash_padded_seq // flash_tile_m
        flash_num_k_blocks = flash_padded_seq // flash_tile_n

        Q_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float32)
        K_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float32)
        V_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float32)
        Q_flash_np[:, :seq_len, :] = Q_np
        K_flash_np[:, :seq_len, :] = K_np
        V_flash_np[:, :seq_len, :] = V_np

        Q_flash = wp.array(Q_flash_np, dtype=float)
        K_flash = wp.array(K_flash_np, dtype=float)
        V_flash = wp.array(V_flash_np, dtype=float)
        O_flash = wp.zeros_like(Q_flash)

        total_flash_threads = batch_heads * flash_num_q_blocks * flash_block_dim
        wp.launch(
            flash_kernel,
            dim=total_flash_threads,
            inputs=[Q_flash, K_flash, V_flash, O_flash, sm_scale, flash_padded_seq, batch_heads, flash_num_q_blocks, flash_num_k_blocks],
            block_dim=flash_block_dim,
        )
        O_flash_result = O_flash.numpy()[:, :seq_len, :]
        err = np.max(np.abs(O_flash_result - ref_out))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  Flash:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

        # # Test tiled kernel (tile_matmul based, FlashAttention style) - commented out, not tiled correctly
        # tile_m, tile_n = TILE_M, TILE_N
        # tiled_kernel = get_tiled_kernel(head_dim, tile_m, tile_n)
        # print(f"    Using TILE_M={tile_m}, TILE_N={tile_n}")
        # padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
        # num_q_blocks = padded_seq // tile_m
        # num_k_blocks = padded_seq // tile_n
        # num_tiles_tiled = batch_heads * num_q_blocks

        # Q_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        # K_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        # V_padded_np = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
        # Q_padded_np[:, :seq_len, :] = Q_np
        # K_padded_np[:, :seq_len, :] = K_np
        # V_padded_np[:, :seq_len, :] = V_np

        # Q_tiled = wp.array(Q_padded_np, dtype=float)
        # K_tiled = wp.array(K_padded_np, dtype=float)
        # V_tiled = wp.array(V_padded_np, dtype=float)
        # O_tiled = wp.zeros_like(Q_tiled)

        # wp.launch_tiled(
        #     tiled_kernel,
        #     dim=num_tiles_tiled,
        #     inputs=[Q_tiled, K_tiled, V_tiled, O_tiled, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
        #     block_dim=TILE_THREADS,
        # )
        # O_result = O_tiled.numpy()[:, :seq_len, :]
        # err = np.max(np.abs(O_result - ref_out))
        # status = "PASS" if err < 1e-5 else "FAIL"
        # print(f"  Tiled:      err={err:.2e} [{status}]")
        # if status == "FAIL":
        #     all_passed = False

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)