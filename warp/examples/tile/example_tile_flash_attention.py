# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Flash Attention v2 implementation in Warp.

Implementations:
1. Naive 3-pass attention (baseline)
2. Flash Attention with online softmax (memory efficient)
3. Flash Attention SIMD - 32 threads cooperate per query
4. Flash Attention tiled - tile_matmul based (tensor cores + SRAM)

Supports head dimensions: 32, 64, 128, 256
Input shape: [batch_heads, seq_len, head_dim]
"""

import numpy as np
import warp as wp

# Constants used by kernels
NEG_INF = wp.constant(-1.0e10)

# Default tile sizes for tile_matmul-based flash attention
TILE_M = 16    # Query block size
TILE_N = 16    # Key/Value block size
TILE_THREADS = 64  # Threads per block for tile operations

# Supported head dimensions
SUPPORTED_HEAD_DIMS = [32, 64, 128, 256]


# Helper functions for tile_map operations
@wp.func
def mul_func(a: float, b: float) -> float:
    return a * b


@wp.func
def div_func(a: float, b: float) -> float:
    return a / b


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


def create_flash_attention_kernel(head_dim: int):
    """Factory to create flash attention kernel for specific head_dim."""
    @wp.kernel(enable_backward=False)
    def flash_attention_kernel(
        Q: wp.array3d(dtype=float),
        K: wp.array3d(dtype=float),
        V: wp.array3d(dtype=float),
        O: wp.array3d(dtype=float),
        sm_scale: float,
        seq_len: int,
    ):
        """Flash Attention with online softmax."""
        query_idx, batch_head = wp.tid()
        if query_idx >= seq_len:
            return

        HEAD_DIM_LOCAL = wp.static(head_dim)

        m_i = float(NEG_INF)
        l_i = float(0.0)

        acc = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
        for d in range(HEAD_DIM_LOCAL):
            acc[d] = float(0.0)

        q = wp.vector(dtype=float, length=HEAD_DIM_LOCAL)
        for d in range(HEAD_DIM_LOCAL):
            q[d] = Q[batch_head, query_idx, d]

        for k_idx in range(seq_len):
            qk = float(0.0)
            for d in range(HEAD_DIM_LOCAL):
                qk += q[d] * K[batch_head, k_idx, d]
            qk = qk * sm_scale

            m_new = wp.max(m_i, qk)
            alpha = wp.exp(m_i - m_new)
            p = wp.exp(qk - m_new)
            l_i = l_i * alpha + p

            for d in range(HEAD_DIM_LOCAL):
                acc[d] = acc[d] * alpha + p * V[batch_head, k_idx, d]

            m_i = m_new

        for d in range(HEAD_DIM_LOCAL):
            O[batch_head, query_idx, d] = acc[d] / l_i

    return flash_attention_kernel


def create_flash_attention_simd_kernel(head_dim: int):
    """Factory to create SIMD flash attention kernel for specific head_dim."""
    d_per_lane = head_dim // 32  # 32 threads per warp

    @wp.kernel(enable_backward=False)
    def flash_attention_simd_kernel(
        Q: wp.array3d(dtype=float),
        K: wp.array3d(dtype=float),
        V: wp.array3d(dtype=float),
        O: wp.array3d(dtype=float),
        sm_scale: float,
        seq_len: int,
        batch_heads: int,
    ):
        """Flash Attention SIMD - 32 threads cooperate per query via warp_reduce_sum."""
        work_idx = wp.tid()
        lane_id = wp.warp_lane_id()

        query_idx = work_idx % seq_len
        batch_head = work_idx // seq_len
        if batch_head >= batch_heads:
            return

        D_PER_LANE_LOCAL = wp.static(d_per_lane)
        d_start = lane_id * D_PER_LANE_LOCAL

        # Load Q elements for this lane
        q_local = wp.vector(dtype=float, length=D_PER_LANE_LOCAL)
        for i in range(D_PER_LANE_LOCAL):
            q_local[i] = Q[batch_head, query_idx, d_start + i]

        m_i = float(NEG_INF)
        l_i = float(0.0)
        acc_local = wp.vector(dtype=float, length=D_PER_LANE_LOCAL)
        for i in range(D_PER_LANE_LOCAL):
            acc_local[i] = float(0.0)

        for k_idx in range(seq_len):
            # Load K elements and compute partial dot product
            partial_qk = float(0.0)
            for i in range(D_PER_LANE_LOCAL):
                k_val = K[batch_head, k_idx, d_start + i]
                partial_qk += q_local[i] * k_val

            qk_val = wp.warp_reduce_sum(partial_qk) * sm_scale

            m_new = wp.max(m_i, qk_val)
            alpha = wp.exp(m_i - m_new)
            p = wp.exp(qk_val - m_new)
            l_i = l_i * alpha + p

            # Load V elements and accumulate
            for i in range(D_PER_LANE_LOCAL):
                v_val = V[batch_head, k_idx, d_start + i]
                acc_local[i] = acc_local[i] * alpha + p * v_val

            m_i = m_new

        # Write output
        for i in range(D_PER_LANE_LOCAL):
            O[batch_head, query_idx, d_start + i] = acc_local[i] / l_i

    return flash_attention_simd_kernel


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


def get_flash_kernel(head_dim: int):
    """Get or create flash attention kernel for head_dim."""
    key = ("flash", head_dim)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_kernel(head_dim)
    return _kernel_cache[key]


def get_simd_kernel(head_dim: int):
    """Get or create SIMD flash attention kernel for head_dim."""
    key = ("simd", head_dim)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        if head_dim % 32 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 32 for SIMD kernel")
        _kernel_cache[key] = create_flash_attention_simd_kernel(head_dim)
    return _kernel_cache[key]


def create_flash_attention_tiled_kernel(head_dim: int, tile_m: int = 16, tile_n: int = 16):
    """Factory to create tiled flash attention kernel using tile_matmul.

    This implements the FlashAttention memory access pattern:
    - K/V blocks are loaded into SRAM (shared memory) once
    - Multiple Q rows in the same tile share the K/V data
    - Online softmax avoids materializing the full attention matrix

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

        Memory access pattern (FlashAttention style):
        - Q block loaded to registers (reused across K/V iterations)
        - K/V blocks loaded to SRAM (shared by all threads in block)
        - O, m, l accumulators in SRAM (updated incrementally)
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

        # Load Q block to registers - reused across all K/V blocks
        # Shape: (1, TILE_M, HEAD_DIM) -> squeeze to (TILE_M, HEAD_DIM)
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Initialize output accumulator in SRAM
        O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=float, storage="shared")

        # Initialize online softmax state per row in SRAM
        m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=float(NEG_INF), dtype=float, storage="shared")
        l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=float, storage="shared")

        # Iterate over K/V blocks (outer loop in FlashAttention paper)
        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            # Load K/V blocks to SRAM - shared by all threads in this block
            # This is the key optimization: K/V loaded from HBM to SRAM once,
            # then reused by all TILE_M query rows in this tile
            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            # Compute S = Q @ K^T on SRAM: (TILE_M, HEAD_DIM) @ (HEAD_DIM, TILE_N) = (TILE_M, TILE_N)
            K_T = wp.tile_transpose(K_tile)
            S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=float)
            wp.tile_matmul(Q_tile, K_T, S)

            # Scale attention scores
            S = S * sm_scale

            # Compute row-wise max for this block
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
            l_block = wp.tile_sum(P, axis=1)
            l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))

            # Update running sum: l_new = l_i * alpha + l_block
            l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
            wp.tile_assign(l_i, l_new)

            # Update output accumulator: O = O * alpha
            alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
            O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)
            wp.tile_assign(O_acc, O_scaled)

            # Add P @ V contribution: O += P @ V
            # (TILE_M, TILE_N) @ (TILE_N, HEAD_DIM) = (TILE_M, HEAD_DIM)
            wp.tile_matmul(P, V_tile, O_acc)

            # Update max for next iteration
            wp.tile_assign(m_i, m_new)

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

        # Test flash kernel (online softmax)
        flash_kernel = get_flash_kernel(head_dim)
        O = wp.zeros_like(Q)
        wp.launch(flash_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
        err = np.max(np.abs(O.numpy() - ref_out))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  Flash:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

        # Test SIMD kernel (warp-parallel)
        simd_kernel = get_simd_kernel(head_dim)
        O = wp.zeros_like(Q)
        wp.launch_tiled(simd_kernel, dim=seq_len * batch_heads, inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads], block_dim=32)
        err = np.max(np.abs(O.numpy() - ref_out))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  SIMD:       err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

        # Test tiled kernel (tile_matmul based, FlashAttention style)
        tiled_kernel = get_tiled_kernel(head_dim)
        tile_m, tile_n = TILE_M, TILE_N
        padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
        num_q_blocks = padded_seq // tile_m
        num_k_blocks = padded_seq // tile_n
        num_tiles_tiled = batch_heads * num_q_blocks

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

        wp.launch_tiled(
            tiled_kernel,
            dim=num_tiles_tiled,
            inputs=[Q_tiled, K_tiled, V_tiled, O_tiled, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
            block_dim=TILE_THREADS,
        )
        O_result = O_tiled.numpy()[:, :seq_len, :]
        err = np.max(np.abs(O_result - ref_out))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  Tiled:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
