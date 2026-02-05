# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Flash Attention v2 implementation in Warp.

Implementations:
1. Naive 3-pass attention (baseline)
2. Flash Attention FP16 - tile_matmul for QK^T and PV with per-thread scalar softmax

Supports head dimensions: 32, 64, 128, 256
Input shape: [batch_heads, seq_len, head_dim]
"""

import numpy as np
import warp as wp

# Constants used by kernels
NEG_INF = wp.constant(-1.0e10)
NEG_INF_F16 = wp.constant(wp.float16(-65504.0))


@wp.func
def mul_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
    return a * b


@wp.func
def div_func_f16(a: wp.float16, b: wp.float16) -> wp.float16:
    return a / b


# Supported head dimensions
SUPPORTED_HEAD_DIMS = [32, 64, 128, 256]


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


def create_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32):
    """Fully tile-based flash attention using tile_matmul, tile_reduce, tile_map, tile_broadcast.

    Uses tensor cores (via tile_matmul) for the two matrix multiplies:
    - S = Q @ K^T  [TILE_M, HEAD_DIM] @ [HEAD_DIM, TILE_N] -> [TILE_M, TILE_N]
    - PV = P @ V   [TILE_M, TILE_N] @ [TILE_N, HEAD_DIM] -> [TILE_M, HEAD_DIM]

    Uses FP16 for tensor core operations with FP32 accumulation for numerical precision.

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size (also determines block_dim)
        tile_n: Key/Value block size
    """

    @wp.kernel(enable_backward=False)
    def flash_attention_kernel(
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
        # TILE_M_LOCAL = wp.static(tile_m)
        # TILE_N_LOCAL = wp.static(tile_n)
        # HEAD_DIM_LOCAL = wp.static(head_dim)

        # tile_idx = wp.tid() // TILE_M_LOCAL
        # batch_head = tile_idx // num_q_blocks
        # q_block_idx = tile_idx % num_q_blocks
        # q_start = q_block_idx * TILE_M_LOCAL

        # if batch_head >= batch_heads:
        #     return

        # Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        # Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=wp.float16, storage="shared")
        # m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=NEG_INF, dtype=wp.float16, storage="shared")
        # l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=wp.float16, storage="shared")

        # for k_block in range(num_k_blocks):
        #     k_start = k_block * TILE_N_LOCAL

        #     K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
        #     K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
        #     V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
        #     V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

        #     K_T = wp.tile_transpose(K_tile)
        #     S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
        #     wp.tile_matmul(Q_tile, K_T, S)
        #     S = S * wp.float16(sm_scale)

        #     m_block = wp.tile_reduce(wp.max, S, axis=1)
        #     m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))
        #     m_new = wp.tile_map(wp.max, m_i, m_block)

        #     alpha = wp.tile_map(wp.exp, m_i - m_new)
        #     m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
        #     P = wp.tile_map(wp.exp, S - m_broadcast)

        #     l_block = wp.tile_sum(P, axis=1)
        #     l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))
        #     l_new = wp.tile_map(mul_func_f16, l_i, alpha) + l_block
        #     wp.tile_assign_to_shared(l_i, l_new)

        #     alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        #     O_scaled = wp.tile_map(mul_func_f16, O_acc, alpha_broadcast)

        #     # Matmul: temp = P @ V
        #     temp_PV = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=wp.float16)
        #     wp.tile_matmul(P, V_tile, temp_PV)

        #     # O_acc = O_scaled + temp_PV
        #     O_acc_new = O_scaled + temp_PV
        #     wp.tile_assign_to_shared(O_acc, O_acc_new)

        #     wp.tile_assign_to_shared(m_i, m_new)

        # l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        # O_final = wp.tile_map(div_func_f16, O_acc, l_broadcast)

        # O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
        # wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

        """Flash Attention FP16 - hybrid tile_matmul + per-thread softmax.

        tile_matmul for QK^T and PV (fp16 tensor cores).
        Per-thread scalar softmax in fp32 for numerical precision.
        """
        TILE_M = wp.static(tile_m)
        TILE_N = wp.static(tile_n)
        HEAD_DIM = wp.static(head_dim)

        tile_idx = wp.tid() // TILE_M
        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks
        q_start = q_block_idx * TILE_M
        row = wp.tid() % TILE_M

        if batch_head >= batch_heads:
            return

        sm_scale_log2 = sm_scale * 1.44269504

        # Load Q tile [TILE_M, HEAD_DIM] fp16 shared
        Q_tile = wp.tile_squeeze(
            wp.tile_load(Q, shape=(1, TILE_M, HEAD_DIM), offset=(batch_head, q_start, 0)),
            axis=(0,)
        )

        # Per-thread scalar accumulators (fp32 for precision)
        m_i = float(NEG_INF)
        l_i = float(0.0)
        o_acc = wp.vector(dtype=float, length=HEAD_DIM)
        for d in range(HEAD_DIM):
            o_acc[d] = float(0.0)

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N

            # Load K tile [TILE_N, HEAD_DIM] fp16 shared
            K_tile = wp.tile_squeeze(
                wp.tile_load(K, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0)),
                axis=(0,)
            )

            # S = Q @ K^T [TILE_M, TILE_N] fp16 tensor cores
            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float16, storage="shared")
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Load V tile early so global memory fetch overlaps with softmax compute
            V_tile = wp.tile_squeeze(
                wp.tile_load(V, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0)),
                axis=(0,)
            )

            # Row-wise max using tile_reduce, then scale
            m_block_tile = wp.tile_reduce(wp.max, S_tile, axis=1)  # (TILE_M,) fp16
            m_block = float(m_block_tile[row]) * sm_scale_log2

            # Online softmax update
            m_new = wp.max(m_i, m_block)
            alpha = wp.exp2(m_i - m_new)

            # Rescale accumulators
            l_i = l_i * alpha
            for d in range(HEAD_DIM):
                o_acc[d] = o_acc[d] * alpha

            # Per-thread: compute P (fp32) and write back as fp16
            l_block = float(0.0)
            for n in range(TILE_N):
                s_val = float(S_tile[row, n]) * sm_scale_log2
                p = wp.exp2(s_val - m_new)
                S_tile[row, n] = wp.float16(p)
                l_block = l_block + p
            l_i = l_i + l_block

            # PV = P @ V [TILE_M, HEAD_DIM] fp16 tensor cores
            PV_tile = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16, storage="shared")
            wp.tile_matmul(S_tile, V_tile, PV_tile)

            # Per-thread: accumulate PV into o_acc (fp32)
            for d in range(HEAD_DIM):
                o_acc[d] = o_acc[d] + float(PV_tile[row, d])

            m_i = m_new

        # Final normalization and store as fp16
        inv_l = float(1.0) / l_i
        for d in range(HEAD_DIM):
            O[batch_head, q_start + row, d] = wp.float16(o_acc[d] * inv_l)

    return flash_attention_kernel


def get_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32):
    """Get or create flash attention kernel (tile_matmul + per-thread streaming).

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size
        tile_n: Key/Value block size
    """
    key = ("flash", head_dim, tile_m, tile_n)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_kernel(head_dim, tile_m, tile_n)
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

        # Test flash attention kernel (fully tile-based, block_dim = tile_m)
        flash_tile_m = 32
        flash_tile_n = 32

        flash_block_dim = flash_tile_m  # block_dim = tile_m (no threads_per_row)
        flash_kernel = get_flash_attention_kernel(head_dim, flash_tile_m, flash_tile_n)
        print(f"    Using TILE_M={flash_tile_m}, TILE_N={flash_tile_n}, "
              f"block_dim={flash_block_dim} ({flash_block_dim//32} warps)")
        flash_padded_seq = ((seq_len + flash_tile_m - 1) // flash_tile_m) * flash_tile_m
        flash_num_q_blocks = flash_padded_seq // flash_tile_m
        flash_num_k_blocks = flash_padded_seq // flash_tile_n

        Q_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float16)
        K_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float16)
        V_flash_np = np.zeros((batch_heads, flash_padded_seq, head_dim), dtype=np.float16)
        Q_flash_np[:, :seq_len, :] = Q_np.astype(np.float16)
        K_flash_np[:, :seq_len, :] = K_np.astype(np.float16)
        V_flash_np[:, :seq_len, :] = V_np.astype(np.float16)

        Q_flash = wp.array(Q_flash_np, dtype=wp.float16)
        K_flash = wp.array(K_flash_np, dtype=wp.float16)
        V_flash = wp.array(V_flash_np, dtype=wp.float16)
        O_flash = wp.zeros_like(Q_flash)

        total_flash_threads = batch_heads * flash_num_q_blocks * flash_block_dim
        wp.launch(
            flash_kernel,
            dim=total_flash_threads,
            inputs=[Q_flash, K_flash, V_flash, O_flash, sm_scale, flash_padded_seq, batch_heads, flash_num_q_blocks, flash_num_k_blocks],
            block_dim=flash_block_dim,
        )
        O_flash_result = O_flash.numpy()[:, :seq_len, :].astype(np.float32)
        err = np.max(np.abs(O_flash_result - ref_out))
        status = "PASS" if err < 1e-2 else "FAIL"  # fp16 has lower precision
        print(f"  Flash:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
