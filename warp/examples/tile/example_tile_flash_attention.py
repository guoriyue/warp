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


def create_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                   threads_per_row: int = 1):
    """Flash attention: tile_matmul for QK^T + per-thread scalar softmax/PV.

    Uses tensor cores (via tile_matmul) for S = Q @ K^T.
    Softmax and PV accumulation are done per-thread in scalar code with FP32
    accumulators in registers — avoiding both __syncthreads overhead and the
    shared memory round-trip that cuBLASDx SMEM API would impose on the O accumulator.

    Multiple threads per row can cooperate to split the HEAD_DIM V accumulation,
    reducing the inner loop from TILE_N * HEAD_DIM to TILE_N * (HEAD_DIM / threads_per_row).

    FP16 I/O with FP32 internal accumulation. Uses exp2 for fast math.

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size
        tile_n: Key/Value block size
        threads_per_row: Number of threads cooperating per Q row (default 1).
            Each thread handles head_dim // threads_per_row elements of V.
            block_dim = tile_m * threads_per_row.
    """
    assert head_dim % threads_per_row == 0
    d_per_thread = head_dim // threads_per_row

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

        # Pre-multiply sm_scale by LOG2E for exp2
        sm_scale_log2 = sm_scale * 1.44269504

        # Load Q block as tile (reused across all K/V blocks)
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Per-thread accumulators in FP32 registers (only D_PER_THREAD elements)
        m_i = float(-1e10)
        l_i = float(0.0)
        o_acc = wp.vector(dtype=float, length=D_PER_THREAD)
        for d in range(D_PER_THREAD):
            o_acc[d] = 0.0

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            # Load K/V blocks
            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            # S = Q @ K^T via tensor cores [TILE_M, TILE_N]
            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Per-thread scalar online softmax (no syncs)
            # All threads in a row redundantly compute softmax state (m, l, p)
            m_block = float(-1e10)
            for n in range(TILE_N_LOCAL):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                m_block = wp.max(m_block, s)

            m_new = wp.max(m_i, m_block)
            alpha = wp.exp2(m_i - m_new)

            # Rescale accumulators (register ops, no SMEM traffic)
            l_i = l_i * alpha
            for d in range(D_PER_THREAD):
                o_acc[d] = o_acc[d] * alpha

            # Fused softmax + V accumulation (per-thread streaming, FP32)
            # Each thread accumulates only its d-slice of o_acc
            for n in range(TILE_N_LOCAL):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                p = wp.exp2(s - m_new)
                l_i = l_i + p
                for d in range(D_PER_THREAD):
                    o_acc[d] = o_acc[d] + p * float(V_tile[n, d_start + d])

            m_i = m_new

        # Final normalization and write output (FP32 -> FP16)
        for d in range(D_PER_THREAD):
            O[batch_head, q_row, d_start + d] = wp.float16(o_acc[d] / l_i)

    return flash_attention_kernel


def get_flash_attention_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                threads_per_row: int = 1):
    """Get or create flash attention kernel (tile_matmul + per-thread streaming).

    Args:
        head_dim: Head dimension (must be in SUPPORTED_HEAD_DIMS)
        tile_m: Query block size
        tile_n: Key/Value block size
        threads_per_row: Number of threads cooperating per Q row
    """
    key = ("flash", head_dim, tile_m, tile_n, threads_per_row)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_kernel(head_dim, tile_m, tile_n, threads_per_row)
    return _kernel_cache[key]



def create_flash_attention_cached_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                          threads_per_row: int = 1):
    """Flash attention with register-cached S_tile values.

    Same as create_flash_attention_kernel but caches S_tile[row, :] in a register
    vector to avoid double SMEM reads (once for max, once for PV).
    """
    assert head_dim % threads_per_row == 0
    d_per_thread = head_dim // threads_per_row

    @wp.kernel(enable_backward=False)
    def flash_attention_cached_kernel(
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
        TILE_M_LOCAL = wp.static(tile_m)
        TILE_N_LOCAL = wp.static(tile_n)
        HEAD_DIM_LOCAL = wp.static(head_dim)
        THREADS_PER_ROW = wp.static(threads_per_row)
        D_PER_THREAD = wp.static(d_per_thread)

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

        sm_scale_log2 = sm_scale * 1.44269504

        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        m_i = float(-1e10)
        l_i = float(0.0)
        o_acc = wp.vector(dtype=float, length=D_PER_THREAD)
        for d in range(D_PER_THREAD):
            o_acc[d] = 0.0

        # Register cache for S_tile row (avoids double SMEM read)
        s_cache = wp.vector(dtype=float, length=TILE_N_LOCAL)

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Pass 1: Read S_tile into register cache + find max
            m_block = float(-1e10)
            for n in range(TILE_N_LOCAL):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                s_cache[n] = s
                m_block = wp.max(m_block, s)

            m_new = wp.max(m_i, m_block)
            alpha = wp.exp2(m_i - m_new)

            l_i = l_i * alpha
            for d in range(D_PER_THREAD):
                o_acc[d] = o_acc[d] * alpha

            # Pass 2: Use register cache (no S_tile SMEM reads)
            for n in range(TILE_N_LOCAL):
                p = wp.exp2(s_cache[n] - m_new)
                l_i = l_i + p
                for d in range(D_PER_THREAD):
                    o_acc[d] = o_acc[d] + p * float(V_tile[n, d_start + d])

            m_i = m_new

        for d in range(D_PER_THREAD):
            O[batch_head, q_row, d_start + d] = wp.float16(o_acc[d] / l_i)

    return flash_attention_cached_kernel


# Cache for cached kernels
_cached_kernel_cache = {}


def get_cached_kernel(head_dim: int, tile_m: int = None, tile_n: int = None,
                      threads_per_row: int = None):
    if tile_m is None or tile_n is None or threads_per_row is None:
        from warp.examples.benchmarks.benchmark_flash_attention import compute_flash_config
        tile_m, tile_n, threads_per_row = compute_flash_config(head_dim)
    key = ("cached", head_dim, tile_m, tile_n, threads_per_row)
    if key not in _cached_kernel_cache:
        _cached_kernel_cache[key] = create_flash_attention_cached_kernel(
            head_dim, tile_m, tile_n, threads_per_row)
    return _cached_kernel_cache[key]


def create_flash_attention_matmul_only_kernel(head_dim: int, tile_m: int = 32, tile_n: int = 32,
                                               threads_per_row: int = 1):
    """Benchmark-only kernel: measures tile_matmul QK^T cost without PV loop.

    Used to isolate tile_matmul overhead from scalar PV overhead.
    """
    assert head_dim % threads_per_row == 0
    d_per_thread = head_dim // threads_per_row

    @wp.kernel(enable_backward=False)
    def flash_attention_matmul_only_kernel(
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
        TILE_M_LOCAL = wp.static(tile_m)
        TILE_N_LOCAL = wp.static(tile_n)
        HEAD_DIM_LOCAL = wp.static(head_dim)
        THREADS_PER_ROW = wp.static(threads_per_row)
        D_PER_THREAD = wp.static(d_per_thread)

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

        sm_scale_log2 = sm_scale * 1.44269504

        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        m_i = float(-1e10)
        l_i = float(0.0)
        o_acc = wp.vector(dtype=float, length=D_PER_THREAD)
        for d in range(D_PER_THREAD):
            o_acc[d] = 0.0

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N_LOCAL

            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))

            # S = Q @ K^T via tensor cores only — no PV, no softmax
            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=wp.float16)
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Minimal work to prevent dead-code elimination: read one element
            m_i = wp.max(m_i, float(S_tile[row_in_tile, 0]) * sm_scale_log2)

        # Write a single value per thread to prevent DCE
        for d in range(D_PER_THREAD):
            O[batch_head, q_row, d_start + d] = wp.float16(m_i)

    return flash_attention_matmul_only_kernel


# Cache for matmul-only kernels
_matmul_only_kernel_cache = {}


def get_matmul_only_kernel(head_dim: int, tile_m: int = None, tile_n: int = None,
                           threads_per_row: int = None):
    if tile_m is None or tile_n is None or threads_per_row is None:
        from warp.examples.benchmarks.benchmark_flash_attention import compute_flash_config
        tile_m, tile_n, threads_per_row = compute_flash_config(head_dim)
    key = ("matmul_only", head_dim, tile_m, tile_n, threads_per_row)
    if key not in _matmul_only_kernel_cache:
        _matmul_only_kernel_cache[key] = create_flash_attention_matmul_only_kernel(
            head_dim, tile_m, tile_n, threads_per_row)
    return _matmul_only_kernel_cache[key]


def create_flash_attention_smem2_kernel(head_dim: int, tile_m: int = 128, tile_n: int = 64,
                                         block_dim: int = 128):
    """Flash attention with tensor-core PV using SMEM O accumulator.

    Uses tile_matmul for BOTH QK^T and PV. The O accumulator stays in shared
    memory, avoiding cuBLASDx RMEM layout-transform overhead.

    Per k_block:
      1. tile_matmul(Q, K^T, S_tile)    — tensor cores for QK^T
      2. Scalar softmax over S_tile rows  — per-thread, no sync needed
      3. O_acc *= alpha                   — element-wise SMEM rescale
      4. Write P to S_tile               — reuse for PV matmul
      5. tile_matmul(S_tile, V, O_acc)   — tensor cores for PV (accumulate)

    Args:
        head_dim: Head dimension (32, 64, 128, 256)
        tile_m: Query block size (rows per Q tile)
        tile_n: Key/Value block size
        block_dim: Thread block size (= tile_m, one thread per row)
    """

    @wp.kernel(enable_backward=False)
    def flash_attention_smem2_kernel(
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
        TILE_M = wp.static(tile_m)
        TILE_N = wp.static(tile_n)
        HEAD_DIM = wp.static(head_dim)

        # Thread indexing: one thread per Q row
        tile_idx = wp.tid() // TILE_M
        row_in_tile = wp.tid() % TILE_M

        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks
        q_start = q_block_idx * TILE_M
        q_row = q_start + row_in_tile

        if batch_head >= batch_heads:
            return
        if q_row >= seq_len:
            return

        sm_scale_log2 = sm_scale * 1.44269504

        # Load Q block (reused across all K/V blocks)
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M, HEAD_DIM), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Per-thread softmax state (FP32 registers)
        m_i = float(-1e10)
        l_i = float(0.0)

        # O accumulator in shared memory — persists across k_blocks
        O_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16)

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N

            # Load K/V blocks
            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0))
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            # S = Q @ K^T via tensor cores [TILE_M, TILE_N]
            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float16, storage="shared")
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Per-thread scalar online softmax (find block max)
            m_block = float(-1e10)
            for n in range(TILE_N):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                m_block = wp.max(m_block, s)

            m_new = wp.max(m_i, m_block)
            alpha = wp.exp2(m_i - m_new)

            # Rescale O accumulator (SMEM read-modify-write) and running sum
            l_i = l_i * alpha
            for d in range(HEAD_DIM):
                O_acc[row_in_tile, d] = wp.float16(float(O_acc[row_in_tile, d]) * alpha)
            wp.tile_sync()

            # Write softmax P values to S_tile (reuse for PV matmul), accumulate l_i
            for n in range(TILE_N):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                p = wp.exp2(s - m_new)
                l_i = l_i + p
                S_tile[row_in_tile, n] = wp.float16(p)

            wp.tile_sync()

            # PV accumulation via tensor cores: O_acc += S_tile @ V_tile
            wp.tile_matmul(S_tile, V_tile, O_acc)

            m_i = m_new

        # Final normalization and write output
        for d in range(HEAD_DIM):
            O[batch_head, q_row, d] = wp.float16(float(O_acc[row_in_tile, d]) / l_i)

    return flash_attention_smem2_kernel


# Cache for SMEM2 kernels
_smem2_kernel_cache = {}


def compute_smem2_config(head_dim: int):
    """Compute (tile_m, tile_n, block_dim) for SMEM2 kernel."""
    tile_m = 128
    tile_n = 64
    block_dim = tile_m  # One thread per row
    return tile_m, tile_n, block_dim


def get_smem2_kernel(head_dim: int, tile_m: int = None, tile_n: int = None,
                     block_dim: int = None):
    """Get or create SMEM2 flash attention kernel."""
    if tile_m is None or tile_n is None or block_dim is None:
        tile_m, tile_n, block_dim = compute_smem2_config(head_dim)
    key = (head_dim, tile_m, tile_n, block_dim)
    if key not in _smem2_kernel_cache:
        _smem2_kernel_cache[key] = create_flash_attention_smem2_kernel(head_dim, tile_m, tile_n, block_dim)
    return _smem2_kernel_cache[key]


def create_flash_attention_rmem_kernel(head_dim: int, tile_m: int = 128, tile_n: int = 64,
                                        block_dim: int = 128):
    """Flash attention with RMEM PV accumulation via cuBLASDx Tensor API.

    Uses tile_matmul (SMEM) for QK^T and tile_matmul (RMEM) for PV.
    The O accumulator lives in register memory (cuBLASDx SUGGESTED_RMEM_C),
    avoiding shared memory round-trips per K-block.

    tile_matmul automatically detects storage="register" on the output tile
    and dispatches to the RMEM path. tile_scale and tile_copy provide
    generic element-wise scaling and rmem→shared copy.

    Args:
        head_dim: Head dimension (32, 64, 128, 256)
        tile_m: Query block size
        tile_n: Key/Value block size
        block_dim: Thread block size
    """

    @wp.kernel(enable_backward=False)
    def flash_attention_rmem_kernel(
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
        TILE_M = wp.static(tile_m)
        TILE_N = wp.static(tile_n)
        HEAD_DIM = wp.static(head_dim)

        # Thread indexing: block_dim threads per tile block
        tile_idx = wp.tid() // TILE_M
        row_in_tile = wp.tid() % TILE_M

        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks
        q_start = q_block_idx * TILE_M
        q_row = q_start + row_in_tile

        if batch_head >= batch_heads:
            return
        if q_row >= seq_len:
            return

        sm_scale_log2 = sm_scale * 1.44269504

        # Load Q block as tile (reused across all K/V blocks)
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M, HEAD_DIM), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Per-thread softmax state (FP32 registers)
        m_i = float(-1e10)
        l_i = float(0.0)

        # O accumulator in register memory — rmem_storage_bytes determined by tile_matmul dispatch
        # accumulator=TILE_N specifies K dimension for the PV matmul: S[M,N] @ V[N,D] → O[M,D]
        O_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16, storage="register", accumulator=TILE_N)

        for k_block in range(num_k_blocks):
            k_start = k_block * TILE_N

            # Load K/V blocks
            K_tile_3d = wp.tile_load(K, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0))
            K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
            V_tile_3d = wp.tile_load(V, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0))
            V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

            # S = Q @ K^T via tensor cores [TILE_M, TILE_N] — shared output for element access
            K_T = wp.tile_transpose(K_tile)
            S_tile = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float16, storage="shared")
            wp.tile_matmul(Q_tile, K_T, S_tile)

            # Per-thread scalar online softmax
            m_block = float(-1e10)
            for n in range(TILE_N):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                m_block = wp.max(m_block, s)

            m_new = wp.max(m_i, m_block)
            alpha = wp.exp2(m_i - m_new)

            # Rescale O accumulator and running sum
            l_i = l_i * alpha
            wp.tile_scale(O_acc, alpha)

            # Write P values to S_tile (reuse for PV matmul), accumulate l_i
            for n in range(TILE_N):
                s = float(S_tile[row_in_tile, n]) * sm_scale_log2
                p = wp.exp2(s - m_new)
                l_i = l_i + p
                S_tile[row_in_tile, n] = wp.float16(p)

            wp.tile_sync()

            # PV accumulation via tensor cores: O_acc += S_tile @ V_tile (RMEM path)
            wp.tile_matmul(S_tile, V_tile, O_acc)

            m_i = m_new

        # Copy register → shared for output
        O_shared = wp.tile_copy(O_acc)

        # Final normalization and write output (FP32 -> FP16)
        for d in range(HEAD_DIM):
            O[batch_head, q_row, d] = wp.float16(float(O_shared[row_in_tile, d]) / l_i)

    return flash_attention_rmem_kernel


def get_flash_attention_rmem_kernel(head_dim: int, tile_m: int = 128, tile_n: int = 64,
                                     block_dim: int = 128):
    """Get or create RMEM flash attention kernel.

    Currently uses scalar PV (same as non-RMEM kernel) as a stepping stone.
    RMEM tensor-core PV accumulation will be integrated once LTO pipeline is validated.
    """
    key = ("flash_rmem", head_dim, tile_m, tile_n, block_dim)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_rmem_kernel(head_dim, tile_m, tile_n, block_dim)
    return _kernel_cache[key]


def create_flash_attention_mma_kernel(head_dim: int, tile_m: int = 128, tile_n: int = 64,
                                      num_warps: int = 8):
    """Factory for native PTX MMA flash attention kernel.

    Uses inline PTX mma.sync.aligned.m16n8k16 instructions instead of cuBLASDx LTO.
    The entire flash attention loop (QK^T, softmax, PV) runs in a fused C++ kernel.

    Args:
        head_dim: Head dimension (32, 64, 128, 256)
        tile_m: Q block size (rows per CTA), default 128
        tile_n: K/V block size (cols per inner loop), default 64
        num_warps: Number of warps per CTA, default 4
    """
    BLOCK_DIM = num_warps * 32

    @wp.kernel(enable_backward=False)
    def flash_attention_mma_kernel(
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
        TILE_M = wp.static(tile_m)
        TILE_N = wp.static(tile_n)
        HEAD_DIM = wp.static(head_dim)
        BD = wp.static(BLOCK_DIM)

        # Each CTA = one (batch_head, q_block) pair
        tile_idx = wp.tid() // BD
        batch_head = tile_idx // num_q_blocks
        q_block_idx = tile_idx % num_q_blocks

        if batch_head >= batch_heads:
            return

        wp.tile_flash_attention_mma(
            Q, K, V, O, sm_scale, seq_len, batch_heads,
            num_q_blocks, num_k_blocks, batch_head, q_block_idx,
            TILE_M, TILE_N, HEAD_DIM,
        )

    return flash_attention_mma_kernel


def get_flash_attention_mma_kernel(head_dim: int, tile_m: int = 128, tile_n: int = 64,
                                   num_warps: int = 8):
    """Get or create native MMA flash attention kernel."""
    key = ("flash_mma", head_dim, tile_m, tile_n, num_warps)
    if key not in _kernel_cache:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim={head_dim} not supported. Use one of {SUPPORTED_HEAD_DIMS}")
        _kernel_cache[key] = create_flash_attention_mma_kernel(head_dim, tile_m, tile_n, num_warps)
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

        # Test flash attention kernel (tile_matmul QK^T + scalar softmax/PV, FP16)
        d_slice = 8
        flash_threads_per_row = head_dim // d_slice
        flash_tile_m = min(128, 1024 // flash_threads_per_row)
        flash_tile_n = 64

        flash_block_dim = flash_tile_m * flash_threads_per_row
        flash_kernel = get_flash_attention_kernel(head_dim, flash_tile_m, flash_tile_n, flash_threads_per_row)
        print(f"    Using TILE_M={flash_tile_m}, TILE_N={flash_tile_n}, "
              f"THREADS_PER_ROW={flash_threads_per_row}, D_PER_THREAD={d_slice}, "
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

        total_threads = batch_heads * flash_num_q_blocks * flash_block_dim
        wp.launch(
            flash_kernel,
            dim=total_threads,
            inputs=[Q_flash, K_flash, V_flash, O_flash, sm_scale, flash_padded_seq, batch_heads, flash_num_q_blocks, flash_num_k_blocks],
            block_dim=flash_block_dim,
        )
        O_flash_result = O_flash.numpy()[:, :seq_len, :].astype(np.float32)
        err = np.max(np.abs(O_flash_result - ref_out))
        status = "PASS" if err < 1e-2 else "FAIL"  # fp16 has lower precision
        print(f"  Flash:      err={err:.2e} [{status}]")
        if status == "FAIL":
            all_passed = False

        # Test RMEM flash attention kernel (tile_matmul PV in register memory)
        if head_dim == 64:  # RMEM kernel currently validated for head_dim=64
            rmem_tile_m = 128
            rmem_tile_n = 64
            rmem_block_dim = 128

            rmem_kernel = get_flash_attention_rmem_kernel(head_dim, rmem_tile_m, rmem_tile_n, rmem_block_dim)
            rmem_padded_seq = ((seq_len + rmem_tile_m - 1) // rmem_tile_m) * rmem_tile_m
            rmem_num_q_blocks = rmem_padded_seq // rmem_tile_m
            rmem_num_k_blocks = rmem_padded_seq // rmem_tile_n

            Q_rmem_np = np.zeros((batch_heads, rmem_padded_seq, head_dim), dtype=np.float16)
            K_rmem_np = np.zeros((batch_heads, rmem_padded_seq, head_dim), dtype=np.float16)
            V_rmem_np = np.zeros((batch_heads, rmem_padded_seq, head_dim), dtype=np.float16)
            Q_rmem_np[:, :seq_len, :] = Q_np.astype(np.float16)
            K_rmem_np[:, :seq_len, :] = K_np.astype(np.float16)
            V_rmem_np[:, :seq_len, :] = V_np.astype(np.float16)

            Q_rmem = wp.array(Q_rmem_np, dtype=wp.float16)
            K_rmem = wp.array(K_rmem_np, dtype=wp.float16)
            V_rmem = wp.array(V_rmem_np, dtype=wp.float16)
            O_rmem = wp.zeros_like(Q_rmem)

            rmem_total_threads = batch_heads * rmem_num_q_blocks * rmem_tile_m
            wp.launch(
                rmem_kernel,
                dim=rmem_total_threads,
                inputs=[Q_rmem, K_rmem, V_rmem, O_rmem, sm_scale, rmem_padded_seq, batch_heads, rmem_num_q_blocks, rmem_num_k_blocks],
                block_dim=rmem_block_dim,
            )
            O_rmem_result = O_rmem.numpy()[:, :seq_len, :].astype(np.float32)
            err = np.max(np.abs(O_rmem_result - ref_out))
            status = "PASS" if err < 0.2 else "FAIL"  # RMEM uses tensor-core PV, fp16 precision
            print(f"  Flash RMEM: err={err:.2e} [{status}]")
            if status == "FAIL":
                all_passed = False

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
