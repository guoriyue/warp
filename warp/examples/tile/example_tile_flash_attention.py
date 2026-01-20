# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Flash Attention v2 implementation in Warp.

Three implementations:
1. Naive 3-pass attention (baseline)
2. Flash Attention with online softmax (memory efficient)
3. Flash Attention SIMD - 32 threads cooperate per query (fastest)

Input shape: [batch_heads, seq_len, head_dim] where head_dim=64
"""

import numpy as np
import warp as wp

HEAD_DIM = wp.constant(64)
D_PER_LANE = wp.constant(2)
NEG_INF = wp.constant(-1.0e10)


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

    q = wp.types.vector(dtype=float, length=HEAD_DIM)
    for d in range(int(HEAD_DIM)):
        q[d] = Q[batch_head, query_idx, d]

    max_score = float(NEG_INF)
    for k_idx in range(seq_len):
        score = float(0.0)
        for d in range(int(HEAD_DIM)):
            score += q[d] * K[batch_head, k_idx, d]
        max_score = wp.max(max_score, score * sm_scale)

    sum_exp = float(0.0)
    for k_idx in range(seq_len):
        score = float(0.0)
        for d in range(int(HEAD_DIM)):
            score += q[d] * K[batch_head, k_idx, d]
        sum_exp += wp.exp(score * sm_scale - max_score)

    acc = wp.types.vector(dtype=float, length=HEAD_DIM)
    for d in range(int(HEAD_DIM)):
        acc[d] = float(0.0)

    for k_idx in range(seq_len):
        score = float(0.0)
        for d in range(int(HEAD_DIM)):
            score += q[d] * K[batch_head, k_idx, d]
        weight = wp.exp(score * sm_scale - max_score) / sum_exp
        for d in range(int(HEAD_DIM)):
            acc[d] += weight * V[batch_head, k_idx, d]

    for d in range(int(HEAD_DIM)):
        O[batch_head, query_idx, d] = acc[d]


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

    m_i = float(NEG_INF)
    l_i = float(0.0)

    acc = wp.types.vector(dtype=float, length=HEAD_DIM)
    for d in range(int(HEAD_DIM)):
        acc[d] = float(0.0)

    q = wp.types.vector(dtype=float, length=HEAD_DIM)
    for d in range(int(HEAD_DIM)):
        q[d] = Q[batch_head, query_idx, d]

    for k_idx in range(seq_len):
        qk = float(0.0)
        for d in range(int(HEAD_DIM)):
            qk += q[d] * K[batch_head, k_idx, d]
        qk = qk * sm_scale

        m_new = wp.max(m_i, qk)
        alpha = wp.exp(m_i - m_new)
        p = wp.exp(qk - m_new)
        l_i = l_i * alpha + p

        for d in range(int(HEAD_DIM)):
            acc[d] = acc[d] * alpha + p * V[batch_head, k_idx, d]

        m_i = m_new

    for d in range(int(HEAD_DIM)):
        O[batch_head, query_idx, d] = acc[d] / l_i


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

    d_start = lane_id * int(D_PER_LANE)
    q_0 = Q[batch_head, query_idx, d_start]
    q_1 = Q[batch_head, query_idx, d_start + 1]

    m_i = float(NEG_INF)
    l_i = float(0.0)
    acc_0 = float(0.0)
    acc_1 = float(0.0)

    for k_idx in range(seq_len):
        k_0 = K[batch_head, k_idx, d_start]
        k_1 = K[batch_head, k_idx, d_start + 1]

        partial_qk = q_0 * k_0 + q_1 * k_1
        qk_val = wp.warp_reduce_sum(partial_qk) * sm_scale

        m_new = wp.max(m_i, qk_val)
        alpha = wp.exp(m_i - m_new)
        p = wp.exp(qk_val - m_new)
        l_i = l_i * alpha + p

        v_0 = V[batch_head, k_idx, d_start]
        v_1 = V[batch_head, k_idx, d_start + 1]
        acc_0 = acc_0 * alpha + p * v_0
        acc_1 = acc_1 * alpha + p * v_1

        m_i = m_new

    O[batch_head, query_idx, d_start] = acc_0 / l_i
    O[batch_head, query_idx, d_start + 1] = acc_1 / l_i


def reference_attention(Q, K, V, sm_scale):
    """NumPy reference."""
    QK = np.matmul(Q, K.transpose(0, 1, 3, 2)) * sm_scale
    QK_max = QK.max(axis=-1, keepdims=True)
    P = np.exp(QK - QK_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


if __name__ == "__main__":
    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    batch_heads = batch * heads

    print("Flash Attention Example")
    print(f"  Shape: [{batch_heads}, {seq_len}, {head_dim}]")

    rng = np.random.default_rng(42)
    Q_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    K_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    V_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    sm_scale = 1.0 / np.sqrt(head_dim)

    Q_4d = Q_np.reshape(batch, heads, seq_len, head_dim)
    K_4d = K_np.reshape(batch, heads, seq_len, head_dim)
    V_4d = V_np.reshape(batch, heads, seq_len, head_dim)
    ref_out = reference_attention(Q_4d, K_4d, V_4d, sm_scale).reshape(batch_heads, seq_len, head_dim)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)

    print("\n--- Naive Attention ---")
    O = wp.zeros_like(Q)
    wp.launch(naive_attention_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    err = np.max(np.abs(O.numpy() - ref_out))
    print(f"  Max error: {err:.2e} [{'PASS' if err < 1e-5 else 'FAIL'}]")

    print("\n--- Flash Attention (scalar) ---")
    O = wp.zeros_like(Q)
    wp.launch(flash_attention_kernel, dim=(seq_len, batch_heads), inputs=[Q, K, V, O, sm_scale, seq_len])
    err = np.max(np.abs(O.numpy() - ref_out))
    print(f"  Max error: {err:.2e} [{'PASS' if err < 1e-5 else 'FAIL'}]")

    print("\n--- Flash Attention (simd) ---")
    O = wp.zeros_like(Q)
    num_queries = seq_len * batch_heads
    wp.launch_tiled(flash_attention_simd_kernel, dim=num_queries, inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads], block_dim=32)
    err = np.max(np.abs(O.numpy() - ref_out))
    print(f"  Max error: {err:.2e} [{'PASS' if err < 1e-5 else 'FAIL'}]")
