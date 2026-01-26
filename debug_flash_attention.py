"""Debug script to isolate flash attention issues with different tile sizes."""

import numpy as np
import warp as wp

NEG_INF = wp.constant(-1.0e10)

def create_debug_kernel(tile_m: int, tile_n: int, head_dim: int):
    """Create a debug kernel to test individual operations."""

    @wp.func
    def mul_func(a: float, b: float) -> float:
        return a * b

    @wp.func
    def div_func(a: float, b: float) -> float:
        return a / b

    @wp.kernel(enable_backward=False)
    def debug_kernel(
        Q: wp.array3d(dtype=float),
        K: wp.array3d(dtype=float),
        V: wp.array3d(dtype=float),
        O: wp.array3d(dtype=float),
        debug_S: wp.array3d(dtype=float),  # Debug: store S matrix
        debug_m: wp.array3d(dtype=float),  # Debug: store row max
        debug_P: wp.array3d(dtype=float),  # Debug: store P matrix
        debug_l: wp.array3d(dtype=float),  # Debug: store row sum
        sm_scale: float,
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

        # Load Q block
        Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, q_start, 0))
        Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

        # Initialize accumulators
        O_acc = wp.tile_zeros(shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL), dtype=float, storage="shared")
        m_i = wp.tile_full(shape=(TILE_M_LOCAL, 1), value=float(NEG_INF), dtype=float, storage="shared")
        l_i = wp.tile_zeros(shape=(TILE_M_LOCAL, 1), dtype=float, storage="shared")

        # Only process first K/V block for debugging
        k_block = 0
        k_start = k_block * TILE_N_LOCAL

        # Load K/V blocks
        K_tile_3d = wp.tile_load(K, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
        K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
        V_tile_3d = wp.tile_load(V, shape=(1, TILE_N_LOCAL, HEAD_DIM_LOCAL), offset=(batch_head, k_start, 0), storage="shared")
        V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

        # Compute S = Q @ K^T
        K_T = wp.tile_transpose(K_tile)
        S = wp.tile_zeros(shape=(TILE_M_LOCAL, TILE_N_LOCAL), dtype=float)
        wp.tile_matmul(Q_tile, K_T, S)

        # Scale
        S = S * sm_scale

        # Debug: Store S matrix
        S_3d = wp.tile_reshape(S, shape=(1, TILE_M_LOCAL, TILE_N_LOCAL))
        wp.tile_store(debug_S, S_3d, offset=(batch_head, q_start, k_start))

        # Compute row-wise max
        m_block = wp.tile_reduce(wp.max, S, axis=1)
        m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))

        # Debug: Store row max
        m_block_3d = wp.tile_reshape(m_block, shape=(1, TILE_M_LOCAL, 1))
        wp.tile_store(debug_m, m_block_3d, offset=(batch_head, q_start, k_block))

        # New running max
        m_new = wp.tile_map(wp.max, m_i, m_block)

        # Compute rescaling factor
        alpha = wp.tile_map(wp.exp, m_i - m_new)

        # Compute P = exp(S - m_new)
        m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
        P = wp.tile_map(wp.exp, S - m_broadcast)

        # Debug: Store P matrix
        P_3d = wp.tile_reshape(P, shape=(1, TILE_M_LOCAL, TILE_N_LOCAL))
        wp.tile_store(debug_P, P_3d, offset=(batch_head, q_start, k_start))

        # Compute row-wise sum
        l_block = wp.tile_sum(P, axis=1)
        l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))

        # Debug: Store row sum
        l_block_3d = wp.tile_reshape(l_block, shape=(1, TILE_M_LOCAL, 1))
        wp.tile_store(debug_l, l_block_3d, offset=(batch_head, q_start, k_block))

        # Update running sum
        l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
        wp.tile_assign(l_i, l_new)

        # Update O accumulator
        alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)
        wp.tile_assign(O_acc, O_scaled)

        # Add P @ V contribution
        wp.tile_matmul(P, V_tile, O_acc)

        # Update max
        wp.tile_assign(m_i, m_new)

        # Final normalization
        l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M_LOCAL, HEAD_DIM_LOCAL))
        O_final = wp.tile_map(div_func, O_acc, l_broadcast)

        # Store output
        O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M_LOCAL, HEAD_DIM_LOCAL))
        wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))

    return debug_kernel


def reference_single_block(Q_block, K_block, V_block, sm_scale):
    """Reference implementation for a single Q/K/V block."""
    # S = Q @ K^T * scale
    S = np.matmul(Q_block, K_block.T) * sm_scale

    # Row-wise max
    m = S.max(axis=-1, keepdims=True)

    # P = exp(S - m)
    P = np.exp(S - m)

    # Row-wise sum
    l = P.sum(axis=-1, keepdims=True)

    # Output = P @ V / l
    O = np.matmul(P, V_block) / l

    return S, m.squeeze(-1), P, l.squeeze(-1), O


def run_debug_test(tile_m, tile_n, tile_threads, head_dim=32):
    """Run debug test with specified tile sizes."""
    print(f"\n{'='*60}")
    print(f"Testing TILE_M={tile_m}, TILE_N={tile_n}, TILE_THREADS={tile_threads}, HEAD_DIM={head_dim}")
    print(f"{'='*60}")

    batch_heads = 1
    seq_len = tile_m  # Use tile_m as seq_len for simplicity

    # Generate test data
    rng = np.random.default_rng(42)
    Q_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    K_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    V_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    sm_scale = 1.0 / np.sqrt(head_dim)

    # Compute reference
    ref_S, ref_m, ref_P, ref_l, ref_O = reference_single_block(
        Q_np[0], K_np[0], V_np[0], sm_scale
    )

    # Create warp arrays
    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    # Debug arrays
    debug_S = wp.zeros((batch_heads, seq_len, seq_len), dtype=float)
    debug_m = wp.zeros((batch_heads, seq_len, 1), dtype=float)
    debug_P = wp.zeros((batch_heads, seq_len, seq_len), dtype=float)
    debug_l = wp.zeros((batch_heads, seq_len, 1), dtype=float)

    # Get kernel
    kernel = create_debug_kernel(tile_m, tile_n, head_dim)

    num_q_blocks = 1
    num_k_blocks = 1
    num_tiles = batch_heads * num_q_blocks

    wp.launch_tiled(
        kernel,
        dim=num_tiles,
        inputs=[Q, K, V, O, debug_S, debug_m, debug_P, debug_l, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks],
        block_dim=tile_threads,
    )

    # Compare results
    S_result = debug_S.numpy()[0]
    m_result = debug_m.numpy()[0, :, 0]
    P_result = debug_P.numpy()[0]
    l_result = debug_l.numpy()[0, :, 0]
    O_result = O.numpy()[0]

    print(f"\nS matrix comparison:")
    print(f"  Max error: {np.max(np.abs(S_result - ref_S)):.2e}")

    print(f"\nRow max (m) comparison:")
    print(f"  Reference: {ref_m[:4]}")
    print(f"  Result:    {m_result[:4]}")
    print(f"  Max error: {np.max(np.abs(m_result - ref_m)):.2e}")

    print(f"\nP matrix comparison:")
    print(f"  Max error: {np.max(np.abs(P_result - ref_P)):.2e}")

    print(f"\nRow sum (l) comparison:")
    print(f"  Reference: {ref_l[:4]}")
    print(f"  Result:    {l_result[:4]}")
    print(f"  Max error: {np.max(np.abs(l_result - ref_l)):.2e}")

    print(f"\nOutput O comparison:")
    print(f"  Max error: {np.max(np.abs(O_result - ref_O)):.2e}")

    # Detailed comparison of first few rows
    print(f"\nDetailed m comparison (first 4 rows):")
    for i in range(min(4, tile_m)):
        print(f"  Row {i}: ref={ref_m[i]:.6f}, result={m_result[i]:.6f}, diff={abs(ref_m[i] - m_result[i]):.2e}")


if __name__ == "__main__":
    # Test with different tile sizes
    print("Testing small tile size (should pass):")
    run_debug_test(tile_m=16, tile_n=16, tile_threads=64)

    print("\n" + "="*80)
    print("Testing larger tile size (may fail):")
    run_debug_test(tile_m=32, tile_n=32, tile_threads=128)
