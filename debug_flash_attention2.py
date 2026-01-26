"""Debug script to test multi-block flash attention iterations."""

import numpy as np
import warp as wp

NEG_INF = wp.constant(-1.0e10)


def create_multiblock_debug_kernel(tile_m: int, tile_n: int, head_dim: int):
    """Create a debug kernel with multiple K/V blocks."""

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

        # Iterate over K/V blocks
        for k_block in range(num_k_blocks):
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

            # Compute row-wise max
            m_block = wp.tile_reduce(wp.max, S, axis=1)
            m_block = wp.tile_reshape(m_block, shape=(TILE_M_LOCAL, 1))

            # New running max
            m_new = wp.tile_map(wp.max, m_i, m_block)

            # Compute rescaling factor
            alpha = wp.tile_map(wp.exp, m_i - m_new)

            # Compute P = exp(S - m_new)
            m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M_LOCAL, TILE_N_LOCAL))
            P = wp.tile_map(wp.exp, S - m_broadcast)

            # Compute row-wise sum
            l_block = wp.tile_sum(P, axis=1)
            l_block = wp.tile_reshape(l_block, shape=(TILE_M_LOCAL, 1))

            # Update running sum: l_new = l_i * alpha + l_block
            l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
            wp.tile_assign(l_i, l_new)

            # Update O accumulator: O = O * alpha
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


def reference_attention(Q, K, V, sm_scale):
    """NumPy reference."""
    # Q, K, V: (batch_heads, seq_len, head_dim)
    QK = np.matmul(Q, K.transpose(0, 2, 1)) * sm_scale
    QK_max = QK.max(axis=-1, keepdims=True)
    P = np.exp(QK - QK_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def run_test(tile_m, tile_n, tile_threads, seq_len, head_dim=32):
    """Run test with specified parameters."""
    print(f"\n{'='*60}")
    print(f"TILE_M={tile_m}, TILE_N={tile_n}, SEQ_LEN={seq_len}, HEAD_DIM={head_dim}")
    print(f"{'='*60}")

    batch_heads = 1

    # Generate test data
    rng = np.random.default_rng(42)
    Q_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    K_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    V_np = rng.random((batch_heads, seq_len, head_dim), dtype=np.float32)
    sm_scale = 1.0 / np.sqrt(head_dim)

    # Pad to tile size
    padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
    Q_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
    K_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
    V_padded = np.zeros((batch_heads, padded_seq, head_dim), dtype=np.float32)
    Q_padded[:, :seq_len, :] = Q_np
    K_padded[:, :seq_len, :] = K_np
    V_padded[:, :seq_len, :] = V_np

    # Compute reference on padded data
    ref_out = reference_attention(Q_padded, K_padded, V_padded, sm_scale)

    # Create warp arrays
    Q = wp.array(Q_padded, dtype=float)
    K = wp.array(K_padded, dtype=float)
    V = wp.array(V_padded, dtype=float)
    O = wp.zeros_like(Q)

    # Get kernel
    kernel = create_multiblock_debug_kernel(tile_m, tile_n, head_dim)

    num_q_blocks = padded_seq // tile_m
    num_k_blocks = padded_seq // tile_n
    num_tiles = batch_heads * num_q_blocks

    print(f"num_q_blocks={num_q_blocks}, num_k_blocks={num_k_blocks}")

    wp.launch_tiled(
        kernel,
        dim=num_tiles,
        inputs=[Q, K, V, O, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
        block_dim=tile_threads,
    )

    # Compare results (only non-padded part)
    O_result = O.numpy()[:, :seq_len, :]
    ref_result = ref_out[:, :seq_len, :]

    err = np.max(np.abs(O_result - ref_result))
    status = "PASS" if err < 1e-5 else "FAIL"
    print(f"Max error: {err:.2e} [{status}]")

    if status == "FAIL":
        # Print some debug info
        print(f"\nDetailed comparison (first 4 rows, first 4 cols):")
        print(f"Reference:\n{ref_result[0, :4, :4]}")
        print(f"Result:\n{O_result[0, :4, :4]}")

    return err, status


if __name__ == "__main__":
    # Test with single K/V block (should always pass)
    print("="*80)
    print("Test 1: Single K/V block")
    print("="*80)
    run_test(tile_m=16, tile_n=16, tile_threads=64, seq_len=16)
    run_test(tile_m=32, tile_n=32, tile_threads=128, seq_len=32)

    # Test with 2 K/V blocks
    print("\n" + "="*80)
    print("Test 2: Two K/V blocks")
    print("="*80)
    run_test(tile_m=16, tile_n=16, tile_threads=64, seq_len=32)
    run_test(tile_m=32, tile_n=32, tile_threads=128, seq_len=64)

    # Test with 4 K/V blocks
    print("\n" + "="*80)
    print("Test 3: Four K/V blocks")
    print("="*80)
    run_test(tile_m=16, tile_n=16, tile_threads=64, seq_len=64)
    run_test(tile_m=32, tile_n=32, tile_threads=128, seq_len=128)
