import numpy as np
import warp as wp

wp.init()

NEG_INF = wp.constant(-1.0e10)


@wp.func
def mul_func(a: float, b: float) -> float:
    return a * b


@wp.func
def div_func(a: float, b: float) -> float:
    return a / b


@wp.kernel(enable_backward=False)
def test_flash_loop_32(
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
    TILE_M = wp.static(32)
    TILE_N = wp.static(32)
    HEAD_DIM = wp.static(64)

    tile_idx = wp.tid()
    batch_head = tile_idx // num_q_blocks
    q_block_idx = tile_idx % num_q_blocks
    q_start = q_block_idx * TILE_M

    if batch_head >= batch_heads:
        return

    # Load Q block to shared memory
    Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M, HEAD_DIM), offset=(batch_head, q_start, 0), storage="shared")
    Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

    # Initialize output accumulator (registers)
    O_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=float, storage="register")

    # Online softmax statistics
    m_i = wp.tile_full(shape=(TILE_M, 1), value=float(NEG_INF), dtype=float, storage="register")
    l_i = wp.tile_zeros(shape=(TILE_M, 1), dtype=float, storage="register")

    # Process K/V blocks
    for k_block in range(num_k_blocks):
        k_start = k_block * TILE_N

        # Load K/V blocks
        K_tile_3d = wp.tile_load(K, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0), storage="shared")
        K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
        V_tile_3d = wp.tile_load(V, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0), storage="shared")
        V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

        # Compute S = Q @ K^T
        K_T = wp.tile_transpose(K_tile)
        S = wp.tile_matmul(Q_tile, K_T)
        S = S * sm_scale

        # Update running max
        m_block = wp.tile_reduce(wp.max, S, axis=1)
        m_block = wp.tile_reshape(m_block, shape=(TILE_M, 1))
        m_new = wp.tile_map(wp.max, m_i, m_block)

        # Compute correction factor
        alpha = wp.tile_map(wp.exp, m_i - m_new)

        # Compute P = exp(S - m_new)
        m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M, TILE_N))
        P = wp.tile_map(wp.exp, S - m_broadcast)

        # Update running sum
        l_block = wp.tile_sum(P, axis=1)
        l_block = wp.tile_reshape(l_block, shape=(TILE_M, 1))
        l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
        wp.tile_assign(l_i, l_new)

        # Rescale previous output
        alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M, HEAD_DIM))
        O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)
        wp.tile_assign(O_acc, O_scaled)

        # Accumulate new contribution
        wp.tile_matmul(P, V_tile, O_acc)

        # Update running max
        wp.tile_assign(m_i, m_new)

    # Final normalization
    l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M, HEAD_DIM))
    O_final = wp.tile_map(div_func, O_acc, l_broadcast)

    # Write back
    O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M, HEAD_DIM))
    wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))


@wp.kernel(enable_backward=False)
def test_flash_loop_64(
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
    TILE_M = wp.static(64)
    TILE_N = wp.static(64)
    HEAD_DIM = wp.static(64)

    tile_idx = wp.tid()
    batch_head = tile_idx // num_q_blocks
    q_block_idx = tile_idx % num_q_blocks
    q_start = q_block_idx * TILE_M

    if batch_head >= batch_heads:
        return

    # Load Q block to shared memory
    Q_tile_3d = wp.tile_load(Q, shape=(1, TILE_M, HEAD_DIM), offset=(batch_head, q_start, 0), storage="shared")
    Q_tile = wp.tile_squeeze(Q_tile_3d, axis=(0,))

    # Initialize output accumulator (registers)
    O_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=float, storage="register")

    # Online softmax statistics
    m_i = wp.tile_full(shape=(TILE_M, 1), value=float(NEG_INF), dtype=float, storage="register")
    l_i = wp.tile_zeros(shape=(TILE_M, 1), dtype=float, storage="register")

    # Process K/V blocks
    for k_block in range(num_k_blocks):
        k_start = k_block * TILE_N

        # Load K/V blocks
        K_tile_3d = wp.tile_load(K, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0), storage="shared")
        K_tile = wp.tile_squeeze(K_tile_3d, axis=(0,))
        V_tile_3d = wp.tile_load(V, shape=(1, TILE_N, HEAD_DIM), offset=(batch_head, k_start, 0), storage="shared")
        V_tile = wp.tile_squeeze(V_tile_3d, axis=(0,))

        # Compute S = Q @ K^T
        K_T = wp.tile_transpose(K_tile)
        S = wp.tile_matmul(Q_tile, K_T)
        S = S * sm_scale

        # Update running max
        m_block = wp.tile_reduce(wp.max, S, axis=1)
        m_block = wp.tile_reshape(m_block, shape=(TILE_M, 1))
        m_new = wp.tile_map(wp.max, m_i, m_block)

        # Compute correction factor
        alpha = wp.tile_map(wp.exp, m_i - m_new)

        # Compute P = exp(S - m_new)
        m_broadcast = wp.tile_broadcast(m_new, shape=(TILE_M, TILE_N))
        P = wp.tile_map(wp.exp, S - m_broadcast)

        # Update running sum
        l_block = wp.tile_sum(P, axis=1)
        l_block = wp.tile_reshape(l_block, shape=(TILE_M, 1))
        l_new = wp.tile_map(mul_func, l_i, alpha) + l_block
        wp.tile_assign(l_i, l_new)

        # Rescale previous output
        alpha_broadcast = wp.tile_broadcast(alpha, shape=(TILE_M, HEAD_DIM))
        O_scaled = wp.tile_map(mul_func, O_acc, alpha_broadcast)
        wp.tile_assign(O_acc, O_scaled)

        # Accumulate new contribution
        wp.tile_matmul(P, V_tile, O_acc)

        # Update running max
        wp.tile_assign(m_i, m_new)

    # Final normalization
    l_broadcast = wp.tile_broadcast(l_i, shape=(TILE_M, HEAD_DIM))
    O_final = wp.tile_map(div_func, O_acc, l_broadcast)

    # Write back
    O_final_3d = wp.tile_reshape(O_final, shape=(1, TILE_M, HEAD_DIM))
    wp.tile_store(O, O_final_3d, offset=(batch_head, q_start, 0))


def reference_attention(Q, K, V, sm_scale):
    """NumPy reference."""
    QK = np.matmul(Q, K.transpose(0, 2, 1)) * sm_scale
    QK_max = QK.max(axis=-1, keepdims=True)
    P = np.exp(QK - QK_max)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


if __name__ == "__main__":
    print("Testing Full Flash Attention Loop Pattern")

    batch_heads = 2
    head_dim = 64
    sm_scale = 1.0 / np.sqrt(head_dim)

    # Test TILE_SIZE=32
    print("\n=== TILE_M=32, TILE_N=32 ===")
    seq_len = 64
    tile_m, tile_n = 32, 32
    threads = 128

    padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
    num_q_blocks = padded_seq // tile_m
    num_k_blocks = padded_seq // tile_n

    Q_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)
    K_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)
    V_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    num_tiles = batch_heads * num_q_blocks

    wp.launch_tiled(
        test_flash_loop_32,
        dim=num_tiles,
        inputs=[Q, K, V, O, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
        block_dim=threads,
    )
    wp.synchronize()

    # Compute reference
    ref_out = reference_attention(Q_np, K_np, V_np, sm_scale)
    result = O.numpy()
    err = np.max(np.abs(result - ref_out))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")

    # Test TILE_SIZE=64
    print("\n=== TILE_M=64, TILE_N=64 ===")
    seq_len = 128
    tile_m, tile_n = 64, 64
    threads = 256

    padded_seq = ((seq_len + tile_m - 1) // tile_m) * tile_m
    num_q_blocks = padded_seq // tile_m
    num_k_blocks = padded_seq // tile_n

    Q_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)
    K_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)
    V_np = np.random.rand(batch_heads, padded_seq, head_dim).astype(np.float32)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    V = wp.array(V_np, dtype=float)
    O = wp.zeros_like(Q)

    num_tiles = batch_heads * num_q_blocks

    wp.launch_tiled(
        test_flash_loop_64,
        dim=num_tiles,
        inputs=[Q, K, V, O, sm_scale, padded_seq, batch_heads, num_q_blocks, num_k_blocks],
        block_dim=threads,
    )
    wp.synchronize()

    # Compute reference
    ref_out = reference_attention(Q_np, K_np, V_np, sm_scale)
    result = O.numpy()
    err = np.max(np.abs(result - ref_out))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")
