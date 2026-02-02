import numpy as np
import warp as wp

wp.init()


@wp.func
def mul_func(a: float, b: float) -> float:
    return a * b


@wp.kernel(enable_backward=False)
def test_flash_pattern_32(
    Q: wp.array2d(dtype=float),
    K: wp.array2d(dtype=float),
    out_max: wp.array2d(dtype=float),
    sm_scale: float,
):
    TILE_M = wp.static(32)
    TILE_N = wp.static(32)
    HEAD_DIM = wp.static(64)

    # Load Q and K to shared memory
    Q_tile = wp.tile_load(Q, shape=(TILE_M, HEAD_DIM), offset=(0, 0), storage="shared")
    K_tile = wp.tile_load(K, shape=(TILE_N, HEAD_DIM), offset=(0, 0), storage="shared")

    # Compute S = Q @ K^T
    K_T = wp.tile_transpose(K_tile)
    S = wp.tile_matmul(Q_tile, K_T)

    # Scale S
    S = S * sm_scale

    # Reduce along axis=1 to get row-wise max
    m_block = wp.tile_reduce(wp.max, S, axis=1)

    # Reshape and store
    m_block_2d = wp.tile_reshape(m_block, shape=(TILE_M, 1))
    wp.tile_store(out_max, m_block_2d, offset=(0, 0))


@wp.kernel(enable_backward=False)
def test_flash_pattern_64(
    Q: wp.array2d(dtype=float),
    K: wp.array2d(dtype=float),
    out_max: wp.array2d(dtype=float),
    sm_scale: float,
):
    TILE_M = wp.static(64)
    TILE_N = wp.static(64)
    HEAD_DIM = wp.static(64)

    # Load Q and K to shared memory
    Q_tile = wp.tile_load(Q, shape=(TILE_M, HEAD_DIM), offset=(0, 0), storage="shared")
    K_tile = wp.tile_load(K, shape=(TILE_N, HEAD_DIM), offset=(0, 0), storage="shared")

    # Compute S = Q @ K^T
    K_T = wp.tile_transpose(K_tile)
    S = wp.tile_matmul(Q_tile, K_T)

    # Scale S
    S = S * sm_scale

    # Reduce along axis=1 to get row-wise max
    m_block = wp.tile_reduce(wp.max, S, axis=1)

    # Reshape and store
    m_block_2d = wp.tile_reshape(m_block, shape=(TILE_M, 1))
    wp.tile_store(out_max, m_block_2d, offset=(0, 0))


if __name__ == "__main__":
    print("Testing Flash Attention Pattern: tile_matmul + scale + tile_reduce")

    head_dim = 64
    sm_scale = 1.0 / np.sqrt(head_dim)

    # Test TILE_SIZE=32
    print("\n=== TILE_M=32, TILE_N=32 ===")
    Q_np = np.random.rand(32, head_dim).astype(np.float32)
    K_np = np.random.rand(32, head_dim).astype(np.float32)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    out_max = wp.zeros((32, 1), dtype=float)

    wp.launch_tiled(test_flash_pattern_32, dim=1, inputs=[Q, K, out_max, sm_scale], block_dim=128)
    wp.synchronize()

    # Compute reference
    S_ref = (Q_np @ K_np.T) * sm_scale
    expected_max = S_ref.max(axis=1)

    result = out_max.numpy().flatten()
    err = np.max(np.abs(result - expected_max))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")

    # Test TILE_SIZE=64
    print("\n=== TILE_M=64, TILE_N=64 ===")
    Q_np = np.random.rand(64, head_dim).astype(np.float32)
    K_np = np.random.rand(64, head_dim).astype(np.float32)

    Q = wp.array(Q_np, dtype=float)
    K = wp.array(K_np, dtype=float)
    out_max = wp.zeros((64, 1), dtype=float)

    wp.launch_tiled(test_flash_pattern_64, dim=1, inputs=[Q, K, out_max, sm_scale], block_dim=256)
    wp.synchronize()

    # Compute reference
    S_ref = (Q_np @ K_np.T) * sm_scale
    expected_max = S_ref.max(axis=1)

    result = out_max.numpy().flatten()
    err = np.max(np.abs(result - expected_max))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")
