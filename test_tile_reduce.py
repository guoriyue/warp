import numpy as np
import warp as wp

wp.init()

# Test kernel for tile_reduce with axis
@wp.kernel(enable_backward=False)
def test_reduce_32(
    A: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    TILE_M = wp.static(32)
    TILE_N = wp.static(32)

    # Load tile to shared memory
    A_tile = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")

    # Reduce along axis=1 (columns)
    row_max = wp.tile_reduce(wp.max, A_tile, axis=1)

    # Reshape to 2D for storing
    row_max_2d = wp.tile_reshape(row_max, shape=(TILE_M, 1))
    wp.tile_store(out, row_max_2d, offset=(0, 0))


@wp.kernel(enable_backward=False)
def test_reduce_64(
    A: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    TILE_M = wp.static(64)
    TILE_N = wp.static(64)

    # Load tile to shared memory
    A_tile = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")

    # Reduce along axis=1 (columns)
    row_max = wp.tile_reduce(wp.max, A_tile, axis=1)

    # Reshape to 2D for storing
    row_max_2d = wp.tile_reshape(row_max, shape=(TILE_M, 1))
    wp.tile_store(out, row_max_2d, offset=(0, 0))


if __name__ == "__main__":
    print("Testing tile_reduce with axis=1")

    # Test TILE_SIZE=32
    print("\n=== TILE_M=32, TILE_N=32 ===")
    A_np = np.random.rand(32, 32).astype(np.float32)
    A = wp.array(A_np, dtype=float)
    out = wp.zeros((32, 1), dtype=float)

    wp.launch_tiled(test_reduce_32, dim=1, inputs=[A, out], block_dim=128)
    wp.synchronize()

    result = out.numpy().flatten()
    expected = A_np.max(axis=1)
    err = np.max(np.abs(result - expected))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")

    # Test TILE_SIZE=64
    print("\n=== TILE_M=64, TILE_N=64 ===")
    A_np = np.random.rand(64, 64).astype(np.float32)
    A = wp.array(A_np, dtype=float)
    out = wp.zeros((64, 1), dtype=float)

    wp.launch_tiled(test_reduce_64, dim=1, inputs=[A, out], block_dim=256)
    wp.synchronize()

    result = out.numpy().flatten()
    expected = A_np.max(axis=1)
    err = np.max(np.abs(result - expected))
    print(f"Error: {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")
