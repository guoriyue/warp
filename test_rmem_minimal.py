"""Minimal test of RMEM tile_matmul path."""
import numpy as np
import warp as wp

wp.init()

TILE_M = 128
TILE_N = 64
HEAD_DIM = 64
BLOCK_DIM = 128

@wp.kernel(enable_backward=False)
def test_rmem_matmul(
    A: wp.array2d(dtype=wp.float16),
    B: wp.array2d(dtype=wp.float16),
    C: wp.array2d(dtype=wp.float16),
):
    # Simple: C = A @ B using RMEM accumulator
    i = wp.tid()

    A_tile = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(0, 0))
    B_tile = wp.tile_load(B, shape=(TILE_N, HEAD_DIM), offset=(0, 0))

    # RMEM accumulator: accumulator=TILE_N specifies K dimension
    C_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16, storage="register", accumulator=TILE_N)

    # This should trigger RMEM path in tile_matmul
    wp.tile_matmul(A_tile, B_tile, C_acc)

    # Copy to shared for output
    C_shared = wp.tile_copy(C_acc)

    row = wp.tid() % TILE_M
    for d in range(HEAD_DIM):
        C[row, d] = C_shared[row, d]


# Create test data
A_np = np.random.randn(TILE_M, TILE_N).astype(np.float16)
B_np = np.random.randn(TILE_N, HEAD_DIM).astype(np.float16)

A_wp = wp.array(A_np, dtype=wp.float16)
B_wp = wp.array(B_np, dtype=wp.float16)
C_wp = wp.zeros((TILE_M, HEAD_DIM), dtype=wp.float16)

print("Launching RMEM matmul test kernel...")
wp.launch(test_rmem_matmul, dim=TILE_M, inputs=[A_wp, B_wp, C_wp], block_dim=BLOCK_DIM)

C_result = C_wp.numpy().astype(np.float32)
C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32))

err = np.max(np.abs(C_result - C_ref))
print(f"Max error: {err:.4e}")
print("PASS" if err < 1.0 else "FAIL")
