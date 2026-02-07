"""Test RMEM tile_scale + multiple matmul accumulations."""
import numpy as np
import warp as wp

wp.init()

TILE_M = 128
TILE_N = 64
HEAD_DIM = 64
BLOCK_DIM = 128

@wp.kernel(enable_backward=False)
def test_rmem_scale_matmul(
    A: wp.array2d(dtype=wp.float16),
    B: wp.array2d(dtype=wp.float16),
    C: wp.array2d(dtype=wp.float16),
    scale_factor: float,
):
    """C = scale_factor * (A @ B) + A @ B = (1 + scale_factor) * A @ B"""
    i = wp.tid()

    A_tile = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(0, 0))
    B_tile = wp.tile_load(B, shape=(TILE_N, HEAD_DIM), offset=(0, 0))

    C_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16, storage="register", accumulator=TILE_N)

    # First matmul
    wp.tile_matmul(A_tile, B_tile, C_acc)

    # Scale the accumulator
    wp.tile_scale(C_acc, scale_factor)

    # Second matmul (accumulate)
    wp.tile_matmul(A_tile, B_tile, C_acc)

    # Copy to shared for output
    C_shared = wp.tile_copy(C_acc)

    row = wp.tid() % TILE_M
    for d in range(HEAD_DIM):
        C[row, d] = C_shared[row, d]


# Create test data
A_np = np.random.RandomState(42).randn(TILE_M, TILE_N).astype(np.float16)
B_np = np.random.RandomState(43).randn(TILE_N, HEAD_DIM).astype(np.float16)

A_wp = wp.array(A_np, dtype=wp.float16)
B_wp = wp.array(B_np, dtype=wp.float16)
C_wp = wp.zeros((TILE_M, HEAD_DIM), dtype=wp.float16)

scale = 0.5

print("Test: C = scale * (A @ B) + A @ B")
wp.launch(test_rmem_scale_matmul, dim=TILE_M, inputs=[A_wp, B_wp, C_wp, scale], block_dim=BLOCK_DIM)

C_result = C_wp.numpy().astype(np.float32)
AB = A_np.astype(np.float32) @ B_np.astype(np.float32)
C_ref = scale * AB + AB  # = (1 + scale) * AB = 1.5 * AB

err = np.max(np.abs(C_result - C_ref))
print(f"Max error: {err:.4e}")
print(f"Expected ~1.5 * AB, got error={err}")
print("PASS" if err < 0.1 else "FAIL")

# Also check: is the scale actually doing anything?
C_noscale_ref = 2.0 * AB  # without scale, it would be 2 * AB
err_noscale = np.max(np.abs(C_result - C_noscale_ref))
print(f"Error vs no-scale (2*AB): {err_noscale:.4e}")
