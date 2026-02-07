import warp as wp
import numpy as np

@wp.kernel(enable_backward=False)
def test_insert_kernel(
    A: wp.array2d(dtype=wp.float16),
    B: wp.array2d(dtype=wp.float16),
):
    i = wp.tid()
    TILE_M = 4
    TILE_N = 4

    # Load tile from A
    tile = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(0, 0))

    # Each thread writes its row index + 1.0 to column 0
    row = i
    wp.tile_insert(tile, row, 0, wp.float16(float(row) + 1.0))
    wp.tile_sync()

    # Store result to B
    wp.tile_store(B, tile, offset=(0, 0))


wp.init()
print("Testing tile_insert...")

A_np = np.zeros((4, 4), dtype=np.float16)
A = wp.array(A_np, dtype=wp.float16)
B = wp.zeros_like(A)

wp.launch(test_insert_kernel, dim=4, inputs=[A, B], block_dim=4)
wp.synchronize()

B_np = B.numpy()
print(f"Result B[:,0] = {B_np[:, 0]}")
print(f"Expected: [1. 2. 3. 4.]")

# Check column 0 has [1, 2, 3, 4], rest is 0
expected = np.zeros((4, 4), dtype=np.float16)
expected[:, 0] = [1.0, 2.0, 3.0, 4.0]
if np.allclose(B_np, expected):
    print("PASS")
else:
    print(f"FAIL: got\n{B_np}")
