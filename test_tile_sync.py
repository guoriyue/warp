import warp as wp

@wp.kernel(enable_backward=False)
def test_sync_kernel(A: wp.array2d(dtype=wp.float16)):
    i = wp.tid()
    wp.tile_sync()

wp.init()
print("tile_sync kernel defined successfully")

import numpy as np
A = wp.array(np.zeros((4, 4), dtype=np.float16), dtype=wp.float16)
wp.launch(test_sync_kernel, dim=4, inputs=[A], block_dim=4)
wp.synchronize()
print("tile_sync kernel executed successfully")
