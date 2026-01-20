"""Test register to shared tile assignment."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def test_tile_assign_register_to_shared(out: wp.array2d(dtype=float)):
    # Create a shared tile
    shared = wp.tile_zeros(dtype=float, shape=(4, 4))

    # Create a register tile via tile_map
    ones = wp.tile_full(dtype=float, shape=(4, 4), value=1.0)
    scale = wp.tile_full(dtype=float, shape=(4, 4), value=2.0)
    twos = wp.tile_map(wp.mul, ones, scale)  # This is a register tile

    # Try to assign register to shared
    wp.tile_assign(shared, twos)

    wp.tile_store(out, shared)

out = wp.zeros((4, 4), dtype=float, device='cuda:0')
try:
    wp.launch_tiled(test_tile_assign_register_to_shared, dim=1, inputs=[out], block_dim=32, device='cuda:0')
    wp.synchronize()
    print('Success!')
    print(out.numpy())
except Exception as e:
    print(f'Error: {e}')
