"""Isolated test for MMA flash attention correctness debugging.

Tests:
1. Full flash attention with small tiles (easy to verify)
2. Directly compares QK^T output against numpy reference
3. Tests different data patterns to isolate the bug
"""
import numpy as np
import warp as wp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "warp", "examples"))
from tile.example_tile_flash_attention import get_flash_attention_mma_kernel

wp.init()


def ref_attention(Q, K, V, sm_scale):
    """NumPy reference flash attention (fp32)."""
    # Q,K,V: [batch_heads, seq_len, head_dim]
    S = Q @ K.transpose(0, 2, 1) * sm_scale  # [B, S, S]
    # Softmax
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    O = P @ V
    return O


def test_mma_flash_attention(tile_m, tile_n, head_dim, seq_len, num_warps, data_scale=1.0, seed=42):
    """Test MMA flash attention kernel with given parameters."""
    rng = np.random.default_rng(seed)
    batch_heads = 1

    shape = (batch_heads, seq_len, head_dim)
    Q_np = (rng.standard_normal(shape) * data_scale).astype(np.float16).astype(np.float32)
    K_np = (rng.standard_normal(shape) * data_scale).astype(np.float16).astype(np.float32)
    V_np = (rng.standard_normal(shape) * data_scale).astype(np.float16).astype(np.float32)

    sm_scale = 1.0 / np.sqrt(head_dim)
    ref = ref_attention(Q_np, K_np, V_np, sm_scale)

    Q = wp.array(Q_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    K = wp.array(K_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    V = wp.array(V_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    O = wp.zeros(shape, dtype=wp.float16, device="cuda:0")

    block_dim = num_warps * 32
    kernel = get_flash_attention_mma_kernel(head_dim, tile_m, tile_n, num_warps=num_warps)

    num_q_blocks = (seq_len + tile_m - 1) // tile_m
    num_k_blocks = (seq_len + tile_n - 1) // tile_n
    total_threads = batch_heads * num_q_blocks * block_dim

    wp.launch(
        kernel,
        dim=total_threads,
        inputs=[Q, K, V, O, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks],
        block_dim=block_dim,
    )
    wp.synchronize()

    O_np = O.numpy().astype(np.float32)
    max_err = np.max(np.abs(O_np - ref))
    mean_err = np.mean(np.abs(O_np - ref))

    return max_err, mean_err, O_np, ref


# Clear kernel cache first
cache_dir = os.path.expanduser("~/.cache/warp")
if os.path.exists(cache_dir):
    import shutil
    for d in os.listdir(cache_dir):
        p = os.path.join(cache_dir, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    print("Cleared warp cache")

print("=" * 70)
print("MMA Flash Attention Correctness Tests")
print("=" * 70)

# Test 1: Smallest config: M=16, N=16, HD=32, seq=16, 1 warp
print("\n--- Test 1: M=16, N=16, HD=32, seq=16, 1 warp, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=16, tile_n=16, head_dim=32, seq_len=16, num_warps=1, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")
print(f"  MMA out[0,0,:8]:  {O_mma[0,0,:8]}")
print(f"  Ref out[0,0,:8]:  {O_ref[0,0,:8]}")

# Test 2: Same but tiny values
print("\n--- Test 2: M=16, N=16, HD=32, seq=16, 1 warp, scale=0.1 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=16, tile_n=16, head_dim=32, seq_len=16, num_warps=1, data_scale=0.1)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 3: M=16, N=16, HD=64, 1 warp
print("\n--- Test 3: M=16, N=16, HD=64, seq=16, 1 warp, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=16, tile_n=16, head_dim=64, seq_len=16, num_warps=1, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 4: 2 k_blocks
print("\n--- Test 4: M=16, N=16, HD=64, seq=32, 1 warp, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=16, tile_n=16, head_dim=64, seq_len=32, num_warps=1, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 5: M=32, N=16, HD=64, 2 warps
print("\n--- Test 5: M=32, N=16, HD=64, seq=32, 2 warps, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=32, tile_n=16, head_dim=64, seq_len=32, num_warps=2, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 6: M=64, N=32, HD=64, 8 warps
print("\n--- Test 6: M=64, N=32, HD=64, seq=64, 8 warps, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=64, tile_n=32, head_dim=64, seq_len=64, num_warps=8, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 7: Production config M=128, N=64, HD=64, 16 warps
print("\n--- Test 7: M=128, N=64, HD=64, seq=128, 16 warps, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=128, tile_n=64, head_dim=64, seq_len=128, num_warps=16, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 8: Production config with small scale
print("\n--- Test 8: M=128, N=64, HD=64, seq=128, 16 warps, scale=0.1 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=128, tile_n=64, head_dim=64, seq_len=128, num_warps=16, data_scale=0.1)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Test 9: Large seq with production config
print("\n--- Test 9: M=128, N=64, HD=64, seq=8192, 16 warps, scale=1.0 ---")
max_err, mean_err, O_mma, O_ref = test_mma_flash_attention(
    tile_m=128, tile_n=64, head_dim=64, seq_len=8192, num_warps=16, data_scale=1.0)
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

print("\n" + "=" * 70)
print("SUMMARY: If small configs pass but large configs fail,")
print("the bug is in multi-warp tile assignment or multi-block accumulation.")
print("If all fail, the bug is in basic MMA fragment loading or matmul.")
print("=" * 70)
