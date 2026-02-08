"""Correctness tests for MMA flash attention kernel.

Tests the native PTX MMA flash attention kernel across various tile sizes,
warp counts, and sequence lengths, comparing against a NumPy reference.
"""
import numpy as np
import warp as wp

from example_tile_flash_attention import get_flash_attention_mma_kernel

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


if __name__ == "__main__":
    print("=" * 70)
    print("MMA Flash Attention Correctness Tests")
    print("=" * 70)

    tests = [
        # (tile_m, tile_n, head_dim, seq_len, num_warps, data_scale)
        (16, 16, 32, 16, 1, 1.0),
        (16, 16, 32, 16, 1, 0.1),
        (16, 16, 64, 16, 1, 1.0),
        (16, 16, 64, 32, 1, 1.0),
        (32, 16, 64, 32, 2, 1.0),
        (64, 32, 64, 64, 8, 1.0),
        (128, 64, 64, 128, 16, 1.0),
        (128, 64, 64, 128, 16, 0.1),
        (128, 64, 64, 8192, 16, 1.0),
    ]

    for i, (tm, tn, hd, sl, nw, ds) in enumerate(tests, 1):
        print(f"\n--- Test {i}: M={tm}, N={tn}, HD={hd}, seq={sl}, {nw} warps, scale={ds} ---")
        max_err, mean_err, _, _ = test_mma_flash_attention(tm, tn, hd, sl, nw, data_scale=ds)
        print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY: If small configs pass but large configs fail,")
    print("the bug is in multi-warp tile assignment or multi-block accumulation.")
    print("If all fail, the bug is in basic MMA fragment loading or matmul.")
    print("=" * 70)
