"""Test: do QK^T matmul only, skip softmax, compare against reference.

This test uses the MMA flash attention kernel but with V=0 to isolate
whether the QK^T matmul or the softmax is the source of error.

Actually, we can't easily extract QK^T from the kernel.
Instead, let's test with data that makes the softmax well-behaved.

Strategy: If Q_i K_j^T is the same for all j (all K rows identical),
then softmax gives uniform weights 1/N for each j.
So output = (1/N) * sum_j V_j = mean(V, axis=0).
This tests softmax but with trivial (uniform) distribution.

Better strategy: Make K = identity-like so QK^T = Q[:, :N].
Then test whether the P=softmax(Q[:,:N] * sm_scale) @ V is correct.
"""
import numpy as np
import warp as wp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "warp", "examples"))
from tile.example_tile_flash_attention import get_flash_attention_mma_kernel

wp.init()


def ref_attention(Q, K, V, sm_scale):
    """NumPy reference flash attention (fp64 for accuracy)."""
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)
    S = Q64 @ K64.transpose(0, 2, 1) * sm_scale
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    O = P @ V64
    return O.astype(np.float32)


def ref_attention_f16_sim(Q, K, V, sm_scale):
    """Simulate fp16 precision: QK^T done in fp16, softmax in fp32, PV in fp16."""
    # QK^T in fp16 (accumulate in fp32 like MMA does)
    Q16 = Q.astype(np.float16)
    K16 = K.astype(np.float16)
    V16 = V.astype(np.float16)

    # MMA computes in fp32 accumulation but fp16 inputs
    S_f32 = Q16.astype(np.float32) @ K16.astype(np.float32).transpose(0, 2, 1)
    # Then stored as fp16 to SMEM
    S_f16 = S_f32.astype(np.float16).astype(np.float32)

    # Softmax in fp32 with log2 scale
    sm_scale_log2 = sm_scale * 1.44269504
    S_scaled = S_f16 * sm_scale_log2
    S_max = S_scaled.max(axis=-1, keepdims=True)
    P = np.exp2(S_scaled - S_max)
    P_sum = P.sum(axis=-1, keepdims=True)
    P_norm = P / P_sum
    # P stored as fp16
    P_f16 = P_norm.astype(np.float16).astype(np.float32)

    # PV in fp16 inputs, fp32 accumulation
    O = P_f16 @ V16.astype(np.float32)
    # Normalize by inv_l (already done above)
    # Then convert to fp16
    return O.astype(np.float16).astype(np.float32)


def run_mma(tile_m, tile_n, head_dim, seq_len, num_warps, Q_np, K_np, V_np, sm_scale):
    Q = wp.array(Q_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    K = wp.array(K_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    V = wp.array(V_np.astype(np.float16), dtype=wp.float16, device="cuda:0")
    O = wp.zeros_like(Q)

    block_dim = num_warps * 32
    kernel = get_flash_attention_mma_kernel(head_dim, tile_m, tile_n, num_warps=num_warps)
    num_q_blocks = (seq_len + tile_m - 1) // tile_m
    num_k_blocks = (seq_len + tile_n - 1) // tile_n
    total_threads = Q_np.shape[0] * num_q_blocks * block_dim

    wp.launch(
        kernel, dim=total_threads,
        inputs=[Q, K, V, O, sm_scale, seq_len, Q_np.shape[0], num_q_blocks, num_k_blocks],
        block_dim=block_dim,
    )
    wp.synchronize()
    return O.numpy().astype(np.float32)


rng = np.random.default_rng(42)

print("=" * 70)
print("Detailed MMA Flash Attention Debugging")
print("=" * 70)

# Config: smallest possible
TM, TN, HD = 16, 16, 32
SEQ = 16
NW = 1
BH = 1

shape = (BH, SEQ, HD)
Q_np = rng.standard_normal(shape).astype(np.float16).astype(np.float32)
K_np = rng.standard_normal(shape).astype(np.float16).astype(np.float32)
V_np = rng.standard_normal(shape).astype(np.float16).astype(np.float32)
sm_scale = 1.0 / np.sqrt(HD)

# Compute references
ref_f64 = ref_attention(Q_np, K_np, V_np, sm_scale)
ref_f16_sim = ref_attention_f16_sim(Q_np, K_np, V_np, sm_scale)

# Run MMA kernel
O_mma = run_mma(TM, TN, HD, SEQ, NW, Q_np, K_np, V_np, sm_scale)

print(f"\nConfig: M={TM}, N={TN}, HD={HD}, seq={SEQ}, {NW} warp, scale=1.0")
print(f"  MMA vs fp64 ref:    max_err={np.max(np.abs(O_mma - ref_f64)):.6f}")
print(f"  MMA vs fp16 sim:    max_err={np.max(np.abs(O_mma - ref_f16_sim)):.6f}")
print(f"  fp16_sim vs fp64:   max_err={np.max(np.abs(ref_f16_sim - ref_f64)):.6f}")

# Show actual QK^T
QKT = Q_np[0] @ K_np[0].T  # [SEQ, SEQ]
print(f"\n  QK^T range: [{QKT.min():.3f}, {QKT.max():.3f}], mean={QKT.mean():.3f}")
print(f"  QK^T * sm_scale range: [{(QKT*sm_scale).min():.3f}, {(QKT*sm_scale).max():.3f}]")
QKT_scaled_log2 = QKT * sm_scale * 1.44269504
print(f"  QK^T * sm_scale_log2 range: [{QKT_scaled_log2.min():.3f}, {QKT_scaled_log2.max():.3f}]")

# Row-by-row comparison
print(f"\n  Row 0 comparison:")
print(f"    MMA:     {O_mma[0,0,:8]}")
print(f"    fp16_sim:{ref_f16_sim[0,0,:8]}")
print(f"    fp64_ref:{ref_f64[0,0,:8]}")

# Now test with SCALED data that makes QK^T stay small
print("\n" + "-" * 70)
print("Test with pre-scaled data (Q *= sm_scale, sm_scale=1.0 in kernel):")
Q_scaled = (Q_np * sm_scale).astype(np.float16).astype(np.float32)
# Use sm_scale=1.0 so the kernel just computes softmax(Q_scaled @ K^T) @ V
O_mma2 = run_mma(TM, TN, HD, SEQ, NW, Q_scaled, K_np, V_np, 1.0)
ref_scaled = ref_attention(Q_scaled, K_np, V_np, 1.0)
print(f"  MMA vs ref: max_err={np.max(np.abs(O_mma2 - ref_scaled)):.6f}")

# Now test: is the problem in QK^T or in softmax?
# Use K = eye-like matrix (first HD columns of seq_len x HD identity)
# Then QK^T = Q[:, :N] (taking first N=16 dims of Q)
print("\n" + "-" * 70)
print("Test with K = identity-like (QK^T becomes trivial):")
K_eye = np.zeros(shape, dtype=np.float32)
for i in range(min(SEQ, HD)):
    K_eye[0, i, i] = 1.0
K_eye = K_eye.astype(np.float16).astype(np.float32)

O_mma3 = run_mma(TM, TN, HD, SEQ, NW, Q_np, K_eye, V_np, sm_scale)
ref_eye = ref_attention(Q_np, K_eye, V_np, sm_scale)
print(f"  MMA vs ref: max_err={np.max(np.abs(O_mma3 - ref_eye)):.6f}")

# QK^T with identity K should be Q[:, :16]
QKT_eye = Q_np[0] @ K_eye[0].T
print(f"  QK^T_eye range: [{QKT_eye.min():.3f}, {QKT_eye.max():.3f}]")

# Test: K = small values (make QK^T small)
print("\n" + "-" * 70)
print("Test with K *= 0.1 (make QK^T smaller):")
K_small = (K_np * 0.1).astype(np.float16).astype(np.float32)
O_mma4 = run_mma(TM, TN, HD, SEQ, NW, Q_np, K_small, V_np, sm_scale)
ref_small = ref_attention(Q_np, K_small, V_np, sm_scale)
print(f"  MMA vs ref: max_err={np.max(np.abs(O_mma4 - ref_small)):.6f}")
QKT_small = Q_np[0] @ K_small[0].T * sm_scale
print(f"  QK^T*scale range: [{QKT_small.min():.3f}, {QKT_small.max():.3f}]")

# Test: make softmax sharper by using large sm_scale
print("\n" + "-" * 70)
print("Test with large sm_scale=10.0 (sharper softmax):")
O_mma5 = run_mma(TM, TN, HD, SEQ, NW, Q_np * 0.1, K_np * 0.1, V_np, 10.0)
ref_sharp = ref_attention(Q_np * 0.1, K_np * 0.1, V_np, 10.0)
QKT_sharp = (Q_np[0] * 0.1) @ (K_np[0] * 0.1).T * 10.0
print(f"  MMA vs ref: max_err={np.max(np.abs(O_mma5 - ref_sharp)):.6f}")
print(f"  QK^T*scale range: [{QKT_sharp.min():.3f}, {QKT_sharp.max():.3f}]")

# Test: uniform K (all rows same) -> softmax should be uniform -> output = mean(V)
print("\n" + "-" * 70)
print("Test with uniform K (all rows identical):")
K_uniform = np.tile(K_np[:, 0:1, :], (1, SEQ, 1))
O_mma6 = run_mma(TM, TN, HD, SEQ, NW, Q_np, K_uniform, V_np, sm_scale)
ref_uniform = ref_attention(Q_np, K_uniform, V_np, sm_scale)
print(f"  MMA vs ref: max_err={np.max(np.abs(O_mma6 - ref_uniform)):.6f}")
expected_mean = V_np[0].mean(axis=0)
print(f"  Output should ≈ mean(V). Actual[0,0,:4]={O_mma6[0,0,:4]}, mean(V)[:4]={expected_mean[:4]}")
