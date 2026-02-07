# RMEM (cuBLASDx Register-Memory) Integration into Warp Tile System

## Session Date: 2026-02-07

## What This Project Is

Integrating cuBLASDx's Tensor API with register-memory (RMEM) accumulators into NVIDIA Warp's tile system for flash attention on RTX 5090 (sm_120). The goal is a clean user API:

```python
O_acc = wp.tile_zeros(shape=(TILE_M, HEAD_DIM), dtype=wp.float16, storage="register", accumulator=TILE_N)
wp.tile_matmul(S_tile, V_tile, O_acc)   # auto-detects register → RMEM LTO path
wp.tile_scale(O_acc, alpha)              # element-wise scale via MAP_IDX2CRD
O_shared = wp.tile_copy(O_acc)           # rmem → shared via MAP_IDX2CRD
```

## Current Status: CORRECTNESS WORKS, PERFORMANCE NEEDS INVESTIGATION

### What Works
- Basic RMEM matmul: max error 0.027 (PASS)
- Scale + matmul accumulation: max error 0.040 (PASS)
- RMEM flash attention: max error 0.109 (PASS, threshold 0.2 for fp16 tensor-core)
- Non-RMEM flash attention: max error 4.49e-04 (PASS)
- All NVRTC compilation issues resolved
- All layout issues resolved (COPY_A/COPY_B + MAP_IDX2CRD)

### What Doesn't Work: PERFORMANCE
Last benchmark (seq_len=8192, head_dim=64, batch=1, heads=8):
```
wp_flash_attn:  21.556ms   (EXPECTED ~6ms - REGRESSION!)
wp_flash_rmem:  20.098ms
triton:          0.873ms
flash_attn:      0.809ms
```

Both Warp kernels are ~25x slower than Triton/FlashAttn. The user said wp_flash_attn used to be ~6ms. This regression was NOT investigated yet.

## Files Modified (from original Warp codebase)

### 1. `warp/native/warp.cu` (~line 4237-4449)
- Core RMEM LTO build function `wp_cuda_compile_dot_rmem()`
- Added `smem_a_sugg_storage_bytes` and `smem_b_sugg_storage_bytes` output params
- Queries both plain and suggested layout sizes from cuBLASDx tensor traits
- Compiles 7 device functions per config: COPY_A, COPY_B, EXECUTE, CLEAR, COPY_C, MAP_IDX2CRD, IS_INDEX_IN_BOUNDS

### 2. `warp/native/warp.h` (~line 506)
- Updated `wp_cuda_compile_dot_rmem()` declaration to match new parameters

### 3. `warp/native/tile.h` (~line 4874-4991)
- **`tile_matmul_rmem` macro**: Allocates separate SMEM for suggested layout via `tile_shared_storage_t::alloc()`, calls COPY_A/COPY_B to rearrange from plain→suggested, then EXECUTE
- **`tile_rmem_copy` macro**: Uses MAP_IDX2CRD element-wise iteration (NOT the COPY device function which outputs in suggested SMEM_C layout)
- **`tile_rmem_scale` macro**: Uses MAP_IDX2CRD + `static_cast<wp::float32>()` (NOT `float()` which conflicts with Warp's `#define float(x) cast_float(x)`)
- **`tile_rmem_t<N>`**: Has template assignment operator to accept scalar from `tile_zeros`
- CPU stubs for all macros

### 4. `warp/_src/build.py` (~line 460-620)
- Added `copy_a_sym`, `copy_b_sym` to metadata dict
- Added extern "C" declarations for all 7 device functions
- Added ctypes output params for `smem_a_sugg_storage_bytes`, `smem_b_sugg_storage_bytes`
- Has `[RMEM DEBUG]` print statements (should be cleaned up)

### 5. `warp/_src/builtins.py` (~line 10920-11300)
- **`tile_matmul_lto_dispatch_func`**: When output tile has `storage="register"`, returns 5-element tuple with `"tile_matmul_rmem"` override. Passes copy_a, copy_b, execute symbols + smem sizes. Backfills `rmem_storage_bytes` on output type.
- **`tile_copy` dispatch**: When source is register/rmem, returns `"tile_rmem_copy"` with MAP/BOUNDS symbols
- **`tile_scale` dispatch**: Returns `"tile_rmem_scale"` with MAP/BOUNDS symbols
- **`_get_or_build_rmem_lto`**: Cache path also has copy_a/copy_b declarations

### 6. `warp/_src/codegen.py` (~line 1594)
- Supports 5-element return from `lto_dispatch_func` (optional func name override)
- Uses override to emit `tile_matmul_rmem(...)` instead of `wp::tile_matmul(...)`

### 7. `warp/_src/types.py` (~line 4251)
- `tile` type: when `storage="register"` and `rmem_storage_bytes > 0`, `ctype()` returns `wp::tile_rmem_t<N>`
- `cinit()` returns zero-initialized rmem buffer

### 8. `warp/_src/build_dll.py` (line 758)
- Added `-ccbin /usr/bin/gcc-12` to nvcc release command (gcc-13 incompatible)

### 9. `warp/native/cuda_util.cpp` (line 79)
- Added `#if CUDA_VERSION >= 12080` guard for `PFN_cuMemcpyBatchAsync_v12080`

### 10. `warp/examples/tile/example_tile_flash_attention.py` (~line 242+)
- `create_flash_attention_rmem_kernel()` function
- RMEM flash attention test in main block (threshold 0.2)

### 11. Test files (repo root)
- `test_rmem_minimal.py` - Basic RMEM matmul test (128x64 x 64x64)
- `test_rmem_scale.py` - tile_scale + multiple matmul accumulations test

## Key Technical Details

### cuBLASDx Tensor API
- `CUBLASDX_API_TENSORS` mode with suggested layouts for SMEM_A/B and RMEM_C
- Warp's `tile_load` produces row-major strided tiles (plain layout)
- cuBLASDx EXECUTE expects suggested layout → COPY_A/COPY_B device functions rearrange
- RMEM_C is opaque register storage, accessed via MAP_IDX2CRD

### 5-Element LTO Dispatch Return
```python
(func_args, template_args, ltoirs, extra_shared_memory, override_native_func)
```
- 5th element overrides C++ function name (suppresses `wp::` namespace prefix)
- Backward-compatible: all existing dispatch functions return 4 elements

### Integer Constants in Dispatch Args
Must use `Var(str(value), str, False, True, False)` pattern (NOT `Var(value, int, ...)`)

### Deferred Variable Declaration
Warp codegen collects Var objects during AST pass, emits C++ declarations after all dispatches. This allows `tile_matmul` dispatch to backfill `rmem_storage_bytes` on the output tile type created by `tile_zeros`.

### The `float()` Cast Problem
Warp codegen emits `#define float(x) cast_float(x)`. This macro-expands `float()` inside C++ macros in tile.h. Fixed by using `static_cast<wp::float32>()` instead.

## Bugs Fixed Across Sessions

1. **NVRTC: `tile_rmem_t` not visible** → moved struct before `#ifdef __CUDACC__`
2. **NVRTC: void macro in assignment** → split declaration from macro call
3. **NVRTC: `wp::` on macro** → 5-element dispatch return to suppress namespace
4. **PTX link: unresolved `copy_a`** → rebuild warp.so with updated warp.cu
5. **Codegen: `int` not `str` in Var** → use `Var(str(value), str, ...)` pattern
6. **CUDA error 700 (illegal memory access)** → account for extra SMEM in dispatch return
7. **Numerical error ~47 (in-place COPY)** → separate SMEM buffers for suggested layout
8. **Numerical error ~47 (COPY output layout)** → use MAP_IDX2CRD instead of COPY for tile_rmem_copy
9. **`cast_float` undefined** → use `static_cast<wp::float32>()` in macros

## Next Task: PERFORMANCE INVESTIGATION

The user's question: "why both get slower? shouldn't the original wp_flash_attn be like 6ms?"

Possible causes to investigate:
1. **Build target**: Did `--quick` build produce sm_52/sm_75 instead of sm_120? Check nvcc arch flags.
2. **`-ccbin /usr/bin/gcc-12`**: Could this affect optimization? Unlikely but check.
3. **Shared memory config**: Benchmark says `MAX_SMEM=99KB` - RTX 5090 supports 228KB/SM. Maybe `cudaFuncSetAttribute` for max dynamic SMEM isn't being called.
4. **Occupancy**: `block_dim=128*8=1024` for wp_flash_attn (128 tile_m * 8 threads_per_row) - that's 32 warps, very high. Check if this was different before.
5. **Launch overhead**: `wp.synchronize()` per iteration includes Python→CUDA overhead. Compare with CUDA events.
6. **Kernel regression**: Check if any of our changes to tile.h or builtins.py accidentally affected the non-RMEM path.
7. **Git bisect**: The user had commits showing 5.9ms at `beebc48c` - compare that commit's code.

### Quick check to do first:
```bash
git stash && python3 benchmark_flash_attention.py --skip-naive
# Then git stash pop to restore
```
This would confirm whether our changes caused the regression or if it's something else.

### Also check:
```bash
# What SM arch is being compiled for?
grep -r "arch=sm_" warp/_src/  # or check NVRTC flags
# What was the kernel at the fast commit?
git show beebc48c:warp/examples/tile/example_tile_flash_attention.py | head -50
```

## Build Instructions

```bash
# Rebuild warp.so (needed after changing warp.cu, tile.h, cuda_util.cpp)
cd /home/mingfeiguo/Desktop/warp
python3 build_lib.py --quick

# Run tests
python3 test_rmem_minimal.py
python3 test_rmem_scale.py
python3 warp/examples/tile/example_tile_flash_attention.py

# Run benchmark
python3 warp/examples/benchmarks/benchmark_flash_attention.py --skip-naive
```

## Plan File
There's also a detailed implementation plan at:
`/home/mingfeiguo/.claude/plans/fuzzy-yawning-eclipse.md`

Done. I've saved a comprehensive session dump to /home/mingfeiguo/Desktop/warp/SESSION_NOTES.md. It covers:

  - Current status: Correctness works, performance needs investigation
  - All files modified with line numbers and descriptions of changes
  - All 9 bugs fixed across sessions
  - Key technical details (cuBLASDx Tensor API, 5-element dispatch, deferred declaration, etc.)
  - Next task: Performance investigation — why both wp_flash_attn (21.6ms, expected ~6ms) and wp_flash_rmem (20.1ms) are ~25x slower than Triton/flash_attn
  - Investigation plan with specific things to check (SM target, shared memory config, occupancy, git bisect to commit beebc48c which had 5.9ms)
  - Build instructions

  There's also the existing plan file at ~/.claude/plans/fuzzy-yawning-eclipse.md which has the original implementation design.

  When you come back, just say "read SESSION_NOTES.md and continue" and we can pick up right where we left off with the performance investigation.