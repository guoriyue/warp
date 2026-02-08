/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "builtin.h"

#include "rand.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

// Check if the CUDA toolkit is available
#if WP_ENABLE_CUDA || defined(__CUDACC_RTC__)

// If NVRTC is being used, do not include extra headers (NVRTC has built-in float4)
#ifdef __CUDACC_RTC__
// NVRTC: Use built-in float4 (no need for extra definitions)
#else
// NVCC: Include vector_types.h to get float4
#include <cuda_runtime.h>
#endif

#else
// If CUDA is not available (e.g., macOS build), manually define float4
struct alignas(16) float4 {
    float x, y, z, w;
};
#endif

#if defined(__CUDA_ARCH__)
#define WP_TILE_SYNC __syncthreads
#else
#define WP_TILE_SYNC void
#endif

#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define WP_PRAGMA_UNROLL _Pragma("unroll")
#define WP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define WP_PRAGMA_UNROLL #pragma unroll
#define WP_PRAGMA_NO_UNROLL #pragma unroll 1
#endif

#else

#define WP_PRAGMA_UNROLL
#define WP_PRAGMA_NO_UNROLL

#endif

#define WP_USE_ASYNC_PIPELINE 1
#define WP_USE_REGISTER_GEMM 0

#if defined(__CUDACC_RTC__)
#define WP_TILE_THREAD_IDX threadIdx.x
#else
#define WP_TILE_THREAD_IDX 0
#endif  //


/* Tile Expressions

[ ] Tiles
    [x] Register, Shared, Global
    [ ] Layouts
        [x] Simple
        [ ] Cute
    [x] Remove Alloc type from tile_shared_t
    [x] wp.launch_tiled() helper
[ ] Creation
    [x] zeros
    [x] ones
    [x] arange
    [x] tile()
    [x] untile()
    [ ] fromfunction()
    [ ] explicit storage
[ ] Load/Store
    [ ] 1D load/store variants
    [ ] max_coord option for non-aligned loads
    [ ] Indexed load
    [x] wp.tile_atomic_add()
[ ] Maps
    [x] Support user functions
    [x] Support built-in functions
    [ ] Support for lambda functions
    [ ] Infer tile_map() output from operator type (e.g.: dot for each element)
[ ] Reductions
    [x] Sum
        [x] Forward
        [x] Reverse
    [x] Min
    [x] Max
    [x] Custom
[x] MatMul
    [x] Forward
    [x] Reverse
[ ] Operators
    [ ] +, -, *, /, @?
    [ ] += for matmul, e.g.: c += a@b, or c = a@b
[ ] Reshape
    [ ] Broadcasting
    [ ] Transpose
        [x] Shared
        [ ] Register
    [ ] Slice
[ ] Runtime
    [x] Compile-time block dimensions
    [x] Switch between SIMT / Tile based execution if `block_dim` not provided to wp.launch()
[ ] Examples
    [ ] Point registration
    [ ] GEMM
    [ ] MLP
    [ ] LayerNorm
    [ ] SoftMax
    [ ] GEMM
    [ ] Batched MLP
    [ ] Layer norm
    [ ] FNO + Burgers equation
    [ ] Stochastic financial modeling
    [ ] Convolution: https://github.com/NVIDIA/MinkowskiEngine/blob/master/src/convolution_kernel.cu#L123
    [ ] MeshCNN (Modulus, Oliver)
    [ ] BioNemo (Ali)
    [ ] Skinning (David/Or/Vismay)
[ ] Error checking
    [ ] Ensure functions passed to tile_map() are compatible with tile type
    [ ] Ensure that args passed to tile ops are compatible
    [ ] Ensure tile load/store operations don't go out of bounds of arrays in debug mode

*/

/*
Notes on shared memory synchronization
======================================

Currently operations that write to shared memory tiles (e.g.: tile_load())
must synchronize before they return through WP_TILE_SYNC(), this
ensures subsequent read operations from the tile do not cause a race condition.

For tile_shared_t adjoints, the gradient accumulation is done through shared
memory atomics, i.e.: atomic_add(), since for broadcast tiles multiple threads
may map to the same location. Synchronization is still required after these
updates, since subsequent operations e.g.: adj_tile_load() will store the
gradients to memory, and all updates must be visible at that point, e.g.:

    a = wp.tile_load(...)
    b = wp.tile_load(...)
    c = wp.tile_matmul(a, b)
    wp.tile_store(c)

    // loads incoming adjoints from global -> shared
    wp.adj_tile_store(c, adj_c)
    // consumes adj_c, requires synchronization
    wp.adj_tile_matmul(a, b, adj_a, adj_b, adj_c)
    // consumes adj_b, requires synchronization
    wp.adj_tile_load(..., adj_b)
    // consumes adj_b, requires synchronization
    wp.adj_tile_load(..., adj_a)

Generally synchronization to adjoint tiles will happen through the
tile_shared_t::add() and tile_shared_t::assign() function automatically,
but in some cases e.g.: tile_matmul() it is done manually.

The current synchronization strategy is conservative, and can lead to more
synchronization than necessary. A more sophisticated strategy would be
to track the 'dirty' state of shared tiles, and synchronize only when
necessary. In addition, custom synchronization for e.g.: tile_load()
operations could be added through a SyncProvider template parameter on
the tile_shared_t type, for example to support barrier synchronization
for asynchronous global to shared loads.
*/

namespace wp {

// Primary template
template <typename T, typename U> struct is_same {
    static constexpr bool value = false;
};

// Specialization for the case when T and U are the same type
template <typename T> struct is_same<T, T> {
    static constexpr bool value = true;
};

// Helper for dependent static_assert failures
template <typename T> struct always_false {
    static constexpr bool value = false;
};

template <typename T> struct remove_reference {
    using type = T;
};
template <typename T> struct remove_reference<T&> {
    using type = T;
};
template <typename T> struct remove_reference<T&&> {
    using type = T;
};

template <int N> struct tile_coord_t {
    int indices[N];

    CUDA_CALLABLE inline int operator[](int i) const
    {
        assert(0 <= i && i < N);
        return indices[i];
    }
    CUDA_CALLABLE inline int& operator[](int i)
    {
        assert(0 <= i && i < N);
        return indices[i];
    }

    CUDA_CALLABLE inline tile_coord_t<N> operator+(const tile_coord_t<N>& c) const
    {
        tile_coord_t<N> out;
        for (int i = 0; i < N; ++i) {
            out.indices[i] = indices[i] + c.indices[i];
        }
        return out;
    }

    static constexpr int size() { return N; }
};

// This function deduces N = sizeof...(Ints)
template <typename... Ints> constexpr tile_coord_t<sizeof...(Ints)> tile_coord(Ints... idxs)
{
    constexpr int N = sizeof...(Ints);

    // Create the result
    tile_coord_t<N> result {};

    // Capture all arguments in a local array
    int arr[] = { static_cast<int>(idxs)... };

    // C++14 or later: 'for' is allowed in a constexpr context
    for (int i = 0; i < N; ++i) {
        result.indices[i] = arr[i];
    }

    return result;
}

// helpers to construct a coord from a set of indices
inline auto tile_coord(int i)
{
    auto c = tile_coord_t<1>();
    c.indices[0] = i;
    return c;
}

inline auto tile_coord(int i, int j)
{
    auto c = tile_coord_t<2>();
    c.indices[0] = i;
    c.indices[1] = j;
    return c;
}

inline auto tile_coord(int i, int j, int k)
{
    auto c = tile_coord_t<3>();
    c.indices[0] = i;
    c.indices[1] = j;
    c.indices[2] = k;
    return c;
}

inline auto tile_coord(int i, int j, int k, int l)
{
    auto c = tile_coord_t<4>();
    c.indices[0] = i;
    c.indices[1] = j;
    c.indices[2] = k;
    c.indices[3] = l;
    return c;
}

// represents a compile time int tuple for strides/shapes/coords
template <int... V> struct tile_tuple_t {
    static constexpr int N = sizeof...(V);
    static_assert(N > 0, "Expected N > 0");

    static constexpr int data[N] = { V... };

    static constexpr int dim(int i)
    {
        assert(i < N);
        return data[i];
    }
    static constexpr int size()
    {
        int res = data[0];
        for (int i = 1; i < N; ++i)
            res *= data[i];

        return res;
    }
};

// simple helper to compute strides from a shape up to 4d
template <typename Shape> struct compute_strides;

// 1D
template <int D0> struct compute_strides<tile_tuple_t<D0>> {
    using Stride = tile_tuple_t<1>;
};
// 2D
template <int D0, int D1> struct compute_strides<tile_tuple_t<D0, D1>> {
    using Stride = tile_tuple_t<D1, 1>;
};
// 3D
template <int D0, int D1, int D2> struct compute_strides<tile_tuple_t<D0, D1, D2>> {
    using Stride = tile_tuple_t<(D1 * D2), D2, 1>;
};
// 4D
template <int D0, int D1, int D2, int D3> struct compute_strides<tile_tuple_t<D0, D1, D2, D3>> {
    using Stride = tile_tuple_t<(D1 * D2 * D3), (D2 * D3), D3, 1>;
};


// alias of tuple to represent shapes
template <int... V> using tile_shape_t = tile_tuple_t<V...>;

// alias of tuple to represent stride
template <int... V> using tile_stride_t = tile_tuple_t<V...>;


// helper to remove a dimension from a shape (used for axis reductions)
template <int Axis, typename Shape> struct tile_shape_remove_dim {
    static_assert(Axis >= 0 && Axis < Shape::N, "Axis out of bounds for tile_shape_remove_dim");
};

// 1D -> scalar
template <int D0> struct tile_shape_remove_dim<0, tile_shape_t<D0>> {
    using type = tile_shape_t<1>;
};

// 2D -> 1D
template <int D0, int D1> struct tile_shape_remove_dim<0, tile_shape_t<D0, D1>> {
    using type = tile_shape_t<D1>;
};

template <int D0, int D1> struct tile_shape_remove_dim<1, tile_shape_t<D0, D1>> {
    using type = tile_shape_t<D0>;
};

// 3D -> 2D
template <int D0, int D1, int D2> struct tile_shape_remove_dim<0, tile_shape_t<D0, D1, D2>> {
    using type = tile_shape_t<D1, D2>;
};

template <int D0, int D1, int D2> struct tile_shape_remove_dim<1, tile_shape_t<D0, D1, D2>> {
    using type = tile_shape_t<D0, D2>;
};

template <int D0, int D1, int D2> struct tile_shape_remove_dim<2, tile_shape_t<D0, D1, D2>> {
    using type = tile_shape_t<D0, D1>;
};

// 4D -> 3D
template <int D0, int D1, int D2, int D3> struct tile_shape_remove_dim<0, tile_shape_t<D0, D1, D2, D3>> {
    using type = tile_shape_t<D1, D2, D3>;
};

template <int D0, int D1, int D2, int D3> struct tile_shape_remove_dim<1, tile_shape_t<D0, D1, D2, D3>> {
    using type = tile_shape_t<D0, D2, D3>;
};

template <int D0, int D1, int D2, int D3> struct tile_shape_remove_dim<2, tile_shape_t<D0, D1, D2, D3>> {
    using type = tile_shape_t<D0, D1, D3>;
};

template <int D0, int D1, int D2, int D3> struct tile_shape_remove_dim<3, tile_shape_t<D0, D1, D2, D3>> {
    using type = tile_shape_t<D0, D1, D2>;
};


// helper to insert an axis value into a coordinate (inverse of removing dimension)
// used for mapping output coordinates back to input coordinates during axis reduction
template <int Axis, int N>
CUDA_CALLABLE constexpr auto tile_coord_insert_axis(const tile_coord_t<N>& coord, int axis_val)
{
    static_assert(Axis >= 0 && Axis <= N, "Axis out of bounds for tile_coord_insert_axis");

    if constexpr (N == 0) {
        // Scalar -> 1D
        static_assert(Axis == 0, "Invalid axis for scalar coordinate");
        return tile_coord(axis_val);
    } else if constexpr (N == 1) {
        // 1D -> 2D
        if constexpr (Axis == 0)
            return tile_coord(axis_val, coord[0]);
        else
            return tile_coord(coord[0], axis_val);
    } else if constexpr (N == 2) {
        // 2D -> 3D
        if constexpr (Axis == 0)
            return tile_coord(axis_val, coord[0], coord[1]);
        else if constexpr (Axis == 1)
            return tile_coord(coord[0], axis_val, coord[1]);
        else
            return tile_coord(coord[0], coord[1], axis_val);
    } else  // N == 3
    {
        // 3D -> 4D
        if constexpr (Axis == 0)
            return tile_coord(axis_val, coord[0], coord[1], coord[2]);
        else if constexpr (Axis == 1)
            return tile_coord(coord[0], axis_val, coord[1], coord[2]);
        else if constexpr (Axis == 2)
            return tile_coord(coord[0], coord[1], axis_val, coord[2]);
        else
            return tile_coord(coord[0], coord[1], coord[2], axis_val);
    }
}


// represents a tile stored in global memory with dynamic strides
// used to represent the source and offset for tile loads to register/shared
// BoundsCheck: when true (default), validates array access bounds; when false, skips validation for performance
template <typename T, typename Shape_, bool BoundsCheck = true> struct tile_global_t {
    using Type = T;
    using Shape = Shape_;
    using Coord = tile_coord_t<Shape::N>;

    array_t<T> data;
    Coord offset;

    tile_global_t(array_t<T>& a, const Coord& c)
        : data(a)
        , offset(c)
    {
    }

    inline CUDA_CALLABLE int index_from_coord(const Coord& coord) const
    {
        // element index
        int index = 0;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Shape::N; ++i) {
            // global = offset + coord
            int c = offset[i] + coord[i];
            index += data.strides[i] * c;
        }

        return index / sizeof(T);
    }

    inline CUDA_CALLABLE bool index(const Coord& coord, int& out) const
    {
        if constexpr (BoundsCheck) {
            // element index
            int index = 0;

            WP_PRAGMA_UNROLL
            for (int i = 0; i < Shape::N; ++i) {
                // global = offset + coord
                int c = offset[i] + coord[i];

                // handle out of bounds case
                if (c >= data.shape[i])
                    return false;
                else
                    index += data.strides[i] * c;
            }

            // array strides are in bytes so we convert to elements
            out = index / sizeof(T);
            return true;
        } else {
            out = index_from_coord(coord);
            return true;
        }
    }

    inline CUDA_CALLABLE T load(const Coord& coord) const
    {
        int i;
        if (index(coord, i))
            return data.data[i];
        else
            return T {};
    }

    inline CUDA_CALLABLE T load_grad(const Coord& coord) const
    {
        int i;
        if (index(coord, i))
            return data.grad[i];
        else
            return T {};
    }

    inline CUDA_CALLABLE void store(const Coord& coord, const T& x) const
    {
        int i;
        if (index(coord, i))
            data.data[i] = x;
    }

    inline CUDA_CALLABLE T atomic_add(const Coord& coord, const T& value) const
    {
        int i;
        if (index(coord, i))
            return wp::atomic_add(&data.data[i], value);
        else
            return T {};
    }

    inline CUDA_CALLABLE T atomic_add_grad(const Coord& coord, const T& grad) const
    {
        int i;
        if (index(coord, i))
            return wp::atomic_add(&data.grad[i], grad);
        else
            return T {};
    }
};


template <typename Shape_> struct tile_layout_register_t {
    using Shape = Shape_;
    using Coord = tile_coord_t<Shape::N>;

    static constexpr int Size = Shape::size();
    static constexpr int NumRegs = (Size + WP_TILE_BLOCK_DIM - 1) / WP_TILE_BLOCK_DIM;
    static constexpr bool Aligned = Size % WP_TILE_BLOCK_DIM == 0;

    static inline CUDA_CALLABLE int linear_from_register(int reg)
    {
        return WP_TILE_THREAD_IDX + reg * WP_TILE_BLOCK_DIM;
    }

    static inline CUDA_CALLABLE int linear_from_coord(Coord c)
    {
        int linear = 0;
        int stride = 1;

        WP_PRAGMA_UNROLL
        for (int i = Shape::N - 1; i >= 0; --i) {
            linear += c[i] * stride;
            stride *= Shape::dim(i);
        }
        return linear;
    }

    static inline CUDA_CALLABLE auto coord_from_linear(int linear)
    {
        Coord c;

        WP_PRAGMA_UNROLL
        for (int i = Shape::N - 1; i >= 0; --i) {
            c[i] = linear % Shape::dim(i);
            linear /= Shape::dim(i);
        }

        return c;
    }

    static inline CUDA_CALLABLE int thread_from_linear(int linear)
    {
        const int thread = linear % WP_TILE_BLOCK_DIM;
        return thread;
    }

    static inline CUDA_CALLABLE int register_from_linear(int linear)
    {
        const int reg = linear / WP_TILE_BLOCK_DIM;
        return reg;
    }

    static inline CUDA_CALLABLE bool valid(int linear)
    {
        if (Aligned || linear < Size)
            return true;
        else
            return false;
    }
};

// Opaque register-memory tile for cuBLASDx Tensor API RMEM accumulator.
// StorageBytes is determined at LTO build time by querying CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES.
// The layout is opaque; only cuBLASDx device functions can read/write the data.
// The tile_matmul_rmem, tile_rmem_copy, tile_rmem_scale macros (below) handle the actual operations.
template <int StorageBytes>
struct tile_rmem_t {
    alignas(16) char buf[StorageBytes];

    // Accept any scalar assignment (e.g. from tile_zeros which returns T{}).
    // Zero-initializes the opaque buffer; for fp16/fp32 zero bits == 0.0.
    template <typename T>
    inline CUDA_CALLABLE tile_rmem_t& operator=(const T&) {
        memset(buf, 0, StorageBytes);
        return *this;
    }
};

// represents a tile stored in registers across a block
template <typename T, typename L> struct tile_register_t {
    using Type = T;
    using Layout = L;

    T data[Layout::NumRegs];

    inline CUDA_CALLABLE tile_register_t(T value = T {})
    {
        // zero-initialize by default necessary for tile adjoints
        // need to check if this results in worse codegen
        // than doing adj_var = tile_zeros() explicitly
        // in backwards pass and letting default constructor
        // avoid initialization

        for (int i = 0; i < Layout::NumRegs; ++i)
            data[i] = value;
    }

    template <bool BoundsCheck>
    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, typename Layout::Shape, BoundsCheck>& t)
    {
        copy_from_global(t);
        return *this;
    }

    // define the += operator which is used during backward pass codegen
    // when returning a register tile from a user defined function
    inline CUDA_CALLABLE auto& operator+=(const tile_register_t<T, Layout>& rhs)
    {
        grad_add(rhs);
        return *this;
    }

    inline CUDA_CALLABLE T& operator()(int reg)
    {
        assert(reg < Layout::NumRegs);
        return data[reg];
    }

    inline CUDA_CALLABLE const T& operator()(int reg) const
    {
        assert(reg < Layout::NumRegs);
        return data[reg];
    }

    inline CUDA_CALLABLE void assign(const tile_register_t<T, Layout>& tile)
    {
        for (int i = 0; i < Layout::NumRegs; ++i)
            data[i] = tile.data[i];
    }

    inline CUDA_CALLABLE void zero()
    {
        for (int i = 0; i < Layout::NumRegs; ++i)
            data[i] = T {};
    }

    // extract a single tile element to a native type
    template <typename Coord> inline CUDA_CALLABLE Type extract(const Coord& c)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = Layout::linear_from_coord(c);
        const int thread = Layout::thread_from_linear(linear);
        const int reg = Layout::register_from_linear(linear);

#if defined(__CUDA_ARCH__)
        __shared__ Type scratch;
#else
        Type scratch;
#endif

        // ensure any previously scheduled threads have finished reading from scratch
        WP_TILE_SYNC();

        if (WP_TILE_THREAD_IDX == thread) {
            scratch = data[reg];
        }

        // ensure extraction thread has updated smem
        WP_TILE_SYNC();

        return scratch;
    }


    // backward version of scalar extract
    template <typename Coord> inline CUDA_CALLABLE void adj_extract(const Coord& c, Type adj_ret)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = Layout::linear_from_coord(c);
        const int thread = Layout::thread_from_linear(linear);
        const int reg = Layout::register_from_linear(linear);

        if (WP_TILE_THREAD_IDX == thread) {
            data[reg] += adj_ret;
        }
    }

    inline CUDA_CALLABLE void print() const;


    // return the in-register version of this tile (nop)
    inline CUDA_CALLABLE auto& copy_to_register() { return *this; }

    inline CUDA_CALLABLE const auto& copy_to_register() const { return *this; }

    // apply a lambda to all valid entries in the tile
    // Op should be a functor that takes a register index and tile_coord_t as input
    template <typename Op> void apply(Op op)
    {
        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;

            auto c = Layout::coord_from_linear(linear);
            op(i, c);
        }
    }


    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero() { zero(); }

    // accumulate gradients onto this tile
    inline CUDA_CALLABLE void grad_add(const tile_register_t<T, Layout>& tile)
    {
        for (int i = 0; i < Layout::NumRegs; ++i)
            data[i] += tile.data[i];
    }

    inline CUDA_CALLABLE void grad_add(const tile_global_t<T, typename Layout::Shape>& global)
    {
        apply([&](int reg, auto c) { data[reg] += global.load_grad(c); });
    }

    // zero the gradient in a global array for the elements covered by this tile
    template <typename Global> inline CUDA_CALLABLE void grad_zero_global(const Global& global)
    {
        apply([&](int reg, auto c) {
            int i;
            if (global.index(c, i))
                global.data.grad[i] = T();
        });
    }

    inline CUDA_CALLABLE auto& grad_to_register()
    {
        // nop for register tiles
        return *this;
    }

    template <typename Global> inline CUDA_CALLABLE void copy_to_global(const Global& dest)
    {
        apply([&](int reg, auto c) { dest.store(c, data[reg]); });
    }

    template <typename Global> inline CUDA_CALLABLE void copy_from_global(const Global& src)
    {
        apply([&](int reg, auto c) { data[reg] = src.load(c); });
    }

    // add a register tile to a global array
    template <typename Global> inline CUDA_CALLABLE auto atomic_add(const Global& dest)
    {
        // allocate a tile to hold previous dest value
        auto previous = *this;

        apply([&](int reg, auto c) { previous.data[reg] = dest.atomic_add(c, data[reg]); });
        return previous;
    }

    // add a register tile to the gradient of a global array
    template <typename Global> inline CUDA_CALLABLE auto atomic_add_grad(const Global& dest)
    {
        // allocate a tile to hold previous dest value
        auto previous = *this;

        apply([&](int reg, auto c) { previous.data[reg] = dest.atomic_add_grad(c, data[reg]); });
        return previous;
    }
};


// helper to allocate a register tile like another tile
// users can either specify a template explicitly or
// pass in another concrete instance
template <typename Tile> auto tile_register_like(Tile* t = nullptr)
{
    using T = typename Tile::Type;
    using L = typename Tile::Layout;

    return tile_register_t<T, tile_layout_register_t<typename L::Shape>>(T {});
}

// helper to construct a register tile from a type and a list of dims
template <typename T, int... Dims> auto tile_register()
{
    return tile_register_t<T, tile_layout_register_t<tile_shape_t<Dims...>>>();
}

inline CUDA_CALLABLE int tile_align(int num_bytes)
{
    // note this much match value in Python types.py
    const int alignment = 16;

    const int num_bytes_abs = num_bytes < 0 ? -num_bytes : num_bytes;
    const int sign = num_bytes < 0 ? -1 : 1;

    return sign * ((num_bytes_abs + alignment - 1) / alignment) * alignment;
}

#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
// On the CPU we use a fixed size block of stack memory for shared tile allocations.
// We store a pointer to the current allocation storage either in a reserved register
// (AArch64) or a static variable (x86-64).
#if !defined(__CUDA_ARCH__)
class tile_shared_storage_t;
#if defined(__aarch64__)
// x28 is is the last callee-saved register on AArch64. This allows us to call externally
// compiled functions without worrying about clobbering the pointer.
// We pass -target-feature +reserve-x28 to Clang to exclude it from register allocation.
register tile_shared_storage_t* shared_tile_storage asm("x28");
#else
// Ideally this would be thread_local, but LLVM's JIT doesn't support TLS yet
// There is also no support for something like -ffixed-r15 either
static tile_shared_storage_t* shared_tile_storage;
#endif
#endif
#endif

// This class manages a block of "shared" memory for use by tiles.
// On the GPU this maps to dynamic shared memory, while on the CPU we allocate
// a fixed size block of memory on the stack and manage allocations from it.
// An instance of this class gets created at the start of a kernel.
class tile_shared_storage_t {
private:
#if !defined(__CUDA_ARCH__)
#define WP_MAX_CPU_SHARED 256*1024
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    tile_shared_storage_t* old_value;
    unsigned int smem_base[WP_TILE_BLOCK_DIM];
    char dynamic_smem_base[WP_MAX_CPU_SHARED];  // on CPU allocate a fixed 256k block to use for shared allocs
#endif
#endif

    // we maintain a per-thread offset into dynamic
    // shared memory that allows us to keep track of
    // current use across dynamic function calls
    static inline CUDA_CALLABLE unsigned int* get_smem_base()
    {
#if defined(__CUDA_ARCH__)
        __shared__ unsigned int smem_base[WP_TILE_BLOCK_DIM];
        return smem_base;
#elif defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        return shared_tile_storage->smem_base;
#else
        static unsigned int smem_base[WP_TILE_BLOCK_DIM];
        return smem_base;
#endif
    }

    static inline CUDA_CALLABLE char* get_dynamic_smem_base()
    {
#if defined(__CUDA_ARCH__)
        extern __shared__ char dynamic_smem_base[];
        return dynamic_smem_base;
#elif defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        return shared_tile_storage->dynamic_smem_base;
#else
        static char dynamic_smem_base[WP_MAX_CPU_SHARED];
        return dynamic_smem_base;
#endif
    }

public:
    // cppcheck-suppress uninitMemberVar
    inline CUDA_CALLABLE tile_shared_storage_t()
    {
#if !defined(__CUDA_ARCH__) && defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        // On the CPU save a pointer to this instance in a reserved register
        // or static variable so it can be accessed from anywhere within a kernel.
        old_value = shared_tile_storage;
        shared_tile_storage = this;
#endif

        init();
    }

    inline CUDA_CALLABLE ~tile_shared_storage_t()
    {
        check();

#if !defined(__CUDA_ARCH__) && defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
        shared_tile_storage = old_value;
#endif
    }

    static inline CUDA_CALLABLE void init()
    {
        unsigned int* smem_base = get_smem_base();

        smem_base[WP_TILE_THREAD_IDX] = 0;
    }

    static inline CUDA_CALLABLE void check()
    {
        unsigned int* smem_base = get_smem_base();

        assert(smem_base[WP_TILE_THREAD_IDX] == 0);
    }

    static inline CUDA_CALLABLE void* alloc(int num_bytes)
    {
        unsigned int* smem_base = get_smem_base();
        char* dynamic_smem_base = get_dynamic_smem_base();

        const unsigned int offset = smem_base[WP_TILE_THREAD_IDX];

        // one entry per-thread so no need for synchronization
        smem_base[WP_TILE_THREAD_IDX] += tile_align(num_bytes);

#if !defined(__CUDA_ARCH__)
        assert(smem_base[WP_TILE_THREAD_IDX] <= WP_MAX_CPU_SHARED);
#endif

        return &(dynamic_smem_base[offset]);
    }
};


template <typename Shape_, typename Stride_ = typename compute_strides<Shape_>::Stride> struct tile_layout_strided_t {
    using Shape = Shape_;
    using Stride = Stride_;
    using Coord = tile_coord_t<Shape::N>;

    static constexpr int Size = Shape::size();
    static constexpr bool Aligned = Size % WP_TILE_BLOCK_DIM == 0;

    static inline CUDA_CALLABLE auto coord_from_linear(int linear)
    {
        assert(linear < Size);

        Coord c;

        WP_PRAGMA_UNROLL
        for (int d = Shape::N - 1; d >= 0; --d) {
            c[d] = linear % Shape::dim(d);
            linear /= Shape::dim(d);
        }

        return c;
    }

    static inline CUDA_CALLABLE int index_from_coord(Coord c)
    {
        int index = 0;

        WP_PRAGMA_UNROLL
        for (int d = 0; d < Shape::N; ++d) {
            assert(c[d] < Shape::dim(d));

            index += c[d] * Stride::dim(d);
        }

        return index;
    }

    // checks whether a strided layout is unique, i.e.: if memory locations are only
    // ever referred to by one element in the tile, this is a basic test that only
    // checks for broadcast dimensions, it would be possible to do the full check
    // using sorted shape/strides in Python and add it as a template parameter to the type
    static constexpr bool is_unique()
    {
        constexpr int N = Shape::N;

        // check for any broadcast dimensions
        for (int i = 0; i < N; ++i)
            if (Stride::dim(i) == 0)
                return false;

        return true;
    }

    static constexpr bool Unique = is_unique();

    static inline CUDA_CALLABLE bool valid(int linear) { return linear < Size; }
};


template <typename T, typename L, bool Owner_ = true> struct tile_shared_t {
    using Type = T;
    using Layout = L;
    static constexpr bool Owner = Owner_;

    struct Storage {
        T* ptr;

        Storage(T* p)
            : ptr(p)
        {
        }

        inline CUDA_CALLABLE T& operator()(typename Layout::Coord c)
        {
            assert(ptr);

            int index = Layout::index_from_coord(c);
            return ptr[index];
        }

        inline CUDA_CALLABLE const T& operator()(typename Layout::Coord c) const
        {
            assert(ptr);

            int index = Layout::index_from_coord(c);
            return ptr[index];
        }

        inline CUDA_CALLABLE T& operator()(int linear)
        {
            assert(ptr);
            assert(Layout::valid(linear));

            auto c = Layout::coord_from_linear(linear);
            return (*this)(c);
        }

        inline CUDA_CALLABLE const T& operator()(int linear) const
        {
            assert(ptr);
            assert(Layout::valid(linear));

            auto c = Layout::coord_from_linear(linear);
            return (*this)(c);
        }
    };

    Storage data;
    Storage grad;

    // we need to track whether or not this tile's data has been initialized.
    // once true, any re-initialization of data that follows needs a WP_TILE_SYNC()
    // call to precede it, to allow threads that are still reading from this tile
    // to complete their work. e.g, in a dynamic loop:
    // for i in range(x):
    //     tile = wp.tile_load(arr, i, TILE_SIZE, storage="shared")
    //     # read from tile...
    bool initialized;

    // default initialization (non-initialized)
    inline CUDA_CALLABLE tile_shared_t()
        : data(nullptr)
        , grad(nullptr)
        , initialized(false)
    {
    }

    // we delete the copy constructor because in the case the shared tile is owning,
    // this leads to a double deallocation.
    // this also forces one to handle copies explicitly
    inline CUDA_CALLABLE tile_shared_t(const tile_shared_t& other)
        : data(other.data)
        , grad(other.grad)
        , initialized(other.initialized)
    {
        static_assert(!Owner, "Copy constructor is only supported for non-owning tiles.");
    }

    // move constructor
    inline CUDA_CALLABLE tile_shared_t(tile_shared_t&& other)
        : data(other.data)
        , grad(other.grad)
        , initialized(other.initialized)
    {
        other.data.ptr = nullptr;
        other.grad.ptr = nullptr;
    }

    template <typename OtherT, typename OtherLayout, bool OtherOwner>
    inline CUDA_CALLABLE tile_shared_t(const tile_shared_t<OtherT, OtherLayout, OtherOwner>& other)
        : data(other.data.ptr)
        , grad(other.grad.ptr)
        , initialized(other.initialized)
    {
        static_assert(!Owner, "Copy constructor is only supported for non-owning tiles.");
        static_assert(Layout::Size == OtherLayout::Size, "Expected Size == OtherLayout::Size");
    }

    // initialize from an existing tile's memory
    inline CUDA_CALLABLE tile_shared_t(T* data, T* grad = nullptr, bool initialized = true)
        : data(data)
        , grad(grad)
        , initialized(initialized)
    {
    }

    inline CUDA_CALLABLE ~tile_shared_t()
    {
        if (Owner) {
            // update our per-thread shared memory allocator
            if (data.ptr)
                tile_shared_storage_t::alloc(-Layout::Size * int(sizeof(T)));

            if (grad.ptr)
                tile_shared_storage_t::alloc(-Layout::Size * int(sizeof(T)));
        }
    }

    // assign from a register tile
    inline CUDA_CALLABLE auto& operator=(const tile_register_t<Type, tile_layout_register_t<typename Layout::Shape>>& t)
    {
        assign(t);
        return *this;
    }

    // construct from another shared tile, this constructor
    // is invoked for reshape operations like `wp.tile_transpose()`
    // or `wp::copy()`
    template <typename OtherT, typename OtherLayout, bool OtherOwner>
    inline CUDA_CALLABLE auto& operator=(const tile_shared_t<OtherT, OtherLayout, OtherOwner>& rhs)
    {
        // check dimensions are compatible
        static_assert(Layout::Size == OtherLayout::Size, "Expected Size == OtherLayout::Size");


        if (Owner) {
            // if the tile owns the data we need to copy
            assign(rhs);
        } else {
            // alias tile directly
            data.ptr = rhs.data.ptr;
            grad.ptr = rhs.grad.ptr;
            initialized = rhs.initialized;
        }

        return *this;
    }

    inline CUDA_CALLABLE auto& operator=(const tile_shared_t& rhs)
    {
        if (Owner) {
            // if the tile owns the data we need to copy
            assign(rhs);
        } else {
            // alias tile directly
            data.ptr = rhs.data.ptr;
            grad.ptr = rhs.grad.ptr;
            initialized = rhs.initialized;
        }

        return *this;
    }

    // assign from a global tile (load)

    template <bool BoundsCheck>
    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, typename Layout::Shape, BoundsCheck>& t)
    {
        copy_from_global(t);
        return *this;
    }

    // assign from a constant value
    inline CUDA_CALLABLE auto& operator=(const T& x)
    {
        // sync if we are re-initializing data so that any threads that are still
        // reading from this tile can complete their work, e.g.: if re-assigning
        // to a tile during a dynamic loop
        if (initialized)
            WP_TILE_SYNC();

        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
            data(i) = x;

        initialized = true;
        WP_TILE_SYNC();
        return *this;
    }

    // define the += operator which is used during backward pass codegen
    // when returning a register tile from a user defined function
    template <typename OtherLayout> inline CUDA_CALLABLE auto& operator+=(const tile_register_t<T, OtherLayout>& rhs)
    {
        grad_add(rhs);
        return *this;
    }

    inline CUDA_CALLABLE auto& operator+=(const tile_shared_t<T, Layout>& rhs)
    {
        grad_add(rhs);
        return *this;
    }

    // in-place zero
    inline CUDA_CALLABLE void zero()
    {
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
            data(i) = T {};

        WP_TILE_SYNC();
    }

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(const typename Layout::Coord& c) { return data(c); }

    // backward of scalar extraction
    inline CUDA_CALLABLE void adj_extract(const typename Layout::Coord& c, Type adj_ret)
    {
        // since multiple threads may extract the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&grad(c), adj_ret);

        WP_TILE_SYNC();
    }

    // direct per-element write (no sync)
    // caller must follow with tile_sync() before any collective op that reads this tile
    inline CUDA_CALLABLE void insert(const typename Layout::Coord& c, const Type& val)
    {
        data(c) = val;
    }

    // backward of scalar insertion
    inline CUDA_CALLABLE void adj_insert(const typename Layout::Coord& c, Type& adj_val)
    {
        adj_val += grad(c);
    }

    // add scalar value onto a single tile element
    inline CUDA_CALLABLE void add_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may add to the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&data(c), x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar addition
    inline CUDA_CALLABLE void adj_add_inplace(const typename Layout::Coord& c, Type& adj_x) { adj_x += grad(c); }

    // subtract scalar value from a single tile element
    inline CUDA_CALLABLE void sub_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may add to the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&data(c), -x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar subtraction
    inline CUDA_CALLABLE void adj_sub_inplace(const typename Layout::Coord& c, Type& adj_x) { adj_x -= grad(c); }

    // perform AND between a scalar value and a single tile element
    inline CUDA_CALLABLE void bit_and_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may access the same element
        // we need to access using atomic operations
        wp::atomic_and(&data(c), x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar AND
    inline CUDA_CALLABLE void adj_bit_and_inplace(const typename Layout::Coord& c, Type& adj_x) { }


    // perform OR between a scalar value and a single tile element
    inline CUDA_CALLABLE void bit_or_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may access the same element
        // we need to access using atomic operations
        wp::atomic_or(&data(c), x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar OR
    inline CUDA_CALLABLE void adj_bit_or_inplace(const typename Layout::Coord& c, Type& adj_x) { }

    // perform XOR between a scalar value and a single tile element
    inline CUDA_CALLABLE void bit_xor_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may access the same element
        // we need to access using atomic operations
        wp::atomic_xor(&data(c), x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar XOR
    inline CUDA_CALLABLE void adj_bit_xor_inplace(const typename Layout::Coord& c, Type& adj_x) { }

    // copy register tile to shared
    template <typename Tile> inline CUDA_CALLABLE void assign(const Tile& tile)
    {
        if (initialized)
            WP_TILE_SYNC();

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Tile::Layout::NumRegs; ++i) {
            const int linear = Tile::Layout::linear_from_register(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Tile::Layout::valid(linear))
                break;

            data(linear) = tile.data[i];
        }

        initialized = true;
        WP_TILE_SYNC();
    }

    // shared tile deep copy
    template <typename OtherT, typename OtherLayout, bool OtherOwner>
    inline CUDA_CALLABLE void assign(const tile_shared_t<OtherT, OtherLayout, OtherOwner>& tile)
    {
        // check dimensions are compatible
        static_assert(Layout::Size == OtherLayout::Size, "Expected Size == OtherLayout::Size");

        if (initialized)
            WP_TILE_SYNC();

        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            data(c) = tile.data(c);
        }

        initialized = true;
        WP_TILE_SYNC();
    }

    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero()
    {
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
            grad(i) = T {};

        WP_TILE_SYNC();
    }


    // accumulate gradients onto this tile
    template <typename Tile> inline CUDA_CALLABLE void grad_add(const Tile& tile)
    {
        WP_PRAGMA_UNROLL
        for (int i = 0; i < Tile::Layout::NumRegs; ++i) {
            const int linear = Tile::Layout::linear_from_register(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Tile::Layout::valid(linear))
                break;

            // if the destination layout is unique (no broadcast dimensions)
            // then we can use regular non-atomic accmulation
            if constexpr (Layout::Unique)
                grad(linear) += tile.data[i];
            else
                // use shared memory atomics to accumulate gradients
                // since for broadcast tiles (e.g.: a bias vector) multiple incoming threads
                // may map to a single location in shared memory
                wp::atomic_add(&grad(linear), tile.data[i]);
        }

        WP_TILE_SYNC();
    }

    // accumulate gradients onto this tile from another shared tile
    inline CUDA_CALLABLE void grad_add(const tile_shared_t<T, Layout>& tile)
    {
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            grad(c) += tile.grad(c);
        }

        WP_TILE_SYNC();
    }

    // accumulate gradient onto this tile from a global array
    inline CUDA_CALLABLE void grad_add(const tile_global_t<T, typename Layout::Shape>& global)
    {
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            T g = global.load_grad(c);

            if constexpr (Layout::Unique) {
                // if the destination layout is unique (no broadcast dimensions)
                // then we can use regular non-atomic accumulation
                grad(c) += g;
            } else {
                // use shared memory atomics to accumulate gradients
                // since for broadcast tiles (e.g.: a bias vector) multiple incoming threads
                // may map to a single location in shared memory
                wp::atomic_add(&grad(c), g);
            }
        }

        WP_TILE_SYNC();
    }

    // zero the gradient in a global array for the elements covered by this tile
    template <typename Global> inline CUDA_CALLABLE void grad_zero_global(const Global& global)
    {
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            int idx;
            if (global.index(c, idx))
                global.data.grad[idx] = T();
        }

        WP_TILE_SYNC();
    }

    // copy shared tile to register
    inline CUDA_CALLABLE auto grad_to_register()
    {
        using Tile = tile_register_t<T, tile_layout_register_t<typename Layout::Shape>>;
        Tile out;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Tile::Layout::NumRegs; ++i) {
            const int linear = Tile::Layout::linear_from_register(i);

            if (!Tile::Layout::valid(linear))
                break;

            out(i) = grad(linear);
        }

        return out;
    }

    // copy shared tile to register
    inline CUDA_CALLABLE auto copy_to_register() const
    {

        auto out = tile_register_like(this);

        using Layout = typename decltype(out)::Layout;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            const int linear = Layout::linear_from_register(i);

            if (!Layout::valid(linear))
                break;

            out(i) = data(linear);
        }

        return out;
    }

    template <typename Global> inline CUDA_CALLABLE void copy_to_global(const Global& dest)
    {

#if defined(__CUDA_ARCH__)
        // vectorized loads for specific input/output shapes
        if constexpr (Layout::Shape::N == 2) {
            constexpr int lastdim = Layout::Shape::N - 1;
            constexpr bool contiguous_src = Layout::Stride::dim(lastdim) == 1;
            const bool contiguous_dest = dest.data.strides[lastdim] == sizeof(T);
            const int elements = min(Layout::Shape::dim(1), (dest.data.shape[lastdim] - dest.offset[lastdim]));
            const bool aligned_size = (elements * sizeof(T)) % sizeof(float4) == 0;
            const bool aligned_stride = (dest.data.strides[0] / sizeof(T)) % Layout::Stride::dim(0) == 0;

            float4* dest128 = (float4*)&dest.data.data[dest.index_from_coord(tile_coord(0, 0))];
            const bool aligned_dst = (uint64_t)(dest128) % sizeof(float4) == 0;

            constexpr int M = Layout::Shape::dim(0);
            constexpr int N = (Layout::Shape::dim(1) * sizeof(T)) / sizeof(float4);

            if (contiguous_dest && contiguous_src && aligned_size && aligned_dst && aligned_stride && N) {
                // alias of shared tile with 128bit type
                using SrcLayout = tile_layout_strided_t<tile_shape_t<M, N>>;
                tile_shared_t<float4, SrcLayout, false> src128((float4*)data.ptr);

                assert(((uint64_t)(data.ptr)) % sizeof(float4) == 0);
                assert(((uint64_t)(dest128)) % sizeof(float4) == 0);

                const int stride_i = dest.data.strides[0] / sizeof(float4);
                const int stride_j = 1;

                WP_PRAGMA_UNROLL
                for (int i = WP_TILE_THREAD_IDX; i < SrcLayout::Size; i += WP_TILE_BLOCK_DIM) {
                    auto c = SrcLayout::coord_from_linear(i);

                    dest128[stride_i * c[0] + stride_j * c[1]] = src128.data(i);
                }

                return;
            }
        }

#endif  // defined(__CUDA_ARCH__)

        // scalar bounds checked path
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            dest.store(c, data(i));
        }
    }

    inline CUDA_CALLABLE void cp_async_global_to_shared_128(float4* shared_dest, const float4* global_src)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

        unsigned long long saddr = 0ULL;
        unsigned long long gaddr = 0ULL;

        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(saddr) : "l"(shared_dest));
        asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(gaddr) : "l"(global_src));

        // Use cp.async on newer architectures
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "l"(saddr), "l"(gaddr));
#else
        // use regular load/store through register on older arches
        *shared_dest = *global_src;
#endif
    }

    inline CUDA_CALLABLE void cp_async_commit_and_wait_all_128()
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile("cp.async.commit_group;\n"
                     "cp.async.wait_group 0;\n" ::);
#endif
    }

    template <typename Global> inline CUDA_CALLABLE void copy_from_global(const Global& src)
    {
        if (initialized)
            WP_TILE_SYNC();

#if defined(__CUDA_ARCH__)

        // vectorized loads for specific input/output shapes
        if constexpr (Layout::Shape::N == 2) {
            constexpr int lastdim = Layout::Shape::N - 1;
            constexpr bool contiguous_dest = Layout::Stride::dim(lastdim) == 1;
            const bool contiguous_src = src.data.strides[lastdim] == sizeof(T);
            const int elements = min(Layout::Shape::dim(1), (src.data.shape[lastdim] - src.offset[lastdim]));
            const bool aligned_size = (elements * sizeof(T)) % sizeof(float4) == 0;
            const bool aligned_stride = (src.data.strides[0] / sizeof(T)) % Layout::Stride::dim(0) == 0;

            float4* src128 = (float4*)&src.data.data[src.index_from_coord(tile_coord(0, 0))];
            const bool aligned_src = (uint64_t)(src128) % sizeof(float4) == 0;

            constexpr int M = Layout::Shape::dim(0);
            constexpr int N = (Layout::Shape::dim(1) * sizeof(T)) / sizeof(float4);

            if (contiguous_dest && contiguous_src && aligned_size && aligned_src && aligned_stride && N) {
                // alias of shared tile with 128bit type
                using DestLayout = tile_layout_strided_t<tile_shape_t<M, N>>;
                tile_shared_t<float4, DestLayout, false> dest128((float4*)data.ptr);

                assert(((uint64_t)(dest128.data.ptr)) % sizeof(float4) == 0);
                assert(((uint64_t)(src128)) % sizeof(float4) == 0);

                const int stride_i = src.data.strides[0] / sizeof(float4);
                const int stride_j = 1;

                WP_PRAGMA_UNROLL
                for (int i = WP_TILE_THREAD_IDX; i < DestLayout::Size; i += WP_TILE_BLOCK_DIM) {
                    auto c = DestLayout::coord_from_linear(i);

#if WP_USE_ASYNC_PIPELINE
                    cp_async_global_to_shared_128(&dest128.data(i), &src128[stride_i * c[0] + stride_j * c[1]]);
#else
                    dest128.data(i) = src128[stride_i * c[0] + stride_j * c[1]];
#endif  // WP_USE_ASYNC_PIPELINE
                }

#if WP_USE_ASYNC_PIPELINE
                cp_async_commit_and_wait_all_128();
#endif  // WP_USE_ASYNC_PIPELINE

                initialized = true;
                WP_TILE_SYNC();
                return;
            }
        }

#endif  // defined(__CUDA_ARCH__)

        // scalar bounds checked path
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = Layout::coord_from_linear(i);
            data(i) = src.load(c);
        }

        initialized = true;
        WP_TILE_SYNC();
    }

    template <typename Global> inline CUDA_CALLABLE auto atomic_add(Global& dest)
    {
        return copy_to_register().atomic_add(dest);
    }

    template <typename Global> inline CUDA_CALLABLE auto atomic_add_grad(Global& dest)
    {
        return grad_to_register().atomic_add_grad(dest);
    }

    // overload for integral types
    inline CUDA_CALLABLE void print_value(int x) const { printf("%d", x); }

    // overload for floating point types
    template <typename ValueType> inline CUDA_CALLABLE void print_value(ValueType x) const { printf("%g", x); }

    template <int Level = 0> inline CUDA_CALLABLE void print_values(const Storage& storage, int index = 0) const
    {
        using Shape = typename Layout::Shape;

        if constexpr (Level < Shape::N) {
            if constexpr (Level == Shape::N - 1) {
                // Special handling for 1D case
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i) {
                    print_value(storage(index + i));

                    if (i < Shape::dim(Level) - 1) {
                        printf(" ");
                    }
                }
                printf("]");
            } else if constexpr (Level == Shape::N - 2) {
                // Special handling for 2D case
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i) {
                    printf("[");
                    for (int j = 0; j < Shape::dim(Level + 1); ++j) {
                        print_value(storage(index));

                        if (j < Shape::dim(Level + 1) - 1) {
                            printf(" ");
                        }

                        ++index;
                    }

                    printf("]");

                    // next row
                    if (i < Shape::dim(Level) - 1) {
                        printf("\n");

                        // indent next row
                        for (int i = 0; i <= Shape::N - 2; ++i)
                            printf(" ");
                    }
                }
                printf("]");
            } else {
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i) {
                    print_values<Level + 1>(storage, index + i * Shape::dim(Level));
                    if (i < Shape::dim(Level) - 1) {
                        printf("\n\n");

                        // indent next row
                        for (int i = 0; i <= Level; ++i)
                            printf(" ");
                    }
                }
                printf("]");
            }
        }
    }

    inline CUDA_CALLABLE void print(bool reverse = false) const
    {
        if (WP_TILE_THREAD_IDX != 0)
            return;

        if (reverse)
            print_values(grad);
        else
            print_values(data);

        printf(" = tile(shape=(");
        for (int i = 0; i < Layout::Shape::N; ++i) {
            printf("%d", Layout::Shape::dim(i));
            if (i != Layout::Shape::N - 1)
                printf(",");
        }

        printf("), storage=shared)\n");
    }
};


template <typename T, typename L> CUDA_CALLABLE void tile_register_t<T, L>::print() const
{
    // create a temporary shared tile so that
    // we can print it deterministically
#if defined(__CUDA_ARCH__)
    __shared__ T smem[L::Size];
#else
    T smem[L::Size];
#endif
    tile_shared_t<T, tile_layout_strided_t<typename L::Shape>, false> scratch(smem, nullptr);

    scratch.assign(*this);

    WP_TILE_SYNC();

    if (WP_TILE_THREAD_IDX == 0) {
        scratch.print_values(scratch.data, 0);

        printf(" = tile(shape=(");
        for (int i = 0; i < L::Shape::N; ++i) {
            printf("%d", L::Shape::dim(i));
            if (i != L::Shape::N - 1)
                printf(",");
        }

        printf("), storage=register)\n");
    }

    WP_TILE_SYNC();
}

// print entry points
template <typename T, typename L> inline CUDA_CALLABLE void print(const tile_register_t<T, L>& t) { t.print(); }

template <typename T, typename L>
inline CUDA_CALLABLE void adj_print(const tile_register_t<T, L>& t, const tile_register_t<T, L>& a)
{
    a.print();
}

template <typename T, typename L, bool Owner> inline CUDA_CALLABLE void print(const tile_shared_t<T, L, Owner>& t)
{
    t.print();
}

template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE void adj_print(const tile_shared_t<T, L, Owner>& t, const tile_shared_t<T, L, Owner>& a)
{
    a.print(true);
}

template <typename T, typename L, bool O> inline CUDA_CALLABLE int len(const tile_shared_t<T, L, O>& t)
{
    return L::Shape::dim(0);
}

template <typename T, typename L, bool O, typename AdjTile>
inline CUDA_CALLABLE void adj_len(const tile_shared_t<T, L, O>& t, const AdjTile& a, int& adj_ret)
{
}

template <typename T, typename L> inline CUDA_CALLABLE int len(const tile_register_t<T, L>& t)
{
    return L::Shape::dim(0);
}

template <typename T, typename L, typename AdjTile>
inline CUDA_CALLABLE void adj_len(const tile_register_t<T, L>& t, const AdjTile& a, int& adj_ret)
{
}

// where specialization for register/shared tiles
template <typename C, typename T, typename LRegister, typename LShared, bool Owner>
inline CUDA_CALLABLE auto
where(const C& cond, const tile_register_t<T, LRegister>& a, const tile_shared_t<T, LShared, Owner>& b)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    return (!!cond) ? a : b.copy_to_register();
}

template <typename C, typename T, typename LRegister, typename LShared, bool Owner>
inline CUDA_CALLABLE auto
where(const C& cond, const tile_shared_t<T, LShared, Owner>& a, const tile_register_t<T, LRegister>& b)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    return (!!cond) ? a.copy_to_register() : b;
}

template <typename C, typename T, typename L, bool Owner>
inline CUDA_CALLABLE auto where(const C& cond, const tile_shared_t<T, L, Owner>& a, const tile_shared_t<T, L, Owner>& b)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    return (!!cond) ? tile_shared_t<T, L, false>(a.data.ptr, a.grad.ptr)
                    : tile_shared_t<T, L, false>(b.data.ptr, b.grad.ptr);
}

template <typename C, typename T, typename L, bool LOwner, bool ROwner>
inline CUDA_CALLABLE auto
where(const C& cond, const tile_shared_t<T, L, LOwner>& a, const tile_shared_t<T, L, ROwner>& b)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    return (!!cond) ? tile_shared_t<T, L, false>(a.data.ptr, a.grad.ptr)
                    : tile_shared_t<T, L, false>(b.data.ptr, b.grad.ptr);
}

// adj_where same as in builtin.h

// copy specialization for shared tiles, the lvalue this gets assigned to is owning, thus, this invokes the copy assign
// path
template <typename T, typename L, bool Owner> inline CUDA_CALLABLE auto copy(const tile_shared_t<T, L, Owner>& t)
{
    return tile_shared_t<T, L, false>(t.data.ptr, t.grad.ptr);
}

template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE void adj_copy(
    const tile_shared_t<T, L, Owner>& src, tile_shared_t<T, L, Owner>& adj_src, tile_shared_t<T, L, Owner>& adj_dest
)
{
    adj_src += adj_dest;
    adj_dest.grad_zero();
}

// helpers to allocate shared tiles
template <typename T, typename Shape, typename Strides, bool RequiresGrad> inline CUDA_CALLABLE auto tile_alloc_empty()
{
    constexpr int size = Shape::size();
    T* data = (T*)tile_shared_storage_t::alloc(size * sizeof(T));
    T* grad = nullptr;

#if FP_CHECK

    // initialize tile to quiet nan
    uint32_t qnanbits = 0x7FC00000;
    float qnan = *(float*)(&qnanbits);

    for (int i = WP_TILE_THREAD_IDX; i < size; i += WP_TILE_BLOCK_DIM)
        data[i] = T(qnan);

    WP_TILE_SYNC();

#endif  // FP_CHECK


    if (RequiresGrad) {
        grad = (T*)tile_shared_storage_t::alloc(size * sizeof(T));

        for (int i = WP_TILE_THREAD_IDX; i < size; i += WP_TILE_BLOCK_DIM)
            grad[i] = T {};

        WP_TILE_SYNC();
    }

    return tile_shared_t<T, tile_layout_strided_t<Shape, Strides>>(data, grad);
}


//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

// construct a tile from a local SIMT value (one per-thread)
template <typename T> inline CUDA_CALLABLE auto tile(const T& x)
{
    tile_register_t<T, tile_layout_register_t<tile_shape_t<WP_TILE_BLOCK_DIM>>> result;

    using Layout = typename decltype(result)::Layout;
    static_assert(Layout::NumRegs == 1, "Expected Layout::NumRegs == 1");

    result.data[0] = x;
    return result;
}

// overload for constructing a tile from a per-thread vector
template <typename T, unsigned Length> inline CUDA_CALLABLE auto tile(const wp::vec_t<Length, T>& x)
{
    tile_register_t<T, tile_layout_register_t<tile_shape_t<Length, WP_TILE_BLOCK_DIM>>> result;

    using Layout = typename decltype(result)::Layout;
    static_assert(Layout::NumRegs == Length, "Expected Layout::NumRegs == Length");

    for (unsigned i = 0; i < Length; ++i)
        result.data[i] = x[i];

    return result;
}

// overload for constructing a tile from a per-thread matrix
template <unsigned Rows, unsigned Cols, typename T> inline CUDA_CALLABLE auto tile(const wp::mat_t<Rows, Cols, T>& x)
{
    tile_register_t<T, tile_layout_register_t<tile_shape_t<Rows, Cols, WP_TILE_BLOCK_DIM>>> result;

    using Layout = typename decltype(result)::Layout;
    static_assert(Layout::NumRegs == Rows * Cols, "Expected Layout::NumRegs == Rows*Cols");

    for (unsigned i = 0; i < Rows; ++i)
        for (unsigned j = 0; j < Cols; ++j)
            result.data[i * Cols + j] = x.data[i][j];

    return result;
}

// it is sufficient to use a single adjoint for all tile overload funcs
// it is also necessary, because we don't provide a dispatch_func for adjoint calls
// so the compiler will default to choosing based on argument types
template <typename T, typename AdjTile> inline CUDA_CALLABLE void adj_tile(const T& x, T& adj_x, AdjTile& adj_ret)
{
    static_assert(
        AdjTile::Layout::Shape::dim(AdjTile::Layout::Shape::N - 1) == WP_TILE_BLOCK_DIM,
        "Expected AdjTile::Layout::Shape::dim(AdjTile::Layout::Shape::N - 1) == WP_TILE_BLOCK_DIM"
    );

    auto adj_reg = adj_ret.copy_to_register();

    if constexpr (AdjTile::Layout::Shape::N == 1) {
        adj_x += adj_reg.data[0];
    } else if constexpr (AdjTile::Layout::Shape::N == 2) {
        for (unsigned i = 0; i < AdjTile::Layout::Shape::dim(0); ++i)
            adj_x[i] += adj_reg.data[i];
    } else if constexpr (AdjTile::Layout::Shape::N == 3) {
        for (unsigned i = 0; i < AdjTile::Layout::Shape::dim(0); ++i)
            for (unsigned j = 0; j < AdjTile::Layout::Shape::dim(1); ++j)
                adj_x.data[i][j] += adj_reg.data[i * AdjTile::Layout::Shape::dim(1) + j];
    }
}


template <typename Tile> inline CUDA_CALLABLE auto untile(Tile& tile)
{
    // code-gen should have set the tile to
    // have exactly the block dimension so
    // there is exactly one value per-thread
    auto reg = tile.copy_to_register();

    constexpr int N = Tile::Layout::Shape::N;

    // scalar case
    if constexpr (N == 1) {
        return reg.data[0];
    }

    // vector case
    if constexpr (N == 2) {
        constexpr int Length = Tile::Layout::Shape::dim(0);
        wp::vec_t<Length, typename Tile::Type> v;
        for (int i = 0; i < Length; ++i)
            v[i] = reg.data[i];

        return v;
    }

    // matrix case
    if constexpr (N == 3) {
        constexpr int Rows = Tile::Layout::Shape::dim(0);
        constexpr int Cols = Tile::Layout::Shape::dim(1);
        wp::mat_t<Rows, Cols, typename Tile::Type> m;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j)
                m.data[i][j] = reg.data[i * Cols + j];

        return m;
    }
}

template <typename Tile, typename Value>
inline CUDA_CALLABLE void adj_untile(Tile& tile, Tile& adj_tile, Value& adj_ret)
{
    auto adj = adj_tile.copy_to_register();

    constexpr int N = Tile::Layout::Shape::N;

    // scalar case
    if constexpr (N == 1) {
        adj.data[0] += adj_ret;
    }

    // vector case
    if constexpr (N == 2) {
        constexpr int Length = Tile::Layout::Shape::dim(0);
        for (int i = 0; i < Length; ++i)
            adj.data[i] += adj_ret[i];
    }

    // matrix case
    if constexpr (N == 3) {
        constexpr int Rows = Tile::Layout::Shape::dim(0);
        constexpr int Cols = Tile::Layout::Shape::dim(1);
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j)
                adj.data[i * Cols + j] += adj_ret.data[i][j];
    }

    adj_tile.assign(adj);
}

// zero initialized tile
template <typename T, unsigned... Shape> inline CUDA_CALLABLE auto tile_zeros()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T {};
}

// one-initialized tile
template <typename T, unsigned... Shape> inline CUDA_CALLABLE auto tile_ones()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T(1);
}

// value-initialized tile
template <typename T, unsigned... Shape> inline CUDA_CALLABLE auto tile_full(T x)
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return x;
}

// tile initialized with random integers
template <unsigned... Shape> inline CUDA_CALLABLE auto tile_randi(uint32 rng)
{
    auto out = tile_register_t<int, tile_layout_register_t<tile_shape_t<Shape...>>>();

    using Layout = typename decltype(out)::Layout;

    uint32 rng_tid = rand_pcg(uint32(WP_TILE_THREAD_IDX) + rng);

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = randi(rng_tid);
    }

    return out;
}

// tile initialized with random integers in range [min, max)
template <unsigned... Shape> inline CUDA_CALLABLE auto tile_randi(uint32 rng, int min, int max)
{
    auto out = tile_register_t<int, tile_layout_register_t<tile_shape_t<Shape...>>>();

    using Layout = typename decltype(out)::Layout;

    uint32 rng_tid = rand_pcg(uint32(WP_TILE_THREAD_IDX) + rng);

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = randi(rng_tid, min, max);
    }

    return out;
}

// tile initialized with random floats in range [0, 1)
template <unsigned... Shape> inline CUDA_CALLABLE auto tile_randf(uint32 rng)
{
    auto out = tile_register_t<float32, tile_layout_register_t<tile_shape_t<Shape...>>>();

    using Layout = typename decltype(out)::Layout;

    uint32 rng_tid = rand_pcg(uint32(WP_TILE_THREAD_IDX) + rng);

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = randf(rng_tid);
    }

    return out;
}

// tile initialized with random floats in range [min, max)
template <unsigned... Shape> inline CUDA_CALLABLE auto tile_randf(uint32 rng, float min, float max)
{
    auto out = tile_register_t<float32, tile_layout_register_t<tile_shape_t<Shape...>>>();

    using Layout = typename decltype(out)::Layout;

    uint32 rng_tid = rand_pcg(uint32(WP_TILE_THREAD_IDX) + rng);

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = randf(rng_tid, min, max);
    }

    return out;
}

// tile with evenly spaced values
template <typename T, int Len> inline CUDA_CALLABLE auto tile_arange(T start, T stop, T step)
{
    auto out = tile_register<T, Len>();

    using Layout = typename decltype(out)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = start + linear * step;
    }

    return out;
}

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_arange(T start, T stop, T step, T& adj_start, T& adj_stop, T& adj_step, AdjTile& adj_ret)
{
}

// entry point for load operations, these just return a reference to a global memory array + coordinate
template <typename T, bool BoundsCheck, unsigned... Shape, typename... Offset>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, Offset... offset)
{
    return tile_global_t<T, tile_shape_t<Shape...>, BoundsCheck>(src, tile_coord(offset...));
}

// used for indexed loads and stores
template <typename T, typename IndicesTile, typename Coord>
inline CUDA_CALLABLE bool
compute_index(array_t<T>& src, IndicesTile& indices, int axis, Coord offset, Coord c, int& out)
{
    int index = 0;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Coord::size(); ++i) {
        if (i == axis) {
            // global = offset_coord + index_mapped_coord
            int index_along_axis = offset[i] + indices.data(c[i]);

            // handle out of bounds case
            if (index_along_axis >= src.shape[i])
                return false;
            else
                index += src.strides[i] * index_along_axis;
        } else {
            // global = offset_coord + coord
            int g = offset[i] + c[i];

            // handle out of bounds case
            if (g >= src.shape[i])
                return false;
            else
                index += src.strides[i] * g;
        }
    }

    // array strides are in bytes so we convert to elements
    out = index / sizeof(T);
    return true;
}


template <unsigned... Shape, typename T, typename IndicesTile, typename... Offset>
inline CUDA_CALLABLE auto tile_load_indexed(array_t<T>& src, IndicesTile& indices, int axis, Offset... offset)
{
    auto out = tile_register_t<T, tile_layout_register_t<tile_shape_t<Shape...>>>();
    auto offset_coord = tile_coord(offset...);

    out.apply([&](int reg, auto c) {
        int i;
        if (compute_index(src, indices, axis, offset_coord, c, i))
            out.data[reg] = src.data[i];
        else
            out.data[reg] = T {};
    });

    return out;
}

// // entry point for tile store operations
// template <typename... Indices, typename T, typename Tile>
// inline CUDA_CALLABLE void tile_store(array_t<T>& dest, Tile& src, Indices... x)
// {
//     src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x)));
// }

// explicit block-level synchronization barrier for tile operations
inline CUDA_CALLABLE void tile_sync()
{
    WP_TILE_SYNC();
}

inline CUDA_CALLABLE void adj_tile_sync()
{
    // no-op for backward pass
}

// entry point for tile store operations
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, Tile& src)
{
    src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck>(dest, tile_coord(x)));
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src)
{
    src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck>(dest, tile_coord(x, y)));
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, int z, Tile& src)
{
    src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck>(dest, tile_coord(x, y, z)));
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, int z, int w, Tile& src)
{
    src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck>(dest, tile_coord(x, y, z, w)));
}

template <typename T, int M, typename Tile, typename Coord>
inline CUDA_CALLABLE void tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    Coord offset,
    Tile& src
)
{
    auto src_reg = src.copy_to_register();

    src_reg.apply([&](int reg, auto c) {
        int i;
        if (compute_index(dest, indices, axis, offset, c, i))
            dest.data[i] = src_reg.data[reg];
    });
}

// entry point for tile index store operations
template <typename T, int M, typename Tile>
inline CUDA_CALLABLE void tile_store_indexed(
    array_t<T>& dest, tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices, int axis, int x, Tile& src
)
{
    tile_store_indexed(dest, indices, axis, tile_coord(x), src);
}
template <typename T, int M, typename Tile>
inline CUDA_CALLABLE void tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    Tile& src
)
{
    tile_store_indexed(dest, indices, axis, tile_coord(x, y), src);
}
template <typename T, int M, typename Tile>
inline CUDA_CALLABLE void tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    Tile& src
)
{
    tile_store_indexed(dest, indices, axis, tile_coord(x, y, z), src);
}
template <typename T, int M, typename Tile>
inline CUDA_CALLABLE void tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    int w,
    Tile& src
)
{
    tile_store_indexed(dest, indices, axis, tile_coord(x, y, z, w), src);
}


// compiler struggles with these if they are one line
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, Tile& src)
{
    tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck> global(dest, tile_coord(x));
    return src.atomic_add(global);
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, Tile& src)
{
    tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck> global(dest, tile_coord(x, y));
    return src.atomic_add(global);
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, int z, Tile& src)
{
    tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck> global(dest, tile_coord(x, y, z));
    return src.atomic_add(global);
}
template <typename T, bool BoundsCheck, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, int z, int w, Tile& src)
{
    tile_global_t<T, typename Tile::Layout::Shape, BoundsCheck> global(dest, tile_coord(x, y, z, w));
    return src.atomic_add(global);
}

template <typename T, int M, typename Tile, typename Coord>
inline CUDA_CALLABLE auto tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    Coord offset,
    Tile& src
)
{
    auto src_reg = src.copy_to_register();
    auto ret_reg = tile_register_like<Tile>();

    src_reg.apply([&](int reg, auto c) {
        int i;
        if (compute_index(dest, indices, axis, offset, c, i))
            ret_reg.data[reg] = wp::atomic_add(&dest.data[i], src_reg.data[reg]);
        else
            ret_reg.data[reg] = T {};
    });

    return ret_reg;
}

// entry point for tile index atomic add operations
template <typename T, int M, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add_indexed(
    array_t<T>& dest, tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices, int axis, int x, Tile& src
)
{
    return tile_atomic_add_indexed(dest, indices, axis, tile_coord(x), src);
}

template <typename T, int M, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    Tile& src
)
{
    return tile_atomic_add_indexed(dest, indices, axis, tile_coord(x, y), src);
}

template <typename T, int M, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    Tile& src
)
{
    return tile_atomic_add_indexed(dest, indices, axis, tile_coord(x, y, z), src);
}

template <typename T, int M, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    int w,
    Tile& src
)
{
    return tile_atomic_add_indexed(dest, indices, axis, tile_coord(x, y, z, w), src);
}


//-------------------------------------
// Adjoints

template <typename T, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, Coord c, array_t<T>& adj_src, Coord adj_c, AdjTile& adj_ret)
{
    tile_global_t<T, typename AdjTile::Layout::Shape> dest(src, c);

    // we allow users to override grad of src
    if (adj_src.data)
        dest.data.grad = adj_src.data;

    adj_ret.atomic_add_grad(dest);
}

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, array_t<T>& adj_src, int adj_x, AdjTile& adj_ret)
{
    adj_tile_load(src, tile_coord(x), adj_src, tile_coord(0), adj_ret);
}
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_load(array_t<T>& src, int x, int y, array_t<T>& adj_src, int adj_x, int adj_y, AdjTile& adj_ret)
{
    adj_tile_load(src, tile_coord(x, y), adj_src, tile_coord(0, 0), adj_ret);
}
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(
    array_t<T>& src, int x, int y, int z, array_t<T>& adj_src, int adj_x, int adj_y, int adj_z, AdjTile& adj_ret
)
{
    adj_tile_load(src, tile_coord(x, y, z), adj_src, tile_coord(0, 0, 0), adj_ret);
}
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(
    array_t<T>& src,
    int x,
    int y,
    int z,
    int w,
    array_t<T>& adj_src,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_ret
)
{
    adj_tile_load(src, tile_coord(x, y, z, w), adj_src, tile_coord(0, 0, 0, 0), adj_ret);
}

template <typename T, typename IndicesTile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_load_indexed(
    array_t<T>& src,
    IndicesTile& indices,
    int axis,
    Coord offset,
    array_t<T>& adj_src,
    IndicesTile& adj_indices,
    int adj_axis,
    Coord adj_offset,
    AdjTile& adj_ret
)
{
    // we allow users to override grad of src
    if (adj_src.data)
        src.grad = adj_src.data;

    auto adj_ret_reg = adj_ret.grad_to_register();

    adj_ret_reg.apply([&](int reg, auto c) {
        int i;
        if (compute_index(src, indices, axis, offset, c, i))
            wp::atomic_add(&src.grad[i], adj_ret_reg.data[reg]);
    });
}

template <typename T, typename IndicesTile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load_indexed(
    array_t<T>& src,
    IndicesTile& indices,
    int axis,
    int x,
    array_t<T>& adj_src,
    IndicesTile& adj_indices,
    int adj_axis,
    int adj_x,
    AdjTile& adj_ret
)
{
    adj_tile_load_indexed(src, indices, axis, tile_coord(x), adj_src, adj_indices, adj_axis, tile_coord(0), adj_ret);
}
template <typename T, typename IndicesTile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load_indexed(
    array_t<T>& src,
    IndicesTile& indices,
    int axis,
    int x,
    int y,
    array_t<T>& adj_src,
    IndicesTile& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    AdjTile& adj_ret
)
{
    adj_tile_load_indexed(
        src, indices, axis, tile_coord(x, y), adj_src, adj_indices, adj_axis, tile_coord(0, 0), adj_ret
    );
}
template <typename T, typename IndicesTile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load_indexed(
    array_t<T>& src,
    IndicesTile& indices,
    int axis,
    int x,
    int y,
    int z,
    array_t<T>& adj_src,
    IndicesTile& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    AdjTile& adj_ret
)
{
    adj_tile_load_indexed(
        src, indices, axis, tile_coord(x, y, z), adj_src, adj_indices, adj_axis, tile_coord(0, 0, 0), adj_ret
    );
}
template <typename T, typename IndicesTile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load_indexed(
    array_t<T>& src,
    IndicesTile& indices,
    int axis,
    int x,
    int y,
    int z,
    int w,
    array_t<T>& adj_src,
    IndicesTile& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_ret
)
{
    adj_tile_load_indexed(
        src, indices, axis, tile_coord(x, y, z, w), adj_src, adj_indices, adj_axis, tile_coord(0, 0, 0, 0), adj_ret
    );
}

template <typename T, typename Tile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void
adj_tile_store(array_t<T>& dest, Coord c, Tile& t, array_t<T>& adj_dest, Coord adj_c, AdjTile& adj_t)
{
    tile_global_t<T, typename AdjTile::Layout::Shape> src(dest, c);

    // we allow users to override grad of src
    if (adj_dest.data)
        src.data.grad = adj_dest.data;

    if (src.data.grad == nullptr)
        return;

    adj_t.grad_add(src);
    adj_t.grad_zero_global(src);
}

template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_store(array_t<T>& dest, int x, Tile& t, array_t<T>& adj_dest, int adj_x, AdjTile& adj_t)
{
    adj_tile_store(dest, tile_coord(x), t, adj_dest, tile_coord(0), adj_t);
}
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t)
{
    adj_tile_store(dest, tile_coord(x, y), t, adj_dest, tile_coord(0, 0), adj_t);
}
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(
    array_t<T>& dest,
    int x,
    int y,
    int z,
    Tile& t,
    array_t<T>& adj_dest,
    int adj_x,
    int adj_y,
    int adj_z,
    AdjTile& adj_t
)
{
    adj_tile_store(dest, tile_coord(x, y, z), t, adj_dest, tile_coord(0, 0, 0), adj_t);
}
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(
    array_t<T>& dest,
    int x,
    int y,
    int z,
    int w,
    Tile& t,
    array_t<T>& adj_dest,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_t
)
{
    adj_tile_store(dest, tile_coord(x, y, z, w), t, adj_dest, tile_coord(0, 0, 0, 0), adj_t);
}

template <typename T, int M, typename Tile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    Coord offset,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    Coord adj_offset,
    AdjTile& adj_t
)
{
    // we allow users to override grad of src
    if (adj_dest.data)
        dest.grad = adj_dest.data;

    if (dest.grad == nullptr)
        return;

    auto adj_t_reg = tile_register_like<Tile>();

    adj_t_reg.apply([&](int reg, auto c) {
        int i;
        if (compute_index(dest, indices, axis, offset, c, i)) {
            adj_t_reg.data[reg] += dest.grad[i];
            dest.grad[i] = T();
        }
    });

    // write adjoints back
    adj_t.grad_add(adj_t_reg);
}

template <typename T, int M, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    AdjTile& adj_t
)
{
    adj_tile_store_indexed(
        dest, indices, axis, tile_coord(x), t, adj_dest, adj_indices, adj_axis, tile_coord(0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    AdjTile& adj_t
)
{
    adj_tile_store_indexed(
        dest, indices, axis, tile_coord(x, y), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    AdjTile& adj_t
)
{
    adj_tile_store_indexed(
        dest, indices, axis, tile_coord(x, y, z), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0, 0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    int w,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_t
)
{
    adj_tile_store_indexed(
        dest, indices, axis, tile_coord(x, y, z, w), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0, 0, 0), adj_t
    );
}

template <typename T, typename Tile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void
adj_tile_atomic_add(array_t<T>& dest, Coord c, Tile& t, array_t<T>& adj_dest, Coord adj_c, AdjTile& adj_t)
{
    tile_global_t<T, typename AdjTile::Layout::Shape> src(dest, c);

    // we allow users to override grad of src
    if (adj_dest.data)
        src.data.grad = adj_dest.data;

    if (src.data.grad == nullptr)
        return;

    adj_t.grad_add(src);
}

template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void
adj_tile_atomic_add(array_t<T>& dest, int x, Tile& t, array_t<T>& adj_dest, int adj_x, AdjTile& adj_t, AdjRet& adj_ret)
{
    adj_tile_atomic_add(dest, tile_coord(x), t, adj_dest, tile_coord(adj_x), adj_t);
}
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(
    array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t, AdjRet& adj_ret
)
{
    adj_tile_atomic_add(dest, tile_coord(x, y), t, adj_dest, tile_coord(adj_x, adj_y), adj_t);
}
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(
    array_t<T>& dest,
    int x,
    int y,
    int z,
    Tile& t,
    array_t<T>& adj_dest,
    int adj_x,
    int adj_y,
    int adj_z,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add(dest, tile_coord(x, y, z), t, adj_dest, tile_coord(adj_x, adj_y, adj_z), adj_t);
}
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(
    array_t<T>& dest,
    int x,
    int y,
    int z,
    int w,
    Tile& t,
    array_t<T>& adj_dest,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add(dest, tile_coord(x, y, z, w), t, adj_dest, tile_coord(adj_x, adj_y, adj_z, adj_w), adj_t);
}

template <typename T, int M, typename Tile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    Coord offset,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    Coord adj_offset,
    AdjTile& adj_t
)
{
    // we allow users to override grad of src
    if (adj_dest.data)
        dest.grad = adj_dest.data;

    if (dest.grad == nullptr)
        return;

    auto adj_t_reg = tile_register_like<Tile>();

    adj_t_reg.apply([&](int reg, auto c) {
        int i;
        if (compute_index(dest, indices, axis, offset, c, i))
            adj_t_reg.data[reg] += dest.grad[i];
    });

    // write adjoints back
    adj_t.grad_add(adj_t_reg);
}

template <typename T, int M, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add_indexed(
        dest, indices, axis, tile_coord(x), t, adj_dest, adj_indices, adj_axis, tile_coord(0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add_indexed(
        dest, indices, axis, tile_coord(x, y), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add_indexed(
        dest, indices, axis, tile_coord(x, y, z), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0, 0), adj_t
    );
}
template <typename T, int M, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add_indexed(
    array_t<T>& dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& indices,
    int axis,
    int x,
    int y,
    int z,
    int w,
    Tile& t,
    array_t<T>& adj_dest,
    tile_shared_t<int, tile_layout_strided_t<tile_shape_t<M>>>& adj_indices,
    int adj_axis,
    int adj_x,
    int adj_y,
    int adj_z,
    int adj_w,
    AdjTile& adj_t,
    AdjRet& adj_ret
)
{
    adj_tile_atomic_add_indexed(
        dest, indices, axis, tile_coord(x, y, z, w), t, adj_dest, adj_indices, adj_axis, tile_coord(0, 0, 0, 0), adj_t
    );
}

// unary map
template <typename Tile, typename Fwd, typename ReturnTile>
inline CUDA_CALLABLE auto tile_map(Fwd op, Tile& a, ReturnTile& r)
{
    // verify shapes and sizes are compatible
    using ShapeIn = typename Tile::Layout::Shape;
    using ShapeOut = typename ReturnTile::Layout::Shape;

    static_assert(ShapeIn::N == ShapeOut::N, "Number of tile dimensions must match for unary map");
    static_assert(ShapeIn::size() == ShapeOut::size(), "Tile sizes must match for unary map");

    auto out = tile_register_like<ReturnTile>();
    auto a_reg = a.copy_to_register();

    using Layout = typename decltype(out)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        out.data[i] = op(a_reg.data[i]);
    }

    return out;
}


template <typename Tile, typename AdjTile, typename Fwd, typename Adj>
inline CUDA_CALLABLE void adj_tile_map(Fwd op, Tile& a, Adj adj_op, Tile& adj_a, AdjTile& adj_ret)
{
    auto a_reg = a.copy_to_register();
    auto adj_a_reg = tile_register_like<Tile>();
    auto adj_ret_reg = adj_ret.grad_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        adj_op(a_reg.data[i], adj_a_reg.data[i], adj_ret_reg.data[i]);
    }

    // write adjoints back
    adj_a.grad_add(adj_a_reg);
}

// binary map
template <typename TileA, typename TileB, typename Fwd, typename ReturnTile>
inline CUDA_CALLABLE auto tile_map(Fwd op, TileA& a, TileB& b, ReturnTile& r)
{
    // verify shapes and sizes are compatible
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeOut = typename ReturnTile::Layout::Shape;

    static_assert(ShapeA::N == ShapeOut::N, "Number of tile dimensions must match for binary map");
    static_assert(ShapeB::N == ShapeOut::N, "Number of tile dimensions must match for binary map");

    static_assert(ShapeA::size() == ShapeOut::size(), "Tile sizes must match for binary map");
    static_assert(ShapeB::size() == ShapeOut::size(), "Tile sizes must match for binary map");

    auto out = tile_register_like<ReturnTile>();

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(out)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        out.data[i] = op(a_reg.data[i], b_reg.data[i]);
    }

    return out;
}

template <typename TileA, typename TileB, typename Fwd, typename Adj, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_map(Fwd op, TileA& a, TileB& b, Adj adj_op, TileA& adj_a, TileB& adj_b, AdjTile& adj_ret)
{
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    // allocate storage for adjoints
    auto adj_a_reg = tile_register_like<TileA>();
    auto adj_b_reg = tile_register_like<TileB>();

    auto adj_ret_reg = adj_ret.grad_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        adj_op(a_reg.data[i], b_reg.data[i], adj_a_reg.data[i], adj_b_reg.data[i], adj_ret_reg.data[i]);
    }

    adj_a.grad_add(adj_a_reg);
    adj_b.grad_add(adj_b_reg);
}

// ============================================================================
// Variadic tile_map (N = 3 to 8 tiles) - for user-defined functions only
// ============================================================================

// N = 3
template <typename ReturnType, typename Fwd, typename T1, typename T2, typename T3>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i] = op(r1.data[i], r2.data[i], r3.data[i]);
    return out;
}

// N = 4
template <typename ReturnType, typename Fwd, typename T1, typename T2, typename T3, typename T4>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3, T4& t4)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i] = op(r1.data[i], r2.data[i], r3.data[i], r4.data[i]);
    return out;
}

// N = 5
template <typename ReturnType, typename Fwd, typename T1, typename T2, typename T3, typename T4, typename T5>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3, T4& t4, T5& t5)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i] = op(r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i]);
    return out;
}

// N = 6
template <
    typename ReturnType,
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i] = op(r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i]);
    return out;
}

// N = 7
template <
    typename ReturnType,
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    auto r7 = t7.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i] = op(r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i], r7.data[i]);
    return out;
}

// N = 8
template <
    typename ReturnType,
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename T8>
inline CUDA_CALLABLE auto tile_map(Fwd op, T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7, T8& t8)
{
    using Shape = typename T1::Layout::Shape;
    auto out = tile_register_t<ReturnType, tile_layout_register_t<Shape>>();
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    auto r7 = t7.copy_to_register();
    auto r8 = t8.copy_to_register();
    using Layout = typename decltype(out)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        out.data[i]
            = op(r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i], r7.data[i], r8.data[i]);
    return out;
}

// N = 3
template <typename Fwd, typename T1, typename T2, typename T3, typename Adj, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_map(Fwd op, T1& t1, T2& t2, T3& t3, Adj adj_op, T1& adj_t1, T2& adj_t2, T3& adj_t3, AdjTile& adj_ret)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(r1.data[i], r2.data[i], r3.data[i], adj_r1.data[i], adj_r2.data[i], adj_r3.data[i], adj_ret_reg.data[i]);
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
}

// N = 4
template <typename Fwd, typename T1, typename T2, typename T3, typename T4, typename Adj, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(
    Fwd op, T1& t1, T2& t2, T3& t3, T4& t4, Adj adj_op, T1& adj_t1, T2& adj_t2, T3& adj_t3, T4& adj_t4, AdjTile& adj_ret
)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_r4 = tile_register_like<T4>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(
            r1.data[i], r2.data[i], r3.data[i], r4.data[i], adj_r1.data[i], adj_r2.data[i], adj_r3.data[i],
            adj_r4.data[i], adj_ret_reg.data[i]
        );
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
    adj_t4.grad_add(adj_r4);
}

// N = 5
template <typename Fwd, typename T1, typename T2, typename T3, typename T4, typename T5, typename Adj, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(
    Fwd op,
    T1& t1,
    T2& t2,
    T3& t3,
    T4& t4,
    T5& t5,
    Adj adj_op,
    T1& adj_t1,
    T2& adj_t2,
    T3& adj_t3,
    T4& adj_t4,
    T5& adj_t5,
    AdjTile& adj_ret
)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_r4 = tile_register_like<T4>();
    auto adj_r5 = tile_register_like<T5>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(
            r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], adj_r1.data[i], adj_r2.data[i], adj_r3.data[i],
            adj_r4.data[i], adj_r5.data[i], adj_ret_reg.data[i]
        );
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
    adj_t4.grad_add(adj_r4);
    adj_t5.grad_add(adj_r5);
}

// N = 6
template <
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename Adj,
    typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(
    Fwd op,
    T1& t1,
    T2& t2,
    T3& t3,
    T4& t4,
    T5& t5,
    T6& t6,
    Adj adj_op,
    T1& adj_t1,
    T2& adj_t2,
    T3& adj_t3,
    T4& adj_t4,
    T5& adj_t5,
    T6& adj_t6,
    AdjTile& adj_ret
)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_r4 = tile_register_like<T4>();
    auto adj_r5 = tile_register_like<T5>();
    auto adj_r6 = tile_register_like<T6>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(
            r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i], adj_r1.data[i], adj_r2.data[i],
            adj_r3.data[i], adj_r4.data[i], adj_r5.data[i], adj_r6.data[i], adj_ret_reg.data[i]
        );
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
    adj_t4.grad_add(adj_r4);
    adj_t5.grad_add(adj_r5);
    adj_t6.grad_add(adj_r6);
}

// N = 7
template <
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename Adj,
    typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(
    Fwd op,
    T1& t1,
    T2& t2,
    T3& t3,
    T4& t4,
    T5& t5,
    T6& t6,
    T7& t7,
    Adj adj_op,
    T1& adj_t1,
    T2& adj_t2,
    T3& adj_t3,
    T4& adj_t4,
    T5& adj_t5,
    T6& adj_t6,
    T7& adj_t7,
    AdjTile& adj_ret
)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    auto r7 = t7.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_r4 = tile_register_like<T4>();
    auto adj_r5 = tile_register_like<T5>();
    auto adj_r6 = tile_register_like<T6>();
    auto adj_r7 = tile_register_like<T7>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(
            r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i], r7.data[i], adj_r1.data[i],
            adj_r2.data[i], adj_r3.data[i], adj_r4.data[i], adj_r5.data[i], adj_r6.data[i], adj_r7.data[i],
            adj_ret_reg.data[i]
        );
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
    adj_t4.grad_add(adj_r4);
    adj_t5.grad_add(adj_r5);
    adj_t6.grad_add(adj_r6);
    adj_t7.grad_add(adj_r7);
}

// N = 8
template <
    typename Fwd,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename T8,
    typename Adj,
    typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(
    Fwd op,
    T1& t1,
    T2& t2,
    T3& t3,
    T4& t4,
    T5& t5,
    T6& t6,
    T7& t7,
    T8& t8,
    Adj adj_op,
    T1& adj_t1,
    T2& adj_t2,
    T3& adj_t3,
    T4& adj_t4,
    T5& adj_t5,
    T6& adj_t6,
    T7& adj_t7,
    T8& adj_t8,
    AdjTile& adj_ret
)
{
    auto r1 = t1.copy_to_register();
    auto r2 = t2.copy_to_register();
    auto r3 = t3.copy_to_register();
    auto r4 = t4.copy_to_register();
    auto r5 = t5.copy_to_register();
    auto r6 = t6.copy_to_register();
    auto r7 = t7.copy_to_register();
    auto r8 = t8.copy_to_register();
    auto adj_r1 = tile_register_like<T1>();
    auto adj_r2 = tile_register_like<T2>();
    auto adj_r3 = tile_register_like<T3>();
    auto adj_r4 = tile_register_like<T4>();
    auto adj_r5 = tile_register_like<T5>();
    auto adj_r6 = tile_register_like<T6>();
    auto adj_r7 = tile_register_like<T7>();
    auto adj_r8 = tile_register_like<T8>();
    auto adj_ret_reg = adj_ret.grad_to_register();
    using Layout = typename decltype(r1)::Layout;
    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i)
        adj_op(
            r1.data[i], r2.data[i], r3.data[i], r4.data[i], r5.data[i], r6.data[i], r7.data[i], r8.data[i],
            adj_r1.data[i], adj_r2.data[i], adj_r3.data[i], adj_r4.data[i], adj_r5.data[i], adj_r6.data[i],
            adj_r7.data[i], adj_r8.data[i], adj_ret_reg.data[i]
        );
    adj_t1.grad_add(adj_r1);
    adj_t2.grad_add(adj_r2);
    adj_t3.grad_add(adj_r3);
    adj_t4.grad_add(adj_r4);
    adj_t5.grad_add(adj_r5);
    adj_t6.grad_add(adj_r6);
    adj_t7.grad_add(adj_r7);
    adj_t8.grad_add(adj_r8);
}

// We wrap the operator in a lambda so that we don't have to do overload resolution for things like e.g.: wp.sin()
// this is important because many of the builtin operators don't follow particular conventions on references for
// the `adj_ret` parameter, which means it's not possible to figure out the overload we need using simple casting
// The r argument is a dummy return tile argument, because we can't template on the return tile type in a macro
// definition. So if we want users to be able to define functions that return a tile type that is different from the
// input type, we must pass an extra dummy return tile argument that is used define the return type of tile_map.

#define tile_unary_map(op, a, r) tile_map([](auto x) { return op(x);}, a, r)
#define adj_tile_unary_map(op, a, r, adj_op, adj_a, adj_r, adj_ret) adj_tile_map([](auto x) { return op(x);}, a, [](auto x, auto& adj_x, auto adj_ret) { adj_op(x, adj_x, adj_ret);}, adj_a, adj_ret)

#define tile_binary_map(op, a, b, r) tile_map([](auto x, auto y) { return op(x, y);}, a, b, r)
#define adj_tile_binary_map(op, a, b, r, adj_op, adj_a, adj_b, adj_r, adj_ret) adj_tile_map([](auto x, auto y) { return op(x, y);}, a, b, [](auto x, auto y, auto& adj_x, auto& adj_y, auto adj_ret) { adj_op(x, y, adj_x, adj_y, adj_ret);}, adj_a, adj_b, adj_ret)

// -tile (unary neg)
template <typename Tile> inline CUDA_CALLABLE auto tile_neg(Tile& a) { return tile_unary_map(wp::neg, a, a); }

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_neg(Tile& a, Tile& adj_a, AdjTile& adj_ret)
{
    adj_tile_unary_map(wp::neg, a, a, wp::adj_neg, adj_a, adj_a, adj_ret);
}


// tile + tile
template <typename TileA, typename TileB> inline CUDA_CALLABLE auto tile_add(TileA& a, TileB& b)
{
    return tile_binary_map(add, a, b, a);
}

// add overloads get called in user function adjoints generated by codegen (adj_tile += adj_ret)
template <typename T, typename L>
inline CUDA_CALLABLE auto add(tile_register_t<T, L>& a, const tile_register_t<T, L>& b)
{
    return tile_add(a, b);
}

template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE auto add(tile_shared_t<T, L, Owner>& a, const tile_shared_t<T, L, Owner>& b)
{
    return tile_add(a, b);
}

template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE auto add(tile_register_t<T, L>& a, const tile_shared_t<T, L, Owner>& b)
{
    return tile_add(a, b);
}

template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE auto add(tile_shared_t<T, L, Owner>& a, const tile_register_t<T, L>& b)
{
    return tile_add(a, b);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_add(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{
    adj_tile_binary_map(add, a, b, a, adj_add, adj_a, adj_b, adj_a, adj_c);
}

// tile - tile
template <typename TileA, typename TileB> inline CUDA_CALLABLE auto tile_sub(TileA& a, TileB& b)
{
    return tile_binary_map(sub, a, b, a);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_sub(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{
    adj_tile_binary_map(sub, a, b, a, adj_sub, adj_a, adj_b, adj_a, adj_c);
}


// tile*scalar
template <typename Tile> inline CUDA_CALLABLE auto tile_mul(Tile& a, const typename Tile::Type& s)
{
    // promote scalar to a constant tile
    auto s_tile = tile_register_t<typename Tile::Type, tile_layout_register_t<typename Tile::Layout::Shape>>(s);

    return tile_binary_map(mul, a, s_tile, a);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_mul(Tile& a, const typename Tile::Type& s, Tile& adj_a, typename Tile::Type& adj_s, AdjTile& adj_c)
{
    auto s_tile = tile_register_like<Tile>();
    auto adj_s_tile = tile_register_like<Tile>();

    using Layout = typename decltype(adj_s_tile)::Layout;

    // initialize to constant
    s_tile = s;

    adj_tile_binary_map(mul, a, s_tile, a, adj_mul, adj_a, adj_s_tile, adj_a, adj_c);

    for (int i = 0; i < Layout::NumRegs; ++i) {
        adj_s += adj_s_tile.data[i];
    }
}


// scalar*tile
template <typename Tile> inline CUDA_CALLABLE auto tile_mul(const typename Tile::Type& s, Tile& a)
{
    return tile_mul(a, s);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void
adj_tile_mul(const typename Tile::Type& s, Tile& a, typename Tile::Type& adj_s, Tile& adj_a, AdjTile& adj_c)
{
    adj_tile_mul(a, s, adj_a, adj_s, adj_c);
}


// tile & tile
template <typename TileA, typename TileB> inline CUDA_CALLABLE auto tile_bit_and(TileA& a, TileB& b)
{
    return tile_binary_map(bit_and, a, b, a);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_bit_and(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{
}

// tile | tile
template <typename TileA, typename TileB> inline CUDA_CALLABLE auto tile_bit_or(TileA& a, TileB& b)
{
    return tile_binary_map(bit_or, a, b, a);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_bit_or(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{
}

// tile ^ tile
template <typename TileA, typename TileB> inline CUDA_CALLABLE auto tile_bit_xor(TileA& a, TileB& b)
{
    return tile_binary_map(bit_xor, a, b, a);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_bit_xor(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{
}


template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_add_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace addition");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace addition");

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(b_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        a_reg.data[i] += b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_add_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace addition");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace addition");

    // allocate storage for adjoints
    auto adj_a_reg = adj_a.grad_to_register();
    auto adj_b_reg = tile_register_like<TileB>();

    using Layout = typename decltype(adj_a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        adj_b_reg.data[i] += adj_a_reg.data[i];
    }

    adj_b.grad_add(adj_b_reg);
}

template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_sub_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace subtraction");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace subtraction");

    // work with register tiles for inplace operations, regardless of the storage type of the input tiles
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        a_reg.data[i] -= b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_sub_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace subtraction");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace subtraction");

    // allocate storage for adjoints
    auto adj_a_reg = adj_a.grad_to_register();
    auto adj_b_reg = tile_register_like<TileB>();

    using Layout = typename decltype(adj_a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        adj_b_reg.data[i] -= adj_a_reg.data[i];
    }

    adj_b.grad_add(adj_b_reg);
}

template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_bit_and_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace bitwise AND");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace bitwise AND");

    // work with register tiles for inplace operations, regardless of the storage type of the input tiles
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        a_reg.data[i] &= b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_bit_and_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
}

template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_bit_or_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace bitwise OR");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace bitwise OR");

    // work with register tiles for inplace operations, regardless of the storage type of the input tiles
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        a_reg.data[i] |= b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_bit_or_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
}

template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_bit_xor_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace bitwise XOR");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace bitwise XOR");

    // work with register tiles for inplace operations, regardless of the storage type of the input tiles
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        a_reg.data[i] ^= b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_bit_xor_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
}


template <typename Tile> typename Tile::Type tile_extract(Tile& t, int i) { return t.extract(tile_coord(i)); }
template <typename Tile> auto tile_extract(Tile& t, int i, int j)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        return t.extract(tile_coord(i))[j];
    } else {
        return t.extract(tile_coord(i, j));
    }
}
template <typename Tile> auto tile_extract(Tile& t, int i, int j, int k)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j))[k];
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        return t.extract(tile_coord(i)).data[j][k];
    } else {
        return t.extract(tile_coord(i, j, k));
    }
}
template <typename Tile> auto tile_extract(Tile& t, int i, int j, int k, int l)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j, k))[l];
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j)).data[k][l];
    } else {
        return t.extract(tile_coord(i, j, k, l));
    }
}
template <typename Tile> auto tile_extract(Tile& t, int i, int j, int k, int l, int m)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j, k, l))[m];
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j, k)).data[l][m];
    } else {
        static_assert(
            always_false<Tile>::value,
            "tile_extract with 5 indices requires a tile of vectors (4D tile) or matrices (3D tile)"
        );
    }
}
template <typename Tile> auto tile_extract(Tile& t, int i, int j, int k, int l, int m, int n)
{
    if constexpr (is_matrix<typename Tile::Type>::value) {
        return t.extract(tile_coord(i, j, k, l)).data[m][n];
    } else {
        static_assert(always_false<Tile>::value, "tile_extract with 6 indices requires a tile of matrices (4D tile)");
    }
}

template <typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, AdjTile& adj_t, int adj_i, typename Tile::Type adj_ret)
{
    adj_t.adj_extract(tile_coord(i), adj_ret);
}
template <typename Tile, typename AdjTile, typename AdjType>
void adj_tile_extract(Tile& t, int i, int j, AdjTile& adj_t, int adj_i, int adj_j, AdjType adj_ret)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        typename Tile::Type vector_adj {};
        vector_adj[j] = adj_ret;
        adj_t.adj_extract(tile_coord(i), vector_adj);
    } else {
        adj_t.adj_extract(tile_coord(i, j), adj_ret);
    }
}
template <typename Tile, typename AdjTile, typename AdjType>
void adj_tile_extract(Tile& t, int i, int j, int k, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, AdjType adj_ret)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        typename Tile::Type vector_adj {};
        vector_adj[k] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j), vector_adj);
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        typename Tile::Type matrix_adj {};
        matrix_adj.data[j][k] = adj_ret;
        adj_t.adj_extract(tile_coord(i), matrix_adj);
    } else {
        adj_t.adj_extract(tile_coord(i, j, k), adj_ret);
    }
}
template <typename Tile, typename AdjTile, typename AdjType>
void adj_tile_extract(
    Tile& t, int i, int j, int k, int l, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, int adj_l, AdjType adj_ret
)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        typename Tile::Type vector_adj {};
        vector_adj[l] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j, k), vector_adj);
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        typename Tile::Type matrix_adj {};
        matrix_adj.data[k][l] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j), matrix_adj);
    } else {
        adj_t.adj_extract(tile_coord(i, j, k, l), adj_ret);
    }
}
template <typename Tile, typename AdjTile, typename AdjType>
void adj_tile_extract(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    int m,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    int adj_m,
    AdjType adj_ret
)
{
    if constexpr (is_vector<typename Tile::Type>::value) {
        typename Tile::Type vector_adj {};
        vector_adj[m] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j, k, l), vector_adj);
    } else if constexpr (is_matrix<typename Tile::Type>::value) {
        typename Tile::Type matrix_adj {};
        matrix_adj.data[l][m] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j, k), matrix_adj);
    } else {
        static_assert(
            always_false<Tile>::value,
            "adj_tile_extract with 5 indices requires a tile of vectors (4D tile) or matrices (3D tile)"
        );
    }
}
template <typename Tile, typename AdjTile, typename AdjType>
void adj_tile_extract(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    int m,
    int n,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    int adj_m,
    int adj_n,
    AdjType adj_ret
)
{
    if constexpr (is_matrix<typename Tile::Type>::value) {
        typename Tile::Type matrix_adj {};
        matrix_adj.data[m][n] = adj_ret;
        adj_t.adj_extract(tile_coord(i, j, k, l), matrix_adj);
    } else {
        static_assert(
            always_false<Tile>::value, "adj_tile_extract with 6 indices requires a tile of matrices (4D tile)"
        );
    }
}


template <typename Tile> void tile_add_inplace(Tile& t, int i, typename Tile::Type value)
{
    t.add_inplace(tile_coord(i), value);
}
template <typename Tile> void tile_add_inplace(Tile& t, int i, int j, typename Tile::Type value)
{
    t.add_inplace(tile_coord(i, j), value);
}
template <typename Tile> void tile_add_inplace(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.add_inplace(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_add_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.add_inplace(tile_coord(i, j, k, l), value);
}

template <typename Tile> void tile_sub_inplace(Tile& t, int i, typename Tile::Type value)
{
    t.sub_inplace(tile_coord(i), value);
}
template <typename Tile> void tile_sub_inplace(Tile& t, int i, int j, typename Tile::Type value)
{
    t.sub_inplace(tile_coord(i, j), value);
}
template <typename Tile> void tile_sub_inplace(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.sub_inplace(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_sub_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.sub_inplace(tile_coord(i, j, k, l), value);
}

template <typename Tile> void tile_bit_and_inplace(Tile& t, int i, typename Tile::Type value)
{
    t.bit_and_inplace(tile_coord(i), value);
}
template <typename Tile> void tile_bit_and_inplace(Tile& t, int i, int j, typename Tile::Type value)
{
    t.bit_and_inplace(tile_coord(i, j), value);
}
template <typename Tile> void tile_bit_and_inplace(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.bit_and_inplace(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_bit_and_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.bit_and_inplace(tile_coord(i, j, k, l), value);
}

template <typename Tile> void tile_bit_or_inplace(Tile& t, int i, typename Tile::Type value)
{
    t.bit_or_inplace(tile_coord(i), value);
}
template <typename Tile> void tile_bit_or_inplace(Tile& t, int i, int j, typename Tile::Type value)
{
    t.bit_or_inplace(tile_coord(i, j), value);
}
template <typename Tile> void tile_bit_or_inplace(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.bit_or_inplace(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_bit_or_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.bit_or_inplace(tile_coord(i, j, k, l), value);
}

template <typename Tile> void tile_bit_xor_inplace(Tile& t, int i, typename Tile::Type value)
{
    t.bit_xor_inplace(tile_coord(i), value);
}
template <typename Tile> void tile_bit_xor_inplace(Tile& t, int i, int j, typename Tile::Type value)
{
    t.bit_xor_inplace(tile_coord(i, j), value);
}
template <typename Tile> void tile_bit_xor_inplace(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.bit_xor_inplace(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_bit_xor_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.bit_xor_inplace(tile_coord(i, j, k, l), value);
}

template <typename Tile, typename AdjTile>
void adj_tile_add_inplace(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
    adj_t.adj_add_inplace(tile_coord(i), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_add_inplace(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_add_inplace(tile_coord(i, j), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_add_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_add_inplace(tile_coord(i, j, k), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_add_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_add_inplace(tile_coord(i, j, k, l), adj_value);
}

template <typename Tile, typename AdjTile>
void adj_tile_sub_inplace(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
    adj_t.adj_sub_inplace(tile_coord(i), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_sub_inplace(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_sub_inplace(tile_coord(i, j), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_sub_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_sub_inplace(tile_coord(i, j, k), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_sub_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_sub_inplace(tile_coord(i, j, k, l), adj_value);
}

template <typename Tile, typename AdjTile>
void adj_tile_bit_and_inplace(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_and_inplace(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_and_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_and_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
}

template <typename Tile, typename AdjTile>
void adj_tile_bit_or_inplace(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_or_inplace(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_or_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_or_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
}

template <typename Tile, typename AdjTile>
void adj_tile_bit_xor_inplace(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_xor_inplace(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_xor_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
}
template <typename Tile, typename AdjTile>
void adj_tile_bit_xor_inplace(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
}

// tile_insert: direct per-element write without sync
// caller must follow with tile_sync() before any collective op that reads this tile
template <typename Tile> void tile_insert(Tile& t, int i, typename Tile::Type value)
{
    t.insert(tile_coord(i), value);
}
template <typename Tile> void tile_insert(Tile& t, int i, int j, typename Tile::Type value)
{
    t.insert(tile_coord(i, j), value);
}
template <typename Tile> void tile_insert(Tile& t, int i, int j, int k, typename Tile::Type value)
{
    t.insert(tile_coord(i, j, k), value);
}
template <typename Tile> void tile_insert(Tile& t, int i, int j, int k, int l, typename Tile::Type value)
{
    t.insert(tile_coord(i, j, k, l), value);
}

template <typename Tile, typename AdjTile>
void adj_tile_insert(
    Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value
)
{
    adj_t.adj_insert(tile_coord(i), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_insert(
    Tile& t,
    int i,
    int j,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_insert(tile_coord(i, j), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_insert(
    Tile& t,
    int i,
    int j,
    int k,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_insert(tile_coord(i, j, k), adj_value);
}
template <typename Tile, typename AdjTile>
void adj_tile_insert(
    Tile& t,
    int i,
    int j,
    int k,
    int l,
    typename Tile::Type value,
    AdjTile& adj_t,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    typename Tile::Type& adj_value
)
{
    adj_t.adj_insert(tile_coord(i, j, k, l), adj_value);
}

namespace partitioned_gemm {

template <typename T> inline CUDA_CALLABLE const T& index(const T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <typename T> inline CUDA_CALLABLE T& index(T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <int PartitionM, int PartitionN, typename Tile> struct partition_t {
    static constexpr int M = PartitionM;
    static constexpr int N = PartitionN;
    static constexpr int Stride = Tile::Layout::Shape::dim(1);

    using T = typename Tile::Type;

    inline partition_t(Tile& A)
    {
        data = A.data.ptr;

        // todo: do ceil div for non-multiples of M,N
        shape[0] = Tile::Layout::Shape::dim(0) / PartitionM;
        shape[1] = Tile::Layout::Shape::dim(1) / PartitionN;
    }

    // underlying data
    T* data;

    // partition dimensions
    int shape[2];
};

template <typename Partition> inline int partition_size(const Partition& part) { return part.shape[0] * part.shape[1]; }

// returns the x, y coordinates of a tile given a linear index
template <typename Partition> inline void partition_coord(const Partition& part, const int t, int& i, int& j)
{
    i = t / part.shape[1];
    j = t % part.shape[1];
}

template <typename Partition> inline auto partition_load(const Partition& tile, int i, int j)
{
    mat_t<Partition::M, Partition::N, typename Partition::T> out;

    const int tile_i = i * Partition::M;
    const int tile_j = j * Partition::N;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            out.data[i][j] = partitioned_gemm::index(tile.data, tile_i + i, tile_j + j, Partition::Stride);
        }
    }

    return out;
}

template <typename Partition, typename Value>
inline void partition_store(const Partition& tile, int i, int j, const Value& value)
{
    const int tile_i = Partition::M * i;
    const int tile_j = Partition::N * j;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            index(tile.data, tile_i + i, tile_j + j, Partition::Stride) = value.data[i][j];
        }
    }
}


template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE void matmul(TileA& A, TileB& B, TileC& out)
{
    const int TILE_M = 4;
    const int TILE_N = 4;
    const int TILE_K = 4;

    auto A_tile = partition_t<TILE_M, TILE_K, TileA>(A);
    auto B_tile = partition_t<TILE_K, TILE_N, TileB>(B);
    auto C_tile = partition_t<TILE_M, TILE_N, TileC>(out);

    // static_assert(is_same<typename TileA::Type, typename TileB::Type>::value);

    const int length = partition_size(C_tile);

    for (int t = WP_TILE_THREAD_IDX; t < length; t += WP_TILE_BLOCK_DIM) {
        int i, j;
        partition_coord(C_tile, t, i, j);

        // accumulator
        auto sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k = 0; k < A_tile.shape[1]; k++) {
            const auto a = partition_load(A_tile, i, k);
            const auto b = partition_load(B_tile, k, j);

            sum += mul(a, b);
        }

        partition_store(C_tile, i, j, sum);
    }
}

template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    for (int t = WP_TILE_THREAD_IDX; t < LayoutC::Size; t += WP_TILE_BLOCK_DIM) {
        auto coord = LayoutC::coord_from_linear(t);

        int i = coord[0];
        int j = coord[1];

        // accumulator
        using TypeC = typename remove_reference<decltype(C(coord))>::type;
        TypeC sum = TypeC(0);

        WP_PRAGMA_UNROLL
        for (int k = 0; k < LayoutA::Shape::dim(1); k++) {
            const auto a = A(tile_coord(i, k));
            const auto b = B(tile_coord(k, j));

            sum = muladd<decltype(sum)>(a, b, sum);
        }

        C(coord) = alpha * sum + beta * C(coord);
    }
}

template <typename TileA, typename TileL> inline CUDA_CALLABLE void scalar_cholesky(TileA& A, TileL& L)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    for (int j = 0; j < n; ++j) {
        T s = A.data(tile_coord(j, j));

        for (int k = 0; k < j; ++k) {
            T r = L.data(tile_coord(j, k));
            s -= r * r;
        }

        s = wp::sqrt(s);
        T invS = 1.0 / s;

        L.data(tile_coord(j, j)) = s;

        for (int i = j + 1; i < n; ++i) {
            s = A.data(tile_coord(i, j));

            for (int k = 0; k < j; ++k) {
                s -= L.data(tile_coord(i, k)) * L.data(tile_coord(j, k));
            }

            L.data(tile_coord(i, j)) = s * invS;
        }

        // zero out upper triangular portion
        for (int k = j + 1; k < n; ++k) {
            L.data(tile_coord(j, k)) = T {};
        }
    }
}

// Writes into X
template <typename TileL, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_forward_substitution(TileL& L, TileX& X, TileY& Y)
{
    using T = typename TileL::Type;

    if constexpr (TileY::Layout::Shape::N == 1) {
        constexpr int n = TileL::Layout::Shape::dim(1);

        for (int i = 0; i < n; ++i) {
            T s = Y.data(tile_coord(i));

            for (int j = 0; j < i; ++j)
                s -= L.data(tile_coord(i, j)) * X.data(tile_coord(j));

            T diag = L.data(tile_coord(i, i));
            X.data(tile_coord(i)) = (diag != T(0.0f)) ? s / diag : s;
        }
    } else if constexpr (TileY::Layout::Shape::N == 2) {
        constexpr int n = TileL::Layout::Shape::dim(1);
        constexpr int m = TileY::Layout::Shape::dim(1);

        for (int k = 0; k < m; ++k) {
            for (int i = 0; i < n; ++i) {
                T s = Y.data(tile_coord(i, k));

                for (int j = 0; j < i; ++j)
                    s -= L.data(tile_coord(i, j)) * X.data(tile_coord(j, k));

                T diag = L.data(tile_coord(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
    }
}

// Reads and writes X
template <typename TileL, typename TileX>
inline CUDA_CALLABLE void scalar_cholesky_back_substitution(TileL& L, TileX& X)
{
    using T = typename TileL::Type;

    if constexpr (TileX::Layout::Shape::N == 1) {
        constexpr int n = TileL::Layout::Shape::dim(1);

        for (int i = n - 1; i >= 0; --i) {
            T s = X.data(tile_coord(i));

            for (int j = i + 1; j < n; ++j)
                s -= L.data(tile_coord(j, i)) * X.data(tile_coord(j));

            T diag = L.data(tile_coord(i, i));
            X.data(tile_coord(i)) = (diag != T(0.0f)) ? s / diag : s;
        }
    } else if constexpr (TileX::Layout::Shape::N == 2) {
        constexpr int n = TileL::Layout::Shape::dim(1);
        constexpr int m = TileX::Layout::Shape::dim(1);

        for (int k = 0; k < m; ++k) {
            for (int i = n - 1; i >= 0; --i) {
                T s = X.data(tile_coord(i, k));

                for (int j = i + 1; j < n; ++j)
                    s -= L.data(tile_coord(j, i)) * X.data(tile_coord(j, k));

                T diag = L.data(tile_coord(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
    }
}

template <typename TileL, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_solve(TileL& L, TileX& X, TileY& Y)
{
    scalar_cholesky_forward_substitution(L, X, Y);
    scalar_cholesky_back_substitution(L, X);
}


}  // namespace partition_gemm


// ============================================================================
// Native MMA Infrastructure (PTX mma.sync.aligned.m16n8k16)
// Bypasses cuBLASDx LTO for direct tensor core access.
// Requires sm_80+ (Ampere/Hopper/Blackwell).
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// Fragment types for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// Each warp (32 threads) computes a 16x8 output tile.
struct mma_frag_a_m16n8k16 { unsigned int x[4]; };  // 8 fp16 packed as 4 uint32
struct mma_frag_b_m16n8k16 { unsigned int x[2]; };  // 4 fp16 packed as 2 uint32
struct mma_frag_acc_m16n8k16 { float x[4]; };        // 4 fp32

// PTX MMA instruction wrapper: D = A * B + C
// A [16,16] row-major fp16, B [16,8] col-major fp16, C/D [16,8] fp32
inline CUDA_CALLABLE void mma_m16n8k16_f16_f32(
    mma_frag_acc_m16n8k16& d,
    const mma_frag_a_m16n8k16& a,
    const mma_frag_b_m16n8k16& b,
    const mma_frag_acc_m16n8k16& c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[0]), "r"(b.x[1]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
    );
}

// Load A fragment from row-major SMEM for mma.sync.m16n8k16.row
// PTX ISA mapping (groupID = lane/4, tid = lane%4):
//   a[0] = {A[groupID,   tid*2],   A[groupID,   tid*2+1]}     (rows 0-7, k 0-7)
//   a[1] = {A[groupID,   tid*2+8], A[groupID,   tid*2+9]}     (rows 0-7, k 8-15)
//   a[2] = {A[groupID+8, tid*2],   A[groupID+8, tid*2+1]}     (rows 8-15, k 0-7)
//   a[3] = {A[groupID+8, tid*2+8], A[groupID+8, tid*2+9]}     (rows 8-15, k 8-15)
inline CUDA_CALLABLE void load_mma_frag_a_rowmajor(
    mma_frag_a_m16n8k16& frag,
    const half* smem,       // pointer to tile start in SMEM
    int m_offset,           // row offset within tile
    int k_offset,           // column offset within tile
    int ld)                 // leading dimension (stride between rows)
{
    int lane = threadIdx.x & 31;
    int group = lane >> 2;       // 0..7
    int tid = lane & 3;          // 0..3

    const half* row0 = smem + (m_offset + group) * ld + k_offset;
    const half* row1 = smem + (m_offset + group + 8) * ld + k_offset;

    frag.x[0] = *reinterpret_cast<const unsigned int*>(&row0[tid * 2]);
    frag.x[1] = *reinterpret_cast<const unsigned int*>(&row0[tid * 2 + 8]);
    frag.x[2] = *reinterpret_cast<const unsigned int*>(&row1[tid * 2]);
    frag.x[3] = *reinterpret_cast<const unsigned int*>(&row1[tid * 2 + 8]);
}

// Load B fragment for mma.sync.m16n8k16.col from col-major data in SMEM.
// PTX ISA mapping:
//   b[0] = {B[tid*2,   groupID], B[tid*2+1,   groupID]}    (k rows 0-7)
//   b[1] = {B[tid*2+8, groupID], B[tid*2+8+1, groupID]}    (k rows 8-15)
// B col-major: B[k,n] at address n*ldb + k
inline CUDA_CALLABLE void load_mma_frag_b_colmajor(
    mma_frag_b_m16n8k16& frag,
    const half* smem,
    int k_offset,
    int n_offset,
    int ldb)                // leading dimension for col-major (= K typically)
{
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int tid = lane & 3;

    const half* col = smem + (n_offset + group) * ldb + k_offset;
    frag.x[0] = *reinterpret_cast<const unsigned int*>(&col[tid * 2]);
    frag.x[1] = *reinterpret_cast<const unsigned int*>(&col[tid * 2 + 8]);
}

// Helper: pack two half values into a uint32 (low bits = first, high bits = second)
inline CUDA_CALLABLE unsigned int mma_pack_half2(half a, half b)
{
    unsigned int result;
    asm("{mov.b32 %0, {%1, %2};}\n" : "=r"(result) : "h"(*reinterpret_cast<unsigned short*>(&a)),
                                                       "h"(*reinterpret_cast<unsigned short*>(&b)));
    return result;
}

// Load B fragment from ROW-MAJOR SMEM (B is logically col-major for MMA).
// Use case: V is stored [K, N] row-major. For MMA B col-major [K, N]:
//   B[k, n] = V[k, n] = smem[k * ld + n]  (row-major access)
// But col-major B would have B[k, n] at n * ldb + k.
// Since they differ, we load individual elements and pack.
inline CUDA_CALLABLE void load_mma_frag_b_from_rowmajor(
    mma_frag_b_m16n8k16& frag,
    const half* smem,       // V stored row-major [K, N]
    int k_offset,
    int n_offset,
    int ld)                 // leading dimension (= N, the number of columns)
{
    int lane = threadIdx.x & 31;
    int group = lane >> 2;       // maps to n dimension
    int tid = lane & 3;          // maps to k dimension

    // B col-major element B[k, n]:
    //   b[0] = {B[tid*2,   group], B[tid*2+1,   group]}
    //   b[1] = {B[tid*2+8, group], B[tid*2+8+1, group]}
    // From V row-major: B[k, n] = V[k, n] = smem[(k_offset + k) * ld + (n_offset + n)]
    // For b[0]: k = tid*2, tid*2+1, n = group
    //   = V[k_offset + tid*2, n_offset + group], V[k_offset + tid*2 + 1, n_offset + group]
    // These are NOT contiguous in memory (stride = ld between them).
    // Must load individually and pack.

    int n = n_offset + group;
    int k0 = k_offset + tid * 2;
    int k1 = k0 + 1;
    int k2 = k_offset + tid * 2 + 8;
    int k3 = k2 + 1;

    half v0 = smem[k0 * ld + n];
    half v1 = smem[k1 * ld + n];
    half v2 = smem[k2 * ld + n];
    half v3 = smem[k3 * ld + n];

    // Pack two half values into uint32
    frag.x[0] = mma_pack_half2(v0, v1);
    frag.x[1] = mma_pack_half2(v2, v3);
}


// ============================================================================
// MMA Accumulator Tile
// Per-thread fp32 register array for M×N output with NumWarps warps.
// Each warp computes a subset of 16×8 output tiles.
// ============================================================================

template <int M, int N, int NumWarps>
struct tile_mma_acc_t {
    static_assert(M % 16 == 0, "M must be multiple of 16");
    static_assert(N % 8 == 0, "N must be multiple of 8");

    static constexpr int m_tiles = M / 16;
    static constexpr int n_tiles = N / 8;
    static constexpr int total_tiles = m_tiles * n_tiles;
    // Round-robin assignment: warp w gets tiles w, w+NumWarps, w+2*NumWarps, ...
    static constexpr int tiles_per_warp = (total_tiles + NumWarps - 1) / NumWarps;
    static constexpr int regs_per_thread = tiles_per_warp * 4;

    float data[regs_per_thread];

    inline CUDA_CALLABLE void zero()
    {
        WP_PRAGMA_UNROLL
        for (int i = 0; i < regs_per_thread; i++)
            data[i] = 0.f;
    }

    // Scale all accumulator values by a per-row factor.
    // row_alpha is a SMEM array of M floats: row_alpha[row] = scale for that row.
    // Each thread's fragment elements map to specific rows based on MMA layout.
    inline CUDA_CALLABLE void scale_rows(const float* row_alpha)
    {
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x & 31;
        int group = lane >> 2;  // 0..7

        WP_PRAGMA_UNROLL
        for (int t = 0; t < tiles_per_warp; t++)
        {
            int tile_idx = warp_id + t * NumWarps;
            if (tile_idx >= total_tiles) break;

            int m_tile = tile_idx / n_tiles;
            int row0 = m_tile * 16 + group;
            int row1 = row0 + 8;

            float s0 = row_alpha[row0];
            float s1 = row_alpha[row1];

            data[t * 4 + 0] *= s0;
            data[t * 4 + 1] *= s0;
            data[t * 4 + 2] *= s1;
            data[t * 4 + 3] *= s1;
        }
    }

    // Scale all elements by a uniform scalar (for cases where all rows share the same alpha).
    inline CUDA_CALLABLE void scale_uniform(float alpha)
    {
        WP_PRAGMA_UNROLL
        for (int i = 0; i < regs_per_thread; i++)
            data[i] *= alpha;
    }

    // Store accumulator to SMEM as fp16, each thread writes its fragment positions.
    inline CUDA_CALLABLE void store_to_smem_f16(half* smem_out, int ld) const
    {
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x & 31;
        int group = lane >> 2;
        int tid = lane & 3;

        WP_PRAGMA_UNROLL
        for (int t = 0; t < tiles_per_warp; t++)
        {
            int tile_idx = warp_id + t * NumWarps;
            if (tile_idx >= total_tiles) break;

            int m_tile = tile_idx / n_tiles;
            int n_tile = tile_idx % n_tiles;

            int r0 = m_tile * 16 + group;
            int c0 = n_tile * 8 + tid * 2;
            int r1 = r0 + 8;

            smem_out[r0 * ld + c0]     = float_to_half(data[t * 4 + 0]);
            smem_out[r0 * ld + c0 + 1] = float_to_half(data[t * 4 + 1]);
            smem_out[r1 * ld + c0]     = float_to_half(data[t * 4 + 2]);
            smem_out[r1 * ld + c0 + 1] = float_to_half(data[t * 4 + 3]);
        }
    }

    // Store to global memory as fp16, with per-row normalization (divide by l_i).
    // row_inv_l is a SMEM array of M floats: 1.0/l_i for each row.
    inline CUDA_CALLABLE void store_to_global_f16_normalized(
        half* global_out, int ld,
        const float* row_inv_l,
        int m_global_offset) const
    {
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x & 31;
        int group = lane >> 2;
        int tid = lane & 3;

        WP_PRAGMA_UNROLL
        for (int t = 0; t < tiles_per_warp; t++)
        {
            int tile_idx = warp_id + t * NumWarps;
            if (tile_idx >= total_tiles) break;

            int m_tile = tile_idx / n_tiles;
            int n_tile = tile_idx % n_tiles;

            int r0 = m_tile * 16 + group;
            int c0 = n_tile * 8 + tid * 2;
            int r1 = r0 + 8;

            float inv_l0 = row_inv_l[r0];
            float inv_l1 = row_inv_l[r1];

            global_out[(m_global_offset + r0) * ld + c0]     = float_to_half(data[t * 4 + 0] * inv_l0);
            global_out[(m_global_offset + r0) * ld + c0 + 1] = float_to_half(data[t * 4 + 1] * inv_l0);
            global_out[(m_global_offset + r1) * ld + c0]     = float_to_half(data[t * 4 + 2] * inv_l1);
            global_out[(m_global_offset + r1) * ld + c0 + 1] = float_to_half(data[t * 4 + 3] * inv_l1);
        }
    }
};


// ============================================================================
// Native MMA Matmul Functions
// ============================================================================

// Compute C[M,N] = A[M,K] @ B^T[K,N] using MMA, write fp16 result to SMEM.
// A is row-major [M,K] in SMEM. B is row-major [N,K] in SMEM (= B^T col-major [K,N]).
// Each warp processes its share of 16×8 output tiles; results written to SMEM immediately.
template <int M, int K, int N, int NumWarps>
inline CUDA_CALLABLE void tile_matmul_mma_to_smem(
    const half* __restrict__ smem_a, int lda,
    const half* __restrict__ smem_b, int ldb,
    half* __restrict__ smem_c, int ldc)
{
    const int warp_id = threadIdx.x / 32;

    constexpr int m_tiles = M / 16;
    constexpr int n_tiles = N / 8;
    constexpr int total_tiles = m_tiles * n_tiles;
    constexpr int k_steps = K / 16;
    constexpr int tiles_per_warp = (total_tiles + NumWarps - 1) / NumWarps;

    _Pragma("unroll 1")
    for (int t = 0; t < tiles_per_warp; t++)
    {
        int tile_idx = warp_id + t * NumWarps;
        if (tile_idx >= total_tiles) break;

        int m_tile = tile_idx / n_tiles;
        int n_tile = tile_idx % n_tiles;

        mma_frag_acc_m16n8k16 c_frag = {0.f, 0.f, 0.f, 0.f};

        WP_PRAGMA_UNROLL
        for (int k = 0; k < k_steps; k++)
        {
            mma_frag_a_m16n8k16 a_frag;
            mma_frag_b_m16n8k16 b_frag;

            // A: Q[m_tile*16..+16, k*16..+16] row-major
            load_mma_frag_a_rowmajor(a_frag, smem_a, m_tile * 16, k * 16, lda);

            // B^T: K is [N,K] row-major = K^T col-major. Load as col-major.
            // Column n_tile*8..+8 of K^T = row n_tile*8..+8 of K.
            // K[n, k] = K^T col-major at column n, row k => ldb = K's leading dim
            load_mma_frag_b_colmajor(b_frag, smem_b, k * 16, n_tile * 8, ldb);

            mma_m16n8k16_f16_f32(c_frag, a_frag, b_frag, c_frag);
        }

        // Write fp32 result as fp16 to SMEM immediately (avoid register pressure)
        int lane = threadIdx.x & 31;
        int group = lane >> 2;
        int tid = lane & 3;

        int r0 = m_tile * 16 + group;
        int c0 = n_tile * 8 + tid * 2;
        int r1 = r0 + 8;

        smem_c[r0 * ldc + c0]     = float_to_half(c_frag.x[0]);
        smem_c[r0 * ldc + c0 + 1] = float_to_half(c_frag.x[1]);
        smem_c[r1 * ldc + c0]     = float_to_half(c_frag.x[2]);
        smem_c[r1 * ldc + c0 + 1] = float_to_half(c_frag.x[3]);
    }
}

// Accumulate into register tile: acc[M,N] += A[M,K] @ B[K,N]
// A is row-major [M,K] in SMEM. B is row-major [K,N] in SMEM.
// B needs transposed fragment loading since MMA expects col-major B.
template <int M, int K, int N, int NumWarps>
inline CUDA_CALLABLE void tile_matmul_mma_accumulate(
    const half* __restrict__ smem_a, int lda,
    const half* __restrict__ smem_b, int ldb,
    tile_mma_acc_t<M, N, NumWarps>& acc)
{
    const int warp_id = threadIdx.x / 32;
    constexpr int k_steps = K / 16;

    _Pragma("unroll 1")
    for (int t = 0; t < tile_mma_acc_t<M, N, NumWarps>::tiles_per_warp; t++)
    {
        int tile_idx = warp_id + t * NumWarps;
        if (tile_idx >= tile_mma_acc_t<M, N, NumWarps>::total_tiles) break;

        int m_tile = tile_idx / tile_mma_acc_t<M, N, NumWarps>::n_tiles;
        int n_tile = tile_idx % tile_mma_acc_t<M, N, NumWarps>::n_tiles;

        mma_frag_acc_m16n8k16 c_frag = {
            acc.data[t * 4 + 0], acc.data[t * 4 + 1],
            acc.data[t * 4 + 2], acc.data[t * 4 + 3]
        };

        WP_PRAGMA_UNROLL
        for (int k = 0; k < k_steps; k++)
        {
            mma_frag_a_m16n8k16 a_frag;
            mma_frag_b_m16n8k16 b_frag;

            // A: P[m_tile*16..+16, k*16..+16] row-major
            load_mma_frag_a_rowmajor(a_frag, smem_a, m_tile * 16, k * 16, lda);

            // B: V[k*16..+16, n_tile*8..+8] row-major -> need transposed load
            load_mma_frag_b_from_rowmajor(b_frag, smem_b, k * 16, n_tile * 8, ldb);

            mma_m16n8k16_f16_f32(c_frag, a_frag, b_frag, c_frag);
        }

        acc.data[t * 4 + 0] = c_frag.x[0];
        acc.data[t * 4 + 1] = c_frag.x[1];
        acc.data[t * 4 + 2] = c_frag.x[2];
        acc.data[t * 4 + 3] = c_frag.x[3];
    }
}

// ============================================================================
// Fused Flash Attention MMA Kernel
// Single function that does the entire flash attention loop for one Q block:
//   - Load Q to SMEM
//   - For each K/V block: load K/V, MMA QK^T, softmax, MMA PV
//   - Normalize and write output
// Template params: TILE_M, TILE_N, HEAD_DIM
// Must be launched with block_dim = NUM_WARPS * 32 (8 warps = 256 threads).
// Each CTA processes one (batch_head, q_block) pair.
// Register-based softmax: each warp handles 1 m_tile (16 rows) × all n_tiles.
// Softmax via quad_allreduce (__shfl_xor_sync across 4 threads sharing a row).
// No SMEM needed for attention scores — only Q, K, V in shared memory.
// ============================================================================
template <int TILE_M, int TILE_N, int HEAD_DIM, int NUM_WARPS = 8>
inline CUDA_CALLABLE void flash_attention_mma_kernel(
    const half* __restrict__ Q,  // [batch_heads, seq_len, HEAD_DIM]
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float sm_scale,
    int seq_len,
    int batch_heads,
    int num_q_blocks,
    int num_k_blocks,
    int batch_head,         // which batch_head this CTA processes
    int q_block_idx)        // which Q block this CTA processes
{
    const int tid = threadIdx.x;
    const int q_start = q_block_idx * TILE_M;

    // Shared memory layout: Q[TILE_M, HEAD_DIM] + K[TILE_N, HEAD_DIM] + V[TILE_N, V_STRIDE]
    // V is padded to avoid bank conflicts
    static constexpr int V_PAD = 8;
    static constexpr int V_STRIDE = HEAD_DIM + V_PAD;
    extern __shared__ char dynamic_smem_base[];
    half* smem_Q = reinterpret_cast<half*>(dynamic_smem_base);
    half* smem_K = smem_Q + TILE_M * HEAD_DIM;
    half* smem_V = smem_K + TILE_N * HEAD_DIM;

    // Warp/lane decomposition
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int group = lane >> 2;   // 0-7: maps to rows within 16-row tile
    const int ltid = lane & 3;    // 0-3: maps to column groups

    // Tile counts
    static constexpr int N_TILES = TILE_N / 8;       // n_tiles for QK^T scores
    static constexpr int HD_TILES = HEAD_DIM / 8;     // n_tiles for O accumulator
    static constexpr int QK_K_STEPS = HEAD_DIM / 16;  // k_steps for QK^T
    static constexpr int PV_K_STEPS = TILE_N / 16;    // k_steps for PV

    // Register accumulators
    float O_regs[HD_TILES * 4];   // output accumulator (persistent)
    float S_regs[N_TILES * 4];    // QK^T scores (reused each k_block)
    float m_r0 = -1e10f, m_r1 = -1e10f;  // running max for 2 rows (group, group+8)
    float l_r0 = 0.f, l_r1 = 0.f;        // running sum for 2 rows

    WP_PRAGMA_UNROLL
    for (int i = 0; i < HD_TILES * 4; i++) O_regs[i] = 0.f;

    const float sm_scale_log2 = sm_scale * 1.44269504f;

    // Load Q block to SMEM [TILE_M, HEAD_DIM]
    const half* Q_src = Q + (batch_head * seq_len + q_start) * HEAD_DIM;
    for (int i = tid; i < TILE_M * HEAD_DIM; i += blockDim.x)
    {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (q_start + row < seq_len)
            smem_Q[i] = Q_src[row * HEAD_DIM + col];
        else
            smem_Q[i] = float_to_half(0.0f);
    }
    __syncthreads();

    for (int k_block = 0; k_block < num_k_blocks; k_block++)
    {
        int k_start = k_block * TILE_N;

        // Load K block to SMEM [TILE_N, HEAD_DIM]
        const half* K_src = K + (batch_head * seq_len + k_start) * HEAD_DIM;
        for (int i = tid; i < TILE_N * HEAD_DIM; i += blockDim.x)
        {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (k_start + row < seq_len)
                smem_K[i] = K_src[row * HEAD_DIM + col];
            else
                smem_K[i] = float_to_half(0.0f);
        }

        // Load V block to SMEM [TILE_N, V_STRIDE] (padded row-major)
        const half* V_src = V + (batch_head * seq_len + k_start) * HEAD_DIM;
        for (int i = tid; i < TILE_N * HEAD_DIM; i += blockDim.x)
        {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (k_start + row < seq_len)
                smem_V[row * V_STRIDE + col] = V_src[row * HEAD_DIM + col];
            else
                smem_V[row * V_STRIDE + col] = float_to_half(0.0f);
        }
        __syncthreads();

        // ===== QK^T: each warp computes ALL n_tiles for its m_tile =====
        // warp_id maps to m_tile (16 rows), iterate over all n_tiles (8 columns each)
        WP_PRAGMA_UNROLL
        for (int j = 0; j < N_TILES; j++)
        {
            mma_frag_acc_m16n8k16 c_frag = {0.f, 0.f, 0.f, 0.f};

            WP_PRAGMA_UNROLL
            for (int k = 0; k < QK_K_STEPS; k++)
            {
                mma_frag_a_m16n8k16 a_frag;
                mma_frag_b_m16n8k16 b_frag;

                load_mma_frag_a_rowmajor(a_frag, smem_Q, warp_id * 16, k * 16, HEAD_DIM);
                load_mma_frag_b_colmajor(b_frag, smem_K, k * 16, j * 8, HEAD_DIM);

                mma_m16n8k16_f16_f32(c_frag, a_frag, b_frag, c_frag);
            }

            S_regs[j * 4 + 0] = c_frag.x[0];  // row group, cols j*8+ltid*2
            S_regs[j * 4 + 1] = c_frag.x[1];  // row group, cols j*8+ltid*2+1
            S_regs[j * 4 + 2] = c_frag.x[2];  // row group+8
            S_regs[j * 4 + 3] = c_frag.x[3];  // row group+8
        }

        // ===== Softmax: register-based with quad_allreduce =====

        // 1. Scale and find max for each row (r0=group, r1=group+8)
        float max_r0 = -1e10f, max_r1 = -1e10f;
        WP_PRAGMA_UNROLL
        for (int j = 0; j < N_TILES; j++)
        {
            S_regs[j * 4 + 0] *= sm_scale_log2;
            S_regs[j * 4 + 1] *= sm_scale_log2;
            S_regs[j * 4 + 2] *= sm_scale_log2;
            S_regs[j * 4 + 3] *= sm_scale_log2;
            max_r0 = fmaxf(max_r0, fmaxf(S_regs[j * 4 + 0], S_regs[j * 4 + 1]));
            max_r1 = fmaxf(max_r1, fmaxf(S_regs[j * 4 + 2], S_regs[j * 4 + 3]));
        }

        // Quad allreduce max across 4 tids (covers all 64 columns)
        max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 1));
        max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 2));
        max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 1));
        max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 2));

        // 2. Rescale O_acc and l_i
        float m_new_r0 = fmaxf(m_r0, max_r0);
        float m_new_r1 = fmaxf(m_r1, max_r1);
        float alpha_r0 = exp2f(m_r0 - m_new_r0);
        float alpha_r1 = exp2f(m_r1 - m_new_r1);
        l_r0 *= alpha_r0;
        l_r1 *= alpha_r1;

        WP_PRAGMA_UNROLL
        for (int h = 0; h < HD_TILES; h++)
        {
            O_regs[h * 4 + 0] *= alpha_r0;
            O_regs[h * 4 + 1] *= alpha_r0;
            O_regs[h * 4 + 2] *= alpha_r1;
            O_regs[h * 4 + 3] *= alpha_r1;
        }

        // 3. Apply exp2f and accumulate sum
        float sum_r0 = 0.f, sum_r1 = 0.f;
        WP_PRAGMA_UNROLL
        for (int j = 0; j < N_TILES; j++)
        {
            S_regs[j * 4 + 0] = exp2f(S_regs[j * 4 + 0] - m_new_r0);
            S_regs[j * 4 + 1] = exp2f(S_regs[j * 4 + 1] - m_new_r0);
            S_regs[j * 4 + 2] = exp2f(S_regs[j * 4 + 2] - m_new_r1);
            S_regs[j * 4 + 3] = exp2f(S_regs[j * 4 + 3] - m_new_r1);
            sum_r0 += S_regs[j * 4 + 0] + S_regs[j * 4 + 1];
            sum_r1 += S_regs[j * 4 + 2] + S_regs[j * 4 + 3];
        }

        sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 1);
        sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 2);
        sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 1);
        sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 2);

        l_r0 += sum_r0;
        l_r1 += sum_r1;
        m_r0 = m_new_r0;
        m_r1 = m_new_r1;

        // ===== PV GEMM: A from registers, B from SMEM V =====
        WP_PRAGMA_UNROLL
        for (int h = 0; h < HD_TILES; h++)
        {
            mma_frag_acc_m16n8k16 c_frag = {
                O_regs[h * 4 + 0], O_regs[h * 4 + 1],
                O_regs[h * 4 + 2], O_regs[h * 4 + 3]
            };

            WP_PRAGMA_UNROLL
            for (int s = 0; s < PV_K_STEPS; s++)
            {
                mma_frag_a_m16n8k16 a_frag;
                mma_frag_b_m16n8k16 b_frag;

                // A from register scores: k_step s uses n_tiles 2s and 2s+1
                int sj0 = 2 * s;
                int sj1 = 2 * s + 1;
                a_frag.x[0] = mma_pack_half2(float_to_half(S_regs[sj0 * 4 + 0]),
                                             float_to_half(S_regs[sj0 * 4 + 1]));
                a_frag.x[1] = mma_pack_half2(float_to_half(S_regs[sj1 * 4 + 0]),
                                             float_to_half(S_regs[sj1 * 4 + 1]));
                a_frag.x[2] = mma_pack_half2(float_to_half(S_regs[sj0 * 4 + 2]),
                                             float_to_half(S_regs[sj0 * 4 + 3]));
                a_frag.x[3] = mma_pack_half2(float_to_half(S_regs[sj1 * 4 + 2]),
                                             float_to_half(S_regs[sj1 * 4 + 3]));

                // B from SMEM V (row-major, with padded stride)
                load_mma_frag_b_from_rowmajor(b_frag, smem_V, s * 16, h * 8, V_STRIDE);

                mma_m16n8k16_f16_f32(c_frag, a_frag, b_frag, c_frag);
            }

            O_regs[h * 4 + 0] = c_frag.x[0];
            O_regs[h * 4 + 1] = c_frag.x[1];
            O_regs[h * 4 + 2] = c_frag.x[2];
            O_regs[h * 4 + 3] = c_frag.x[3];
        }

        __syncthreads();  // Ensure all warps done reading K/V before next iteration overwrites
    }

    // ===== Final output: write from registers to global memory =====
    float inv_l0 = (l_r0 > 0.f) ? (1.f / l_r0) : 0.f;
    float inv_l1 = (l_r1 > 0.f) ? (1.f / l_r1) : 0.f;
    half* O_out = O + (batch_head * seq_len + q_start) * HEAD_DIM;
    int r0 = warp_id * 16 + group;
    int r1 = r0 + 8;

    if (q_start + r0 < seq_len)
    {
        WP_PRAGMA_UNROLL
        for (int h = 0; h < HD_TILES; h++)
        {
            int c0 = h * 8 + ltid * 2;
            O_out[r0 * HEAD_DIM + c0]     = float_to_half(O_regs[h * 4 + 0] * inv_l0);
            O_out[r0 * HEAD_DIM + c0 + 1] = float_to_half(O_regs[h * 4 + 1] * inv_l0);
        }
    }
    if (q_start + r1 < seq_len)
    {
        WP_PRAGMA_UNROLL
        for (int h = 0; h < HD_TILES; h++)
        {
            int c0 = h * 8 + ltid * 2;
            O_out[r1 * HEAD_DIM + c0]     = float_to_half(O_regs[h * 4 + 2] * inv_l1);
            O_out[r1 * HEAD_DIM + c0 + 1] = float_to_half(O_regs[h * 4 + 3] * inv_l1);
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800


template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta>
TileC& tile_matmul(
    Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, Alpha& alpha, Beta& beta
)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");


    using T = typename TileC::Type;

    T alphaT = T(alpha);
    T betaT = T(beta);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_matmul<typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
        A.data, B.data, C.data, alphaT, betaT
    );
#else
    fun_forward(&alphaT, A.data.ptr, B.data.ptr, &betaT, C.data.ptr);
#endif

    WP_TILE_SYNC();

    return C;
}


// backward for the wp.tile_matmul(a, b, out) syntax
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_matmul(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    partitioned_gemm::scalar_matmul<typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul<typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
    fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
#endif

    if (T_C(beta) != T_C(1.0)) {
        for (int i = WP_TILE_THREAD_IDX; i < TileC::Layout::Size; i += WP_TILE_BLOCK_DIM)
            adj_C.grad(i) *= T_C(beta);
    }

    WP_TILE_SYNC();
}

// backward for the out = wp.tile_matmul(a, b) syntax
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_matmul(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta,
    TileC& adj_ret
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    partitioned_gemm::scalar_matmul<typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul<typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
    fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
#endif

    WP_TILE_SYNC();
}

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

#define tile_fft()
#define tile_ifft()

#define adj_tile_fft()
#define adj_tile_ifft()

// RMEM stubs for CPU
#define tile_rmem_clear(fn_clear, var_out)
#define adj_tile_rmem_clear(fn_clear, var_out, adj_fn_clear, adj_var_out)
#define tile_matmul_rmem(fn_copy_a, fn_copy_b, fn_execute, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes)
#define adj_tile_matmul_rmem(fn_copy_a, fn_copy_b, fn_execute, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes, adj_fn_copy_a, adj_fn_copy_b, adj_fn_execute, adj_A, adj_B, adj_var_C, adj_smem_a_sugg_bytes, adj_smem_b_sugg_bytes)
#define tile_rmem_scale(fn_map, fn_bounds, var_C, alpha, logical_size)
#define adj_tile_rmem_scale(fn_map, fn_bounds, var_C, alpha, logical_size, adj_fn_map, adj_fn_bounds, adj_var_C, adj_alpha, adj_logical_size)
#define tile_rmem_copy(fn_map, fn_bounds, var_src, var_dst, logical_size)
#define adj_tile_rmem_copy(fn_map, fn_bounds, var_src, var_dst, logical_size, adj_fn_map, adj_fn_bounds, adj_var_src, adj_var_dst, adj_logical_size)

// Native MMA stubs for CPU
#define tile_mma_smem(A, B, C)
#define adj_tile_mma_smem(A, B, C, adj_A, adj_B, adj_C)
#define tile_mma_acc_op(A, B, acc)
#define adj_tile_mma_acc_op(A, B, acc, adj_A, adj_B, adj_acc)
#define tile_mma_scale_rows(acc, alpha_smem)
#define adj_tile_mma_scale_rows(acc, alpha_smem, adj_acc, adj_alpha_smem)
#define tile_mma_store_normalized(acc, out_ptr, inv_l_smem, ld, m_offset)
#define adj_tile_mma_store_normalized(acc, out_ptr, inv_l_smem, ld, m_offset, adj_acc, adj_out_ptr, adj_inv_l_smem, adj_ld, adj_m_offset)
#define tile_flash_attention_mma(Q_ptr, K_ptr, V_ptr, O_ptr, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks, batch_head, q_block_idx, TILE_M_val, TILE_N_val, HEAD_DIM_val)
#define adj_tile_flash_attention_mma(Q_ptr, K_ptr, V_ptr, O_ptr, sm_scale, seq_len, batch_heads, num_q_blocks, num_k_blocks, batch_head, q_block_idx, TILE_M_val, TILE_N_val, HEAD_DIM_val, adj_Q_ptr, adj_K_ptr, adj_V_ptr, adj_O_ptr, adj_sm_scale, adj_seq_len, adj_batch_heads, adj_num_q_blocks, adj_num_k_blocks, adj_batch_head, adj_q_block_idx, adj_TILE_M_val, adj_TILE_N_val, adj_HEAD_DIM_val)

#else

// TODO(lcambier): use a properly overaligned complex type that matches cuFFTDx's expectation
// and remove the need for __align__(16) dtypes data[...]
#define tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, Xinout) \
     do { \
         void function_name(dtype*, char*); \
         char* buffer = (char*)wp::tile_shared_storage_t::alloc(shared_memory_size); \
         __align__(16) dtype data[ept]; \
         for(int b = 0; b < (int)batch_size; b++) { \
             dtype* inout = Xinout.data + (int)b * (int)ept; \
             memcpy(data, inout, sizeof(dtype) * ept); \
             function_name(data, buffer); \
             memcpy(inout, data, sizeof(dtype) * ept); \
             WP_TILE_SYNC(); \
         } \
         wp::tile_shared_storage_t::alloc(-shared_memory_size); \
     } while (0)

#define tile_ifft tile_fft

// adj_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept are all ignored

#define adj_tile_fft(                                                                                                  \
    function_name, dtype, shared_memory_size, batch_size, ept, Xinout, adj_function_name, adj_dtype,                   \
    adj_shared_memory_size, adj_batch_size, adj_ept, adj_Xinout                                                        \
) \
     do { \
         tile_ifft(function_name, dtype, shared_memory_size, batch_size, ept, adj_Xinout); \
     } while (0)

#define adj_tile_ifft(                                                                                                 \
    function_name, dtype, shared_memory_size, batch_size, ept, Xinout, adj_function_name, adj_dtype,                   \
    adj_shared_memory_size, adj_batch_size, adj_ept, adj_Xinout                                                        \
) \
     do { \
         tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, adj_Xinout); \
     } while (0)

// ============================================================================
// RMEM tile macros (cuBLASDx Tensor API)
// ============================================================================
// tensor_t for cuBLASDx Tensor API: struct { void* ptr; }
struct tile_rmem_tensor_t { void* ptr; };

// tile_rmem_clear: call LTO clear on the rmem tile
#define tile_rmem_clear(fn_clear, var_out) \
    do { \
        void fn_clear(tile_rmem_tensor_t); \
        tile_rmem_tensor_t _tc = { var_out.buf }; \
        fn_clear(_tc); \
    } while (0)

#define adj_tile_rmem_clear(fn_clear, var_out, adj_fn_clear, adj_var_out)

// tile_matmul_rmem: C_rmem += A_smem @ B_smem
// First copies A,B from plain SMEM layout to suggested SMEM layout (separate buffers),
// then executes matmul. The copy functions rearrange data from the user's tile_load layout
// to cuBLASDx's optimal layout. Temporary shared memory is allocated for the suggested copies.
#define tile_matmul_rmem(fn_copy_a, fn_copy_b, fn_execute, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes) \
    do { \
        void fn_copy_a(tile_rmem_tensor_t, tile_rmem_tensor_t); \
        void fn_copy_b(tile_rmem_tensor_t, tile_rmem_tensor_t); \
        void fn_execute(tile_rmem_tensor_t, tile_rmem_tensor_t, tile_rmem_tensor_t); \
        char* _buf_a_sugg = (char*)wp::tile_shared_storage_t::alloc(smem_a_sugg_bytes); \
        char* _buf_b_sugg = (char*)wp::tile_shared_storage_t::alloc(smem_b_sugg_bytes); \
        tile_rmem_tensor_t _ta_plain = { A.data.ptr }; \
        tile_rmem_tensor_t _tb_plain = { B.data.ptr }; \
        tile_rmem_tensor_t _ta_sugg = { _buf_a_sugg }; \
        tile_rmem_tensor_t _tb_sugg = { _buf_b_sugg }; \
        fn_copy_a(_ta_plain, _ta_sugg); \
        fn_copy_b(_tb_plain, _tb_sugg); \
        WP_TILE_SYNC(); \
        tile_rmem_tensor_t _tc = { var_C.buf }; \
        fn_execute(_ta_sugg, _tb_sugg, _tc); \
        WP_TILE_SYNC(); \
        wp::tile_shared_storage_t::alloc(-(smem_a_sugg_bytes)); \
        wp::tile_shared_storage_t::alloc(-(smem_b_sugg_bytes)); \
    } while (0)

#define adj_tile_matmul_rmem(fn_copy_a, fn_copy_b, fn_execute, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes, \
    adj_fn_copy_a, adj_fn_copy_b, adj_fn_execute, adj_A, adj_B, adj_var_C, adj_smem_a_sugg_bytes, adj_smem_b_sugg_bytes)

// tile_matmul_rmem_beta: C_rmem = A_smem @ B_smem + beta * C_rmem
// Same as tile_matmul_rmem but with beta scaling of accumulator before execute.
// Uses AXPBY (D = alpha*C + beta*D) with alpha=0 to scale C_rmem by beta first.
// This is the standard GEMM equation: C = alpha*A@B + beta*C (with alpha=1 for matmul).
#define tile_matmul_rmem_beta(fn_copy_a, fn_copy_b, fn_execute, fn_axpby, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes, beta) \
    do { \
        void fn_copy_a(tile_rmem_tensor_t, tile_rmem_tensor_t); \
        void fn_copy_b(tile_rmem_tensor_t, tile_rmem_tensor_t); \
        void fn_execute(tile_rmem_tensor_t, tile_rmem_tensor_t, tile_rmem_tensor_t); \
        void fn_axpby(void*, tile_rmem_tensor_t, void*, tile_rmem_tensor_t); \
        char* _buf_a_sugg = (char*)wp::tile_shared_storage_t::alloc(smem_a_sugg_bytes); \
        char* _buf_b_sugg = (char*)wp::tile_shared_storage_t::alloc(smem_b_sugg_bytes); \
        tile_rmem_tensor_t _ta_plain = { A.data.ptr }; \
        tile_rmem_tensor_t _tb_plain = { B.data.ptr }; \
        tile_rmem_tensor_t _ta_sugg = { _buf_a_sugg }; \
        tile_rmem_tensor_t _tb_sugg = { _buf_b_sugg }; \
        fn_copy_a(_ta_plain, _ta_sugg); \
        fn_copy_b(_tb_plain, _tb_sugg); \
        WP_TILE_SYNC(); \
        tile_rmem_tensor_t _tc = { var_C.buf }; \
        /* Skip AXPBY when beta=1.0 (identity operation) to avoid overhead */ \
        if (static_cast<float>(beta) != 1.0f) { \
            wp::float16 _alpha_val = wp::float16(0.0f); \
            wp::float16 _beta_val = wp::float16(static_cast<float>(beta)); \
            fn_axpby(&_alpha_val, _tc, &_beta_val, _tc); \
        } \
        fn_execute(_ta_sugg, _tb_sugg, _tc); \
        WP_TILE_SYNC(); \
        wp::tile_shared_storage_t::alloc(-(smem_a_sugg_bytes)); \
        wp::tile_shared_storage_t::alloc(-(smem_b_sugg_bytes)); \
    } while (0)

#define adj_tile_matmul_rmem_beta(fn_copy_a, fn_copy_b, fn_execute, fn_axpby, A, B, var_C, smem_a_sugg_bytes, smem_b_sugg_bytes, beta, \
    adj_fn_copy_a, adj_fn_copy_b, adj_fn_execute, adj_fn_axpby, adj_A, adj_B, adj_var_C, adj_smem_a_sugg_bytes, adj_smem_b_sugg_bytes, adj_beta)

// tile_rmem_scale: element-wise scale RMEM tile
// For scalar multiplication, we can directly iterate over the raw buffer
// since fp16 * scalar is element-wise regardless of layout.
// Skip scaling when alpha is very close to 1.0 (common in online softmax).
#define tile_rmem_scale(fn_map, fn_bounds, var_C, alpha, logical_size) \
    do { \
        wp::float32 _alpha32 = static_cast<wp::float32>(alpha); \
        /* Early exit if alpha is close to 1.0 (common when max doesn't change) */ \
        if (_alpha32 < 0.99999f || _alpha32 > 1.00001f) { \
            wp::float16* _buf = reinterpret_cast<wp::float16*>(var_C.buf); \
            _Pragma("unroll") \
            for (int _i = 0; _i < (int)(logical_size); _i++) { \
                _buf[_i] = wp::float16(static_cast<wp::float32>(_buf[_i]) * _alpha32); \
            } \
        } \
    } while (0)

#define adj_tile_rmem_scale(fn_map, fn_bounds, var_C, alpha, logical_size, \
    adj_fn_map, adj_fn_bounds, adj_var_C, adj_alpha, adj_logical_size)

// tile_rmem_copy: copy RMEM tile to shared memory tile using MAP_IDX2CRD
// Uses element-wise access to handle the layout mismatch between RMEM and strided shared tiles.
#define tile_rmem_copy(fn_map, fn_bounds, var_src, var_dst, logical_size) \
    do { \
        void fn_map(tile_rmem_tensor_t, int*, int*, int*, void**); \
        void fn_bounds(int*, int*); \
        tile_rmem_tensor_t _tc = { var_src.buf }; \
        for (int _idx = 0; _idx < (int)(logical_size); _idx++) { \
            int _in_bounds = 0; \
            fn_bounds(&_idx, &_in_bounds); \
            if (!_in_bounds) continue; \
            int _i, _j; \
            void* _elem_ptr = nullptr; \
            fn_map(_tc, &_idx, &_i, &_j, &_elem_ptr); \
            if (_elem_ptr) { \
                wp::float16* _p = static_cast<wp::float16*>(_elem_ptr); \
                var_dst.data(wp::tile_coord_t<2>{_i, _j}) = *_p; \
            } \
        } \
        WP_TILE_SYNC(); \
    } while (0)

#define adj_tile_rmem_copy(fn_map, fn_bounds, var_src, var_dst, logical_size, \
    adj_fn_map, adj_fn_bounds, adj_var_src, adj_var_dst, adj_logical_size)

// ============================================================================
// Native MMA macros (GPU path) — called by codegen via override_native_func
// ============================================================================

// tile_mma_smem: C_smem = A_smem @ B_transposed_smem using native PTX MMA
// A is tile_shared_t [M, K] fp16 row-major, B is tile_shared_t [N, K] row-major (transposed tile),
// C is tile_shared_t [M, N] fp16 row-major.
// B's logical shape after tile_transpose is [K, N], but physically stored as [N, K] row-major.
#define tile_mma_smem(A, B, C) \
    do { \
        using _ShapeA = typename wp::remove_reference<decltype(A)>::type::Layout::Shape; \
        using _ShapeB_phys = typename wp::remove_reference<decltype(B)>::type::Layout::Shape; \
        using _ShapeC = typename wp::remove_reference<decltype(C)>::type::Layout::Shape; \
        constexpr int _M = _ShapeA::dim(0); \
        constexpr int _K = _ShapeA::dim(1); \
        constexpr int _N = _ShapeC::dim(1); \
        wp::tile_matmul_mma_to_smem<_M, _K, _N>( \
            reinterpret_cast<const half*>(A.data.ptr), _K, \
            reinterpret_cast<const half*>(B.data.ptr), _K, \
            reinterpret_cast<half*>(C.data.ptr), _N); \
        WP_TILE_SYNC(); \
    } while (0)

#define adj_tile_mma_smem(A, B, C, adj_A, adj_B, adj_C)

// tile_mma_acc_op: acc += A_smem @ B_smem using native PTX MMA with register accumulator
// A is tile_shared_t [M, K] fp16 row-major, B is tile_shared_t [K, N] fp16 row-major,
// acc is tile_mma_acc_t<M, N, NumWarps>.
// B needs transposed fragment loading (row-major to col-major for MMA).
#define tile_mma_acc_op(A, B, acc) \
    do { \
        using _ShapeA = typename wp::remove_reference<decltype(A)>::type::Layout::Shape; \
        constexpr int _M = _ShapeA::dim(0); \
        constexpr int _K = _ShapeA::dim(1); \
        constexpr int _N = (int)(sizeof(acc.data) / sizeof(float) / (wp::tile_mma_acc_t<_M, 8, (WP_TILE_BLOCK_DIM/32)>::tiles_per_warp)); \
        constexpr int _NumWarps = WP_TILE_BLOCK_DIM / 32; \
        wp::tile_matmul_mma_accumulate<_M, _K, _N, _NumWarps>( \
            reinterpret_cast<const half*>(A.data.ptr), _K, \
            reinterpret_cast<const half*>(B.data.ptr), _N, \
            acc); \
    } while (0)

#define adj_tile_mma_acc_op(A, B, acc, adj_A, adj_B, adj_acc)

// tile_mma_scale_rows: scale accumulator by per-row alpha from SMEM
#define tile_mma_scale_rows(acc, alpha_smem) \
    do { \
        acc.scale_rows(reinterpret_cast<const float*>(alpha_smem.data.ptr)); \
    } while (0)

#define adj_tile_mma_scale_rows(acc, alpha_smem, adj_acc, adj_alpha_smem)

// tile_mma_store_normalized: write accumulator to global memory with per-row normalization
#define tile_mma_store_normalized(acc, out_ptr, inv_l_smem, ld, m_offset) \
    do { \
        acc.store_to_global_f16_normalized( \
            reinterpret_cast<half*>(out_ptr), ld, \
            reinterpret_cast<const float*>(inv_l_smem.data.ptr), m_offset); \
    } while (0)

#define adj_tile_mma_store_normalized(acc, out_ptr, inv_l_smem, ld, m_offset, \
    adj_acc, adj_out_ptr, adj_inv_l_smem, adj_ld, adj_m_offset)

// tile_flash_attention_mma: fused flash attention kernel using native MMA
// Calls the C++ template function that does the entire flash attention loop.
// TILE_M_val, TILE_N_val, HEAD_DIM_val must be compile-time constants.
#define tile_flash_attention_mma(Q_ptr, K_ptr, V_ptr, O_ptr, sm_scale, seq_len, batch_heads, \
    num_q_blocks, num_k_blocks, batch_head, q_block_idx, TILE_M_val, TILE_N_val, HEAD_DIM_val) \
    do { \
        wp::flash_attention_mma_kernel<TILE_M_val, TILE_N_val, HEAD_DIM_val, (WP_TILE_BLOCK_DIM/32)>( \
            reinterpret_cast<const wp::half*>((Q_ptr).data), \
            reinterpret_cast<const wp::half*>((K_ptr).data), \
            reinterpret_cast<const wp::half*>((V_ptr).data), \
            reinterpret_cast<wp::half*>((O_ptr).data), \
            static_cast<float>(sm_scale), \
            static_cast<int>(seq_len), \
            static_cast<int>(batch_heads), \
            static_cast<int>(num_q_blocks), \
            static_cast<int>(num_k_blocks), \
            static_cast<int>(batch_head), \
            static_cast<int>(q_block_idx)); \
    } while (0)

#define adj_tile_flash_attention_mma(Q_ptr, K_ptr, V_ptr, O_ptr, sm_scale, seq_len, batch_heads, \
    num_q_blocks, num_k_blocks, batch_head, q_block_idx, TILE_M_val, TILE_N_val, HEAD_DIM_val, \
    adj_Q_ptr, adj_K_ptr, adj_V_ptr, adj_O_ptr, adj_sm_scale, adj_seq_len, adj_batch_heads, \
    adj_num_q_blocks, adj_num_k_blocks, adj_batch_head, adj_q_block_idx, adj_TILE_M_val, adj_TILE_N_val, adj_HEAD_DIM_val)

#endif  // !defined(__CUDA_ARCH__)

template <typename Fwd, typename TileA, typename TileL>
CUDA_CALLABLE TileL& tile_cholesky(Fwd fun_forward, TileA& A, TileL& L)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileL::Layout::Shape::N == 2, "Expected TileL::Layout::Shape::N == 2");

    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");
    static_assert(TileL::Layout::Shape::dim(0) == TileL::Layout::Shape::dim(1), "Expected TileL to be square");
    static_assert(
        TileA::Layout::Shape::dim(0) == TileL::Layout::Shape::dim(0), "Expected A and L to have the same number of rows"
    );
    static_assert(
        TileA::Layout::Shape::dim(1) == TileL::Layout::Shape::dim(1),
        "Expected A and L to have the same number of columns"
    );

    // Copy to L
    L = A;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky(A, L);

#else

    // TODO: for batched Cholesky, need one info per batch
    __shared__ int info[1];

    if (WP_TILE_THREAD_IDX == 0) {
        info[0] = 0;
    }

    // Call cholesky on L
    WP_TILE_SYNC();

    fun_forward(L.data.ptr, info);

    WP_TILE_SYNC();

    // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
    if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
        printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
    }
#endif

    // Zero-out the upper triangular part of L

    WP_PRAGMA_UNROLL
    for (int i = WP_TILE_THREAD_IDX; i < TileL::Layout::Size; i += WP_TILE_BLOCK_DIM) {
        auto c = TileL::Layout::coord_from_linear(i);

        if (c[0] < c[1])
            L.data(c) = 0.0;
    }

    WP_TILE_SYNC();

#endif

    return L;
}

template <typename Fwd, typename TileA> CUDA_CALLABLE void tile_cholesky_inplace(Fwd fun_forward, TileA& A)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky(A, A);

#else

    // TODO: for batched Cholesky, need one info per batch
    __shared__ int info[1];

    if (WP_TILE_THREAD_IDX == 0) {
        info[0] = 0;
    }

    // Call cholesky on A
    WP_TILE_SYNC();

    fun_forward(A.data.ptr, info);

    WP_TILE_SYNC();

    // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
    if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
        printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
    }
#endif

    // Zero-out the upper triangular part of L

    WP_PRAGMA_UNROLL
    for (int i = WP_TILE_THREAD_IDX; i < TileA::Layout::Size; i += WP_TILE_BLOCK_DIM) {
        auto c = TileA::Layout::coord_from_linear(i);

        if (c[0] < c[1])
            A.data(c) = 0.0;
    }

    WP_TILE_SYNC();

#endif
}

#define adj_tile_cholesky(function_name, A, L, adj_function_name, adj_A, adj_L, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_cholesky_inplace(function_name, A, adj_function_name, adj_A) \
     do { \
         assert(false); \
     } while (0)

template <typename Fwd, typename TileL, typename TileY, typename TileX>
TileX& tile_cholesky_solve(Fwd fun_forward, TileL& L, TileY& Y, TileX& X)
{
    // Copy y to x

    X = Y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky_solve(L, X, Y);

#else

    // Call cholesky solve on L & x

    WP_TILE_SYNC();

    fun_forward(L.data.ptr, X.data.ptr);

    WP_TILE_SYNC();

#endif

    return X;
}

template <typename Fwd, typename TileL, typename TileY>
void tile_cholesky_solve_inplace(Fwd fun_forward, TileL& L, TileY& Y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky_solve(L, Y, Y);

#else

    // Call cholesky solve on L & y
    fun_forward(L.data.ptr, Y.data.ptr);

    WP_TILE_SYNC();

#endif
}

#define adj_tile_cholesky_solve(function_name, L, Y, X, adj_function_name, adj_L, adj_Y, adj_X, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_cholesky_solve_inplace(function_name, L, Y, adj_function_name, adj_L, adj_Y) \
     do { \
         assert(false); \
     } while (0)


template <typename Fwd, typename TileL, typename TileY, typename TileZ>
TileZ& tile_lower_solve(Fwd fun_forward, TileL& L, TileY& y, TileZ& z)
{
    // Copy y to z
    z = y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky_forward_substitution(L, z, y);

#else

    // Call cholesky solve on L & z

    WP_TILE_SYNC();

    fun_forward(L.data.ptr, z.data.ptr);

    WP_TILE_SYNC();

#endif

    return z;
}

template <typename Fwd, typename TileL, typename TileY>
void tile_lower_solve_inplace(Fwd fun_forward, TileL& L, TileY& y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky_forward_substitution(L, y, y);

#else

    // Call cholesky solve on L & y

    WP_TILE_SYNC();

    fun_forward(L.data.ptr, y.data.ptr);

    WP_TILE_SYNC();

#endif
}

#define adj_tile_lower_solve(function_name, L, y, z, adj_function_name, adj_L, adj_y, adj_z, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_lower_solve_inplace(function_name, L, y, adj_function_name, adj_L, adj_y) \
     do { \
         assert(false); \
     } while (0)


template <typename Fwd, typename TileU, typename TileZ, typename TileX>
TileX& tile_upper_solve(Fwd fun_forward, TileU& U, TileZ& z, TileX& x)
{
    // Copy z to x
    x = z;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    auto L = tile_transpose(U);
    partitioned_gemm::scalar_cholesky_back_substitution(L, x);

#else

    // Call cholesky solve on U & x

    WP_TILE_SYNC();

    fun_forward(U.data.ptr, x.data.ptr);

    WP_TILE_SYNC();

#endif

    return x;
}

template <typename Fwd, typename TileU, typename TileZ>
void tile_upper_solve_inplace(Fwd fun_forward, TileU& U, TileZ& z)
{

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    auto L = tile_transpose(U);
    partitioned_gemm::scalar_cholesky_back_substitution(L, z);

#else

    // Call cholesky solve on U & z

    WP_TILE_SYNC();

    fun_forward(U.data.ptr, z.data.ptr);

    WP_TILE_SYNC();

#endif
}

#define adj_tile_upper_solve(function_name, U, z, x, adj_function_name, adj_U, adj_z, adj_x, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_upper_solve_inplace(function_name, U, z, adj_function_name, adj_U, adj_z) \
     do { \
         assert(false); \
     } while (0)


template <typename Tile> inline CUDA_CALLABLE auto tile_transpose(Tile& t)
{
    static_assert(Tile::Layout::Shape::N == 2, "Expected Tile::Layout::Shape::N == 2");

    // alias incoming tile
    constexpr int M = Tile::Layout::Shape::dim(0);
    constexpr int N = Tile::Layout::Shape::dim(1);

    constexpr int StrideM = Tile::Layout::Stride::dim(0);
    constexpr int StrideN = Tile::Layout::Stride::dim(1);

    return tile_shared_t<
        typename Tile::Type, tile_layout_strided_t<tile_shape_t<N, M>, tile_stride_t<StrideN, StrideM>>, false>(
        t.data.ptr, t.grad.ptr
    );
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_transpose(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    auto a = tile_transpose(adj_ret);
    auto& b = adj_t;

    adj_t.assign(tile_add(a, b));
}

template <int N, int StrideN, typename Tile> inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<N>, tile_stride_t<StrideN>>, false>(
        t.data.ptr, t.grad.ptr
    );
}

template <int M, int N, int StrideM, int StrideN, typename Tile> inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<
        typename Tile::Type, tile_layout_strided_t<tile_shape_t<M, N>, tile_stride_t<StrideM, StrideN>>, false>(
        t.data.ptr, t.grad.ptr
    );
}

template <int M, int N, int O, int StrideM, int StrideN, int StrideO, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<
        typename Tile::Type, tile_layout_strided_t<tile_shape_t<M, N, O>, tile_stride_t<StrideM, StrideN, StrideO>>,
        false>(t.data.ptr, t.grad.ptr);
}

template <int M, int N, int O, int P, int StrideM, int StrideN, int StrideO, int StrideP, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<
        typename Tile::Type,
        tile_layout_strided_t<tile_shape_t<M, N, O, P>, tile_stride_t<StrideM, StrideN, StrideO, StrideP>>, false>(
        t.data.ptr, t.grad.ptr
    );
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_broadcast(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // nop, since memory is aliased, grads already accumulated
}


template <typename ReturnTile, typename Tile, typename... Indices>
inline CUDA_CALLABLE auto tile_view(Tile& t, Indices... indices)
{
    auto c = tile_coord(indices...);

    // return new tile with same strides
    typename Tile::Type* data_ptr = &t.data(c);
    typename Tile::Type* grad_ptr = nullptr;

    if (t.grad.ptr)
        grad_ptr = &t.grad(c);

    return ReturnTile(data_ptr, grad_ptr);
}


template <typename ReturnTile, typename Tile> inline CUDA_CALLABLE auto tile_squeeze(Tile& t)
{
    // ReturnTile layout is set in builtins.py
    typename Tile::Type* data_ptr = t.data.ptr;
    typename Tile::Type* grad_ptr = nullptr;

    if (t.grad.ptr)
        grad_ptr = t.grad.ptr;

    return ReturnTile(data_ptr, grad_ptr);
}

template <typename Tile, typename AdjTile, typename AdjReturnTile>
inline CUDA_CALLABLE void adj_tile_squeeze(Tile& t, AdjTile& adj_t, AdjReturnTile& adj_ret)
{
    // nop, since memory is aliased, grads already accumulated
}


template <typename ReturnTile, typename Tile> inline CUDA_CALLABLE auto tile_reshape(Tile& t)
{
    // ReturnTile layout is set in builtins.py
    typename Tile::Type* data_ptr = t.data.ptr;
    typename Tile::Type* grad_ptr = nullptr;

    if (t.grad.ptr)
        grad_ptr = t.grad.ptr;

    return ReturnTile(data_ptr, grad_ptr);
}

template <typename Tile, typename AdjTile, typename AdjReturnTile>
inline CUDA_CALLABLE void adj_tile_reshape(Tile& t, AdjTile& adj_t, AdjReturnTile& adj_ret)
{
    // nop, since memory is aliased, grads already accumulated
}


template <typename ReturnTile, typename Tile> inline CUDA_CALLABLE auto tile_astype(Tile& t)
{
    // verify shapes and sizes are compatible
    using ShapeIn = typename Tile::Layout::Shape;
    using ShapeOut = typename ReturnTile::Layout::Shape;

    static_assert(ShapeIn::N == ShapeOut::N, "Tile shapes must match for data type casting");
    static_assert(ShapeIn::size() == ShapeOut::size(), "Tile sizes must match for data type casting");

    // work with register tiles for type casting
    auto t_reg = t.copy_to_register();
    auto result = tile_register_like<ReturnTile>();

    using Layout = typename decltype(result)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);

        if (!Layout::valid(linear))
            break;

        result.data[i] = static_cast<typename ReturnTile::Type>(t_reg.data[i]);
    }

    return result;
}

template <typename Tile, typename AdjTile, typename AdjReturnTile>
inline CUDA_CALLABLE void adj_tile_astype(Tile& t, AdjTile& adj_t, AdjReturnTile& adj_ret)
{
    // gradients only flow between float conversions
    if constexpr ((is_same<typename AdjTile::Type, wp::float16>::value
                   || is_same<typename AdjTile::Type, wp::float32>::value
                   || is_same<typename AdjTile::Type, wp::float64>::value)
                  && (is_same<typename AdjReturnTile::Type, wp::float16>::value
                      || is_same<typename AdjReturnTile::Type, wp::float32>::value
                      || is_same<typename AdjReturnTile::Type, wp::float64>::value)) {
        auto adj_ret_reg = adj_ret.grad_to_register();
        auto adj_t_reg = tile_register_like<AdjTile>();

        using Layout = typename decltype(adj_t_reg)::Layout;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            adj_t_reg.data[i] += static_cast<typename AdjTile::Type>(adj_ret_reg.data[i]);
        }

        adj_t.grad_add(adj_t_reg);
    }
}


template <typename TileA, typename Scalar> inline CUDA_CALLABLE void assign(TileA& dest, int i, const Scalar& src)
{
    dest.data(tile_coord(i)) = src;
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, const Scalar& src)
{
    if constexpr (is_vector<typename TileA::Type>::value) {
        dest.data(tile_coord(i))[j] = src;
    } else {
        dest.data(tile_coord(i, j)) = src;
    }
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, const Scalar& src)
{
    if constexpr (is_vector<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j))[k] = src;
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        dest.data(tile_coord(i)).data[j][k] = src;
    } else {
        dest.data(tile_coord(i, j, k)) = src;
    }
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, int l, const Scalar& src)
{
    if constexpr (is_vector<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j, k))[l] = src;
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j)).data[k][l] = src;
    } else {
        dest.data(tile_coord(i, j, k, l)) = src;
    }
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, int l, int m, const Scalar& src)
{
    if constexpr (is_vector<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j, k, l))[m] = src;
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j, k)).data[l][m] = src;
    } else {
        static_assert(
            always_false<TileA>::value,
            "assign with 5 indices requires a tile of vectors (4D tile) or matrices (3D tile)"
        );
    }
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, int l, int m, int n, const Scalar& src)
{
    if constexpr (is_matrix<typename TileA::Type>::value) {
        dest.data(tile_coord(i, j, k, l)).data[m][n] = src;
    } else {
        static_assert(always_false<TileA>::value, "assign with 6 indices requires a tile of matrices (4D tile)");
    }
    WP_TILE_SYNC();
}


template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void
adj_assign(TileA& dest, int i, const Scalar& src, AdjTileA& adj_dest, int adj_i, Scalar& adj_src)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    adj_src += dest.grad(tile_coord(i));
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void
adj_assign(TileA& dest, int i, int j, const Scalar& src, AdjTileA& adj_dest, int adj_i, int adj_j, Scalar& adj_src)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    if constexpr (is_vector<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i))[j];
    } else {
        adj_src += dest.grad(tile_coord(i, j));
    }
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(
    TileA& dest,
    int i,
    int j,
    int k,
    const Scalar& src,
    AdjTileA& adj_dest,
    int adj_i,
    int adj_j,
    int adj_k,
    Scalar& adj_src
)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    if constexpr (is_vector<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j))[k];
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i)).data[j][k];
    } else {
        adj_src += dest.grad(tile_coord(i, j, k));
    }
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(
    TileA& dest,
    int i,
    int j,
    int k,
    int l,
    const Scalar& src,
    AdjTileA& adj_dest,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    Scalar& adj_src
)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    if constexpr (is_vector<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j, k))[l];
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j)).data[k][l];
    } else {
        adj_src += dest.grad(tile_coord(i, j, k, l));
    }
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(
    TileA& dest,
    int i,
    int j,
    int k,
    int l,
    int m,
    const Scalar& src,
    AdjTileA& adj_dest,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    int adj_m,
    Scalar& adj_src
)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    if constexpr (is_vector<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j, k, l))[m];
    } else if constexpr (is_matrix<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j, k)).data[l][m];
    } else {
        static_assert(
            always_false<TileA>::value,
            "adj_assign with 5 indices requires a tile of vectors (4D tile) or matrices (3D tile)"
        );
    }
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(
    TileA& dest,
    int i,
    int j,
    int k,
    int l,
    int m,
    int n,
    const Scalar& src,
    AdjTileA& adj_dest,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    int adj_m,
    int adj_n,
    Scalar& adj_src
)
{
    if (dest.grad.ptr == nullptr) {
        return;
    }

    if constexpr (is_matrix<typename TileA::Type>::value) {
        adj_src += dest.grad(tile_coord(i, j, k, l)).data[m][n];
    } else {
        static_assert(always_false<TileA>::value, "adj_assign with 6 indices requires a tile of matrices (4D tile)");
    }
}

// Specialization for register→shared tile assignment (whole tile copy, zero offset)
// This uses the tile_shared_t::assign() method which correctly handles register tile data access
template <typename T, typename SharedLayout, bool Owner, typename RegT, typename RegLayout>
inline CUDA_CALLABLE void tile_assign(
    tile_shared_t<T, SharedLayout, Owner>& dest,
    const tile_register_t<RegT, RegLayout>& src,
    const tile_coord_t<SharedLayout::Shape::N>& offset)
{
    // Check if offset is zero - if so, use the optimized assign() method
    bool is_zero_offset = true;
    for (int i = 0; i < SharedLayout::Shape::N; ++i) {
        if (offset[i] != 0) {
            is_zero_offset = false;
            break;
        }
    }

    if (is_zero_offset) {
        // Use the member assign() which correctly handles register tile data access
        dest.assign(src);
    } else {
        // Non-zero offset: need to do manual copy with offset
        // This iterates over register elements and writes to shared memory at offset
        using Layout = RegLayout;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            const int linear = Layout::linear_from_register(i);

            if (!Layout::valid(linear))
                break;

            auto c = Layout::coord_from_linear(linear);
            dest.data(c + offset) = src.data[i];
        }

        WP_TILE_SYNC();
    }
}

template <typename TileA, typename TileB, typename Coord>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, const Coord& offset)
{
    using Layout = typename TileB::Layout;

    for (int t = WP_TILE_THREAD_IDX; t < Layout::Size; t += WP_TILE_BLOCK_DIM) {
        auto c = Layout::coord_from_linear(t);
        dest.data(c + offset) = src.data(c);
    }

    WP_TILE_SYNC();
}

// Specialization for adj_tile_assign: register→shared backward pass
// Accumulates gradients from shared tile destination back to register tile source
template <typename T, typename SharedLayout, bool Owner, typename RegT, typename RegLayout,
          typename AdjT, typename AdjSharedLayout, bool AdjOwner, typename AdjRegT, typename AdjRegLayout,
          typename Coord, typename AdjCoord>
inline CUDA_CALLABLE void adj_tile_assign(
    tile_shared_t<T, SharedLayout, Owner>& dest,
    const tile_register_t<RegT, RegLayout>& src,
    Coord offset,
    tile_shared_t<AdjT, AdjSharedLayout, AdjOwner>& adj_dest,
    tile_register_t<AdjRegT, AdjRegLayout>& adj_src,
    AdjCoord adj_offset)
{
    // Check if offset is zero
    bool is_zero_offset = true;
    for (int i = 0; i < SharedLayout::Shape::N; ++i) {
        if (offset[i] != 0) {
            is_zero_offset = false;
            break;
        }
    }

    if (is_zero_offset) {
        // Use grad_add which correctly handles register tile gradient accumulation
        adj_dest.grad_add(adj_src);
    } else {
        // Non-zero offset: iterate over register elements
        using Layout = RegLayout;

        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            const int linear = Layout::linear_from_register(i);

            if (!Layout::valid(linear))
                break;

            auto c = Layout::coord_from_linear(linear);
            adj_src.data[i] += adj_dest.grad(c + offset);
        }

        WP_TILE_SYNC();
    }
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename Coord, typename AdjCoord>
inline CUDA_CALLABLE void
adj_tile_assign(TileA& dest, TileB& src, Coord offset, AdjTileA& adj_dest, AdjTileB& adj_src, AdjCoord adj_offset)
{
    using Layout = typename TileB::Layout;

    for (int t = WP_TILE_THREAD_IDX; t < Layout::Size; t += WP_TILE_BLOCK_DIM) {
        auto c = Layout::coord_from_linear(t);
        src.grad(c) += dest.grad(c + offset);
    }

    WP_TILE_SYNC();
}


// codegen entry points, which emit calls like `tile_assign(dest, src, i, j, k)`
// a better approach here would be for codegen to just directly generate `tile_assign(dest, src, tile_coord(i, j, k))`
// i.e.: call the above implementation methods directly, then we could remove these overloads
// Note: For register→shared assignment, use tile_assign_to_shared() which handles the template matching correctly.
template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i)
{
    tile_assign(dest, src, tile_coord(i));
}
template <typename TileA, typename TileB> inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j)
{
    tile_assign(dest, src, tile_coord(i, j));
}
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j, int k)
{
    tile_assign(dest, src, tile_coord(i, j, k));
}
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j, int k, int l)
{
    tile_assign(dest, src, tile_coord(i, j, k, l));
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, int i, AdjTileA& adj_dest, AdjTileB& adj_src, int)
{
    adj_tile_assign(dest, src, tile_coord(i), adj_dest, adj_src, tile_coord(0));
}
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void
adj_tile_assign(TileA& dest, TileB& src, int i, int j, AdjTileA& adj_dest, AdjTileB& adj_src, int, int)
{
    adj_tile_assign(dest, src, tile_coord(i, j), adj_dest, adj_src, tile_coord(0));
}
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void
adj_tile_assign(TileA& dest, TileB& src, int i, int j, int k, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int)
{
    adj_tile_assign(dest, src, tile_coord(i, j, k), adj_dest, adj_src, tile_coord(0));
}
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(
    TileA& dest, TileB& src, int i, int j, int k, int l, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int, int
)
{
    adj_tile_assign(dest, src, tile_coord(i, j, k, l), adj_dest, adj_src, tile_coord(0));
}


// tile_assign_to_shared: explicit register→shared tile assignment
// These use generic template parameters because the storage mutation bug in Python
// may have incorrectly labeled register tiles as shared tiles. We use static_cast
// to force the const-qualification that enables the register→shared specialization.
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign_to_shared(TileA& dest, TileB& src, int i)
{
    tile_assign(dest, static_cast<const TileB&>(src), tile_coord(i));
}

template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign_to_shared(TileA& dest, TileB& src, int i, int j)
{
    tile_assign(dest, static_cast<const TileB&>(src), tile_coord(i, j));
}

template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign_to_shared(TileA& dest, TileB& src, int i, int j, int k)
{
    tile_assign(dest, static_cast<const TileB&>(src), tile_coord(i, j, k));
}

template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign_to_shared(TileA& dest, TileB& src, int i, int j, int k, int l)
{
    tile_assign(dest, static_cast<const TileB&>(src), tile_coord(i, j, k, l));
}

// adj_tile_assign_to_shared: backward pass for register→shared assignment
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign_to_shared(TileA& dest, TileB& src, int i, AdjTileA& adj_dest, AdjTileB& adj_src, int)
{
    adj_tile_assign(dest, src, tile_coord(i), adj_dest, adj_src, tile_coord(0));
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign_to_shared(TileA& dest, TileB& src, int i, int j, AdjTileA& adj_dest, AdjTileB& adj_src, int, int)
{
    adj_tile_assign(dest, src, tile_coord(i, j), adj_dest, adj_src, tile_coord(0));
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign_to_shared(TileA& dest, TileB& src, int i, int j, int k, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int)
{
    adj_tile_assign(dest, src, tile_coord(i, j, k), adj_dest, adj_src, tile_coord(0));
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign_to_shared(TileA& dest, TileB& src, int i, int j, int k, int l, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int, int)
{
    adj_tile_assign(dest, src, tile_coord(i, j, k, l), adj_dest, adj_src, tile_coord(0));
}


template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE TileC& tile_diag_add(TileA& a, TileB& b, TileC& c)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::dim(0) == ShapeA::dim(1), "Expected ShapeA::dim(0) == ShapeA::dim(1)");
    static_assert(ShapeB::dim(0) == ShapeA::dim(0), "Expected ShapeB::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeC::dim(1), "Expected ShapeC::dim(0) == ShapeC::dim(1)");

    c = a;

    for (int t = WP_TILE_THREAD_IDX; t < ShapeA::dim(0); t += WP_TILE_BLOCK_DIM) {
        c.data(tile_coord(t, t)) += b.data(tile_coord(t));
    }

    WP_TILE_SYNC();

    return c;
}

template <typename TileA, typename TileB, typename TileC, typename AdjTileA, typename AdjTileB, typename AdjTileC>
inline CUDA_CALLABLE void
adj_tile_diag_add(TileA& a, TileB& b, TileC& c, AdjTileA& adj_a, AdjTileB& adj_b, AdjTileC& adj_c, AdjTileC& adj_ret)
{
}


}  // namespace wp


#ifdef __clang__
#pragma clang diagnostic pop
#endif
