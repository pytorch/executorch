/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared Metal-shader-side helpers for cooperative tile loading from
// device memory into threadgroup memory. Used by GEMM-style kernels
// (matmul_simd, matmul_nt, matmul_tn, conv2d, etc).
// Mirrors the load loops in mlx/backend/metal/kernels/steel/gemm/loader.h.
// Usage from a host .mm:
//   #include <executorch/backends/metal/kernels/TileLoad.h>
//   const char* MyOp::kernelSource() const {
//     static const std::string source = std::string(kTileLoadMetalSource) + R"(
//       // ... your kernel using cooperativeLoadTileVec4 ...
//     )";
//     return source.c_str();
//   }

namespace executorch {
namespace backends {
namespace metal_v2 {

inline constexpr const char* kTileLoadMetalSource = R"METAL(
//===----------------------------------------------------------------------===//
// cooperativeLoadTileVec4
// Cooperatively load a ROWS x COLS tile from row-major device memory `src`
// (full tensor of size srcRows x srcCols, row stride `srcStride`) into a
// padded threadgroup-memory tile `smem` of shape ROWS x SMEM_STRIDE.
// The tile origin in `src` is (baseRow, baseCol). NUM_THREADS threads
// (typically a full threadgroup of 128) cooperate; each thread loads
// ceil(ROWS*COLS/4 / NUM_THREADS) vec<T,4> chunks.
// Vectorized (vec<T,4>) load is used when the 4-element column block is
// fully in bounds; scalar fallback handles the boundary tile.
// Constraints:
//   - COLS must be a multiple of 4
//   - SMEM_STRIDE >= COLS (the extra columns are usually padding to avoid
//     bank conflicts -- e.g. SMEM_STRIDE = COLS + 4)
//   - NUM_THREADS should evenly divide ROWS*COLS/4 for best efficiency.
//===----------------------------------------------------------------------===//

template <typename T, int ROWS, int COLS, int SMEM_STRIDE,
          int NUM_THREADS = 128>
inline void cooperativeLoadTileVec4(
    threadgroup T (&smem)[ROWS][SMEM_STRIDE],
    device const T* src,
    int srcRows, int srcCols, int srcStride,
    int baseRow, int baseCol,
    uint tid) {

  static_assert(COLS % 4 == 0, "COLS must be a multiple of 4");
  constexpr int VECS_PER_ROW = COLS / 4;
  constexpr int TOTAL_VECS = ROWS * VECS_PER_ROW;

  for (int v = int(tid); v < TOTAL_VECS; v += NUM_THREADS) {
    int row = v / VECS_PER_ROW;
    int col4 = v % VECS_PER_ROW;
    int gRow = baseRow + row;
    int gCol = baseCol + col4 * 4;

    if (gRow < srcRows && gCol + 3 < srcCols) {
      // Aligned 4-wide fast path
      metal::vec<T, 4> vv = *((device metal::vec<T, 4>*)(src + gRow * srcStride + gCol));
      smem[row][col4 * 4    ] = vv[0];
      smem[row][col4 * 4 + 1] = vv[1];
      smem[row][col4 * 4 + 2] = vv[2];
      smem[row][col4 * 4 + 3] = vv[3];
    } else {
      // Edge: scalar fallback with per-element bounds check
      for (int d = 0; d < 4; d++) {
        int c = col4 * 4 + d;
        smem[row][c] = (gRow < srcRows && (baseCol + c) < srcCols)
                       ? src[gRow * srcStride + baseCol + c]
                       : T(0);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// cooperativeLoadTileTransposedVec4
// Same as cooperativeLoadTileVec4 but loads a logical ROWS x COLS tile from
// a PHYSICALLY TRANSPOSED source: src is stored as [srcCols x srcRows] (i.e.
// src[c * srcStride + r] = logical(r, c)). srcStride is the physical row
// stride (in elements) of src, which equals the logical row count.
// Used by matmul_nt (loads B which is logically [K, N] but stored [N, K])
// and matmul_tn (loads A which is logically [M, K] but stored [K, M]).
// Vec4 coalescing: we vectorize along the PHYSICAL row direction (= logical
// row direction), so each thread loads 4 logical rows for one logical col.
// This requires ROWS to be a multiple of 4.
//===----------------------------------------------------------------------===//

template <typename T, int ROWS, int COLS, int SMEM_STRIDE,
          int NUM_THREADS = 128>
inline void cooperativeLoadTileTransposedVec4(
    threadgroup T (&smem)[ROWS][SMEM_STRIDE],
    device const T* src,
    int srcRows, int srcCols, int srcStride,
    int baseRow, int baseCol,
    uint tid) {

  static_assert(ROWS % 4 == 0, "ROWS must be a multiple of 4 for transposed vec4");
  constexpr int VECS_PER_COL = ROWS / 4;
  constexpr int TOTAL_VECS = VECS_PER_COL * COLS;

  for (int v = int(tid); v < TOTAL_VECS; v += NUM_THREADS) {
    int col   = v / VECS_PER_COL;       // logical col within tile
    int row4  = v % VECS_PER_COL;       // group of 4 logical rows
    int gRow  = baseRow + row4 * 4;     // first logical row in group
    int gCol  = baseCol + col;          // logical col in src

    // Physical: src[gCol * srcStride + gRow .. gRow+3]
    if (gCol < srcCols && gRow + 3 < srcRows) {
      metal::vec<T, 4> vv = *((device metal::vec<T, 4>*)(src + gCol * srcStride + gRow));
      smem[row4 * 4    ][col] = vv[0];
      smem[row4 * 4 + 1][col] = vv[1];
      smem[row4 * 4 + 2][col] = vv[2];
      smem[row4 * 4 + 3][col] = vv[3];
    } else {
      for (int d = 0; d < 4; d++) {
        int r = row4 * 4 + d;
        smem[r][col] = (gCol < srcCols && (baseRow + r) < srcRows)
                       ? src[gCol * srcStride + baseRow + r]
                       : T(0);
      }
    }
  }
}

)METAL";

} // namespace metal_v2
} // namespace backends
} // namespace executorch
