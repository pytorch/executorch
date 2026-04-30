/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/backends/portable/runtime/metal_v2/OpUtils.h>

#include <algorithm>  // std::max
#include <cstdint>

#import <Metal/Metal.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Shared host-side helpers used by every metal_v2 matmul-family op
// (MatMulOp / AddMMOp / BAddBMMOp / BatchedMatMulOp). Promoted from the old
// anonymous namespace inside MatMulOp.mm into this header so each op file
// can include them. All marked inline / constexpr to keep ODR-safe.

struct MatMulThresholds {
  int simdMNK;       // min M,N,K to pick Simd over Tiled/Naive
  int gemvMK;        // min M (or N for gemv_t) to use the simdgroup gemv path
};

constexpr MatMulThresholds thresholdsForTier(DeviceTier tier) {
  switch (tier) {
    case DeviceTier::Phone:    return {32, 16};
    case DeviceTier::MacUltra: return {64, 32};
    case DeviceTier::MacBase:
    default:                   return {48, 24};
  }
}

// MLX-style fp32 NAX gating. Mirrors MLX's env::enable_tf32(). When set,
// fp32 matmuls become eligible for NAX (cooperative-tensor matmul on
// Apple9+ with truncated mantissa). Trades 6 bits of fp32 mantissa for
// substantial speedup (~2-3x on prefill matmul / SDPA shapes; see
// metal_v2_mlx_sdpa_vendor plan A/B numbers).
// Resolution order:
//   1. Env var `MLX_ENABLE_TF32` if set — takes precedence.
//      "1" or "true" → enabled.  "0" or anything else → disabled.
//   2. Compile-time default `EXECUTORCH_METAL_TF32_DEFAULT_ENABLED`
//      (set via CMake option `EXECUTORCH_METAL_ENABLE_TF32_DEFAULT`,
//       which defaults ON for the v2 backend).
// Cached on first call (the resolution doesn't change at runtime).
inline bool tf32Enabled() {
  static const bool enabled = []() {
    const char* env = getenv("MLX_ENABLE_TF32");
    if (env != nullptr) {
      // Explicit env-var setting wins regardless of compile-time default.
      return strcmp(env, "1") == 0 || strcmp(env, "true") == 0;
    }
#if defined(EXECUTORCH_METAL_TF32_DEFAULT_ENABLED) && (EXECUTORCH_METAL_TF32_DEFAULT_ENABLED + 0 != 0)
    return true;
#else
    return false;
#endif
  }();
  return enabled;
}


// dtype family classifier matching MLX's GEMM_TPARAM_MACRO usage.
// MLX branches between complex64, float32, and "half/bfloat" (everything
// else). complex64 is only special-cased on small devices ('g'/'p'); we
// don't ship complex64 tiles so map it to fp32 here.
inline bool isFloat32(executorch::aten::ScalarType dtype) {
  return dtype == executorch::aten::ScalarType::Float;
}

// Build the (align_M, align_N, align_K, use_out_source, do_axpby) function-
// constant tuple kept here as historical context — the legacy
// matmul_simd_addmm_t kernel was DELETED in the MLX-JIT cleanup. Active
// FC-tuple builders for MLX live in MatMulMlxJit.h
// (mlx_jit_helpers::makeMlxFusedFCs uses MLX's slot numbering 10/100/110/
// 200/201/202; the legacy 0/1/2/3/4 slots used here are no longer
// reachable from any caller). Removed entirely.

//===----------------------------------------------------------------------===//
// Tile picker for the SIMD-MMA matmul family. Owned by AddMMOp /
// BAddBMMOp / MatMulOp routing logic; the chosen tile is then handed to
// MatMulMlxJit.h's dispatchGemmViaMlxJit which assembles the per-shape
// MLX `gemm` instantiation.
// `transposed_NN_only` is true for layouts where only the (64,64,16,2,2)
// tile is supported by the AddMM family's NT/TN paths.
// `batch` is grid.z; pass 1 for non-batched ops.
//===----------------------------------------------------------------------===//

struct AddmmTilePick {
  const char* tile_suffix;  // e.g. "64_32_32_2_2_" — kept for logging only
  int BM, BN, BK;
  uvec3 grid;               // (ceildiv(N,BN), ceildiv(M,BM), batch)
  // Threads per threadgroup. Computed from (WM*WN)*32 — most tiles use
  // 4 simdgroups (128 threads); the (1,2) tile uses 2 (64 threads).
  int block_threads = 128;
};

// MLX-faithful tile-shape enum. Mirrors the (BM, BN, BK, WM, WN) shapes
// MLX's GEMM_TPARAM_MACRO selects in matmul.cpp:88-169. We don't have
// MLX's full 6-tile catalog instantiated; substitutions are documented
// per-enum.
enum class MatMulTileShape {
  // (64, 64, 16, 2, 2) — MLX default. Our `Simd`.
  S_64_64_16_2_2,
  // (64, 32, 32, 2, 2) — MLX `nt` and `'d' small fp32 nn`. Our `Simd_BN32`.
  S_64_32_32_2_2,
  // (64, 64, 16, 1, 2) — MLX `'d' large modest-K half/bf` & `'d' small half/bf nn`
  // & `'g'/'p' half/bf non-nt`. Our `Simd_W12`.
  S_64_64_16_1_2,
  // (32, 64, 16, 1, 2) — MLX `'d' large nn-deep-K half/bf` & `'d' small fp32 nt`.
  // Real tile (was substituted by Simd_M32 before Phase A).
  S_32_64_16_1_2,
};

inline AddmmTilePick pickAddmmTile(
    int32_t M, int32_t N, int32_t K,
    int32_t batch = 1,
    bool transposed_NN_only = false,
    // New args (all defaulted to stay source-compatible):
    executorch::aten::ScalarType dtype =
        executorch::aten::ScalarType::Float,
    DeviceTier tier = DeviceTier::MacBase) {
  AddmmTilePick t{};
  if (transposed_NN_only) {
    // NT/TN: only 64x64x16 has tile instantiations.
    t.tile_suffix = "64_64_16_2_2_";
    t.BM = 64; t.BN = 64; t.BK = 16;
    t.grid = uvec3((N + 63) / 64, (M + 63) / 64, batch);
    return t;
  }
  if (M < 64 || N < 64 || K < 16) {
    // Below the simd-MMA path's lower bound. Fall through to Simd_M32 for
    // skinny-M cases; tiny shapes use the 64x64 kernel with bounds-safe
    // store (the rest of the original ladder).
    if (M >= 2 && N >= 64 && K >= 16) {
      t.tile_suffix = "32_64_32_1_4_";
      t.BM = 32; t.BN = 64; t.BK = 32;
      t.grid = uvec3((N + 63) / 64, (M + 31) / 32, batch);
    } else {
      t.tile_suffix = "64_64_16_2_2_";
      t.BM = 64; t.BN = 64; t.BK = 16;
      t.grid = uvec3((N + 63) / 64, (M + 63) / 64, batch);
    }
    return t;
  }

  // MLX's GEMM_TPARAM_MACRO transcribed verbatim from
  // mlx/backend/metal/matmul.cpp:88-169. Device-class mapping:
  //   'g'/'p' (small Apple Silicon, iPhone/iPad)  → Phone
  //   'd'     (Mac Max/Ultra)                     → MacUltra
  //   else    (medium: M-base/Pro, 's', 'c')      → MacBase (defaults only)
  // All callers in this file path are NN layout (transpose_a=false,
  // transpose_b=false) — NT/TN go through the transposed_NN_only branch
  // above. So MLX's `nt = (!transpose_a && transpose_b)` is always false
  // here and the `nt` branches are dead-coded out of the transcription.
  const bool fp32 = isFloat32(dtype);

  // MLX defaults (matmul.cpp:398-399): (64, 64, 16, 2, 2).
  MatMulTileShape shape = MatMulTileShape::S_64_64_16_2_2;

  if (tier == DeviceTier::Phone) {
    // matmul.cpp:89-108  ('g'/'p' branch)
    if (!fp32) {
      // half/bf, nn → matmul.cpp:103-107
      shape = MatMulTileShape::S_64_64_16_1_2;
    }
    // else fp32 nn → keep defaults (S_64_64_16_2_2).
  } else if (tier == DeviceTier::MacUltra) {
    // matmul.cpp:109-162  ('d' branch)
    const int64_t area = int64_t(batch) * int64_t(M) * int64_t(N);
    if (area >= (int64_t(1) << 20)) {
      // matmul.cpp:110  large matmul
      if (!fp32) {
        // matmul.cpp:111
        if (2 * std::max(M, N) > K) {
          // matmul.cpp:113-117  reasonable K
          shape = MatMulTileShape::S_64_64_16_1_2;
        } else {
          // matmul.cpp:124-129  nn with large K
          shape = MatMulTileShape::S_32_64_16_1_2;
        }
      }
      // else fp32 large 'd' → defaults (matmul.cpp:131).
    } else {
      // matmul.cpp:132  smaller matmul
      if (!fp32) {
        // matmul.cpp:140-145  half/bf nn
        shape = MatMulTileShape::S_64_64_16_1_2;
      } else {
        // matmul.cpp:154-159  fp32 nn
        shape = MatMulTileShape::S_64_32_32_2_2;
      }
    }
  }
  // else: DeviceTier::MacBase (medium) → matmul.cpp:163-168 defaults.

  switch (shape) {
    case MatMulTileShape::S_64_64_16_2_2:
      t.tile_suffix = "64_64_16_2_2_";
      t.BM = 64; t.BN = 64; t.BK = 16;
      t.grid = uvec3((N + 63) / 64, (M + 63) / 64, batch);
      t.block_threads = 128;
      break;
    case MatMulTileShape::S_64_32_32_2_2:
      t.tile_suffix = "64_32_32_2_2_";
      t.BM = 64; t.BN = 32; t.BK = 32;
      t.grid = uvec3((N + 31) / 32, (M + 63) / 64, batch);
      t.block_threads = 128;
      break;
    case MatMulTileShape::S_64_64_16_1_2:
      t.tile_suffix = "64_64_16_1_2_";
      t.BM = 64; t.BN = 64; t.BK = 16;
      t.grid = uvec3((N + 63) / 64, (M + 63) / 64, batch);
      t.block_threads = 64;  // WM*WN*32 = 1*2*32
      break;
    case MatMulTileShape::S_32_64_16_1_2:
      // Phase A: real (32, 64, 16, 1, 2) tile (BM=32, BN=64, BK=16, 1×2
      // simd layout). 64 threads/tg.
      t.tile_suffix = "32_64_16_1_2_";
      t.BM = 32; t.BN = 64; t.BK = 16;
      t.grid = uvec3((N + 63) / 64, (M + 31) / 32, batch);
      t.block_threads = 64;  // WM*WN*32 = 1*2*32
      break;
  }
  return t;
}

// Block-swizzle log heuristic — for grids large enough that L2 contention
// matters, group threadgroups into a 4x4 cluster (log2=2). Otherwise no
// swizzle (log2=0). Used by every simd-MMA dispatch.
inline int32_t pickSwizzleLog(uvec3 grid) {
  return (grid.y >= 4 && grid.x >= 4) ? 2 : 0;
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
