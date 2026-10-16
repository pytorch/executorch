/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/CoopmatFamily.h>

namespace vkcompute {

namespace {

bool has_shape(
    const CoopmatDeviceKey& device,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const CoopmatComponentType ab,
    const CoopmatComponentType result) {
  for (const CoopmatShape& s : device.coopmat_shapes) {
    if (s.m == m && s.n == n && s.k == k && s.ab == ab && s.result == result) {
      return true;
    }
  }
  return false;
}

bool subgroup_size_available(
    const CoopmatDeviceKey& device,
    const uint32_t size) {
  return device.subgroup_size == size ||
      (device.subgroup_size_control && size >= device.min_subgroup_size &&
       size <= device.max_subgroup_size);
}

//
// Intel Xe2 rows. MMA 8x16x16 / 8x16x32 (the only fp16 / int8 cooperative
// matrix shapes Xe2 exposes), subgroup 16.
//

// Weight-heavy projections run their workgroups in blocks of N tiles.
bool xe2_weight_heavy(const WmmaShape& s) {
  return s.K * s.N >= 24000000;
}

const std::vector<WmmaRow>& xe2_rows() {
  static const std::vector<WmmaRow> rows = {
      // 4w, texture3d: banded drain, 8 x 2 subgroups.
      {WmmaOp::kQ4gsw,
       "linear_q4gsw_wmma_t128x128k16g82s16m8flib",
       {128, 128, 16, 8, 2, 16, 8, false},
       kWmmaTex3dTex2d,
       nullptr},
      // 4w, buffer IO: 4 x 4 subgroups.
      {WmmaOp::kQ4gsw,
       "linear_q4gsw_wmma_t128x128k16g44s16m8fli",
       {128, 128, 16, 4, 4, 16, 8, false},
       kWmmaBufTex2d,
       nullptr},
      // 8da4w: texel-wise B staging and a full A thread map; blocked
      // workgroup order on weight-heavy projections, dispatch order elsewhere.
      {WmmaOp::kDq8caQ4gsw,
       "linear_dq8ca_q4gsw_wmma_t128x128k64g84s16m8n13",
       {128, 128, 64, 8, 4, 16, 8, false},
       kWmmaTex3dTex2d | kWmmaBufTex2d,
       xe2_weight_heavy},
      {WmmaOp::kDq8caQ4gsw,
       "linear_dq8ca_q4gsw_wmma_t128x128k64g84s16m8i",
       {128, 128, 64, 8, 4, 16, 8, false},
       kWmmaTex3dTex2d | kWmmaBufTex2d,
       nullptr},
  };
  return rows;
}

//
// NVIDIA rows. MMA 16x16x16 (fp16) / 16x16x32 (int8), subgroup 32,
// texture3d IO only.
//

// Only the prefill projections of Llama 3.2 1B / 3B and Llama 3.1 8B at group
// size 128 and M <= 2048 were validated on the family's members; every other
// shape keeps the existing kernels.
bool nv_measured(const WmmaShape& s) {
  const int64_t K = s.K;
  const int64_t N = s.N;
  const bool projection = (K == 2048 && (N == 512 || N == 2048 || N == 8192)) ||
      (K == 3072 && (N == 1024 || N == 3072 || N == 8192)) ||
      (K == 4096 && (N == 1024 || N == 4096 || N == 14336)) ||
      (K == 8192 && (N == 2048 || N == 3072)) || (K == 14336 && N == 4096);
  return projection && s.group_size == 128 && s.M <= 2048;
}
// fp16 accumulation loses accuracy beyond K = 8192: fp32 accumulation there.
bool nv_large_k(const WmmaShape& s) {
  return nv_measured(s) && s.K > 8192;
}
bool nv_256(const WmmaShape& s) {
  return nv_measured(s) && s.K <= 8192 && (s.M != 256 || s.N % 2048 == 0);
}
bool nv_default(const WmmaShape& s) {
  return nv_measured(s) && s.K <= 8192;
}

const std::vector<WmmaRow>& nvidia_rows() {
  static const std::vector<WmmaRow> rows = {
      // 4w, K > 8192: fp32 accumulation, texel-wise B staging, drain in the
      // A tile.
      {WmmaOp::kQ4gsw,
       "linear_q4gsw_wmma_bx_t128x128k32g42s32f32c",
       {128, 128, 32, 4, 2, 32, 16, true},
       kWmmaTex3dTex2d,
       nv_large_k},
      // 4w, K <= 8192: N-major B in shared memory.
      {WmmaOp::kQ4gsw,
       "linear_q4gsw_wmma_t256x128k16g42s32bt",
       {256, 128, 16, 4, 2, 32, 16, false},
       kWmmaTex3dTex2d,
       nv_256},
      // 4w, the remaining validated shapes (M not a multiple of 256, and
      // M = 256 at N % 2048 != 0).
      {WmmaOp::kQ4gsw,
       "linear_q4gsw_wmma_t128x128k16g22s32",
       {128, 128, 16, 2, 2, 32, 16, false},
       kWmmaTex3dTex2d,
       nv_default},
      // 8da4w: row-major int8 activations, 2 x 4 subgroups.
      {WmmaOp::kDq8caQ4gsw,
       "linear_dq8ca_q4gsw_wmma_rowmajor_t128x128k64g24s32mk32",
       {128, 128, 64, 2, 4, 32, 16, true},
       kWmmaTex3dTex2d,
       nv_measured,
       /*rowmajor_a=*/true},
  };
  return rows;
}

} // namespace

CoopmatFamily coopmat_family(const CoopmatDeviceKey& device) {
  if (!device.coopmat || !device.int8_coopmat) {
    return CoopmatFamily::kNone;
  }
  using T = CoopmatComponentType;
  if (device.vendor_id == kVendorIdIntel &&
      device.driver_id == kDriverIdIntelMesa &&
      has_shape(device, 8, 16, 16, T::kFp16, T::kFp16) &&
      has_shape(device, 8, 16, 32, T::kSint8, T::kSint32) &&
      subgroup_size_available(device, 16)) {
    return CoopmatFamily::kXe2;
  }
  if (device.vendor_id == kVendorIdNvidia &&
      device.driver_id == kDriverIdNvidiaProprietary &&
      has_shape(device, 16, 16, 16, T::kFp16, T::kFp16) &&
      has_shape(device, 16, 16, 16, T::kFp16, T::kFp32) &&
      has_shape(device, 16, 16, 32, T::kSint8, T::kSint32) &&
      subgroup_size_available(device, 32)) {
    return CoopmatFamily::kNvidia;
  }
  return CoopmatFamily::kNone;
}

const char* coopmat_family_name(const CoopmatFamily family) {
  switch (family) {
    case CoopmatFamily::kXe2:
      return "xe2";
    case CoopmatFamily::kNvidia:
      return "nvidia";
    case CoopmatFamily::kNone:
      break;
  }
  return "none";
}

const std::vector<WmmaRow>& coopmat_family_rows(const CoopmatFamily family) {
  static const std::vector<WmmaRow> kNoRows;
  switch (family) {
    case CoopmatFamily::kXe2:
      return xe2_rows();
    case CoopmatFamily::kNvidia:
      return nvidia_rows();
    case CoopmatFamily::kNone:
      break;
  }
  return kNoRows;
}

bool wmma_row_fits(
    const CoopmatDeviceKey& device,
    const WmmaShape& shape,
    const WmmaRow& row) {
  const WmmaTile& tile = row.tile;
  if (shape.op != row.op) {
    return false;
  }
  // The kernels are built without bias and read no leading dim.
  if (shape.has_bias || shape.gemv || shape.batch != 1 || !shape.half) {
    return false;
  }
  if (!device.coopmat) {
    return false;
  }
  if (shape.op == WmmaOp::kDq8caQ4gsw) {
    if (!device.int8_coopmat) {
      return false;
    }
    const WmmaInt8Layout want =
        row.rowmajor_a ? WmmaInt8Layout::kRowMajor : WmmaInt8Layout::k4H4W;
    if (shape.int8_layout != WmmaInt8Layout::kAny &&
        shape.int8_layout != want) {
      return false;
    }
  }
  // The pipeline requires the kernel's subgroup size.
  if (!subgroup_size_available(device, tile.subgroup_size)) {
    return false;
  }
  // One IO_STORAGE parameter covers input and output.
  uint8_t combo = 0;
  if (shape.output == WmmaStorage::kBuffer &&
      shape.input == WmmaStorage::kBuffer &&
      shape.weight == WmmaStorage::kTexture2D) {
    combo = kWmmaBufTex2d;
  } else if (
      shape.output == WmmaStorage::kTexture3D &&
      shape.input == WmmaStorage::kTexture3D && shape.io_width_packed &&
      shape.weight == WmmaStorage::kTexture2D) {
    combo = kWmmaTex3dTex2d;
    // The texture epilogue stages SG_GRID_Y * MMA_M rows x WG_TILE_N fp16 on
    // top of the A/B tiles unless it reuses the A tile. A tile over the
    // shared-memory limit must not reach pipeline creation.
    if (!tile.csh_in_ash) {
      const uint64_t csh_bytes =
          uint64_t(tile.sg_grid_y) * tile.mma_m * tile.n * 2u;
      if (csh_bytes >= device.max_shared_bytes) {
        return false;
      }
    }
  }
  if ((combo & row.storages) == 0) {
    return false;
  }
  if (!shape.ignore_alignment &&
      (shape.M % tile.m != 0 || shape.N % tile.n != 0 ||
       shape.K % tile.k != 0 || shape.group_size % tile.k != 0)) {
    return false;
  }
  return row.shape_ok == nullptr || row.shape_ok(shape);
}

std::optional<WmmaChoice> select_wmma_kernel(
    const CoopmatDeviceKey& device,
    const WmmaShape& shape) {
  for (const WmmaRow& row : coopmat_family_rows(coopmat_family(device))) {
    if (wmma_row_fits(device, shape, row)) {
      return WmmaChoice{row.kernel_base, row.tile, row.rowmajor_a};
    }
  }
  return std::nullopt;
}

std::optional<WmmaTile> wmma_tile_for_kernel(const std::string& kernel_name) {
  for (const CoopmatFamily family :
       {CoopmatFamily::kXe2, CoopmatFamily::kNvidia}) {
    for (const WmmaRow& row : coopmat_family_rows(family)) {
      const std::string base(row.kernel_base);
      if (kernel_name.size() > base.size() &&
          kernel_name.compare(0, base.size(), base) == 0 &&
          kernel_name[base.size()] == '_') {
        return row.tile;
      }
    }
  }
  return std::nullopt;
}

} // namespace vkcompute
