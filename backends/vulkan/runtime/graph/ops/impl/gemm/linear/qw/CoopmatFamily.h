/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Cooperative-matrix architecture families for the 4-bit quantized linear ops
// (et_vk.linear_q4gsw / et_vk.q4gsw_linear and et_vk.linear_dq8ca_q4gsw).
//
// A device belongs to a family by its vendor ID, its driver ID and a property
// signature (cooperative-matrix shapes, available subgroup sizes), never by
// its name. Each family has a small set of kernel rows; a device that matches
// no family, or a shape no row of its family accepts, keeps the existing
// kernels unchanged.
//
// Pure C++ (no Vulkan or ComputeGraph types) so it can be unit-tested on a
// host without a GPU (test/custom_ops/coopmat_family_test.cpp). The Vulkan
// side builds a CoopmatDeviceKey from the adapter (WmmaLinear.h).

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vkcompute {

// Component types of a cooperative-matrix configuration (VkComponentTypeKHR,
// reduced to the types the family keys test).
enum class CoopmatComponentType { kOther, kFp16, kFp32, kSint8, kSint32 };

// One subgroup-scope VkCooperativeMatrixPropertiesKHR entry with A == B and
// C == Result types.
struct CoopmatShape {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  CoopmatComponentType ab; // A and B
  CoopmatComponentType result; // C and Result
};

// VkPhysicalDeviceProperties::vendorID and VkDriverId values the family keys
// use.
constexpr uint32_t kVendorIdIntel = 0x8086;
constexpr uint32_t kVendorIdNvidia = 0x10de;
// VK_DRIVER_ID_NVIDIA_PROPRIETARY
constexpr uint32_t kDriverIdNvidiaProprietary = 4;
// VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA
constexpr uint32_t kDriverIdIntelMesa = 6;

struct CoopmatDeviceKey {
  uint32_t vendor_id = 0;
  uint32_t driver_id = 0; // VkDriverId, 0 when not reported
  uint32_t subgroup_size = 0; // default subgroup size
  uint32_t min_subgroup_size = 0;
  uint32_t max_subgroup_size = 0;
  bool subgroup_size_control = false; // a required subgroup size can be set
  bool coopmat = false; // VK_KHR_cooperative_matrix is usable
  bool int8_coopmat = false; // an int8 cooperative-matrix shape is enumerated
  uint32_t max_shared_bytes = 0; // maxComputeSharedMemorySize
  std::vector<CoopmatShape> coopmat_shapes; // subgroup scope only
};

enum class CoopmatFamily {
  kNone,
  // Intel Xe2 on the Mesa driver: fp16 8x16x16 (fp16 result), int8 8x16x32,
  // subgroup size 16 available.
  kXe2,
  // NVIDIA proprietary driver: fp16 16x16x16 (fp16 and fp32 result), int8
  // 16x16x32, subgroup size 32. Measured on Ampere (Jetson Orin) and Ada; the
  // key does not separate them from other NVIDIA generations with the same
  // shapes.
  kNvidia,
};

// The only keying point. kNone keeps the existing kernels.
CoopmatFamily coopmat_family(const CoopmatDeviceKey& device);

const char* coopmat_family_name(CoopmatFamily family);

//
// Kernel selection
//

enum class WmmaOp {
  kQ4gsw, // 4w: fp16 activations, 4-bit group-symmetric weights
  kDq8caQ4gsw, // 8da4w: dynamic int8 activations, 4-bit weights
};

enum class WmmaStorage { kBuffer, kTexture2D, kTexture3D, kOther };

// Layout of the int8 activations of kDq8caQ4gsw. The 4h4w kernels read
// utils::kPackedInt8_4H4W blocks; the row-major kernels read
// utils::kPackedInt8_4W.
enum class WmmaInt8Layout { kAny, k4H4W, kRowMajor };

struct WmmaShape {
  WmmaOp op = WmmaOp::kQ4gsw;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t group_size = 0;
  int64_t batch = 1; // product of the leading dims
  bool gemv = false;
  bool has_bias = false;
  bool half = false;
  WmmaStorage input = WmmaStorage::kOther;
  WmmaStorage output = WmmaStorage::kOther;
  WmmaStorage weight = WmmaStorage::kOther;
  bool io_width_packed = false;
  WmmaInt8Layout int8_layout = WmmaInt8Layout::kAny; // kDq8caQ4gsw only
  // Skip the M/N/K tile-alignment check (row-major 8da4w kernels, whose
  // activation layout is fixed at build time; the grid rounds M up).
  bool ignore_alignment = false;
};

// The kQ4gsw texture-IO kernels accept a partial last M tile. Their rows take
// an M of at least this many rows that is not a multiple of their tile, and are
// selected as for M rounded up to a multiple of it (the smallest WG_TILE_M of
// the family rows), so an unaligned M gets the kernel of the next aligned M.
// Smaller M, buffer IO and kDq8caQ4gsw keep the alignment requirement.
constexpr int64_t kWmmaPartialTileM = 128;

struct WmmaTile {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t sg_grid_x;
  uint32_t sg_grid_y;
  uint32_t subgroup_size;
  uint32_t mma_m;
  bool csh_in_ash; // texture3d drain staged in the A tile (no extra Csh)
  uint32_t wg_size() const {
    return sg_grid_x * sg_grid_y * subgroup_size;
  }
};

// Storage combinations (IO_STORAGE x WEIGHT_STORAGE) a kernel is built for;
// the yaml contains <kernel_base>_<io>_<weight>_half for each set bit.
enum WmmaStorageSet : uint8_t {
  kWmmaTex3dTex2d = 1u << 0, // texture3d input/output, texture2d weight
  kWmmaBufTex2d = 1u << 1, // buffer input/output, texture2d weight
};

struct WmmaRow {
  WmmaOp op;
  const char* kernel_base; // shader name without _<io>_<weight>_<dtype>
  WmmaTile tile;
  uint8_t storages; // WmmaStorageSet bits
  // Extra shape requirement (beyond tile alignment), or null. Rows are tried
  // in order, so a later row can serve the shapes an earlier one rejects.
  bool (*shape_ok)(const WmmaShape&);
  bool rowmajor_a = false; // kDq8caQ4gsw: reads row-major int8 activations
  // kQ4gsw: the texture3d variant reads A through a storage image, so the
  // input must be bound in VK_IMAGE_LAYOUT_GENERAL.
  bool image_a = false;
};

struct WmmaChoice {
  std::string kernel_base;
  WmmaTile tile;
  bool rowmajor_a = false;
  bool image_a = false;
};

// The rows of a family, in selection order (empty for kNone).
const std::vector<WmmaRow>& coopmat_family_rows(CoopmatFamily family);

// The device, storage, alignment and shape constraints of `row`.
bool wmma_row_fits(
    const CoopmatDeviceKey& device,
    const WmmaShape& shape,
    const WmmaRow& row);

// The first row of the device's family that fits `shape`, or std::nullopt
// (no family, or no row applies): the op keeps its existing kernels.
std::optional<WmmaChoice> select_wmma_kernel(
    const CoopmatDeviceKey& device,
    const WmmaShape& shape);

// Tile of a kernel named <kernel_base>_<suffixes> for any family row.
std::optional<WmmaTile> wmma_tile_for_kernel(const std::string& kernel_name);

} // namespace vkcompute
