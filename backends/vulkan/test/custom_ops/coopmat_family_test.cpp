// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Host-only tests of the cooperative-matrix family key and the family kernel
// selection (CoopmatFamily.h). No GPU needed.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/CoopmatFamily.h>

#include <gtest/gtest.h>

#include <fstream>
#include <functional>
#include <sstream>
#include <string>

namespace vkcompute {
namespace {

using T = CoopmatComponentType;

// VkDriverId values used below.
constexpr uint32_t kDriverMesaRadv = 3;
constexpr uint32_t kDriverIntelWindows = 5;
constexpr uint32_t kDriverMesaLlvmpipe = 13;
constexpr uint32_t kDriverMesaNvk = 24;

CoopmatDeviceKey
base_key(const uint32_t sg, const uint32_t sg_min, const uint32_t sg_max) {
  CoopmatDeviceKey d;
  d.subgroup_size = sg;
  d.min_subgroup_size = sg_min;
  d.max_subgroup_size = sg_max;
  d.subgroup_size_control = true;
  d.coopmat = true;
  d.int8_coopmat = true;
  d.max_shared_bytes = 65536;
  return d;
}

// Keys as Xe2 (Arc B580 / Arc Pro B70) and NVIDIA (Ampere / Ada) devices
// report them.
CoopmatDeviceKey xe2_key(const uint32_t driver = kDriverIdIntelMesa) {
  CoopmatDeviceKey d = base_key(32, 8, 32);
  d.vendor_id = kVendorIdIntel;
  d.driver_id = driver;
  d.coopmat_shapes = {
      {8, 16, 16, T::kFp16, T::kFp16},
      {8, 16, 16, T::kFp16, T::kFp32},
      {8, 16, 32, T::kSint8, T::kSint32}};
  return d;
}

CoopmatDeviceKey nvidia_key(
    const uint32_t driver = kDriverIdNvidiaProprietary) {
  CoopmatDeviceKey d = base_key(32, 32, 32);
  d.vendor_id = kVendorIdNvidia;
  d.driver_id = driver;
  d.coopmat_shapes = {
      {16, 16, 16, T::kFp16, T::kFp16},
      {16, 8, 16, T::kFp16, T::kFp16},
      {16, 16, 16, T::kFp16, T::kFp32},
      {16, 16, 32, T::kSint8, T::kSint32}};
  return d;
}

// Mesa llvmpipe (lavapipe): 8x8x8 shapes only, fixed subgroup 8.
CoopmatDeviceKey llvmpipe_key() {
  CoopmatDeviceKey d = base_key(8, 8, 8);
  d.vendor_id = 0x10005;
  d.driver_id = kDriverMesaLlvmpipe;
  d.max_shared_bytes = 32768;
  d.coopmat_shapes = {
      {8, 8, 8, T::kFp16, T::kFp16},
      {8, 8, 8, T::kFp16, T::kFp32},
      {8, 8, 8, T::kOther, T::kOther}, // u8 x u8 -> u32
      {8, 8, 8, T::kSint8, T::kSint32}};
  return d;
}

// AMD RDNA3 under RADV: no family yet.
CoopmatDeviceKey radv_key() {
  CoopmatDeviceKey d = base_key(64, 32, 64);
  d.vendor_id = 0x1002;
  d.driver_id = kDriverMesaRadv;
  d.coopmat_shapes = {
      {16, 16, 16, T::kFp16, T::kFp16},
      {16, 16, 16, T::kFp16, T::kFp32},
      {16, 16, 16, T::kSint8, T::kSint32}};
  return d;
}

WmmaShape prefill(
    const WmmaOp op,
    const int64_t M,
    const int64_t K,
    const int64_t N,
    const WmmaStorage io) {
  WmmaShape s;
  s.op = op;
  s.M = M;
  s.K = K;
  s.N = N;
  s.group_size = 128;
  s.half = true;
  s.input = io;
  s.output = io;
  s.weight = WmmaStorage::kTexture2D;
  s.io_width_packed = true;
  return s;
}

std::string selected(const CoopmatDeviceKey& d, const WmmaShape& s) {
  const std::optional<WmmaChoice> c = select_wmma_kernel(d, s);
  return c.has_value() ? c->kernel_base : "";
}

// (K, N) of every 4-bit linear in Llama 3.2 1B / 3B and Llama 3.1 8B.
const int64_t kShapes[][2] = {
    {2048, 2048},
    {2048, 512},
    {2048, 8192},
    {8192, 2048},
    {3072, 3072},
    {3072, 1024},
    {3072, 8192},
    {8192, 3072},
    {4096, 4096},
    {4096, 1024},
    {4096, 14336},
    {14336, 4096},
};

const std::string kQ = "linear_q4gsw_wmma_";
const std::string kD = "linear_dq8ca_q4gsw_wmma_";

using Rule = std::function<std::string(const WmmaShape&)>;

// The intended kernel per family, restated independently of the tables.
std::string xe2_rule(const WmmaShape& s) {
  if (s.op == WmmaOp::kQ4gsw) {
    return kQ +
        (s.output == WmmaStorage::kTexture3D ? "t128x128k16g82s16m8flib"
                                             : "t128x128k16g44s16m8fli");
  }
  return kD +
      (s.K * s.N >= 20000000 ? "t128x128k64g84s16m8n13"
                             : "t128x128k64g84s16m8i");
}

std::string nvidia_rule(const WmmaShape& s) {
  if (s.output != WmmaStorage::kTexture3D) {
    return "";
  }
  if (s.op == WmmaOp::kQ4gsw) {
    return kQ +
        (s.K > 8192 ? "bx_t128x128k32g42s32f32c" : "t256x128k16g42s32bt");
  }
  return kD + "rowmajor_t128x128k64g24s32mk32";
}

std::string none_rule(const WmmaShape&) {
  return "";
}

void check_llama_shapes(
    const char* label,
    const CoopmatDeviceKey& d,
    const Rule& rule) {
  for (const WmmaOp op : {WmmaOp::kQ4gsw, WmmaOp::kDq8caQ4gsw}) {
    for (const auto& kn : kShapes) {
      for (const WmmaStorage io :
           {WmmaStorage::kTexture3D, WmmaStorage::kBuffer}) {
        const WmmaShape s = prefill(op, 2048, kn[0], kn[1], io);
        EXPECT_EQ(selected(d, s), rule(s))
            << label << " op=" << static_cast<int>(op) << " K=" << kn[0]
            << " N=" << kn[1]
            << " io=" << (io == WmmaStorage::kBuffer ? "buffer" : "texture3d");
      }
    }
  }
}

TEST(CoopmatFamilyTest, FamilyKey) {
  EXPECT_EQ(coopmat_family(xe2_key()), CoopmatFamily::kXe2);
  EXPECT_EQ(coopmat_family(nvidia_key()), CoopmatFamily::kNvidia);

  // Same silicon under another driver: no family.
  EXPECT_EQ(coopmat_family(xe2_key(kDriverIntelWindows)), CoopmatFamily::kNone);
  EXPECT_EQ(coopmat_family(nvidia_key(kDriverMesaNvk)), CoopmatFamily::kNone);

  // The int8 shape is part of the key.
  CoopmatDeviceKey no_int8 = nvidia_key();
  no_int8.int8_coopmat = false;
  EXPECT_EQ(coopmat_family(no_int8), CoopmatFamily::kNone);
  CoopmatDeviceKey no_int8_shape = xe2_key();
  no_int8_shape.coopmat_shapes.pop_back();
  EXPECT_EQ(coopmat_family(no_int8_shape), CoopmatFamily::kNone);

  // Xe2 needs subgroup size 16 to be available.
  CoopmatDeviceKey xe2_sg8 = xe2_key();
  xe2_sg8.subgroup_size = 8;
  xe2_sg8.min_subgroup_size = 8;
  xe2_sg8.max_subgroup_size = 8;
  EXPECT_EQ(coopmat_family(xe2_sg8), CoopmatFamily::kNone);
  CoopmatDeviceKey xe2_no_control = xe2_key();
  xe2_no_control.subgroup_size_control = false;
  EXPECT_EQ(coopmat_family(xe2_no_control), CoopmatFamily::kNone);

  // No family for vendors without validated kernels.
  EXPECT_EQ(coopmat_family(radv_key()), CoopmatFamily::kNone);

  // No cooperative matrix at all.
  CoopmatDeviceKey no_coopmat = xe2_key();
  no_coopmat.coopmat = false;
  EXPECT_EQ(coopmat_family(no_coopmat), CoopmatFamily::kNone);
}

TEST(CoopmatFamilyTest, LlvmpipeHasNoFamily) {
  const CoopmatDeviceKey d = llvmpipe_key();
  EXPECT_EQ(coopmat_family(d), CoopmatFamily::kNone);
  EXPECT_TRUE(coopmat_family_rows(coopmat_family(d)).empty());
  check_llama_shapes("llvmpipe", d, none_rule);
  // Shapes aligned to every family tile.
  for (const WmmaOp op : {WmmaOp::kQ4gsw, WmmaOp::kDq8caQ4gsw}) {
    for (const WmmaStorage io :
         {WmmaStorage::kTexture3D, WmmaStorage::kBuffer}) {
      EXPECT_FALSE(
          select_wmma_kernel(d, prefill(op, 256, 256, 256, io)).has_value());
    }
  }
}

TEST(CoopmatFamilyTest, UnmatchedKeysSelectNothing) {
  check_llama_shapes("radv", radv_key(), none_rule);
  check_llama_shapes("xe2-windows", xe2_key(kDriverIntelWindows), none_rule);
  check_llama_shapes("nvk", nvidia_key(kDriverMesaNvk), none_rule);
}

TEST(CoopmatFamilyTest, Xe2Rows) {
  check_llama_shapes("xe2", xe2_key(), xe2_rule);
  // The weight-heavy rule: K * N >= 20e6.
  const CoopmatDeviceKey d = xe2_key();
  EXPECT_EQ(
      selected(
          d,
          prefill(WmmaOp::kDq8caQ4gsw, 256, 2048, 9856, WmmaStorage::kBuffer)),
      kD + "t128x128k64g84s16m8n13"); // 20.2e6
  EXPECT_EQ(
      selected(
          d,
          prefill(WmmaOp::kDq8caQ4gsw, 256, 2048, 9728, WmmaStorage::kBuffer)),
      kD + "t128x128k64g84s16m8i"); // 19.9e6
  // Any aligned shape, not only the Llama projections.
  EXPECT_EQ(
      selected(d, prefill(WmmaOp::kQ4gsw, 128, 256, 384, WmmaStorage::kBuffer)),
      kQ + "t128x128k16g44s16m8fli");
}

TEST(CoopmatFamilyTest, NvidiaRows) {
  check_llama_shapes("nvidia", nvidia_key(), nvidia_rule);
  const CoopmatDeviceKey d = nvidia_key();
  struct Edge {
    int64_t M, K, N;
    const char* want; // nullptr: existing kernels
  };
  const Edge edges[] = {
      // M not a multiple of 256: the 128-row tiles.
      {384, 2048, 2048, "t128x128k16g22s32"},
      // M = 256 at N % 2048 != 0: not the 256-row tile.
      {256, 3072, 1024, "t128x128k16g22s32"},
      {256, 2048, 8192, "t256x128k16g42s32bt"},
      // K > 8192 with M not a multiple of 256: fp32 accumulation.
      {384, 14336, 4096, "bx_t128x128k32g42s32f32c"},
      // Any aligned shape, not only the Llama projections (Qwen2.5 0.5B,
      // Gemma 2 2B), and M beyond 2048.
      {512, 896, 4864, "t256x128k16g42s32bt"},
      {512, 9216, 2304, "bx_t128x128k32g42s32f32c"},
      {4096, 2048, 2048, "t256x128k16g42s32bt"},
      // N not a multiple of the 128-column tile.
      {2048, 2048, 4160, nullptr},
  };
  for (const Edge& e : edges) {
    const WmmaShape s =
        prefill(WmmaOp::kQ4gsw, e.M, e.K, e.N, WmmaStorage::kTexture3D);
    EXPECT_EQ(selected(d, s), e.want == nullptr ? "" : kQ + e.want)
        << "M=" << e.M << " K=" << e.K << " N=" << e.N;
  }
  // Group sizes other than 128, down to the K tile of the row.
  for (const WmmaOp op : {WmmaOp::kQ4gsw, WmmaOp::kDq8caQ4gsw}) {
    WmmaShape g64 = prefill(op, 2048, 2048, 2048, WmmaStorage::kTexture3D);
    g64.group_size = 64;
    EXPECT_EQ(selected(d, g64), nvidia_rule(g64));
  }
  WmmaShape g32 =
      prefill(WmmaOp::kDq8caQ4gsw, 2048, 2048, 2048, WmmaStorage::kTexture3D);
  g32.group_size = 32;
  EXPECT_EQ(selected(d, g32), "");
}

TEST(CoopmatFamilyTest, ShapesThatKeepTheExistingKernels) {
  for (const CoopmatDeviceKey& d : {xe2_key(), nvidia_key()}) {
    for (const WmmaOp op : {WmmaOp::kQ4gsw, WmmaOp::kDq8caQ4gsw}) {
      const WmmaShape s =
          prefill(op, 2048, 4096, 4096, WmmaStorage::kTexture3D);
      ASSERT_FALSE(selected(d, s).empty());

      // Below kWmmaPartialTileM no row takes an unaligned M.
      WmmaShape unaligned = s;
      unaligned.M = 120;
      WmmaShape gemv = s;
      gemv.M = 1;
      gemv.gemv = true;
      WmmaShape bias = s;
      bias.has_bias = true;
      WmmaShape fp32 = s;
      fp32.half = false;
      WmmaShape mixed = s;
      mixed.input = WmmaStorage::kBuffer;
      WmmaShape batched = s;
      batched.batch = 2;
      WmmaShape buffer_weight = s;
      buffer_weight.weight = WmmaStorage::kBuffer;
      WmmaShape channels_packed = s;
      channels_packed.io_width_packed = false;
      for (const WmmaShape* x :
           {&unaligned,
            &gemv,
            &bias,
            &fp32,
            &mixed,
            &batched,
            &buffer_weight,
            &channels_packed}) {
        EXPECT_FALSE(select_wmma_kernel(d, *x).has_value())
            << coopmat_family_name(coopmat_family(d))
            << " op=" << static_cast<int>(op);
      }
    }
  }
}

TEST(CoopmatFamilyTest, PartialMTile) {
  struct Case {
    CoopmatDeviceKey device;
    WmmaOp op;
    int64_t M, K, N;
    WmmaStorage io;
    std::string want; // empty: existing kernels
  };
  const Case cases[] = {
      // 4w texture IO: selected as for M rounded up to 128 rows.
      {xe2_key(),
       WmmaOp::kQ4gsw,
       511,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       kQ + "t128x128k16g82s16m8flib"},
      {xe2_key(),
       WmmaOp::kQ4gsw,
       129,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       kQ + "t128x128k16g82s16m8flib"},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       511,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       kQ + "t256x128k16g42s32bt"},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       300,
       3072,
       1024,
       WmmaStorage::kTexture3D,
       kQ + "t128x128k16g22s32"},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       129,
       14336,
       4096,
       WmmaStorage::kTexture3D,
       kQ + "bx_t128x128k32g42s32f32c"},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       2047,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       kQ + "t256x128k16g42s32bt"},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       2049,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       kQ + "t128x128k16g22s32"},
      // Below kWmmaPartialTileM.
      {xe2_key(), WmmaOp::kQ4gsw, 127, 2048, 2048, WmmaStorage::kTexture3D, ""},
      {nvidia_key(),
       WmmaOp::kQ4gsw,
       127,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       ""},
      // Buffer IO and 8da4w keep the alignment requirement.
      {xe2_key(), WmmaOp::kQ4gsw, 511, 2048, 2048, WmmaStorage::kBuffer, ""},
      {xe2_key(),
       WmmaOp::kDq8caQ4gsw,
       511,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       ""},
      {nvidia_key(),
       WmmaOp::kDq8caQ4gsw,
       511,
       2048,
       2048,
       WmmaStorage::kTexture3D,
       ""},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(selected(c.device, prefill(c.op, c.M, c.K, c.N, c.io)), c.want)
        << coopmat_family_name(coopmat_family(c.device))
        << " op=" << static_cast<int>(c.op) << " M=" << c.M << " K=" << c.K
        << " N=" << c.N << " io=" << static_cast<int>(c.io);
  }
}

TEST(CoopmatFamilyTest, Int8Layout) {
  const CoopmatDeviceKey d = nvidia_key();
  WmmaShape s =
      prefill(WmmaOp::kDq8caQ4gsw, 2048, 4096, 4096, WmmaStorage::kTexture3D);
  const std::optional<WmmaChoice> aligned = select_wmma_kernel(d, s);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_TRUE(aligned->rowmajor_a);

  // A row-major op keeps its kernel for unaligned M when alignment is waived.
  s.int8_layout = WmmaInt8Layout::kRowMajor;
  s.M = 1304;
  s.ignore_alignment = true;
  const std::optional<WmmaChoice> forced = select_wmma_kernel(d, s);
  ASSERT_TRUE(forced.has_value());
  EXPECT_EQ(forced->kernel_base, aligned->kernel_base);

  // 4h4w activations never get a row-major kernel, and vice versa.
  WmmaShape h = s;
  h.M = 2048;
  h.ignore_alignment = false;
  h.int8_layout = WmmaInt8Layout::k4H4W;
  EXPECT_FALSE(select_wmma_kernel(d, h).has_value());
  WmmaShape x =
      prefill(WmmaOp::kDq8caQ4gsw, 2048, 4096, 4096, WmmaStorage::kTexture3D);
  x.int8_layout = WmmaInt8Layout::kRowMajor;
  EXPECT_FALSE(select_wmma_kernel(xe2_key(), x).has_value());
}

TEST(CoopmatFamilyTest, SharedMemoryLimit) {
  // The texture3d drain of the Xe2 4w tile needs 2 * 8 * 128 * 2 = 4 KiB of
  // shared memory on top of the A/B tiles.
  CoopmatDeviceKey d = xe2_key();
  d.max_shared_bytes = 4096;
  EXPECT_EQ(
      selected(
          d,
          prefill(WmmaOp::kQ4gsw, 2048, 4096, 4096, WmmaStorage::kTexture3D)),
      "");
  EXPECT_EQ(
      selected(
          d, prefill(WmmaOp::kQ4gsw, 2048, 4096, 4096, WmmaStorage::kBuffer)),
      kQ + "t128x128k16g44s16m8fli");
}

TEST(CoopmatFamilyTest, TileForKernel) {
  for (const CoopmatFamily f : {CoopmatFamily::kXe2, CoopmatFamily::kNvidia}) {
    for (const WmmaRow& row : coopmat_family_rows(f)) {
      const std::optional<WmmaTile> tile = wmma_tile_for_kernel(
          std::string(row.kernel_base) + "_texture3d_texture2d_half");
      ASSERT_TRUE(tile.has_value()) << row.kernel_base;
      EXPECT_EQ(tile->m, row.tile.m) << row.kernel_base;
      EXPECT_EQ(tile->n, row.tile.n) << row.kernel_base;
      EXPECT_EQ(tile->wg_size(), row.tile.wg_size()) << row.kernel_base;
    }
  }
  // The existing kernels are not family kernels.
  EXPECT_FALSE(
      wmma_tile_for_kernel("linear_q4gsw_coopmat_buffer_texture2d_half")
          .has_value());
  EXPECT_FALSE(
      wmma_tile_for_kernel("linear_q4gsw_tiled_texture3d_texture2d_half")
          .has_value());
  EXPECT_FALSE(wmma_tile_for_kernel("linear_q4gsw_wmma_t128x128k16g82s16m8fli")
                   .has_value());
}

#ifdef COOPMAT_FAMILY_TEST_GLSL_DIR
std::string read_file(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Every kernel name a row can produce must be a variant in its yaml.
TEST(CoopmatFamilyTest, EveryRowHasShaderVariants) {
  const std::string dir =
      std::string(COOPMAT_FAMILY_TEST_GLSL_DIR) + "/gemm/linear/qw/";
  const std::string yamls = read_file(dir + "q4gsw/linear_q4gsw_wmma.yaml") +
      read_file(dir + "q4gsw/linear_q4gsw_wmma_bx.yaml") +
      read_file(dir + "dq8ca_q4gsw/linear_dq8ca_q4gsw_wmma.yaml") +
      read_file(dir + "dq8ca_q4gsw/linear_dq8ca_q4gsw_wmma_rowmajor.yaml");
  if (yamls.empty()) {
    GTEST_SKIP() << "shader sources not found under " << dir;
  }
  const std::pair<uint8_t, const char*> combos[] = {
      {kWmmaTex3dTex2d, "_texture3d_texture2d_half"},
      {kWmmaBufTex2d, "_buffer_texture2d_half"}};
  for (const CoopmatFamily f : {CoopmatFamily::kXe2, CoopmatFamily::kNvidia}) {
    for (const WmmaRow& row : coopmat_family_rows(f)) {
      for (const auto& c : combos) {
        if (row.storages & c.first) {
          const std::string name = std::string(row.kernel_base) + c.second;
          EXPECT_NE(yamls.find("NAME: " + name + "\n"), std::string::npos)
              << "no shader variant " << name;
        }
      }
    }
  }
}
#endif // COOPMAT_FAMILY_TEST_GLSL_DIR

} // namespace
} // namespace vkcompute
