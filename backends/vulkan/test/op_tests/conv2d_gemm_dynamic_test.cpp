/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dGemm.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <xnnpack.h>

#include <algorithm>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;
using namespace vkcompute;

//
// Dynamic-shape (resize) test for the tiled im2col conv2d path.
//
// The im2col + GEMM conv path (conv2d_gemm_impl) materializes its im2col matrix
// in tiles of output-height rows bounded to a byte budget. The static op-test
// suite (test/custom_ops/test_conv2d.cpp) only validates a single shape per
// graph. This test exercises the path under an actual trigger_resize: it builds
// the graph at an upper-bound input shape that forces MULTIPLE tiles, then
// resizes the input across tile boundaries (so trailing tiles must no-op via
// the shader's `oh < H_out` guard and the scratch must track the smaller shape)
// and verifies the output against a reference at every shape.
//
// Build-time upper bound: storage and the fixed num_tiles are built at the
// initial input shape, so every resized shape MUST be <= the initial shape per
// dim (resize-down is the supported direction; resizing back up toward — never
// above — the bound is fine). The harness asserts this so misuse fails loudly.
//
// Multi-tiling is forced by a real shape at the production 16 MB budget (no
// test knob): C_in=64, 3x3, 128x128 gives K_total = 9*64 = 576, so one
// output-height row of im2col is W_out * K_total * elem = 128 * 576 * 4 = 288
// KB; oh_tile = 16 MB / 288 KB = 56 rows, and H_out=128 needs ceil(128/56) = 3
// tiles. The resize sweep (H = 64 / 56 / 112 / 128) needs 2 / 1 / 2 / 3 active
// tiles, crossing tile boundaries down to a single tile and back up, leaving
// surplus build-time tiles to no-op. The reference is computed by XNNPACK so
// this medium shape stays cheap.
//
// The test sweeps the full io_storage x im2col_storage matrix (3 x 3 = 9
// combos): io_storage = the input/output tensor storage; im2col_storage = the
// scratch storage forced into conv2d_gemm_impl.

namespace {

//
// Test specification
//

struct Conv2dTestConfig {
  // Conv params.
  int64_t in_channels;
  int64_t out_channels;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t padding_h;
  int64_t padding_w;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t groups; // only groups == 1 is supported by conv2d_gemm_impl
  bool has_bias;

  // Initial (BUILD-time, upper-bound) input spatial extents. The input is
  // always [1, in_channels, init_h, init_w]; weight is
  // [out_channels, in_channels, kernel_h, kernel_w].
  int64_t init_h;
  int64_t init_w;

  // Storage type for the conv INPUT and OUTPUT tensors.
  utils::StorageType io_storage;

  // Storage type for the im2col SCRATCH tensor (distinct from io_storage).
  utils::StorageType im2col_storage;

  // Resized input (h, w) shapes to sweep after the initial build. Each MUST be
  // <= (init_h, init_w) per dim (build-time upper bound).
  std::vector<std::pair<int64_t, int64_t>> resize_hw;
};

const char* storage_type_name(utils::StorageType st) {
  switch (st) {
    case utils::kBuffer:
      return "buffer";
    case utils::kTexture2D:
      return "texture2d";
    case utils::kTexture3D:
      return "texture3d";
    default:
      return "unknown";
  }
}

// Human-readable dump of a Conv2dTestConfig, used both as per-test SCOPED_TRACE
// context and in per-shape mismatch messages so a failing run names the exact
// config + storage that failed.
std::string to_string(const Conv2dTestConfig& cfg) {
  std::ostringstream os;
  os << "Conv2dTestConfig{in_ch=" << cfg.in_channels
     << " out_ch=" << cfg.out_channels << " kernel=" << cfg.kernel_h << "x"
     << cfg.kernel_w << " stride=" << cfg.stride_h << "x" << cfg.stride_w
     << " pad=" << cfg.padding_h << "x" << cfg.padding_w
     << " dilation=" << cfg.dilation_h << "x" << cfg.dilation_w
     << " groups=" << cfg.groups << " has_bias=" << (cfg.has_bias ? "1" : "0")
     << " init=" << cfg.init_h << "x" << cfg.init_w
     << " io_storage=" << storage_type_name(cfg.io_storage)
     << " im2col_storage=" << storage_type_name(cfg.im2col_storage)
     << " resize_hw=[";
  for (size_t i = 0; i < cfg.resize_hw.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << cfg.resize_hw[i].first << "x" << cfg.resize_hw[i].second;
  }
  os << "]}";
  return os.str();
}

int64_t conv_out_dim(
    int64_t in,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
  return (in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

std::vector<float> rand_floats(size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
  return data;
}

size_t numel(const std::vector<int64_t>& sizes) {
  size_t n = 1;
  for (auto s : sizes) {
    n *= static_cast<size_t>(s);
  }
  return n;
}

std::vector<int32_t> to_int32(const std::vector<int64_t>& v) {
  return std::vector<int32_t>(v.begin(), v.end());
}

//
// Reference: XNNPACK f32 conv2d (fast). XNNPACK is NHWC with OHWI weights; the
// graph I/O is NCHW (channels-packed), so this converts in -> NHWC and weight
// -> OHWI before the op and the output NHWC -> NCHW after, returning an
// NCHW-order reference to compare against the staging read-back.
//
// (A naive nested-loop reference was the reason large shapes couldn't be
// tested; XNNPACK runs the reference fast enough that the build shape can be a
// true upper bound without the CPU reference dominating runtime.)
//
std::vector<float> conv2d_ref_xnnpack(
    const std::vector<float>& input_nchw, // [C_in, H, W]
    const std::vector<float>& weight_nchw, // [C_out, C_in, K_h, K_w]
    const std::vector<float>& bias, // [C_out] or empty
    const Conv2dTestConfig& cfg,
    int64_t H_in,
    int64_t W_in) {
  const int64_t C_in = cfg.in_channels;
  const int64_t C_out = cfg.out_channels;
  const int64_t K_h = cfg.kernel_h;
  const int64_t K_w = cfg.kernel_w;
  const int64_t H_out =
      conv_out_dim(H_in, K_h, cfg.stride_h, cfg.padding_h, cfg.dilation_h);
  const int64_t W_out =
      conv_out_dim(W_in, K_w, cfg.stride_w, cfg.padding_w, cfg.dilation_w);

  // NCHW -> NHWC input.
  std::vector<float> input_nhwc(static_cast<size_t>(C_in * H_in * W_in));
  for (int64_t c = 0; c < C_in; ++c) {
    for (int64_t h = 0; h < H_in; ++h) {
      for (int64_t w = 0; w < W_in; ++w) {
        input_nhwc[static_cast<size_t>((h * W_in + w) * C_in + c)] =
            input_nchw[static_cast<size_t>(c * (H_in * W_in) + h * W_in + w)];
      }
    }
  }

  // NCHW -> OHWI weight ([C_out, C_in, K_h, K_w] -> [C_out, K_h, K_w, C_in]).
  std::vector<float> weight_ohwi(static_cast<size_t>(C_out * K_h * K_w * C_in));
  for (int64_t co = 0; co < C_out; ++co) {
    for (int64_t ci = 0; ci < C_in; ++ci) {
      for (int64_t kh = 0; kh < K_h; ++kh) {
        for (int64_t kw = 0; kw < K_w; ++kw) {
          weight_ohwi[static_cast<size_t>(
              ((co * K_h + kh) * K_w + kw) * C_in + ci)] =
              weight_nchw[static_cast<size_t>(
                  ((co * C_in + ci) * K_h + kh) * K_w + kw)];
        }
      }
    }
  }

  EXPECT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);

  // XNNPACK failures throw rather than return a zeroed result: a silent zero
  // output would masquerade as a Vulkan-vs-reference mismatch downstream. The
  // operator handle is deleted before throwing so it does not leak on the
  // error path.
  xnn_operator_t op = nullptr;
  const float out_min = -std::numeric_limits<float>::infinity();
  const float out_max = std::numeric_limits<float>::infinity();
  const xnn_status create_status = xnn_create_convolution2d_nhwc_f32(
      static_cast<uint32_t>(cfg.padding_h),
      static_cast<uint32_t>(cfg.padding_w),
      static_cast<uint32_t>(cfg.padding_h),
      static_cast<uint32_t>(cfg.padding_w),
      static_cast<uint32_t>(K_h),
      static_cast<uint32_t>(K_w),
      static_cast<uint32_t>(cfg.stride_h),
      static_cast<uint32_t>(cfg.stride_w),
      static_cast<uint32_t>(cfg.dilation_h),
      static_cast<uint32_t>(cfg.dilation_w),
      /*groups=*/1,
      /*group_input_channels=*/static_cast<size_t>(C_in),
      /*group_output_channels=*/static_cast<size_t>(C_out),
      /*input_channel_stride=*/static_cast<size_t>(C_in),
      /*output_channel_stride=*/static_cast<size_t>(C_out),
      weight_ohwi.data(),
      bias.empty() ? nullptr : bias.data(),
      out_min,
      out_max,
      /*flags=*/0,
      /*code_cache=*/nullptr,
      /*weights_cache=*/nullptr,
      &op);
  if (create_status != xnn_status_success || op == nullptr) {
    xnn_delete_operator(op);
    throw std::runtime_error(
        "xnn_create_convolution2d_nhwc_f32 failed with status " +
        std::to_string(static_cast<int>(create_status)));
  }

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  size_t out_h = 0;
  size_t out_w = 0;
  const xnn_status reshape_status = xnn_reshape_convolution2d_nhwc_f32(
      op,
      /*batch_size=*/1,
      static_cast<size_t>(H_in),
      static_cast<size_t>(W_in),
      &workspace_size,
      &workspace_alignment,
      &out_h,
      &out_w,
      /*threadpool=*/nullptr);
  if (reshape_status != xnn_status_success) {
    xnn_delete_operator(op);
    throw std::runtime_error(
        "xnn_reshape_convolution2d_nhwc_f32 failed with status " +
        std::to_string(static_cast<int>(reshape_status)));
  }

  std::vector<float> output_nhwc(static_cast<size_t>(C_out * H_out * W_out));
  // XNN_ALLOCATION_ALIGNMENT-aligned workspace (a bare vector is only
  // max_align_t aligned).
  std::vector<char> workspace(workspace_size + workspace_alignment);
  void* ws_ptr = workspace.data();
  if (workspace_alignment > 0) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ws_ptr);
    const uintptr_t aligned =
        (addr + workspace_alignment - 1) & ~(workspace_alignment - 1);
    ws_ptr = reinterpret_cast<void*>(aligned);
  }

  const xnn_status setup_status = xnn_setup_convolution2d_nhwc_f32(
      op, ws_ptr, input_nhwc.data(), output_nhwc.data());
  if (setup_status != xnn_status_success) {
    xnn_delete_operator(op);
    throw std::runtime_error(
        "xnn_setup_convolution2d_nhwc_f32 failed with status " +
        std::to_string(static_cast<int>(setup_status)));
  }
  const xnn_status run_status = xnn_run_operator(op, /*threadpool=*/nullptr);
  if (run_status != xnn_status_success) {
    xnn_delete_operator(op);
    throw std::runtime_error(
        "xnn_run_operator failed with status " +
        std::to_string(static_cast<int>(run_status)));
  }
  xnn_delete_operator(op);

  // NHWC -> NCHW output.
  std::vector<float> output_nchw(static_cast<size_t>(C_out * H_out * W_out));
  for (int64_t c = 0; c < C_out; ++c) {
    for (int64_t h = 0; h < H_out; ++h) {
      for (int64_t w = 0; w < W_out; ++w) {
        output_nchw[static_cast<size_t>(c * (H_out * W_out) + h * W_out + w)] =
            output_nhwc[static_cast<size_t>((h * W_out + w) * C_out + c)];
      }
    }
  }
  return output_nchw;
}

//
// Graph construction
//

// Handles needed to drive a built conv graph through resizes. ComputeGraph is
// move-only; hold it by value alongside the input/output handles.
struct ConvGraph {
  ComputeGraph graph;
  IOValueRef input;
  ValueRef staging_out;
  std::vector<float> weight_nchw; // kept alive for the reference
  std::vector<float> bias; // empty if cfg.has_bias == false
};

// Build the tiled-im2col conv graph at cfg's INITIAL (upper-bound) input shape.
ConvGraph build_graph(const Conv2dTestConfig& cfg) {
  GraphConfig graph_config;
  // Force resize fns to run on every execute() (they also run because the input
  // shape changes; this just matches how the runtime exercises the path).
  graph_config.force_resize = true;

  std::vector<float> weight_nchw = rand_floats(
      numel({cfg.out_channels, cfg.in_channels, cfg.kernel_h, cfg.kernel_w}),
      11);
  std::vector<float> bias = cfg.has_bias
      ? rand_floats(static_cast<size_t>(cfg.out_channels), 22)
      : std::vector<float>{};

  ConvGraph cg{
      ComputeGraph(graph_config),
      IOValueRef{},
      kDummyValueRef,
      std::move(weight_nchw),
      std::move(bias)};
  ComputeGraph& graph = cg.graph;

  // Conv requires channels-packed input/output (check_conv_args). io_storage
  // selects the input/output tensor storage. Build at the upper-bound (init)
  // shape so the fixed build-time tile count covers every resized shape.
  cg.input = graph.add_input_tensor(
      {1, cfg.in_channels, cfg.init_h, cfg.init_w},
      vkapi::kFloat,
      cfg.io_storage,
      utils::kChannelsPacked);
  const ValueRef r_weight = graph.add_tensorref(
      {cfg.out_channels, cfg.in_channels, cfg.kernel_h, cfg.kernel_w},
      vkapi::kFloat,
      cg.weight_nchw.data());

  ValueRef r_bias = kDummyValueRef;
  if (cfg.has_bias) {
    r_bias =
        graph.add_tensorref({cfg.out_channels}, vkapi::kFloat, cg.bias.data());
  } else {
    r_bias = graph.add_none();
  }

  const ValueRef r_stride =
      graph.add_scalar_list<int64_t>({cfg.stride_h, cfg.stride_w});
  const ValueRef r_padding =
      graph.add_scalar_list<int64_t>({cfg.padding_h, cfg.padding_w});
  const ValueRef r_dilation =
      graph.add_scalar_list<int64_t>({cfg.dilation_h, cfg.dilation_w});

  const int64_t H_out_max = conv_out_dim(
      cfg.init_h, cfg.kernel_h, cfg.stride_h, cfg.padding_h, cfg.dilation_h);
  const int64_t W_out_max = conv_out_dim(
      cfg.init_w, cfg.kernel_w, cfg.stride_w, cfg.padding_w, cfg.dilation_w);
  const ValueRef r_out = graph.add_tensor(
      {1, cfg.out_channels, H_out_max, W_out_max},
      vkapi::kFloat,
      cfg.io_storage,
      utils::kChannelsPacked);

  // Route straight to conv2d_gemm_impl with a forced im2col storage so the test
  // is device-independent (the registered op auto-selects storage per device).
  conv2d_gemm_impl(
      graph,
      cg.input.value,
      r_weight,
      r_bias,
      r_stride,
      r_padding,
      r_dilation,
      r_out,
      /*clamp_out=*/false,
      /*out_min_val=*/0.0f,
      /*out_max_val=*/0.0f,
      cfg.im2col_storage);

  cg.staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  return cg;
}

//
// Test driver
//

void run_dynamic_conv2d_resize_test(const Conv2dTestConfig& cfg) {
  // Attach the config to every assertion in this test (gtest prints active
  // SCOPED_TRACEs on failure) so a failing run names the exact config +
  // storage.
  SCOPED_TRACE(to_string(cfg));
  ASSERT_EQ(cfg.groups, 1) << "conv2d_gemm_impl only supports groups == 1";

  ConvGraph cg = build_graph(cfg);
  TensorFactory<ScalarType::Float> tf;

  // The build shape is the upper bound; the first run is at the initial shape,
  // followed by each resized shape (each <= init per dim, asserted below).
  std::vector<std::pair<int64_t, int64_t>> shapes;
  shapes.emplace_back(cfg.init_h, cfg.init_w);
  for (const auto& hw : cfg.resize_hw) {
    shapes.push_back(hw);
  }

  unsigned seed = 100;
  for (const auto& hw : shapes) {
    const int64_t H = hw.first;
    const int64_t W = hw.second;
    ASSERT_LE(H, cfg.init_h)
        << "resized H must be <= build-time upper bound init_h";
    ASSERT_LE(W, cfg.init_w)
        << "resized W must be <= build-time upper bound init_w";

    const int64_t H_out = conv_out_dim(
        H, cfg.kernel_h, cfg.stride_h, cfg.padding_h, cfg.dilation_h);
    const int64_t W_out = conv_out_dim(
        W, cfg.kernel_w, cfg.stride_w, cfg.padding_w, cfg.dilation_w);
    const std::vector<int64_t> in_shape = {1, cfg.in_channels, H, W};
    const std::vector<int64_t> out_shape = {1, cfg.out_channels, H_out, W_out};
    const size_t in_n = numel(in_shape);
    const size_t out_n = numel(out_shape);

    std::vector<float> x_data = rand_floats(in_n, seed++);
    std::vector<float> ref =
        conv2d_ref_xnnpack(x_data, cg.weight_nchw, cg.bias, cfg, H, W);

    cg.graph.resize_input(0, in_shape);
    cg.graph.propagate_resize();
    cg.graph.maybe_cast_and_copy_into_staging(
        cg.input.staging, x_data.data(), in_n, vkapi::kFloat);

    cg.graph.execute();

    std::vector<float> vk_data(out_n);
    cg.graph.maybe_cast_and_copy_from_staging(
        cg.staging_out, vk_data.data(), out_n, vkapi::kFloat);

    Tensor ref_t = tf.make(to_int32(out_shape), ref);
    Tensor vk_t = tf.make(to_int32(out_shape), vk_data);
    EXPECT_TENSOR_CLOSE_WITH_TOL(ref_t, vk_t, 1e-3, 1e-3)
        << "Mismatch at resized H=" << H << " W=" << W << " (H_out=" << H_out
        << ", W_out=" << W_out << ") for " << to_string(cfg);
  }
}

// Sweep the resize test over the three im2col SCRATCH storage variants (buffer
// / texture2d / texture3d), reusing the caller's config (taken by value) and
// overriding only im2col_storage each iteration. SCOPED_TRACE / to_string
// identifies the im2col_storage of any failing variant. All three are
// supported, so this is a safe single in-process loop (no variant crashes).
void test_wrapper(Conv2dTestConfig cfg) {
  for (const auto im2col_storage :
       {utils::kBuffer, utils::kTexture2D, utils::kTexture3D}) {
    cfg.im2col_storage = im2col_storage;
    run_dynamic_conv2d_resize_test(cfg);
  }
}

} // namespace

TEST(VulkanConv2dGemmDynamicTest, im2col_storage_sweep_resize) {
  // A 128x128 build shape with C_in=64, 3x3 s1p1 at the production 16 MB budget
  // tiles into 3 output-height tiles (oh_tile=56). The resize sweep
  // 64/56/112/128 needs 2/1/2/3 active tiles — crossing tile boundaries down to
  // a single tile and back up to the bound, all <= 128. XNNPACK computes the
  // reference, so the medium shape stays cheap.
  //
  // io_storage (the conv input/output tensor storage) is pinned to kTexture3D:
  // conv2d_gemm I/O is texture3d-only. The conv shaders declare t_in / t_out as
  // texture3d and use ivec3 addressing (conv2d_im2col reads texelFetch(t_in,
  // ivec3); conv2d_gemm writes imageStore(t_out, ivec3)), so a buffer-backed
  // I/O tensor bound to a texture descriptor crashes the driver, and a
  // texture2d channels-packed 4D tensor has a different physical layout than
  // the ivec3 addressing assumes, producing numerically wrong output (both were
  // confirmed on Mali + Adreno). Only the im2col SCRATCH storage is
  // parameterized — the im2col / GEMM shaders DO have buffer / tex2d / tex3d
  // codegen variants for it. io_storage is therefore pinned to kTexture3D (and
  // the buffer / tex2d I/O combinations are intentionally NOT exercised)
  // until/unless buffer / tex2d conv-I/O shader variants are added; it stays an
  // explicit config field (printed by to_string) so the test is trivial to
  // extend when they land.
  //
  // im2col_storage is overwritten by test_wrapper for each scratch-storage
  // variant; the value set here is just a placeholder.
  const Conv2dTestConfig cfg{
      /*in_channels=*/64,
      /*out_channels=*/64,
      /*kernel_h=*/3,
      /*kernel_w=*/3,
      /*stride_h=*/1,
      /*stride_w=*/1,
      /*padding_h=*/1,
      /*padding_w=*/1,
      /*dilation_h=*/1,
      /*dilation_w=*/1,
      /*groups=*/1,
      /*has_bias=*/true,
      /*init_h=*/128,
      /*init_w=*/128,
      /*io_storage=*/utils::kTexture3D,
      /*im2col_storage=*/utils::kTexture3D,
      /*resize_hw=*/{{64, 128}, {56, 128}, {112, 128}, {128, 128}}};
  test_wrapper(cfg);
}
