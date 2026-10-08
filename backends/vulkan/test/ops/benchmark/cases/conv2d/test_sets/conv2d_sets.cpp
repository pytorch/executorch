// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

struct Conv2dCaseGroup {
  std::vector<Conv2dTestConfig> configs;
  std::vector<vkapi::ScalarType> dtypes;
  std::vector<std::string> impls;
};

std::vector<TestCase> generate_test_cases(
    const std::vector<Conv2dCaseGroup>& groups) {
  std::vector<TestCase> test_cases;

  for (const auto& group : groups) {
    for (const auto& config : group.configs) {
      for (auto dtype : group.dtypes) {
        for (const auto& impl : group.impls) {
          test_cases.push_back(create_conv2d_test_case(
              config, dtype, kStorageType, kMemoryLayout, impl));
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("conv2d", "correctness") {
  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dTestConfig> accuracy_configs = {
      // 3x3 stride=1 pad=1 same-channels (the bottleneck pattern in TinyCNN)
      {InputDims(1, 8, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      {InputDims(1, 16, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // 3x3 stride=2 (downsample) with channel expansion
      {InputDims(1, 8, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=1 with channel reduction
      {InputDims(1, 16, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // Non-multiple-of-4 channels
      {InputDims(1, 11, 8, 8),
       13,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3-channel input (like RGB stem)
      {InputDims(1, 3, 16, 16),
       8,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Small shapes used to exercise each im2col intermediate-storage variant
  // (buffer / texture2d / texture3d) deterministically and independently of
  // the device's auto-selection. All dims < kRefDimSizeLimit so both the FP32
  // and FP16 references validate them. The first two shapes keep M (= H_out *
  // W_out) tiny, so the im2col intermediate [1, K_total, H_out, W_out] is small
  // and the forced texture3d variant never exercises the large-M texture3d
  // Z-layout the auto-selector actually falls back to. The third shape (48x48)
  // is included specifically so the forced im2col_tex3d variant exercises the
  // texture3d layout (K4 along Z, many spatial rows) at a non-trivial M, while
  // still keeping every dim under kRefDimSizeLimit for reference validation.
  std::vector<Conv2dTestConfig> per_variant_configs = {
      // 3x3 s1 p1, channels multiple of 4
      {InputDims(1, 16, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // Non-multiple-of-4 channels exercise the Cin padding path
      {InputDims(1, 11, 12, 12),
       13,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // Larger spatial extent (M = 48*48 = 2304) exercises the texture3d im2col
      // layout [1, K_total, H_out, W_out] with K4 along Z at a non-trivial M,
      // while all dims stay < kRefDimSizeLimit so both FP32 and FP16 references
      // validate it. C_in=16 keeps K = 3*3*16 = 144 (same as the 16x16 case)
      // so FP16 accumulation stays within tolerance.
      {InputDims(1, 16, 48, 48),
       16,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
  };

  // Implementation variants exercised for every small ACCU shape:
  //   ""       -> heuristic-routed (should_use_conv2d_im2col picks direct on
  //               Adreno for small c_out, im2col on Mali)
  //   "im2col" -> forced im2col/GEMM path
  //   "direct" -> forced direct sliding-window path (force_direct=true)
  // Including "direct" guarantees the direct shader gets reference-checked on
  // BOTH devices — without it, Mali would always route "" to im2col and never
  // exercise the direct path.
  const std::vector<std::string> impls = {"", "im2col", "direct"};
  // Forced-storage im2col variants for the per-variant ACCU coverage.
  const std::vector<std::string> forced_storage_impls = {
      "im2col_buffer", "im2col_tex2d", "im2col_tex3d"};

  // FP16 small shapes get a real reference check (gated in
  // conv2d_reference_impl); we run both dtypes so we catch correctness
  // regressions in either path. Large-K half stays timing-only via the
  // reference's PERF-shape throw.
  const std::vector<vkapi::ScalarType> accu_dtypes = {
      vkapi::kFloat, vkapi::kHalf};

  const std::vector<Conv2dCaseGroup> groups = {
      // Accuracy test cases for all impls and both dtypes.
      {accuracy_configs, accu_dtypes, impls},
      // Per-variant forced-storage ACCU cases (FP32 and FP16) so all three
      // im2col intermediate-storage variants get deterministic,
      // device-independent, reference-checked coverage at small K.
      {per_variant_configs, accu_dtypes, forced_storage_impls},
  };
  return {
      generate_test_cases(groups),
      conv2d_reference_impl,
      conv2d_flop_calculator,
      /*warmup_runs = */ 5,
      /*benchmark_runs = */ 20};
}

REGISTER_TEST_CASE_SET("conv2d", "performance") {
  // TinyCNN depth estimator hotspots (from profiling
  // UNTRAINED_TinyCNNDepthEstimatorRealTime_Vulkan.pte).
  // Each entry lists (C_in, H, W) -> C_out, all 3x3 stride=1 pad=1 unless
  // noted. Together the first 6 entries account for ~89% of all conv time.
  std::vector<Conv2dTestConfig> perf_configs = {
      // #1: 21.25% — (1,128,36,48)->(1,128,36,48)
      {InputDims(1, 128, 36, 48),
       128,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #2: 20.68% — (1,256,18,24)->(1,256,18,24)
      {InputDims(1, 256, 18, 24),
       256,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #3: 20.01% — (1,64,72,96)->(1,64,72,96)
      {InputDims(1, 64, 72, 96),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #4: 13.25% — (1,32,144,192)->(1,32,144,192)
      {InputDims(1, 32, 144, 192),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #5: 6.74% — (1,64,36,48)->(1,64,36,48)
      {InputDims(1, 64, 36, 48),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #6: 5.90% — (1,32,72,96)->(1,32,72,96)
      {InputDims(1, 32, 72, 96),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // Secondary cases
      // 3x3 stride=2 downsample with channel expansion: 1.52%
      {InputDims(1, 32, 72, 96),
       128,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=1 same-shape, smaller spatial: 1.51%
      {InputDims(1, 128, 18, 24),
       128,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // 3x3 stride=1, channel reduction
      {InputDims(1, 128, 18, 24),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 64, 36, 48),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=2 downsample, same channels
      {InputDims(1, 32, 72, 96),
       32,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 64, 36, 48),
       64,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // RGB stem
      {InputDims(1, 3, 144, 192),
       32,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Boundary pair straddling the should_use_conv2d_im2col() c_out >= 128
  // routing threshold. Spatial dims are tiny (8x8) so the FP32 float reference
  // stays cheap, but c_out = 64 / 128 are both >= kRefDimSizeLimit, so these
  // get the PERF label. FP32 PERF cases are still numerically VERIFIED (the
  // reference's invalid_argument throw that skips the check only fires for
  // half), so both implementations are cross-checked against the float
  // reference at the boundary. Run all three impls: at c_out = 64 the heuristic
  // ("") picks direct on Adreno / im2col on Mali; at c_out = 128 it picks
  // im2col on both — and "direct"/"im2col" force each path regardless, proving
  // the two implementations agree at the boundary on either device.
  std::vector<Conv2dTestConfig> boundary_configs = {
      // c_out = 64 (< 128): below the threshold
      {InputDims(1, 16, 8, 8),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // c_out = 128 (== 128): at/above the threshold
      {InputDims(1, 16, 8, 8),
       128,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  const std::vector<std::string> impls = {"", "im2col", "direct"};

  const std::vector<Conv2dCaseGroup> groups = {
      // The c_out boundary pair (FP32 only) through all three impls. FP32 PERF
      // cases are reference-VERIFIED, so the direct and im2col paths are both
      // cross-checked against the float reference at the routing threshold.
      {boundary_configs, {vkapi::kFloat}, impls},
      // Performance test cases (float and half) for all impls.
      {perf_configs, {vkapi::kFloat, vkapi::kHalf}, impls},
  };
  return {
      generate_test_cases(groups),
      conv2d_reference_impl,
      conv2d_flop_calculator,
      /*warmup_runs = */ 5,
      /*benchmark_runs = */ 20};
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
