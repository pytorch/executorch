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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/runtime.h>

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace vkcompute;
using AVParams = std::tuple<int, int, int, vkapi::ScalarType, int64_t>;

template <typename T>
void check_av(const AVParams& params) {
  const auto [group, dim, storage, dtype, variant] = params;
  constexpr int kv_heads = 2;
  constexpr int capacity = 1024;
  const int heads = group * kv_heads;
  const auto io = storage == 0 ? utils::kTexture3D : utils::kBuffer;
  const auto cache = storage == 2 ? utils::kBuffer : utils::kTexture3D;
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  config.set_storage_type_override(io);
  config.set_memory_layout_override(utils::kWidthPacked);
  ComputeGraph graph(config);
  const auto p = graph.add_input_tensor({1, heads, 8, capacity}, dtype);
  const auto v = graph.add_tensor(
      {1, capacity, kv_heads, dim}, dtype, cache, utils::kWidthPacked);
  const auto vstage = graph.set_input_tensor(v);
  const auto q = graph.add_tensor({1, 8, heads, dim}, dtype);
  const auto out = graph.add_tensor({1, 8, heads, dim}, dtype);
  const auto position = graph.add_symint(0);
  add_sdpa_compute_out_node(
      graph,
      p.value,
      v,
      q,
      v,
      position,
      out,
      SDPAMode::LLM,
      graph.add_scalar<int64_t>(variant));
  const auto ostage = graph.set_output_tensor(out);
  graph.prepare();
  graph.prepack();

  const std::vector<std::pair<int, int>> steps = {
      {1, 1}, {3, 1}, {65, 1}, {257, 1}, {511, 3}, {1024, 1}, {7, 1}, {1, 1}};
  for (const auto& [context, sequence] : steps) {
    SCOPED_TRACE(context);
    SCOPED_TRACE(sequence);
    const int ca = utils::align_up_4(context);
    const int sa = utils::align_up_4(sequence);
    graph.resize_input(0, {1, heads, sa, ca});
    graph.virtual_resize(q, {1, sequence, heads, dim});
    graph.set_symint(position, context - sequence);
    graph.propagate_resize();

    std::vector<T> values(
        capacity * kv_heads * dim, T(std::numeric_limits<float>::quiet_NaN()));
    for (int c = 0; c < context; ++c) {
      for (int h = 0; h < kv_heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          values[(c * kv_heads + h) * dim + d] =
              T(float((c * 13 + h * 7 + d * 3) % 31 - 15) / 16);
        }
      }
    }
    graph.maybe_cast_and_copy_into_staging(
        vstage, values.data(), values.size(), dtype);

    for (const bool one_hot : {false, true}) {
      SCOPED_TRACE(one_hot);
      std::vector<T> probabilities(heads * sa * ca, T(0.0f));
      for (int h = 0; h < heads; ++h) {
        for (int s = 0; s < sequence; ++s) {
          const int live = context - sequence + s + 1;
          const int offset = (h * sa + s) * ca;
          if (one_hot) {
            probabilities[offset + live - 1] = T(1.0f);
          } else {
            for (int i = 0; i < 64; ++i) {
              const int at = offset + (i * 37 + h * 11 + s * 7) % live;
              probabilities[at] = T(float(probabilities[at]) + 1.0f / 64);
            }
          }
        }
      }
      graph.maybe_cast_and_copy_into_staging(
          p.staging, probabilities.data(), probabilities.size(), dtype);
      graph.execute();
      std::vector<T> actual(sequence * heads * dim);
      graph.maybe_cast_and_copy_from_staging(
          ostage, actual.data(), actual.size(), dtype);

      // These dyadic inputs keep every partial sum exactly representable in
      // FP16, including texture stores with implementation-defined rounding.
      for (int s = 0; s < sequence; ++s) {
        for (int h = 0; h < heads; ++h) {
          for (int d = 0; d < dim; ++d) {
            double reference = 0;
            for (int c = 0; c < context; ++c) {
              reference += double(float(probabilities[(h * sa + s) * ca + c])) *
                  double(float(values[(c * kv_heads + h / group) * dim + d]));
            }
            ASSERT_EQ(
                double(float(actual[(s * heads + h) * dim + d])), reference)
                << "row=" << s << " head=" << h << " dim=" << d;
          }
        }
      }
    }
  }
}

class VulkanSDPAAVTest : public ::testing::TestWithParam<AVParams> {};

TEST_P(VulkanSDPAAVTest, MatchesCPUAcrossDecodeAndPrefill) {
  executorch::runtime::runtime_init();
  if (std::get<3>(GetParam()) == vkapi::kHalf) {
    check_av<executorch::aten::Half>(GetParam());
  } else {
    check_av<float>(GetParam());
  }
}

INSTANTIATE_TEST_SUITE_P(
    GroupedAttention,
    VulkanSDPAAVTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5, 6, 7, 8),
        ::testing::Values(12, 128),
        ::testing::Values(0, 1, 2),
        ::testing::Values(vkapi::kFloat, vkapi::kHalf),
        ::testing::Values(
            kShaderOverrideForceBase,
            kShaderOverrideForceTile2)));

} // namespace
