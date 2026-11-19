// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <cstdint>
#include <functional>

namespace vkcompute::shader_profiling {

// Copies between two scratch buffers before each repetition, evicting the
// measured work's data from the GPU caches.
class CacheFlush {
 private:
  size_t first_;
  size_t end_;

 public:
  // Adds the copy's execute nodes, rounding bytes up to a multiple of 4 KiB.
  // Adds nothing when bytes is 0.
  CacheFlush(ComputeGraph& graph, uint64_t bytes);

  // Encodes the copy behind a full barrier. The caller must place a full
  // barrier between the copy and the measured work.
  void encode(ComputeGraph& graph) const;
};

Report measure(
    const Json& spec,
    ComputeGraph& graph,
    const std::function<void(uint32_t)>& encode_batch,
    const Json& work_spec = Json::object());

} // namespace vkcompute::shader_profiling
