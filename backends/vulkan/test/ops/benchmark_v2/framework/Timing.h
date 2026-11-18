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

// Copies between two scratch buffers of cache_flush_bytes before each
// repetition, evicting the measured work's data from the GPU caches.
class CacheFlush {
 public:
  // Adds the copy's execute nodes when the spec sets cache_flush_bytes.
  CacheFlush(ComputeGraph& graph, const Json& spec);

  // Encodes the copy behind a full barrier. The caller must place a full
  // barrier between the copy and the measured work.
  void encode(ComputeGraph& graph) const;

 private:
  size_t first_;
  size_t end_;
};

Report measure(
    const Json& spec,
    ComputeGraph& graph,
    const std::function<void(uint32_t)>& encode_batch,
    const Json& work_spec = Json::object());

} // namespace vkcompute::shader_profiling
