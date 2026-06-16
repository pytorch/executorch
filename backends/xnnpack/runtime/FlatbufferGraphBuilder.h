#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace executorch::backends::xnnpack {

struct FlatbufferBuildResult {
  graph::Graph graph;
  std::vector<uint32_t> input_external_ids;
  std::vector<uint32_t> output_external_ids;
};

struct FlatbufferGraphBuilder {
  static executorch::runtime::Result<FlatbufferBuildResult> build(
      const void* buffer,
      size_t size,
      const executorch::runtime::NamedDataMap* named_data_map = nullptr);
};

} // namespace executorch::backends::xnnpack
