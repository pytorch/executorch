#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>

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
    static FlatbufferBuildResult build(const void* buffer, size_t size);
};

}
