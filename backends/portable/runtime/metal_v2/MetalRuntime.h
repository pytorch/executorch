/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/GraphRuntime.h>

#include <memory>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Forward declarations - avoid including Metal headers in C++ files
class MetalStream;
class MetalOp;
class MetalOpRegistry;

//===----------------------------------------------------------------------===//
// MetalRuntime - GraphRuntime implementation using MetalStream
//===----------------------------------------------------------------------===//

class MetalRuntime : public portable::GraphRuntime {
public:
  MetalRuntime();
  ~MetalRuntime() override;

  //=== GraphRuntime interface ===

  const char* name() const override { return "MetalRuntime_v2"; }
  bool is_available() const override;

  bool has_op(const portable::OperatorCall& op, const portable::Graph& graph) const override;

  runtime::Error init(
      const portable::Graph& graph,
      runtime::ArrayRef<portable::ExecutionSegment> segments) override;

  runtime::Error initialize_constants(
      runtime::Span<const uint32_t> value_indices) override;

  runtime::Error initialize_buffers(
      runtime::Span<const uint32_t> value_indices) override;

  runtime::Error execute_segment(
      size_t segment_index,
      runtime::Span<runtime::EValue> values) override;

  runtime::Error upload_values(
      runtime::Span<const runtime::EValue> cpu_values,
      runtime::Span<const uint32_t> indices) override;

  runtime::Error download_values(
      runtime::Span<runtime::EValue> cpu_values,
      runtime::Span<const uint32_t> indices) override;

  void destroy() override;

private:
  MetalStream* stream_;
  const portable::Graph* graph_;
  std::vector<portable::ExecutionSegment> segments_;

  // Value index -> GPU buffer mapping
  std::unordered_map<uint32_t, void*> valueBuffers_;

  // Cached sizes for detecting reallocation needs
  std::unordered_map<uint32_t, size_t> cachedSizes_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
