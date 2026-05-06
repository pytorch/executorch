/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUBackend.h>
#include <executorch/backends/webgpu/runtime/WebGPUDelegateHeader.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#include <new>

namespace executorch {
namespace backends {
namespace webgpu {

// vkgraph namespace is declared at global scope in the generated FlatBuffer
// header

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::register_backend;
using executorch::runtime::Result;
using executorch::runtime::Span;

// Test-only global; overwritten on each init() call.
static WebGPUMemoryStats s_last_memory_stats_for_testing;

WebGPUMemoryStats get_last_memory_stats() {
  return s_last_memory_stats_for_testing;
}

bool WebGPUBackend::is_available() const {
  return true;
}

Result<DelegateHandle*> WebGPUBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // Allocate graph on the runtime allocator
  WebGPUGraph* graph =
      context.get_runtime_allocator()->allocateInstance<WebGPUGraph>();
  if (graph == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  new (graph) WebGPUGraph();

  // Parse header to locate flatbuffer and constant data
  Result<WebGPUDelegateHeader> header =
      WebGPUDelegateHeader::parse(processed->data());
  if (!header.ok()) {
    ET_LOG(Error, "WebGPUDelegateHeader may be corrupt");
    return header.error();
  }

  const uint8_t* buffer_start =
      reinterpret_cast<const uint8_t*>(processed->data());
  const uint8_t* flatbuffer_data = buffer_start + header->flatbuffer_offset;
  const uint8_t* constant_data = buffer_start + header->bytes_offset;

  // Verify FlatBuffer identifier
  if (!vkgraph::VkGraphBufferHasIdentifier(flatbuffer_data)) {
    ET_LOG(
        Error,
        "WebGPU delegate FlatBuffer identifier mismatch (expected VK00)");
    return Error::DelegateInvalidCompatibility;
  }

  try {
    graph->build(flatbuffer_data, constant_data);
  } catch (const std::exception& e) {
    ET_LOG(Error, "WebGPU graph build failed: %s", e.what());
    graph->~WebGPUGraph();
    return Error::DelegateInvalidCompatibility;
  }

  s_last_memory_stats_for_testing = graph->memory_stats();

  processed->Free();

  return graph;
}

Error WebGPUBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  WebGPUGraph* graph = static_cast<WebGPUGraph*>(handle);

  const size_t num_inputs = graph->input_ids().size();
  const size_t num_outputs = graph->output_ids().size();

  // Copy inputs from EValue tensors to GPU buffers
  std::vector<std::pair<const void*, size_t>> inputs;
  inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    const auto& tensor = args[i]->toTensor();
    inputs.emplace_back(tensor.const_data_ptr(), tensor.nbytes());
  }
  graph->copy_inputs(inputs);

  // Execute the compute graph
  graph->execute();

  // Copy outputs from GPU staging buffers to EValue tensor data pointers
  std::vector<std::pair<void*, size_t>> outputs;
  outputs.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    const size_t arg_idx = num_inputs + i;
    auto& tensor = args[arg_idx]->toTensor();
    outputs.emplace_back(tensor.mutable_data_ptr(), tensor.nbytes());
  }
  graph->copy_outputs(outputs);

  return Error::Ok;
}

void WebGPUBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    WebGPUGraph* graph = static_cast<WebGPUGraph*>(handle);
    graph->~WebGPUGraph();
  }
}

namespace {
auto cls = WebGPUBackend();
Backend backend{"VulkanBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace webgpu
} // namespace backends
} // namespace executorch
