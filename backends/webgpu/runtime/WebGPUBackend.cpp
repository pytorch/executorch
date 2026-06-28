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
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <vector>

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
using executorch::runtime::resize_tensor;
using executorch::runtime::Result;
using executorch::runtime::Span;

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
    graph->build(flatbuffer_data, constant_data, context.get_named_data_map());
  } catch (const std::exception& e) {
    ET_LOG(Error, "WebGPU graph build failed: %s", e.what());
    graph->~WebGPUGraph();
    return Error::DelegateInvalidCompatibility;
  }

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
  std::vector<InputData> inputs;
  inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    const auto& tensor = args[i]->toTensor();
    const bool host_is_int64 =
        tensor.scalar_type() == executorch::aten::ScalarType::Long;
    inputs.push_back({tensor.const_data_ptr(), tensor.nbytes(), host_is_int64});
  }
  // Fail loud as a runtime Error so a throw never crosses the backend boundary.
  try {
    // Dynamic shapes: shrink each input to its live sizes before upload
    // (mirrors Vulkan maybe_resize_input). No-op when unchanged, so a static
    // graph is byte-identical.
    for (size_t i = 0; i < num_inputs; i++) {
      const auto sizes = args[i]->toTensor().sizes();
      std::vector<int64_t> new_dims(sizes.begin(), sizes.end());
      graph->resize_input(graph->input_ids()[i], new_dims);
    }
    graph->copy_inputs(inputs);
    graph->update_symints_from_inputs(inputs);
    graph->propagate_resize();
    // Resize each output EValue to its live shape so the readback length is
    // correct (mirrors Vulkan maybe_resize_output).
    for (size_t i = 0; i < num_outputs; i++) {
      const auto& cd = graph->cur_dims(graph->output_ids()[i]);
      std::vector<executorch::aten::SizesType> osizes(cd.begin(), cd.end());
      Error e = resize_tensor(
          args[num_inputs + i]->toTensor(),
          ArrayRef<executorch::aten::SizesType>(osizes.data(), osizes.size()));
      if (e != Error::Ok) {
        ET_LOG(Error, "WebGPU: output %zu resize failed", i);
        return Error::Internal;
      }
    }
  } catch (const std::exception& e) {
    ET_LOG(Error, "WebGPU input/output resize / copy failed: %s", e.what());
    return Error::Internal;
  }

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
