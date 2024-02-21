/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/vulkan/graph/Arithmetic.h>
#include <ATen/native/vulkan/graph/Graph.h>

#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>
#include <executorch/backends/vulkan/schema_generated.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/profiler.h>

#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>
#include <type_traits>

namespace torch {
namespace executor {
namespace vulkan {
namespace {

// Flatbuffer types
using VkGraphPtr = const vkgraph::VkGraph*;
using VkNodePtr = const vkgraph::VkNode*;
using VkValuePtr = const vkgraph::VkValue*;
using VkTensorPtr = const vkgraph::VkTensor*;
using VkBytesPtr = const vkgraph::VkBytes*;

// Flatbuffer vector types
using VkValuesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkValue>>*;
using BytesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkBytes>>*;
using UIntVector = const flatbuffers::Vector<uint32_t>*;

const uint8_t* getConstantDataPtr(
    VkGraphPtr flatbuffer_graph,
    const int32_t buffer_idx,
    const uint8_t* constant_data) {
  VkBytesPtr constant_bytes = flatbuffer_graph->constants()->Get(buffer_idx);
  return constant_data + constant_bytes->offset();
}

using namespace at::native::vulkan;

class VulkanBackend final : public PyTorchBackendInterface {
 public:
  ~VulkanBackend() override = default;

  bool is_available() const override {
    return true;
  }

  arithmetic::OpType get_native_op_type(
      const vkgraph::VkArithmeticOpType& delegate_op_type) const {
    switch (delegate_op_type) {
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_add): {
        return arithmetic::OpType::ADD;
      }
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_sub): {
        return arithmetic::OpType::SUB;
      }
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_mul): {
        return arithmetic::OpType::MUL;
      }
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_div): {
        return arithmetic::OpType::DIV;
      }
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_floor_div): {
        return arithmetic::OpType::FLOOR_DIV;
      }
      case (vkgraph::VkArithmeticOpType::vk_arithmetic_op_type_pow): {
        return arithmetic::OpType::POW;
      }
    }
  }

  api::ScalarType get_scalar_type(
      const vkgraph::VkDatatype& vk_datatype) const {
    switch (vk_datatype) {
      case (vkgraph::VkDatatype::vk_datatype_fp32): {
        return api::kFloat;
      }
    }
  }

  ValueRef get_value_ref(
      const uint32_t value_id,
      VkGraphPtr flatbuffer_graph,
      ComputeGraph* compute_graph,
      std::unordered_map<uint32_t, ValueRef>& ref_mapping,
      VkValuesVector value_mapping,
      const uint8_t* constant_data) const {
    const std::unordered_map<uint32_t, ValueRef>::iterator found_ref =
        ref_mapping.find(value_id);

    if (found_ref != ref_mapping.end()) {
      return found_ref->second;
    }

    VkValuePtr vk_value = value_mapping->Get(value_id);
    VkTensorPtr vk_tensor = vk_value->value();

    ET_CHECK_MSG(
        vk_tensor->constant_buffer_idx() >= 0,
        "Only constant buffers are supported when adding tensors to compute graph (indicated by constant_buffer_idx == 0), but got constant_buffer_idx of %d",
        vk_tensor->constant_buffer_idx());

    const api::ScalarType& tensor_dtype =
        get_scalar_type(vk_tensor->datatype());

    UIntVector tensor_dims_fb = vk_tensor->dims();
    const std::vector<int64_t> tensor_dims_vector(
        tensor_dims_fb->cbegin(), tensor_dims_fb->cend());

    const uint8_t* tensor_data = getConstantDataPtr(
        flatbuffer_graph, vk_tensor->constant_buffer_idx(), constant_data);

    const ValueRef value_ref = compute_graph->add_tensorref(
        tensor_dims_vector, tensor_dtype, tensor_data);

    ref_mapping[value_id] = value_ref;

    return value_ref;
  }

  GraphConfig generate_config() const {
    const uint32_t submit_frequency = UINT32_MAX;

    const api::CommandPoolConfig cmd_config{
        4u, // cmdPoolInitialSize
        2u, // cmdPoolBatchSize
    };

    const api::DescriptorPoolConfig descriptor_pool_config{
        1024u, // descriptorPoolMaxSets
        1024u, // descriptorUniformBufferCount
        1024u, // descriptorStorageBufferCount
        1024u, // descriptorCombinedSamplerCount
        1024u, // descriptorStorageImageCount
        32u, // descriptorPileSizes
    };

    const api::QueryPoolConfig query_pool_config{};

    const api::ContextConfig context_config{
        submit_frequency, // cmdSubmitFrequency
        cmd_config, // cmdPoolConfig
        descriptor_pool_config, // descriptorPoolConfig
        query_pool_config, // queryPoolConfig
    };

    const GraphConfig graph_config{
        context_config,
    };

    return graph_config;
  }

  __ET_NODISCARD Error
  compileModel(const void* buffer_pointer, ComputeGraph* compute_graph) const {
    Result<VulkanDelegateHeader> header =
        VulkanDelegateHeader::Parse(buffer_pointer);
    const uint8_t* flatbuffer_data = nullptr;
    const uint8_t* constant_data = nullptr;

    if (header.ok()) {
      const uint8_t* buffer_start =
          reinterpret_cast<const uint8_t*>(buffer_pointer);
      flatbuffer_data = buffer_start + header->flatbuffer_offset;
      constant_data = buffer_start + header->bytes_offset;
    } else {
      ET_LOG(Error, "VulkanDelegateHeader may be corrupt");
      return header.error();
    }

    ET_CHECK_OR_RETURN_ERROR(
        vkgraph::VkGraphBufferHasIdentifier(flatbuffer_data),
        DelegateInvalidCompatibility,
        "Vulkan Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(flatbuffer_data),
        vkgraph::VkGraphIdentifier());

    VkGraphPtr flatbuffer_graph = vkgraph::GetVkGraph(flatbuffer_data);

    // Mapping from serialized VkValue ids to compute graph ValueRefs
    // This will be populated as the compute graph is built
    std::unordered_map<uint32_t, ValueRef> ref_mapping;

    // A vector which acts as a mapping from VkValue ids (vector indices) to
    // VkValues
    VkValuesVector value_mapping = flatbuffer_graph->values();

    // 1. Add all inputs (and corresponding tensors) to the compute graph
    UIntVector input_ids = flatbuffer_graph->input_ids();

    for (size_t input_index = 0; input_index < input_ids->size();
         input_index++) {
      const uint32_t input_id = input_ids->Get(input_index);
      VkValuePtr input_vk_value = value_mapping->Get(input_id);

      VkTensorPtr input_vk_tensor = input_vk_value->value();

      ET_CHECK_MSG(
          input_vk_tensor->constant_buffer_idx() < 0,
          "Expected constant buffer index for input at index %zu with id %d to be < 0 (since it is non-constant), but got: %d",
          input_index,
          input_id,
          input_vk_tensor->constant_buffer_idx());

      const api::ScalarType& input_dtype =
          get_scalar_type(input_vk_tensor->datatype());

      UIntVector input_dims_fb = input_vk_tensor->dims();
      const std::vector<int64_t> input_dims_vector(
          input_dims_fb->cbegin(), input_dims_fb->cend());

      const ValueRef input_ref = compute_graph->add_tensor(
          input_dims_vector, input_dtype, input_vk_tensor->mem_obj_id());

      ref_mapping[input_id] = input_ref;
      compute_graph->set_input_tensor(input_ref);
    }

    // 2. Add all ops to the graph
    for (VkNodePtr node : *(flatbuffer_graph->chain())) {
      const vkgraph::VkArithmeticNode* typed_node = node->node();

      const uint32_t input1_id = typed_node->input1_id();
      const uint32_t input2_id = typed_node->input2_id();
      const uint32_t output_id = typed_node->output_id();

      const ValueRef input1_ref = get_value_ref(
          input1_id,
          flatbuffer_graph,
          compute_graph,
          ref_mapping,
          value_mapping,
          constant_data);

      const ValueRef input2_ref = get_value_ref(
          input2_id,
          flatbuffer_graph,
          compute_graph,
          ref_mapping,
          value_mapping,
          constant_data);

      const ValueRef output_ref = add_arithmetic_node(
          *compute_graph,
          input1_ref,
          input2_ref,
          1.0,
          get_native_op_type(typed_node->op_type()),
          value_mapping->Get(output_id)->value()->mem_obj_id());

      ref_mapping[output_id] = output_ref;
    }

    // 3. Add all outputs to the compute graph
    for (const uint32_t output_id : *flatbuffer_graph->output_ids()) {
      const ValueRef output_ref = ref_mapping[output_id];
      compute_graph->set_output_tensor(output_ref);
    }

    compute_graph->encode_prepack();
    compute_graph->prepack();

    compute_graph->encode_execute();

    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    ComputeGraph* compute_graph = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), ComputeGraph);

    new (compute_graph) ComputeGraph(generate_config());

    Error err = compileModel(processed->data(), compute_graph);

    if (err != Error::Ok) {
      return err;
    }

    return compute_graph;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    EXECUTORCH_SCOPE_PROF("VulkanBackend::execute");

    ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);

    const size_t num_inputs = compute_graph->inputs().size();
    for (size_t i = 0; i < num_inputs; i++) {
      compute_graph->copy_into_staging(
          compute_graph->inputs()[i],
          args[i]->toTensor().const_data_ptr(),
          args[i]->toTensor().numel());
    }

    compute_graph->execute();

    for (size_t i = 0; i < compute_graph->outputs().size(); i++) {
      // args holds inputs directly followed by outputs, so the i'th output
      // for compute_graph corresponds to the (i + num_inputs)'th arg
      compute_graph->copy_from_staging(
          compute_graph->outputs()[i],
          args[num_inputs + i]->toTensor().mutable_data_ptr(),
          args[num_inputs + i]->toTensor().numel());
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);
      // ComputeGraph is not trivially destructible. Since
      // this was constructed manually in init(), we must destroy it manually
      // here.
      compute_graph->~ComputeGraph();
    }
  }
};

auto cls = VulkanBackend();
Backend backend{"VulkanBackend", &cls};
static auto success_with_compiler = register_backend(backend);

} // namespace
} // namespace vulkan
} // namespace executor
} // namespace torch
