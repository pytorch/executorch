/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/vulkan/graph/Arithmetic.h>
#include <ATen/native/vulkan/graph/Graph.h>
#include <executorch/backends/vulkan/serialization/schema/schema_generated.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/profiler.h>
#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>
#include <type_traits>

namespace torch {
namespace executor {

class VulkanBackend final : public PyTorchBackendInterface {
 public:
  ~VulkanBackend() override = default;

  bool is_available() const override {
    return true;
  }

  at::native::vulkan::arithmetic::OpType get_native_op_type(
      const at::vulkan::delegate::VkArithmeticOpType& delegate_op_type) const {
    switch (delegate_op_type) {
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_add): {
        return at::native::vulkan::arithmetic::OpType::ADD;
      }
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_sub): {
        return at::native::vulkan::arithmetic::OpType::SUB;
      }
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_mul): {
        return at::native::vulkan::arithmetic::OpType::MUL;
      }
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_div): {
        return at::native::vulkan::arithmetic::OpType::DIV;
      }
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_floor_div): {
        return at::native::vulkan::arithmetic::OpType::FLOOR_DIV;
      }
      case (at::vulkan::delegate::VkArithmeticOpType::
                vk_arithmetic_op_type_pow): {
        return at::native::vulkan::arithmetic::OpType::POW;
      }
    }
  }

  at::native::vulkan::api::ScalarType get_scalar_type(
      const at::vulkan::delegate::VkDatatype& vk_datatype) const {
    switch (vk_datatype) {
      case (at::vulkan::delegate::VkDatatype::vk_datatype_fp32): {
        return at::native::vulkan::api::kFloat;
      }
    }
  }

  at::native::vulkan::ValueRef get_value_ref(
      const uint32_t value_id,
      at::native::vulkan::ComputeGraph* compute_graph,
      std::unordered_map<uint32_t, at::native::vulkan::ValueRef>& ref_mapping,
      const flatbuffers_fbsource::Vector<
          flatbuffers_fbsource::Offset<at::vulkan::delegate::VkValue>>*
          value_mapping,
      const flatbuffers_fbsource::Vector<
          flatbuffers_fbsource::Offset<at::vulkan::delegate::Buffer>>*
          constant_buffer) const {
    const std::unordered_map<uint32_t, at::native::vulkan::ValueRef>::iterator
        found_ref = ref_mapping.find(value_id);

    if (found_ref != ref_mapping.end()) {
      return found_ref->second;
    }

    const at::vulkan::delegate::VkValue* vk_value =
        value_mapping->Get(value_id);

    const at::vulkan::delegate::VkTensor* vk_tensor = vk_value->value();

    ET_CHECK_MSG(
        vk_tensor->constant_buffer_idx() != 0,
        "Only constant buffers are supported when adding tensors to compute graph (indicated by constant_buffer_idx == 0), but got constant_buffer_idx of %d",
        vk_tensor->constant_buffer_idx());

    const at::native::vulkan::api::ScalarType& tensor_dtype =
        get_scalar_type(vk_tensor->datatype());

    const flatbuffers_fbsource::Vector<uint32_t>* tensor_dims_fb =
        vk_tensor->dims();
    const std::vector<int64_t> tensor_dims_vector(
        tensor_dims_fb->cbegin(), tensor_dims_fb->cend());

    const uint8_t* tensor_data =
        constant_buffer->Get(vk_tensor->constant_buffer_idx())
            ->storage()
            ->data();

    const at::native::vulkan::ValueRef value_ref = compute_graph->add_tensorref(
        tensor_dims_vector, tensor_dtype, tensor_data);

    ref_mapping[value_id] = value_ref;

    return value_ref;
  }

  at::native::vulkan::GraphConfig generate_config() const {
    const uint32_t submit_frequency = UINT32_MAX;

    const at::native::vulkan::api::CommandPoolConfig cmd_config{
        4u, // cmdPoolInitialSize
        2u, // cmdPoolBatchSize
    };

    const at::native::vulkan::api::DescriptorPoolConfig descriptor_pool_config{
        1024u, // descriptorPoolMaxSets
        1024u, // descriptorUniformBufferCount
        1024u, // descriptorStorageBufferCount
        1024u, // descriptorCombinedSamplerCount
        1024u, // descriptorStorageImageCount
        32u, // descriptorPileSizes
    };

    const at::native::vulkan::api::QueryPoolConfig query_pool_config{};

    const at::native::vulkan::api::ContextConfig context_config{
        submit_frequency, // cmdSubmitFrequency
        cmd_config, // cmdPoolConfig
        descriptor_pool_config, // descriptorPoolConfig
        query_pool_config, // queryPoolConfig
    };

    const at::native::vulkan::GraphConfig graph_config{
        context_config,
    };

    return graph_config;
  }

  __ET_NODISCARD Error compileModel(
      const void* buffer_pointer,
      at::native::vulkan::ComputeGraph* compute_graph) const {
    const at::vulkan::delegate::VkGraph* flatbuffer_graph =
        at::vulkan::delegate::GetVkGraph(buffer_pointer);

    // Mapping from serialized VkValue ids to compute graph ValueRefs
    // This will be populated as the compute graph is built
    std::unordered_map<uint32_t, at::native::vulkan::ValueRef> ref_mapping;

    // A vector which acts as a mapping from VkValue ids (vector indices) to
    // VkValues
    const flatbuffers_fbsource::Vector<
        flatbuffers_fbsource::Offset<at::vulkan::delegate::VkValue>>*
        value_mapping = flatbuffer_graph->vkvalues();

    // 1. Add all inputs (and corresponding tensors) to the compute graph
    const flatbuffers_fbsource::Vector<uint32_t>* input_ids =
        flatbuffer_graph->input_ids();

    for (size_t input_index = 0; input_index < input_ids->size();
         input_index++) {
      const uint32_t input_id = input_ids->Get(input_index);
      const at::vulkan::delegate::VkValue* input_vk_value =
          value_mapping->Get(input_id);

      const at::vulkan::delegate::VkTensor* input_vk_tensor =
          input_vk_value->value();

      ET_CHECK_MSG(
          input_vk_tensor->constant_buffer_idx() == 0,
          "Expected constant buffer index for input at index %zu with id %d to be 0 (since it is non-constant), but got: %d",
          input_index,
          input_id,
          input_vk_tensor->constant_buffer_idx());

      const at::native::vulkan::api::ScalarType& input_dtype =
          get_scalar_type(input_vk_tensor->datatype());

      const flatbuffers_fbsource::Vector<uint32_t>* input_dims_fb =
          input_vk_tensor->dims();
      const std::vector<int64_t> input_dims_vector(
          input_dims_fb->cbegin(), input_dims_fb->cend());

      const at::native::vulkan::ValueRef input_ref =
          compute_graph->add_tensor(input_dims_vector, input_dtype);

      ref_mapping[input_id] = input_ref;
      compute_graph->set_input_tensor(input_ref);
    }

    // 2. Add all ops to the graph
    const flatbuffers_fbsource::Vector<
        flatbuffers_fbsource::Offset<at::vulkan::delegate::Buffer>>*
        constant_buffer = flatbuffer_graph->constant_buffer();

    for (const at::vulkan::delegate::VkNode* node :
         *flatbuffer_graph->vknodes()) {
      const at::vulkan::delegate::VkArithmeticNode* typed_node = node->node();

      const uint32_t input1_id = typed_node->input1_id();
      const uint32_t input2_id = typed_node->input2_id();
      const uint32_t output_id = typed_node->output_id();

      const at::native::vulkan::ValueRef input1_ref = get_value_ref(
          input1_id,
          compute_graph,
          ref_mapping,
          value_mapping,
          constant_buffer);

      const at::native::vulkan::ValueRef input2_ref = get_value_ref(
          input2_id,
          compute_graph,
          ref_mapping,
          value_mapping,
          constant_buffer);

      const at::native::vulkan::ValueRef output_ref =
          at::native::vulkan::add_arithmetic_node(
              *compute_graph,
              input1_ref,
              input2_ref,
              1.0,
              get_native_op_type(typed_node->op_type()));

      ref_mapping[output_id] = output_ref;
    }

    // 3. Add all outputs to the compute graph
    for (const uint32_t output_id : *flatbuffer_graph->output_ids()) {
      const at::native::vulkan::ValueRef output_ref = ref_mapping[output_id];
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
    ET_CHECK_OR_RETURN_ERROR(
        at::vulkan::delegate::VkGraphBufferHasIdentifier(processed->data()),
        DelegateInvalidCompatibility,
        "Vulkan Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
        flatbuffers_fbsource::GetBufferIdentifier(processed->data()),
        at::vulkan::delegate::VkGraphIdentifier());

    at::native::vulkan::ComputeGraph* compute_graph =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
            context.get_runtime_allocator(), at::native::vulkan::ComputeGraph);

    new (compute_graph) at::native::vulkan::ComputeGraph(generate_config());

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

    at::native::vulkan::ComputeGraph* compute_graph =
        static_cast<at::native::vulkan::ComputeGraph*>(handle);

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
      at::native::vulkan::ComputeGraph* compute_graph =
          static_cast<at::native::vulkan::ComputeGraph*>(handle);
      // at::native::vulkan::ComputeGraph is not trivially destructible. Since
      // this was constructed manually in init(), we must destroy it manually
      // here.
      compute_graph->~ComputeGraph();
    }
  }
};

namespace {
auto cls = VulkanBackend();
Backend backend{"VulkanBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace executor
} // namespace torch
