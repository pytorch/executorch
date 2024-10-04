/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>
#include <executorch/backends/vulkan/schema_generated.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#endif // ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/profiler.h>

#include <cstdio>
#include <cstdlib> /* strtol */
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace executorch {
namespace backends {
namespace vulkan {
namespace {

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::Result;

using namespace vkcompute;

// Flatbuffer types
using VkGraphPtr = const vkgraph::VkGraph*;
using OpCallPtr = const vkgraph::OperatorCall*;
using VkValuePtr = const vkgraph::VkValue*;
using VkTensorPtr = const vkgraph::VkTensor*;
using VkBytesPtr = const vkgraph::VkBytes*;

// Flatbuffer vector types
using VkValuesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkValue>>*;
using BytesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkBytes>>*;
using UIntVector = const flatbuffers::Vector<uint32_t>*;

const uint8_t* get_constant_data_ptr(
    VkGraphPtr flatbuffer_graph,
    const int32_t buffer_idx,
    const uint8_t* constant_data) {
  VkBytesPtr constant_bytes = flatbuffer_graph->constants()->Get(buffer_idx);
  return constant_data + constant_bytes->offset();
}

vkapi::ScalarType get_scalar_type(const vkgraph::VkDataType& vk_datatype) {
  switch (vk_datatype) {
    case vkgraph::VkDataType::BOOL:
      return vkapi::kBool;
    case vkgraph::VkDataType::UINT8:
      return vkapi::kByte;
    case vkgraph::VkDataType::INT8:
      return vkapi::kChar;
    case vkgraph::VkDataType::INT32:
      return vkapi::kInt;
    case vkgraph::VkDataType::FLOAT16:
      return vkapi::kHalf;
    case vkgraph::VkDataType::FLOAT32:
      return vkapi::kFloat;
  }
}

utils::StorageType get_storage_type(
    const vkgraph::VkStorageType& vk_storage_type) {
  switch (vk_storage_type) {
    case vkgraph::VkStorageType::BUFFER:
      return utils::kBuffer;
    case vkgraph::VkStorageType::TEXTURE_3D:
      return utils::kTexture3D;
    case vkgraph::VkStorageType::TEXTURE_2D:
      return utils::kTexture2D;
    default:
      break;
  }
  VK_THROW("Invalid storage type encountered!");
}

utils::GPUMemoryLayout get_memory_layout(
    const vkgraph::VkMemoryLayout& vk_memory_layout) {
  switch (vk_memory_layout) {
    case vkgraph::VkMemoryLayout::TENSOR_WIDTH_PACKED:
      return utils::kWidthPacked;
    case vkgraph::VkMemoryLayout::TENSOR_HEIGHT_PACKED:
      return utils::kHeightPacked;
    case vkgraph::VkMemoryLayout::TENSOR_CHANNELS_PACKED:
      return utils::kChannelsPacked;
    default:
      break;
  }
  VK_THROW("Invalid memory layout encountered!");
}

GraphConfig get_graph_config(ArrayRef<CompileSpec>& compile_specs) {
  GraphConfig config = GraphConfig();

  for (const CompileSpec& spec : compile_specs) {
    const uint8_t* value_data = (const uint8_t*)spec.value.buffer;
    const size_t value_size = spec.value.nbytes;
    if (strcmp(spec.key, "storage_type_override") == 0) {
      ET_CHECK_MSG(value_size == sizeof(int32_t), "Unexpected value size!");
      int value_as_int = static_cast<int>(getUInt32LE(value_data));
      utils::StorageType storage_type =
          static_cast<utils::StorageType>(value_as_int);

      config.set_storage_type_override(storage_type);
    }
    if (strcmp(spec.key, "memory_layout_override") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint32_t), "Unexpected value size!");
      uint32_t value_as_int = getUInt32LE(value_data);
      utils::GPUMemoryLayout memory_layout =
          static_cast<utils::GPUMemoryLayout>(value_as_int);

      config.set_memory_layout_override(memory_layout);
    }
  }
#ifdef ET_EVENT_TRACER_ENABLED
  config.enable_querypool = true;
#endif // ET_EVENT_TRACER_ENABLED
  return config;
}

class GraphBuilder {
  ComputeGraph* compute_graph_;
  VkGraphPtr flatbuffer_;
  const uint8_t* constant_data_;

  std::unordered_map<uint32_t, ValueRef> ref_mapping_;

 public:
  explicit GraphBuilder(
      ComputeGraph* compute_graph,
      VkGraphPtr flatbuffer,
      const uint8_t* constant_data)
      : compute_graph_(compute_graph),
        flatbuffer_(flatbuffer),
        constant_data_(constant_data),
        ref_mapping_() {}

  bool fb_id_exists(const uint32_t fb_id) {
    const std::unordered_map<uint32_t, ValueRef>::iterator found_ref =
        ref_mapping_.find(fb_id);

    return found_ref != ref_mapping_.end();
  }

  ValueRef get_fb_id_valueref(const uint32_t fb_id) {
    const std::unordered_map<uint32_t, ValueRef>::iterator found_ref =
        ref_mapping_.find(fb_id);

    ET_CHECK_MSG(
        found_ref != ref_mapping_.end(),
        "Trying to extract a value that hasn't yet been added to the graph.");

    return found_ref->second;
  }

  void add_tensor_to_graph(const uint32_t fb_id, VkTensorPtr tensor_fb) {
    const vkapi::ScalarType& dtype = get_scalar_type(tensor_fb->datatype());
    utils::StorageType storage_type =
        tensor_fb->storage_type() == vkgraph::VkStorageType::DEFAULT_STORAGE
        ? compute_graph_->suggested_storage_type()
        : get_storage_type(tensor_fb->storage_type());

    UIntVector dims_fb = tensor_fb->dims();
    const std::vector<int64_t> dims_vector(dims_fb->cbegin(), dims_fb->cend());

    utils::GPUMemoryLayout memory_layout =
        tensor_fb->memory_layout() == vkgraph::VkMemoryLayout::DEFAULT_LAYOUT
        ? compute_graph_->suggested_memory_layout(dims_vector)
        : get_memory_layout(tensor_fb->memory_layout());

    ValueRef ref;
    if (tensor_fb->constant_id() >= 0) {
      const uint8_t* tensor_data = get_constant_data_ptr(
          flatbuffer_, tensor_fb->constant_id(), constant_data_);

      ref = compute_graph_->add_tensorref(dims_vector, dtype, tensor_data);
    } else {
      ref = compute_graph_->add_tensor(
          dims_vector,
          dtype,
          storage_type,
          memory_layout,
          tensor_fb->mem_obj_id());
    }

    ref_mapping_[fb_id] = ref;
  }

  void add_none_to_graph(const uint32_t fb_id) {
    ValueRef ref = compute_graph_->add_none();
    ref_mapping_[fb_id] = ref;
  }

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, void>::type
  add_scalar_to_graph(const uint32_t fb_id, T value) {
    ValueRef ref = compute_graph_->add_scalar(value);
    ref_mapping_[fb_id] = ref;
  }

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, void>::type
  add_scalar_list_to_graph(const uint32_t fb_id, std::vector<T>&& value) {
    ValueRef ref = compute_graph_->add_scalar_list(std::move(value));
    ref_mapping_[fb_id] = ref;
  }

  void add_value_list_to_graph(
      const uint32_t fb_id,
      std::vector<ValueRef>&& value) {
    ValueRef ref = compute_graph_->add_value_list(std::move(value));
    ref_mapping_[fb_id] = ref;
  }

  void add_string_to_graph(const uint32_t fb_id, VkValuePtr value) {
    const auto fb_str = value->value_as_String()->string_val();
    std::string string(fb_str->cbegin(), fb_str->cend());
    ValueRef ref = compute_graph_->add_string(std::move(string));
    ref_mapping_[fb_id] = ref;
  }

  void add_value_to_graph(const uint32_t fb_id, VkValuePtr value) {
    ET_CHECK_MSG(
        !fb_id_exists(fb_id),
        "Trying to add a value that has already been added to the graph.");

    switch (value->value_type()) {
      case vkgraph::GraphTypes::Null:
        add_none_to_graph(fb_id);
        break;
      case vkgraph::GraphTypes::Int:
        add_scalar_to_graph(fb_id, value->value_as_Int()->int_val());
        break;
      case vkgraph::GraphTypes::Double:
        add_scalar_to_graph(fb_id, value->value_as_Double()->double_val());
        break;
      case vkgraph::GraphTypes::Bool:
        add_scalar_to_graph(fb_id, value->value_as_Bool()->bool_val());
        break;
      case vkgraph::GraphTypes::VkTensor:
        add_tensor_to_graph(fb_id, value->value_as_VkTensor());
        break;
      case vkgraph::GraphTypes::IntList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<int64_t>(
                value->value_as_IntList()->items()->cbegin(),
                value->value_as_IntList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::DoubleList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<double>(
                value->value_as_DoubleList()->items()->cbegin(),
                value->value_as_DoubleList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::BoolList:
        add_scalar_list_to_graph(
            fb_id,
            std::vector<bool>(
                value->value_as_BoolList()->items()->cbegin(),
                value->value_as_BoolList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::ValueList:
        add_value_list_to_graph(
            fb_id,
            std::vector<ValueRef>(
                value->value_as_ValueList()->items()->cbegin(),
                value->value_as_ValueList()->items()->cend()));
        break;
      case vkgraph::GraphTypes::String:
        add_string_to_graph(fb_id, value);
        break;
      default:
        ET_CHECK_MSG(false, "Unsupported value type.");
    }
  }

  void build_graph() {
    // First, add all values to the graph
    for (uint32_t fb_id = 0; fb_id < flatbuffer_->values()->size(); ++fb_id) {
      VkValuePtr value = flatbuffer_->values()->Get(fb_id);
      add_value_to_graph(fb_id, value);
    }

    // Parse the inputs
    for (const uint32_t fb_id : *flatbuffer_->input_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      compute_graph_->set_input_tensor(ref);
    }

    // Parse the operators
    uint32_t last_prepack_node_ct = 0;
    uint32_t last_execute_node_ct = 0;

    for (OpCallPtr op_call : *(flatbuffer_->chain())) {
      std::string op_name = op_call->name()->str();
      ET_CHECK_MSG(VK_HAS_OP(op_name), "Missing operator: %s", op_name.c_str());

      const std::vector<int> arg_fb_ids(
          op_call->args()->cbegin(), op_call->args()->cend());

      std::vector<ValueRef> args;
      for (const int arg_fb_id : arg_fb_ids) {
        args.push_back(get_fb_id_valueref(arg_fb_id));
      }

      auto vkFn = VK_GET_OP_FN(op_name);
      vkFn(*compute_graph_, args);
      if (compute_graph_->graphconfig().enable_querypool) {
        for (uint32_t idx_prepack = last_prepack_node_ct;
             idx_prepack < compute_graph_->prepack_nodes().size();
             idx_prepack++) {
          compute_graph_->prepack_nodes()[idx_prepack]->set_node_id(
              op_call->node_id());
        }
        for (uint32_t idx_execute = last_execute_node_ct;
             idx_execute < compute_graph_->execute_nodes().size();
             idx_execute++) {
          compute_graph_->execute_nodes()[idx_execute]->set_node_id(
              op_call->node_id());
        }
        last_prepack_node_ct = compute_graph_->prepack_nodes().size();
        last_execute_node_ct = compute_graph_->execute_nodes().size();
      }
    }

    // Parse the outputs
    for (const uint32_t fb_id : *flatbuffer_->output_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      compute_graph_->set_output_tensor(ref);
    }
  }
};

//
// Execution tools
//

bool maybe_resize_input(
    ComputeGraph* graph,
    const size_t input_i,
    executorch::aten::Tensor& et_tensor) {
  ValueRef in_tensor_ref = graph->inputs()[input_i].value;
  vTensorPtr in_tensor = graph->get_tensor(in_tensor_ref);

  ET_CHECK_MSG(
      et_tensor.dim() == in_tensor->sizes().size(),
      "Cannot resize input tensor: old ndim %zu does not match new ndim %zu",
      static_cast<size_t>(in_tensor->sizes().size()),
      static_cast<size_t>(et_tensor.dim()));

  bool should_resize = false;
  std::vector<int64_t> new_sizes(et_tensor.dim());
  for (size_t i = 0; i < et_tensor.dim(); i++) {
    if (in_tensor->sizes()[i] != et_tensor.sizes()[i]) {
      should_resize = true;
    }
    new_sizes.at(i) = et_tensor.sizes()[i];
  }

  if (should_resize) {
    graph->resize_input(input_i, new_sizes);
  }

  ET_CHECK_MSG(
      in_tensor->numel() == et_tensor.numel(),
      "Vulkan tensor numel %zu does not match ET tensor numel %zu",
      static_cast<size_t>(in_tensor->numel()),
      static_cast<size_t>(et_tensor.numel()));

  return should_resize;
}

void maybe_resize_output(
    ComputeGraph* graph,
    const size_t output_i,
    executorch::aten::Tensor& et_tensor) {
  ValueRef out_tensor_ref = graph->outputs()[output_i].value;
  vTensorPtr out_tensor = graph->get_tensor(out_tensor_ref);

  executorch::aten::SizesType new_output_size[kTensorDimensionLimit];
  size_t ndim = out_tensor->sizes().size();
  for (int i = 0; i < ndim; ++i) {
    new_output_size[i] = out_tensor->sizes()[i];
  }

  executorch::aten::ArrayRef<executorch::aten::SizesType> output_size{
      new_output_size, ndim};
  Error err = resize_tensor(et_tensor, output_size);

  ET_CHECK_MSG(err == Error::Ok, "Failed to resize output tensor.");
}

//
// VulkanBackend class
//

class VulkanBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~VulkanBackend() override = default;

  bool is_available() const override {
    // TODO(ssjia): replace with an actual Vulkan runtime availability check
    return true;
  }

  ET_NODISCARD Error
  compileModel(const void* buffer_pointer, ComputeGraph* compute_graph) const {
    Result<VulkanDelegateHeader> header =
        VulkanDelegateHeader::parse(buffer_pointer);

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

    GraphBuilder builder =
        GraphBuilder(compute_graph, flatbuffer_graph, constant_data);

    builder.build_graph();

    compute_graph->prepare();

    compute_graph->encode_prepack();
    compute_graph->prepack();

    compute_graph->encode_execute();

    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ComputeGraph* compute_graph = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), ComputeGraph);

    new (compute_graph) ComputeGraph(get_graph_config(compile_specs));

    Error err = compileModel(processed->data(), compute_graph);

    // This backend does not need its processed data after compiling the model.
    processed->Free();

    if (err != Error::Ok) {
      return err;
    }

    return compute_graph;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    EXECUTORCH_SCOPE_PROF("VulkanBackend::execute");

    ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);

    const size_t num_inputs = compute_graph->inputs().size();
    bool should_propagate_resize = false;
    for (size_t i = 0; i < num_inputs; i++) {
      bool was_resized =
          maybe_resize_input(compute_graph, i, args[i]->toTensor());
      should_propagate_resize = should_propagate_resize || was_resized;
      compute_graph->copy_into_staging(
          compute_graph->inputs()[i].staging,
          args[i]->toTensor().const_data_ptr(),
          args[i]->toTensor().numel());
    }

    if (should_propagate_resize) {
      compute_graph->propagate_resize();
    }
    compute_graph->execute();

    for (size_t i = 0; i < compute_graph->outputs().size(); i++) {
      maybe_resize_output(compute_graph, i, args[num_inputs + i]->toTensor());
      // args holds inputs directly followed by outputs, so the i'th output
      // for compute_graph corresponds to the (i + num_inputs)'th arg
      compute_graph->copy_from_staging(
          compute_graph->outputs()[i].staging,
          args[num_inputs + i]->toTensor().mutable_data_ptr(),
          args[num_inputs + i]->toTensor().numel());
    }

#ifdef ET_EVENT_TRACER_ENABLED
    EventTracer* event_tracer = context.event_tracer();
    compute_graph->context()->querypool().extract_results();
    for (const auto& tup :
         compute_graph->context()->querypool().get_shader_timestamp_data()) {
      std::string event_name =
          std::get<0>(tup) + "_" + std::to_string(std::get<1>(tup));
      event_tracer_log_profiling_delegate(
          event_tracer,
          event_name.c_str(),
          -1,
          std::get<2>(tup),
          std::get<3>(tup));
    }
#endif // ET_EVENT_TRACER_ENABLED

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
} // namespace backends
} // namespace executorch
