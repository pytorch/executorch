/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>
#include <executorch/backends/vulkan/serialization/schema_generated.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#endif // ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>
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
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;

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
    case vkgraph::VkDataType::INT64:
      return vkapi::kLong;
    case vkgraph::VkDataType::FLOAT16:
      return vkapi::kHalf;
    case vkgraph::VkDataType::FLOAT32:
      return vkapi::kFloat;
    case vkgraph::VkDataType::FLOAT64:
      return vkapi::kDouble;
    default:
      VK_THROW("Invalid VkDataType type encountered!");
  }
}

vkapi::ScalarType equivalent_scalar_type(
    const executorch::runtime::etensor::ScalarType& et_datatype) {
  switch (et_datatype) {
    case executorch::runtime::etensor::ScalarType::Byte:
      return vkapi::kByte;
    case executorch::runtime::etensor::ScalarType::Char:
      return vkapi::kChar;
    case executorch::runtime::etensor::ScalarType::Int:
      return vkapi::kInt;
    case executorch::runtime::etensor::ScalarType::Long:
      return vkapi::kLong;
    case executorch::runtime::etensor::ScalarType::Half:
      return vkapi::kHalf;
    case executorch::runtime::etensor::ScalarType::Float:
      return vkapi::kFloat;
    case executorch::runtime::etensor::ScalarType::Double:
      return vkapi::kDouble;
    case executorch::runtime::etensor::ScalarType::Bool:
      return vkapi::kBool;
    default:
      VK_THROW("Invalid etensor::ScalarType encountered!");
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
    case vkgraph::VkMemoryLayout::PACKED_INT8_4W4C:
      return utils::kPackedInt8_4W4C;
    case vkgraph::VkMemoryLayout::PACKED_INT8_4H4W:
      return utils::kPackedInt8_4H4W;
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
    if (strcmp(spec.key, "require_dynamic_shapes") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint8_t), "Unexpected value size!");
      bool value = getBool(value_data);

      if (value) {
        config.expect_dynamic_shapes = true;
      }
    }
    if (strcmp(spec.key, "warmup_execute_after_compile") == 0) {
      ET_CHECK_MSG(value_size == sizeof(uint8_t), "Unexpected value size!");
      bool value = getBool(value_data);

      config.warmup_execute_after_compile = value;
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
  const NamedDataMap* named_data_map_;
  std::vector<FreeableBuffer> loaded_buffers_from_map_;

  std::vector<ValueRef> ref_mapping_;

 public:
  explicit GraphBuilder(
      ComputeGraph* compute_graph,
      VkGraphPtr flatbuffer,
      const uint8_t* constant_data,
      const NamedDataMap* named_data_map)
      : compute_graph_(compute_graph),
        flatbuffer_(flatbuffer),
        constant_data_(constant_data),
        named_data_map_(named_data_map),
        loaded_buffers_from_map_(),
        ref_mapping_() {}

  void resize(uint32_t size) {
    ref_mapping_.resize(size, INT32_MAX);
  }

  bool fb_id_exists(const uint32_t fb_id) {
    return fb_id < ref_mapping_.size() && ref_mapping_[fb_id] != INT32_MAX;
  }

  ValueRef get_fb_id_valueref(const uint32_t fb_id) {
    ET_CHECK_MSG(
        fb_id_exists(fb_id),
        "Trying to extract a value that hasn't yet been added to the graph.");

    return ref_mapping_[fb_id];
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
      VkBytesPtr constant_bytes =
          flatbuffer_->constants()->Get(tensor_fb->constant_id());

      if (constant_bytes->named_key() != nullptr &&
          constant_bytes->offset() == UINT64_MAX &&
          named_data_map_ != nullptr) {
        const std::string& data_name = constant_bytes->named_key()->str();
        Result<FreeableBuffer> buffer =
            named_data_map_->get_data(data_name.c_str());

        VK_CHECK_COND(
            buffer.ok(),
            "Failed to get constant data for key %s from named_data_map. Error code: %u",
            data_name.c_str(),
            static_cast<uint32_t>(buffer.error()));
        ref = compute_graph_->add_tensorref(
            dims_vector, dtype, std::move(buffer.get()));
      } else {
        const uint8_t* tensor_data = constant_data_ + constant_bytes->offset();
        ref = compute_graph_->add_tensorref(dims_vector, dtype, tensor_data);
      }
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

  void add_symint_to_graph(const uint32_t fb_id, VkValuePtr value) {
    const int32_t fb_symint = value->value_as_SymInt()->value();
    ValueRef ref = compute_graph_->add_symint(fb_symint);
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
      case vkgraph::GraphTypes::SymInt:
        add_symint_to_graph(fb_id, value);
        break;
      default:
        ET_CHECK_MSG(false, "Unsupported value type.");
    }
  }

  vkapi::ScalarType get_staging_scalar_type_of(const uint32_t fb_id) {
    VkTensorPtr tensor_fb =
        flatbuffer_->values()->Get(fb_id)->value_as_VkTensor();
    if (tensor_fb->staging_datatype() == vkgraph::VkDataType::UNSET) {
      return get_scalar_type(tensor_fb->datatype());
    }
    return get_scalar_type(tensor_fb->staging_datatype());
  }

  void build_graph() {
    // Resize the mapping to the number of values in the flatbuffer
    resize(flatbuffer_->values()->size());

    // First, add all values to the graph
    for (uint32_t fb_id = 0; fb_id < flatbuffer_->values()->size(); ++fb_id) {
      VkValuePtr value = flatbuffer_->values()->Get(fb_id);
      add_value_to_graph(fb_id, value);
    }

    // Parse the inputs, which will be tensors most of the time but can also be
    // symints and tensorrefs (which will be the case if the original graph had)
    // mutable buffers.
    for (const uint32_t fb_id : *flatbuffer_->input_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      if (compute_graph_->val_is_tensor(ref)) {
        compute_graph_->set_input_tensor(
            ref, get_staging_scalar_type_of(fb_id));
      } else {
        compute_graph_->set_val_as_input(ref);
      }
    }

    // Parse the operators
    for (OpCallPtr op_call : *(flatbuffer_->chain())) {
      std::string op_name = op_call->name()->str();
      ET_CHECK_MSG(VK_HAS_OP(op_name), "Missing operator: %s", op_name.c_str());

      std::vector<ValueRef> args;
      args.reserve(op_call->args()->size());
      for (const auto arg_fb_id : *op_call->args()) {
        args.push_back(get_fb_id_valueref(static_cast<int>(arg_fb_id)));
      }

      auto vkFn = VK_GET_OP_FN(op_name);
      vkFn(*compute_graph_, args);
    }

    // Parse the outputs, which will be mostly tensors but may contain tensorref
    // values as well if the source graph returns parameter nodes.
    for (const uint32_t fb_id : *flatbuffer_->output_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      if (compute_graph_->val_is_tensor(ref)) {
        compute_graph_->set_output_tensor(
            ref, get_staging_scalar_type_of(fb_id));
      } else {
        compute_graph_->set_output_value(ref);
      }
    }

    if (compute_graph_->graphconfig().enable_querypool) {
      for (uint32_t i = 0; i < compute_graph_->prepack_nodes().size(); ++i) {
        compute_graph_->prepack_nodes()[i]->set_node_id(i);
      }
      for (uint32_t i = 0; i < compute_graph_->execute_nodes().size(); ++i) {
        compute_graph_->execute_nodes()[i]->set_node_id(i);
      }
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

  const std::vector<int64_t> in_tensor_vk_sizes =
      graph->sizes_of(in_tensor_ref);

  ET_CHECK_MSG(
      et_tensor.dim() == in_tensor_vk_sizes.size(),
      "Cannot resize input tensor: old ndim %zu does not match new ndim %zu",
      static_cast<size_t>(in_tensor_vk_sizes.size()),
      static_cast<size_t>(et_tensor.dim()));

  bool should_resize = false;
  std::vector<int64_t> new_sizes(et_tensor.dim());
  for (size_t i = 0; i < et_tensor.dim(); i++) {
    if (in_tensor_vk_sizes[i] != et_tensor.sizes()[i]) {
      should_resize = true;
    }
    new_sizes.at(i) = et_tensor.sizes()[i];
  }

  if (should_resize) {
    graph->resize_input(input_i, new_sizes);
  }

  const size_t in_tensor_vk_numel = graph->numel_of(in_tensor_ref);
  ET_CHECK_MSG(
      in_tensor_vk_numel == et_tensor.numel(),
      "Vulkan tensor numel %zu does not match ET tensor numel %zu",
      static_cast<size_t>(in_tensor_vk_numel),
      static_cast<size_t>(et_tensor.numel()));

  return should_resize;
}

bool maybe_update_scalar_tensor(
    ComputeGraph* graph,
    const ValueRef ref,
    executorch::aten::Tensor& scalar_tensor_src) {
  const int32_t cur_val = graph->read_symint(ref);
  int32_t scalar_tensor_val = 0;
  executorch::aten::ScalarType dtype = scalar_tensor_src.scalar_type();
  if (dtype == executorch::aten::ScalarType::Int) {
    scalar_tensor_val = *scalar_tensor_src.const_data_ptr<int32_t>();
  } else if (dtype == executorch::aten::ScalarType::Long) {
    scalar_tensor_val = int32_t(*scalar_tensor_src.const_data_ptr<int64_t>());
  }
  bool was_updated = false;
  if (scalar_tensor_val != cur_val) {
    graph->set_symint(ref, scalar_tensor_val);
    was_updated = true;
  }
  return was_updated;
}

void maybe_resize_output(
    ComputeGraph* graph,
    const size_t output_i,
    executorch::aten::Tensor& et_tensor) {
  ValueRef out_tensor_ref = graph->outputs()[output_i].value;

  const std::vector<int64_t> out_tensor_vk_sizes =
      graph->sizes_of(out_tensor_ref);

  executorch::aten::SizesType new_output_size[kTensorDimensionLimit];
  size_t ndim = out_tensor_vk_sizes.size();
  for (int i = 0; i < ndim; ++i) {
    new_output_size[i] = out_tensor_vk_sizes[i];
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

  ET_NODISCARD Error compileModel(
      const void* buffer_pointer,
      ComputeGraph* compute_graph,
      const NamedDataMap* named_data_map) const {
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

    GraphBuilder builder(
        compute_graph, flatbuffer_graph, constant_data, named_data_map);

    builder.build_graph();

    compute_graph->prepare();
    compute_graph->prepare_pipelines();

    compute_graph->prepack();

    compute_graph->optional_warmup_execute();

    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ComputeGraph* compute_graph =
        context.get_runtime_allocator()->allocateInstance<ComputeGraph>();
    if (compute_graph == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    GraphConfig graph_config = get_graph_config(compile_specs);
    graph_config.external_adapter = vkapi::set_and_get_external_adapter();
    new (compute_graph) ComputeGraph(graph_config);

    const NamedDataMap* named_data_map = context.get_named_data_map();
    Error err = compileModel(processed->data(), compute_graph, named_data_map);

    // This backend does not need its processed data after compiling the
    // model.
    processed->Free();

    if (err != Error::Ok) {
      return err;
    }

    return compute_graph;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    EXECUTORCH_SCOPE_PROF("VulkanBackend::execute");

    ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);

    const size_t num_inputs = compute_graph->inputs().size();
    bool should_propagate_resize = false;
    for (size_t i = 0; i < num_inputs; i++) {
      const ValueRef iref = compute_graph->inputs()[i].value;
      if (compute_graph->val_is_tensor(iref)) {
        VK_CHECK_COND(args[i]->isTensor());
        bool was_resized =
            maybe_resize_input(compute_graph, i, args[i]->toTensor());
        should_propagate_resize = should_propagate_resize || was_resized;
        compute_graph->maybe_cast_and_copy_into_staging(
            compute_graph->inputs()[i].staging,
            args[i]->toTensor().const_data_ptr(),
            args[i]->toTensor().numel(),
            equivalent_scalar_type(args[i]->toTensor().scalar_type()));
      } else if (compute_graph->val_is_symint(iref)) {
        VK_CHECK_COND(
            args[i]->isTensor(),
            "Cannot handle symint arg to graph that is not derived from a "
            "scalar tensor at the moment.");
        bool was_updated = maybe_update_scalar_tensor(
            compute_graph, iref, args[i]->toTensor());
        // Since symint inputs may impact tensor's sizes, trigger a resize if
        // any symbolic integer shapes are updated.
        should_propagate_resize = should_propagate_resize || was_updated;
      } else {
        VK_THROW(
            "Could not handle input with type ",
            compute_graph->get_val_type(iref));
      }
    }

    if (should_propagate_resize || compute_graph->has_data_dependent_shapes()) {
      compute_graph->propagate_resize();
    }

    compute_graph->execute();

    for (size_t i = 0; i < compute_graph->outputs().size(); i++) {
      const size_t o = i + num_inputs;
      const ValueRef oref = compute_graph->outputs()[i].value;
      if (compute_graph->val_is_tensor(oref)) {
        VK_CHECK_COND(args[o]->isTensor());
        maybe_resize_output(compute_graph, i, args[o]->toTensor());
        // args holds inputs directly followed by outputs, so the i'th output
        // for compute_graph corresponds to the o'th arg
        compute_graph->maybe_cast_and_copy_from_staging(
            compute_graph->outputs()[i].staging,
            args[o]->toTensor().mutable_data_ptr(),
            args[o]->toTensor().numel(),
            equivalent_scalar_type(args[o]->toTensor().scalar_type()));
      }
      // TensorRef values represent constant tensors which will not have been
      // modified by the graph execution. Therefore, if a constant tensor is
      // returned as an output, no action is required.
      else if (compute_graph->val_is_tref(oref)) {
        continue;
      } else {
        VK_THROW(
            "Could not handle output with type ",
            compute_graph->get_val_type(oref));
      }
    }

#ifdef ET_EVENT_TRACER_ENABLED
    runtime::EventTracer* event_tracer = context.event_tracer();
    compute_graph->context()->querypool().extract_results();
    for (const auto& r :
         compute_graph->context()->querypool().get_shader_timestamp_data()) {
      std::string event_name =
          r.kernel_name + "_" + std::to_string(r.dispatch_id);
      event_tracer_log_profiling_delegate(
          event_tracer,
          event_name.c_str(),
          /* delegate_debug_id = */ -1,
          r.start_time_ns,
          r.end_time_ns,
          (void*)(&r.metadata),
          sizeof(r.metadata));
    }
#endif // ET_EVENT_TRACER_ENABLED

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);
      compute_graph->context()
          ->adapter_ptr()
          ->compute_pipeline_cache()
          .save_cache();
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
