/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY
// facebook-security-vulnerable-integer-sign-conversion

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/api/containers/StagingBuffer.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#ifdef ET_EVENT_TRACER_ENABLED
std::string& set_and_get_current_operator_json(const std::string& json) {
  static std::string current_operator_json;
  if (json.size() > 0) {
    current_operator_json = json;
  }
  return current_operator_json;
}

size_t get_current_operator_count(const bool increment) {
  static int count = 0;
  if (increment) {
    count++;
  }
  return count;
}
#endif /* ET_EVENT_TRACER_ENABLED */

namespace vkcompute {

//
// VTensorPtr
//

#define VALUE_PTR_CLASS_IMPL(classname, ctype, type_name)                 \
  classname::classname(ComputeGraph* const graph, const ValueRef idx)     \
      : graph_(graph), ptr_(&(graph_->values_.at(idx).to##type_name())) { \
    graph_->values_in_use_++;                                             \
  }                                                                       \
  ctype* classname::operator->() const {                                  \
    return ptr_;                                                          \
  }                                                                       \
  ctype& classname::operator*() const {                                   \
    return *ptr_;                                                         \
  }                                                                       \
  classname::~classname() {                                               \
    graph_->values_in_use_--;                                             \
  }

VALUE_PTR_CLASS_IMPL(vTensorPtr, api::vTensor, Tensor)
VALUE_PTR_CLASS_IMPL(TensorRefPtr, TensorRef, TensorRef)
VALUE_PTR_CLASS_IMPL(StagingPtr, api::StagingBuffer, Staging)
VALUE_PTR_CLASS_IMPL(IntListPtr, std::vector<int64_t>, IntList)
VALUE_PTR_CLASS_IMPL(DoubleListPtr, std::vector<double>, DoubleList)
VALUE_PTR_CLASS_IMPL(BoolListPtr, std::vector<bool>, BoolList)
VALUE_PTR_CLASS_IMPL(ValueListPtr, std::vector<ValueRef>, ValueList)
VALUE_PTR_CLASS_IMPL(SymIntPtr, SymInt, SymInt)

#undef VALUE_PTR_CLASS_IMPL

//
// TmpTensor
//

TmpTensor::TmpTensor(
    ComputeGraph* const graph_ptr,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout)
    : graph_p(graph_ptr),
      sobj_idx(get_sobj_idx()),
      vref(graph_p->add_tensor(
          sizes,
          dtype,
          storage_type,
          memory_layout,
          sobj_idx)) {}

TmpTensor::TmpTensor(
    ComputeGraph* const graph_ptr,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type)
    : graph_p(graph_ptr),
      sobj_idx(get_sobj_idx()),
      vref(graph_p->add_tensor(sizes, dtype, storage_type, sobj_idx)) {}

TmpTensor::TmpTensor(
    ComputeGraph* const graph_ptr,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::GPUMemoryLayout memory_layout)
    : graph_p(graph_ptr),
      sobj_idx(get_sobj_idx()),
      vref(graph_p->add_tensor(sizes, dtype, memory_layout, sobj_idx)) {}

TmpTensor::TmpTensor(
    ComputeGraph* const graph_ptr,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype)
    : graph_p(graph_ptr),
      sobj_idx(get_sobj_idx()),
      vref(graph_p->add_tensor(sizes, dtype, sobj_idx)) {}

TmpTensor::~TmpTensor() {
  // Lifetime of this temporary tensor is expired; return the shared object to
  // the pool, as long as the sobj index is valid
  if (sobj_idx >= 0) {
    graph_p->tmp_shared_object_idxs_.emplace(sobj_idx);
  }
}

int64_t TmpTensor::get_sobj_idx() {
  int64_t sobj_idx;
  // If no available temporary shared objects, request a new one to be created
  if (graph_p->tmp_shared_object_idxs_.empty()) {
    sobj_idx = graph_p->shared_objects_.size();
  } else {
    // Get the first available shared object idx
    sobj_idx = graph_p->tmp_shared_object_idxs_.top();
    graph_p->tmp_shared_object_idxs_.pop();
  }
  return sobj_idx;
}

//
// ComputeGraph
//

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      prepack_descriptor_counts_{},
      execute_descriptor_counts_{},
      context_{new api::Context(
          config.external_adapter ? config.external_adapter
                                  : vkapi::runtime()->get_adapter_p(),
          config_.context_config)},
      shared_objects_{},
      values_{},
      param_ubos_{},
      prepack_nodes_{},
      execute_nodes_{},
      inputs_{},
      outputs_{} {
  // Ensure that descriptor counts are initialized to 0
  prepack_descriptor_counts_.descriptor_pool_max_sets = 0;
  prepack_descriptor_counts_.descriptor_uniform_buffer_count = 0;
  prepack_descriptor_counts_.descriptor_storage_buffer_count = 0;
  prepack_descriptor_counts_.descriptor_combined_sampler_count = 0;
  prepack_descriptor_counts_.descriptor_storage_image_count = 0;

  execute_descriptor_counts_.descriptor_pool_max_sets = 0;
  execute_descriptor_counts_.descriptor_uniform_buffer_count = 0;
  execute_descriptor_counts_.descriptor_storage_buffer_count = 0;
  execute_descriptor_counts_.descriptor_combined_sampler_count = 0;
  execute_descriptor_counts_.descriptor_storage_image_count = 0;

  // If certain graph config variables are not specified, then set them
  // automatically.
  if (config_.prepack_threshold_nbytes == 0) {
    config_.prepack_threshold_nbytes = 10 * MB;
    config_.prepack_initial_threshold_nbytes = 10 * MB;
  }
  if (config_.execute_threshold_node_count == 0) {
    config_.execute_threshold_node_count = 128;
    config_.execute_initial_threshold_node_count = 64;
  }

  // Check if the underlying GPU can access accelerated integer dot product
  // instructions
  can_use_int8_dot_product_ =
      context_->adapter_ptr()->supports_int8_dot_product();
}

ComputeGraph::~ComputeGraph() {
  // Wait for all currently executing commands to complete before cleaning up.
  // If wait_for_queue() throws an exception, still proceed with cleanup.
  try {
    context_->wait_for_queue();
  } catch (...) {
  }

  // Wrap in try/catch to ensure that destructor does not throw
  try {
    values_.clear();

    prepack_nodes_.clear();
    execute_nodes_.clear();
    clear_deferred_cmds();
  } catch (...) {
  }
}

std::vector<int64_t> ComputeGraph::extract_int_or_symint_list(
    const ValueRef idx) {
  const Value& val = values_.at(idx);
  std::vector<int64_t> result;

  if (val.isIntList()) {
    // If it's an IntList, return a copy of the list
    return val.toConstIntList();
  } else if (val.isValueList()) {
    // If it's a ValueList, extract each element as an Int or SymInt
    const std::vector<ValueRef>& value_list = val.toConstValueList();
    result.reserve(value_list.size());

    for (const ValueRef& ref : value_list) {
      const Value& element = values_.at(ref);
      if (element.isInt()) {
        result.push_back(element.toInt());
      } else if (element.isSymInt()) {
        result.push_back(read_symint(ref));
      } else {
        VK_THROW(
            "ValueList element is neither Int nor SymInt, but has type ",
            element.type());
      }
    }
    return result;
  }

  VK_THROW(
      "Cannot extract int or symint list from Value with type ", val.type());
}

utils::StorageType ComputeGraph::suggested_storage_type() {
  if (config_.enable_storage_type_override) {
    return config_.storage_type_override;
  }
  return utils::kTexture3D;
}

bool ComputeGraph::was_value_updated(const ValueRef idx) const noexcept {
  if (!is_valid_value_idx(idx)) {
    return false;
  }

  // Check if this ValueRef itself was updated
  if (updated_values_.find(idx) != updated_values_.end()) {
    return true;
  }

  // If this is a ValueList, check each ValueRef in the list
  if (val_is_value_list(idx)) {
    const auto& value_list = values_.at(idx).toConstValueList();
    for (const auto& nested_idx : value_list) {
      if (was_value_updated(nested_idx)) {
        return true;
      }
    }
  }

  return false;
}

utils::GPUMemoryLayout ComputeGraph::suggested_memory_layout(
    const std::vector<int64_t>& sizes) {
  if (config_.enable_memory_layout_override) {
    return config_.memory_layout_override;
  }
  if (sizes.size() < 3) {
    return utils::kWidthPacked;
  }
  // For 3 dimensional tensors that only have a channels dimension of 1, still
  // prefer width packed.
  if (utils::val_at(-3, sizes) == 1) {
    return utils::kWidthPacked;
  }
  return utils::kChannelsPacked;
}

bool ComputeGraph::device_name_contains(const char* substr) {
  return context_->adapter_ptr()->device_name().find(substr) !=
      std::string::npos;
}

void ComputeGraph::check_no_active_value_ptrs() {
  VK_CHECK_COND(
      values_in_use_ == 0,
      "Make sure that there are no pointers stored from the return values of "
      "`ComputeGraph::get_*()` functions in scope before adding Values to the "
      "graph. Modifying the graph's values may cause existing pointers to be "
      "invalidated.");
}

bool ComputeGraph::is_valid_value_idx(const ValueRef idx) const noexcept {
  return idx >= 0 && idx < static_cast<int>(values_.size());
}

std::vector<int64_t> ComputeGraph::sizes_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().sizes();
  } else if (val.isTensorRef()) {
    return val.toConstTensorRef().sizes;
  }
  VK_THROW("Could not get sizes of value with type ", val.type());
}

std::vector<int64_t> ComputeGraph::padded_sizes_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().padded_sizes();
  }
  VK_THROW("Could not get padded sizes of value with type ", val.type());
}

int64_t ComputeGraph::dim_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().dim();
  } else if (val.isTensorRef()) {
    return val.toConstTensorRef().sizes.size();
  }
  VK_THROW("Could not get dim of value with type ", val.type());
}

std::vector<int64_t> ComputeGraph::dim_order_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().dim_order();
  }
  VK_THROW("Could not get dim order of value with type ", val.type());
}

std::vector<int64_t> ComputeGraph::strides_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().strides();
  }
  VK_THROW("Could not get strides of value with type ", val.type());
}

vkapi::ScalarType ComputeGraph::dtype_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().dtype();
  } else if (val.isTensorRef()) {
    return val.toConstTensorRef().dtype;
  } else if (val.isStaging()) {
    return val.toConstStaging().dtype();
  } else if (val.isBool()) {
    return vkapi::ScalarType::Bool;
  } else if (val.isDouble()) {
    // We downcast anyway in the shader and we want to avoid having to
    // write special cases there.
    return vkapi::ScalarType::Float;
  } else if (val.isInt()) {
    return vkapi::ScalarType::Int;
  }
  VK_THROW("Could not get dtype of value with type ", val.type());
}

vkapi::ScalarType ComputeGraph::get_staging_dtype_for(
    const ValueRef idx) const {
  return api::get_staging_dtype(context_.get(), dtype_of(idx));
}

bool ComputeGraph::is_contiguous_buffer_tensor(const ValueRef idx) const {
  if (!val_is_tensor(idx)) {
    return false;
  }
  if (!is_buffer_storage(idx)) {
    return false;
  }
  return is_contiguous(idx);
}

bool ComputeGraph::is_contiguous_texture_tensor(const ValueRef idx) const {
  if (!val_is_tensor(idx)) {
    return false;
  }
  if (is_buffer_storage(idx)) {
    return false;
  }
  return has_standard_axis_map(idx) && packed_dim_of(idx) == 0;
}

bool ComputeGraph::is_standard_channels_packed_texture_tensor(
    const ValueRef idx) const {
  if (!val_is_tensor(idx)) {
    return false;
  }
  if (is_buffer_storage(idx)) {
    return false;
  }
  return has_standard_axis_map(idx) && packed_dim_of(idx) == 2;
}

bool ComputeGraph::is_2d_matrix(const ValueRef idx) const {
  std::vector<int64_t> sizes = sizes_of(idx);
  const size_t ndim = sizes.size();
  if (sizes.size() < 2) {
    return false;
  }
  if (sizes.size() == 2) {
    return true;
  }

  // Check that outermost dims have size of 1
  for (int d = 0; d < ndim - 2; d++) {
    if (sizes[d] != 1) {
      return false;
    }
  }

  return true;
}

bool ComputeGraph::is_vectorizable_contiguous_2d_matrix(
    const ValueRef idx) const {
  if (!is_2d_matrix(idx)) {
    return false;
  }
  if (is_buffer_storage(idx)) {
    return is_contiguous_buffer_tensor(idx) &&
        size_at<int32_t>(-1, idx) % 4 == 0;
  }
  return is_contiguous_texture_tensor(idx);
}

bool ComputeGraph::is_vectorizable_width_packed_tensor(
    const ValueRef idx) const {
  // Not a tensor - return false
  if (!val_is_tensor(idx)) {
    return false;
  }
  if (is_buffer_storage(idx)) {
    return is_contiguous_buffer_tensor(idx) &&
        size_at<int32_t>(-1, idx) % 4 == 0;
  }

  return is_standard_channels_packed_texture_tensor(idx);
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout,
    const int64_t shared_object_idx,
    const utils::AxisMapLayout axis_map_layout) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(api::vTensor(
      context(),
      sizes,
      dtype,
      storage_type,
      memory_layout,
      false,
      axis_map_layout));

  if (shared_object_idx >= 0) {
    get_shared_object(shared_object_idx).add_user(this, idx);
  }
  return idx;
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type,
    const int64_t shared_object_idx,
    const utils::AxisMapLayout axis_map_layout) {
  return add_tensor(
      sizes,
      dtype,
      storage_type,
      suggested_memory_layout(sizes),
      shared_object_idx,
      axis_map_layout);
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::GPUMemoryLayout memory_layout,
    const int64_t shared_object_idx,
    const utils::AxisMapLayout axis_map_layout) {
  return add_tensor(
      sizes,
      dtype,
      suggested_storage_type(),
      memory_layout,
      shared_object_idx,
      axis_map_layout);
}

ValueRef ComputeGraph::add_tensor_like(
    const ValueRef idx,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout,
    const utils::AxisMapLayout axis_map_layout) {
  return add_tensor(
      sizes_of(idx),
      dtype_of(idx),
      storage_type,
      memory_layout,
      -1,
      axis_map_layout);
}

ValueRef ComputeGraph::add_tensor_like(
    const ValueRef idx,
    const utils::GPUMemoryLayout memory_layout,
    const utils::AxisMapLayout axis_map_layout) {
  return add_tensor(
      sizes_of(idx),
      dtype_of(idx),
      storage_type_of(idx),
      memory_layout,
      -1,
      axis_map_layout);
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const int64_t shared_object_idx,
    const utils::AxisMapLayout axis_map_layout) {
  return add_tensor(
      sizes,
      dtype,
      suggested_memory_layout(sizes),
      shared_object_idx,
      axis_map_layout);
}

ValueRef ComputeGraph::add_tensor(const vkapi::VulkanImage& image) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(api::vTensor(context(), image));
  return idx;
}

ValueRef ComputeGraph::add_tensor_view(const ValueRef vref) {
  const vTensorPtr t = get_tensor(vref);
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(api::vTensor(*t));
  return idx;
}

ValueRef ComputeGraph::add_tensor_view(
    const ValueRef vref,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  const vTensorPtr t = get_tensor(vref);
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(api::vTensor(*t, sizes, strides));
  return idx;
}

ValueRef ComputeGraph::add_tensorref(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const void* const data) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(TensorRef(sizes, dtype, data));
  total_constant_nbytes_ += values_.back().toConstTensorRef().nbytes();
  return idx;
}

ValueRef ComputeGraph::add_tensorref(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    executorch::runtime::FreeableBuffer&& buffer) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(TensorRef(sizes, dtype, std::move(buffer)));
  total_constant_nbytes_ += values_.back().toConstTensorRef().nbytes();
  return idx;
}

ValueRef ComputeGraph::add_staging(
    const vkapi::ScalarType dtype,
    const size_t numel,
    const vkapi::CopyDirection direction) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(api::StagingBuffer(context(), dtype, numel, direction));
  return idx;
}

ValueRef ComputeGraph::add_none() {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back();
  return idx;
}

ValueRef ComputeGraph::add_value_list(std::vector<ValueRef>&& value) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(std::move(value));
  return idx;
}

ValueRef ComputeGraph::add_string(std::string&& str) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(std::move(str));
  return idx;
}

ValueRef ComputeGraph::add_symint(const int32_t val) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(SymInt(context(), val));
  return idx;
}

ValueRef ComputeGraph::get_or_add_value_for_int(const int64_t val) {
  for (int i = 0; i < values_.size(); ++i) {
    if (values_.at(i).isInt() && values_.at(i).toInt() == val) {
      return i;
    }
  }
  return add_scalar(val);
}

ValueRef ComputeGraph::set_input_tensor(
    const ValueRef idx,
    vkapi::ScalarType staging_dtype) {
  // For texture storage, the buffer size needs to account for the zero
  // padding applied by unused texel elements.
  size_t buf_numel = get_tensor(idx)->staging_buffer_numel();
  ValueRef staging_idx = add_staging(
      staging_dtype, buf_numel, vkapi::CopyDirection::HOST_TO_DEVICE);
  add_staging_to_tensor_node(*this, staging_idx, idx);
  inputs_.push_back({idx, staging_idx});
  return staging_idx;
}

ValueRef ComputeGraph::set_input_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vkapi::ScalarType dtype = get_tensor(idx)->dtype();
    return set_input_tensor(idx, dtype);
  } else {
    inputs_.push_back({idx, kDummyValueRef});
    return idx;
  }
}

ValueRef ComputeGraph::set_output_tensor(
    const ValueRef idx,
    vkapi::ScalarType staging_dtype) {
  // For texture storage, the buffer size needs to account for the zero
  // padding applied by unused texel elements.
  size_t buf_numel = get_tensor(idx)->staging_buffer_numel();
  ValueRef staging_idx = add_staging(
      staging_dtype, buf_numel, vkapi::CopyDirection::DEVICE_TO_HOST);
  // We only run this when the tensor is non-empty.  When the underlying
  // tensor is empty (e.g. padded_numel == 0), we do not allocate a VkImage to
  // tensor, we will not be able to bind the node for execution.
  if (buf_numel > 0) {
    add_tensor_to_staging_node(*this, idx, staging_idx);
  }
  outputs_.push_back({idx, staging_idx});
  return staging_idx;
}

ValueRef ComputeGraph::set_output_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vkapi::ScalarType dtype = get_tensor(idx)->dtype();
    return set_output_tensor(idx, dtype);
  } else {
    outputs_.push_back({idx, kDummyValueRef});
    return idx;
  }
}

ValueRef ComputeGraph::set_output_value(const ValueRef idx) {
  if (values_.at(idx).isTensor()) {
    return set_output_tensor(idx);
  }
  outputs_.push_back({idx, kDummyValueRef});
  return idx;
}

vkapi::BufferBindInfo ComputeGraph::get_or_create_int_param_buffer(
    const ValueRef idx) {
  if (values_.at(idx).isInt()) {
    const int32_t val = extract_scalar<int32_t>(idx);
    return create_params_buffer(val);
  } else if (values_.at(idx).isSymInt()) {
    SymIntPtr symint = get_symint(idx);
    return vkapi::BufferBindInfo(symint->gpu_buffer.buffer());
  }
  VK_THROW("Cannot create a int param buffer for the given value");
}

vkapi::BufferBindInfo ComputeGraph::get_or_create_int_param_buffer(
    const ValueRef idx,
    const int32_t default_val) {
  if (values_.at(idx).isNone()) {
    return create_params_buffer(default_val);
  } else {
    return get_or_create_int_param_buffer(idx);
  }
}

void ComputeGraph::set_symint(const ValueRef idx, const int32_t val) {
  int32_t cur_val = read_symint(idx);
  if (cur_val != val) {
    get_symint(idx)->set(val);
    // Track that this ValueRef was updated
    updated_values_.insert(idx);
  }
}

int32_t ComputeGraph::read_symint(const ValueRef idx) {
  return get_symint(idx)->get();
}

ValueRef ComputeGraph::staging_of(const ValueRef idx) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].value == idx) {
      if (is_valid(inputs_[i].staging)) {
        return inputs_[i].staging;
      }
    }
  }
  VK_THROW("Could not find staging buffer for value at index ", idx);
}

SharedObject& ComputeGraph::get_shared_object(const int64_t idx) {
  if (idx >= shared_objects_.size()) {
    shared_objects_.resize(static_cast<size_t>(idx + 1));
  }
  return shared_objects_.at(idx);
}

void ComputeGraph::create_dedicated_allocation_for(const ValueRef idx) {
  vTensorPtr tensor = get_tensor(idx);
  if (!tensor->memory_is_bound()) {
    VmaAllocationCreateInfo alloc_create_info =
        context()->adapter_ptr()->vma().gpuonly_resource_create_info();
    tensor->acquire_allocation(
        context()->adapter_ptr()->vma().create_allocation(
            tensor->get_memory_requirements(), alloc_create_info));
  }
}

void ComputeGraph::update_descriptor_counts(
    const vkapi::ShaderInfo& shader_info,
    bool execute) {
  vkapi::DescriptorPoolConfig* config =
      execute ? &execute_descriptor_counts_ : &prepack_descriptor_counts_;

  config->descriptor_pool_max_sets += 1;
  for (const VkDescriptorType arg_type : shader_info.kernel_layout) {
    switch (arg_type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        config->descriptor_uniform_buffer_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        config->descriptor_storage_buffer_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        config->descriptor_combined_sampler_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        config->descriptor_storage_image_count += 1;
        break;
      default:
        VK_THROW("Unsupported descriptor type!");
    }
  }
}

void ComputeGraph::register_pipeline_to_create(
    const vkapi::ShaderInfo& shader_info,
    const utils::WorkgroupSize& local_workgroup_size,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<PushConstantDataInfo>& push_constants) {
  VkDescriptorSetLayout shader_layout =
      context()->shader_layout_cache().retrieve(shader_info.kernel_layout);

  uint32_t pc_offset = 0;
  std::array<uint8_t, kMaxPushConstantSize> pc_data;
  for (const auto& pc : push_constants) {
    pc_offset += pc.write(pc_data.data(), pc_offset, kMaxPushConstantSize);
  }

  vkapi::SpecVarList spec_constants = {
      SV(local_workgroup_size[0u]),
      SV(local_workgroup_size[1u]),
      SV(local_workgroup_size[2u])};

  spec_constants.append(spec_vars);

  const vkapi::ComputePipelineCache::Key desc = {
      context()->pipeline_layout_cache().retrieve(shader_layout, pc_offset),
      context()->shader_cache().retrieve(shader_info),
      spec_constants};

  if (context_->pipeline_cache().contains(desc)) {
    return;
  }
  auto it = pipeline_descriptors_.find(desc);
  if (it != pipeline_descriptors_.cend()) {
    return;
  }
  pipeline_descriptors_.insert(desc);
}

utils::uvec3 ComputeGraph::create_global_wg_size(const ValueRef idx) {
  if (is_buffer_storage(idx)) {
    return {uint32_t(numel_of(idx)), 1u, 1u};
  }
  return logical_limits_of(idx);
}

utils::uvec3 ComputeGraph::create_local_wg_size(
    const utils::uvec3 global_wg_size) {
  if (config_.enable_local_wg_size_override) {
    return config_.local_wg_size_override;
  }

  // array containing axis index and global workgroup size
  std::pair<uint32_t, uint32_t> global_wg_size_desc[] = {
      {0u, global_wg_size[0]},
      {1u, global_wg_size[1]},
      {2u, global_wg_size[2]}};

  // sort the global workgroup size in descending order
  if (global_wg_size_desc[0].second < global_wg_size_desc[1].second) {
    std::swap(global_wg_size_desc[0], global_wg_size_desc[1]);
  }
  if (global_wg_size_desc[1].second < global_wg_size_desc[2].second) {
    std::swap(global_wg_size_desc[1], global_wg_size_desc[2]);
  }
  if (global_wg_size_desc[0].second < global_wg_size_desc[1].second) {
    std::swap(global_wg_size_desc[0], global_wg_size_desc[1]);
  }

  utils::uvec3 local_group_size = {
      8,
      std::max(1u, std::min(4u, global_wg_size_desc[1].second)),
      std::max(1u, std::min(2u, global_wg_size_desc[2].second))};

  if (global_wg_size_desc[2u].second == 1) {
    if (global_wg_size_desc[1u].second == 1) {
      local_group_size[0u] = 64;
      local_group_size[1u] = 1;
    } else if (global_wg_size_desc[1u].second % 4 == 0) {
      local_group_size[0u] = 16;
      local_group_size[1u] = 4;
    } else {
      local_group_size[0u] = 32;
      local_group_size[1u] = 2;
    }
  }

  return {
      local_group_size[global_wg_size_desc[0].first],
      local_group_size[global_wg_size_desc[1].first],
      local_group_size[global_wg_size_desc[2].first]};
}

utils::uvec3 ComputeGraph::create_local_wg_size(const ValueRef idx) {
  return create_local_wg_size(create_global_wg_size(idx));
}

void ComputeGraph::bind_tensor_to_descriptor_set(
    const ValueRef ref,
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::MemoryAccessFlags access_type,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  vTensorPtr tensor = get_tensor(ref);
  if (tensor->buffer()) {
    vkapi::VulkanBuffer& buffer = tensor->buffer(
        pipeline_barrier, vkapi::PipelineStage::COMPUTE, access_type);
    descriptor_set.bind(idx, buffer);
  } else {
    vkapi::VulkanImage& image = tensor->image(
        pipeline_barrier, vkapi::PipelineStage::COMPUTE, access_type);
    descriptor_set.bind(idx, image);
  }
}

void ComputeGraph::bind_value_to_descriptor_set(
    const ValueRef ref,
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::MemoryAccessFlags access_type,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  if (val_is_tensor(ref)) {
    bind_tensor_to_descriptor_set(
        ref, pipeline_barrier, access_type, descriptor_set, idx);
  } else if (val_is_staging(ref)) {
    descriptor_set.bind(idx, get_staging(ref)->buffer());
  }
}

void ComputeGraph::copy_into_staging(
    const ValueRef idx,
    const void* data,
    const size_t numel) {
  StagingPtr staging = get_staging(idx);
  size_t nbytes = numel * vkapi::element_size(staging->dtype());
  staging->copy_from(data, nbytes);
}

void ComputeGraph::maybe_cast_and_copy_into_staging(
    const ValueRef idx,
    const void* data,
    const size_t numel,
    const vkapi::ScalarType src_data_dtype) {
  StagingPtr staging = get_staging(idx);
  vkapi::ScalarType staging_dtype = staging->dtype();
  if (src_data_dtype == staging_dtype) {
    size_t nbytes = numel * vkapi::element_size(staging_dtype);
    staging->copy_from(data, nbytes);
    return;
  } else {
    // Hard-coded type conversion cases
    if (src_data_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
      const int64_t* casted_data = reinterpret_cast<const int64_t*>(data);
      staging->cast_and_copy_from<int64_t, int32_t>(casted_data, numel);
    } else if (
        src_data_dtype == vkapi::kDouble && staging_dtype == vkapi::kFloat) {
      const double* casted_data = reinterpret_cast<const double*>(data);
      staging->cast_and_copy_from<double, float>(casted_data, numel);
    } else if (
        src_data_dtype == vkapi::kHalf && staging_dtype == vkapi::kFloat) {
      const uint16_t* casted_data = reinterpret_cast<const uint16_t*>(data);
      staging->cast_half_to_float_and_copy_from(casted_data, numel);
    } else {
      VK_THROW(
          "Unsupported type conversion from ",
          src_data_dtype,
          " to staging dtype ",
          staging_dtype);
    }
  }
}

void ComputeGraph::copy_from_staging(
    const ValueRef idx,
    void* data,
    const size_t numel) {
  StagingPtr staging = get_staging(idx);
  size_t nbytes = numel * vkapi::element_size(staging->dtype());
  staging->copy_to(data, nbytes);
}

void ComputeGraph::maybe_cast_and_copy_from_staging(
    const ValueRef idx,
    void* data,
    const size_t numel,
    const vkapi::ScalarType dst_data_dtype) {
  StagingPtr staging = get_staging(idx);
  vkapi::ScalarType staging_dtype = staging->dtype();
  if (dst_data_dtype == staging_dtype) {
    size_t nbytes = numel * vkapi::element_size(staging_dtype);
    staging->copy_to(data, nbytes);
    return;
  } else {
    // Hard-coded type conversion cases
    if (dst_data_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
      int64_t* casted_data = reinterpret_cast<int64_t*>(data);
      staging->cast_and_copy_to<int32_t, int64_t>(casted_data, numel);
    } else if (
        dst_data_dtype == vkapi::kDouble && staging_dtype == vkapi::kFloat) {
      double* casted_data = reinterpret_cast<double*>(data);
      staging->cast_and_copy_to<float, double>(casted_data, numel);
    } else if (
        dst_data_dtype == vkapi::kHalf && staging_dtype == vkapi::kFloat) {
      uint16_t* casted_data = reinterpret_cast<uint16_t*>(data);
      staging->cast_float_to_half_and_copy_to(casted_data, numel);
    } else {
      VK_THROW(
          "Unsupported type conversion from staging dtype ",
          staging_dtype,
          " to ",
          dst_data_dtype);
    }
  }
}

void ComputeGraph::prepare() {
#define MERGE_FIELD(field)                    \
  static_cast<uint32_t>(std::ceil(            \
      std::max(                               \
          execute_descriptor_counts_.field,   \
          prepack_descriptor_counts_.field) * \
      config_.descriptor_pool_safety_factor))

  uint32_t max_sets = MERGE_FIELD(descriptor_pool_max_sets);
  vkapi::DescriptorPoolConfig config{
      max_sets,
      std::max(MERGE_FIELD(descriptor_uniform_buffer_count), max_sets),
      std::max(MERGE_FIELD(descriptor_storage_buffer_count), max_sets),
      std::max(MERGE_FIELD(descriptor_combined_sampler_count), max_sets),
      std::max(MERGE_FIELD(descriptor_storage_image_count), max_sets),
      1u,
  };

  if (!context_->descriptor_pool()) {
    context_->descriptor_pool().init(config);
  }
#undef MERGE_FIELD

  if (config_.enable_querypool) {
    context_->initialize_querypool();
  }

  // Calculate the threshold at which a new command buffer should be created
  // during execute()
  const size_t total_node_count = execute_nodes_.size();
  size_t init_threshold = config_.execute_initial_threshold_node_count;
  size_t count_threshold = config_.execute_threshold_node_count;

  // If max command buffer count is set, we need to adjust the thresholds to
  // accommodate execution within the limit, if total command buffers with
  // current thresholds would exceed execute_max_cmds
  if (config_.execute_max_cmds > 0) {
    // Worse case scenario we have one command buffer for nodes before init
    // threshold and config_.execute_max_cmds - 1 command buffers for the rest
    // of dispatches

    // If command buffers created after offsetting init_threshold would exceed
    // max command buffer count, we need to adjust init and count thresholds
    const bool slicing_exceeds_max_cmds = (total_node_count - init_threshold) >
        count_threshold * (config_.execute_max_cmds - 1);
    if (total_node_count > init_threshold && slicing_exceeds_max_cmds) {
      // Increase count threshold so remaining nodes after offsetting init fits
      // in config_.execute_max_cmds - 1
      count_threshold = static_cast<size_t>(ceil(
          (total_node_count - init_threshold) /
          double(config_.execute_max_cmds - 1)));
    }
  }

  execute_threshold_node_count_ = count_threshold;
}

void ComputeGraph::prepare_pipelines() {
  for (std::unique_ptr<PrepackNode>& node : prepack_nodes_) {
    node->prepare_pipelines(this);
  }
  for (std::unique_ptr<ExecuteNode>& node : execute_nodes_) {
    node->prepare_pipelines(this);
  }
  context_->pipeline_cache().create_pipelines(pipeline_descriptors_);

  pipeline_descriptors_ = std::unordered_set<
      vkapi::ComputePipelineCache::Key,
      vkapi::ComputePipelineCache::Hasher>();
}

void ComputeGraph::submit_current_cmd(const bool final_use) {
  context_->submit_cmd_to_gpu(VK_NULL_HANDLE, final_use);
}

void ComputeGraph::submit_current_cmd_and_wait(const bool final_use) {
  vkapi::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle(), final_use);
  fence.wait();
  context_->fences().return_fence(fence);
}

void ComputeGraph::submit_cmd(vkapi::CommandBuffer& cmd_buf, VkFence fence) {
  if (cmd_buf) {
    cmd_buf.end();
    context_->adapter_ptr()->submit_cmd(
        context_->queue(), cmd_buf.get_submit_handle(false), fence);
  }
}

void ComputeGraph::submit_deferred_cmds_and_wait() {
  vkapi::VulkanFence fence = context_->fences().get_fence();

  for (uint32_t i = 0; i < deferred_cmd_list_.size(); i++) {
    auto& cmd = deferred_cmd_list_[i];

    submit_cmd(
        cmd,
        i == (deferred_cmd_list_.size() - 1) ? fence.get_submit_handle()
                                             : VK_NULL_HANDLE);
  }
  fence.wait();
  context_->fences().return_fence(fence);
}

void ComputeGraph::clear_deferred_cmds() {
  for (auto& cmd : deferred_cmd_list_) {
    if (cmd) {
      cmd.end();
      cmd.invalidate();
    }
  }
  deferred_cmd_list_.clear();
}

void ComputeGraph::prepack() {
  int i = 0;
  bool submitted = false;
  const bool reduce_peak_memory = total_constant_nbytes_ > 500 * MB;
  // int count = 0;
  context_->set_cmd();
  for (std::unique_ptr<PrepackNode>& node : prepack_nodes_) {
    // Do not trigger on the first or last prepack node.
    const bool not_terminal = i != 0 && i != (prepack_nodes_.size() - 1);
    size_t threshold = submitted ? config_.prepack_threshold_nbytes
                                 : config_.prepack_initial_threshold_nbytes;
    if (not_terminal && staging_nbytes_in_cmd_ > threshold) {
      // If reducing peak memory usage, wait for the current command buffer to
      // finish executing and flush to recycle the staging memory. This will
      // reduce peak memory usage, but will slightly increase load latency.
      // Otherwise, just submit the current command buffer for execution and
      // proceed. This results in lower load latency at the cost of higher peak
      // memory usage.
      if (reduce_peak_memory) {
        submit_current_cmd_and_wait();
        context_->flush();
      } else {
        submit_current_cmd();
      }
      staging_nbytes_in_cmd_ = 0;
      context_->set_cmd();
      submitted = true;
    }

    node->encode(this);
    i++;
  }
  submit_current_cmd_and_wait(/*final_use=*/true);
  context_->flush();
  staging_nbytes_in_cmd_ = 0;

  // Initialize allocations for intermediate tensors
  for (SharedObject& shared_object : shared_objects_) {
    shared_object.allocate(this);
    shared_object.bind_users(this);
  }
  // Make sure all remaining tensors have allocations
  for (int i = 0; i < values_.size(); i++) {
    if (values_.at(i).isTensor()) {
      create_dedicated_allocation_for(i);
    }
  }
}

void ComputeGraph::optional_warmup_execute() {
  if (config_.warmup_execute_after_compile) {
    execute();
  }
}

void ComputeGraph::execute() {
  if (deferred_cmd_list_.empty()) {
    context_->flush();
    context_->set_cmd(/*reusable = */ true);

    context_->cmd_reset_querypool();
    const size_t total_node_count = execute_nodes_.size();
    uint32_t encoded_node_count = 0;

    for (std::unique_ptr<ExecuteNode>& node : execute_nodes_) {
      node->encode(this);
      encoded_node_count++;

      // Threshold is reached when the node count reached
      // execute_initial_threshold_node_count or if its a multiple of
      // execute_threshold_node_count.
      const bool reached_threshold =
          encoded_node_count >= config_.execute_initial_threshold_node_count &&
          ((encoded_node_count - config_.execute_initial_threshold_node_count) %
               execute_threshold_node_count_ ==
           0);

      // Create a new command buffer when threashold is reached
      // But avoid it if this is the last node, since last cmd buf is submitted
      // after the loop
      if (reached_threshold && encoded_node_count != total_node_count) {
        context_->submit_cmd_to_gpu(VK_NULL_HANDLE, false);
        deferred_cmd_list_.emplace_back(std::move(context_->extract_cmd()));
        context_->set_cmd(true);
      }
    }

    vkapi::VulkanFence fence = context_->fences().get_fence();
    context_->submit_cmd_to_gpu(fence.get_submit_handle(), false);
    fence.wait();
    context_->fences().return_fence(fence);
    deferred_cmd_list_.emplace_back(std::move(context_->extract_cmd()));
  } else {
    submit_deferred_cmds_and_wait();
  }

  execute_count_++;

  // Clear the set of updated values at the end of inference
  updated_values_.clear();

  // Reset the re-encoding flag at the end of inference
  requires_reencode_ = false;
}

void ComputeGraph::virtual_clone(const ValueRef dst, const ValueRef src) {
  get_tensor(dst)->virtual_clone(*get_tensor(src));
}

void ComputeGraph::virtual_transpose(
    const ValueRef tensor,
    const int64_t dim0,
    const int64_t dim1) {
  get_tensor(tensor)->virtual_transpose(dim0, dim1);
}

void ComputeGraph::resize_input(
    const int64_t idx,
    const std::vector<int64_t>& new_sizes) {
  IOValueRef io_val = inputs_.at(idx);
  virtual_resize(io_val.value, new_sizes);
  updated_values_.insert(io_val.staging);
}

void ComputeGraph::virtual_resize(
    const ValueRef idx,
    const std::vector<int64_t>& new_sizes) {
  std::vector<int64_t> cur_sizes = sizes_of(idx);
  if (cur_sizes != new_sizes) {
    get_tensor(idx)->virtual_resize(new_sizes);
    // Track that this ValueRef was updated
    updated_values_.insert(idx);
  }
}

void ComputeGraph::propagate_resize() {
  for (std::unique_ptr<ExecuteNode>& node : execute_nodes_) {
    node->trigger_resize(this);
  }
  // A command buffer re-encode will be needed if:
  // 1. Any push constant data (used for tensor metadata) was updated
  // 2. Compute shader dispatch parameters (i.e. compute shader, global and
  //    local work group sizes) were updated
  if (requires_reencode_) {
    clear_deferred_cmds();
  }
}

} // namespace vkcompute
