// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanEngineExecutable.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/MemoryPlanning.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/Scalar.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>
#include <executorch/backends/native/runtime/vulkan/VulkanConstantMaterializationTracker.h>
#include <executorch/backends/native/runtime/vulkan/passes/InsertPrepack.h>
#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>
#include <executorch/runtime/core/freeable_buffer.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace ptn {
namespace {

using vkcompute::ComputeGraph;
using vkcompute::GraphConfig;
using VkRef = vkcompute::ValueRef;
constexpr VkRef kNoOutputStaging = -1;
namespace vkapi = vkcompute::vkapi;
namespace utils = vkcompute::utils;

void release_owned_bytes(void* context, void*, size_t) {
  delete static_cast<std::shared_ptr<OwnedBytes>*>(context);
}

// Native ScalarType -> ET-VK vkapi::ScalarType. The ids are pinned to the same
// ExecuTorch base, but map explicitly so an unsupported dtype fails by name
// instead of being reinterpreted at another width.
vkapi::ScalarType to_vk_dtype(ScalarType t) {
  switch (t) {
    case ScalarType::Float:
      return vkapi::kFloat;
    case ScalarType::Half:
      return vkapi::kHalf;
    case ScalarType::Double:
      return vkapi::kDouble;
    case ScalarType::Int:
      return vkapi::kInt;
    case ScalarType::Long:
      return vkapi::kLong;
    case ScalarType::Char:
      return vkapi::kChar;
    case ScalarType::Byte:
      return vkapi::kByte;
    case ScalarType::Bool:
      return vkapi::kBool;
    case ScalarType::Short:
    case ScalarType::BFloat16:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      break;
  }
  throw std::runtime_error(
      std::string("vulkan: unsupported dtype ") + scalar_type_name(t));
}

ScalarType device_dtype(ScalarType dtype) {
  return dtype == ScalarType::Long ? ScalarType::Int : dtype;
}

size_t logical_numel(const TensorMeta& meta) {
  const int64_t numel = meta.numel();
  if (static_cast<uint64_t>(numel) > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("vulkan: tensor element count overflows");
  }
  return static_cast<size_t>(numel);
}

size_t logical_numel(const std::vector<int64_t>& sizes) {
  TensorMeta meta;
  meta.sizes = sizes;
  return logical_numel(meta);
}

size_t logical_nbytes(const TensorMeta& meta) {
  const size_t element_bytes = element_size(meta.dtype);
  if (element_bytes == 0) {
    throw std::runtime_error("vulkan: tensor dtype has zero element size");
  }
  const size_t numel = logical_numel(meta);
  if (numel > std::numeric_limits<size_t>::max() / element_bytes) {
    throw std::runtime_error("vulkan: tensor byte size overflows");
  }
  return numel * element_bytes;
}

// ET-VK operator-registry key for a native fx target: strips the leading
// "torch.ops." namespace (torch.ops.aten.add.Tensor -> aten.add.Tensor) and
// renames targets whose ET-VK kernel is registered under another key.
std::string registry_key(const std::string& target) {
  constexpr std::string_view kPrefix = "torch.ops.";
  std::string key =
      target.starts_with(kPrefix) ? target.substr(kPrefix.size()) : target;
  if (key == "aten.relu_.default") {
    key = "aten.relu.default";
  } else if (key == "llama.update_cache.default") {
    key = "update_cache.default";
  } else if (key == "aten.sym_size.int") {
    key = "sym_size.int";
  } else {
    constexpr std::array<std::string_view, 4> kSymbolicOps{
        "add", "floordiv", "mul", "sub"};
    constexpr std::string_view kOperatorPrefix = "_operator.";
    if (key.starts_with(kOperatorPrefix)) {
      const std::string_view candidate =
          std::string_view(key).substr(kOperatorPrefix.size());
      if (std::ranges::find(kSymbolicOps, candidate) != kSymbolicOps.end()) {
        key = candidate;
      }
    }
  }
  return key;
}

// In-place targets that registry_key maps to a functional ET-VK kernel.
bool is_functional_in_place(const std::string& target) {
  return target == "torch.ops.aten.relu_.default" ||
      target == "aten.relu_.default";
}

// Ops whose ET-VK implementations require width-packed operands. This list
// seeds layout assignment; propagation below determines the final layouts.
bool is_width_packed_op(std::string_view key) {
  constexpr std::array<std::string_view, 9> kOps{
      "aten.linear.default",
      "aten.addmm.default",
      "aten.mm.default",
      "aten.bmm.default",
      "et_vk.linear_dq8ca_q4gsw.default",
      "et_vk.rms_norm.default",
      "llama.custom_sdpa.default",
      "update_cache.default",
      "torchao.choose_qparams_affine.default"};
  return std::ranges::find(kOps, key) != kOps.end();
}

bool preserves_packed_layout(std::string_view key) {
  constexpr std::array<std::string_view, 5> kOps{
      "aten._to_copy.default",
      "aten.add.Tensor",
      "aten.mul.Tensor",
      "aten.sigmoid.default",
      "aten.sub.Tensor"};
  return std::ranges::find(kOps, key) != kOps.end();
}

struct Reduce2dLayoutConstraint {
  ValueId input_id;
  std::array<bool, 3> allowed_packed_dims;
};

std::optional<Reduce2dLayoutConstraint> reduce2d_layout_constraint(
    const Graph& graph,
    const Node& node) {
  constexpr std::array<std::string_view, 4> kReduceOps{
      "aten.amax.default",
      "aten.amin.default",
      "aten.mean.dim",
      "aten.sum.dim_IntList",
  };
  if (std::ranges::find(kReduceOps, registry_key(node.target)) ==
      kReduceOps.end()) {
    return std::nullopt;
  }

  const auto input = std::ranges::find_if(node.inputs, [](const auto& arg) {
    return arg.name == "self" && arg.arg.kind() == ArgKind::Tensor;
  });
  const auto dims = std::ranges::find_if(node.inputs, [](const auto& arg) {
    return arg.name == "dim" && arg.arg.kind() == ArgKind::IntList;
  });
  if (input == node.inputs.end() || dims == node.inputs.end() ||
      dims->arg.as_int_list().values.size() != 2) {
    return std::nullopt;
  }
  if (!dims->arg.as_int_list().ids.empty()) {
    throw std::runtime_error(
        "vulkan: symbolic dimensions are not supported for 2D reductions");
  }

  const ValueId input_id = input->arg.as_tensor().id;
  const int64_t ndim =
      static_cast<int64_t>(graph.value(input_id).tensor_meta().ndim());
  std::array<bool, 3> allowed{true, true, true};
  for (int64_t dim : dims->arg.as_int_list().values) {
    dim = dim < 0 ? dim + ndim : dim;
    if (dim < 0 || dim >= ndim) {
      throw std::runtime_error("vulkan: reduction dimension is out of range");
    }
    const int64_t whcn_dim = ndim - 1 - dim;
    if (whcn_dim < static_cast<int64_t>(allowed.size())) {
      allowed[static_cast<size_t>(whcn_dim)] = false;
    }
  }
  return Reduce2dLayoutConstraint{input_id, allowed};
}

int32_t packed_dim(utils::GPUMemoryLayout layout) {
  if (layout == utils::kWidthPacked) {
    return 0;
  }
  if (layout == utils::kHeightPacked) {
    return 1;
  }
  if (layout == utils::kChannelsPacked) {
    return 2;
  }
  throw std::runtime_error("vulkan: unsupported packed layout");
}

utils::GPUMemoryLayout layout_for_packed_dim(int32_t dim) {
  constexpr std::array<utils::GPUMemoryLayout, 3> kLayouts{
      utils::kWidthPacked,
      utils::kHeightPacked,
      utils::kChannelsPacked,
  };
  return kLayouts.at(static_cast<size_t>(dim));
}

bool exceeds_texture_limit(
    const TensorMeta& meta,
    utils::GPUMemoryLayout layout,
    uint32_t limit) {
  if (meta.sizes.size() > 4) {
    return true;
  }
  std::array<uint64_t, 4> sizes{1, 1, 1, 1};
  std::copy(
      meta.sizes.begin(), meta.sizes.end(), sizes.end() - meta.sizes.size());
  if (layout == utils::kWidthPacked) {
    sizes[3] = (sizes[3] + 3) / 4;
  } else if (layout == utils::kHeightPacked) {
    sizes[2] = (sizes[2] + 3) / 4;
  } else if (layout == utils::kChannelsPacked) {
    sizes[1] = (sizes[1] + 3) / 4;
  }
  return sizes[3] > limit || sizes[2] > limit || sizes[0] * sizes[1] > limit;
}

bool requires_bias_tensor(std::string_view key) {
  return key == "aten.linear.default" || key == "aten.convolution.default";
}

bool is_metadata_only_view(std::string_view key) {
  constexpr std::array<std::string_view, 7> kOps{
      "aten.alias.default",
      "aten.permute.default",
      "aten.select.int",
      "aten.squeeze.dim",
      "aten.squeeze.dims",
      "aten.transpose.int",
      "aten.unsqueeze.default",
  };
  return key == "aten.view.default" ||
      std::ranges::find(kOps, key) != kOps.end();
}

MemoryKind memory_kind(utils::StorageType storage) {
  switch (storage) {
    case utils::StorageType::BUFFER:
      return 0;
    case utils::StorageType::TEXTURE_3D:
      return 1;
    case utils::StorageType::TEXTURE_2D:
      return 2;
  }
  throw std::runtime_error("vulkan: unrecognized storage type");
}

template <typename Fn>
void for_each_output_value(const Node& node, Fn&& fn) {
  for (const Output& output : node.outputs) {
    if (output.kind == OutputValueKind::TensorList) {
      for (const ValueId id : output.elem_ids) {
        if (valid(id)) {
          fn(id);
        }
      }
    } else if (valid(output.value_id)) {
      fn(output.value_id);
    }
  }
}

// cppcheck-suppress-begin useStlAlgorithm
void validate_user_io(const Method& method) {
  for (const ValueId id : method.graph.input_ids) {
    if (!method.graph.value(id).is_tensor()) {
      throw std::runtime_error("vulkan: user inputs must be tensors");
    }
  }
  for (const ValueId id : method.graph.output_ids) {
    if (!method.graph.value(id).is_tensor()) {
      throw std::runtime_error("vulkan: graph outputs must be tensors");
    }
  }
  const auto is_quantized = [&method](ValueId id) {
    return method.graph.value(id).tensor_meta().quant.has_value();
  };
  if (std::ranges::any_of(method.graph.input_ids, is_quantized) ||
      std::ranges::any_of(method.graph.output_ids, is_quantized)) {
    throw std::runtime_error(
        "vulkan: quantized graph inputs and outputs are not supported");
  }
  // set_input and get_output copy contiguous bytes.
  const auto is_contiguous = [&method](ValueId id) {
    return method.graph.value(id).tensor_meta().is_contiguous();
  };
  if (!std::ranges::all_of(method.graph.input_ids, is_contiguous) ||
      !std::ranges::all_of(method.graph.output_ids, is_contiguous)) {
    throw std::runtime_error(
        "vulkan: graph inputs and outputs must be contiguous");
  }
}
// cppcheck-suppress-end useStlAlgorithm

void validate_int64_to_int32(const void* data, size_t numel) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < numel; ++i) {
    int64_t value;
    std::memcpy(&value, bytes + i * sizeof(value), sizeof(value));
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(
          "vulkan: int64 input value is outside int32 range");
    }
  }
}

// Holds a method's non-empty mutable DataBinding keys in `owners` for the
// lifetime of its executable.
class MutableStateClaim {
 public:
  MutableStateClaim(VulkanMutableStateOwners& owners, const Method& method)
      : owners_(owners) {
    for (const DataBinding& binding : method.data_bindings) {
      if (!binding.mutated || binding.key.empty()) {
        continue;
      }
      const auto owner = owners_.find(binding.key);
      if (owner != owners_.end()) {
        // TODO(Native-VK): share it instead: the context would own one buffer
        // per key and each graph would wrap it via
        // ComputeGraph::add_tensor(const vkapi::VulkanBuffer&).
        throw std::runtime_error(
            "vulkan: mutable state '" + binding.key + "' of method '" +
            method.name + "' is already held by a live executable of method '" +
            owner->second +
            "'; sharing mutable state across executables is not implemented");
      }
    }
    for (const DataBinding& binding : method.data_bindings) {
      if (binding.mutated && !binding.key.empty() &&
          owners_.try_emplace(binding.key, method.name).second) {
        keys_.push_back(binding.key);
      }
    }
  }

  MutableStateClaim(const MutableStateClaim&) = delete;
  MutableStateClaim& operator=(const MutableStateClaim&) = delete;
  MutableStateClaim(MutableStateClaim&&) = delete;
  MutableStateClaim& operator=(MutableStateClaim&&) = delete;

  ~MutableStateClaim() {
    for (const std::string& key : keys_) {
      owners_.erase(key);
    }
  }

 private:
  VulkanMutableStateOwners& owners_;
  std::vector<std::string> keys_;
};

// One region of a Method lowered onto an ET-VK ComputeGraph.
//
// Deliberately not in the header: EngineContext::compile hands this back
// through the base pointer, so nothing outside this file needs to name it.
class VulkanEngineExecutable final : public EngineExecutable {
 private:
  Method method_;
  VulkanConstantMaterializationTracker& materializations_;
  MutableStateClaim mutable_state_claim_;

  // The region this executable covers. Today this is the whole method. The
  // lowering helpers still inspect the full method graph, so narrower runtime
  // partitioning requires auditing those walks first.
  std::vector<NodeId> nodes_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
  // Public output index -> graph output position. Mutation outputs remain in
  // outputs_ for execution but are not part of the EngineExecutable API.
  std::vector<size_t> user_output_positions_;

  std::unique_ptr<ComputeGraph> graph_;
  // native ValueId -> ET-VK ValueRef, one slot per value in the method graph.
  std::vector<VkRef> vref_;
  // Dynamic integer outputs need mutable ET-VK SymInt values rather than the
  // placeholder None payload carried by the serialized native Value.
  std::vector<uint8_t> symint_;
  // native ValueId -> the method binding that supplies its storage, null for a
  // value with none. Indexed so the value walk can ask about external storage
  // without scanning the binding list per value.
  std::vector<const DataBinding*> binding_;
  // Chosen memory layout per native value; channels-packed backbone by default.
  std::vector<utils::GPUMemoryLayout> layout_;
  std::vector<utils::StorageType> storage_;
  MemoryPlan memory_plan_;
  VkRef none_ref_ = -1;
  // Staging refs, aligned to inputs_ / outputs_.
  std::vector<VkRef> input_staging_;
  std::vector<VkRef> output_staging_;
  // Backing bytes for synthesized zero biases. add_tensorref keeps each data
  // pointer until prepack.
  std::deque<std::vector<uint8_t>> synth_bias_;
  std::vector<uint8_t> zero_state_;
  // Canonical owner -> bytes shared by tensor references until ET-VK consumes
  // and releases them during prepack.
  std::unordered_map<std::string, std::shared_ptr<OwnedBytes>> constants_;

  // vref_ and layout_ are sized to the method graph's value list, so every
  // ValueId the graph hands out addresses them. An id that does not means the
  // tables have gone out of sync with the graph; throw rather than read past
  // the end, matching Graph's own bounds-checked accessors.
  template <typename Vec>
  static auto& checked_slot(Vec& vec, ValueId id, const char* what) {
    if (!in_bounds(id, vec.size())) {
      throw std::runtime_error(
          std::string("vulkan: ") + what + " id " + std::to_string(id) +
          " does not address the value list");
    }
    return vec[static_cast<size_t>(id)];
  }
  VkRef& vref_at(ValueId id) {
    return checked_slot(vref_, id, "vref");
  }
  const VkRef& vref_at(ValueId id) const {
    return checked_slot(vref_, id, "vref");
  }
  utils::GPUMemoryLayout& layout_at(ValueId id) {
    return checked_slot(layout_, id, "layout");
  }
  utils::StorageType& storage_at(ValueId id) {
    return checked_slot(storage_, id, "storage");
  }
  const DataBinding* binding_at(ValueId id) {
    return checked_slot(binding_, id, "binding");
  }

 public:
  VulkanEngineExecutable(
      const Method& method,
      const Package& package,
      VulkanConstantMaterializationTracker& materializations,
      VulkanMutableStateOwners& mutable_state_owners,
      const GraphConfig& config)
      : method_(method),
        materializations_(materializations),
        mutable_state_claim_(mutable_state_owners, method),
        graph_(std::make_unique<ComputeGraph>(config)) {
    method_.graph.rebuild_def_use();
    vulkan::insert_prepack_nodes(method_);
    validate_graph(method_.graph);
    validate_user_io(method_);
    nodes_ = method_.graph.schedule;
    inputs_ = method_.graph.input_ids;
    outputs_ = method_.graph.output_ids;
    if (!method_.output_specs.empty() &&
        method_.output_specs.size() != outputs_.size()) {
      throw std::runtime_error(
          "vulkan: output_specs must be empty or match graph outputs");
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (method_.output_specs.empty() ||
          method_.output_specs.at(i).kind == OutputKind::UserOutput) {
        user_output_positions_.push_back(i);
      }
    }
    build(package);
  }

  size_t num_inputs() const override {
    return inputs_.size();
  }

  size_t num_outputs() const override {
    return user_output_positions_.size();
  }

  std::vector<int64_t> input_sizes(size_t i) const override {
    return graph_->sizes_of(vref_at(inputs_.at(i)));
  }

  std::vector<int64_t> output_sizes(size_t i) const override {
    return graph_->sizes_of(vref_at(outputs_.at(user_output_positions_.at(i))));
  }

  ScalarType input_dtype(size_t i) const override {
    return meta_of(inputs_.at(i)).dtype;
  }

  ScalarType output_dtype(size_t i) const override {
    return meta_of(outputs_.at(user_output_positions_.at(i))).dtype;
  }

  void resize_input(size_t i, const std::vector<int64_t>& sizes) override {
    const ValueId input = inputs_.at(i);
    if (!meta_of(input).accepts_sizes(sizes)) {
      throw std::runtime_error(
          "vulkan: input shape is outside serialized bounds");
    }
    graph_->resize_input(static_cast<int64_t>(i), sizes);
  }

  void set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype)
      override {
    const TensorMeta& meta = meta_of(inputs_.at(i));
    if (numel != logical_numel(input_sizes(i))) {
      throw std::runtime_error(
          "vulkan: input element count does not match its current shape");
    }
    if (numel == 0) {
      return;
    }
    if (data == nullptr) {
      throw std::runtime_error("vulkan: input data is null");
    }
    if (src_dtype == ScalarType::Long &&
        device_dtype(meta.dtype) == ScalarType::Int) {
      validate_int64_to_int32(data, numel);
    }
    graph_->maybe_cast_and_copy_into_staging(
        input_staging_.at(i), data, numel, to_vk_dtype(src_dtype));
  }

  void execute() override {
    graph_->propagate_resize();
    graph_->execute();
  }

  void get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype)
      override {
    const size_t output_position = user_output_positions_.at(i);
    if (numel != logical_numel(output_sizes(i))) {
      throw std::runtime_error(
          "vulkan: output element count does not match its current shape");
    }
    if (numel == 0) {
      return;
    }
    if (data == nullptr) {
      throw std::runtime_error("vulkan: output data is null");
    }
    graph_->maybe_cast_and_copy_from_staging(
        output_staging_.at(output_position),
        data,
        numel,
        to_vk_dtype(dst_dtype));
  }

 private:
  const Graph& g() const {
    return method_.graph;
  }

  const TensorMeta& meta_of(ValueId id) const {
    return g().value(id).tensor_meta();
  }

  void assign_layouts() {
    layout_.assign(g().values.size(), utils::kChannelsPacked);
    storage_.assign(g().values.size(), utils::kTexture3D);

    const auto mark_width = [&](ValueId id) {
      if (!valid(id) || !g().value(id).is_tensor() ||
          layout_at(id) == utils::kWidthPacked) {
        return false;
      }
      layout_at(id) = utils::kWidthPacked;
      return true;
    };
    const auto mark_node_width = [&](const Node& node) {
      bool changed = false;
      for (const ValueId input_id : node.input_value_ids()) {
        changed |= mark_width(input_id);
      }
      for_each_output_value(
          node, [&](ValueId output_id) { changed |= mark_width(output_id); });
      return changed;
    };
    const auto node_is_width = [&](const Node& node) {
      if (std::ranges::any_of(node.input_value_ids(), [&](ValueId input_id) {
            return g().value(input_id).is_tensor() &&
                layout_at(input_id) == utils::kWidthPacked;
          })) {
        return true;
      }
      bool output_is_width = false;
      for_each_output_value(node, [&](ValueId output_id) {
        output_is_width |= g().value(output_id).is_tensor() &&
            layout_at(output_id) == utils::kWidthPacked;
      });
      return output_is_width;
    };

    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call() || !is_width_packed_op(registry_key(n.target))) {
        continue;
      }
      mark_node_width(n);
    }

    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call() || registry_key(n.target) != "aten.index.Tensor") {
        continue;
      }
      const std::vector<ValueId> inputs = n.input_value_ids();
      if (inputs.empty() || !valid(inputs.front()) ||
          !g().value(inputs.front()).is_tensor() ||
          g().value(inputs.front()).tensor_meta().sizes.size() <= 1) {
        continue;
      }
      mark_node_width(n);
      for (const ValueId input_id : inputs) {
        if (valid(input_id) && g().value(input_id).is_tensor()) {
          storage_at(input_id) = utils::kBuffer;
        }
      }
      for_each_output_value(n, [&](ValueId output_id) {
        if (g().value(output_id).is_tensor()) {
          storage_at(output_id) = utils::kBuffer;
        }
      });
    }

    bool changed;
    do {
      changed = false;
      for (const NodeId node_id : nodes_) {
        const Node& node = g().node(node_id);
        if (node.is_call() &&
            preserves_packed_layout(registry_key(node.target)) &&
            node_is_width(node)) {
          changed |= mark_node_width(node);
        }
      }
      for (ValueId id = 0; id < static_cast<ValueId>(g().values.size()); ++id) {
        const ValueId source = g().value(id).alias_id;
        if (valid(source) &&
            (layout_at(id) == utils::kWidthPacked ||
             layout_at(source) == utils::kWidthPacked)) {
          changed |= mark_width(id);
          changed |= mark_width(source);
        }
      }
    } while (changed);

    for (const NodeId node_id : nodes_) {
      const Node& node = g().node(node_id);
      if (!node.is_call()) {
        continue;
      }
      const auto constraint = reduce2d_layout_constraint(g(), node);
      if (!constraint) {
        continue;
      }
      utils::GPUMemoryLayout required = layout_at(constraint->input_id);
      if (!constraint->allowed_packed_dims.at(
              static_cast<size_t>(packed_dim(required)))) {
        for (int32_t dim = 2; dim >= 0; --dim) {
          if (constraint->allowed_packed_dims.at(static_cast<size_t>(dim))) {
            required = layout_for_packed_dim(dim);
            break;
          }
        }
      }
      layout_at(constraint->input_id) = required;
      for_each_output_value(
          node, [&](ValueId output_id) { layout_at(output_id) = required; });
    }

    const uint32_t texture_limit =
        graph_->context()->adapter_ptr()->max_texture3d_dim();
    for (ValueId id = 0; id < static_cast<ValueId>(g().values.size()); ++id) {
      const Value& value = g().value(id);
      if (value.is_tensor() &&
          exceeds_texture_limit(
              value.tensor_meta(), layout_at(id), texture_limit)) {
        storage_at(id) = utils::kBuffer;
      }
    }
    // Sibling views and alias chains need repeated passes before every alias
    // agrees with the storage of its root.
    do {
      changed = false;
      for (ValueId id = 0; id < static_cast<ValueId>(g().values.size()); ++id) {
        const ValueId source = g().value(id).alias_id;
        if (!valid(source)) {
          continue;
        }
        if (storage_at(id) == utils::kBuffer &&
            storage_at(source) != utils::kBuffer) {
          storage_at(source) = utils::kBuffer;
          changed = true;
        }
        if (storage_at(id) != storage_at(source)) {
          storage_at(id) = storage_at(source);
          changed = true;
        }
      }
    } while (changed);
  }

  void materialize_packed_views() {
    std::vector<ValueId> value_ids;
    for (const NodeId node_id : nodes_) {
      const Node& node = g().node(node_id);
      if (!node.is_call() || registry_key(node.target) != "aten.view.default") {
        continue;
      }
      for_each_output_value(node, [&](ValueId output_id) {
        const ValueId source_id = g().value(output_id).alias_id;
        if (valid(source_id) && layout_at(output_id) != layout_at(source_id) &&
            storage_at(output_id) != utils::kBuffer) {
          value_ids.push_back(output_id);
        }
      });
    }
    vulkan::materialize_view_copies(method_.graph, value_ids);
    validate_graph(method_.graph);
  }

  // Resolve a bound value's package bytes and verify them against the
  // serialized metadata before upload.
  executorch::runtime::FreeableBuffer constant_buffer(
      const Package& package,
      const DataBinding& b,
      const Value& v) {
    const std::optional<ConstantInfo> info = package.constant_info(b.key);
    if (!info) {
      throw std::runtime_error(
          "vulkan: package holds no constant for key '" + b.key + "'");
    }
    if (info->dtype != v.tensor_meta().dtype) {
      throw std::runtime_error(
          "vulkan: constant '" + b.key + "' is " +
          scalar_type_name(info->dtype) + " but the program declares " +
          scalar_type_name(v.tensor_meta().dtype));
    }
    const size_t expected = logical_nbytes(v.tensor_meta());
    if (info->nbytes != expected) {
      throw std::runtime_error(
          "vulkan: constant '" + b.key + "' holds " +
          std::to_string(info->nbytes) + " bytes but its shape and dtype " +
          "need " + std::to_string(expected));
    }
    std::shared_ptr<OwnedBytes>& storage = constants_[info->owner];
    if (!storage) {
      std::optional<OwnedBytes> acquired =
          package.acquire_constant(info->owner);
      if (!acquired) {
        throw std::runtime_error(
            "vulkan: package could not load constant for key '" + b.key + "'");
      }
      const ByteSpan source = acquired->span();
      if (source.size() != expected) {
        throw std::runtime_error(
            "vulkan: loaded constant '" + b.key + "' has " +
            std::to_string(source.size()) + " bytes but metadata declares " +
            std::to_string(expected));
      }
      if (info->dtype == ScalarType::Long) {
        std::vector<uint8_t> converted(
            static_cast<size_t>(v.tensor_meta().numel()) * sizeof(int32_t));
        for (size_t i = 0; i < converted.size() / sizeof(int32_t); ++i) {
          int64_t value;
          std::memcpy(&value, source.data() + i * sizeof(value), sizeof(value));
          if (value < std::numeric_limits<int32_t>::min() ||
              value > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(
                "vulkan: int64 constant value is outside int32 range");
          }
          const int32_t downcast = static_cast<int32_t>(value);
          std::memcpy(
              converted.data() + i * sizeof(downcast),
              &downcast,
              sizeof(downcast));
        }
        storage = std::make_shared<OwnedBytes>(
            OwnedBytes::from_vector(std::move(converted)));
      } else {
        storage = std::make_shared<OwnedBytes>(std::move(acquired.value()));
      }
      materializations_.record(info->package_id, info->owner, info->nbytes);
    }
    const ByteSpan bytes = storage->span();
    auto* context = new std::shared_ptr<OwnedBytes>(storage);
    return executorch::runtime::FreeableBuffer(
        bytes.data(), bytes.size(), release_owned_bytes, context);
  }

  VkRef make_value(const Package& package, const Value& v, ValueId native_id) {
    switch (v.kind()) {
      case ValueKind::None:
        return checked_slot(symint_, native_id, "symint") != 0
            ? graph_->add_symint(0)
            : graph_->add_none();
      case ValueKind::Scalar: {
        const Scalar& s = v.scalar();
        if (s.is_int()) {
          return graph_->add_scalar<int64_t>(s.to_int());
        }
        if (s.is_double()) {
          return graph_->add_scalar<double>(s.to_double());
        }
        return graph_->add_scalar<bool>(s.to_bool());
      }
      case ValueKind::List: {
        const std::vector<ValueId>& ids = v.content_ids();
        std::vector<VkRef> refs(ids.size());
        std::transform(ids.begin(), ids.end(), refs.begin(), [this](ValueId e) {
          return valid(e) ? vref_at(e) : none_ref_;
        });
        return graph_->add_value_list(std::move(refs));
      }
      case ValueKind::Tensor: {
        const TensorMeta& m = v.tensor_meta();
        const std::vector<int64_t>& sizes = m.sizes;
        if (valid(v.alias_id)) {
          const VkRef source = vref_at(v.alias_id);
          const TensorMeta& source_meta = g().value(v.alias_id).tensor_meta();
          if (m.dtype != source_meta.dtype ||
              logical_nbytes(m) > logical_nbytes(source_meta)) {
            throw std::runtime_error(
                "vulkan: alias must match its source dtype and fit within it");
          }
          std::vector<int64_t> dim_order(
              m.dim_order_hint.begin(), m.dim_order_hint.end());
          if (dim_order.empty()) {
            dim_order.resize(m.sizes.size());
            std::iota(dim_order.begin(), dim_order.end(), 0);
          }
          return graph_->add_tensor_view(source, sizes, dim_order);
        }
        const vkapi::ScalarType dt = to_vk_dtype(device_dtype(m.dtype));
        const DataBinding* b = binding_at(native_id);
        if (b != nullptr && b->has_data) {
          const VkRef source =
              graph_->add_tensorref(sizes, dt, constant_buffer(package, *b, v));
          if (!b->mutated) {
            return source;
          }
          const VkRef tensor = graph_->add_tensor(
              sizes,
              dt,
              storage_at(native_id),
              layout_at(native_id),
              memory_plan_.allocation_id(native_id));
          VK_GET_OP_FN("et_vk.prepack.default")(*graph_, {source, tensor});
          return tensor;
        }
        if (b != nullptr && b->mutated) {
          return zero_initialized_tensor(v, native_id);
        }
        return graph_->add_tensor(
            sizes,
            dt,
            storage_at(native_id),
            layout_at(native_id),
            memory_plan_.allocation_id(native_id));
      }
    }
    throw std::runtime_error("vulkan: unhandled value kind");
  }

  // Resolve one native argument to an ET-VK ValueRef, materializing scalar and
  // list literals as fresh graph values.
  VkRef resolve_arg(const Argument& arg) {
    switch (arg.kind()) {
      case ArgKind::None:
        return none_ref_;
      case ArgKind::Tensor: {
        const ValueId t = arg.as_tensor().id;
        return valid(t) ? vref_at(t) : none_ref_;
      }
      case ArgKind::Int: {
        const IntArg& a = arg.as_int();
        return valid(a.id) ? vref_at(a.id)
                           : graph_->add_scalar<int64_t>(a.value);
      }
      case ArgKind::Float: {
        const FloatArg& a = arg.as_float();
        return valid(a.id) ? vref_at(a.id)
                           : graph_->add_scalar<double>(a.value);
      }
      case ArgKind::Bool: {
        const BoolArg& a = arg.as_bool();
        return valid(a.id) ? vref_at(a.id) : graph_->add_scalar<bool>(a.value);
      }
      case ArgKind::IntList: {
        const IntListArg& list = arg.as_int_list();
        if (list.ids.empty()) {
          return graph_->add_scalar_list(std::vector<int64_t>(list.values));
        }
        if (list.ids.size() != list.values.size()) {
          throw std::runtime_error(
              "vulkan: symbolic IntList ids and values must have equal size");
        }
        std::vector<VkRef> refs;
        refs.reserve(list.values.size());
        for (size_t i = 0; i < list.values.size(); ++i) {
          refs.push_back(
              valid(list.ids.at(i))
                  ? vref_at(list.ids.at(i))
                  : graph_->add_scalar<int64_t>(list.values[i]));
        }
        return graph_->add_value_list(std::move(refs));
      }
      case ArgKind::FloatList:
        return graph_->add_scalar_list(
            std::vector<double>(arg.as_float_list().values));
      case ArgKind::BoolList:
        return graph_->add_scalar_list(
            std::vector<bool>(arg.as_bool_list().values));
      case ArgKind::String:
        return graph_->add_string(std::string(arg.as_string().value));
      case ArgKind::ScalarType:
        return graph_->add_scalar<int64_t>(
            static_cast<int64_t>(arg.as_scalar_type().value));
      case ArgKind::TensorList: {
        const std::vector<ValueId>& ids = arg.as_tensor_list().ids;
        std::vector<VkRef> refs(ids.size());
        std::transform(ids.begin(), ids.end(), refs.begin(), [this](ValueId e) {
          return valid(e) ? vref_at(e) : none_ref_;
        });
        return graph_->add_value_list(std::move(refs));
      }
      case ArgKind::OptionalTensorList: {
        const std::vector<ValueId>& ids = arg.as_optional_tensor_list().ids;
        std::vector<VkRef> refs(ids.size());
        std::transform(ids.begin(), ids.end(), refs.begin(), [this](ValueId e) {
          return valid(e) ? vref_at(e) : none_ref_;
        });
        return graph_->add_value_list(std::move(refs));
      }
      case ArgKind::Graph:
        throw std::runtime_error(
            "vulkan: higher-order-op subgraphs not supported");
    }
    throw std::runtime_error("vulkan: unhandled argument kind");
  }

  // Insert a same-shape clone into a fresh width-packed tensor when an
  // activation's current layout or storage cannot satisfy its consumer.
  VkRef repack_to_width(VkRef src, const Value& v, utils::StorageType storage) {
    const std::vector<int64_t>& sizes = v.tensor_meta().sizes;
    const VkRef dst = graph_->add_tensor(
        sizes,
        to_vk_dtype(device_dtype(v.tensor_meta().dtype)),
        storage,
        utils::kWidthPacked);
    std::vector<VkRef> args = {src, none_ref_, dst};
    VK_GET_OP_FN("aten.clone.default")(*graph_, args);
    return dst;
  }

  // ET-VK conv and linear require a real bias tensor, but the native graph
  // carries bias=None for bias-free layers. Synthesize a zero bias shaped
  // [out_channels] in the weight's dtype. A transposed conv weight is
  // [in_channels, out_channels / groups, ...].
  //
  // Finds the weight by argument name, which is what the current op set gives
  // us; a schema-driven lookup would be needed for anything beyond conv/linear.
  static const Argument* find_input(const Node& n, std::string_view name) {
    const auto it = std::find_if(
        n.inputs.begin(), n.inputs.end(), [name](const NamedArgument& na) {
          return na.name == name;
        });
    return it == n.inputs.end() ? nullptr : &it->arg;
  }

  static bool
  constant_bool_input(const Node& n, std::string_view name, bool fallback) {
    const Argument* arg = find_input(n, name);
    if (arg == nullptr) {
      return fallback;
    }
    if (arg->kind() != ArgKind::Bool || valid(arg->as_bool().id)) {
      throw std::runtime_error(
          "vulkan: '" + std::string(name) + "' must be a constant bool");
    }
    return arg->as_bool().value;
  }

  static int64_t
  constant_int_input(const Node& n, std::string_view name, int64_t fallback) {
    const Argument* arg = find_input(n, name);
    if (arg == nullptr) {
      return fallback;
    }
    if (arg->kind() != ArgKind::Int || valid(arg->as_int().id)) {
      throw std::runtime_error(
          "vulkan: '" + std::string(name) + "' must be a constant int");
    }
    return arg->as_int().value;
  }

  VkRef synth_zero_bias(const Node& n) {
    const auto weight = std::find_if(
        n.inputs.begin(), n.inputs.end(), [](const NamedArgument& na) {
          return na.name == "weight" && na.arg.kind() == ArgKind::Tensor &&
              valid(na.arg.as_tensor().id);
        });
    if (weight == n.inputs.end()) {
      throw std::runtime_error(
          "vulkan: cannot synthesize a bias (no weight argument found)");
    }
    const TensorMeta& m = meta_of(weight->arg.as_tensor().id);
    if (m.sizes.empty()) {
      throw std::runtime_error(
          "vulkan: cannot synthesize a bias from a scalar weight");
    }
    const bool transposed = constant_bool_input(n, "transposed", false);
    const int64_t out_channels = transposed
        ? m.sizes.at(1) * constant_int_input(n, "groups", 1)
        : m.sizes[0];
    const ScalarType dtype = device_dtype(m.dtype);
    const TensorMeta bias_meta{
        .dtype = dtype,
        .sizes = {out_channels},
    };
    synth_bias_.emplace_back(logical_nbytes(bias_meta), 0);
    return graph_->add_tensorref(
        {out_channels}, to_vk_dtype(dtype), synth_bias_.back().data());
  }

  // A data-less mutable buffer starts as zeros. Every such buffer prepacks from
  // one shared host zero block sized for the largest of them.
  VkRef zero_initialized_tensor(const Value& v, ValueId native_id) {
    if (zero_state_.empty()) {
      size_t nbytes = 0;
      for (const DataBinding& b : method_.data_bindings) {
        if (b.mutated && !b.has_data) {
          nbytes = std::max(nbytes, logical_nbytes(device_meta(b.value_id)));
        }
      }
      zero_state_.assign(nbytes, 0);
    }
    const std::vector<int64_t>& sizes = v.tensor_meta().sizes;
    const vkapi::ScalarType dt =
        to_vk_dtype(device_dtype(v.tensor_meta().dtype));
    const VkRef source = graph_->add_tensorref(sizes, dt, zero_state_.data());
    const VkRef tensor = graph_->add_tensor(
        sizes,
        dt,
        storage_at(native_id),
        layout_at(native_id),
        memory_plan_.allocation_id(native_id));
    VK_GET_OP_FN("et_vk.prepack.default")(*graph_, {source, tensor});
    return tensor;
  }

  TensorMeta device_meta(ValueId id) const {
    const TensorMeta& m = meta_of(id);
    return TensorMeta{.dtype = device_dtype(m.dtype), .sizes = m.sizes};
  }

  void dispatch(const Node& n) {
    const std::string key = registry_key(n.target);
    if (!VK_HAS_OP(key)) {
      throw std::runtime_error(
          "vulkan: unsupported op '" + key + "' (target " + n.target + ")");
    }
    const bool width = is_width_packed_op(key);
    std::optional<utils::StorageType> activation_storage;
    if (key == "et_vk.linear_dq8ca_q4gsw.default") {
      const auto output =
          std::ranges::find_if(n.outputs, [this](const Output& candidate) {
            return valid(candidate.value_id) &&
                g().value(candidate.value_id).is_tensor();
          });
      if (output != n.outputs.end()) {
        activation_storage = storage_at(output->value_id);
      }
    }

    std::vector<VkRef> args;
    args.reserve(n.inputs.size() + 1);
    for (const NamedArgument& na : n.inputs) {
      if (na.name == "bias" && na.arg.kind() == ArgKind::None &&
          requires_bias_tensor(key)) {
        args.push_back(synth_zero_bias(n));
        continue;
      }
      VkRef a = resolve_arg(na.arg);
      if (width && na.arg.kind() == ArgKind::Tensor) {
        const ValueId value_id = na.arg.as_tensor().id;
        if (valid(value_id)) {
          const Value& v = g().value(value_id);
          const bool activation = v.role == ValueRole::Intermediate ||
              v.role == ValueRole::UserInput;
          const bool storage_mismatch = activation_storage.has_value() &&
              na.name == "input" && storage_at(value_id) != *activation_storage;
          if (activation &&
              (layout_at(value_id) != utils::kWidthPacked ||
               storage_mismatch)) {
            a = repack_to_width(
                a, v, activation_storage.value_or(storage_at(value_id)));
          }
        }
      }
      args.push_back(a);
    }

    // Outputs: a single tensor output is appended directly; otherwise every
    // output ref is grouped into one ValueList, which is ET-VK's multi-return
    // and Tensor[] calling convention, even for a one-element Tensor[].
    std::vector<VkRef> outs;
    for_each_output_value(
        n, [&](ValueId output_id) { outs.push_back(vref_at(output_id)); });
    const bool returns_list =
        std::ranges::any_of(n.outputs, [](const Output& output) {
          return output.kind == OutputValueKind::TensorList;
        });
    if (outs.size() == 1 && !returns_list) {
      args.push_back(outs[0]);
    } else if (!outs.empty()) {
      args.push_back(graph_->add_value_list(std::move(outs)));
    }

    VK_GET_OP_FN(key)(*graph_, args);
  }

  void index_output_kinds() {
    symint_.assign(g().values.size(), 0);
    for (const NodeId node_id : nodes_) {
      for (const Output& output : g().node(node_id).outputs) {
        if (output.kind == OutputValueKind::Int && valid(output.value_id)) {
          checked_slot(symint_, output.value_id, "symint") = 1;
        }
      }
    }
  }

  // The bindings live in the Method copy owned by this executable, so the
  // indexed pointers stay valid.
  void index_bindings() {
    binding_.assign(g().values.size(), nullptr);
    for (const DataBinding& b : method_.data_bindings) {
      if (in_bounds(b.value_id, binding_.size())) {
        binding_.at(static_cast<size_t>(b.value_id)) = &b;
      }
    }
  }

  // A functional kernel whose output views its input binds one storage for
  // both read and write. ET-VK records only the last binding's access, so the
  // write is forgotten and the next reader of the output gets no barrier. Such
  // ops run out of place instead, which is only equivalent while nothing else
  // reads the mutated tensor. Runs after layout assignment so the detached
  // output keeps its input's layout and storage.
  void detach_in_place_outputs() {
    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call() || !is_functional_in_place(n.target) ||
          n.inputs.empty() || n.inputs[0].arg.kind() != ArgKind::Tensor ||
          n.outputs.size() != 1) {
        continue;
      }
      const ValueId self_id = n.inputs[0].arg.as_tensor().id;
      const ValueId out_id = n.outputs[0].value_id;
      if (!valid(self_id) || !valid(out_id) ||
          g().value(out_id).alias_id != self_id) {
        continue;
      }
      const Value& self = g().value(self_id);
      const bool observed = (self.role != ValueRole::Intermediate &&
                             self.role != ValueRole::UserInput) ||
          valid(self.alias_id) || self.consumer_ids.size() != 1 ||
          std::ranges::find(outputs_, self_id) != outputs_.end();
      if (observed) {
        throw std::runtime_error(
            "vulkan: node '" + n.name + "' (" + n.target +
            "): in-place update of '" + self.name +
            "' is read elsewhere; only an unshared input can run out of place");
      }
      method_.graph.value(out_id).alias_id = kInvalid;
    }
  }

  void plan_allocations() {
    std::vector<AllocationRequest> requests;
    requests.reserve(g().values.size());
    for (ValueId id = 0; id < static_cast<ValueId>(g().values.size()); ++id) {
      const Value& value = g().value(id);
      if (!value.is_tensor()) {
        continue;
      }
      if (valid(value.alias_id) || value.role == ValueRole::Parameter ||
          value.role == ValueRole::ConstantTensor) {
        continue;
      }
      const DataBinding* binding = binding_at(id);
      // Prepacked values get a dedicated allocation during prepack.
      if (binding != nullptr && (binding->has_data || binding->mutated)) {
        continue;
      }
      const bool active = valid(value.producer_id) ||
          !value.consumer_ids.empty() || value.role == ValueRole::Buffer;
      if (!active) {
        continue;
      }
      requests.push_back(AllocationRequest{
          .value_id = id,
          .size_bytes = logical_nbytes(value.tensor_meta()),
          .memory_kind = memory_kind(storage_at(id)),
      });
    }
    memory_plan_ = plan_memory(g(), requests);
  }

  // execute() never copies a mutation output into its target, so the output
  // must already be the target's storage: the buffer itself or an alias chain
  // ending at it. Writes to a user input never reach the caller's host copy.
  void check_mutation_outputs() const {
    for (size_t i = 0; i < method_.output_specs.size(); ++i) {
      const OutputSpec& spec = method_.output_specs[i];
      if (spec.kind == OutputKind::UserOutput) {
        continue;
      }
      if (spec.kind == OutputKind::UserInputMutation) {
        throw std::runtime_error(
            "vulkan: mutation of user input '" +
            g().value(spec.target_id).name +
            "' is not supported; results are not copied back to the caller");
      }
      ValueId id = outputs_.at(i);
      while (id != spec.target_id && valid(g().value(id).alias_id)) {
        id = g().value(id).alias_id;
      }
      if (id != spec.target_id) {
        throw std::runtime_error(
            "vulkan: mutation output '" + g().value(outputs_.at(i)).name +
            "' is not written in place to '" + g().value(spec.target_id).name +
            "'; copying mutation results back is not implemented");
      }
    }
  }

  std::vector<ValueId> value_dependencies(const Value& v) const {
    if (v.kind() == ValueKind::List) {
      std::vector<ValueId> ids;
      std::ranges::copy_if(
          v.content_ids(), std::back_inserter(ids), [](ValueId id) {
            return valid(id);
          });
      return ids;
    }
    if (v.is_tensor() && valid(v.alias_id)) {
      return {v.alias_id};
    }
    return {};
  }

  // Creates values dependencies first: a view or list may precede the values
  // it refers to, e.g. when a pass appends the source of an existing view.
  void make_values(const Package& package) {
    none_ref_ = graph_->add_none();
    vref_.assign(g().values.size(), -1);
    std::vector<bool> on_path(g().values.size());
    std::vector<ValueId> stack;
    for (ValueId root = 0; root < static_cast<ValueId>(g().values.size());
         ++root) {
      stack.push_back(root);
      while (!stack.empty()) {
        const ValueId id = stack.back();
        if (vref_at(id) >= 0) {
          stack.pop_back();
          continue;
        }
        const Value& v = g().values[id];
        on_path.at(id) = true;
        const std::vector<ValueId> deps = value_dependencies(v);
        const auto missing = std::ranges::find_if(
            deps, [this](ValueId dep) { return vref_at(dep) < 0; });
        if (missing != deps.end()) {
          if (on_path.at(*missing)) {
            throw std::runtime_error(
                "vulkan: value '" + v.name + "' depends on itself");
          }
          stack.push_back(*missing);
          continue;
        }
        try {
          vref_at(id) = make_value(package, v, id);
        } catch (const std::exception& e) {
          throw std::runtime_error(
              "vulkan: value[" + std::to_string(id) + "] '" + v.name +
              "': " + e.what());
        }
        on_path.at(id) = false;
        stack.pop_back();
      }
    }
  }

  // ET-VK views share their source's base address, so select is only a view
  // at index 0.
  void check_view_offset(const Node& n) const {
    if (registry_key(n.target) != "aten.select.int") {
      return;
    }
    const auto arg = [&n](std::string_view name) -> const IntArg* {
      const auto it = std::ranges::find(n.inputs, name, &NamedArgument::name);
      return it == n.inputs.end() || it->arg.kind() != ArgKind::Int ||
              valid(it->arg.as_int().id)
          ? nullptr
          : &it->arg.as_int();
    };
    const IntArg* dim = arg("dim");
    const IntArg* index = arg("index");
    const std::vector<int64_t>& sizes =
        g().value(n.inputs.at(0).arg.as_tensor().id).tensor_meta().sizes;
    const int64_t ndim = static_cast<int64_t>(sizes.size());
    if (dim == nullptr || index == nullptr || dim->value < -ndim ||
        dim->value >= ndim) {
      throw std::runtime_error("vulkan: select view needs a constant index");
    }
    const int64_t extent =
        sizes.at(dim->value < 0 ? dim->value + ndim : dim->value);
    if (index->value != 0 && index->value != -extent) {
      throw std::runtime_error(
          "vulkan: select view at a nonzero index is not supported");
    }
  }

  void build(const Package& package) {
    check_mutation_outputs();
    assign_layouts();
    materialize_packed_views();
    detach_in_place_outputs();
    index_output_kinds();
    index_bindings();
    plan_allocations();

    // 1. Materialize every native value as an ET-VK value.
    make_values(package);

    // 2. Register the region's inputs, creating host->device staging.
    input_staging_.resize(inputs_.size());
    std::transform(
        inputs_.begin(),
        inputs_.end(),
        input_staging_.begin(),
        [this](ValueId in) { return graph_->set_input_tensor(vref_at(in)); });

    // 3. Dispatch executable call nodes in schedule order.
    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call()) {
        continue;
      }
      if (is_metadata_only_view(registry_key(n.target)) &&
          std::ranges::all_of(n.outputs, [&](const Output& output) {
            return valid(output.value_id) &&
                valid(g().value(output.value_id).alias_id);
          })) {
        check_view_offset(n);
        continue;
      }
      try {
        dispatch(n);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            "vulkan: node '" + n.name + "' (" + n.target + "): " + e.what());
      }
    }

    // 4. Register graph outputs. User outputs get host staging; mutation
    // outputs remain device-resident.
    output_staging_.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const VkRef output = vref_at(outputs_[i]);
      const bool is_mutation = !method_.output_specs.empty() &&
          method_.output_specs.at(i).kind != OutputKind::UserOutput;
      if (is_mutation) {
        graph_->set_output_tensor(output, false);
        output_staging_.push_back(kNoOutputStaging);
      } else {
        const ScalarType output_type = meta_of(outputs_[i]).dtype;
        const vkapi::ScalarType staging_type = output_type == ScalarType::Half
            ? vkapi::kFloat
            : to_vk_dtype(device_dtype(output_type));
        output_staging_.push_back(
            graph_->set_output_tensor(output, staging_type));
      }
    }

    // 5. Finalize and upload constants. TensorRef owns the remaining shared
    // references and releases each source after its final prepack consumer.
    constants_.clear();
    graph_->prepare();
    graph_->prepare_pipelines();
    graph_->prepack();
  }
};

} // namespace

// cppcheck-suppress unusedFunction
std::unique_ptr<EngineExecutable> create_vulkan_engine_executable(
    const Method& method,
    const Package& package,
    VulkanConstantMaterializationTracker& materializations,
    VulkanMutableStateOwners& mutable_state_owners,
    const GraphConfig& config) {
  return std::make_unique<VulkanEngineExecutable>(
      method, package, materializations, mutable_state_owners, config);
}

} // namespace ptn
