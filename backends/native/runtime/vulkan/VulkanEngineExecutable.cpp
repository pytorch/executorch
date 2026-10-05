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
#include <limits>
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
#include <executorch/runtime/core/freeable_buffer.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace ptn {
namespace {

using vkcompute::ComputeGraph;
using vkcompute::GraphConfig;
using VkRef = vkcompute::ValueRef;
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

// Strip the leading "torch.ops." namespace so a native fx target becomes an
// ET-VK operator-registry key (torch.ops.aten.add.Tensor -> aten.add.Tensor).
std::string registry_key(const std::string& target) {
  constexpr std::string_view kPrefix = "torch.ops.";
  if (target.starts_with(kPrefix)) {
    return target.substr(kPrefix.size());
  }
  return target;
}

// Ops whose ET-VK implementations require width-packed operands. This list
// seeds layout assignment; propagation below determines the final layouts.
bool is_width_packed_op(std::string_view key) {
  constexpr std::array<std::string_view, 4> kOps{
      "aten.linear.default",
      "aten.addmm.default",
      "aten.mm.default",
      "aten.bmm.default"};
  return std::ranges::find(kOps, key) != kOps.end();
}

bool requires_bias_tensor(std::string_view key) {
  return key == "aten.linear.default" || key == "aten.convolution.default";
}

bool requires_buffer_storage(ScalarType dtype) {
  return dtype != ScalarType::Float && dtype != ScalarType::Half;
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

// cppcheck-suppress-begin useStlAlgorithm
void validate_supported_method(const Method& method) {
  const Graph& graph = method.graph;
  for (const Value& value : graph.values) {
    if (valid(value.alias_id)) {
      throw std::runtime_error(
          "vulkan: tensor aliases are not supported by this engine version");
    }
  }
  for (const ValueId id : graph.input_ids) {
    if (!graph.value(id).is_tensor()) {
      throw std::runtime_error("vulkan: user inputs must be tensors");
    }
  }
  for (const ValueId id : graph.output_ids) {
    if (!graph.value(id).is_tensor()) {
      throw std::runtime_error("vulkan: user outputs must be tensors");
    }
  }
  for (const DataBinding& binding : method.data_bindings) {
    if (binding.mutated || !binding.has_data) {
      throw std::runtime_error(
          "vulkan: mutable buffers are not supported by this engine version");
    }
  }
  for (const OutputSpec& spec : method.output_specs) {
    if (spec.kind != OutputKind::UserOutput) {
      throw std::runtime_error(
          "vulkan: mutation outputs are not supported by this engine version");
    }
  }
  for (const NodeId id : graph.schedule) {
    for (const NamedArgument& input : graph.node(id).inputs) {
      if (input.mutated) {
        throw std::runtime_error(
            "vulkan: mutating operators are not supported by this engine version");
      }
    }
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

// One region of a Method lowered onto an ET-VK ComputeGraph.
//
// Deliberately not in the header: EngineContext::compile hands this back
// through the base pointer, so nothing outside this file needs to name it.
class VulkanEngineExecutable final : public EngineExecutable {
 private:
  Method method_;
  VulkanConstantMaterializationTracker& materializations_;

  // The region this executable covers. Today this is the whole method. The
  // lowering helpers still inspect the full method graph, so narrower runtime
  // partitioning requires auditing those walks first.
  std::vector<NodeId> nodes_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;

  std::unique_ptr<ComputeGraph> graph_;
  // native ValueId -> ET-VK ValueRef, one slot per value in the method graph.
  std::vector<VkRef> vref_;
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
      const GraphConfig& config)
      : method_(method),
        materializations_(materializations),
        graph_(std::make_unique<ComputeGraph>(config)) {
    method_.graph.rebuild_def_use();
    vulkan::insert_prepack_nodes(method_);
    validate_graph(method_.graph);
    validate_supported_method(method_);
    nodes_ = method_.graph.schedule;
    inputs_ = method_.graph.input_ids;
    outputs_ = method_.graph.output_ids;
    build(package);
  }

  size_t num_inputs() const override {
    return inputs_.size();
  }

  size_t num_outputs() const override {
    return outputs_.size();
  }

  std::vector<int64_t> input_sizes(size_t i) const override {
    return meta_of(inputs_.at(i)).sizes;
  }

  std::vector<int64_t> output_sizes(size_t i) const override {
    return meta_of(outputs_.at(i)).sizes;
  }

  ScalarType input_dtype(size_t i) const override {
    return meta_of(inputs_.at(i)).dtype;
  }

  ScalarType output_dtype(size_t i) const override {
    return meta_of(outputs_.at(i)).dtype;
  }

  void set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype)
      override {
    const TensorMeta& meta = meta_of(inputs_.at(i));
    if (numel != logical_numel(meta)) {
      throw std::runtime_error(
          "vulkan: input element count does not match its shape");
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
    graph_->execute();
  }

  void get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype)
      override {
    if (numel != logical_numel(meta_of(outputs_.at(i)))) {
      throw std::runtime_error(
          "vulkan: output element count does not match its shape");
    }
    if (numel == 0) {
      return;
    }
    if (data == nullptr) {
      throw std::runtime_error("vulkan: output data is null");
    }
    graph_->maybe_cast_and_copy_from_staging(
        output_staging_.at(i), data, numel, to_vk_dtype(dst_dtype));
  }

 private:
  const Graph& g() const {
    return method_.graph;
  }

  const TensorMeta& meta_of(ValueId id) const {
    return g().value(id).tensor_meta();
  }

  // Everything is channels-packed except the operands of the width-packed ops,
  // whose producing values are created width-packed directly so only genuine
  // mid-graph transitions need a repack.
  void assign_layouts() {
    layout_.assign(g().values.size(), utils::kChannelsPacked);
    storage_.assign(g().values.size(), utils::kTexture3D);
    for (ValueId id = 0; id < static_cast<ValueId>(g().values.size()); ++id) {
      const Value& value = g().value(id);
      if (value.is_tensor() &&
          requires_buffer_storage(value.tensor_meta().dtype)) {
        storage_at(id) = utils::kBuffer;
        layout_at(id) = utils::kWidthPacked;
      }
    }
    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call() || !is_width_packed_op(registry_key(n.target))) {
        continue;
      }
      for (const Output& o : n.outputs) {
        if (o.kind == OutputValueKind::TensorList) {
          for (const ValueId e : o.elem_ids) {
            if (valid(e)) {
              layout_at(e) = utils::kWidthPacked;
            }
          }
        } else if (valid(o.value_id)) {
          layout_at(o.value_id) = utils::kWidthPacked;
        }
      }
    }
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
        return graph_->add_none();
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
        const vkapi::ScalarType dt = to_vk_dtype(device_dtype(m.dtype));
        const DataBinding* b = binding_at(native_id);
        if (b != nullptr && b->has_data) {
          return graph_->add_tensorref(
              sizes, dt, constant_buffer(package, *b, v));
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
      case ArgKind::IntList:
        return graph_->add_scalar_list(
            std::vector<int64_t>(arg.as_int_list().values));
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

  // Insert a same-shape view_copy that repacks a channels-packed activation
  // into a fresh width-packed tensor, and return the width-packed ref.
  VkRef repack_to_width(VkRef src, const Value& v) {
    const std::vector<int64_t>& sizes = v.tensor_meta().sizes;
    const VkRef dst = graph_->add_tensor(
        sizes,
        to_vk_dtype(device_dtype(v.tensor_meta().dtype)),
        utils::kTexture3D,
        utils::kWidthPacked);
    const VkRef size_list =
        graph_->add_scalar_list(std::vector<int64_t>(sizes));
    std::vector<VkRef> args = {src, size_list, dst};
    VK_GET_OP_FN("aten.view_copy.default")(*graph_, args);
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

  void dispatch(const Node& n) {
    const std::string key = registry_key(n.target);
    if (!VK_HAS_OP(key)) {
      throw std::runtime_error(
          "vulkan: unsupported op '" + key + "' (target " + n.target + ")");
    }
    const bool width = is_width_packed_op(key);

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
          if (activation && layout_at(value_id) != utils::kWidthPacked) {
            a = repack_to_width(a, v);
          }
        }
      }
      args.push_back(a);
    }

    // Outputs: exactly one tensor is appended directly; otherwise every output
    // ref is grouped into one ValueList, which is ET-VK's multi-return calling
    // convention.
    std::vector<VkRef> outs;
    for (const Output& o : n.outputs) {
      if (o.kind == OutputValueKind::TensorList) {
        std::transform(
            o.elem_ids.begin(),
            o.elem_ids.end(),
            std::back_inserter(outs),
            [this](ValueId e) { return vref_at(e); });
      } else if (valid(o.value_id)) {
        outs.push_back(vref_at(o.value_id));
      }
    }
    if (outs.size() == 1) {
      args.push_back(outs[0]);
    } else if (outs.size() > 1) {
      args.push_back(graph_->add_value_list(std::move(outs)));
    }

    VK_GET_OP_FN(key)(*graph_, args);
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
      if (binding != nullptr && binding->has_data) {
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

  void build(const Package& package) {
    assign_layouts();
    index_bindings();
    plan_allocations();

    // 1. Materialize every native value as an ET-VK value.
    none_ref_ = graph_->add_none();
    vref_.assign(g().values.size(), -1);
    for (ValueId i = 0; i < static_cast<ValueId>(g().values.size()); ++i) {
      try {
        vref_at(i) = make_value(package, g().values[i], i);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            "vulkan: value[" + std::to_string(i) + "] '" + g().values[i].name +
            "': " + e.what());
      }
    }

    // 2. Register the region's inputs, creating host->device staging.
    input_staging_.resize(inputs_.size());
    std::transform(
        inputs_.begin(),
        inputs_.end(),
        input_staging_.begin(),
        [this](ValueId in) { return graph_->set_input_tensor(vref_at(in)); });

    // 3. Dispatch each call node in schedule order.
    for (const NodeId nid : nodes_) {
      const Node& n = g().node(nid);
      if (!n.is_call()) {
        continue;
      }
      try {
        dispatch(n);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            "vulkan: node '" + n.name + "' (" + n.target + "): " + e.what());
      }
    }

    // 4. Register the region's outputs, creating device->host staging.
    output_staging_.resize(outputs_.size());
    std::transform(
        outputs_.begin(),
        outputs_.end(),
        output_staging_.begin(),
        [this](ValueId out) {
          return graph_->set_output_tensor(vref_at(out));
        });

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
    const GraphConfig& config) {
  return std::make_unique<VulkanEngineExecutable>(
      method, package, materializations, config);
}

} // namespace ptn
