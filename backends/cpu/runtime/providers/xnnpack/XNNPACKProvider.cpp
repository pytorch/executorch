// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/op_helpers/Convolution.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <xnnpack.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string_view>
#include <unordered_set>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;

Error check(xnn_status status, const char* operation) {
  if (status == xnn_status_success) {
    return Error::Ok;
  }
  ET_LOG(
      Error, "CPU XNNPACK %s failed: %d", operation, static_cast<int>(status));
  return status == xnn_status_out_of_memory ? Error::MemoryAllocationFailed
                                            : Error::NotSupported;
}

#define CPU_XNN_CHECK(expression)                                \
  do {                                                           \
    const auto cpu_xnn_error = check((expression), #expression); \
    if (cpu_xnn_error != Error::Ok) {                            \
      return cpu_xnn_error;                                      \
    }                                                            \
  } while (false)

bool constant(const ptn::Value& value) {
  return value.role == ptn::ValueRole::Parameter ||
      value.role == ptn::ValueRole::ConstantTensor ||
      value.role == ptn::ValueRole::Buffer;
}

const ptn::Argument& arg(const Kernel& node, size_t index) {
  return node.inputs.at(index).arg;
}

std::vector<ptn::ValueId> tensor_values(const Kernel& node) {
  auto values = node.input_value_ids();
  for (const auto& output : node.outputs) {
    if (output.kind == ptn::OutputValueKind::Tensor) {
      values.push_back(output.value_id);
    }
  }
  return values;
}

bool fp32(const Kernel& node, const ptn::Graph& graph) {
  const auto values = tensor_values(node);
  return std::all_of(values.begin(), values.end(), [&graph](auto id) {
    return graph.value(id).tensor_meta().dtype == ptn::ScalarType::Float;
  });
}

class XNNExecutable;

struct XNNOp {
  std::string_view target;
  bool (*supports)(const Kernel&, const ptn::Graph&);
  Error (XNNExecutable::*define)(const Kernel&);
};

const XNNOp* find_op(std::string_view target);

class XNNState {
 public:
  XNNState() = default;
  ~XNNState() {
    if (workspace) {
      xnn_release_workspace(workspace);
    }
    if (cache) {
      xnn_delete_weights_cache(cache);
    }
  }
  XNNState(const XNNState&) = delete;
  XNNState& operator=(const XNNState&) = delete;
  XNNState(XNNState&&) = delete;
  XNNState& operator=(XNNState&&) = delete;
  Error initialize() {
    if (workspace) {
      return Error::Ok;
    }
    CPU_XNN_CHECK(xnn_initialize(nullptr));
    CPU_XNN_CHECK(xnn_create_weights_cache(&cache));
    CPU_XNN_CHECK(xnn_create_workspace(&workspace));
    return Error::Ok;
  }
  std::map<std::pair<ptn::ValueId, bool>, Buffer> converted_weights;
  xnn_workspace_t workspace = nullptr;
  xnn_weights_cache_t cache = nullptr;
};

class XNNExecutable final : public Executable {
 public:
  XNNExecutable(PreparationContext context, XNNState& state)
      : context_(context), state_(state) {}
  ~XNNExecutable() override {
    if (runtime_) {
      xnn_delete_runtime(runtime_);
    }
    if (subgraph_) {
      xnn_delete_subgraph(subgraph_);
    }
  }
  XNNExecutable(const XNNExecutable&) = delete;
  XNNExecutable& operator=(const XNNExecutable&) = delete;
  XNNExecutable(XNNExecutable&&) = delete;
  XNNExecutable& operator=(XNNExecutable&&) = delete;

  Error initialize(const KernelRegion& region) {
    CPU_XNN_CHECK(xnn_create_subgraph(
        static_cast<uint32_t>(context_.graph.values.size()), 0, &subgraph_));
    std::unordered_set<ptn::ValueId> inputs(
        region.inputs.begin(), region.inputs.end());
    std::unordered_set<ptn::ValueId> outputs(
        region.outputs.begin(), region.outputs.end());
    std::unordered_set<ptn::ValueId> used;
    for (auto id : region.nodes) {
      for (auto value : context_.graph.node(id).input_value_ids()) {
        used.insert(value);
      }
      for (const auto& output : context_.graph.node(id).outputs) {
        used.insert(output.value_id);
      }
    }
    bindings_.reserve(inputs.size() + outputs.size());
    for (auto id : used) {
      const auto& value = context_.graph.value(id);
      const auto shape = dimensions(id);
      const void* data = nullptr;
      uint32_t flags = 0;
      if (constant(value)) {
        const auto& buffer = context_.buffers.at(id);
        ET_CHECK_OR_RETURN_ERROR(
            buffer.accepts(tensor_bytes(value).get(), false),
            InvalidArgument,
            "XNNPACK constant lacks alignment/readable tail: %s",
            value.name.c_str());
        data = buffer.data;
      } else {
        if (inputs.count(id)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        }
        if (outputs.count(id)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        }
      }
      uint32_t defined = XNN_INVALID_VALUE_ID;
      CPU_XNN_CHECK(xnn_define_tensor_value(
          subgraph_,
          xnn_datatype_fp32,
          shape.size(),
          shape.data(),
          data,
          id,
          flags,
          &defined));
      if (flags) {
        bindings_.push_back({id, flags});
      }
    }
    for (auto id : region.nodes) {
      const auto error = define(context_.graph.node(id));
      if (error != Error::Ok) {
        ET_LOG(
            Error,
            "XNNPACK semantic lowering failed: %s",
            context_.graph.node(id).target.c_str());
        return error;
      }
    }
    CPU_XNN_CHECK(xnn_create_runtime_v4(
        subgraph_,
        state_.cache,
        state_.workspace,
        context_.execution.threadpool,
        0,
        &runtime_));
    return Error::Ok;
  }

  Error reshape() override {
    CPU_XNN_CHECK(xnn_reshape_runtime(runtime_));
    for (const auto& binding : bindings_) {
      size_t rank = 0;
      std::array<size_t, XNN_MAX_TENSOR_DIMS> shape{};
      CPU_XNN_CHECK(xnn_get_external_value_shape(
          runtime_, binding.id, &rank, shape.data()));
      const auto expected = dimensions(binding.id);
      ET_CHECK_OR_RETURN_ERROR(
          rank == expected.size() &&
              std::equal(expected.begin(), expected.end(), shape.begin()),
          InvalidProgram,
          "XNNPACK inferred a different semantic shape for %s",
          context_.graph.value(binding.id).name.c_str());
    }
    return Error::Ok;
  }

  Error bind(const Buffer&) override {
    bound_ = false;
    external_.clear();
    external_.reserve(bindings_.size());
    for (const auto& binding : bindings_) {
      const auto& buffer = context_.buffers.at(binding.id);
      const bool output = binding.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
      ET_CHECK_OR_RETURN_ERROR(
          buffer.accepts(
              tensor_bytes(context_.graph.value(binding.id)).get(), output),
          InvalidArgument,
          "XNNPACK binding lacks required bounds/alignment: %s",
          context_.graph.value(binding.id).name.c_str());
      external_.push_back({static_cast<uint32_t>(binding.id), buffer.data});
    }
    bound_ = true;
    return Error::Ok;
  }

  Error run(const ExecutionContext&) override {
    ET_CHECK_OR_RETURN_ERROR(
        bound_, InvalidState, "XNNPACK requires a successful bind");
    // Shared workspace growth can relocate another region's internal values.
    CPU_XNN_CHECK(
        xnn_setup_runtime_v2(runtime_, external_.size(), external_.data()));
    CPU_XNN_CHECK(xnn_invoke_runtime(runtime_));
    return Error::Ok;
  }

 private:
  friend const XNNOp* find_op(std::string_view target);

  std::vector<size_t> dimensions(ptn::ValueId id) const {
    const auto& sizes = context_.graph.value(id).tensor_meta().sizes;
    return {sizes.begin(), sizes.end()};
  }

  Error tensor(
      const std::vector<size_t>& shape,
      uint32_t& id,
      const void* data = nullptr) {
    CPU_XNN_CHECK(xnn_define_tensor_value(
        subgraph_,
        xnn_datatype_fp32,
        shape.size(),
        shape.data(),
        data,
        XNN_INVALID_VALUE_ID,
        0,
        &id));
    return Error::Ok;
  }

  static bool supports_convolution(
      const Kernel& node,
      const ptn::Graph& graph) {
    const auto parsed = parse_convolution(node);
    if (!parsed || parsed->transposed || !fp32(node, graph)) {
      return false;
    }
    const auto& input = graph.value(parsed->input).tensor_meta().sizes;
    const auto& weight = graph.value(parsed->weight);
    const auto& filter = weight.tensor_meta().sizes;
    const auto groups = parsed->groups;
    if (input.size() != 4 || filter.size() != 4 || !constant(weight) ||
        groups <= 0 || input[1] != filter[1] * groups ||
        filter[0] % groups != 0) {
      return false;
    }
    const auto positive = [](const auto& list) {
      return list.size() == 2 &&
          std::all_of(list.begin(), list.end(), [](auto v) { return v >= 1; });
    };
    if (!positive(parsed->stride) || !positive(parsed->dilation) ||
        parsed->padding.size() != 2 ||
        std::any_of(
            parsed->padding.begin(),
            parsed->padding.end(),
            [](auto value) { return value < 0; }) ||
        parsed->output_padding != std::vector<int64_t>{0, 0}) {
      return false;
    }
    return parsed->bias == ptn::kInvalid || constant(graph.value(parsed->bias));
  }

  Error convolution(const Kernel& node) {
    const auto parsed = parse_convolution(node);
    ET_CHECK_OR_RETURN_ERROR(
        parsed, InvalidProgram, "XNNPACK convolution schema mismatch");
    const auto input = parsed->input;
    const auto weight = parsed->weight;
    const auto bias =
        parsed->bias == ptn::kInvalid ? XNN_INVALID_VALUE_ID : parsed->bias;
    const auto output = parsed->output;
    const auto in = dimensions(input);
    const auto out = dimensions(output);
    const auto filter = dimensions(weight);
    const auto& stride = parsed->stride;
    const auto& padding = parsed->padding;
    const auto& dilation = parsed->dilation;
    const size_t groups = parsed->groups;
    uint32_t nhwc_in, nhwc_out, packed_weight;
    auto error = tensor({in[0], in[2], in[3], in[1]}, nhwc_in);
    if (error != Error::Ok) {
      return error;
    }
    error = tensor({out[0], out[2], out[3], out[1]}, nhwc_out);
    if (error != Error::Ok) {
      return error;
    }
    constexpr std::array<size_t, 4> to_nhwc{0, 2, 3, 1};
    constexpr std::array<size_t, 4> to_nchw{0, 3, 1, 2};
    CPU_XNN_CHECK(xnn_define_static_transpose(
        subgraph_, 4, to_nhwc.data(), input, nhwc_in, 0));
    const bool depthwise =
        groups == in[1] && filter[1] == 1 && filter[0] == groups;
    const auto key = std::make_pair(weight, depthwise);
    const float* packed =
        static_cast<const float*>(context_.buffers.at(weight).data);
    const bool compatible = depthwise
        ? (filter[0] == 1 || filter[2] * filter[3] == 1)
        : (filter[1] == 1 || filter[2] * filter[3] == 1);
    if (!compatible) {
      const auto cached = state_.converted_weights.find(key);
      if (cached != state_.converted_weights.end()) {
        packed = static_cast<const float*>(cached->second.data);
      } else {
        auto storage = allocate_buffer(
            context_.allocator,
            tensor_bytes(context_.graph.value(weight)).get());
        if (!storage.ok()) {
          return storage.error();
        }
        context_.private_constant_bytes += storage->readable_bytes;
        auto* converted = static_cast<float*>(storage->data);
        const auto* source =
            static_cast<const float*>(context_.buffers.at(weight).data);
        for (size_t oc = 0; oc < filter[0]; ++oc) {
          for (size_t ic = 0; ic < filter[1]; ++ic) {
            for (size_t y = 0; y < filter[2]; ++y) {
              for (size_t x = 0; x < filter[3]; ++x) {
                const size_t from =
                    ((oc * filter[1] + ic) * filter[2] + y) * filter[3] + x;
                const size_t to = depthwise
                    ? (y * filter[3] + x) * filter[0] + oc
                    : ((oc * filter[2] + y) * filter[3] + x) * filter[1] + ic;
                converted[to] = source[from];
              }
            }
          }
        }
        state_.converted_weights.emplace(key, storage.get());
        packed = converted;
      }
    }
    error = tensor(
        depthwise
            ? std::vector<size_t>{1, filter[2], filter[3], filter[0]}
            : std::vector<size_t>{filter[0], filter[2], filter[3], filter[1]},
        packed_weight,
        packed);
    if (error != Error::Ok) {
      return error;
    }
    const auto infinity = std::numeric_limits<float>::infinity();
    if (depthwise) {
      CPU_XNN_CHECK(xnn_define_depthwise_convolution_2d(
          subgraph_,
          static_cast<uint32_t>(padding[0]),
          static_cast<uint32_t>(padding[1]),
          static_cast<uint32_t>(padding[0]),
          static_cast<uint32_t>(padding[1]),
          static_cast<uint32_t>(filter[2]),
          static_cast<uint32_t>(filter[3]),
          static_cast<uint32_t>(stride[0]),
          static_cast<uint32_t>(stride[1]),
          static_cast<uint32_t>(dilation[0]),
          static_cast<uint32_t>(dilation[1]),
          1,
          in[1],
          -infinity,
          infinity,
          nhwc_in,
          packed_weight,
          bias,
          nhwc_out,
          0));
    } else {
      CPU_XNN_CHECK(xnn_define_convolution_2d(
          subgraph_,
          static_cast<uint32_t>(padding[0]),
          static_cast<uint32_t>(padding[1]),
          static_cast<uint32_t>(padding[0]),
          static_cast<uint32_t>(padding[1]),
          static_cast<uint32_t>(filter[2]),
          static_cast<uint32_t>(filter[3]),
          static_cast<uint32_t>(stride[0]),
          static_cast<uint32_t>(stride[1]),
          static_cast<uint32_t>(dilation[0]),
          static_cast<uint32_t>(dilation[1]),
          static_cast<uint32_t>(groups),
          filter[1],
          filter[0] / groups,
          -infinity,
          infinity,
          nhwc_in,
          packed_weight,
          bias,
          nhwc_out,
          0));
    }
    CPU_XNN_CHECK(xnn_define_static_transpose(
        subgraph_, 4, to_nchw.data(), nhwc_out, output, 0));
    return Error::Ok;
  }

  static bool supports_layer_norm(const Kernel& node, const ptn::Graph& graph) {
    if (!fp32(node, graph) || node.inputs.size() != 5 ||
        node.outputs.size() != 3) {
      return false;
    }
    const auto& shape =
        graph.value(arg(node, 0).as_tensor().id).tensor_meta().sizes;
    const auto& normalized = arg(node, 1).as_int_list().values;
    return !normalized.empty() && normalized.size() <= shape.size() &&
        std::equal(normalized.rbegin(), normalized.rend(), shape.rbegin()) &&
        arg(node, 4).as_float().value >= 0 &&
        std::isfinite(arg(node, 4).as_float().value);
  }

  Error layer_norm(const Kernel& node) {
    const auto input = arg(node, 0).as_tensor().id;
    const auto output = node.outputs.at(0).value_id;
    const auto mean = node.outputs.at(1).value_id;
    const auto rstd = node.outputs.at(2).value_id;
    const auto shape = dimensions(input);
    const auto reduced_shape = dimensions(mean);
    const size_t normalized_rank = arg(node, 1).as_int_list().values.size();
    std::vector<int64_t> axes(normalized_rank);
    std::iota(axes.begin(), axes.end(), shape.size() - normalized_rank);
    uint32_t centered = XNN_INVALID_VALUE_ID;
    uint32_t squared = XNN_INVALID_VALUE_ID;
    uint32_t variance = XNN_INVALID_VALUE_ID;
    uint32_t shifted = XNN_INVALID_VALUE_ID;
    uint32_t normalized = XNN_INVALID_VALUE_ID;
    for (auto* id : {&centered, &squared, &normalized}) {
      const auto error = tensor(shape, *id);
      if (error != Error::Ok) {
        return error;
      }
    }
    for (auto* id : {&variance, &shifted}) {
      const auto error = tensor(reduced_shape, *id);
      if (error != Error::Ok) {
        return error;
      }
    }
    auto epsilon = allocate_buffer(context_.allocator, sizeof(float));
    if (!epsilon.ok()) {
      return epsilon.error();
    }
    *static_cast<float*>(epsilon->data) =
        static_cast<float>(arg(node, 4).as_float().value);
    context_.private_constant_bytes += epsilon->readable_bytes;
    uint32_t epsilon_id;
    const auto error = tensor({}, epsilon_id, epsilon->data);
    if (error != Error::Ok) {
      return error;
    }
    CPU_XNN_CHECK(xnn_define_static_reduce_v2(
        subgraph_,
        xnn_reduce_mean,
        axes.size(),
        axes.data(),
        input,
        mean,
        XNN_FLAG_KEEP_DIMS));
    CPU_XNN_CHECK(xnn_define_binary(
        subgraph_, xnn_binary_subtract, nullptr, input, mean, centered, 0));
    CPU_XNN_CHECK(xnn_define_unary(
        subgraph_, xnn_unary_square, nullptr, centered, squared, 0));
    CPU_XNN_CHECK(xnn_define_static_reduce_v2(
        subgraph_,
        xnn_reduce_mean,
        axes.size(),
        axes.data(),
        squared,
        variance,
        XNN_FLAG_KEEP_DIMS));
    CPU_XNN_CHECK(xnn_define_binary(
        subgraph_, xnn_binary_add, nullptr, variance, epsilon_id, shifted, 0));
    CPU_XNN_CHECK(xnn_define_unary(
        subgraph_,
        xnn_unary_reciprocal_square_root,
        nullptr,
        shifted,
        rstd,
        0));
    const bool weight = arg(node, 2).kind() == ptn::ArgKind::Tensor;
    const bool bias = arg(node, 3).kind() == ptn::ArgKind::Tensor;
    CPU_XNN_CHECK(xnn_define_binary(
        subgraph_,
        xnn_binary_multiply,
        nullptr,
        centered,
        rstd,
        weight || bias ? normalized : output,
        0));
    uint32_t scaled = normalized;
    if (weight) {
      if (bias) {
        const auto result = tensor(shape, scaled);
        if (result != Error::Ok) {
          return result;
        }
      } else {
        scaled = output;
      }
      CPU_XNN_CHECK(xnn_define_binary(
          subgraph_,
          xnn_binary_multiply,
          nullptr,
          normalized,
          arg(node, 2).as_tensor().id,
          scaled,
          0));
    }
    if (bias) {
      CPU_XNN_CHECK(xnn_define_binary(
          subgraph_,
          xnn_binary_add,
          nullptr,
          scaled,
          arg(node, 3).as_tensor().id,
          output,
          0));
    }
    return Error::Ok;
  }

  template <xnn_unary_operator Operator>
  Error unary(const Kernel& node) {
    CPU_XNN_CHECK(xnn_define_unary(
        subgraph_,
        Operator,
        nullptr,
        arg(node, 0).as_tensor().id,
        node.outputs.at(0).value_id,
        0));
    return Error::Ok;
  }

  static bool supports_linear(const Kernel& node, const ptn::Graph& graph) {
    if (!fp32(node, graph) || node.inputs.size() != 3 ||
        node.outputs.size() != 1) {
      return false;
    }
    const auto& weight = graph.value(arg(node, 1).as_tensor().id);
    const auto& filter = weight.tensor_meta().sizes;
    if (!constant(weight) || filter.size() != 2) {
      return false;
    }
    if (arg(node, 2).kind() == ptn::ArgKind::None) {
      return true;
    }
    const auto& bias = graph.value(arg(node, 2).as_tensor().id);
    const auto& shape = bias.tensor_meta().sizes;
    return constant(bias) &&
        (shape.empty() ||
         (shape.size() == 1 && (shape[0] == 1 || shape[0] == filter[0])));
  }

  static bool supports_softmax(const Kernel& node, const ptn::Graph& graph) {
    if (node.inputs.size() != 3 || node.outputs.size() != 1 ||
        node.outputs[0].kind != ptn::OutputValueKind::Tensor ||
        arg(node, 0).kind() != ptn::ArgKind::Tensor ||
        arg(node, 1).kind() != ptn::ArgKind::Int ||
        arg(node, 2).kind() != ptn::ArgKind::Bool || !fp32(node, graph)) {
      return false;
    }
    const auto& input =
        graph.value(arg(node, 0).as_tensor().id).tensor_meta().sizes;
    const auto& output =
        graph.value(node.outputs[0].value_id).tensor_meta().sizes;
    const auto dim = arg(node, 1).as_int();
    const auto half_to_float = arg(node, 2).as_bool();
    return !input.empty() && input == output && dim.id == ptn::kInvalid &&
        half_to_float.id == ptn::kInvalid && !half_to_float.value &&
        (dim.value == -1 ||
         dim.value == static_cast<int64_t>(input.size()) - 1);
  }

  Error softmax(const Kernel& node) {
    CPU_XNN_CHECK(xnn_define_softmax(
        subgraph_,
        arg(node, 0).as_tensor().id,
        node.outputs.at(0).value_id,
        0));
    return Error::Ok;
  }

  Error linear(const Kernel& node) {
    const auto weight = arg(node, 1).as_tensor().id;
    const size_t channels = dimensions(weight)[0];
    uint32_t bias = arg(node, 2).kind() == ptn::ArgKind::None
        ? XNN_INVALID_VALUE_ID
        : arg(node, 2).as_tensor().id;
    if (bias != XNN_INVALID_VALUE_ID &&
        dimensions(bias) != std::vector<size_t>{channels}) {
      auto broadcast =
          allocate_buffer(context_.allocator, channels * sizeof(float));
      if (!broadcast.ok()) {
        return broadcast.error();
      }
      std::fill_n(
          static_cast<float*>(broadcast->data),
          channels,
          *static_cast<const float*>(context_.buffers.at(bias).data));
      context_.private_constant_bytes += broadcast->readable_bytes;
      const auto error = tensor({channels}, bias, broadcast->data);
      if (error != Error::Ok) {
        return error;
      }
    }
    const float infinity = std::numeric_limits<float>::infinity();
    CPU_XNN_CHECK(xnn_define_fully_connected(
        subgraph_,
        -infinity,
        infinity,
        arg(node, 0).as_tensor().id,
        weight,
        bias,
        node.outputs.at(0).value_id,
        0));
    return Error::Ok;
  }

  static bool supports_add(const Kernel& node, const ptn::Graph& graph) {
    return fp32(node, graph) && node.inputs.size() == 3 &&
        node.outputs.size() == 1 &&
        ((arg(node, 2).kind() == ptn::ArgKind::Int &&
          arg(node, 2).as_int().value == 1) ||
         (arg(node, 2).kind() == ptn::ArgKind::Float &&
          arg(node, 2).as_float().value == 1));
  }

  static bool supports_mul(const Kernel& node, const ptn::Graph& graph) {
    return fp32(node, graph) && node.inputs.size() == 2 &&
        node.outputs.size() == 1;
  }

  template <xnn_binary_operator Operator>
  Error binary(const Kernel& node) {
    CPU_XNN_CHECK(xnn_define_binary(
        subgraph_,
        Operator,
        nullptr,
        arg(node, 0).as_tensor().id,
        arg(node, 1).as_tensor().id,
        node.outputs.at(0).value_id,
        0));
    return Error::Ok;
  }

  static bool supports_gelu(const Kernel& node, const ptn::Graph& graph) {
    return fp32(node, graph) && node.inputs.size() == 2 &&
        node.outputs.size() == 1 && arg(node, 1).as_string().value == "none";
  }

  static bool supports_mean(const Kernel& node, const ptn::Graph& graph) {
    return fp32(node, graph) && node.inputs.size() == 4 &&
        node.outputs.size() == 1 &&
        arg(node, 1).kind() == ptn::ArgKind::IntList &&
        !arg(node, 1).as_int_list().values.empty() &&
        (arg(node, 3).kind() == ptn::ArgKind::None ||
         arg(node, 3).as_scalar_type().value == ptn::ScalarType::Float);
  }

  Error mean(const Kernel& node) {
    if (dimensions(arg(node, 0).as_tensor().id).empty()) {
      return static_reshape(node);
    }
    const auto& axes = arg(node, 1).as_int_list().values;
    CPU_XNN_CHECK(xnn_define_static_reduce_v2(
        subgraph_,
        xnn_reduce_mean,
        axes.size(),
        axes.data(),
        arg(node, 0).as_tensor().id,
        node.outputs.at(0).value_id,
        arg(node, 2).as_bool().value ? XNN_FLAG_KEEP_DIMS : 0));
    return Error::Ok;
  }

  static bool supports_shape_copy(const Kernel& node, const ptn::Graph& graph) {
    return fp32(node, graph) && node.inputs.size() == 2 &&
        node.outputs.size() == 1;
  }

  Error permute(const Kernel& node) {
    const auto& order = arg(node, 1).as_int_list().values;
    if (order.empty()) {
      return static_reshape(node);
    }
    std::vector<size_t> permutation(order.size());
    std::transform(
        order.begin(), order.end(), permutation.begin(), [&](auto dimension) {
          return dimension < 0 ? dimension + order.size() : dimension;
        });
    CPU_XNN_CHECK(xnn_define_static_transpose(
        subgraph_,
        permutation.size(),
        permutation.data(),
        arg(node, 0).as_tensor().id,
        node.outputs.at(0).value_id,
        0));
    return Error::Ok;
  }

  static bool supports_as_strided(const Kernel& node, const ptn::Graph& graph) {
    if (!fp32(node, graph) || node.inputs.size() != 4 ||
        node.outputs.size() != 1 ||
        (arg(node, 3).kind() != ptn::ArgKind::None &&
         arg(node, 3).as_int().value != 0)) {
      return false;
    }
    const auto& shape = arg(node, 1).as_int_list().values;
    const auto& strides = arg(node, 2).as_int_list().values;
    if (shape.size() > XNN_MAX_TENSOR_DIMS || shape.size() != strides.size() ||
        shape != graph.value(node.outputs[0].value_id).tensor_meta().sizes) {
      return false;
    }
    int64_t expected_stride = 1;
    for (size_t reverse = shape.size(); reverse > 0; --reverse) {
      const auto dimension = reverse - 1;
      if (shape[dimension] <= 0 || shape[dimension] > INT32_MAX ||
          expected_stride > INT32_MAX / shape[dimension] ||
          (shape[dimension] > 1 && strides[dimension] != expected_stride)) {
        return false;
      }
      expected_stride *= shape[dimension];
    }
    int64_t input_elements = 1;
    for (auto size :
         graph.value(arg(node, 0).as_tensor().id).tensor_meta().sizes) {
      if (size > INT32_MAX || input_elements > INT32_MAX / size) {
        return false;
      }
      input_elements *= size;
    }
    return input_elements == expected_stride;
  }

  Error static_reshape(const Kernel& node) {
    const auto output = node.outputs.at(0).value_id;
    const auto shape = dimensions(output);
    CPU_XNN_CHECK(xnn_define_static_reshape(
        subgraph_,
        shape.size(),
        shape.data(),
        arg(node, 0).as_tensor().id,
        output,
        0));
    return Error::Ok;
  }

  Error define(const Kernel& node) {
    const auto* op = find_op(node.target);
    return op ? (this->*op->define)(node) : Error::NotSupported;
  }

  struct Binding {
    ptn::ValueId id;
    uint32_t flags;
  };
  PreparationContext context_;
  XNNState& state_;
  xnn_subgraph_t subgraph_ = nullptr;
  xnn_runtime_t runtime_ = nullptr;
  bool bound_ = false;
  std::vector<Binding> bindings_;
  std::vector<xnn_external_value> external_;
};

const XNNOp* find_op(std::string_view target) {
  using E = XNNExecutable;
  static constexpr std::array<XNNOp, 15> kOps{{
      {"torch.ops.aten.add.Tensor",
       &E::supports_add,
       &E::binary<xnn_binary_add>},
      {"torch.ops.aten.as_strided_copy.default",
       &E::supports_as_strided,
       &E::static_reshape},
      {"torch.ops.aten._softmax.default", &E::supports_softmax, &E::softmax},
      {"torch.ops.aten.convolution.default",
       &E::supports_convolution,
       &E::convolution},
      {"torch.ops.aten.gelu.default",
       &E::supports_gelu,
       &E::unary<xnn_unary_gelu>},
      {"torch.ops.aten.linear.default", &E::supports_linear, &E::linear},
      {"torch.ops.aten.mean.dim", &E::supports_mean, &E::mean},
      {"torch.ops.aten.mul.Tensor",
       &E::supports_mul,
       &E::binary<xnn_binary_multiply>},
      {"torch.ops.aten.native_layer_norm.default",
       &E::supports_layer_norm,
       &E::layer_norm},
      {"torch.ops.aten.permute_copy.default",
       &E::supports_shape_copy,
       &E::permute},
      {"torch.ops.aten.view_copy.default",
       &E::supports_shape_copy,
       &E::static_reshape},
  }};
  const auto found =
      std::find_if(kOps.begin(), kOps.end(), [&](const auto& candidate) {
        return candidate.target == target;
      });
  return found == kOps.end() ? nullptr : &*found;
}

class XNNImplementation final : public KernelImplementation {
 public:
  explicit XNNImplementation(XNNState& state) : state_(state) {}
  std::string_view name() const override {
    return "subgraph";
  }
  int baseline_priority() const override {
    return 100;
  }
  bool accepts_regions() const override {
    return true;
  }

  Support supports(
      const Kernel& node,
      const ptn::Graph& graph,
      const ExecutionContext&) const override {
    const bool supported = supports_node(node, graph);
    return {
        supported,
        supported
            ? "XNNPACK FP32 route"
            : "unsupported XNNPACK schema, attributes or tensor metadata"};
  }

  bool supports_node(const Kernel& node, const ptn::Graph& graph) const {
    const auto* op = find_op(node.target);
    if (!op) {
      return false;
    }
    const auto values = tensor_values(node);
    return std::all_of(
               values.begin(),
               values.end(),
               [&graph](auto id) {
                 const auto& value = graph.value(id);
                 if (!value.is_tensor()) {
                   return false;
                 }
                 const auto& meta = value.tensor_meta();
                 return meta.is_contiguous() &&
                     meta.sizes.size() <= XNN_MAX_TENSOR_DIMS &&
                     std::all_of(
                            meta.sizes.begin(),
                            meta.sizes.end(),
                            [](auto size) { return size > 0; });
               }) &&
        op->supports(node, graph);
  }

  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion& region,
      PreparationContext& context) override {
    auto error = state_.initialize();
    if (error != Error::Ok) {
      return error;
    }
    auto executable = std::make_unique<XNNExecutable>(context, state_);
    error = executable->initialize(region);
    if (error != Error::Ok) {
      return error;
    }
    return std::unique_ptr<Executable>(std::move(executable));
  }

 private:
  XNNState& state_;
};

class XNNProvider final : public KernelProvider {
 public:
  XNNProvider() : implementation_(state_) {}
  std::string_view name() const override {
    return "XNNPACK";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&implementation_};
  }
  Error finish() override {
    if (state_.cache) {
      CPU_XNN_CHECK(xnn_finalize_weights_cache(
          state_.cache, xnn_weights_cache_finalization_kind_hard));
    }
    return Error::Ok;
  }

 private:
  XNNState state_;
  XNNImplementation implementation_;
};
} // namespace
// Referenced by generated provider registries and provider tests.
// cppcheck-suppress unusedFunction
std::unique_ptr<KernelProvider> create_xnnpack_provider() {
  return std::make_unique<XNNProvider>();
}
} // namespace executorch::backends::cpu
