// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/cpu/CPUEngineExecutable.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace ptn {
namespace {
namespace cpu = executorch::backends::cpu;

void check_error(executorch::runtime::Error error, const char* operation) {
  if (error != executorch::runtime::Error::Ok) {
    throw std::runtime_error(
        std::string("cpu: ") + operation + " failed (error " +
        std::to_string(static_cast<uint32_t>(error)) + ")");
  }
}

void validate_supported_method(const Method& method) {
  const auto& graph = method.graph;
  validate_graph(graph);
  if (!graph.subgraphs.empty()) {
    throw std::runtime_error("cpu: control flow is not supported");
  }
  for (const auto& value : graph.values) {
    if (valid(value.alias_id)) {
      throw std::runtime_error("cpu: tensor aliases are not supported");
    }
    const auto bytes = cpu::tensor_bytes(value);
    check_error(bytes.error(), "tensor validation");
    const auto& meta = value.tensor_meta();
    if (meta.quant) {
      throw std::runtime_error(
          "cpu: quantized tensor metadata is not supported");
    }
    if (!meta.lower_bounds.empty() && meta.lower_bounds != meta.sizes) {
      throw std::runtime_error("cpu: dynamic shapes are not supported");
    }
  }
  for (const auto& binding : method.data_bindings) {
    if (binding.mutated || !binding.has_data) {
      throw std::runtime_error("cpu: mutable buffers are not supported");
    }
  }
  for (const auto& spec : method.output_specs) {
    if (spec.kind != OutputKind::UserOutput) {
      throw std::runtime_error("cpu: mutation outputs are not supported");
    }
  }
  for (const auto id : graph.schedule) {
    for (const auto& input : graph.node(id).inputs) {
      if (input.mutated) {
        throw std::runtime_error("cpu: mutating operators are not supported");
      }
    }
  }
}

bool same_layout(const TensorMeta& left, const TensorMeta& right) {
  return left.dtype == right.dtype && left.sizes == right.sizes &&
      (left.dim_order_hint == right.dim_order_hint ||
       (left.is_contiguous() && right.is_contiguous()));
}

bool contiguous_reshape(const Node& node, const Graph& graph) {
  if (node.inputs.size() != 4 ||
      node.inputs[1].arg.kind() != ArgKind::IntList ||
      node.inputs[2].arg.kind() != ArgKind::IntList ||
      (node.inputs[3].arg.kind() != ArgKind::None &&
       (node.inputs[3].arg.kind() != ArgKind::Int ||
        node.inputs[3].arg.as_int().value != 0))) {
    return false;
  }
  const auto& source = graph.value(node.inputs[0].arg.as_tensor().id);
  if (valid(source.alias_id)) {
    return false;
  }
  const auto& meta = source.tensor_meta();
  int32_t previous = -1;
  for (const auto dim : meta.dim_order_hint) {
    if (meta.sizes[dim] > 1) {
      if (dim < previous) {
        return false;
      }
      previous = dim;
    }
  }
  const auto& sizes = node.inputs[1].arg.as_int_list().values;
  const auto& strides = node.inputs[2].arg.as_int_list().values;
  if (sizes != graph.value(node.output_value_ids()[0]).tensor_meta().sizes ||
      sizes.size() != strides.size()) {
    return false;
  }
  int64_t extent = 1;
  for (size_t reverse = sizes.size(); reverse > 0; --reverse) {
    const auto dim = reverse - 1;
    if (sizes[dim] <= 0 || extent > INT64_MAX / sizes[dim] ||
        (sizes[dim] > 1 && strides[dim] != extent)) {
      return false;
    }
    extent *= sizes[dim];
  }
  return extent == meta.numel();
}

void redirect_mutated_aliases(
    Graph& graph,
    NodeId id,
    ValueId source,
    ValueId output,
    const std::vector<ValueId>& roots,
    const std::vector<size_t>& position) {
  for (ValueId alias = 0; alias < graph.values.size(); ++alias) {
    const auto& value = graph.value(alias);
    if (roots[alias] != roots[source] || !valid(value.producer_id) ||
        position[value.producer_id] >= position[id]) {
      continue;
    }
    const auto consumers = value.consumer_ids;
    const bool live = std::any_of(
                          consumers.begin(),
                          consumers.end(),
                          [&](const auto consumer) {
                            return position[consumer] > position[id];
                          }) ||
        std::find(graph.output_ids.begin(), graph.output_ids.end(), alias) !=
            graph.output_ids.end();
    if (!live) {
      continue;
    }
    if (!same_layout(value.tensor_meta(), graph.value(output).tensor_meta())) {
      throw std::runtime_error(
          "cpu: mutation through a live view is not supported");
    }
    for (const auto consumer : consumers) {
      if (position[consumer] > position[id]) {
        graph.replace_input(consumer, alias, output);
      }
    }
    std::replace(
        graph.output_ids.begin(), graph.output_ids.end(), alias, output);
  }
}

// Native preprocessing preserves views and may reinplace activations. CPUPlan
// consumes functional, contiguous values, so materialize them at engine compile
// time. A mutation can be functionalized only when every live alias can be
// redirected to the new value without changing its logical indexing.
void lower_native_method(Method& method) {
  auto& graph = method.graph;
  validate_graph(graph);
  std::vector<size_t> position(graph.nodes.size());
  for (size_t index = 0; index < graph.schedule.size(); ++index) {
    position[graph.schedule[index]] = index;
  }
  std::vector<ValueId> roots(graph.values.size());
  for (ValueId id = 0; id < graph.values.size(); ++id) {
    auto root = id;
    while (valid(graph.value(root).alias_id)) {
      root = graph.value(root).alias_id;
    }
    roots[id] = root;
  }
  for (const auto id : graph.schedule) {
    auto& node = graph.node(id);
    if (!node.is_call()) {
      continue;
    }
    const bool mutating = std::any_of(
        node.inputs.begin(), node.inputs.end(), [](const auto& input) {
          return input.mutated;
        });
    const auto outputs = node.output_value_ids();
    const bool aliases =
        std::any_of(outputs.begin(), outputs.end(), [&](const auto output) {
          return valid(graph.value(output).alias_id);
        });
    if (!mutating && !aliases) {
      continue;
    }
    if (node.inputs.empty() || node.inputs[0].arg.kind() != ArgKind::Tensor ||
        outputs.size() != 1 || !graph.value(outputs[0]).is_tensor()) {
      throw std::runtime_error("cpu: unsupported Native alias or mutation");
    }
    const auto source = node.inputs[0].arg.as_tensor().id;
    const auto output = outputs[0];
    if (roots[source] != roots[output]) {
      throw std::runtime_error("cpu: Native alias does not match its operand");
    }
    if (mutating) {
      if ((node.target != "torch.ops.aten.relu_.default" &&
           node.target != "torch.ops.aten.gelu_.default" &&
           node.target != "torch.ops.aten.sigmoid_.default") ||
          !node.inputs[0].mutated ||
          std::any_of(
              node.inputs.begin() + 1,
              node.inputs.end(),
              [](const auto& input) { return input.mutated; }) ||
          graph.value(roots[source]).role != ValueRole::Intermediate ||
          !same_layout(
              graph.value(source).tensor_meta(),
              graph.value(output).tensor_meta())) {
        throw std::runtime_error("cpu: unsupported Native mutation");
      }
      redirect_mutated_aliases(graph, id, source, output, roots, position);
      node.inputs[0].mutated = false;
      node.target.erase(node.target.find("_.default"), 1);
    } else if (node.target == "torch.ops.aten.as_strided.default") {
      if (!contiguous_reshape(node, graph)) {
        throw std::runtime_error(
            "cpu: Native as_strided requires a contiguous full-span reshape");
      }
      node.target = "torch.ops.aten.as_strided_copy.default";
    } else {
      constexpr std::pair<std::string_view, std::string_view> views[] = {
          {"torch.ops.aten.view.default", "torch.ops.aten.view_copy.default"},
          {"torch.ops.aten.permute.default",
           "torch.ops.aten.permute_copy.default"},
          {"torch.ops.aten.transpose.int",
           "torch.ops.aten.transpose_copy.int"}};
      const auto found = std::find_if(
          std::begin(views), std::end(views), [&](const auto& view) {
            return node.target == view.first;
          });
      if (found == std::end(views)) {
        throw std::runtime_error("cpu: unsupported Native view " + node.target);
      }
      node.target = found->second;
    }
  }
  for (ValueId id = 0; id < graph.values.size(); ++id) {
    auto& value = graph.value(id);
    value.alias_id = kInvalid;
    if (value.is_tensor() && value.role == ValueRole::Intermediate &&
        std::find(graph.output_ids.begin(), graph.output_ids.end(), id) ==
            graph.output_ids.end()) {
      value.tensor_meta().dim_order_hint.clear();
    }
  }
  graph.rebuild_def_use();
}

template <ScalarType Type>
using HostType = std::conditional_t<
    Type == ScalarType::Half,
    executorch::aten::Half,
    std::conditional_t<
        Type == ScalarType::BFloat16,
        executorch::aten::BFloat16,
        cpp_type_t<Type>>>;

template <typename Function>
void visit_dtype(ScalarType dtype, Function&& function) {
  switch (dtype) {
#define CPU_VISIT_TYPE(cpp_type, name, id)                      \
  case ScalarType::name:                                        \
    function.template operator()<HostType<ScalarType::name>>(); \
    return;
    PTN_FORALL_SCALAR_TYPES(CPU_VISIT_TYPE)
#undef CPU_VISIT_TYPE
  }
  throw std::runtime_error("cpu: unsupported host dtype");
}

template <typename Destination, typename Source>
Destination convert_element(Source value) {
  if constexpr (
      std::is_integral_v<Destination> && !std::is_same_v<Destination, bool>) {
    if constexpr (std::is_integral_v<Source>) {
      if constexpr (!std::is_same_v<Source, bool>) {
        if (!std::in_range<Destination>(value)) {
          throw std::runtime_error("cpu: integer conversion is out of range");
        }
      }
    } else {
      const double number = static_cast<double>(value);
      const double upper =
          std::ldexp(1.0, std::numeric_limits<Destination>::digits);
      const double lower = std::is_signed_v<Destination> ? -upper : 0;
      if (!std::isfinite(number) || number < lower || number >= upper) {
        throw std::runtime_error(
            "cpu: floating-point conversion is out of range");
      }
    }
  }
  return executorch::runtime::convert<Destination, Source>(value);
}

// Host pointers need not have the alignment or readable tail required by a
// provider. Copy only their logical span, using memcpy for scalar loads too.
void copy_tensor(
    void* destination,
    ScalarType destination_dtype,
    const void* source,
    ScalarType source_dtype,
    size_t numel) {
  const auto source_size = element_size(source_dtype);
  const auto destination_size = element_size(destination_dtype);
  if (numel > SIZE_MAX / std::max(source_size, destination_size)) {
    throw std::runtime_error("cpu: host tensor byte size overflows");
  }
  if (numel == 0) {
    return;
  }
  if (source_dtype == destination_dtype) {
    std::memcpy(destination, source, numel * source_size);
    return;
  }
  visit_dtype(destination_dtype, [&]<typename Destination>() {
    visit_dtype(source_dtype, [&]<typename Source>() {
      const auto* input = static_cast<const uint8_t*>(source);
      auto* output = static_cast<uint8_t*>(destination);
      for (size_t index = 0; index < numel; ++index) {
        Source value;
        std::memcpy(&value, input + index * sizeof(Source), sizeof(Source));
        const auto converted = convert_element<Destination>(value);
        std::memcpy(
            output + index * sizeof(Destination),
            &converted,
            sizeof(Destination));
      }
    });
  });
}

class CPUEngineExecutable final : public EngineExecutable {
 public:
  CPUEngineExecutable(
      const Method& method,
      const Package& package,
      CPUConstantStore& constants,
      cpu::RuntimeConfiguration configuration,
      const cpu::ExecutionContext& execution)
      : method_(method),
        buffers_(method.graph.values.size()),
        execution_(execution) {
    lower_native_method(method_);
    validate_supported_method(method_);
    plan_ = std::make_unique<cpu::CPUPlan>(
        method_.graph,
        buffers_,
        std::move(configuration),
        execution_);
    check_error(plan_->select(), "provider selection");
    for (const auto& binding : method_.data_bindings) {
      buffers_.at(binding.value_id) = constants.acquire(
          package, binding.key, plan_->requirements().at(binding.value_id));
    }
    check_error(plan_->prepare(allocator_), "plan preparation");
    input_staging_.reserve(num_inputs());
    input_ready_.resize(num_inputs(), false);
    for (const auto id : method_.graph.input_ids) {
      auto staging = cpu::allocate_buffer(
          allocator_,
          static_cast<size_t>(method_.graph.value(id).tensor_meta().numel()) *
              element_size(method_.graph.value(id).tensor_meta().dtype));
      check_error(staging.error(), "input staging allocation");
      input_staging_.push_back(staging.get());
    }
  }

  size_t num_inputs() const override {
    return method_.graph.input_ids.size();
  }
  size_t num_outputs() const override {
    return method_.graph.output_ids.size();
  }

  std::vector<int64_t> input_sizes(size_t index) const override {
    return input_meta(index).sizes;
  }
  std::vector<int64_t> output_sizes(size_t index) const override {
    return output_meta(index).sizes;
  }
  ScalarType input_dtype(size_t index) const override {
    return input_meta(index).dtype;
  }
  ScalarType output_dtype(size_t index) const override {
    return output_meta(index).dtype;
  }

  void set_input(size_t index, const void* data, size_t numel, ScalarType dtype)
      override {
    const auto& meta = input_meta(index);
    validate_io(meta, data, numel);
    input_ready_[index] = false;
    output_ready_ = false;
    copy_tensor(input_staging_[index].data, meta.dtype, data, dtype, numel);
    input_ready_[index] = true;
  }

  void execute() override {
    output_ready_ = false;
    if (std::find(input_ready_.begin(), input_ready_.end(), false) !=
        input_ready_.end()) {
      throw std::runtime_error("cpu: every input must be set before execution");
    }
    for (size_t index = 0; index < num_inputs(); ++index) {
      const auto id = method_.graph.input_ids[index];
      const auto& meta = input_meta(index);
      const auto bytes =
          static_cast<size_t>(meta.numel()) * element_size(meta.dtype);
      if (bytes) {
        std::memcpy(buffers_.at(id).data, input_staging_[index].data, bytes);
      }
    }
    check_error(plan_->execute(execution_), "execution");
    output_ready_ = true;
  }

  void get_output(size_t index, void* data, size_t numel, ScalarType dtype)
      override {
    const auto& meta = output_meta(index);
    validate_io(meta, data, numel);
    if (!output_ready_) {
      throw std::runtime_error(
          "cpu: output is unavailable before successful execution");
    }
    copy_tensor(
        data,
        dtype,
        buffers_.at(method_.graph.output_ids[index]).data,
        meta.dtype,
        numel);
  }

 private:
  const TensorMeta& input_meta(size_t index) const {
    if (index >= num_inputs()) {
      throw std::runtime_error("cpu: input index out of range");
    }
    return method_.graph.value(method_.graph.input_ids[index]).tensor_meta();
  }
  const TensorMeta& output_meta(size_t index) const {
    if (index >= num_outputs()) {
      throw std::runtime_error("cpu: output index out of range");
    }
    return method_.graph.value(method_.graph.output_ids[index]).tensor_meta();
  }
  static void
  validate_io(const TensorMeta& meta, const void* data, size_t numel) {
    if (numel != static_cast<size_t>(meta.numel()) ||
        (numel != 0 && data == nullptr)) {
      throw std::runtime_error(
          "cpu: host tensor element count or data pointer is invalid");
    }
  }

  // Destruction order keeps graph, constants (context-owned), and allocations
  // alive through CPUPlan and provider executable destruction.
  Method method_;
  executorch::extension::MallocMemoryAllocator allocator_;
  std::vector<cpu::Buffer> buffers_;
  std::vector<cpu::Buffer> input_staging_;
  std::vector<bool> input_ready_;
  cpu::ExecutionContext execution_;
  bool output_ready_ = false;
  std::unique_ptr<cpu::CPUPlan> plan_;
};

} // namespace

std::unique_ptr<EngineExecutable> create_cpu_engine_executable(
    const Method& method,
    const Package& package,
    CPUConstantStore& constants,
    cpu::RuntimeConfiguration configuration,
    const cpu::ExecutionContext& execution) {
  return std::make_unique<CPUEngineExecutable>(
      method, package, constants, std::move(configuration), execution);
}

} // namespace ptn
