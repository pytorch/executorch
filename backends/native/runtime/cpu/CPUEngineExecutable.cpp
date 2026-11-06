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
#include <type_traits>
#include <utility>

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/utils/Aliases.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

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
  if (!method.graph.subgraphs.empty()) {
    throw std::runtime_error("cpu: control flow is not supported");
  }
  for (const auto& value : method.graph.values) {
    if (!value.is_tensor()) {
      continue;
    }
    const auto& meta = value.tensor_meta();
    if (meta.quant) {
      throw std::runtime_error(
          "cpu: quantized tensor metadata is not supported");
    }
    if (!meta.lower_bounds.empty() && meta.lower_bounds != meta.sizes) {
      throw std::runtime_error("cpu: dynamic shapes are not supported");
    }
  }
  if (std::any_of(
          method.data_bindings.begin(),
          method.data_bindings.end(),
          [](const auto& binding) {
            return binding.mutated || !binding.has_data;
          })) {
    throw std::runtime_error("cpu: mutable buffers are not supported");
  }
  if (std::any_of(
          method.output_specs.begin(),
          method.output_specs.end(),
          [](const auto& spec) {
            return spec.kind != OutputKind::UserOutput;
          })) {
    throw std::runtime_error("cpu: mutation outputs are not supported");
  }
}

[[noreturn]] void fail_matmul(const char* reason) {
  throw std::runtime_error(std::string("cpu: matmul ") + reason);
}

void lower_matmul(Graph& graph, NodeId id) {
  const auto node = graph.node(id);
  if (node.inputs.size() != 2 || node.outputs.size() != 1 ||
      node.outputs[0].kind != OutputValueKind::Tensor ||
      node.inputs[0].arg.kind() != ArgKind::Tensor ||
      node.inputs[1].arg.kind() != ArgKind::Tensor) {
    fail_matmul("requires two tensor inputs and one tensor output");
  }
  const auto left_id = node.inputs[0].arg.as_tensor().id;
  const auto right_id = node.inputs[1].arg.as_tensor().id;
  const auto& left_value = graph.value(left_id);
  const auto& right_value = graph.value(right_id);
  if (!left_value.is_tensor() || !right_value.is_tensor() ||
      left_value.tensor_meta().dtype != ScalarType::Float ||
      right_value.tensor_meta().dtype != ScalarType::Float) {
    fail_matmul("requires FP32 operands");
  }
  const auto left = left_value.tensor_meta().sizes;
  const auto right = right_value.tensor_meta().sizes;
  if (left.size() < 2 || right.size() < 2 ||
      std::ranges::any_of(left, [](int64_t size) { return size <= 0; }) ||
      std::ranges::any_of(right, [](int64_t size) { return size <= 0; })) {
    fail_matmul("requires positive matrix shapes of rank at least two");
  }
  const size_t rank = std::max(left.size(), right.size());
  std::vector<int64_t> expanded_left(rank, 1), expanded_right(rank, 1);
  std::copy(left.begin(), left.end(), expanded_left.end() - left.size());
  std::copy(right.begin(), right.end(), expanded_right.end() - right.size());
  if (expanded_left[rank - 1] != expanded_right[rank - 2]) {
    fail_matmul("has incompatible contraction dimensions");
  }
  std::vector<int64_t> output_shape(rank);
  int64_t batches = 1;
  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    const auto a = expanded_left[dim];
    const auto b = expanded_right[dim];
    if (a != b && a != 1 && b != 1) {
      fail_matmul("has incompatible broadcast dimensions");
    }
    const auto extent = std::max(a, b);
    if (batches > std::numeric_limits<int64_t>::max() / extent) {
      fail_matmul("has an overflowing batch extent");
    }
    batches *= extent;
    output_shape[dim] = extent;
    expanded_left[dim] = extent;
    expanded_right[dim] = extent;
  }
  const auto rows = left[left.size() - 2];
  const auto inner = left.back();
  const auto columns = right.back();
  output_shape[rank - 2] = rows;
  output_shape[rank - 1] = columns;
  const auto& output = graph.value(node.outputs[0].value_id);
  if (!output.is_tensor() || output.tensor_meta().dtype != ScalarType::Float ||
      output.tensor_meta().sizes != output_shape) {
    fail_matmul("has incompatible output metadata");
  }
  if (rank == 2 ||
      (rank == 3 && left.size() == 3 && right.size() == 3 &&
       left[0] == right[0])) {
    auto& replacement = graph.node(id);
    replacement.target =
        rank == 2 ? "torch.ops.aten.mm.default" : "torch.ops.aten.bmm.default";
    replacement.inputs[1].name = "mat2";
    return;
  }
  const auto tensor = [](ValueId value) -> NamedArgument {
    return {.name = "self", .arg = TensorArg{value}};
  };
  const auto call = [&](const char* suffix,
                        const char* target,
                        std::vector<NamedArgument> arguments,
                        std::vector<int64_t> shape) {
    const std::string name = node.name + "__cpu_matmul_" + suffix;
    const auto value =
        graph.append_value(Value(name, ScalarType::Float, std::move(shape)));
    graph.insert_node_before(
        id,
        Node{
            .name = name,
            .op_kind = OpKind::CallFunction,
            .target = target,
            .inputs = std::move(arguments),
            .outputs =
                {{.kind = OutputValueKind::Tensor,
                  .value_id = value,
                  .elem_ids = {}}},
            .attrs = {},
        });
    return value;
  };
  const auto expand = [&](const char* suffix,
                          ValueId source,
                          const std::vector<int64_t>& original,
                          const std::vector<int64_t>& shape) {
    if (original == shape) {
      return source;
    }
    return call(
        suffix,
        "torch.ops.aten.expand_copy.default",
        {tensor(source),
         {.name = "size", .arg = IntListArg{.values = shape, .ids = {}}},
         {.name = "implicit", .arg = BoolArg{false}}},
        shape);
  };
  const auto reshape = [&](const char* suffix,
                           ValueId source,
                           const std::vector<int64_t>& shape) {
    return call(
        suffix,
        "torch.ops.aten.view_copy.default",
        {tensor(source),
         {.name = "size", .arg = IntListArg{.values = shape, .ids = {}}}},
        shape);
  };
  const auto expanded_a = expand("left_expand", left_id, left, expanded_left);
  const auto expanded_b =
      expand("right_expand", right_id, right, expanded_right);
  const auto a = reshape("left", expanded_a, {batches, rows, inner});
  const auto b = reshape("right", expanded_b, {batches, inner, columns});
  const auto result = call(
      "result",
      "torch.ops.aten.bmm.default",
      {tensor(a), {.name = "mat2", .arg = TensorArg{b}}},
      {batches, rows, columns});
  auto& replacement = graph.node(id);
  replacement.target = "torch.ops.aten.view_copy.default";
  replacement.inputs = {
      tensor(result),
      {.name = "size", .arg = IntListArg{.values = output_shape, .ids = {}}}};
  graph.rebuild_def_use();
}

void lower_native_method(Graph& graph) {
  functionalize_aliases(graph);
  const auto schedule = graph.schedule;
  for (const auto id : schedule) {
    if (graph.node(id).is_call() &&
        graph.node(id).target == "torch.ops.aten.matmul.default") {
      lower_matmul(graph, id);
    }
  }
  for (ValueId id = 0; std::cmp_less(id, graph.values.size()); ++id) {
    auto& value = graph.value(id);
    if (value.is_tensor() && value.role == ValueRole::Intermediate &&
        std::find(graph.output_ids.begin(), graph.output_ids.end(), id) ==
            graph.output_ids.end()) {
      value.tensor_meta().dim_order_hint.clear();
    }
  }
  const auto unsupported =
      std::find_if(graph.values.begin(), graph.values.end(), [](const auto& v) {
        return !cpu::tensor_bytes(v).ok();
      });
  if (unsupported != graph.values.end()) {
    throw std::runtime_error(
        "cpu: value '" + unsupported->name +
        "' is not a static contiguous tensor of rank <= 16");
  }
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
      const double number = std::trunc(static_cast<double>(value));
      const double upper =
          std::ldexp(1.0, std::numeric_limits<Destination>::digits);
      const double lower = std::is_signed_v<Destination> ? -upper : 0;
      if (!std::isfinite(number) || number < lower || number >= upper) {
        throw std::runtime_error(
            "cpu: floating-point conversion is out of range");
      }
    }
  }
  return static_cast<Destination>(value);
}

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
        Source value{};
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
      cpu::RuntimeConfiguration configuration)
      : method_(method) {
    validate_supported_method(method_);
    lower_native_method(method_.graph);
    buffers_.resize(method_.graph.values.size());
    plan_ = std::make_unique<cpu::CPUPlan>(
        method_.graph,
        buffers_,
        std::move(configuration),
        cpu::current_execution_context());
    check_error(plan_->select(), "provider selection");
    for (const auto& binding : method_.data_bindings) {
      buffers_.at(binding.value_id) = constants.acquire(
          package, binding.key, plan_->requirements().at(binding.value_id));
    }
    for (const auto id : method_.graph.input_ids) {
      const auto bytes = cpu::tensor_bytes(method_.graph.value(id));
      check_error(bytes.error(), "input validation");
      auto requirements = cpu::tensor_requirements(bytes.get(), true);
      check_error(requirements.error(), "input validation");
      requirements->merge(plan_->requirements().at(id));
      auto buffer = cpu::allocate_buffer(allocator_, requirements.get());
      check_error(buffer.error(), "input allocation");
      buffers_.at(id) = buffer.get();
    }
    check_error(plan_->prepare(allocator_), "plan preparation");
    input_ready_.resize(num_inputs(), false);
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
    input_ready_.at(index) = false;
    output_ready_ = false;
    copy_tensor(
        buffers_.at(method_.graph.input_ids.at(index)).data,
        meta.dtype,
        data,
        dtype,
        numel);
    input_ready_.at(index) = true;
  }

  void execute() override {
    output_ready_ = false;
    if (std::find(input_ready_.begin(), input_ready_.end(), false) !=
        input_ready_.end()) {
      throw std::runtime_error("cpu: every input must be set before execution");
    }
    check_error(plan_->execute(cpu::current_execution_context()), "execution");
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
        buffers_.at(method_.graph.output_ids.at(index)).data,
        meta.dtype,
        numel);
  }

 private:
  const TensorMeta& input_meta(size_t index) const {
    if (index >= num_inputs()) {
      throw std::runtime_error("cpu: input index out of range");
    }
    return method_.graph.value(method_.graph.input_ids.at(index)).tensor_meta();
  }
  const TensorMeta& output_meta(size_t index) const {
    if (index >= num_outputs()) {
      throw std::runtime_error("cpu: output index out of range");
    }
    return method_.graph.value(method_.graph.output_ids.at(index))
        .tensor_meta();
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
  std::vector<bool> input_ready_;
  bool output_ready_ = false;
  std::unique_ptr<cpu::CPUPlan> plan_;
};

} // namespace

std::unique_ptr<EngineExecutable> create_cpu_engine_executable(
    const Method& method,
    const Package& package,
    CPUConstantStore& constants,
    cpu::RuntimeConfiguration configuration) {
  return std::make_unique<CPUEngineExecutable>(
      method, package, constants, std::move(configuration));
}

} // namespace ptn
