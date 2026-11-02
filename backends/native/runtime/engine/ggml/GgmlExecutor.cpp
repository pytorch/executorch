// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlExecutor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlHostInts.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>
#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn {
namespace {

using namespace ggml_detail;

// Multiplies each slice of the leading dimension by its scale.
// data is a byte buffer holding count floats, as ggml tensor data is.
void scale_slices(void* data, size_t count, const std::vector<float>& scales) {
  float* values = static_cast<float*>(data);
  const size_t slice = count / scales.size();
  for (size_t c = 0; c < scales.size(); ++c) {
    for (size_t i = 0; i < slice; ++i) {
      values[c * slice + i] *= scales[c];
    }
  }
}

// An attention mask written as a comparison between a per-key index and
// per-query positions, both small integer functions of the graph inputs (the
// causal mask of a static KV cache, for instance), attends only keys below a
// host-computable extent: past it every query is masked. Attention, and the
// mask, can then cover just that key prefix exactly. Extents round up to a
// bucket so replays keep reusing one graph while the prefix grows.
constexpr int64_t kKeyExtentBucket = 256;

struct MaskExtent {
  std::string compare; // le or lt
  bool key_lhs = true;
  ValueId key_operand = kInvalid;
  ValueId query_operand = kInvalid;
  int64_t keys = 0;
  // The SDPA mask, then each value up to and including the compare output.
  std::vector<ValueId> chain;
};

// The bucketed prefix of keys that some query attends.
int64_t attended_key_extent(
    const MaskExtent& mask,
    const HostInts& key_index,
    const HostInts& query) {
  if (key_index.data.size() != static_cast<size_t>(mask.keys) ||
      query.data.empty()) {
    return mask.keys;
  }
  // Key j is attended by some query iff compare(key[j], query) holds for
  // the extreme query value; compare(query, key) is the mirrored
  // compare(key, query).
  const auto [lo, hi] = std::ranges::minmax(query.data);
  const std::string& op = mask.compare;
  const auto attended = [&](int64_t key) {
    if (op == "le") {
      return mask.key_lhs ? key <= hi : key >= lo;
    }
    return mask.key_lhs ? key < hi : key > lo;
  };
  int64_t last = -1;
  for (int64_t j = mask.keys - 1; j >= 0 && last < 0; --j) {
    if (attended(key_index.data[static_cast<size_t>(j)])) {
      last = j;
    }
  }
  const int64_t needed = std::max<int64_t>(last + 1, 1);
  return std::min(
      mask.keys,
      (needed + kKeyExtentBucket - 1) / kKeyExtentBucket * kKeyExtentBucket);
}

// Holds a method's non-empty mutable buffer keys in `owners` for the lifetime
// of its executable.
class MutableKeyClaim {
 public:
  MutableKeyClaim(GgmlMutableKeyOwners& owners, const Method& method)
      : owners_(owners) {
    for (const DataBinding& binding : method.data_bindings) {
      if (!binding.mutated || binding.key.empty()) {
        continue;
      }
      const auto owner = owners_.find(binding.key);
      if (owner != owners_.end()) {
        throw std::runtime_error(
            "ggml: mutable buffer '" + binding.key + "' of method '" +
            method.name +
            "' is already bound by a live executable of method '" +
            owner->second +
            "'; sharing mutable buffers across executables is not implemented");
      }
    }
    for (const DataBinding& binding : method.data_bindings) {
      if (binding.mutated && !binding.key.empty() &&
          owners_.try_emplace(binding.key, method.name).second) {
        keys_.push_back(binding.key);
      }
    }
  }

  MutableKeyClaim(const MutableKeyClaim&) = delete;
  MutableKeyClaim& operator=(const MutableKeyClaim&) = delete;

  ~MutableKeyClaim() {
    for (const std::string& key : keys_) {
      owners_.erase(key);
    }
  }

 private:
  GgmlMutableKeyOwners& owners_;
  std::vector<std::string> keys_;
};

class GgmlExecutable final : public EngineExecutable {
 public:
  GgmlExecutable(
      const Method& method,
      const Package& package,
      const GgmlEngineOptions& options,
      size_t* materialized_constant_bytes,
      size_t* heap_copied_constant_bytes,
      GgmlMutableKeyOwners& mutable_key_owners)
      : mutable_key_claim_(mutable_key_owners, method),
        method_(method),
        package_(package),
        options_(options) {
    index_signature();
    initialize_backends();
    fold_batch_norms();
    initialize_persistent_tensors(materialized_constant_bytes);
    index_rope_frequencies();
    index_mask_extents();
    if (heap_copied_constant_bytes != nullptr) {
      *heap_copied_constant_bytes += heap_copied_constant_bytes_;
    }
    // Rejects unsupported ops and shapes at compile time.
    build_graph(full_key_extents());
  }

  ~GgmlExecutable() override {
    for (CachedGraph& cached : cached_graphs_) {
      ggml_backend_sched_synchronize(cached.scheduler.get());
    }
    if (scheduler_) {
      ggml_backend_sched_synchronize(scheduler_.get());
      ggml_backend_sched_reset(scheduler_.get());
    }
  }

  size_t num_inputs() const override {
    return inputs_.size();
  }

  size_t num_outputs() const override {
    return outputs_.size();
  }

  std::vector<int64_t> input_sizes(size_t i) const override {
    return inputs_.at(i).sizes;
  }

  std::vector<int64_t> output_sizes(size_t i) const override {
    return outputs_.at(i).sizes;
  }

  ScalarType input_dtype(size_t i) const override {
    return inputs_.at(i).meta.dtype;
  }

  ScalarType output_dtype(size_t i) const override {
    return outputs_.at(i).meta.dtype;
  }

  void resize_input(size_t i, const std::vector<int64_t>& sizes) override {
    InputSlot& slot = inputs_.at(i);
    if (!slot.meta.accepts_sizes(sizes)) {
      throw std::invalid_argument(
          "ggml: input shape exceeds serialized bounds");
    }
    slot.sizes = sizes;
    slot.bytes.clear();
    executed_ = false;
  }

  void set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype)
      override {
    InputSlot& slot = inputs_.at(i);
    const size_t expected = checked_numel(slot.sizes);
    if (numel != expected) {
      throw std::invalid_argument(
          "ggml: input element count does not match shape");
    }
    if (src_dtype != slot.meta.dtype) {
      throw std::invalid_argument(
          "ggml: input conversion is unsupported; input dtype must match "
          "the graph");
    }
    const size_t nbytes = checked_nbytes(slot.sizes, src_dtype);
    if (nbytes != 0 && data == nullptr) {
      throw std::invalid_argument("ggml: input data is null");
    }
    slot.bytes.resize(nbytes);
    if (nbytes != 0) {
      std::memcpy(slot.bytes.data(), data, nbytes);
    }
    executed_ = false;
  }

  void execute() override {
    executed_ = false;
    if (std::ranges::any_of(inputs_, [](const InputSlot& input) {
          return input.bytes.size() !=
              checked_nbytes(input.sizes, input.meta.dtype);
        })) {
      throw std::logic_error("ggml: every input must be set before execute");
    }

    if (options_.max_cached_graphs == 0) {
      ggml_backend_sched_reset(scheduler_.get());
      GraphInstance instance = build_graph(evaluate_key_extents());
      allocate(scheduler_.get(), instance);
      try {
        run(scheduler_.get(), instance);
      } catch (...) {
        ggml_backend_sched_reset(scheduler_.get());
        throw;
      }
      ggml_backend_sched_reset(scheduler_.get());
    } else {
      CachedGraph& cached = cached_graph();
      try {
        run(cached.scheduler.get(), cached.instance);
      } catch (...) {
        ggml_backend_sched_synchronize(cached.scheduler.get());
        cached_graphs_.pop_back();
        throw;
      }
    }
    executed_ = true;
  }

  void get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype)
      override {
    const OutputSlot& slot = outputs_.at(i);
    if (!executed_) {
      throw std::logic_error("ggml: output requested before execute");
    }
    if (numel != checked_numel(slot.sizes)) {
      throw std::invalid_argument(
          "ggml: output element count does not match shape");
    }
    if (numel != 0 && data == nullptr) {
      throw std::invalid_argument("ggml: output destination is null");
    }
    copy_output(slot, data, dst_dtype);
  }

 private:
  struct CachedGraph {
    std::vector<std::vector<int64_t>> input_sizes;
    // Attention key extents follow input values (positions), so they are
    // part of the key along with the input shapes.
    std::vector<int64_t> key_extents;
    GraphInstance instance;
    ggml_backend_sched_ptr scheduler;
  };

  // Lowering depends on input shapes, constants and the attention key extents
  // computed from input values, so a graph built for the same key replays
  // with fresh input bytes. The most recently used graph is last.
  CachedGraph& cached_graph() {
    std::vector<int64_t> key_extents = evaluate_key_extents();
    const auto same_key = [this, &key_extents](const CachedGraph& cached) {
      if (cached.key_extents != key_extents) {
        return false;
      }
      for (size_t i = 0; i < inputs_.size(); ++i) {
        if (cached.input_sizes[i] != inputs_[i].sizes) {
          return false;
        }
      }
      return true;
    };
    const auto hit = std::ranges::find_if(cached_graphs_, same_key);
    if (hit != cached_graphs_.end()) {
      std::rotate(hit, hit + 1, cached_graphs_.end());
      refresh_inputs(cached_graphs_.back().instance);
      return cached_graphs_.back();
    }
    if (cached_graphs_.size() >= options_.max_cached_graphs) {
      ggml_backend_sched_synchronize(cached_graphs_.front().scheduler.get());
      cached_graphs_.erase(cached_graphs_.begin());
    }
    CachedGraph cached{
        .key_extents = key_extents,
        .instance = build_graph(key_extents),
        .scheduler = make_scheduler()};
    for (const InputSlot& input : inputs_) {
      cached.input_sizes.push_back(input.sizes);
    }
    allocate(cached.scheduler.get(), cached.instance);
    cached_graphs_.push_back(std::move(cached));
    return cached_graphs_.back();
  }

  void refresh_inputs(GraphInstance& instance) const {
    for (const RowIndexCheck& check : instance.row_index_checks) {
      check_index_bytes(
          *std::ranges::find(inputs_, check.input_id, &InputSlot::id),
          check.rows);
    }
    for (Upload& upload : instance.uploads) {
      if (upload.input) {
        const InputSlot& input = inputs_.at(*upload.input);
        upload.bytes = input.meta.dtype == ScalarType::Long
            ? narrow_to_int32(input.bytes)
            : input.bytes;
      }
    }
  }

  void allocate(ggml_backend_sched_t scheduler, const GraphInstance& instance) {
    // The scheduler otherwise keeps graph inputs on the CPU backend and copies
    // them into GPU splits once per consuming view.
    for (const Upload& upload : instance.uploads) {
      ggml_backend_sched_set_tensor_backend(
          scheduler, upload.tensor, primary_backend_);
    }
    if (!ggml_backend_sched_alloc_graph(scheduler, instance.graph)) {
      ggml_backend_sched_reset(scheduler);
      throw std::runtime_error("ggml: failed to allocate execution graph");
    }
  }

  void run(ggml_backend_sched_t scheduler, const GraphInstance& instance) {
    for (const Upload& upload : instance.uploads) {
      if (upload.tensor->buffer == nullptr) {
        continue;
      }
      ggml_backend_tensor_set(
          upload.tensor, upload.bytes.data(), 0, upload.bytes.size());
    }
    const ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, instance.graph);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error(
          "ggml: graph execution failed with status " +
          std::to_string(static_cast<int>(status)));
    }
    const size_t staged = std::accumulate(
        instance.outputs.begin(),
        instance.outputs.end(),
        size_t{0},
        [](size_t total, const TensorRef& output) {
          return total +
              GGML_PAD(ggml_nbytes(output.tensor), kStagingAlignment);
        });
    uint8_t* staging = output_staging(staged);
    for (size_t i = 0; i < outputs_.size(); ++i) {
      OutputSlot& output = outputs_[i];
      ggml_tensor* tensor = instance.outputs.at(i).tensor;
      output.sizes = instance.outputs.at(i).sizes;
      output.data = staging;
      output.nbytes = ggml_nbytes(tensor);
      ggml_backend_tensor_get(tensor, staging, 0, output.nbytes);
      staging += GGML_PAD(output.nbytes, kStagingAlignment);
    }
  }

  void index_signature() {
    for (const ValueId id : method_.graph.input_ids) {
      const TensorMeta& meta = contiguous_meta(id);
      if (meta.quant) {
        throw std::runtime_error(
            "ggml: quantized graph inputs are unsupported");
      }
      inputs_.push_back(InputSlot{.id = id, .meta = meta, .sizes = meta.sizes});
    }
    if (method_.output_specs.size() != method_.graph.output_ids.size()) {
      throw std::runtime_error("ggml: inconsistent output specifications");
    }
    for (size_t i = 0; i < method_.graph.output_ids.size(); ++i) {
      const OutputKind kind = method_.output_specs[i].kind;
      if (kind == OutputKind::UserInputMutation) {
        throw std::runtime_error("ggml: user input mutation is unsupported");
      }
      if (kind != OutputKind::UserOutput) {
        continue;
      }
      const ValueId id = method_.graph.output_ids[i];
      const TensorMeta& meta = contiguous_meta(id);
      if (meta.quant) {
        throw std::runtime_error(
            "ggml: quantized graph outputs are unsupported");
      }
      outputs_.push_back(
          OutputSlot{.id = id, .meta = meta, .sizes = meta.sizes});
    }
    for (const DataBinding& binding : method_.data_bindings) {
      if (binding.has_data && valid(binding.value_id) &&
          method_.graph.value(binding.value_id).is_tensor()) {
        contiguous_meta(binding.value_id);
      }
    }
  }

  // Bytes that cross the engine boundary are laid out as dim_order_hint says.
  const TensorMeta& contiguous_meta(ValueId id) const {
    const TensorMeta& meta = method_.graph.value(id).tensor_meta();
    if (!meta.is_contiguous()) {
      throw std::runtime_error(
          "ggml: tensor '" + method_.graph.value(id).name +
          "' has a non-contiguous dim order");
    }
    return meta;
  }

  void initialize_backends() {
    if (options_.device == GgmlDevice::Gpu) {
      ggml_backend_dev_t device = select_ggml_gpu_device(options_);
      ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
      if (!gpu) {
        throw std::runtime_error(
            std::string("ggml: failed to initialize ") +
            ggml_backend_dev_name(device));
      }
      primary_backend_ = gpu.get();
      backends_.push_back(std::move(gpu));
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!cpu) {
      throw std::runtime_error("ggml: failed to initialize CPU backend");
    }
    if (options_.cpu_threads > 0) {
      ggml_backend_cpu_set_n_threads(cpu.get(), options_.cpu_threads);
    }
    if (primary_backend_ == nullptr) {
      primary_backend_ = cpu.get();
    }
    backends_.push_back(std::move(cpu));

    if (options_.max_cached_graphs == 0) {
      scheduler_ = make_scheduler();
    }
  }

  ggml_backend_sched_ptr make_scheduler() const {
    std::vector<ggml_backend_t> raw(backends_.size());
    std::ranges::transform(backends_, raw.begin(), [](const auto& backend) {
      return backend.get();
    });
    ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(
        raw.data(),
        nullptr,
        static_cast<int>(raw.size()),
        options_.graph_size,
        /*parallel=*/false,
        /*op_offload=*/true));
    if (!scheduler) {
      throw std::runtime_error("ggml: failed to create backend scheduler");
    }
    return scheduler;
  }

  // Quantization parameters are bound but only read through their weight.
  bool is_used(ValueId id) const {
    const std::vector<ValueId>& outputs = method_.graph.output_ids;
    return !method_.graph.value(id).consumer_ids.empty() ||
        std::find(outputs.begin(), outputs.end(), id) != outputs.end();
  }

  bool is_persistent(const DataBinding& binding) const {
    return valid(binding.value_id) &&
        method_.graph.value(binding.value_id).is_tensor() &&
        is_used(binding.value_id);
  }

  static size_t stored_nbytes(const TensorMeta& meta) {
    if (packed_quant(meta) != nullptr) {
      return packed_nbytes(meta);
    }
    if (affine_quant(meta) != nullptr) {
      return checked_numel(meta.sizes) / 2;
    }
    return checked_nbytes(meta.sizes, meta.dtype);
  }

  void check_q4_consumers(ValueId id, const std::string& key) const {
    for (const NodeId consumer : method_.graph.value(id).consumer_ids) {
      const Node& node = method_.graph.node(consumer);
      const std::string op = op_key(node.target);
      const NamedArgument* weight = op == "aten.linear.default"
          ? find_arg(node, "weight", 1)
          : op == "aten.embedding.default" ? find_arg(node, "weight", 0)
                                           : nullptr;
      if (weight == nullptr || weight->arg.kind() != ArgKind::Tensor ||
          weight->arg.as_tensor().id != id) {
        throw std::runtime_error(
            "ggml: quantized weight '" + key +
            "' may only feed linear or embedding weights, not '" + node.target +
            "'");
      }
    }
  }

  // Converters read some dtypes through typed pointers, so the package's bytes
  // are viewed in place only when aligned to the element size.
  std::optional<OwnedBytes> read_constant(const std::string& key) const {
    const std::optional<ConstantInfo> info = package_.constant_info(key);
    if (!info) {
      return std::nullopt;
    }
    std::optional<OwnedBytes> view = package_.view_constant(key);
    if (view &&
        reinterpret_cast<uintptr_t>(view->span().data()) %
                std::max<size_t>(element_size(info->dtype), 1) ==
            0) {
      return view;
    }
    return acquire_constant(key);
  }

  std::optional<OwnedBytes> acquire_constant(const std::string& key) const {
    std::optional<OwnedBytes> bytes = package_.acquire_constant(key);
    if (bytes) {
      heap_copied_constant_bytes_ += bytes->span().size();
    }
    return bytes;
  }

  std::vector<uint8_t> load_q4_0(const TensorMeta& meta, ByteSpan packed)
      const {
    const AffineGroupQuant& quant = q4_affine(meta);
    const std::optional<OwnedBytes> scales =
        read_constant(quant.scale_data_key);
    if (!scales) {
      throw std::runtime_error(
          "ggml: package is missing scales '" + quant.scale_data_key + "'");
    }
    if (!quant.zero_point_data_key.empty()) {
      const std::optional<OwnedBytes> zeros =
          read_constant(quant.zero_point_data_key);
      if (!zeros || std::ranges::any_of(zeros->span(), [](uint8_t byte) {
            return byte != 0;
          })) {
        throw std::runtime_error(
            "ggml: Q4_0 needs zero points of zero, see '" +
            quant.zero_point_data_key + "'");
      }
    }
    return affine_int4_to_q4_0(
        meta, packed, to_float_vector(scales->span(), quant.scale_dtype));
  }

  // The binding of a constant with data that no op writes.
  const DataBinding* constant_binding(ValueId id) const {
    const auto binding = std::ranges::find_if(
        method_.data_bindings, [id](const DataBinding& candidate) {
          return candidate.value_id == id && candidate.has_data;
        });
    if (binding == method_.data_bindings.end() ||
        std::ranges::any_of(method_.output_specs, [id](const OutputSpec& spec) {
          return spec.kind == OutputKind::BufferMutation &&
              spec.target_id == id;
        })) {
      return nullptr;
    }
    const auto mutates_id = [id](const NamedArgument& input) {
      return input.mutated && input.arg.kind() == ArgKind::Tensor &&
          input.arg.as_tensor().id == id;
    };
    if (std::ranges::any_of(
            method_.graph.value(id).consumer_ids, [&](NodeId consumer) {
              return std::ranges::any_of(
                  method_.graph.node(consumer).inputs, mutates_id);
            })) {
      return nullptr;
    }
    return &*binding;
  }

  // The values of an unquantized floating-point constant.
  std::optional<std::vector<float>> constant_floats(ValueId id) const {
    const DataBinding* binding = constant_binding(id);
    const TensorMeta& meta = method_.graph.value(id).tensor_meta();
    if (binding == nullptr || packed_quant(meta) != nullptr ||
        affine_quant(meta) != nullptr ||
        (meta.dtype != ScalarType::Float && meta.dtype != ScalarType::Half &&
         meta.dtype != ScalarType::BFloat16)) {
      return std::nullopt;
    }
    const std::optional<OwnedBytes> bytes = read_constant(binding->key);
    if (!bytes) {
      return std::nullopt;
    }
    return to_float_vector(bytes->span(), meta.dtype);
  }

  // An optional tensor argument's constant values: nullopt if it is present
  // but not constant, an empty vector if absent.
  std::optional<std::vector<float>> optional_constant(
      const Node& node,
      std::string_view name,
      size_t positional) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() == ArgKind::None) {
      return std::vector<float>{};
    }
    if (found->arg.kind() != ArgKind::Tensor ||
        !valid(found->arg.as_tensor().id)) {
      return std::nullopt;
    }
    return constant_floats(found->arg.as_tensor().id);
  }

  // The convolution producing `id` if it may absorb a batch norm reading
  // `id`: nothing else reads its output, and nothing else reads its
  // unquantized F32 constant weight, whose leading dimension is the output
  // channels and which the loader scales.
  const Node* absorbing_convolution(ValueId id) const {
    const Value& value = method_.graph.value(id);
    if (value.consumer_ids.size() != 1 || !valid(value.producer_id) ||
        std::ranges::find(method_.graph.output_ids, id) !=
            method_.graph.output_ids.end()) {
      return nullptr;
    }
    const Node& conv = method_.graph.node(value.producer_id);
    if (!conv.is_call() || op_key(conv.target) != "aten.convolution.default" ||
        output_ids(conv) != std::vector<ValueId>{id}) {
      return nullptr;
    }
    const NamedArgument* transposed = find_arg(conv, "transposed", 6);
    const NamedArgument* weight = find_arg(conv, "weight", 1);
    if (transposed == nullptr || transposed->arg.kind() != ArgKind::Bool ||
        transposed->arg.as_bool().value ||
        valid(transposed->arg.as_bool().id) || weight == nullptr ||
        weight->arg.kind() != ArgKind::Tensor) {
      return nullptr;
    }
    const ValueId weight_id = weight->arg.as_tensor().id;
    if (!valid(weight_id) || constant_binding(weight_id) == nullptr) {
      return nullptr;
    }
    const Value& weight_value = method_.graph.value(weight_id);
    if (weight_value.consumer_ids.size() != 1 ||
        weight_value.tensor_meta().dtype != ScalarType::Float ||
        weight_value.tensor_meta().quant) {
      return nullptr;
    }
    return &conv;
  }

  void fold_batch_norms() {
    // y = x * scale + shift with scale = weight / sqrt(var + eps) and
    // shift = bias - mean * scale.
    struct Fold {
      ValueId output = kInvalid;
      ValueId conv_output = kInvalid;
      std::vector<float> scale;
      std::vector<float> shift;
    };
    std::vector<Fold> folds;
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() ||
          op_key(node.target) !=
              "aten._native_batch_norm_legit_no_training.default") {
        continue;
      }
      const NamedArgument* input = find_arg(node, "input", 0);
      const NamedArgument* eps = find_arg(node, "eps", 6);
      const NamedArgument* mean = find_arg(node, "running_mean", 3);
      const NamedArgument* var = find_arg(node, "running_var", 4);
      if (input == nullptr || input->arg.kind() != ArgKind::Tensor ||
          eps == nullptr || eps->arg.kind() != ArgKind::Float ||
          valid(eps->arg.as_float().id) || mean == nullptr ||
          mean->arg.kind() != ArgKind::Tensor || var == nullptr ||
          var->arg.kind() != ArgKind::Tensor) {
        continue;
      }
      const auto means = constant_floats(mean->arg.as_tensor().id);
      const auto vars = constant_floats(var->arg.as_tensor().id);
      const auto weights = optional_constant(node, "weight", 1);
      const auto biases = optional_constant(node, "bias", 2);
      const std::vector<ValueId> outputs = output_ids(node);
      if (!means || !vars || !weights || !biases || outputs.empty() ||
          vars->size() != means->size() ||
          (!weights->empty() && weights->size() != means->size()) ||
          (!biases->empty() && biases->size() != means->size())) {
        continue;
      }
      Fold entry{.output = outputs.front()};
      const size_t channels = means->size();
      entry.scale.resize(channels);
      entry.shift.resize(channels);
      for (size_t c = 0; c < channels; ++c) {
        const double scale = (weights->empty() ? 1.0 : (*weights)[c]) /
            std::sqrt(static_cast<double>((*vars)[c]) +
                      eps->arg.as_float().value);
        entry.scale[c] = static_cast<float>(scale);
        entry.shift[c] = static_cast<float>(
            (biases->empty() ? 0.0 : (*biases)[c]) - (*means)[c] * scale);
      }
      const ValueId input_id = input->arg.as_tensor().id;
      if (const Node* conv =
              valid(input_id) ? absorbing_convolution(input_id) : nullptr) {
        const ValueId weight_id =
            find_arg(*conv, "weight", 1)->arg.as_tensor().id;
        const auto conv_biases = optional_constant(*conv, "bias", 2);
        if (conv_biases &&
            method_.graph.value(weight_id).tensor_meta().sizes.at(0) ==
                static_cast<int64_t>(channels) &&
            (conv_biases->empty() || conv_biases->size() == channels)) {
          if (!conv_biases->empty()) {
            for (size_t c = 0; c < channels; ++c) {
              entry.shift[c] += (*conv_biases)[c] * entry.scale[c];
            }
          }
          weight_scales_.emplace(weight_id, std::move(entry.scale));
          entry.scale.clear();
          entry.conv_output = input_id;
        }
      }
      folds.push_back(std::move(entry));
    }
    if (folds.empty()) {
      return;
    }

    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * (2 * folds.size() + 1),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    batch_norm_context_.reset(ggml_init(params));
    if (!batch_norm_context_) {
      throw std::runtime_error("ggml: failed to create batch norm context");
    }
    const auto new_vector = [this](size_t size) {
      return ggml_new_tensor_1d(
          batch_norm_context_.get(), GGML_TYPE_F32, static_cast<int64_t>(size));
    };
    for (const Fold& entry : folds) {
      if (valid(entry.conv_output)) {
        batch_norms_.absorbed.insert(entry.output);
        batch_norms_.conv_bias.emplace(
            entry.conv_output, new_vector(entry.shift.size()));
      } else {
        batch_norms_.affine.emplace(
            entry.output,
            std::pair{
                new_vector(entry.scale.size()),
                new_vector(entry.shift.size())});
      }
    }
    batch_norm_buffer_.reset(ggml_backend_alloc_ctx_tensors(
        batch_norm_context_.get(), primary_backend_));
    if (!batch_norm_buffer_) {
      throw std::runtime_error("ggml: failed to allocate batch norm constants");
    }
    const auto upload = [](ggml_tensor* tensor,
                           const std::vector<float>& data) {
      ggml_backend_tensor_set(
          tensor, data.data(), 0, data.size() * sizeof(float));
    };
    for (const Fold& entry : folds) {
      if (valid(entry.conv_output)) {
        upload(batch_norms_.conv_bias.at(entry.conv_output), entry.shift);
      } else {
        const auto [scale, shift] = batch_norms_.affine.at(entry.output);
        upload(scale, entry.scale);
        upload(shift, entry.shift);
      }
    }
    ggml_backend_buffer_set_usage(
        batch_norm_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }

  void initialize_persistent_tensors(size_t* materialized_constant_bytes) {
    const size_t count = static_cast<size_t>(std::ranges::count_if(
        method_.data_bindings,
        [this](const DataBinding& binding) { return is_persistent(binding); }));
    if (count == 0) {
      return;
    }
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * (count + 16),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    persistent_context_.reset(ggml_init(params));
    if (!persistent_context_) {
      throw std::runtime_error(
          "ggml: failed to create persistent tensor context");
    }

    for (const DataBinding& binding : method_.data_bindings) {
      if (!is_persistent(binding)) {
        continue;
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      if (affine_quant(meta) != nullptr || packed_quant(meta) != nullptr) {
        check_q4_consumers(binding.value_id, binding.key);
      }
      ggml_tensor* tensor =
          new_tensor(persistent_context_.get(), compute_type(meta), meta.sizes);
      ggml_set_name(tensor, binding.key.c_str());
      persistent_.emplace(binding.value_id, tensor);
    }

    persistent_buffer_.reset(ggml_backend_alloc_ctx_tensors(
        persistent_context_.get(), primary_backend_));
    if (!persistent_buffer_) {
      throw std::runtime_error("ggml: failed to allocate persistent tensors");
    }

    const bool host_buffer =
        ggml_backend_buffer_is_host(persistent_buffer_.get());
    size_t loaded = 0;
    for (const DataBinding& binding : method_.data_bindings) {
      const auto found = persistent_.find(binding.value_id);
      if (found == persistent_.end()) {
        continue;
      }
      ggml_tensor* tensor = found->second;
      if (!binding.has_data) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
        continue;
      }
      const std::optional<ConstantInfo> info =
          package_.constant_info(binding.key);
      if (!info) {
        throw std::runtime_error(
            "ggml: package is missing constant '" + binding.key + "'");
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      const size_t expected = stored_nbytes(meta);
      if (info->nbytes != expected) {
        throw std::runtime_error(
            "ggml: stored byte count disagrees with PTG tensor '" +
            binding.key + "'");
      }
      const bool is_q4 = affine_quant(meta) != nullptr;
      const bool narrow =
          tensor->type == GGML_TYPE_I32 && meta.dtype == ScalarType::Long;
      const auto scales = weight_scales_.find(binding.value_id);
      const bool scaled = scales != weight_scales_.end();
      if (!is_q4 && !narrow && host_buffer) {
        if (ggml_nbytes(tensor) != expected) {
          throw std::runtime_error(
              "ggml: stored byte count disagrees with ggml tensor '" +
              binding.key + "'");
        }
        package_.load_constant_into(
            binding.key,
            MutableByteSpan(static_cast<uint8_t*>(tensor->data), expected));
        if (scaled) {
          scale_slices(tensor->data, expected / sizeof(float), scales->second);
        }
        loaded += expected;
        continue;
      }
      const std::optional<OwnedBytes> bytes = is_q4 || narrow
          ? read_constant(binding.key)
          // Device uploads go through a heap copy: ROCm fails to pin a
          // file-backed mapping, and ggml_backend_tensor_set from one is
          // several times slower.
          : acquire_constant(binding.key);
      if (!bytes) {
        throw std::runtime_error(
            "ggml: package is missing constant '" + binding.key + "'");
      }
      ByteSpan span = bytes->span();
      std::vector<uint8_t> converted;
      if (is_q4) {
        converted = load_q4_0(meta, span);
        span = converted;
      } else if (narrow) {
        converted = narrow_to_int32(span);
        span = converted;
      } else if (scaled) {
        converted.assign(span.begin(), span.end());
        scale_slices(
            converted.data(), converted.size() / sizeof(float), scales->second);
        span = converted;
      }
      if (ggml_nbytes(tensor) != span.size()) {
        throw std::runtime_error(
            "ggml: converted byte count disagrees with tensor '" + binding.key +
            "'");
      }
      ggml_backend_tensor_set(tensor, span.data(), 0, span.size());
      loaded += span.size();
    }
    ggml_backend_buffer_set_usage(
        persistent_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (materialized_constant_bytes != nullptr) {
      *materialized_constant_bytes += loaded;
    }
  }

  void index_rope_frequencies() {
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() || op_key(node.target) != "native.rope.default") {
        continue;
      }
      const Argument& arg = require_arg(node, "inv_freq", 2).arg;
      if (arg.kind() != ArgKind::Tensor) {
        throw std::runtime_error("ggml: rope inv_freq must be a tensor");
      }
      const ValueId inv_freq = arg.as_tensor().id;
      if (rope_freq_factors_.contains(inv_freq)) {
        continue;
      }
      const auto binding = std::ranges::find_if(
          method_.data_bindings, [inv_freq](const DataBinding& candidate) {
            return candidate.value_id == inv_freq && candidate.has_data;
          });
      const std::optional<OwnedBytes> bytes =
          binding == method_.data_bindings.end() ? std::nullopt
                                                 : read_constant(binding->key);
      if (!bytes) {
        throw std::runtime_error("ggml: rope inv_freq must be a constant");
      }
      std::vector<float> factors = to_float_vector(
          bytes->span(), method_.graph.value(inv_freq).tensor_meta().dtype);
      // ggml_rope_ext with freq_base 1 computes theta = pos / factor.
      for (float& factor : factors) {
        if (factor == 0.0f) {
          throw std::runtime_error("ggml: rope inv_freq has a zero entry");
        }
        factor = 1.0f / factor;
      }
      std::vector<uint8_t> raw(factors.size() * sizeof(float));
      std::memcpy(raw.data(), factors.data(), raw.size());
      rope_freq_factors_.emplace(inv_freq, std::move(raw));
    }
  }

  static void
  copy_output(const OutputSlot& slot, void* destination, ScalarType dst_dtype) {
    const size_t count = checked_numel(slot.sizes);
    const bool narrowed = slot.meta.dtype == ScalarType::Long;
    if (slot.nbytes !=
        (narrowed ? count * sizeof(int32_t)
                  : checked_nbytes(slot.sizes, slot.meta.dtype))) {
      throw std::logic_error(
          "ggml: output byte count disagrees with its dtype");
    }
    if (slot.meta.dtype == dst_dtype) {
      if (narrowed) {
        const auto* src = reinterpret_cast<const int32_t*>(slot.data);
        std::copy(src, src + count, static_cast<int64_t*>(destination));
        return;
      }
      std::memcpy(destination, slot.data, slot.nbytes);
      return;
    }
    if (dst_dtype != ScalarType::Float) {
      throw std::invalid_argument(
          "ggml: output conversion only supports conversion to Float");
    }
    float* out = static_cast<float*>(destination);
    if (slot.meta.dtype == ScalarType::Half) {
      ggml_fp16_to_fp32_row(
          reinterpret_cast<const ggml_fp16_t*>(slot.data),
          out,
          static_cast<int64_t>(count));
      return;
    }
    if (slot.meta.dtype == ScalarType::BFloat16) {
      ggml_bf16_to_fp32_row(
          reinterpret_cast<const ggml_bf16_t*>(slot.data),
          out,
          static_cast<int64_t>(count));
      return;
    }
    throw std::invalid_argument(
        "ggml: unsupported output dtype conversion to Float");
  }

  GraphInstance build_graph(const std::vector<int64_t>& key_extents);

  static std::optional<int64_t> static_dim(const TensorMeta& meta, size_t i) {
    if (i >= meta.sizes.size() ||
        (!meta.lower_bounds.empty() &&
         meta.lower_bounds.at(i) != meta.sizes[i])) {
      return std::nullopt;
    }
    return meta.sizes[i];
  }

  // Matches mask = wrappers(compare(key_index, query)) with key_index varying
  // only along the last (key) axis and query constant along it. Every value
  // in the chain must feed nothing but the next wrapper, or SDPA masks.
  std::optional<MaskExtent> match_mask_extent(ValueId mask_id, int64_t keys)
      const {
    const Graph& graph = method_.graph;
    // A boolean mask, where true means attend; additive masks never match.
    if (!graph.value(mask_id).is_tensor() ||
        graph.value(mask_id).tensor_meta().dtype != ScalarType::Bool) {
      return std::nullopt;
    }
    MaskExtent out;
    out.keys = keys;
    ValueId id = mask_id;
    for (size_t depth = 0; depth < 16; ++depth) {
      const Value& value = graph.value(id);
      if (!value.is_tensor() ||
          static_dim(value.tensor_meta(), value.tensor_meta().ndim() - 1) !=
              keys ||
          std::ranges::find(graph.output_ids, id) != graph.output_ids.end()) {
        return std::nullopt;
      }
      out.chain.push_back(id);
      const NodeId producer = value.producer_id;
      if (!valid(producer) || !graph.node(producer).is_call()) {
        return std::nullopt;
      }
      const Node& node = graph.node(producer);
      const std::string op = op_key(node.target);
      const NamedArgument* self = find_arg(node, "self", 0);
      if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
        return std::nullopt;
      }
      if (op == "aten.le.Tensor" || op == "aten.lt.Tensor") {
        const NamedArgument* other = find_arg(node, "other", 1);
        if (other == nullptr || other->arg.kind() != ArgKind::Tensor) {
          return std::nullopt;
        }
        const auto is_key = [&](ValueId operand) {
          const TensorMeta& meta = graph.value(operand).tensor_meta();
          for (size_t i = 0; i + 1 < meta.ndim(); ++i) {
            if (static_dim(meta, i) != 1) {
              return false;
            }
          }
          return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == keys;
        };
        const auto is_query = [&](ValueId operand) {
          const TensorMeta& meta = graph.value(operand).tensor_meta();
          return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == 1;
        };
        const ValueId a = self->arg.as_tensor().id;
        const ValueId b = other->arg.as_tensor().id;
        out.compare = op.substr(5, 2);
        if (is_key(a) && is_query(b)) {
          out.key_operand = a;
          out.query_operand = b;
        } else if (is_key(b) && is_query(a)) {
          out.key_lhs = false;
          out.key_operand = b;
          out.query_operand = a;
        } else {
          return std::nullopt;
        }
        break;
      }
      if (op == "aten.expand.default" || op == "aten.expand_copy.default") {
        // The target size of the key axis is rewritten with the extent.
        const NamedArgument* size = find_arg(node, "size", 1);
        if (size == nullptr || size->arg.kind() != ArgKind::IntList) {
          return std::nullopt;
        }
        const IntListArg& list = size->arg.as_int_list();
        if (list.values.empty() ||
            (!list.ids.empty() && valid(list.ids.back())) ||
            (list.values.back() != -1 && list.values.back() != keys)) {
          return std::nullopt;
        }
      } else if (
          op != "aten.unsqueeze.default" &&
          op != "aten.unsqueeze_copy.default" && op != "aten.alias.default" &&
          op != "aten.clone.default") {
        return std::nullopt;
      }
      id = self->arg.as_tensor().id;
    }
    if (out.compare.empty()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < out.chain.size(); ++i) {
      for (const NodeId consumer : graph.value(out.chain[i]).consumer_ids) {
        const Node& node = graph.node(consumer);
        const bool next_wrapper =
            i > 0 && graph.value(out.chain[i - 1]).producer_id == consumer;
        const NamedArgument* mask = find_arg(node, "attn_mask", 3);
        const bool sdpa_mask = i == 0 && node.is_call() &&
            op_key(node.target) ==
                "aten.scaled_dot_product_attention.default" &&
            mask != nullptr && mask->arg.kind() == ArgKind::Tensor &&
            mask->arg.as_tensor().id == out.chain[i];
        if (!next_wrapper && !sdpa_mask) {
          return std::nullopt;
        }
      }
    }
    return out;
  }

  void index_mask_extents() {
    std::unordered_map<ValueId, std::optional<MaskExtent>> by_mask;
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() ||
          op_key(node.target) != "aten.scaled_dot_product_attention.default") {
        continue;
      }
      const NamedArgument* mask = find_arg(node, "attn_mask", 3);
      const NamedArgument* key = find_arg(node, "key", 1);
      if (mask == nullptr || mask->arg.kind() != ArgKind::Tensor ||
          !valid(mask->arg.as_tensor().id) || key == nullptr ||
          key->arg.kind() != ArgKind::Tensor) {
        continue;
      }
      const TensorMeta& key_meta =
          method_.graph.value(key->arg.as_tensor().id).tensor_meta();
      const std::optional<int64_t> keys = static_dim(key_meta, 2);
      const ValueId mask_id = mask->arg.as_tensor().id;
      auto [it, inserted] = by_mask.try_emplace(mask_id);
      if (inserted) {
        if (key_meta.ndim() == 4 && keys) {
          it->second = match_mask_extent(mask_id, *keys);
        }
      } else if (it->second && it->second->keys != keys) {
        // Every attention reading this mask must cover the same keys.
        it->second.reset();
      }
    }
    for (auto& [mask_id, extent] : by_mask) {
      if (extent) {
        mask_extents_.push_back(std::move(*extent));
      }
    }
  }

  std::vector<int64_t> full_key_extents() const {
    std::vector<int64_t> out(mask_extents_.size());
    std::ranges::transform(mask_extents_, out.begin(), &MaskExtent::keys);
    return out;
  }

  std::vector<int64_t> evaluate_key_extents() {
    if (mask_extents_.empty()) {
      return full_key_extents();
    }
    HostIntEvaluator host(
        method_,
        inputs_,
        [this](ValueId id) { return host_constant(id); },
        host_static_values_);
    std::vector<int64_t> out;
    for (const MaskExtent& mask : mask_extents_) {
      const HostInts* key_index = host.tensor(mask.key_operand);
      const HostInts* query = host.tensor(mask.query_operand);
      out.push_back(
          key_index != nullptr && query != nullptr
              ? attended_key_extent(mask, *key_index, *query)
              : mask.keys);
    }
    return out;
  }

  // Small integer constants, e.g. a key offset, read once from the package.
  std::optional<HostInts> host_constant(ValueId id) {
    if (const auto found = host_constants_.find(id);
        found != host_constants_.end()) {
      return found->second;
    }
    std::optional<HostInts> out;
    const auto binding = std::ranges::find_if(
        method_.data_bindings, [id](const DataBinding& candidate) {
          return candidate.value_id == id && candidate.has_data &&
              !candidate.mutated;
        });
    const TensorMeta* meta = method_.graph.value(id).is_tensor()
        ? &method_.graph.value(id).tensor_meta()
        : nullptr;
    if (binding != method_.data_bindings.end() && meta != nullptr &&
        !meta->quant && checked_numel(meta->sizes) <= 64 &&
        (meta->dtype == ScalarType::Long || meta->dtype == ScalarType::Int)) {
      if (const std::optional<OwnedBytes> bytes = read_constant(binding->key)) {
        out = HostInts{meta->sizes, {}};
        const ByteSpan span = bytes->span();
        if (meta->dtype == ScalarType::Long) {
          out->data.resize(span.size() / sizeof(int64_t));
          std::memcpy(out->data.data(), span.data(), span.size());
        } else {
          std::vector<int32_t> narrow(span.size() / sizeof(int32_t));
          std::memcpy(narrow.data(), span.data(), span.size());
          out->data.assign(narrow.begin(), narrow.end());
        }
        if (out->data.size() != checked_numel(meta->sizes)) {
          out.reset();
        }
      }
    }
    host_constants_.emplace(id, out);
    return out;
  }

  static constexpr size_t kStagingAlignment = 64;

  // Device-to-host copies into pinned memory skip the driver's pageable
  // bounce buffer. Devices without a host buffer type use plain memory.
  uint8_t* output_staging(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(primary_backend_));
    if (host == nullptr) {
      output_host_.resize(std::max(output_host_.size(), bytes));
      return output_host_.data();
    }
    if (!output_staging_ ||
        ggml_backend_buffer_get_size(output_staging_.get()) < bytes) {
      output_staging_.reset(ggml_backend_buft_alloc_buffer(host, bytes));
      if (!output_staging_) {
        throw std::runtime_error("ggml: failed to allocate output staging");
      }
    }
    return static_cast<uint8_t*>(
        ggml_backend_buffer_get_base(output_staging_.get()));
  }

  MutableKeyClaim mutable_key_claim_;
  // Accumulated while compiling, reported to the context at the end.
  mutable size_t heap_copied_constant_bytes_ = 0;
  const Method& method_;
  const Package& package_;
  GgmlEngineOptions options_;
  std::vector<InputSlot> inputs_;
  std::vector<OutputSlot> outputs_;
  std::vector<ggml_backend_ptr> backends_;
  ggml_backend_t primary_backend_ = nullptr;
  ggml_backend_sched_ptr scheduler_;
  std::vector<CachedGraph> cached_graphs_;
  std::vector<MaskExtent> mask_extents_;
  std::unordered_map<ValueId, std::optional<HostInts>> host_constants_;
  HostIntEvaluator::StaticCache host_static_values_;
  ggml_context_ptr persistent_context_;
  ggml_backend_buffer_ptr persistent_buffer_;
  ggml_backend_buffer_ptr output_staging_;
  std::vector<uint8_t> output_host_;
  std::unordered_map<ValueId, ggml_tensor*> persistent_;
  bool executed_ = false;
  std::unordered_map<ValueId, std::vector<uint8_t>> rope_freq_factors_;
  // Per-output-channel factors applied to convolution weights at load.
  std::unordered_map<ValueId, std::vector<float>> weight_scales_;
  FoldedBatchNorms batch_norms_;
  ggml_context_ptr batch_norm_context_;
  ggml_backend_buffer_ptr batch_norm_buffer_;
};

GraphInstance GgmlExecutable::build_graph(
    const std::vector<int64_t>& key_extents) {
  std::unordered_map<ValueId, int64_t> truncated;
  for (size_t i = 0; i < mask_extents_.size(); ++i) {
    if (key_extents.at(i) < mask_extents_[i].keys) {
      for (const ValueId id : mask_extents_[i].chain) {
        truncated.emplace(id, key_extents[i]);
      }
    }
  }
  GraphInstance instance = GraphBuilder(
                               method_,
                               inputs_,
                               outputs_,
                               persistent_,
                               rope_freq_factors_,
                               batch_norms_,
                               std::move(truncated),
                               options_.graph_size)
                               .build();
  if (options_.print_graph_ops) {
    std::map<std::string, int> counts;
    for (int i = 0; i < ggml_graph_n_nodes(instance.graph); ++i) {
      ++counts[ggml_op_desc(ggml_graph_node(instance.graph, i))];
    }
    std::string line = "ggml graph ops:";
    for (const auto& [op, count] : counts) {
      line += " " + op + "=" + std::to_string(count);
    }
    std::fprintf(stderr, "%s\n", line.c_str());
  }
  return instance;
}

} // namespace

ggml_backend_dev_t select_ggml_gpu_device(const GgmlEngineOptions& options) {
  int remaining = options.device_index;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
      continue;
    }
    if (remaining-- == 0) {
      return device;
    }
  }
  throw std::runtime_error(
      "ggml: GPU device " + std::to_string(options.device_index) +
      " is unavailable (" + std::to_string(options.device_index - remaining) +
      " GPU devices registered by the linked ggml backends)");
}

// cppcheck-suppress unusedFunction
std::unique_ptr<EngineExecutable> make_ggml_executable(
    const Method& method,
    const Package& package,
    const GgmlEngineOptions& options,
    size_t* materialized_constant_bytes,
    size_t* heap_copied_constant_bytes,
    GgmlMutableKeyOwners& mutable_key_owners) {
  return std::make_unique<GgmlExecutable>(
      method,
      package,
      options,
      materialized_constant_bytes,
      heap_copied_constant_bytes,
      mutable_key_owners);
}

} // namespace ptn
