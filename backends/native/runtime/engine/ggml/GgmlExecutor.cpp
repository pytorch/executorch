// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlExecutor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>
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

// How one persistent tensor's package bytes become its device bytes. Each
// load-time transform is decided once, in plan_load(), which records it in
// `transforms`; the loader applies only what the plan holds. A constant is
// shared under its source bytes, buffer type, device type and shape and
// that chain, so a transform that changes device bytes changes the key.
struct GgmlLoadPlan {
  std::string source;
  std::string buffer_type;
  ggml_type type = GGML_TYPE_F32;
  std::vector<int64_t> sizes;
  bool q4 = false;
  std::optional<std::vector<float>> row_scales;
  bool narrow = false;
  // Weight-only quantized weights dequantized to F32, with activation
  // quantization off (allow_activation_quant_for_weight_only).
  bool dequant = false;
  const std::vector<float>* scales = nullptr;
  std::string transforms;

  std::string key() const {
    std::string out =
        source + '|' + buffer_type + '|' + ggml_type_name(type) + '[';
    for (const int64_t size : sizes) {
      out += std::to_string(size) + ',';
    }
    return out + ']' + transforms;
  }

  bool operator==(const GgmlLoadPlan& other) const {
    return source == other.source && buffer_type == other.buffer_type &&
        type == other.type && sizes == other.sizes &&
        transforms == other.transforms;
  }
};

namespace {

using namespace ggml_detail;
namespace cache = executorch::extension::llm::cache;
using executorch::extension::llm::cache::Run;
using executorch::extension::llm::cache::SeqStepPlan;

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

// An attention mask written as comparisons between a per-key index and
// per-query positions, both small integer functions of the graph inputs, and
// anded together as a product of bools (the causal or sliding-window mask of a
// static KV cache, for instance), attends only keys below a host-computable
// extent: past it every query is masked. Attention, and the mask, can then
// cover just that key prefix exactly. Extents round up to a bucket so replays
// keep reusing one graph while the prefix grows.
constexpr int64_t kKeyExtentBucket = 256;

struct KeyCompare {
  std::string compare; // le or lt
  bool key_lhs = true;
  ValueId key_operand = kInvalid;
  ValueId query_operand = kInvalid;
};

// Masks that share values, which are computed over one key prefix: the
// largest that any of the masks attends.
struct MaskExtent {
  int64_t keys = 0;
  // Each SDPA mask as the comparisons it ands together.
  std::vector<std::vector<KeyCompare>> masks;
  // The masks and every value they are computed from over the key axis.
  std::vector<ValueId> chain;
};

// The bucketed prefix of keys that some query attends.
int64_t attended_key_extent(
    const KeyCompare& mask,
    int64_t keys,
    const HostInts& key_index,
    const HostInts& query) {
  if (key_index.data.size() != static_cast<size_t>(keys) ||
      query.data.empty()) {
    return keys;
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
  for (int64_t j = keys - 1; j >= 0 && last < 0; --j) {
    if (attended(key_index.data[static_cast<size_t>(j)])) {
      last = j;
    }
  }
  const int64_t needed = std::max<int64_t>(last + 1, 1);
  return std::min(
      keys,
      (needed + kKeyExtentBucket - 1) / kKeyExtentBucket * kKeyExtentBucket);
}

// Storage of the mutable buffers or constants one compile introduced to its
// context.
struct StateBlock {
  ggml_context_ptr context;
  ggml_backend_buffer_ptr buffer;
};

class GgmlExecutable final : public EngineExecutable {
 public:
  GgmlExecutable(
      const Method& method,
      const Package& package,
      const GgmlEngineOptions& options,
      size_t* materialized_constant_bytes,
      size_t* heap_copied_constant_bytes,
      size_t* f16_kv_cache_bytes,
      GgmlContextState& state,
      GgmlKvCache kv_cache)
      : state_(state),
        method_(method),
        package_(package),
        options_(options),
        kv_cache_(kv_cache) {
    index_signature();
    initialize_backends();
    fold_batch_norms();
    initialize_persistent_tensors(
        materialized_constant_bytes, f16_kv_cache_bytes);
    index_rope_frequencies();
    index_mask_extents();
    index_offgraph_attention();
    fold_constants(materialized_constant_bytes);
    if (heap_copied_constant_bytes != nullptr) {
      *heap_copied_constant_bytes += heap_copied_constant_bytes_;
    }
    // Rejects unsupported ops and shapes at compile time, and sizes the
    // context's compute buffers for the largest inputs and key extents.
    worst_case_ =
        std::make_shared<const GraphInstance>(build_graph(full_key_extents()));
    state_.reserved.push_back(worst_case_);
    check_host_integers_boundable(*worst_case_);
    reserve(*worst_case_);
    // Only a compiled executable has initialized its fresh storage.
    for (auto& [key, storage] : fresh_state_) {
      state_.storage.emplace(key, std::move(storage));
    }
    fresh_state_.clear();
  }

  ~GgmlExecutable() override {
    if (state_.scheduler) {
      ggml_backend_sched_synchronize(state_.scheduler.get());
      ggml_backend_sched_reset(state_.scheduler.get());
      state_.allocated = nullptr;
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
    // A step starts where its sequences end and commits only once it has run,
    // so a sequence step that fails leaves the sequence unchanged: whatever
    // it wrote lies past the length. A cell step ends with its forward, giving
    // back the cells it claimed if the forward fails.
    if (valid(offgraph_position_)) {
      offgraph_step_ = checked_offgraph_step();
    }

    std::optional<SeqStepPlan> step;
    try {
      if (valid(offgraph_position_)) {
        bind_offgraph(offgraph_step_->rows);
      }
      if (options_.max_cached_graphs == 0) {
        GraphInstance instance = build_graph(evaluate_key_extents());
        check_host_integers(instance);
        grow_scheduler(scheduler_size(instance));
        step = prepare_offgraph(instance);
        try {
          activate(instance);
          run(instance);
        } catch (...) {
          release_allocation();
          throw;
        }
        release_allocation();
      } else {
        CachedGraph& cached = cached_graph();
        step = prepare_offgraph(cached.instance);
        try {
          activate(cached.instance);
          run(cached.instance);
        } catch (...) {
          release_allocation();
          cached_graphs_.pop_back();
          throw;
        }
      }
    } catch (...) {
      if (valid(offgraph_position_) && kv_cache_.cells != nullptr) {
        kv_cache_.cells->abandon_step();
      }
      throw;
    }
    if (step) {
      kv_cache_.sequence->commit(*step);
    }
    if (valid(offgraph_position_) && kv_cache_.cells != nullptr) {
      kv_cache_.cells->end_step();
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
      check_host_integers(cached_graphs_.back().instance);
      return cached_graphs_.back();
    }
    if (cached_graphs_.size() >= options_.max_cached_graphs) {
      if (state_.allocated == cached_graphs_.front().instance.graph) {
        release_allocation();
      }
      cached_graphs_.erase(cached_graphs_.begin());
    }
    GraphInstance instance = build_graph(key_extents);
    check_host_integers(instance);
    grow_scheduler(scheduler_size(instance));
    CachedGraph cached{
        .key_extents = key_extents, .instance = std::move(instance)};
    for (const InputSlot& input : inputs_) {
      cached.input_sizes.push_back(input.sizes);
    }
    cached_graphs_.push_back(std::move(cached));
    return cached_graphs_.back();
  }

  void refresh_inputs(GraphInstance& instance) const {
    for (const RowIndexCheck& check : instance.row_index_checks) {
      check_index_bytes(
          *std::ranges::find(inputs_, check.input_id, &InputSlot::id),
          check.rows);
    }
    for (const ValueId id : instance.exact_integer_checks) {
      check_exact_integer_bytes(
          *std::ranges::find(inputs_, id, &InputSlot::id));
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

  // The scheduler otherwise keeps graph inputs on the CPU backend and copies
  // them into GPU splits once per consuming view.
  void place_uploads(const GraphInstance& instance) const {
    for (const Upload& upload : instance.uploads) {
      ggml_backend_sched_set_tensor_backend(
          state_.scheduler.get(), upload.tensor, primary_backend_);
    }
  }

  // Allocates `instance` from the context's compute buffers unless it holds
  // their allocation already. The buffers are reserved for every method's
  // worst case, so this only lays the graph out in them, from a plan made for
  // it: ggml's allocator reuses the last graph's plan for any graph with as
  // many nodes and leafs that fit in it, whatever their views and lifetimes.
  void activate(const GraphInstance& instance) {
    ggml_backend_sched_t scheduler = state_.scheduler.get();
    if (state_.allocated == instance.graph) {
      return;
    }
    reserve_in_scheduler(instance);
    restore_built_state(instance);
    place_uploads(instance);
    if (!ggml_backend_sched_alloc_graph(scheduler, instance.graph)) {
      ggml_backend_sched_reset(scheduler);
      throw std::runtime_error("ggml: failed to allocate execution graph");
    }
    state_.allocated = instance.graph;
    track_compute_buffers();
  }

  void release_allocation() {
    ggml_backend_sched_synchronize(state_.scheduler.get());
    ggml_backend_sched_reset(state_.scheduler.get());
    state_.allocated = nullptr;
  }

  // Growing the scheduler reserves `instance` with the others.
  void reserve(const GraphInstance& instance) {
    if (!grow_scheduler(scheduler_size(instance))) {
      reserve_in_scheduler(instance);
    }
  }

  // Returns a graph to its state as built, for a scheduler that may have
  // split and allocated others since: sources as built, and no tensor
  // pointing into compute buffers that may since have been freed, whose
  // earlier layout ggml's allocator would otherwise keep for it.
  static void restore_built_state(const GraphInstance& instance) {
    for (int i = 0; i < ggml_graph_n_nodes(instance.graph); ++i) {
      std::ranges::copy(
          instance.sources.at(static_cast<size_t>(i)),
          ggml_graph_node(instance.graph, i)->src);
    }
    for (const auto& [tensor, data] : instance.unallocated) {
      tensor->buffer = nullptr;
      tensor->data = data;
    }
  }

  void reserve_in_scheduler(const GraphInstance& instance) {
    ggml_backend_sched_t scheduler = state_.scheduler.get();
    ggml_backend_sched_reset(scheduler);
    state_.allocated = nullptr;
    restore_built_state(instance);
    place_uploads(instance);
    if (!ggml_backend_sched_reserve(scheduler, instance.graph)) {
      throw std::runtime_error("ggml: failed to reserve compute buffers");
    }
    track_compute_buffers();
  }

  // A scheduler holds as many graph nodes and leafs as its size; a larger
  // graph needs a new one, which reserves every live method's worst case
  // again so that no later call grows its buffers. Returns whether it grew.
  bool grow_scheduler(size_t size) {
    if (size <= state_.scheduler_size) {
      return false;
    }
    if (state_.scheduler) {
      ggml_backend_sched_synchronize(state_.scheduler.get());
    }
    state_.allocated = nullptr;
    state_.scheduler = make_scheduler(size);
    state_.scheduler_size = size;
    state_.compute_buffer_bytes = 0;
    std::erase_if(
        state_.reserved, [](const auto& graph) { return graph.expired(); });
    for (const auto& reserved : state_.reserved) {
      if (const std::shared_ptr<const GraphInstance> graph = reserved.lock()) {
        reserve_in_scheduler(*graph);
      }
    }
    return true;
  }

  void track_compute_buffers() {
    size_t bytes = 0;
    // cppcheck-suppress-begin useStlAlgorithm ; the explicit loop reads clearer
    // here
    for (const ggml_backend_ptr& backend : state_.backends) {
      bytes += ggml_backend_sched_get_buffer_size(
          state_.scheduler.get(), backend.get());
    }
    // cppcheck-suppress-end useStlAlgorithm
    if (bytes > state_.compute_buffer_bytes) {
      ++state_.compute_buffer_allocations;
    }
    state_.compute_buffer_bytes = bytes;
  }

  void run(const GraphInstance& instance) {
    ggml_backend_sched_t scheduler = state_.scheduler.get();
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

  // The context's executables share its backends, created by the first.
  void initialize_backends() {
    const bool gpu = options_.device == GgmlDevice::Gpu;
    ggml_backend_dev_t device =
        gpu ? select_ggml_gpu_device(options_) : nullptr;
    const std::string_view registry =
        gpu ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)) : "";
    f32_accumulation_ =
        options_.cuda_f32_accumulation_for_testing && registry == "CUDA";
    f16_kv_caches_ = !options_.f32_gpu_kv_cache_for_testing &&
        (registry == "CUDA" || registry == "ROCm");
    gpu_device_ = device;
    // A failed setup publishes nothing, so a later compile starts it over.
    if (state_.backends.empty()) {
      std::vector<ggml_backend_ptr> backends;
      if (gpu) {
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        if (!backend) {
          throw std::runtime_error(
              std::string("ggml: failed to initialize ") +
              ggml_backend_dev_name(device));
        }
        backends.push_back(std::move(backend));
      }
      ggml_backend_ptr cpu(ggml_backend_cpu_init());
      if (!cpu) {
        throw std::runtime_error("ggml: failed to initialize CPU backend");
      }
      if (options_.cpu_threads > 0) {
        ggml_backend_cpu_set_n_threads(cpu.get(), options_.cpu_threads);
      }
      backends.push_back(std::move(cpu));
      state_.backends = std::move(backends);
      state_.primary_backend = state_.backends.front().get();
    }
    primary_backend_ = state_.primary_backend;
  }

  // The scheduler hashes a graph's nodes and leafs and splits it at most once
  // per node, and ggml mallocs its split context in proportion to this size,
  // so it is the built graph's count, not the graph's capacity; at least
  // twice graph_size, the configured room. A leaf is a source without an op,
  // or an output without one.
  size_t scheduler_size(const GraphInstance& instance) const {
    std::unordered_set<const ggml_tensor*> leafs;
    const int nodes = ggml_graph_n_nodes(instance.graph);
    for (int i = 0; i < nodes; ++i) {
      for (const ggml_tensor* source :
           ggml_graph_node(instance.graph, i)->src) {
        if (source != nullptr && source->op == GGML_OP_NONE) {
          leafs.insert(source);
        }
      }
    }
    for (const TensorRef& output : instance.outputs) {
      if (output.tensor->op == GGML_OP_NONE) {
        leafs.insert(output.tensor);
      }
    }
    return std::max(
        2 * options_.graph_size, static_cast<size_t>(nodes) + leafs.size());
  }

  ggml_backend_sched_ptr make_scheduler(size_t size) const {
    std::vector<ggml_backend_t> raw(state_.backends.size());
    std::ranges::transform(
        state_.backends, raw.begin(), [](const auto& backend) {
          return backend.get();
        });
    ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(
        raw.data(),
        nullptr,
        static_cast<int>(raw.size()),
        size,
        /*parallel=*/false,
        /*op_offload=*/true));
    if (!scheduler) {
      throw std::runtime_error("ggml: failed to create backend scheduler");
    }
    return scheduler;
  }

  // Quantization parameters are bound but only read through their weight.
  bool is_used(ValueId id) const {
    return is_used(method_, id);
  }

  static bool is_used(const Method& method, ValueId id) {
    const std::vector<ValueId>& outputs = method.graph.output_ids;
    return !method.graph.value(id).consumer_ids.empty() ||
        std::find(outputs.begin(), outputs.end(), id) != outputs.end();
  }

  bool is_persistent(const DataBinding& binding) const {
    return valid(binding.value_id) &&
        method_.graph.value(binding.value_id).is_tensor() &&
        is_used(binding.value_id);
  }

  // Whether the GPU runs this attention over F16 keys and values. The SDPA
  // lowering's shape checks come first: ggml asserts on the shapes they
  // refuse, which then load as F32 caches and fail in the lowering.
  bool supports_f16_attention(
      const Method& method,
      const Node& node,
      bool custom) const {
    const auto sizes = [&](std::string_view name, size_t position) {
      return method.graph
          .value(require_arg(node, name, position).arg.as_tensor().id)
          .tensor_meta()
          .sizes;
    };
    const std::vector<int64_t> q = sizes("query", 0);
    const std::vector<int64_t> k = sizes("key", 1);
    const std::vector<int64_t> v = sizes("value", 2);
    if (custom ? !custom_sdpa_shapes_compatible(q, k, v)
               : !sdpa_shapes_compatible(q, k, v)) {
      return false;
    }
    if (options_.gpu_runs_f16_attention_for_testing) {
      return options_.gpu_runs_f16_attention_for_testing();
    }
    return gpu_runs_f16_attention(method, node, custom);
  }

  // A zero-initialized F32 buffer on a backend that keeps KV caches in F16,
  // whose every use writes rows into it or reads it as attention keys or
  // values, in every method of the context that uses it, since they share
  // its storage; see GgmlEngineOptions::f32_gpu_kv_cache_for_testing.
  bool stores_f16_kv_cache(const DataBinding& binding) const {
    if (!stores_f16_kv_cache(method_, binding)) {
      return false;
    }
    for (const Method* method : state_.methods) {
      if (method == &method_) {
        continue;
      }
      // cppcheck-suppress-begin useStlAlgorithm ; the explicit loop reads
      // clearer here
      for (const DataBinding& other : method->data_bindings) {
        if (other.key == binding.key && valid(other.value_id) &&
            method->graph.value(other.value_id).is_tensor() &&
            is_used(*method, other.value_id) &&
            !stores_f16_kv_cache(*method, other)) {
          return false;
        }
      }
      // cppcheck-suppress-end useStlAlgorithm
    }
    return true;
  }

  bool stores_f16_kv_cache(const Method& method, const DataBinding& binding)
      const {
    const bool f16_kv_caches = options_.gpu_runs_f16_attention_for_testing
        ? !options_.f32_gpu_kv_cache_for_testing
        : f16_kv_caches_;
    if (!f16_kv_caches || binding.has_data) {
      return false;
    }
    const TensorMeta& meta = method.graph.value(binding.value_id).tensor_meta();
    return meta.dtype == ScalarType::Float && !meta.quant &&
        only_f16_gpu_attention_reads(method, binding.value_id);
  }

  // Whether every use of `id`, followed through views and the results of row
  // writes into it, writes rows into it or reads it as attention keys or
  // values, at least one attention reads it, and the GPU runs each of those
  // attentions over F16 keys and values. A node that uses `id` more times
  // than it has those roles also reads it some other way.
  bool only_f16_gpu_attention_reads(const Method& method, ValueId id) const {
    static const std::unordered_set<std::string> kViews = {
        "aten._unsafe_view.default",
        "aten.alias.default",
        "aten.permute.default",
        "aten.permute_copy.default",
        "aten.reshape.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.dims",
        "aten.squeeze_copy.dims",
        "aten.transpose_copy.int",
        "aten.unsqueeze.default",
        "aten.unsqueeze_copy.default",
        "aten.view.default",
        "aten.view_copy.default",
    };
    std::vector<ValueId> pending = {id};
    std::unordered_set<ValueId> seen;
    bool attended = false;
    while (!pending.empty()) {
      const ValueId value = pending.back();
      pending.pop_back();
      if (!seen.insert(value).second) {
        continue;
      }
      const std::vector<ValueId>& outputs = method.graph.output_ids;
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == value &&
            method.output_specs[i].kind != OutputKind::BufferMutation) {
          return false;
        }
      }
      for (const NodeId consumer : method.graph.value(value).consumer_ids) {
        const Node& node = method.graph.node(consumer);
        if (node.is_output()) {
          continue;
        }
        const std::string op = node.is_call() ? op_key(node.target) : "";
        const auto is_arg = [&](std::string_view name, size_t position) {
          const NamedArgument* arg = find_arg(node, name, position);
          return arg != nullptr && arg->arg.kind() == ArgKind::Tensor &&
              arg->arg.as_tensor().id == value;
        };
        const bool attention =
            op == "aten.scaled_dot_product_attention.default" ||
            op == "llama.custom_sdpa.default";
        size_t roles = 0;
        if ((op == "aten.index_put_.default" && is_arg("self", 0)) ||
            (op == "llama.update_cache.default" && is_arg("cache", 1)) ||
            (kViews.contains(op) && is_arg("self", 0))) {
          ++roles;
          // cppcheck-suppress-begin useStlAlgorithm ; the explicit loop reads
          // clearer here
          for (const ValueId result : node.output_value_ids()) {
            pending.push_back(result);
          }
          // cppcheck-suppress-end useStlAlgorithm
        }
        if (attention && (is_arg("key", 1) || is_arg("value", 2))) {
          if (!supports_f16_attention(
                  method, node, op == "llama.custom_sdpa.default")) {
            return false;
          }
          roles += (is_arg("key", 1) ? 1 : 0) + (is_arg("value", 2) ? 1 : 0);
          attended = true;
        }
        const auto uses = std::ranges::count(node.input_value_ids(), value);
        if (roles == 0 || static_cast<size_t>(uses) != roles) {
          return false;
        }
      }
    }
    return attended;
  }

  // Whether the GPU runs this attention's flash attention, for one query and
  // for all of them, over F16 keys and values, with the mask the lowering
  // passes: some kernels need one (ggml-cuda's head size 512, say), and some
  // take only one mask plane (ggml-cuda's, again). ggml leaves an attention
  // the GPU lacks a kernel for to the CPU, whose kernel reads F32 keys and
  // values without rounding them; its caches stay F32.
  bool gpu_runs_f16_attention(
      const Method& method,
      const Node& node,
      bool custom) const {
    const auto meta = [&](std::string_view name, size_t position) {
      return method.graph
          .value(require_arg(node, name, position).arg.as_tensor().id)
          .tensor_meta()
          .sizes;
    };
    const std::vector<int64_t> q = meta("query", 0);
    const std::vector<int64_t> k = meta("key", 1);
    const std::vector<int64_t> v = meta("value", 2);
    if (q.size() != 4 || k.size() != 4 || v.size() != 4) {
      return false;
    }
    const NamedArgument* mask = find_arg(node, "attn_mask", custom ? 4 : 3);
    const bool explicit_mask = mask != nullptr &&
        mask->arg.kind() == ArgKind::Tensor && valid(mask->arg.as_tensor().id);
    // ggml's mask is [S, T, mask heads, mask batch]: the explicit mask's dims
    // before its last two, which must divide the query's heads and batch.
    int64_t mask_heads = 1;
    int64_t mask_batch = 1;
    if (explicit_mask) {
      const std::vector<int64_t>& m =
          method.graph.value(mask->arg.as_tensor().id).tensor_meta().sizes;
      if (m.size() < 2 || m.size() > 4) {
        return false;
      }
      mask_heads = m.size() > 2 ? m[m.size() - 3] : 1;
      mask_batch = m.size() > 3 ? m[0] : 1;
    }
    const NamedArgument* causal = find_arg(node, "is_causal", custom ? 6 : 5);
    const bool is_causal = causal != nullptr &&
        causal->arg.kind() == ArgKind::Bool && causal->arg.as_bool().value;
    // [B, H, T, D] for aten SDPA; [B, T, H, D] for custom_sdpa, which always
    // passes a mask, and without an explicit one attends start_pos + T keys,
    // any count up to the cache's.
    const size_t heads = custom ? 2 : 1;
    const size_t tokens = custom ? 1 : 2;
    const bool masked = custom || explicit_mask || is_causal;
    if (mask_heads <= 0 || mask_batch <= 0 || q[heads] % mask_heads != 0 ||
        q[0] % mask_batch != 0) {
      return false;
    }
    std::vector<int64_t> key_counts = {k[tokens]};
    if (custom && !explicit_mask) {
      key_counts.push_back(1);
    }
    const ggml_init_params params{
        .mem_size = 24 * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    const ggml_context_ptr ctx(ggml_init(params));
    for (const int64_t rows : {int64_t{1}, q[tokens]}) {
      for (const int64_t keys : key_counts) {
        ggml_tensor* attention = ggml_flash_attn_ext(
            ctx.get(),
            ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_F32, q[3], rows, q[heads], q[0]),
            ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_F16, k[3], keys, k[heads], k[0]),
            ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_F16, v[3], keys, v[heads], v[0]),
            masked ? ggml_new_tensor_4d(
                         ctx.get(),
                         GGML_TYPE_F16,
                         keys,
                         rows,
                         mask_heads,
                         mask_batch)
                   : nullptr,
            1.0f,
            0.0f,
            0.0f);
        ggml_prec_set_acc(attention, GGML_PREC_F32);
        if (!ggml_backend_dev_supports_op(gpu_device_, attention)) {
          return false;
        }
      }
    }
    return true;
  }

  // Bound to storage the context shares, not to a private constant.
  bool is_state(const DataBinding& binding) const {
    return state_.keys.contains(binding.key);
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
    if (method_.graph.value(id).tensor_meta().ndim() == 3) {
      check_expert_consumers(id, key);
      return;
    }
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
      if (op == "aten.linear.default" &&
          !options_.allow_activation_quant_for_weight_only &&
          !dequantizes_to_f32(id) && !has_dequantized_input(node)) {
        throw std::runtime_error(
            "ggml: weight-only quantized linear '" + node.name +
            "' shares its weight with a linear that quantizes its own "
            "activations, so it needs allow_activation_quant_for_weight_only");
      }
    }
  }

  // Without activation quantization: a quantized weight that a matmul reads,
  // a linear or an expert gather, unless a linear reading it quantizes its
  // own activations. Embeddings gather rows exactly from either form.
  bool dequantizes_to_f32(ValueId id) const {
    const TensorMeta& meta = method_.graph.value(id).tensor_meta();
    if (options_.allow_activation_quant_for_weight_only ||
        (packed_quant(meta) == nullptr && affine_quant(meta) == nullptr)) {
      return false;
    }
    bool matmul = false;
    for (const NodeId consumer : method_.graph.value(id).consumer_ids) {
      const Node& node = method_.graph.node(consumer);
      const std::string op = op_key(node.target);
      if (op == "aten.linear.default") {
        if (has_dequantized_input(node)) {
          return false;
        }
        matmul = true;
      } else if (expert_matmul(method_, node) != nullptr) {
        matmul = true;
      }
    }
    return matmul;
  }

  // Quantized experts are only read by expert gathers, whose bmm runs as
  // ggml_mul_mat_id and, like a linear, quantizes its activations.
  void check_expert_consumers(ValueId id, const std::string& key) const {
    for (const NodeId consumer : method_.graph.value(id).consumer_ids) {
      const Node& node = method_.graph.node(consumer);
      const NamedArgument* self = find_arg(node, "self", 0);
      if (self == nullptr || self->arg.kind() != ArgKind::Tensor ||
          self->arg.as_tensor().id != id ||
          expert_matmul(method_, node) == nullptr) {
        throw std::runtime_error(
            "ggml: 3-D quantized weight '" + key +
            "' may only feed expert gathers (aten.index.Tensor on dim 0 read "
            "only by bmm's weight side), not '" +
            node.target + "'");
      }
    }
  }

  bool has_dequantized_input(const Node& linear) const {
    const NamedArgument* input = find_arg(linear, "input", 0);
    if (input == nullptr || input->arg.kind() != ArgKind::Tensor) {
      return false;
    }
    const NodeId producer =
        method_.graph.value(input->arg.as_tensor().id).producer_id;
    return valid(producer) &&
        op_key(method_.graph.node(producer).target) ==
        "torchao.dequantize_affine.default";
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

  // Scales that fp16 would round, of an int4 constant only embeddings read.
  // Q4_0 blocks hold fp16 scales, and a linear's mul_mat reads them there, so
  // only such a weight loads as unit-scale blocks whose gathered rows are
  // multiplied by these scales.
  std::optional<std::vector<float>> exact_row_scales(ValueId id) const {
    const TensorMeta& meta = method_.graph.value(id).tensor_meta();
    const AffineGroupQuant* quant = affine_quant(meta);
    if (quant == nullptr || quant->scale_dtype == ScalarType::Half ||
        constant_binding(id) == nullptr ||
        !std::ranges::all_of(
            method_.graph.value(id).consumer_ids, [this](NodeId consumer) {
              return op_key(method_.graph.node(consumer).target) ==
                  "aten.embedding.default";
            })) {
      return std::nullopt;
    }
    std::vector<float> scales = load_q4_scales(meta);
    if (std::ranges::all_of(scales, [](float scale) {
          return ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale)) == scale;
        })) {
      return std::nullopt;
    }
    return scales;
  }

  std::vector<float> load_q4_scales(const TensorMeta& meta) const {
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
    return to_float_vector(scales->span(), quant.scale_dtype);
  }

  // The binding of a constant with data that no method writes.
  const DataBinding* constant_binding(ValueId id) const {
    const auto binding = std::ranges::find_if(
        method_.data_bindings, [id](const DataBinding& candidate) {
          return candidate.value_id == id && candidate.has_data;
        });
    if (binding == method_.data_bindings.end() || is_state(*binding) ||
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

  using LoadPlan = GgmlLoadPlan;

  // The package key that owns `key`'s bytes: methods of one package name the
  // same bytes by different data keys.
  std::string owner_key(const std::string& key) const {
    const std::optional<ConstantInfo> info = package_.constant_info(key);
    return info ? info->owner : key;
  }

  LoadPlan plan_load(
      const DataBinding& binding,
      std::optional<std::vector<float>> row_scales) const {
    const TensorMeta& meta =
        method_.graph.value(binding.value_id).tensor_meta();
    LoadPlan plan{
        .source = owner_key(binding.key),
        .buffer_type = ggml_backend_buft_name(
            ggml_backend_get_default_buffer_type(primary_backend_)),
        .type = compute_type(meta),
        .sizes = meta.sizes,
    };
    const auto record = [&plan](const std::string& transform) {
      plan.transforms += '|' + transform;
    };
    if (const PackedQuant* packed = packed_quant(meta)) {
      record("packed:" + packed->codec);
    }
    const AffineGroupQuant* quant = affine_quant(meta);
    if (quant != nullptr) {
      record(
          "int4:" + owner_key(quant->scale_data_key) + ',' +
          owner_key(quant->zero_point_data_key) + ',' +
          std::to_string(quant->group_size));
    }
    if (dequantizes_to_f32(binding.value_id)) {
      plan.type = GGML_TYPE_F32;
      plan.dequant = true;
      record("dequant_f32");
    } else if (quant != nullptr) {
      plan.q4 = true;
      record("q4_0");
      if (row_scales) {
        plan.row_scales = std::move(row_scales);
        record("row_scales");
      }
    }
    if (plan.type == GGML_TYPE_I32 && meta.dtype == ScalarType::Long) {
      plan.narrow = true;
      record("i64_to_i32");
    }
    if (const auto scales = weight_scales_.find(binding.value_id);
        scales != weight_scales_.end()) {
      plan.scales = &scales->second;
      std::string factors(scales->second.size() * sizeof(float), '\0');
      std::memcpy(factors.data(), scales->second.data(), factors.size());
      record("scaled:" + factors);
    }
    return plan;
  }

  // A tensor bound by key must be the one this plan would have created.
  static void check_shared_constant(
      const ggml_tensor* live,
      const LoadPlan& recorded,
      const LoadPlan& plan) {
    const std::array<int64_t, GGML_MAX_DIMS> ne = reversed_extents(plan.sizes);
    const size_t nbytes = ggml_row_size(plan.type, ne[0]) *
        static_cast<size_t>(ne[1] * ne[2] * ne[3]);
    if (!(recorded == plan) || live->type != plan.type ||
        !std::equal(ne.begin(), ne.end(), live->ne) ||
        !ggml_is_contiguous(live) || ggml_nbytes(live) != nbytes ||
        plan.buffer_type !=
            ggml_backend_buft_name(
                ggml_backend_buffer_get_type(live->buffer))) {
      throw std::logic_error(
          "ggml: shared constant '" + plan.source +
          "' does not match the tensor its key names");
    }
  }

  // Binds each constant to the device tensor the context holds for its plan,
  // allocating tensors, still to be loaded, for plans no live executable
  // holds. Returns the values bound to loaded tensors.
  std::unordered_set<ValueId> bind_constants(
      const std::unordered_map<ValueId, LoadPlan>& plans) {
    static constexpr std::string_view kRowScales = "|row_scales";
    std::unordered_set<ValueId> loaded;
    std::unordered_map<std::string, ggml_tensor*> fresh;
    std::shared_ptr<StateBlock> block;
    for (const DataBinding& binding : method_.data_bindings) {
      if (!is_persistent(binding) || is_state(binding)) {
        continue;
      }
      const LoadPlan& plan = plans.at(binding.value_id);
      const std::string key = plan.key();
      const std::string scales_key = key + std::string(kRowScales);
      GgmlContextState::Constant& shared = state_.constants[key];
      if (std::shared_ptr<ggml_tensor> live = shared.tensor.lock()) {
        check_shared_constant(live.get(), *shared.plan, plan);
        persistent_.emplace(binding.value_id, live.get());
        if (plan.row_scales) {
          std::shared_ptr<ggml_tensor> scales =
              state_.constants.at(scales_key).tensor.lock();
          row_scales_.emplace(binding.value_id, scales.get());
          state_storage_.push_back(std::move(scales));
        }
        loaded.insert(binding.value_id);
        state_storage_.push_back(std::move(live));
        continue;
      }
      if (const auto created = fresh.find(key); created != fresh.end()) {
        persistent_.emplace(binding.value_id, created->second);
        if (plan.row_scales) {
          row_scales_.emplace(binding.value_id, fresh.at(scales_key));
        }
        loaded.insert(binding.value_id);
        continue;
      }
      if (!block) {
        block = std::make_shared<StateBlock>();
        ggml_init_params params{
            .mem_size =
                ggml_tensor_overhead() * (2 * method_.data_bindings.size() + 1),
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        block->context.reset(ggml_init(params));
        if (!block->context) {
          throw std::runtime_error(
              "ggml: failed to create persistent tensor context");
        }
      }
      ggml_tensor* tensor =
          new_tensor(block->context.get(), plan.type, plan.sizes);
      ggml_set_name(tensor, binding.key.c_str());
      persistent_.emplace(binding.value_id, tensor);
      fresh.emplace(key, tensor);
      auto recorded = std::make_shared<LoadPlan>(plan);
      recorded->row_scales.reset();
      recorded->scales = nullptr;
      shared.plan = std::move(recorded);
      if (plan.row_scales) {
        const AffineGroupQuant& quant =
            q4_affine(method_.graph.value(binding.value_id).tensor_meta());
        ggml_tensor* scales = new_tensor(
            block->context.get(),
            GGML_TYPE_F32,
            {plan.sizes[0], plan.sizes[1] / quant.group_size});
        ggml_set_name(scales, quant.scale_data_key.c_str());
        row_scales_.emplace(binding.value_id, scales);
        fresh.emplace(scales_key, scales);
      }
    }
    if (!block) {
      return loaded;
    }
    block->buffer.reset(
        ggml_backend_alloc_ctx_tensors(block->context.get(), primary_backend_));
    if (!block->buffer) {
      throw std::runtime_error("ggml: failed to allocate persistent tensors");
    }
    ggml_backend_buffer_set_usage(
        block->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [key, tensor] : fresh) {
      std::shared_ptr<ggml_tensor> storage(block, tensor);
      state_.constants[key].tensor = storage;
      state_storage_.push_back(std::move(storage));
    }
    return loaded;
  }

  // Binds each mutable buffer key to the storage the context holds for it,
  // allocating storage, still to be initialized, for keys it does not hold
  // yet. Returns the values bound to storage that is already initialized.
  // Methods sharing a KV cache must agree on storing it in F16.
  std::unordered_set<ValueId> bind_state(size_t* f16_kv_cache_bytes) {
    std::unordered_set<ValueId> initialized;
    std::unordered_map<std::string, ggml_tensor*> fresh;
    std::shared_ptr<StateBlock> block;
    for (const DataBinding& binding : method_.data_bindings) {
      if (!is_persistent(binding) || !is_state(binding)) {
        continue;
      }
      if (const auto held = state_.storage.find(binding.key);
          held != state_.storage.end()) {
        const std::shared_ptr<ggml_tensor>& live = held->second;
        persistent_.emplace(binding.value_id, live.get());
        initialized.insert(binding.value_id);
        state_storage_.push_back(live);
        continue;
      }
      if (const auto created = fresh.find(binding.key);
          created != fresh.end()) {
        persistent_.emplace(binding.value_id, created->second);
        initialized.insert(binding.value_id);
        continue;
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      const bool f16_kv_cache = stores_f16_kv_cache(binding);
      const ggml_type type = f16_kv_cache ? GGML_TYPE_F16 : compute_type(meta);
      if (!block) {
        block = std::make_shared<StateBlock>();
        ggml_init_params params{
            .mem_size =
                ggml_tensor_overhead() * (method_.data_bindings.size() + 1),
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        block->context.reset(ggml_init(params));
        if (!block->context) {
          throw std::runtime_error("ggml: failed to create state context");
        }
      }
      ggml_tensor* tensor = new_tensor(block->context.get(), type, meta.sizes);
      if (f16_kv_cache && f16_kv_cache_bytes != nullptr) {
        *f16_kv_cache_bytes += ggml_nbytes(tensor);
      }
      ggml_set_name(tensor, binding.key.c_str());
      persistent_.emplace(binding.value_id, tensor);
      fresh.emplace(binding.key, tensor);
    }
    if (!block) {
      return initialized;
    }
    block->buffer.reset(
        ggml_backend_alloc_ctx_tensors(block->context.get(), primary_backend_));
    if (!block->buffer) {
      throw std::runtime_error("ggml: failed to allocate mutable buffers");
    }
    ggml_backend_buffer_set_usage(
        block->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [key, tensor] : fresh) {
      std::shared_ptr<ggml_tensor> storage(block, tensor);
      state_storage_.push_back(storage);
      fresh_state_.emplace_back(key, std::move(storage));
    }
    return initialized;
  }

  void initialize_persistent_tensors(
      size_t* materialized_constant_bytes,
      size_t* f16_kv_cache_bytes) {
    std::unordered_map<ValueId, LoadPlan> plans;
    for (const DataBinding& binding : method_.data_bindings) {
      if (!is_persistent(binding)) {
        continue;
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      if (affine_quant(meta) != nullptr || packed_quant(meta) != nullptr) {
        check_q4_consumers(binding.value_id, binding.key);
      }
      LoadPlan plan = plan_load(binding, exact_row_scales(binding.value_id));
      constant_keys_.emplace(binding.value_id, plan.key());
      plans.emplace(binding.value_id, std::move(plan));
    }
    std::unordered_set<ValueId> initialized = bind_constants(plans);
    initialized.merge(bind_state(f16_kv_cache_bytes));

    size_t loaded = 0;
    for (const DataBinding& binding : method_.data_bindings) {
      const auto found = persistent_.find(binding.value_id);
      if (found == persistent_.end() ||
          initialized.contains(binding.value_id)) {
        continue;
      }
      ggml_tensor* tensor = found->second;
      const bool host_buffer = ggml_backend_buffer_is_host(tensor->buffer);
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
      const LoadPlan& plan = plans.at(binding.value_id);
      if (!plan.q4 && !plan.narrow && !plan.dequant && host_buffer) {
        if (ggml_nbytes(tensor) != expected) {
          throw std::runtime_error(
              "ggml: stored byte count disagrees with ggml tensor '" +
              binding.key + "'");
        }
        package_.load_constant_into(
            binding.key,
            MutableByteSpan(static_cast<uint8_t*>(tensor->data), expected));
        if (plan.scales != nullptr) {
          scale_slices(tensor->data, expected / sizeof(float), *plan.scales);
        }
        loaded += expected;
        continue;
      }
      const std::optional<OwnedBytes> bytes =
          plan.q4 || plan.narrow || plan.dequant
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
      if (plan.q4) {
        if (!plan.row_scales) {
          converted = affine_int4_to_q4_0(meta, span, load_q4_scales(meta));
        } else {
          converted = affine_int4_to_q4_0(
              meta, span, std::vector<float>(plan.row_scales->size(), 1.0f));
          ggml_tensor* row_scales = row_scales_.at(binding.value_id);
          ggml_backend_tensor_set(
              row_scales, plan.row_scales->data(), 0, ggml_nbytes(row_scales));
          loaded += ggml_nbytes(row_scales);
        }
        span = converted;
      } else if (plan.narrow) {
        converted = narrow_to_int32(span);
        span = converted;
      } else if (plan.dequant && affine_quant(meta) != nullptr) {
        converted = affine_int4_to_f32(meta, span, load_q4_scales(meta));
        span = converted;
      } else if (plan.dequant) {
        const int64_t numel = ggml_nelements(tensor);
        converted.resize(static_cast<size_t>(numel) * sizeof(float));
        ggml_get_type_traits(compute_type(meta))
            ->to_float(
                // cppcheck-suppress invalidPointerCast ; the byte buffer is
                // sized for floats and only accessed as float
                span.data(),
                // cppcheck-suppress invalidPointerCast ; the byte buffer is
                // sized for floats and only accessed as float
                reinterpret_cast<float*>(converted.data()),
                numel);
        span = converted;
      } else if (plan.scales != nullptr) {
        converted.assign(span.begin(), span.end());
        scale_slices(
            converted.data(), converted.size() / sizeof(float), *plan.scales);
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
    if (materialized_constant_bytes != nullptr) {
      *materialized_constant_bytes += loaded;
    }
  }

  // Ops that compute a pure function of their tensor arguments, which the
  // executable may run once when all of those are constants.
  static bool is_foldable_op(const std::string& key) {
    static const std::unordered_set<std::string> kOps = {
        "aten._to_copy.default",     "aten._unsafe_view.default",
        "aten.abs.default",          "aten.add.Tensor",
        "aten.alias.default",        "aten.clamp.default",
        "aten.clone.default",        "aten.contiguous.default",
        "aten.cos.default",          "aten.div.Tensor",
        "aten.exp.default",          "aten.expand.default",
        "aten.expand_copy.default",  "aten.hardtanh.default",
        "aten.log.default",          "aten.mul.Tensor",
        "aten.neg.default",          "aten.permute.default",
        "aten.permute_copy.default", "aten.relu.default",
        "aten.reshape.default",      "aten.select.int",
        "aten.select_copy.int",      "aten.sigmoid.default",
        "aten.silu.default",         "aten.sin.default",
        "aten.slice.Tensor",         "aten.slice_copy.Tensor",
        "aten.sqrt.default",         "aten.squeeze.dims",
        "aten.squeeze_copy.dims",    "aten.sub.Tensor",
        "aten.tanh.default",         "aten.transpose_copy.int",
        "aten.unsqueeze.default",    "aten.unsqueeze_copy.default",
        "aten.view.default",         "aten.view_copy.default",
    };
    return kOps.contains(key);
  }

  // Unquantized constants no method writes: packed and int4 weights stay as
  // they are for mul_mat, and state may change between calls.
  std::unordered_set<ValueId> immutable_constants() const {
    std::unordered_set<ValueId> constants;
    for (const DataBinding& binding : method_.data_bindings) {
      if (persistent_.contains(binding.value_id) && !binding.mutated &&
          constant_binding(binding.value_id) != nullptr &&
          !method_.graph.value(binding.value_id).tensor_meta().quant) {
        constants.insert(binding.value_id);
      }
    }
    return constants;
  }

  // Whether `node` computes a float result no larger than its inputs (or
  // small), from `constants` alone, with static arguments, writing nothing.
  bool is_foldable(
      const Node& node,
      const std::unordered_set<ValueId>& constants,
      const std::unordered_set<ValueId>& excluded) const {
    constexpr size_t kSmallResultBytes = size_t{1} << 20;
    const std::vector<ValueId> outputs = output_ids(node);
    if (!node.is_call() || !is_foldable_op(op_key(node.target)) ||
        outputs.size() != 1 || excluded.contains(outputs.front())) {
      return false;
    }
    const TensorMeta& meta = method_.graph.value(outputs.front()).tensor_meta();
    if ((meta.dtype != ScalarType::Float && meta.dtype != ScalarType::Half &&
         meta.dtype != ScalarType::BFloat16) ||
        !meta.lower_bounds.empty() || checked_numel(meta.sizes) == 0) {
      return false;
    }
    size_t input_bytes = 0;
    for (const NamedArgument& input : node.inputs) {
      const Argument& arg = input.arg;
      const bool symbolic =
          (arg.kind() == ArgKind::Int && valid(arg.as_int().id)) ||
          (arg.kind() == ArgKind::Float && valid(arg.as_float().id)) ||
          (arg.kind() == ArgKind::Bool && valid(arg.as_bool().id)) ||
          (arg.kind() == ArgKind::IntList &&
           std::ranges::any_of(arg.as_int_list().ids, valid));
      if (input.mutated || symbolic || arg.kind() == ArgKind::TensorList ||
          arg.kind() == ArgKind::OptionalTensorList ||
          arg.kind() == ArgKind::Graph) {
        return false;
      }
      if (arg.kind() == ArgKind::Tensor) {
        const ValueId id = arg.as_tensor().id;
        if (!constants.contains(id)) {
          return false;
        }
        const TensorMeta& source = method_.graph.value(id).tensor_meta();
        input_bytes += checked_nbytes(source.sizes, source.dtype);
      }
    }
    const size_t bytes = checked_nbytes(meta.sizes, meta.dtype);
    const auto writes_result = [&](NodeId consumer) {
      return std::ranges::any_of(
          method_.graph.node(consumer).inputs, [&](const NamedArgument& in) {
            return in.mutated && in.arg.kind() == ArgKind::Tensor &&
                in.arg.as_tensor().id == outputs.front();
          });
    };
    return input_bytes > 0 &&
        (bytes <= input_bytes || bytes <= kSmallResultBytes) &&
        std::ranges::none_of(
               method_.graph.value(outputs.front()).consumer_ids,
               writes_result);
  }

  // Computes once the nodes that read only immutable constants (directly or
  // through other such nodes), in their own lowering, so every call's graph
  // reads the same bits it would have computed. Results that other nodes read
  // join the persistent tensors, shared by the context's executables that
  // compute the same result; graph outputs and attention-mask chains, whose
  // lowering depends on the call, are left alone.
  void fold_constants(size_t* materialized_constant_bytes) {
    if (!options_.fold_constants) {
      return;
    }
    std::unordered_set<ValueId> constants = immutable_constants();
    std::unordered_set<ValueId> excluded(
        method_.graph.output_ids.begin(), method_.graph.output_ids.end());
    for (const MaskExtent& mask : mask_extents_) {
      excluded.insert(mask.chain.begin(), mask.chain.end());
    }
    std::vector<NodeId> nodes;
    std::unordered_set<NodeId> folded;
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (is_foldable(node, constants, excluded)) {
        nodes.push_back(id);
        folded.insert(id);
        constants.insert(output_ids(node).front());
      }
    }
    std::vector<ValueId> results;
    for (const NodeId id : nodes) {
      const ValueId result = output_ids(method_.graph.node(id)).front();
      if (std::ranges::any_of(
              method_.graph.value(result).consumer_ids,
              [&](NodeId consumer) { return !folded.contains(consumer); })) {
        results.push_back(result);
      }
    }
    if (results.empty()) {
      return;
    }
    const std::unordered_map<ValueId, std::string> keys = fold_keys(nodes);
    std::vector<ValueId> missing;
    std::unordered_map<std::string, ValueId> first;
    for (const ValueId result : results) {
      const std::string& key = keys.at(result);
      if (std::shared_ptr<ggml_tensor> live =
              state_.constants[key].tensor.lock()) {
        check_shared_fold(live.get(), result);
        persistent_.emplace(result, live.get());
        state_storage_.push_back(std::move(live));
      } else if (first.try_emplace(key, result).second) {
        missing.push_back(result);
      }
    }
    if (missing.empty()) {
      return;
    }
    FoldGraph fold = graph_builder({}).build_folds(nodes, missing);
    if (!fold.values.empty()) {
      store_folds(fold, keys, materialized_constant_bytes);
    }
    // Results equal to another of this method's share its tensor.
    for (const ValueId result : results) {
      const auto computed = first.find(keys.at(result));
      if (persistent_.contains(result) || computed == first.end()) {
        continue;
      }
      if (const auto shared = persistent_.find(computed->second);
          shared != persistent_.end()) {
        persistent_.emplace(result, shared->second);
      }
    }
  }

  // The context-wide identity of each folded node's result: its op chain, as
  // op names and static arguments down to the storage keys of the constants
  // it reads, and its declared dtype and shape. Executables of one context
  // lower ops alike, so equal keys compute equal bits.
  std::unordered_map<ValueId, std::string> fold_keys(
      const std::vector<NodeId>& nodes) const {
    std::unordered_map<ValueId, std::string> keys;
    const auto tensor_key = [&](ValueId id) {
      if (const auto found = keys.find(id); found != keys.end()) {
        return found->second;
      }
      return "c(" + constant_keys_.at(id) + ')';
    };
    for (const NodeId id : nodes) {
      const Node& node = method_.graph.node(id);
      std::string key = op_key(node.target) + '(';
      for (const NamedArgument& input : node.inputs) {
        key += input.name + '=' +
            (input.arg.kind() == ArgKind::Tensor
                 ? tensor_key(input.arg.as_tensor().id)
                 : static_argument_key(input.arg)) +
            ';';
      }
      const ValueId result = output_ids(node).front();
      const TensorMeta& meta = method_.graph.value(result).tensor_meta();
      key += ")->" + std::string(scalar_type_name(meta.dtype));
      for (const int64_t size : meta.sizes) {
        key += ',' + std::to_string(size);
      }
      keys.emplace(result, std::move(key));
    }
    return keys;
  }

  // A folded result bound by key must have the shape it computes.
  void check_shared_fold(const ggml_tensor* live, ValueId result) const {
    const std::array<int64_t, GGML_MAX_DIMS> ne =
        reversed_extents(method_.graph.value(result).tensor_meta().sizes);
    if (!std::equal(ne.begin(), ne.end(), live->ne)) {
      throw std::logic_error(
          "ggml: shared constant result '" + method_.graph.value(result).name +
          "' does not match the tensor its key names");
    }
  }

  // A non-tensor argument of a folded node, exactly: is_foldable admits only
  // static ones.
  static std::string static_argument_key(const Argument& arg) {
    const auto exact = [](double value) {
      char text[32];
      const int size = std::snprintf(text, sizeof(text), "%a", value);
      return std::string(text, static_cast<size_t>(size));
    };
    std::string key = std::to_string(static_cast<int>(arg.kind())) + ':';
    switch (arg.kind()) {
      case ArgKind::Int:
        return key + std::to_string(arg.as_int().value);
      case ArgKind::Float:
        return key + exact(arg.as_float().value);
      case ArgKind::Bool:
        return key + (arg.as_bool().value ? "1" : "0");
      case ArgKind::String:
        return key + arg.as_string().value;
      case ArgKind::ScalarType:
        return key +
            std::to_string(static_cast<int>(arg.as_scalar_type().value));
      case ArgKind::IntList:
        for (const int64_t value : arg.as_int_list().values) {
          key += std::to_string(value) + ',';
        }
        return key;
      case ArgKind::FloatList:
        for (const double value : arg.as_float_list().values) {
          key += exact(value) + ',';
        }
        return key;
      case ArgKind::BoolList:
        for (const bool value : arg.as_bool_list().values) {
          key += value ? '1' : '0';
        }
        return key;
      default:
        return key;
    }
  }

  // Copies the computed results into tensors the context holds under their
  // keys, beside its constants.
  void store_folds(
      const FoldGraph& fold,
      const std::unordered_map<ValueId, std::string>& keys,
      size_t* materialized_constant_bytes) {
    ggml_backend_sched_ptr scheduler =
        make_scheduler(scheduler_size(fold.instance));
    if (!ggml_backend_sched_alloc_graph(scheduler.get(), fold.instance.graph) ||
        ggml_backend_sched_graph_compute(
            scheduler.get(), fold.instance.graph) != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("ggml: failed to compute constant results");
    }
    auto block = std::make_shared<StateBlock>();
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * fold.values.size(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    block->context.reset(ggml_init(params));
    if (!block->context) {
      throw std::runtime_error(
          "ggml: failed to create constant result context");
    }
    std::vector<ggml_tensor*> tensors;
    for (size_t i = 0; i < fold.values.size(); ++i) {
      const TensorRef& result = fold.instance.outputs[i];
      tensors.push_back(
          new_tensor(block->context.get(), result.tensor->type, result.sizes));
      ggml_set_name(
          tensors.back(), method_.graph.value(fold.values[i]).name.c_str());
    }
    block->buffer.reset(
        ggml_backend_alloc_ctx_tensors(block->context.get(), primary_backend_));
    if (!block->buffer) {
      throw std::runtime_error("ggml: failed to allocate constant results");
    }
    size_t loaded = 0;
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < fold.values.size(); ++i) {
      bytes.resize(ggml_nbytes(tensors[i]));
      ggml_backend_tensor_get(
          fold.instance.outputs[i].tensor, bytes.data(), 0, bytes.size());
      ggml_backend_tensor_set(tensors[i], bytes.data(), 0, bytes.size());
      persistent_.emplace(fold.values[i], tensors[i]);
      std::shared_ptr<ggml_tensor> storage(block, tensors[i]);
      state_.constants[keys.at(fold.values[i])].tensor = storage;
      state_storage_.push_back(std::move(storage));
      loaded += bytes.size();
    }
    ggml_backend_sched_synchronize(scheduler.get());
    ggml_backend_buffer_set_usage(
        block->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (materialized_constant_bytes != nullptr) {
      *materialized_constant_bytes += loaded;
    }
  }

  // Rope frequency factors are uploaded once, one tensor per distinct table:
  // layers often hold equal inv_freq constants.
  void index_rope_frequencies() {
    std::map<std::vector<uint8_t>, std::vector<ValueId>> tables;
    std::unordered_set<ValueId> seen;
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
      if (!seen.insert(inv_freq).second) {
        continue;
      }
      const auto binding = std::ranges::find_if(
          method_.data_bindings,
          [this, inv_freq](const DataBinding& candidate) {
            return candidate.value_id == inv_freq && candidate.has_data &&
                !is_state(candidate);
          });
      const std::optional<OwnedBytes> bytes =
          binding == method_.data_bindings.end() ? std::nullopt
                                                 : read_constant(binding->key);
      if (!bytes) {
        throw std::runtime_error("ggml: rope inv_freq must be a constant");
      }
      std::vector<float> factors = to_float_vector(
          bytes->span(), method_.graph.value(inv_freq).tensor_meta().dtype);
      // ggml_rope_ext with freq_base 1 computes theta = pos / factor. A zero
      // frequency leaves its dims unrotated, as partial RoPE does: an
      // infinite factor gives theta = 0.
      std::ranges::transform(factors, factors.begin(), [](float factor) {
        return factor == 0.0f ? std::numeric_limits<float>::infinity()
                              : 1.0f / factor;
      });
      std::vector<uint8_t> raw(factors.size() * sizeof(float));
      std::memcpy(raw.data(), factors.data(), raw.size());
      tables[std::move(raw)].push_back(inv_freq);
    }
    if (tables.empty()) {
      return;
    }
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * tables.size(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    rope_context_.reset(ggml_init(params));
    if (!rope_context_) {
      throw std::runtime_error("ggml: failed to create rope factor context");
    }
    for (const auto& [raw, users] : tables) {
      ggml_tensor* tensor = ggml_new_tensor_1d(
          rope_context_.get(),
          GGML_TYPE_F32,
          static_cast<int64_t>(raw.size() / sizeof(float)));
      for (const ValueId inv_freq : users) {
        rope_factors_.emplace(inv_freq, tensor);
      }
    }
    rope_buffer_.reset(
        ggml_backend_alloc_ctx_tensors(rope_context_.get(), primary_backend_));
    if (!rope_buffer_) {
      throw std::runtime_error("ggml: failed to allocate rope factors");
    }
    for (const auto& [raw, users] : tables) {
      ggml_backend_tensor_set(
          rope_factors_.at(users.front()), raw.data(), 0, raw.size());
    }
    ggml_backend_buffer_set_usage(
        rope_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
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

  GraphBuilder graph_builder(
      std::unordered_map<ValueId, int64_t> truncated,
      const OffGraphCacheBinding* offgraph_cache = nullptr);
  GraphInstance build_graph(const std::vector<int64_t>& key_extents);

  static std::optional<int64_t> static_dim(const TensorMeta& meta, size_t i) {
    if (i >= meta.sizes.size() ||
        (!meta.lower_bounds.empty() &&
         meta.lower_bounds.at(i) != meta.sizes[i])) {
      return std::nullopt;
    }
    return meta.sizes[i];
  }

  // Matches mask = wrappers(compare(key_index, query)), or a product of such
  // boolean masks, with key_index varying only along the last (key) axis and
  // query constant along it. Appends the comparisons the mask ands together
  // to `compares`, and the values computed over the key axis to out.chain.
  bool match_mask_extent(
      ValueId id,
      MaskExtent& out,
      std::vector<KeyCompare>& compares,
      size_t depth = 0) const {
    const Graph& graph = method_.graph;
    const Value& value = graph.value(id);
    // A boolean mask, where true means attend; additive masks never match.
    if (depth >= 16 || !value.is_tensor() ||
        value.tensor_meta().dtype != ScalarType::Bool ||
        static_dim(value.tensor_meta(), value.tensor_meta().ndim() - 1) !=
            out.keys ||
        std::ranges::find(graph.output_ids, id) != graph.output_ids.end()) {
      return false;
    }
    if (std::ranges::find(out.chain, id) == out.chain.end()) {
      out.chain.push_back(id);
    }
    const NodeId producer = value.producer_id;
    if (!valid(producer) || !graph.node(producer).is_call()) {
      return false;
    }
    const Node& node = graph.node(producer);
    const std::string op = op_key(node.target);
    const NamedArgument* self = find_arg(node, "self", 0);
    if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
      return false;
    }
    const NamedArgument* other = find_arg(node, "other", 1);
    const bool other_tensor =
        other != nullptr && other->arg.kind() == ArgKind::Tensor;
    if (op == "aten.le.Tensor" || op == "aten.lt.Tensor") {
      if (!other_tensor) {
        return false;
      }
      const auto is_key = [&](ValueId operand) {
        const TensorMeta& meta = graph.value(operand).tensor_meta();
        for (size_t i = 0; i + 1 < meta.ndim(); ++i) {
          if (static_dim(meta, i) != 1) {
            return false;
          }
        }
        return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == out.keys;
      };
      const auto is_query = [&](ValueId operand) {
        const TensorMeta& meta = graph.value(operand).tensor_meta();
        return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == 1;
      };
      const ValueId a = self->arg.as_tensor().id;
      const ValueId b = other->arg.as_tensor().id;
      KeyCompare compare{.compare = op.substr(5, 2)};
      if (is_key(a) && is_query(b)) {
        compare.key_operand = a;
        compare.query_operand = b;
      } else if (is_key(b) && is_query(a)) {
        compare.key_lhs = false;
        compare.key_operand = b;
        compare.query_operand = a;
      } else {
        return false;
      }
      compares.push_back(std::move(compare));
      return true;
    }
    if (op == "aten.mul.Tensor") {
      // A product of bools is their logical and.
      return other_tensor &&
          match_mask_extent(
                 self->arg.as_tensor().id, out, compares, depth + 1) &&
          match_mask_extent(
                 other->arg.as_tensor().id, out, compares, depth + 1);
    }
    if (op == "aten.expand.default" || op == "aten.expand_copy.default") {
      // The target size of the key axis is rewritten with the extent.
      const NamedArgument* size = find_arg(node, "size", 1);
      if (size == nullptr || size->arg.kind() != ArgKind::IntList) {
        return false;
      }
      const IntListArg& list = size->arg.as_int_list();
      if (list.values.empty() ||
          (!list.ids.empty() && valid(list.ids.back())) ||
          (list.values.back() != -1 && list.values.back() != out.keys)) {
        return false;
      }
    } else if (
        op != "aten.unsqueeze.default" && op != "aten.unsqueeze_copy.default" &&
        op != "aten.alias.default" && op != "aten.clone.default") {
      return false;
    }
    return match_mask_extent(
        self->arg.as_tensor().id, out, compares, depth + 1);
  }

  // Every value in the chain feeds nothing but the producer of another one,
  // or SDPA masks over the same keys.
  bool feeds_only_masks(const MaskExtent& extent) const {
    const Graph& graph = method_.graph;
    for (const ValueId id : extent.chain) {
      for (const NodeId consumer : graph.value(id).consumer_ids) {
        const Node& node = graph.node(consumer);
        const bool in_chain =
            std::ranges::any_of(output_ids(node), [&](ValueId output) {
              return std::ranges::find(extent.chain, output) !=
                  extent.chain.end();
            });
        const NamedArgument* mask = find_arg(node, "attn_mask", 3);
        const NamedArgument* key = find_arg(node, "key", 1);
        const bool sdpa_mask = node.is_call() &&
            op_key(node.target) ==
                "aten.scaled_dot_product_attention.default" &&
            mask != nullptr && mask->arg.kind() == ArgKind::Tensor &&
            mask->arg.as_tensor().id == id && key != nullptr &&
            key->arg.kind() == ArgKind::Tensor &&
            static_dim(graph.value(key->arg.as_tensor().id).tensor_meta(), 2) ==
                extent.keys;
        if (!in_chain && !sdpa_mask) {
          return false;
        }
      }
    }
    return true;
  }

  void index_mask_extents() {
    // Masks in schedule order, with the keys every attention reading them
    // covers.
    std::vector<std::pair<ValueId, std::optional<int64_t>>> masks;
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
      const std::optional<int64_t> keys =
          key_meta.ndim() == 4 ? static_dim(key_meta, 2) : std::nullopt;
      const ValueId mask_id = mask->arg.as_tensor().id;
      const auto found = std::ranges::find(
          masks, mask_id, &std::pair<ValueId, std::optional<int64_t>>::first);
      if (found == masks.end()) {
        masks.emplace_back(mask_id, keys);
      } else if (found->second != keys) {
        found->second.reset();
      }
    }
    // Masks that share a value are computed over one prefix.
    std::vector<std::optional<MaskExtent>> groups;
    for (const auto& [mask_id, keys] : masks) {
      if (!keys) {
        continue;
      }
      MaskExtent extent;
      extent.keys = *keys;
      std::vector<KeyCompare> compares;
      if (!match_mask_extent(mask_id, extent, compares)) {
        continue;
      }
      extent.masks.push_back(std::move(compares));
      // Every chain value ends in its group's keys, so groups that share a
      // value agree on them.
      for (std::optional<MaskExtent>& group : groups) {
        if (!group || std::ranges::none_of(group->chain, [&](ValueId id) {
              return std::ranges::find(extent.chain, id) != extent.chain.end();
            })) {
          continue;
        }
        for (std::vector<KeyCompare>& mask : group->masks) {
          extent.masks.push_back(std::move(mask));
        }
        for (const ValueId id : group->chain) {
          if (std::ranges::find(extent.chain, id) == extent.chain.end()) {
            extent.chain.push_back(id);
          }
        }
        group.reset();
      }
      groups.emplace_back(std::move(extent));
    }
    for (std::optional<MaskExtent>& group : groups) {
      if (group && feeds_only_masks(*group)) {
        mask_extents_.push_back(std::move(*group));
      }
    }
  }

  std::vector<int64_t> full_key_extents() const {
    std::vector<int64_t> out(mask_extents_.size());
    std::ranges::transform(mask_extents_, out.begin(), &MaskExtent::keys);
    if (valid(offgraph_position_)) {
      out.push_back(offgraph_storage_->rows);
    }
    return out;
  }

  HostIntEvaluator host_evaluator() {
    return HostIntEvaluator(
        method_,
        inputs_,
        [this](ValueId id) { return host_constant(id); },
        host_static_values_);
  }

  std::vector<int64_t> evaluate_key_extents() {
    if (mask_extents_.empty() && !valid(offgraph_position_)) {
      return full_key_extents();
    }
    HostIntEvaluator host = host_evaluator();
    std::vector<int64_t> out;
    for (const MaskExtent& extent : mask_extents_) {
      // A mask attends the keys its every comparison attends.
      int64_t widest = 0;
      for (const std::vector<KeyCompare>& mask : extent.masks) {
        int64_t attended = extent.keys;
        for (const KeyCompare& compare : mask) {
          const HostInts* key_index = host.tensor(compare.key_operand);
          const HostInts* query = host.tensor(compare.query_operand);
          if (key_index != nullptr && query != nullptr) {
            attended = std::min(
                attended,
                attended_key_extent(compare, extent.keys, *key_index, *query));
          }
        }
        widest = std::max(widest, attended);
      }
      out.push_back(widest);
    }
    if (valid(offgraph_position_)) {
      // The cache rows the step reads, bucketed like masks.
      const int64_t needed = offgraph_step_->rows;
      out.push_back(std::min<int64_t>(
          offgraph_storage_->rows,
          (needed + kKeyExtentBucket - 1) / kKeyExtentBucket *
              kKeyExtentBucket));
    }
    return out;
  }

  // Binds the off-graph cache to a method that calls update_and_attend.
  void index_offgraph_attention() {
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() ||
          op_key(node.target) != "kvcache.update_and_attend.default") {
        continue;
      }
      const NamedArgument* position = find_arg(node, "position", 3);
      if (position == nullptr || position->arg.kind() != ArgKind::Tensor) {
        throw std::runtime_error(
            "ggml: update_and_attend is missing its position tensor");
      }
      const ValueId position_id = position->arg.as_tensor().id;
      if (valid(offgraph_position_) && position_id != offgraph_position_) {
        throw std::runtime_error(
            "ggml: every update_and_attend in a method must read the same "
            "position tensor");
      }
      offgraph_position_ = position_id;
    }
    if (!valid(offgraph_position_)) {
      return;
    }
    if (!kv_cache_) {
      throw std::runtime_error(
          "ggml: method '" + method_.name +
          "' calls kvcache.update_and_attend but its context has no KV "
          "cache; create the context with the llm_cache_registry_key option");
    }
    bind_offgraph(1);
  }

  // Grows the cache to `rows` and, if its storage moved, rebinds it, drops
  // the graphs built over the old storage, which this executable keeps alive
  // until then, and reserves the context's buffers for the worst case over
  // the new storage.
  void bind_offgraph(int64_t rows) {
    kv_cache_.pool().reserve(primary_backend_, static_cast<int>(rows));
    if (kv_cache_.pool().storage() == offgraph_storage_) {
      return;
    }
    if (state_.scheduler) {
      release_allocation();
    }
    cached_graphs_.clear();
    offgraph_storage_ = kv_cache_.pool().storage();
    offgraph_cache_.layers.clear();
    for (const GgmlKvPool::Layer& layer : offgraph_storage_->layers) {
      offgraph_cache_.layers.push_back(
          {.k = layer.k,
           .v = layer.v,
           .window = layer.window,
           .ring = layer.ring});
    }
    if (worst_case_) {
      worst_case_ = std::make_shared<const GraphInstance>(
          build_graph(full_key_extents()));
      state_.reserved.push_back(worst_case_);
      reserve(*worst_case_);
    }
  }

  struct OffGraphStep {
    // Per token, its logical position.
    std::vector<int32_t> positions;
    // Cache rows the step reads, which storage must hold.
    int64_t rows;
  };

  // Reads the step's positions on the host, one per token, and checks them
  // against the cache.
  OffGraphStep checked_offgraph_step() {
    HostIntEvaluator host = host_evaluator();
    const HostInts* values = host.tensor(offgraph_position_);
    if (values == nullptr || values->data.empty()) {
      throw std::runtime_error(
          "ggml: update_and_attend positions must be computable on the host "
          "from graph inputs");
    }
    const std::vector<int64_t>& sizes = values->sizes;
    if (sizes.empty() || sizes.size() > 2 ||
        (sizes.size() == 2 && sizes[1] != 1)) {
      throw std::invalid_argument(
          "ggml: update_and_attend positions must be [T] or [T, 1]");
    }
    OffGraphStep step;
    for (const int64_t position : values->data) {
      if (position < 0 || position >= kv_cache_.capacity()) {
        throw std::out_of_range(
            "ggml: KV cache position " + std::to_string(position) +
            " is outside the cache capacity of " +
            std::to_string(kv_cache_.capacity()));
      }
      step.positions.push_back(static_cast<int32_t>(position));
    }
    if (kv_cache_.cells != nullptr) {
      // Placing the step claims its cells for every layer of the forward.
      const cache::CellStep* placed = kv_cache_.cells->place_step(
          0, step.positions.data(), static_cast<int>(step.positions.size()));
      if (placed == nullptr) {
        throw std::invalid_argument(
            "ggml: the cell cache refused the step: declare it first, with "
            "positions continuing each sequence and room in the pool");
      }
      step.rows = placed->read_len;
      return step;
    }
    step.rows = checked_sequence_step(step.positions);
    return step;
  }

  // Checks a single-sequence step's positions are consecutive, fit the cache,
  // and start at the sequence's length, and returns the rows it reads.
  // Re-running positions takes an explicit rewind of the cache.
  int64_t checked_sequence_step(const std::vector<int32_t>& positions) {
    GgmlSequenceCache& sequence = *kv_cache_.sequence;
    const int64_t start = positions.front();
    for (size_t i = 0; i < positions.size(); ++i) {
      if (positions[i] != start + static_cast<int64_t>(i)) {
        throw std::invalid_argument(
            "ggml: update_and_attend positions must be consecutive");
      }
    }
    const auto tokens = static_cast<int64_t>(positions.size());
    if (start + tokens > sequence.capacity()) {
      throw std::out_of_range(
          "ggml: KV cache step [" + std::to_string(start) + ", " +
          std::to_string(start + tokens) + ") exceeds the cache capacity of " +
          std::to_string(sequence.capacity()));
    }
    if (start != sequence.length()) {
      throw std::invalid_argument(
          "ggml: KV cache step starts at position " + std::to_string(start) +
          ", not at the sequence's length of " +
          std::to_string(sequence.length()) +
          "; rewind the cache to run earlier positions again");
    }
    return start + tokens;
  }

  // Plans the step on the cache and fills each window's cache rows and mask.
  // Returns a plan to commit once the step has run.
  std::optional<SeqStepPlan> prepare_offgraph(GraphInstance& instance) {
    if (!instance.offgraph) {
      return std::nullopt;
    }
    const OffGraphUploads& slots = *instance.offgraph;
    if (static_cast<int64_t>(offgraph_step_->positions.size()) !=
        slots.tokens) {
      throw std::runtime_error(
          "ggml: update_and_attend needs one position per token");
    }
    if (kv_cache_.cells != nullptr) {
      prepare_cell_step(instance);
      return std::nullopt;
    }
    const int64_t start = offgraph_step_->positions.front();
    const auto plan_layer = [&](int layer) {
      const std::optional<SeqStepPlan> plan = kv_cache_.sequence->plan(
          layer, static_cast<int>(start), static_cast<int>(slots.tokens));
      if (!plan) {
        throw std::runtime_error(
            "ggml: the KV cache cannot plan a " + std::to_string(slots.tokens) +
            "-token step at position " + std::to_string(start) + " for layer " +
            std::to_string(layer));
      }
      return *plan;
    };
    // Layers outside the method's calls must fit the step too.
    for (size_t layer = 0; layer < offgraph_cache_.layers.size(); ++layer) {
      plan_layer(static_cast<int>(layer));
    }
    const ggml_fp16_t attend = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t masked =
        ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    std::optional<SeqStepPlan> committed;
    for (const OffGraphUploads::Group& group : slots.groups) {
      const SeqStepPlan plan = plan_layer(group.layer);
      // The logical position each row holds this step, or -1 if it is unread.
      std::vector<int64_t> positions(static_cast<size_t>(group.keys), -1);
      int64_t position = plan.read_base_pos;
      for (int r = 0; r < plan.n_read; ++r) {
        const Run read = plan.read[r];
        if (read.start < 0 || read.start + read.len > group.keys) {
          throw std::logic_error(
              "ggml: KV cache read exceeds the graph's keys");
        }
        for (int i = 0; i < read.len; ++i) {
          positions[static_cast<size_t>(read.start + i)] = position++;
        }
      }
      std::vector<uint8_t>& rows = instance.uploads[group.rows].bytes;
      rows.resize(static_cast<size_t>(slots.tokens) * sizeof(int32_t));
      auto* row = reinterpret_cast<int32_t*>(rows.data());
      int64_t written = 0;
      for (int w = 0; w < plan.n_write; ++w) {
        for (int i = 0; i < plan.write[w].len && written < slots.tokens; ++i) {
          row[written++] = plan.write[w].start + i;
        }
      }
      if (written != slots.tokens) {
        throw std::logic_error("ggml: KV cache planned the wrong rows");
      }
      std::vector<uint8_t>& mask = instance.uploads[group.mask].bytes;
      mask.resize(
          static_cast<size_t>(slots.tokens * group.keys) * sizeof(ggml_fp16_t));
      auto* bias = reinterpret_cast<ggml_fp16_t*>(mask.data());
      for (int64_t i = 0; i < slots.tokens; ++i) {
        const int64_t query = start + i;
        for (int64_t j = 0; j < group.keys; ++j) {
          const int64_t key = positions[static_cast<size_t>(j)];
          bias[i * group.keys + j] = key >= 0 && key <= query &&
                  (group.window == 0 || query - key < group.window)
              ? attend
              : masked;
        }
      }
      committed = plan;
    }
    return committed;
  }

  // Fills each window's cache rows and mask from the placed cell step, which
  // every layer of the forward shares.
  void prepare_cell_step(GraphInstance& instance) {
    const OffGraphUploads& slots = *instance.offgraph;
    const ggml_fp16_t attend = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t masked =
        ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (const OffGraphUploads::Group& group : slots.groups) {
      const cache::CellStep* step = kv_cache_.cells->place_step(
          group.layer,
          offgraph_step_->positions.data(),
          static_cast<int>(slots.tokens));
      if (step == nullptr || step->read_len > group.keys ||
          static_cast<int64_t>(step->cells.size()) != slots.tokens) {
        throw std::logic_error("ggml: the cell step does not fit the graph");
      }
      std::vector<uint8_t>& rows = instance.uploads[group.rows].bytes;
      rows.resize(step->cells.size() * sizeof(int32_t));
      std::memcpy(rows.data(), step->cells.data(), rows.size());
      std::vector<uint8_t>& mask = instance.uploads[group.mask].bytes;
      mask.resize(
          static_cast<size_t>(slots.tokens * group.keys) * sizeof(ggml_fp16_t));
      auto* bias = reinterpret_cast<ggml_fp16_t*>(mask.data());
      for (int64_t i = 0; i < slots.tokens; ++i) {
        const uint8_t* bits = step->mask_bits.data() + i * step->read_len;
        for (int64_t j = 0; j < group.keys; ++j) {
          bias[i * group.keys + j] =
              j < step->read_len && bits[j] != 0 ? attend : masked;
        }
      }
    }
  }

  // The slow path, for integers whose bound needs this run's input values,
  // e.g. positions a graph input offsets; build_graph drops the others.
  void check_host_integers(const GraphInstance& instance) {
    if (instance.host_integer_checks.empty()) {
      return;
    }
    HostIntEvaluator host(
        method_,
        inputs_,
        [this](ValueId id) { return host_constant(id); },
        host_static_values_);
    for (const HostIntegerCheck& check : instance.host_integer_checks) {
      const std::optional<int64_t> peak = host.peak_bound(check.id);
      if (!peak) {
        throw unbounded_host_integer(check);
      }
      if (*peak > kExactF32Integer) {
        throw inexact_host_integer(check);
      }
    }
  }

  // Whether the host can bound an integer at all depends on the graph alone,
  // so one that no run could bound is refused at compile; each run checks
  // only that its inputs keep the bound within +-2^24.
  void check_host_integers_boundable(const GraphInstance& instance) {
    HostIntEvaluator host = host_evaluator();
    for (const HostIntegerCheck& check : instance.host_integer_checks) {
      const std::optional<int64_t> floor = host.static_peak_bound(check.id);
      if (!floor) {
        throw unbounded_host_integer(check);
      }
      if (*floor > kExactF32Integer) {
        throw inexact_host_integer(check);
      }
    }
  }

  static std::runtime_error unbounded_host_integer(
      const HostIntegerCheck& check) {
    return std::runtime_error(
        "ggml: " + check.what +
        " must be an integer F32 holds exactly: a graph input, a small "
        "constant or an arange, through integer casts, views or repeats, "
        "adds or subtracts");
  }

  static std::out_of_range inexact_host_integer(const HostIntegerCheck& check) {
    return std::out_of_range(
        "ggml: " + check.what +
        " is computed through integers that can leave +-2^24, which F32 "
        "holds exactly");
  }

  // Small integer constants, e.g. a key offset or the key positions of a
  // static KV cache, read once from the package.
  std::optional<HostInts> host_constant(ValueId id) {
    if (const auto found = host_constants_.find(id);
        found != host_constants_.end()) {
      return found->second;
    }
    std::optional<HostInts> out;
    const auto binding = std::ranges::find_if(
        method_.data_bindings, [this, id](const DataBinding& candidate) {
          return candidate.value_id == id && candidate.has_data &&
              !candidate.mutated && !is_state(candidate);
        });
    const TensorMeta* meta = method_.graph.value(id).is_tensor()
        ? &method_.graph.value(id).tensor_meta()
        : nullptr;
    if (binding != method_.data_bindings.end() && meta != nullptr &&
        !meta->quant && checked_numel(meta->sizes) <= kHostEvalMaxElements &&
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

  GgmlContextState& state_;
  // Declared before every graph and scheduler that reads it.
  std::vector<std::shared_ptr<ggml_tensor>> state_storage_;
  // Storage bind_state allocated, handed to the context once compiled.
  std::vector<std::pair<std::string, std::shared_ptr<ggml_tensor>>>
      fresh_state_;
  // Accumulated while compiling, reported to the context at the end.
  mutable size_t heap_copied_constant_bytes_ = 0;
  const Method& method_;
  const Package& package_;
  GgmlEngineOptions options_;
  GgmlKvCache kv_cache_;
  // The position tensor of the method's update_and_attend calls, if any.
  ValueId offgraph_position_ = kInvalid;
  // Cache storage; build_graph sets the keys per graph.
  OffGraphCacheBinding offgraph_cache_;
  // The storage offgraph_cache_ and cached graphs point into. Declared before
  // cached_graphs_ so it outlives them.
  std::shared_ptr<const GgmlKvPool::Storage> offgraph_storage_;
  // The step execute() is running, checked once.
  std::optional<OffGraphStep> offgraph_step_;
  std::vector<InputSlot> inputs_;
  std::vector<OutputSlot> outputs_;
  ggml_backend_t primary_backend_ = nullptr;
  bool f32_accumulation_ = false;
  bool f16_kv_caches_ = false;
  ggml_backend_dev_t gpu_device_ = nullptr;
  std::vector<CachedGraph> cached_graphs_;
  // The graph at the largest inputs and full key extents, reserved in the
  // context's compute buffers.
  std::shared_ptr<const GraphInstance> worst_case_;
  std::vector<MaskExtent> mask_extents_;
  std::unordered_map<ValueId, std::optional<HostInts>> host_constants_;
  HostIntEvaluator::StaticCache host_static_values_;
  ggml_backend_buffer_ptr output_staging_;
  std::vector<uint8_t> output_host_;
  std::unordered_map<ValueId, ggml_tensor*> persistent_;
  // Each persistent tensor's load-plan key, which folded results' keys name.
  std::unordered_map<ValueId, std::string> constant_keys_;
  // Float scales of the Q4_0 weights with unit block scales, by weight.
  std::unordered_map<ValueId, ggml_tensor*> row_scales_;
  bool executed_ = false;
  std::unordered_map<ValueId, ggml_tensor*> rope_factors_;
  ggml_context_ptr rope_context_;
  ggml_backend_buffer_ptr rope_buffer_;
  // Per-output-channel factors applied to convolution weights at load.
  std::unordered_map<ValueId, std::vector<float>> weight_scales_;
  FoldedBatchNorms batch_norms_;
  ggml_context_ptr batch_norm_context_;
  ggml_backend_buffer_ptr batch_norm_buffer_;
};

GraphBuilder GgmlExecutable::graph_builder(
    std::unordered_map<ValueId, int64_t> truncated,
    const OffGraphCacheBinding* offgraph_cache) {
  return GraphBuilder(
      method_,
      inputs_,
      outputs_,
      persistent_,
      row_scales_,
      rope_factors_,
      batch_norms_,
      std::move(truncated),
      options_.graph_size,
      options_.device == GgmlDevice::Gpu,
      options_.exact_topk_for_testing,
      options_.exact_gelu_for_testing,
      f32_accumulation_,
      offgraph_cache);
}

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
  OffGraphCacheBinding offgraph_cache = offgraph_cache_;
  if (valid(offgraph_position_)) {
    offgraph_cache.keys = key_extents.back();
  }
  GraphInstance instance =
      graph_builder(
          std::move(truncated),
          valid(offgraph_position_) ? &offgraph_cache : nullptr)
          .build();
  // Boundability without a per-run check: shapes are part of a graph's key,
  // and every run rejects row indices outside [0, rows), such as the positions
  // a KV cache is written at, as a hard error before computing (when building
  // the graph, or in refresh_inputs on a replay). So an integer that
  // constants, shapes and the row indices of tables of at most 2^24 rows bound
  // within +-2^24 needs no check per run; check_host_integers bounds the rest.
  if (!instance.host_integer_checks.empty()) {
    std::unordered_map<ValueId, int64_t> row_peaks;
    for (const RowIndexCheck& check : instance.row_index_checks) {
      if (check.rows > kExactF32Integer) {
        continue;
      }
      const int64_t peak = std::max<int64_t>(check.rows - 1, 0);
      int64_t& known =
          row_peaks.try_emplace(check.input_id, peak).first->second;
      known = std::min(known, peak);
    }
    HostIntEvaluator host = host_evaluator();
    std::erase_if(
        instance.host_integer_checks,
        [&host, &row_peaks](const HostIntegerCheck& check) {
          const std::optional<int64_t> bound =
              host.shape_bound(check.id, row_peaks);
          return bound && *bound <= kExactF32Integer;
        });
  }
  for (int i = 0; i < ggml_graph_n_nodes(instance.graph); ++i) {
    std::array<ggml_tensor*, GGML_MAX_SRC> sources{};
    std::ranges::copy(ggml_graph_node(instance.graph, i)->src, sources.begin());
    instance.sources.push_back(sources);
  }
  ggml_context* ctx = instance.context.get();
  for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    if (t->buffer == nullptr) {
      instance.unallocated.emplace_back(t, t->data);
    }
  }
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
    if (!instance.host_integer_checks.empty()) {
      std::fprintf(
          stderr,
          "ggml host integer checks per run: %zu\n",
          instance.host_integer_checks.size());
    }
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
    size_t* f16_kv_cache_bytes,
    GgmlContextState& state,
    GgmlKvCache kv_cache) {
  return std::make_unique<GgmlExecutable>(
      method,
      package,
      options,
      materialized_constant_bytes,
      heap_copied_constant_bytes,
      f16_kv_cache_bytes,
      state,
      kv_cache);
}

} // namespace ptn
