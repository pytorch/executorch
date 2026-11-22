// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <ggml-cpp.h>
#include <ggml.h>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>

namespace ptn {

struct Method;

namespace ggml_detail {

std::string op_key(std::string_view target);

const NamedArgument*
find_arg(const Node& node, std::string_view name, size_t positional);

const NamedArgument&
require_arg(const Node& node, std::string_view name, size_t positional);

std::vector<ValueId> output_ids(const Node& node);

int64_t normalize_dim(int64_t dim, size_t rank);

struct TensorRef {
  ggml_tensor* tensor = nullptr;
  std::vector<int64_t> sizes;
};

// A torchao activation quantize/dequantize chain in flight; `input` is the
// float tensor it started from.
struct DynamicQuant {
  TensorRef input;
};

using RuntimeValue = std::
    variant<std::monostate, TensorRef, DynamicQuant, int64_t, double, bool>;

struct InputSlot {
  ValueId id = kInvalid;
  TensorMeta meta;
  std::vector<int64_t> sizes;
  std::vector<uint8_t> bytes;
};

struct OutputSlot {
  ValueId id = kInvalid;
  TensorMeta meta;
  std::vector<int64_t> sizes;
  // Points into the executable's output staging until the next execute.
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
};

struct Upload {
  ggml_tensor* tensor = nullptr;
  std::vector<uint8_t> bytes;
  // Signature position of the graph input these bytes come from; empty for
  // tensors that lowering fills itself, such as causal masks.
  std::optional<size_t> input;
};

struct RowIndexCheck {
  ValueId input_id = kInvalid;
  int64_t rows = 0;
};

// The off-graph KV cache as one graph sees it.
struct OffGraphCacheBinding {
  struct Layer {
    // K and V storage, ggml [head_dim, rows, n_kv_heads].
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
    // The layer's sliding window, 0 for none; layers sharing one share a mask.
    int window = 0;
    // A ring layer attends all its rows; any other, the first `keys`.
    bool ring = false;
  };
  std::vector<Layer> layers;
  // Rows a non-ring layer's attention reads, a prefix of its storage.
  int64_t keys = 0;
};

// Uploads of a graph's off-graph attention, which the executor fills from the
// cache's plan before every run. Layers with the same window share them.
struct OffGraphUploads {
  struct Group {
    int window = 0;
    // A layer with this window, whose plan fills the uploads.
    int layer = 0;
    // I32 [tokens]: the cache row each token writes.
    size_t rows = 0;
    // F16 [tokens, keys]: additive mask of the rows each token attends.
    size_t mask = 0;
    int64_t keys = 0;
  };
  int64_t tokens = 0;
  std::vector<Group> groups;
};

// An integer a comparison leaves to the host, such as an attention position
// that adds and subtracts compute from a graph input, which the host bounds
// on every run.
struct HostIntegerCheck {
  ValueId id = kInvalid;
  std::string what;
};

struct GraphInstance {
  ggml_context_ptr context;
  ggml_cgraph* graph = nullptr;
  std::vector<Upload> uploads;
  std::vector<TensorRef> outputs;
  // Input-driven row indices, rechecked on every replay.
  std::vector<RowIndexCheck> row_index_checks;
  std::optional<OffGraphUploads> offgraph;
  // Integer inputs some op computes on in F32, rechecked on every replay.
  std::vector<ValueId> exact_integer_checks;
  std::vector<HostIntegerCheck> host_integer_checks;
};

// Inference batch norm is the per-channel map x * scale + shift of its
// constants, computed once at load. A convolution read only by the batch norm
// absorbs it: its weight is scaled at load and its bias replaced, leaving the
// batch norm a no-op. Keyed by output value.
struct FoldedBatchNorms {
  // Batch norm outputs absorbed by their convolution.
  std::unordered_set<ValueId> absorbed;
  // The other batch norm outputs: scale and shift.
  std::unordered_map<ValueId, std::pair<ggml_tensor*, ggml_tensor*>> affine;
  // Absorbing convolution outputs: the bias replacing the convolution's own.
  std::unordered_map<ValueId, ggml_tensor*> conv_bias;
};

// Node capacity of a method's graphs: at least `graph_size`, and at least the
// metadata room GraphBuilder sizes for every lowering's tensors, so no graph
// can outgrow it.
size_t graph_capacity(const Method& method, size_t graph_size);

const ggml_tensor* storage_root(const ggml_tensor* tensor);

void check_index_bytes(const InputSlot& input, int64_t rows);

// F32 holds every integer within +-2^24 exactly.
inline constexpr int64_t kExactF32Integer = int64_t{1} << 24;

bool is_integer(ScalarType dtype);
bool keeps_integer_values(const std::string& key);
void check_exact_integer_bytes(const InputSlot& input);

// Whether the attention lowerings accept these query, key and value sizes:
// [B, H, T, D] for aten SDPA, [B, T, H, D] for custom_sdpa. ggml asserts on
// the sizes they refuse.
bool sdpa_shapes_compatible(
    const std::vector<int64_t>& query,
    const std::vector<int64_t>& key,
    const std::vector<int64_t>& value);
bool custom_sdpa_shapes_compatible(
    const std::vector<int64_t>& query,
    const std::vector<int64_t>& key,
    const std::vector<int64_t>& value);

ggml_tensor* reshape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes);

ggml_tensor* view_with_shape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes,
    size_t offset);

// The bmm that reads `index`'s result as its weight side when `index` gathers
// whole matrices of a 3-D constant [E, O, I] by one 1-D integer index on dim 0
// and that bmm, with activations [S, I, 1], is its only reader: the batched
// MoE expert form, lowered to one ggml_mul_mat_id. Otherwise nullptr.
const Node* expert_matmul(const Method& method, const Node& index);

// The expert gather whose expert_matmul is `matmul`, or nullptr.
const Node* expert_gather(const Method& method, const Node& matmul);

class GraphBuilder {
 public:
  GraphBuilder(
      const Method& method,
      const std::vector<InputSlot>& inputs,
      const std::vector<OutputSlot>& outputs,
      const std::unordered_map<ValueId, ggml_tensor*>& persistent,
      const std::unordered_map<ValueId, ggml_tensor*>& row_scales,
      const std::unordered_map<ValueId, std::vector<uint8_t>>&
          rope_factor_bytes,
      const FoldedBatchNorms& batch_norms,
      std::unordered_map<ValueId, int64_t> key_extents,
      size_t graph_size,
      bool gpu,
      bool exact_topk,
      bool exact_gelu,
      bool f32_accumulation,
      const OffGraphCacheBinding* offgraph_cache = nullptr);

  GraphInstance build();

 private:
  ggml_tensor* as_output(ggml_tensor* result, const TensorMeta& meta);
  void expand_write(ggml_tensor* write);
  ggml_tensor* write_back(ValueId value_id, ValueId target_id);
  ggml_tensor* snapshot(ggml_tensor* tensor);
  void seed_values();
  void set(ValueId id, RuntimeValue value);
  ggml_tensor* narrowed_to_declared(ValueId id, ggml_tensor* tensor);
  const RuntimeValue& value(ValueId id) const;
  TensorRef view(ValueId id) const;
  TensorRef tensor(ValueId id) const;
  TensorRef
  tensor_arg(const Node& node, std::string_view name, size_t positional) const;
  TensorRef view_arg(const Node& node, std::string_view name, size_t positional)
      const;
  TensorRef rows_arg(const Node& node, std::string_view name, size_t positional)
      const;
  ValueId tensor_arg_id(
      const Node& node,
      std::string_view name,
      size_t positional) const;
  void set_view(ValueId id, TensorRef ref);
  void set_single_view(const Node& node, TensorRef result);
  std::optional<int64_t> key_extent_of(const Node& node) const;
  TensorRef last_axis_prefix(TensorRef ref, int64_t extent);
  int64_t int_value(const Argument& arg) const;
  int64_t int_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      int64_t fallback,
      bool optional = false) const;
  double float_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      double fallback,
      bool optional = false) const;
  bool bool_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      bool fallback) const;
  std::vector<int64_t> int_list(const Argument& arg) const;
  void set_single_output(const Node& node, TensorRef result);
  void lower(const Node& node);
  void lower_linear(const Node& node);
  void lower_row_gather(const Node& node);
  void lower_expert_matmul(const Node& bmm, const Node& index);
  void check_row_indices(ValueId id, int64_t rows);
  void require_exact_integers(
      ValueId id,
      const std::string& what,
      bool defer_to_host = false);
  void lower_rms_norm(const Node& node);
  void lower_layer_norm(const Node& node);
  void lower_gelu(const Node& node);
  void lower_constant_pad(const Node& node);
  void lower_reshape(const Node& node);
  void lower_permute(const Node& node);
  void lower_transpose(const Node& node);
  void lower_copy_or_alias(const Node& node, bool alias);
  void lower_cast(const Node& node);
  ggml_tensor* as_f32(ggml_tensor* tensor);
  // An F16 or BF16 tensor in F32, for lowerings that compute in F32 only.
  ggml_tensor* half_as_f32(ggml_tensor* tensor);
  ggml_tensor* as_i32(ggml_tensor* tensor);
  void invalidate_cached_transforms(const ggml_tensor* target);
  ggml_tensor* contiguous(ggml_tensor* tensor);
  ggml_tensor* mul_mat(ggml_tensor* a, ggml_tensor* b);
  void index_written_values();
  bool touches_written_value(const Node& node) const;
  ggml_tensor* materialize(ggml_tensor* tensor);
  void mark_written(const ggml_tensor* target);

  struct BinaryOperands {
    ggml_tensor* lhs = nullptr;
    ggml_tensor* rhs = nullptr;
    std::vector<int64_t> sizes;
  };

  BinaryOperands tensor_operands(
      const TensorRef& left,
      const TensorRef& right,
      bool force_f32);
  void lower_binary(const Node& node, const std::string& key);
  static void require_float(const Node& node, const ggml_tensor* tensor);

  using Unary = ggml_tensor* (*)(ggml_context*, ggml_tensor*);

  void lower_unary(const Node& node, Unary fn);
  void lower_sqrt(const Node& node);
  void lower_hardtanh(const Node& node);
  void lower_elu(const Node& node);
  std::array<int, 2> pair_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      std::array<int, 2> fallback,
      int64_t min_value) const;
  std::optional<TensorRef> optional_tensor(
      const Node& node,
      std::string_view name,
      size_t positional) const;
  void require_only_first_output_used(const Node& node) const;
  static void require_f32(const Node& node, const ggml_tensor* tensor);
  void lower_convolution(const Node& node);
  void lower_batch_norm(const Node& node);
  void lower_max_pool2d(const Node& node);
  void lower_mean(const Node& node);
  void lower_sum(const Node& node);
  void lower_amax(const Node& node);
  void lower_softmax(const Node& node);
  void lower_softplus(const Node& node);
  ggml_tensor* nan_mask(ggml_tensor* x);
  TensorRef permuted(const TensorRef& input, const std::vector<int64_t>& dims);
  // The `self` of a reduction as ggml rows of the reduced elements, and the
  // result sizes.
  std::pair<ggml_tensor*, std::vector<int64_t>> reduced_rows(
      const Node& node,
      const std::string& op);
  ggml_tensor* row_argmax(ggml_tensor* rows);
  ggml_tensor* row_max(ggml_tensor* rows);
  ggml_tensor* nan_where_rows_hold_nan(ggml_tensor* rows, ggml_tensor* values);
  void lower_split(const Node& node, const std::string& key);
  void lower_topk(const Node& node);
  void set_as_output_type(ValueId id, TensorRef result);
  ggml_tensor* last_dim_gather(ggml_tensor* table, ggml_tensor* indices);
  void lower_slice(const Node& node, bool copy);
  void lower_select(const Node& node, bool copy);
  void lower_unsqueeze(const Node& node);
  void lower_squeeze(const Node& node);
  void lower_expand(const Node& node);
  void lower_repeat(const Node& node);
  void lower_cat(const Node& node);
  void lower_matmul(const Node& node);
  void lower_update_cache(const Node& node);
  TensorRef active_cache(const TensorRef& cache, int64_t length);
  TensorRef causal_mask(int64_t tokens, int64_t keys, int64_t start);
  void lower_custom_sdpa(const Node& node);
  enum class Comparison { Le, Lt, Ne };
  void lower_compare(const Node& node, Comparison comparison);
  void lower_arange(const Node& node);
  DynamicQuant dynamic_quant_arg(
      const Node& node,
      std::string_view name,
      size_t positional) const;
  void set_dynamic_quant(const Node& node, const DynamicQuant& quant);
  void lower_choose_qparams(const Node& node);
  void lower_quantize_affine(const Node& node);
  void lower_dequantize_affine(const Node& node);
  ggml_tensor* rope_factor_tensor(ValueId inv_freq);
  void lower_rope(const Node& node);
  void lower_gated_delta_rule(const Node& node);
  ggml_tensor* skip_f16_cast(ggml_tensor* rows) const;
  void lower_index_put(const Node& node);
  ggml_tensor* attention_mask(ValueId id, int64_t tokens, int64_t keys);
  void lower_sdpa(const Node& node);
  void lower_update_and_attend(const Node& node);
  void lower_sym_size(const Node& node);
  int64_t symbolic_operand(const Argument& arg) const;
  void lower_symbolic(const Node& node, const std::string& key);

  const Method& method_;
  const std::vector<InputSlot>& inputs_;
  const std::vector<OutputSlot>& outputs_;
  const std::unordered_map<ValueId, ggml_tensor*>& persistent_;
  const std::unordered_map<ValueId, ggml_tensor*>& row_scales_;
  const std::unordered_map<ValueId, std::vector<uint8_t>>& rope_factor_bytes_;
  const FoldedBatchNorms& batch_norms_;
  // Mask values (and their producers) computed over a key prefix only.
  std::unordered_map<ValueId, int64_t> key_extents_;
  size_t graph_size_;
  // The primary backend is a GPU.
  bool gpu_;
  // GgmlEngineOptions::exact_topk_for_testing.
  bool exact_topk_;
  // GgmlEngineOptions::exact_gelu_for_testing.
  bool exact_gelu_;
  // Matrix multiplies with an F16 operand request F32 accumulation.
  bool f32_accumulation_;
  const OffGraphCacheBinding* offgraph_cache_;
  std::vector<RuntimeValue> values_;
  std::unordered_map<ValueId, ggml_tensor*> rope_factor_tensors_;
  std::map<std::pair<ValueId, int64_t>, ggml_tensor*> attention_masks_;
  std::unordered_map<ggml_tensor*, ggml_tensor*> i32_casts_;
  std::unordered_set<const ggml_tensor*> persistent_storage_;
  std::unordered_set<const ggml_tensor*> aliased_storage_;
  std::unordered_set<const ggml_tensor*> written_storage_;
  std::unordered_set<ValueId> written_values_;
  // Set while lowering a node that touches a written value.
  bool copy_required_ = false;
  // Values lowered as non-contiguous views of shared storage, and their
  // copies.
  std::unordered_set<ValueId> deferred_;
  mutable std::unordered_map<ValueId, ggml_tensor*> deferred_copies_;
  GraphInstance instance_;
};

} // namespace ggml_detail
} // namespace ptn
