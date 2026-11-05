// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>

namespace ptn::ggml_detail {

void GraphBuilder::lower_update_cache(const Node& node) {
  const TensorRef update = tensor_arg(node, "value", 0);
  const TensorRef cache = tensor_arg(node, "cache", 1);
  const int64_t start = int_arg(node, "start_pos", 2, 0);
  if (cache.sizes.size() != update.sizes.size() || cache.sizes.size() != 4 ||
      update.sizes[0] != cache.sizes[0] || update.sizes[2] != cache.sizes[2] ||
      update.sizes[3] != cache.sizes[3] || start < 0 ||
      start > cache.sizes[1] - update.sizes[1]) {
    throw std::runtime_error("ggml: invalid KV cache update");
  }
  std::vector<int64_t> sizes = cache.sizes;
  sizes[1] = update.sizes[1];
  const size_t offset = static_cast<size_t>(start) * cache.tensor->nb[2];
  ggml_tensor* destination =
      view_with_shape(instance_.context.get(), cache.tensor, sizes, offset);
  ggml_tensor* source = cache.tensor->type == GGML_TYPE_F16
      ? skip_f16_cast(update.tensor)
      : update.tensor;
  invalidate_cached_transforms(destination);
  mark_written(destination);
  ggml_tensor* write = ggml_cpy(instance_.context.get(), source, destination);
  expand_write(write);
  const std::vector<ValueId> ids = output_ids(node);
  if (!ids.empty()) {
    set(ids.front(), TensorRef{write, sizes});
  }
}

TensorRef GraphBuilder::active_cache(const TensorRef& cache, int64_t length) {
  if (cache.sizes.size() != 4 || length <= 0 || length > cache.sizes[1]) {
    throw std::runtime_error("ggml: invalid active KV-cache length");
  }
  std::vector<int64_t> sizes = cache.sizes;
  sizes[1] = length;
  return TensorRef{
      view_with_shape(instance_.context.get(), cache.tensor, sizes, 0), sizes};
}

TensorRef
GraphBuilder::causal_mask(int64_t tokens, int64_t keys, int64_t start) {
  const std::vector<int64_t> sizes{1, 1, tokens, keys};
  ggml_tensor* mask = new_tensor(instance_.context.get(), GGML_TYPE_F16, sizes);
  ggml_set_input(mask);
  Upload upload;
  upload.tensor = mask;
  upload.bytes.resize(static_cast<size_t>(tokens * keys) * sizeof(ggml_fp16_t));
  auto* data = reinterpret_cast<ggml_fp16_t*>(upload.bytes.data());
  for (int64_t query = 0; query < tokens; ++query) {
    for (int64_t key = 0; key < keys; ++key) {
      const float bias =
          key <= start + query ? 0.0f : -std::numeric_limits<float>::infinity();
      data[query * keys + key] = ggml_fp32_to_fp16(bias);
    }
  }
  instance_.uploads.push_back(std::move(upload));
  return TensorRef{mask, sizes};
}

void GraphBuilder::lower_custom_sdpa(const Node& node) {
  const TensorRef query = tensor_arg(node, "query", 0);
  const TensorRef key_cache = tensor_arg(node, "key", 1);
  const TensorRef value_cache = tensor_arg(node, "value", 2);
  if (query.sizes.size() != 4) {
    throw std::runtime_error("ggml: custom_sdpa query must be [B,T,H,D]");
  }
  if (key_cache.sizes.size() != 4 || value_cache.sizes != key_cache.sizes ||
      key_cache.sizes[0] != query.sizes[0] ||
      key_cache.sizes[3] != query.sizes[3] || key_cache.sizes[2] <= 0 ||
      query.sizes[2] % key_cache.sizes[2] != 0) {
    throw std::runtime_error("ggml: custom_sdpa q/k/v shapes are incompatible");
  }
  const int64_t start = int_arg(node, "start_pos", 3, 0);
  const int64_t tokens = query.sizes[1];
  if (start < 0 || start > key_cache.sizes[1] - tokens) {
    throw std::runtime_error(
        "ggml: custom_sdpa start_pos is outside the KV cache");
  }
  const NamedArgument* mask_arg = find_arg(node, "attn_mask", 4);
  const bool masked = mask_arg != nullptr &&
      mask_arg->arg.kind() == ArgKind::Tensor &&
      valid(mask_arg->arg.as_tensor().id);
  // As in ExecuTorch's kernel, a masked call ignores start_pos and attends
  // the whole cache.
  const int64_t keys = masked ? key_cache.sizes[1] : start + tokens;
  const bool is_causal = bool_arg(node, "is_causal", 6, false);
  if (float_arg(node, "dropout_p", 5, 0.0, true) != 0.0) {
    throw std::runtime_error("ggml: custom_sdpa dropout is unsupported");
  }
  const TensorRef key = active_cache(key_cache, keys);
  const TensorRef val = active_cache(value_cache, keys);
  // flash_attn_ext kernels read the query as F32.
  ggml_tensor* q = query.tensor->type == GGML_TYPE_F32
      ? query.tensor
      : ggml_cast(instance_.context.get(), query.tensor, GGML_TYPE_F32);
  q = ggml_permute(instance_.context.get(), q, 0, 2, 1, 3);
  ggml_tensor* k =
      ggml_permute(instance_.context.get(), key.tensor, 0, 2, 1, 3);
  ggml_tensor* v =
      ggml_permute(instance_.context.get(), val.tensor, 0, 2, 1, 3);

  ggml_tensor* mask = nullptr;
  if (masked) {
    if (is_causal) {
      throw std::runtime_error(
          "ggml: custom_sdpa cannot take a mask and is_causal");
    }
    const TensorRef attn_mask = tensor(mask_arg->arg.as_tensor().id);
    const size_t rank = attn_mask.sizes.size();
    // flash_attn_ext reads [tokens, keys] mask rows without bounds checks.
    if (rank < 2 || attn_mask.sizes[rank - 2] < tokens ||
        attn_mask.sizes[rank - 1] < keys) {
      throw std::runtime_error(
          "ggml: custom_sdpa mask does not cover every query and key");
    }
    mask = ggml_cast(instance_.context.get(), attn_mask.tensor, GGML_TYPE_F16);
  } else if (is_causal && tokens > 1) {
    mask = causal_mask(tokens, keys, start).tensor;
  }
  const double default_scale =
      1.0 / std::sqrt(static_cast<double>(query.sizes.back()));
  const float scale =
      static_cast<float>(float_arg(node, "scale", 7, default_scale, true));
  ggml_tensor* result = ggml_flash_attn_ext(
      instance_.context.get(), q, k, v, mask, scale, 0.0f, 0.0f);
  ggml_prec_set_acc(result, GGML_PREC_F32);
  set_single_output(node, TensorRef{result, query.sizes});
}

ggml_tensor* GraphBuilder::rope_factor_tensor(ValueId inv_freq) {
  if (const auto cached = rope_factor_tensors_.find(inv_freq);
      cached != rope_factor_tensors_.end()) {
    return cached->second;
  }
  const auto found = rope_factor_bytes_.find(inv_freq);
  if (found == rope_factor_bytes_.end()) {
    throw std::runtime_error("ggml: rope inv_freq was not indexed at load");
  }
  ggml_tensor* factors = ggml_new_tensor_1d(
      instance_.context.get(),
      GGML_TYPE_F32,
      static_cast<int64_t>(found->second.size() / sizeof(float)));
  ggml_set_input(factors);
  instance_.uploads.push_back(Upload{factors, found->second});
  rope_factor_tensors_.emplace(inv_freq, factors);
  return factors;
}

void GraphBuilder::lower_rope(const Node& node) {
  // ggml_rope_ext reads strided rows; rotating the [D, heads, tokens] view
  // of a transposed projection then needs no copy.
  const TensorRef input = rows_arg(node, "input", 0);
  const TensorRef positions = tensor_arg(node, "position_ids", 1);
  ggml_tensor* factors =
      rope_factor_tensor(require_arg(node, "inv_freq", 2).arg.as_tensor().id);
  const bool interleaved = bool_arg(node, "interleaved", 3, false);
  const double scale = float_arg(node, "attention_scale", 4, 1.0, true);
  const int n_dims = 2 * static_cast<int>(ggml_nelements(factors));
  if (input.sizes.size() != 4 || positions.sizes.size() != 2 ||
      positions.sizes[0] != 1 || positions.sizes[1] != input.sizes[2] ||
      n_dims > input.sizes[3]) {
    throw std::runtime_error(
        "ggml: rope expects input [B,H,T,D] and position_ids [1,T]");
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* pos = ggml_reshape_1d(
      ctx, contiguous(as_i32(positions.tensor)), positions.sizes[1]);
  // ggml_rope_ext rotates [D, heads, tokens, B]; torch order is [B,H,T,D].
  ggml_tensor* rotated = ggml_rope_ext(
      ctx,
      ggml_permute(ctx, input.tensor, 0, 2, 1, 3),
      pos,
      factors,
      n_dims,
      interleaved ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX,
      /*n_ctx_orig=*/0,
      /*freq_base=*/1.0f,
      /*freq_scale=*/1.0f,
      /*ext_factor=*/0.0f,
      static_cast<float>(scale),
      /*beta_fast=*/0.0f,
      /*beta_slow=*/0.0f);
  set_single_view(
      node, TensorRef{ggml_permute(ctx, rotated, 0, 2, 1, 3), input.sizes});
}

// set_rows into an F16 destination and attention over F16 keys round F32
// values exactly like a cast, so they can read a fresh F32 -> F16 cast's
// source directly. ggml_cast is a CPY whose destination is its own result.
// Skipping the cast moves the read of its source to the consumer, so a
// source in a persistent buffer or in storage the graph writes in place
// keeps its cast.
ggml_tensor* GraphBuilder::skip_f16_cast(ggml_tensor* rows) const {
  ggml_tensor* source = rows->src[0];
  if (rows->type != GGML_TYPE_F16 || rows->op != GGML_OP_CPY ||
      rows->src[1] != rows || source->type != GGML_TYPE_F32 ||
      !ggml_are_same_shape(rows, source) || !ggml_is_contiguous_rows(source)) {
    return rows;
  }
  const ggml_tensor* root = storage_root(source);
  return persistent_storage_.contains(root) || written_storage_.contains(root)
      ? rows
      : source;
}

void GraphBuilder::lower_index_put(const Node& node) {
  const TensorRef self = tensor_arg(node, "self", 0);
  const Argument& indices_arg = require_arg(node, "indices", 1).arg;
  // set_rows reads rows with any outer strides, so the values may be a
  // deferred permutation.
  const TensorRef values = rows_arg(node, "values", 2);
  const size_t rank = self.sizes.size();
  const std::vector<ValueId> indices =
      indices_arg.kind() == ArgKind::OptionalTensorList
      ? indices_arg.as_optional_tensor_list().ids
      : std::vector<ValueId>{};
  const auto is_index = [](ValueId id) { return valid(id); };
  const bool rows_only = rank >= 2 && indices.size() == rank - 1 &&
      is_index(indices.back()) &&
      std::none_of(indices.begin(), indices.end() - 1, is_index);
  const TensorRef index = rows_only ? tensor(indices.back()) : TensorRef{};
  std::vector<int64_t> row_sizes = self.sizes;
  if (rows_only && index.sizes.size() == 1) {
    row_sizes[rank - 2] = index.sizes[0];
  }
  if (!rows_only || index.sizes.size() != 1 || values.sizes != row_sizes ||
      bool_arg(node, "accumulate", 3, false)) {
    throw std::runtime_error(
        "ggml: index_put_ only supports writing whole rows along dim -2");
  }
  const ScalarType index_dtype =
      method_.graph.value(indices.back()).tensor_meta().dtype;
  if (index_dtype != ScalarType::Long && index_dtype != ScalarType::Int) {
    throw std::runtime_error("ggml: index_put_ indices must be int64 or int32");
  }
  check_row_indices(indices.back(), self.sizes[rank - 2]);
  ggml_tensor* rows = values.tensor->type == GGML_TYPE_F16
      ? values.tensor
      : as_f32(values.tensor);
  if (self.tensor->type == GGML_TYPE_F16) {
    rows = skip_f16_cast(rows);
  }
  invalidate_cached_transforms(self.tensor);
  mark_written(self.tensor);
  ggml_tensor* result = ggml_set_rows(
      instance_.context.get(), self.tensor, rows, as_i32(index.tensor));
  expand_write(result);
  set_single_output(node, TensorRef{result, self.sizes});
}

ggml_tensor*
GraphBuilder::attention_mask(ValueId id, int64_t tokens, int64_t keys) {
  const TensorRef mask = tensor(id);
  const size_t rank = mask.sizes.size();
  if (rank < 2 || rank > 4 ||
      (mask.sizes[rank - 2] != tokens && mask.sizes[rank - 2] != 1) ||
      mask.sizes.back() != keys) {
    throw std::runtime_error("ggml: attention mask must end in [T|1, S]");
  }
  // Out-of-place repeat/scale results, and views of them, are stable until
  // an explicit graph write to their storage (which invalidates this cache).
  // Reuse their additive F16 mask across attention ops; never cache
  // writable inputs.
  const ggml_tensor* root = storage_root(mask.tensor);
  const bool reusable = root->op == GGML_OP_REPEAT || root->op == GGML_OP_SCALE;
  const auto cache_key = std::make_pair(id, tokens);
  if (reusable) {
    const auto found = attention_masks_.find(cache_key);
    if (found != attention_masks_.end()) {
      return found->second;
    }
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* additive = mask.tensor;
  if (method_.graph.value(id).tensor_meta().dtype == ScalarType::Bool) {
    // max - max is exactly 0; -max saturates to -inf in F16.
    constexpr float kMax = std::numeric_limits<float>::max();
    additive =
        ggml_scale_bias(ctx, contiguous(as_f32(mask.tensor)), kMax, -kMax);
  }
  // flash_attn_ext broadcasts the mask over heads and batch but reads one
  // row per query.
  if (mask.sizes[rank - 2] != tokens) {
    std::vector<int64_t> rows = mask.sizes;
    rows[rank - 2] = tokens;
    additive = ggml_repeat(
        ctx, as_f32(additive), new_tensor(ctx, GGML_TYPE_F32, rows));
  }
  // flash_attn_ext needs a contiguous F16 mask.
  ggml_tensor* result = additive->type == GGML_TYPE_F16
      ? contiguous(additive)
      : ggml_cast(ctx, additive, GGML_TYPE_F16);
  if (reusable) {
    attention_masks_.emplace(cache_key, result);
  }
  return result;
}

void GraphBuilder::lower_sdpa(const Node& node) {
  // flash_attn_ext takes strided q/k/v with contiguous rows, e.g. the
  // head-transposed projections, without copies.
  const TensorRef query = rows_arg(node, "query", 0);
  const TensorRef key = rows_arg(node, "key", 1);
  const TensorRef val = rows_arg(node, "value", 2);
  const bool is_causal = bool_arg(node, "is_causal", 5, false);
  if (query.sizes.size() != 4 || key.sizes.size() != 4 ||
      val.sizes.size() != 4 ||
      float_arg(node, "dropout_p", 4, 0.0, true) != 0.0) {
    throw std::runtime_error(
        "ggml: scaled_dot_product_attention expects [B,H,T,D] without dropout");
  }
  if (key.sizes[0] != query.sizes[0] || val.sizes[0] != query.sizes[0] ||
      key.sizes[1] != val.sizes[1] || key.sizes[1] <= 0 ||
      query.sizes[1] % key.sizes[1] != 0 || key.sizes[2] != val.sizes[2] ||
      key.sizes[3] != query.sizes[3]) {
    throw std::runtime_error(
        "ggml: scaled_dot_product_attention q/k/v shapes are incompatible");
  }
  const int64_t tokens = query.sizes[2];
  int64_t keys = key.sizes[2];
  ggml_tensor* k = key.tensor;
  ggml_tensor* v = val.tensor;
  ggml_tensor* mask = nullptr;
  const NamedArgument* mask_arg = find_arg(node, "attn_mask", 3);
  if (mask_arg != nullptr && mask_arg->arg.kind() == ArgKind::Tensor &&
      valid(mask_arg->arg.as_tensor().id)) {
    if (is_causal) {
      throw std::runtime_error("ggml: sdpa cannot take a mask and is_causal");
    }
    const ValueId mask_id = mask_arg->arg.as_tensor().id;
    if (const auto extent = key_extents_.find(mask_id);
        extent != key_extents_.end()) {
      // Every query masks the keys past the extent: attend the prefix.
      keys = extent->second;
      const auto prefix = [&](ggml_tensor* t) {
        return ggml_view_4d(
            instance_.context.get(),
            t,
            t->ne[0],
            keys,
            t->ne[2],
            t->ne[3],
            t->nb[1],
            t->nb[2],
            t->nb[3],
            0);
      };
      k = prefix(k);
      v = prefix(v);
    }
    mask = attention_mask(mask_id, tokens, keys);
  } else if (is_causal) {
    mask = causal_mask(tokens, keys, 0).tensor;
  }
  const double default_scale =
      1.0 / std::sqrt(static_cast<double>(query.sizes.back()));
  const float scale =
      static_cast<float>(float_arg(node, "scale", 6, default_scale, true));
  ggml_context* ctx = instance_.context.get();
  // The kernels read the query as F32 and round it to the F16 key's dot
  // product type themselves (CPU vec_dot, Metal and CUDA shared memory), so
  // an F32 -> F16 cast that only makes the query match an F16 cache for
  // torch's SDPA is the identity on the result. Use its F32 source, where
  // skip_f16_cast allows, rather than round trip through F16.
  ggml_tensor* q = query.tensor;
  if (key.tensor->type == GGML_TYPE_F16) {
    q = skip_f16_cast(q);
  }
  ggml_tensor* result =
      ggml_flash_attn_ext(ctx, as_f32(q), k, v, mask, scale, 0.0f, 0.0f);
  ggml_prec_set_acc(result, GGML_PREC_F32);
  // flash_attn_ext returns [Dv, H, T, B]; torch order is [B, H, T, Dv].
  set_single_view(
      node,
      TensorRef{
          ggml_permute(ctx, result, 0, 2, 1, 3),
          {query.sizes[0], query.sizes[1], tokens, val.sizes[3]}});
}

} // namespace ptn::ggml_detail
