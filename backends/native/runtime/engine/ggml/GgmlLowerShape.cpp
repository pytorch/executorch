// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>

namespace ptn::ggml_detail {

// pad lists (left, right) pairs from the last dim inward, which is ggml's
// ne order.
void GraphBuilder::lower_constant_pad(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const std::vector<int64_t> pad = int_list(require_arg(node, "pad", 1).arg);
  const size_t rank = input.sizes.size();
  if (pad.size() % 2 != 0 || pad.size() / 2 > rank || rank > 4) {
    throw std::runtime_error("ggml: invalid constant_pad_nd padding");
  }
  std::array<int, 8> amounts{};
  std::vector<int64_t> sizes = input.sizes;
  for (size_t i = 0; i < pad.size(); ++i) {
    if (pad[i] < 0) {
      throw std::runtime_error("ggml: constant_pad_nd cannot crop");
    }
    amounts[i] = static_cast<int>(pad[i]);
    sizes[rank - 1 - i / 2] += pad[i];
  }
  if (std::ranges::all_of(amounts, [](int p) { return p == 0; })) {
    set_single_output(node, TensorRef{materialize(input.tensor), input.sizes});
    return;
  }
  if (float_arg(node, "value", 2, 0.0, true) != 0.0) {
    throw std::runtime_error("ggml: constant_pad_nd only pads with zeros");
  }
  // Exact in F32; set() rounds an F16 or BF16 result back.
  ggml_tensor* x = half_as_f32(input.tensor);
  require_f32(node, x);
  ggml_tensor* result = ggml_pad_ext(
      instance_.context.get(),
      contiguous(x),
      amounts[0],
      amounts[1],
      amounts[2],
      amounts[3],
      amounts[4],
      amounts[5],
      amounts[6],
      amounts[7]);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_reshape(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  if (op_key(node.target) == "aten.view_copy.default") {
    input.tensor = snapshot(input.tensor);
  }
  std::vector<int64_t> sizes = int_list(require_arg(node, "shape", 1).arg);
  int64_t inferred = -1;
  size_t known = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == -1) {
      if (inferred != -1) {
        throw std::runtime_error("ggml: reshape has multiple inferred dims");
      }
      inferred = static_cast<int64_t>(i);
    } else if (sizes[i] < 0) {
      throw std::runtime_error("ggml: reshape has invalid dimension");
    } else {
      known *= static_cast<size_t>(sizes[i]);
    }
  }
  if (inferred >= 0) {
    const size_t total = checked_numel(input.sizes);
    if (known == 0 || total % known != 0) {
      throw std::runtime_error("ggml: reshape cannot infer dimension");
    }
    sizes[static_cast<size_t>(inferred)] = static_cast<int64_t>(total / known);
  }
  if (checked_numel(sizes) != checked_numel(input.sizes)) {
    throw std::runtime_error("ggml: reshape changes the element count");
  }
  set_single_output(
      node,
      TensorRef{reshape(instance_.context.get(), input.tensor, sizes), sizes});
}

void GraphBuilder::lower_permute(const Node& node) {
  const TensorRef input = view_arg(node, "self", 0);
  const std::vector<int64_t> dims = int_list(require_arg(node, "dims", 1).arg);
  if (dims.size() != input.sizes.size() || dims.size() > 4) {
    throw std::runtime_error("ggml: invalid permutation rank");
  }
  std::array<int, 4> axes{0, 1, 2, 3};
  std::vector<int64_t> sizes(dims.size());
  std::vector<bool> seen(dims.size());
  for (size_t out_dim = 0; out_dim < dims.size(); ++out_dim) {
    const int64_t src_dim = normalize_dim(dims[out_dim], dims.size());
    if (seen[static_cast<size_t>(src_dim)]) {
      throw std::runtime_error("ggml: repeated permutation dimension");
    }
    seen[static_cast<size_t>(src_dim)] = true;
    sizes[out_dim] = input.sizes[static_cast<size_t>(src_dim)];
  }
  // ggml_permute maps source axis i to result axis axes[i].
  for (size_t out_dim = 0; out_dim < dims.size(); ++out_dim) {
    const int64_t src_dim = normalize_dim(dims[out_dim], dims.size());
    axes[dims.size() - 1 - static_cast<size_t>(src_dim)] =
        static_cast<int>(dims.size() - 1 - out_dim);
  }
  ggml_tensor* result = ggml_permute(
      instance_.context.get(),
      input.tensor,
      axes[0],
      axes[1],
      axes[2],
      axes[3]);
  set_single_view(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_transpose(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const int64_t d0 =
      normalize_dim(int_arg(node, "dim0", 1, 0), input.sizes.size());
  const int64_t d1 =
      normalize_dim(int_arg(node, "dim1", 2, 0), input.sizes.size());
  std::vector<int64_t> dims(input.sizes.size());
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[static_cast<size_t>(d0)], dims[static_cast<size_t>(d1)]);
  Node synthetic = node;
  synthetic.inputs = {
      {.name = "self", .arg = require_arg(node, "self", 0).arg},
      {.name = "dims", .arg = IntListArg{dims, {}}},
  };
  lower_permute(synthetic);
}

void GraphBuilder::lower_copy_or_alias(const Node& node, bool alias) {
  const TensorRef input = tensor_arg(node, "self", 0);
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* in = input.tensor;
  ggml_tensor* result = nullptr;
  if (alias) {
    // ggml_view_tensor sets only view_src, so the graph would not reach the
    // producer of the aliased tensor.
    result = ggml_view_4d(
        ctx,
        in,
        in->ne[0],
        in->ne[1],
        in->ne[2],
        in->ne[3],
        in->nb[1],
        in->nb[2],
        in->nb[3],
        0);
  } else {
    result = materialize(in);
  }
  set_single_output(node, TensorRef{result, input.sizes});
}

void GraphBuilder::lower_slice(const Node& node, bool copy) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const int64_t dim =
      normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
  int64_t start = int_arg(node, "start", 2, 0, true);
  int64_t end =
      int_arg(node, "end", 3, input.sizes[static_cast<size_t>(dim)], true);
  const int64_t step = int_arg(node, "step", 4, 1, true);
  if (step != 1) {
    throw std::runtime_error("ggml: slice step must be one");
  }
  const int64_t extent = input.sizes[static_cast<size_t>(dim)];
  start = std::clamp(start < 0 ? start + extent : start, int64_t{0}, extent);
  end = std::clamp(end < 0 ? end + extent : end, start, extent);
  std::vector<int64_t> sizes = input.sizes;
  sizes[static_cast<size_t>(dim)] = end - start;
  const size_t gdim = sizes.size() - 1 - static_cast<size_t>(dim);
  ggml_tensor* result = view_with_shape(
      instance_.context.get(),
      input.tensor,
      sizes,
      static_cast<size_t>(start) * input.tensor->nb[gdim]);
  if (copy) {
    result = materialize(result);
  }
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_select(const Node& node, bool copy) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const int64_t dim =
      normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
  int64_t index = int_arg(node, "index", 2, 0);
  const int64_t extent = input.sizes[static_cast<size_t>(dim)];
  index = index < 0 ? index + extent : index;
  if (index < 0 || index >= extent) {
    throw std::runtime_error("ggml: select index is out of range");
  }
  std::vector<int64_t> view_sizes = input.sizes;
  view_sizes[static_cast<size_t>(dim)] = 1;
  const size_t gdim = view_sizes.size() - 1 - static_cast<size_t>(dim);
  ggml_tensor* result = view_with_shape(
      instance_.context.get(),
      input.tensor,
      view_sizes,
      static_cast<size_t>(index) * input.tensor->nb[gdim]);
  result = copy ? materialize(result) : contiguous(result);
  std::vector<int64_t> sizes = input.sizes;
  sizes.erase(sizes.begin() + dim);
  result = reshape(instance_.context.get(), result, sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_unsqueeze(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  if (op_key(node.target) == "aten.unsqueeze_copy.default") {
    input.tensor = snapshot(input.tensor);
  }
  int64_t dim = int_arg(node, "dim", 1, 0);
  const int64_t rank = static_cast<int64_t>(input.sizes.size() + 1);
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::runtime_error("ggml: unsqueeze dimension is out of range");
  }
  std::vector<int64_t> sizes = input.sizes;
  sizes.insert(sizes.begin() + dim, 1);
  set_single_output(
      node,
      TensorRef{reshape(instance_.context.get(), input.tensor, sizes), sizes});
}

void GraphBuilder::lower_squeeze(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  if (op_key(node.target) == "aten.squeeze_copy.dims") {
    input.tensor = snapshot(input.tensor);
  }
  std::vector<int64_t> dims = int_list(require_arg(node, "dim", 1).arg);
  std::vector<bool> remove(input.sizes.size());
  for (int64_t dim : dims) {
    dim = normalize_dim(dim, input.sizes.size());
    if (input.sizes[static_cast<size_t>(dim)] == 1) {
      remove[static_cast<size_t>(dim)] = true;
    }
  }
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < input.sizes.size(); ++i) {
    if (!remove[i]) {
      sizes.push_back(input.sizes[i]);
    }
  }
  set_single_output(
      node,
      TensorRef{reshape(instance_.context.get(), input.tensor, sizes), sizes});
}

void GraphBuilder::lower_expand(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  std::vector<int64_t> sizes = int_list(require_arg(node, "size", 1).arg);
  if (sizes.size() < input.sizes.size()) {
    throw std::runtime_error("ggml: expand cannot drop dimensions");
  }
  // A mask expanded over its key prefix only (see key_extents_).
  if (const std::optional<int64_t> extent = key_extent_of(node)) {
    sizes.back() = *extent;
  }
  const size_t offset = sizes.size() - input.sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int64_t in = i < offset ? 1 : input.sizes[i - offset];
    if (sizes[i] == -1 && i >= offset) {
      sizes[i] = in;
    }
    if (sizes[i] < 0 || (in != 1 && in != sizes[i])) {
      throw std::runtime_error("ggml: expand size is incompatible with input");
    }
  }
  ggml_tensor* shape =
      new_tensor(instance_.context.get(), input.tensor->type, sizes);
  ggml_tensor* result =
      ggml_repeat(instance_.context.get(), input.tensor, shape);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_repeat(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  if (input.tensor == nullptr) {
    throw std::runtime_error("ggml: repeat input has no tensor");
  }
  std::vector<int64_t> sizes = int_list(require_arg(node, "repeats", 1).arg);
  if (sizes.size() < input.sizes.size() || sizes.size() > 4) {
    throw std::runtime_error(
        "ggml: repeat needs input rank <= repeats rank <= 4");
  }
  const size_t offset = sizes.size() - input.sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int64_t extent = i < offset ? 1 : input.sizes[i - offset];
    if (extent <= 0 || sizes[i] <= 0 ||
        sizes[i] > std::numeric_limits<int64_t>::max() / extent) {
      throw std::runtime_error("ggml: repeat requires positive, bounded sizes");
    }
    sizes[i] *= extent;
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* shape = new_tensor(ctx, input.tensor->type, sizes);
  ggml_tensor* result = ggml_repeat(ctx, input.tensor, shape);
  // A no-op repeat must still snapshot storage that can be written later.
  set_single_output(node, TensorRef{snapshot(result), std::move(sizes)});
}

void GraphBuilder::lower_cat(const Node& node) {
  const Argument& tensors_arg = require_arg(node, "tensors", 0).arg;
  if (tensors_arg.kind() != ArgKind::TensorList) {
    throw std::runtime_error("ggml: cat expects a tensor list");
  }
  const std::vector<ValueId>& ids = tensors_arg.as_tensor_list().ids;
  if (ids.empty()) {
    throw std::runtime_error("ggml: cat tensor list is empty");
  }
  TensorRef result = tensor(ids.front());
  const int64_t dim =
      normalize_dim(int_arg(node, "dim", 1, 0, true), result.sizes.size());
  const int gdim = static_cast<int>(result.sizes.size() - 1 - dim);
  // Producers may carry one dtype as different ggml types (an F16 linear
  // result is F32), so mismatched operands take the output's type. ggml
  // converts I32 only to F32.
  const auto to_output_type = [&](ggml_tensor* t) {
    ggml_context* ctx = instance_.context.get();
    const ggml_type type =
        compute_type(method_.graph.value(output_ids(node).at(0)).tensor_meta());
    if (t->type == GGML_TYPE_I32 && type != GGML_TYPE_I32) {
      t = ggml_cast(ctx, t, GGML_TYPE_F32);
    }
    return t->type == type ? t : ggml_cast(ctx, t, type);
  };
  for (size_t i = 1; i < ids.size(); ++i) {
    TensorRef next = tensor(ids[i]);
    bool compatible = next.sizes.size() == result.sizes.size();
    for (size_t d = 0; compatible && d < next.sizes.size(); ++d) {
      compatible =
          d == static_cast<size_t>(dim) || next.sizes[d] == result.sizes[d];
    }
    if (!compatible) {
      throw std::runtime_error(
          "ggml: cat inputs must match in rank and every other dim");
    }
    if (next.tensor->type != result.tensor->type) {
      result.tensor = to_output_type(result.tensor);
      next.tensor = to_output_type(next.tensor);
    }
    result.tensor =
        ggml_concat(instance_.context.get(), result.tensor, next.tensor, gdim);
    result.sizes[static_cast<size_t>(dim)] +=
        next.sizes.at(static_cast<size_t>(dim));
  }
  set_single_output(node, std::move(result));
}

void GraphBuilder::lower_split(const Node& node, const std::string& key) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const size_t rank = input.sizes.size();
  const auto dim = static_cast<size_t>(
      normalize_dim(int_arg(node, "dim", 2, 0, true), rank));
  const int64_t extent = input.sizes[dim];
  if (node.outputs.size() != 1 ||
      node.outputs.front().kind != OutputValueKind::TensorList) {
    throw std::runtime_error("ggml: split must produce one tensor per piece");
  }
  const size_t outputs = node.outputs.front().elem_ids.size();
  std::vector<int64_t> pieces;
  if (key == "aten.split_with_sizes_copy.default" ||
      key == "aten.split_with_sizes.default") {
    pieces = int_list(require_arg(node, "split_sizes", 1).arg);
  } else {
    const int64_t size = int_arg(node, "split_size", 1, 0);
    if (size <= 0) {
      throw std::runtime_error("ggml: split size must be positive");
    }
    // Counted before the pieces are listed, so a huge extent costs nothing.
    if (static_cast<uint64_t>(extent / size + (extent % size != 0)) !=
        outputs) {
      throw std::runtime_error("ggml: split must produce one tensor per piece");
    }
    for (int64_t start = 0; start < extent; start += size) {
      pieces.push_back(std::min(size, extent - start));
    }
  }
  if (pieces.size() != outputs) {
    throw std::runtime_error("ggml: split must produce one tensor per piece");
  }
  int64_t total = 0;
  for (const int64_t size : pieces) {
    if (size < 0 || __builtin_add_overflow(total, size, &total)) {
      throw std::runtime_error(
          "ggml: split sizes must add up to the dimension's size");
    }
  }
  if (total != extent) {
    throw std::runtime_error(
        "ggml: split sizes must add up to the dimension's size");
  }
  const size_t gdim = rank - 1 - dim;
  int64_t start = 0;
  for (size_t i = 0; i < pieces.size(); ++i) {
    std::vector<int64_t> sizes = input.sizes;
    sizes[dim] = pieces[i];
    if (const ValueId id = node.outputs.front().elem_ids[i]; valid(id)) {
      set_view(
          id,
          TensorRef{
              view_with_shape(
                  instance_.context.get(),
                  input.tensor,
                  sizes,
                  static_cast<size_t>(start) * input.tensor->nb[gdim]),
              sizes});
    }
    start += pieces[i];
  }
}

// result[..., j] = table[..., indices[..., j]] for an F32 or I32 table and
// I32 indices with the same leading dims: each table row is a batch of
// one-element rows for ggml_get_rows. Returns [1, k, rows].
ggml_tensor* GraphBuilder::last_dim_gather(
    ggml_tensor* table,
    ggml_tensor* indices) {
  ggml_context* ctx = instance_.context.get();
  table = contiguous(table);
  indices = contiguous(indices);
  const int64_t rows = ggml_nrows(table);
  return ggml_get_rows(
      ctx,
      ggml_reshape_3d(ctx, table, 1, table->ne[0], rows),
      ggml_reshape_2d(ctx, indices, indices->ne[0], rows));
}

// Always sorted. Ties may come out in another order than PyTorch's.
void GraphBuilder::lower_topk(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const size_t rank = input.sizes.size();
  if (rank == 0 ||
      normalize_dim(int_arg(node, "dim", 2, -1, true), rank) !=
          static_cast<int64_t>(rank) - 1) {
    throw std::runtime_error("ggml: topk supports the last dimension only");
  }
  const int64_t k = int_arg(node, "k", 1, 0);
  if (k < 1 || k > input.sizes.back()) {
    throw std::runtime_error(
        "ggml: topk needs k between one and the dimension's size");
  }
  if (node.outputs.size() != 2) {
    throw std::runtime_error("ggml: topk must produce values and indices");
  }
  require_exact_integers(tensor_arg_id(node, "self", 0), "topk's self");
  ggml_context* ctx = instance_.context.get();
  const bool largest = bool_arg(node, "largest", 3, true);
  ggml_tensor* values = contiguous(as_f32(input.tensor));
  const int64_t n = values->ne[0];
  std::vector<int64_t> sizes = input.sizes;
  sizes.back() = k;
  if (!exact_topk_) {
    // llama.cpp's ggml_argsort_top_k: one argsort, its first k and their
    // values. ggml's argsort compares values as they are, so a row holding
    // NaN comes out in an unspecified order, its other values included.
    ggml_tensor* rows = ggml_reshape_2d(ctx, values, n, ggml_nrows(values));
    ggml_tensor* ids = nullptr;
    if (largest) {
      ids = ggml_argsort_top_k(ctx, rows, static_cast<int>(k));
    } else {
      ggml_tensor* order = ggml_argsort(ctx, rows, GGML_SORT_ORDER_ASC);
      ids = ggml_view_2d(ctx, order, k, order->ne[1], order->nb[1], 0);
    }
    if (const ValueId id = node.outputs[0].value_id; valid(id)) {
      ggml_tensor* picked = ggml_get_rows(
          ctx, ggml_reshape_3d(ctx, rows, 1, n, rows->ne[1]), ids);
      set_as_output_type(id, TensorRef{reshape(ctx, picked, sizes), sizes});
    }
    if (const ValueId id = node.outputs[1].value_id; valid(id)) {
      set(id, TensorRef{reshape(ctx, ids, sizes), sizes});
    }
    return;
  }
  // Eager ranks NaN above every value, +inf included. A first sort ranks NaN
  // as +inf. A second sort moves each row's NaNs ahead of the rest (behind it
  // for the smallest), keeping the first order within both: position j sorts
  // as j - n for a NaN when largest, j + n when smallest. Both stay within
  // +-2^24, which F32 holds exactly, for rows of up to 2^23 values.
  constexpr int64_t kMaxRow = kExactF32Integer / 2;
  if (n > kMaxRow) {
    throw std::runtime_error("ggml: topk supports rows of up to 2^23 values");
  }
  const int64_t count = ggml_nelements(values);
  const auto rows_like = [&](ggml_tensor* t) {
    return ggml_reshape_4d(
        ctx, t, values->ne[0], values->ne[1], values->ne[2], values->ne[3]);
  };
  ggml_tensor* infinities = ggml_fill(
      ctx,
      new_tensor(ctx, GGML_TYPE_F32, input.sizes),
      std::numeric_limits<float>::infinity());
  ggml_tensor* keys = rows_like(ggml_get_rows(
      ctx,
      ggml_concat(
          ctx,
          ggml_reshape_3d(ctx, values, 1, 1, count),
          ggml_reshape_3d(ctx, infinities, 1, 1, count),
          /*dim=*/1),
      ggml_cast(
          ctx,
          ggml_reshape_2d(ctx, nan_mask(values), 1, count),
          GGML_TYPE_I32)));
  ggml_tensor* first = ggml_argsort(
      ctx, keys, largest ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC);
  ggml_tensor* sorted = rows_like(last_dim_gather(values, first));
  ggml_tensor* positions = ggml_repeat(
      ctx, ggml_arange(ctx, 0.0f, static_cast<float>(n), 1.0f), sorted);
  ggml_tensor* second = ggml_argsort(
      ctx,
      ggml_add(
          ctx,
          positions,
          ggml_scale(
              ctx, nan_mask(sorted), static_cast<float>(largest ? -n : n))),
      GGML_SORT_ORDER_ASC);
  ggml_tensor* order = ggml_cast(
      ctx,
      rows_like(last_dim_gather(ggml_cast(ctx, first, GGML_TYPE_F32), second)),
      GGML_TYPE_I32);
  ggml_tensor* indices = ggml_cont(
      ctx,
      ggml_view_4d(
          ctx,
          order,
          k,
          order->ne[1],
          order->ne[2],
          order->ne[3],
          order->nb[1],
          order->nb[2],
          order->nb[3],
          0));
  if (const ValueId id = node.outputs[0].value_id; valid(id)) {
    set_as_output_type(
        id,
        TensorRef{
            reshape(ctx, last_dim_gather(values, indices), sizes), sizes});
  }
  if (const ValueId id = node.outputs[1].value_id; valid(id)) {
    set(id, TensorRef{reshape(ctx, indices, sizes), sizes});
  }
}

} // namespace ptn::ggml_detail
