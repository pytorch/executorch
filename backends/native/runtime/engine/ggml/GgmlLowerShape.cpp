// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <array>
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
  require_f32(node, input.tensor);
  ggml_tensor* result = ggml_pad_ext(
      instance_.context.get(),
      contiguous(input.tensor),
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

} // namespace ptn::ggml_detail
