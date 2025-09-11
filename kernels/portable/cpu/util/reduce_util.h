/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <cstring>
#include <tuple>

namespace torch {
namespace executor {
namespace {

template <typename Fn>
void apply_on_flat_ix_with_stride_and_base(
    const Fn& fn,
    const size_t stride,
    const size_t base,
    const size_t start,
    const size_t end) {
  for (size_t i = start; i <= end; i++) {
    fn(base + i * stride);
  }
}

template <typename Fn>
void apply_on_flat_and_dim_ix_with_stride_and_base(
    const Fn& fn,
    const size_t stride,
    const size_t base,
    const size_t start,
    const size_t end) {
  for (size_t i = start; i <= end; i++) {
    fn(base + i * stride, i);
  }
}

template <typename Fn>
void apply_on_flat_ix_with_dim_mask_and_base(
    const Fn& fn,
    const Tensor& in,
    const bool* dim_mask,
    const size_t base,
    const size_t start,
    const size_t end) {
  // Compute innermost dim from dim list
  int64_t inner_dim = in.dim() - 1;
  while (!dim_mask[inner_dim]) {
    inner_dim--;
  }

  // Initialize array of indices per dimension. This array is used to maintain
  // the per-dimension index of the element in `in` that is being reduced over
  // Only the dims that are in the dim list are relevant.
  int64_t dim_index[kTensorDimensionLimit];
  for (int64_t d = 0; d < in.dim(); d++) {
    dim_index[d] = 0;
  }

  // Gather strides
  const auto strides = in.strides();

  // curr_index will always be index of the element from `in` we are currently
  // reducing. Initialized to the first index from `in` that maps to `out_ix`
  size_t curr_index = base;

  size_t apply_fun_counter = 0;
  while (true) {
    // Apply reduction to current index
    if (apply_fun_counter >= start && apply_fun_counter <= end) {
      fn(curr_index);
    }
    apply_fun_counter += 1;
    if (apply_fun_counter > end) {
      return;
    }

    // Next index to reduce. Increase dim_index[inner_dim] by 1, and curr_index
    // by strides[inner_dim].
    dim_index[inner_dim]++;
    curr_index += strides[inner_dim];

    // Check if we have reached the end of the innermost dimension
    if (dim_index[inner_dim] == in.size(inner_dim)) {
      // If we reached the end, we need to update the indices in dim_index. We
      // do this by resetting dim_index[inner_dim] to 0, and then incrementing
      // the index of the next innermost dimension from the dim list by 1.
      // If when we do this increment, we also reach the end of that dimension,
      // we need to keep repeating that procedure.
      // This is similar to doing the carry over when adding 1 to a number.

      // curr_dim will be the dim from the dim list we are currently updating
      int64_t curr_dim = inner_dim;

      while (dim_index[curr_dim] == in.size(curr_dim)) {
        if (curr_dim == 0) {
          // Exit function if we've reached the end of the outermost dimension
          return;
        }
        // Reset dim_index[curr_dim] to 0. We need to update curr_index
        // accordingly. Reseting dim_index[curr_dim] from in.size(curr_dim)
        // to 0 means we need to subtract in.size(curr_dim) * strides[curr_dim]
        // from curr_index. However in.size(curr_dim) * strides[curr_dim] is
        // equal to strides[curr_dim - 1]. Notice that curr_dim > 0 at this
        // point in the execution
        dim_index[curr_dim] = 0;
        curr_index -= strides[curr_dim - 1];

        // Decrease current dim
        curr_dim--;
        while (curr_dim >= 0) {
          // Stop if curr_dim is in the dim list
          if (dim_mask[curr_dim]) {
            break;
          }
          // Keep decreasing if curr_dim is not in the dim list
          curr_dim--;
        }
        // Exit function if curr_dim was decreased to -1. This means we have
        // reduced over all the elements we needed to.
        if (curr_dim < 0) {
          return;
        }

        // At this point in the execution, curr_dim is the next innermost
        // dimension. Increase dim_index[curr_dim] by 1 and update curr_index
        // accordingly.
        dim_index[curr_dim]++;
        curr_index += strides[curr_dim];
      }
    }
  }
}

} // namespace

//
// Helper Functions
//

ET_NODISCARD bool check_dim_list_is_valid(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list);

bool check_dim_in_dim_list(
    const size_t dim,
    const size_t max_dim,
    const executorch::aten::ArrayRef<int64_t>& dim_list);

size_t get_reduced_dim_product(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim);

size_t get_reduced_dim_product(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list);

// Resolve ambiguity between the above two overloads -- ArrayRef and
// optional are both implicitly constructible from int64_t.
inline size_t get_reduced_dim_product(
    const executorch::aten::Tensor& in,
    int64_t dim) {
  return get_reduced_dim_product(in, std::optional<int64_t>(dim));
}

size_t get_out_numel(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim);

size_t get_out_numel(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list);

// Resolve ambiguity between the above two overloads -- ArrayRef and
// optional are both implicitly constructible from int64_t.
inline size_t get_out_numel(const executorch::aten::Tensor& in, int64_t dim) {
  return get_out_numel(in, std::optional<int64_t>(dim));
}

size_t get_init_index(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    const size_t out_ix);

size_t get_init_index(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    const size_t out_ix);

inline size_t get_init_index(
    const executorch::aten::Tensor& in,
    int64_t dim,
    const size_t out_ix) {
  return get_init_index(in, std::optional<int64_t>(dim), out_ix);
}
//
// Iteration Functions
//

/**
 * Useful to reduce a tensor `in` over a given dimension `dim` using the
 * reduce function `fn`, which should have the following signature:
 * void fn(const size_t size, const size_t stride, const size_t base_ix)
 * where `size` and `stride` are the size and stride of the dimension being
 * reduced and `base_ix` is the index of the first element of the reduction.
 */
template <typename Fn>
void apply_over_dim(
    const Fn& fn,
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim) {
  // If dim is null, apply fn over the entire tensor
  if (!dim.has_value()) {
    fn(in.numel(), 1, 0);
    return;
  }

  if (in.dim() != 0) {
    ET_CHECK_VALID_DIM(dim.value(), in.dim());
  } else {
    // Special handling for 0-D tensor; 0 or -1 is valid for PyTorch code
    // `torch.mean(torch.tensor(2, dtype=float), dim=-1)`
    ET_CHECK(dim.value() == 0 || dim.value() == -1);
    fn(in.numel(), 1, 0);
    return;
  }

  if (in.numel() == 0) {
    return;
  }

  const size_t d = ET_NORMALIZE_IX(dim.value(), in.dim());

  const size_t size = in.size(d);
  const size_t stride = in.strides()[d];
  const size_t outer_size = getLeadingDims(in, d);
  const size_t outer_stride = size * stride;
  // Loop through all outer dimensions
  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    size_t outer = outer_idx * outer_stride;
    // Loop through all inner dimensions
    for (size_t inner_idx = 0; inner_idx < stride; ++inner_idx) {
      size_t base = outer + inner_idx;
      fn(size, stride, base);
    }
  }
}

/**
 * Useful to reduce a tensor `in` over a given dimension `dim` for the output
 * element at index `out_ix` using the reduce function `fn`, which
 * should have the following signature:
 * `void fn(const size_t in_ix, const size_t dim_ix)`
 * where `in_ix` is the flat index of each element from `in` that maps to
 * `out_ix` and `dim_ix` is its index along `dim`.
 */
template <typename Fn>
void apply_over_dim(
    const Fn& fn,
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    const size_t out_ix,
    const int64_t start = 0,
    const int64_t end = -1) {
  if (dim.has_value()) {
    if (in.dim() != 0) {
      ET_CHECK_VALID_DIM(dim.value(), in.dim());
    } else {
      ET_CHECK(dim.value() == 0 || dim.value() == -1);
    }
  }
  ET_CHECK_MSG(
      out_ix < get_out_numel(in, dim),
      "Out index %zd is out of bounds",
      out_ix);

  if (in.numel() == 0) {
    return;
  }

  const size_t iter_length = get_reduced_dim_product(in, dim);
  const size_t normalized_start = ET_NORMALIZE_IX(start, iter_length);
  const size_t normalized_end = ET_NORMALIZE_IX(end, iter_length);
  const size_t ustart = std::max(normalized_start, size_t(0));
  const size_t uend = std::min(normalized_end, iter_length - 1);

  // If dim is null, iterate over the entire tensor
  if (!dim.has_value()) {
    apply_on_flat_and_dim_ix_with_stride_and_base(
        fn, /*stride=*/1, /*base=*/0, ustart, uend);
    return;
  }

  // Compute the starting base index
  const size_t base = get_init_index(in, dim, out_ix);

  // Compute non-negative dimension value from dim value
  const size_t d = ET_NORMALIZE_IX(dim.value(), in.dim());

  if (in.dim() == 0) {
    fn(base, ustart);
  } else {
    apply_on_flat_and_dim_ix_with_stride_and_base(
        fn, in.strides()[d], base, ustart, uend);
  }
}

/**
 * Execution plan for repeated apply_over_dim_list with the same
 * function, input tensor, dim list, start, and end but varying
 * out_ix, as done (via {map_,}reduce_over_dim_list) in reductions.
 */
class ApplyOverDimListPlan {
 public:
  ApplyOverDimListPlan(
      const executorch::aten::Tensor& in,
      // If set, lifetime must last until execute() returns.
      const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
      const int64_t start = 0,
      const int64_t end = -1)
      : dim_list_(dim_list), in_(in) {
    ET_CHECK(check_dim_list_is_valid(in, dim_list));
    out_numel_ = get_out_numel(in_, dim_list);
    if (in.numel() == 0) {
      mode_ = ExecutionMode::NothingToDo;
      return;
    }
    const size_t iter_length = get_reduced_dim_product(in, dim_list);
    const size_t normalized_start = ET_NORMALIZE_IX(start, iter_length);
    const size_t normalized_end = ET_NORMALIZE_IX(end, iter_length);
    ustart_ = std::max(normalized_start, size_t(0));
    uend_ = std::min(normalized_end, iter_length - 1);
    if (!dim_list.has_value() || dim_list.value().size() == 0 ||
        in.dim() == 0) {
      mode_ = ExecutionMode::NoDimMaskOrZeroDimension;
      return;
    }
    dim_list_ = dim_list.value();
    if (dim_list_.value().size() == 1) {
      mode_ = ExecutionMode::OnlyOneDim;
      return;
    }
    is_in_dim_list_.fill(0);
    for (const auto& d : dim_list.value()) {
      const size_t non_neg_d = d < 0 ? d + in.dim() : d;
      is_in_dim_list_[non_neg_d] = true;
    }

    mode_ = ExecutionMode::NormalDimMask;
  }

  template <typename Fn>
  void execute(const Fn& fn, const size_t out_ix) const {
    ET_CHECK_MSG(out_ix < out_numel_, "Out index %zd is out of bounds", out_ix);

    switch (mode_) {
      case ExecutionMode::NothingToDo:
        return;
      case ExecutionMode::NoDimMaskOrZeroDimension:
        apply_on_flat_ix_with_stride_and_base(
            fn, /*stride=*/1, /*base=*/0, ustart_, uend_);
        return;
      case ExecutionMode::OnlyOneDim:
        apply_on_flat_and_dim_ix_with_stride_and_base(
            [&](const auto in_ix, const auto dim_ix) { fn(in_ix); },
            in_.strides()[ET_NORMALIZE_IX(dim_list_.value()[0], in_.dim())],
            get_init_index(in_, dim_list_.value(), out_ix),
            ustart_,
            uend_);
        return;
      case ExecutionMode::NormalDimMask:
        apply_on_flat_ix_with_dim_mask_and_base(
            fn,
            in_,
            is_in_dim_list_.data(),
            get_init_index(in_, dim_list_.value(), out_ix),
            ustart_,
            uend_);
        return;
    }
  }

  const executorch::aten::Tensor& get_input_tensor() const {
    return in_;
  }

  const std::optional<executorch::aten::ArrayRef<int64_t>>& get_dim_list()
      const {
    return dim_list_;
  }

 private:
  // Start argument to apply_on_flat_ix_with_{stride,dim_mask}_and_base.
  size_t ustart_;
  // End argument to apply_on_flat_ix_with_{stride,dim_mask}_and_base.
  size_t uend_;
  enum class ExecutionMode {
    // Empty input, no work to do.
    NothingToDo,
    // Iterate over the entire tensor with
    // apply_on_flat_ix_with_stride_and_base.
    NoDimMaskOrZeroDimension,
    // dim_list has size 1, iterate with
    // apply_on_flat_and_dim_ix_with_stride_and_base
    OnlyOneDim,
    // General mode, iterate with
    // apply_on_flat_ix_with_dim_mask_and_base.
    NormalDimMask
  };
  ExecutionMode mode_;
  size_t out_numel_;
  std::optional<executorch::aten::ArrayRef<int64_t>> dim_list_;
  std::array<bool, kTensorDimensionLimit> is_in_dim_list_;
  const executorch::aten::Tensor& in_;
};

/**
 * Useful to reduce a tensor `in` over a given list of dimensions `dim_list`
 * for the output element at index `out_ix` using the reduce function
 * `fn`, which should have the following signature:
 * `void fn(const size_t in_ix)`
 * where `in_ix` is the index of each element from `in` that maps to `out_ix`
 */
template <typename Fn>
void apply_over_dim_list(
    const Fn& fn,
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    const size_t out_ix,
    const int64_t start = 0,
    const int64_t end = -1) {
  ApplyOverDimListPlan plan(in, dim_list, start, end);
  plan.execute(fn, out_ix);
}

//
// Reduce Functions
//

/**
 * Useful to reduce a tensor `in` over a dimension `dim` for the output element
 * at index `out_ix`, first applying the map `map_fun` to each element of `in`,
 * which should have the signature: CTYPE_OUT map_fun(CTYPE_IN v)
 * and then reducing using `reduce_fun`, which should have the signature:
 * `CTYPE_OUT reduce_fun(CTYPE_OUT val, long ix, CTYPE_OUT acc_val, long
 * acc_ix)`
 *
 * Common usage:
 *
 * CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
 * for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
 *   out_data[out_ix] = map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>(
 *       [](CTYPE_IN v) {
 *         // map operation on `v`, outputs `val`
 *       },
 *       [](CTYPE_OUT val, long ix, CTYPE_OUT acc_val, long acc_ix) {
 *         // reduce operation on `acc_val` and `acc_ix` using `val` and `ix`,
 *         // outputs {`acc_val`, `acc_ix`} pair
 *       in,
 *       dim_list,
 *       out_ix);
 * }
 */
template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename MapOp,
    typename ReduceOp>
std::tuple<CTYPE_OUT, long> map_reduce_over_dim(
    const MapOp& map_fun,
    const ReduceOp& reduce_fun,
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    const size_t out_ix) {
  if (dim.has_value()) {
    if (in.dim() != 0) {
      ET_CHECK_VALID_DIM(dim.value(), in.dim());
    } else {
      ET_CHECK(dim.value() == 0 || dim.value() == -1);
    }
  }

  ET_CHECK_MSG(
      out_ix < get_out_numel(in, dim),
      "Out index %zd is out of bounds",
      out_ix);

  ET_CHECK_MSG(in.numel() > 0, "Input tensor must be nonempty");

  const size_t init_index = get_init_index(in, dim, out_ix);

  const CTYPE_IN* const in_data = in.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT acc_val = map_fun(in_data[init_index]);
  long acc_ix = 0;

  if (in.numel() == 1) {
    return std::tuple<CTYPE_OUT, long>{acc_val, acc_ix};
  }

  apply_over_dim(
      [&acc_val, &acc_ix, reduce_fun, map_fun, in_data](
          const size_t in_ix, const size_t dim_ix) {
        std::tuple<CTYPE_OUT, long> res =
            reduce_fun(map_fun(in_data[in_ix]), dim_ix, acc_val, acc_ix);
        acc_val = std::get<0>(res);
        acc_ix = std::get<1>(res);
      },
      in,
      dim,
      out_ix,
      1,
      -1);

  return std::tuple<CTYPE_OUT, long>{acc_val, acc_ix};
}

/**
 * Execution plan for repeated map_reduce_over_dim_list with the same
 * function, input tensor, and dim_list but varying out_ix.
 */
class MapReduceOverDimListPlan {
 public:
  MapReduceOverDimListPlan(
      const executorch::aten::Tensor& in,
      const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list)
      : plan_(in, dim_list, 1, -1) {
    ET_CHECK_MSG(in.numel() > 0, "Input tensor must be nonempty");
  }

  template <
      typename CTYPE_IN,
      typename CTYPE_OUT,
      typename MapOp,
      typename ReduceOp>
  CTYPE_OUT execute(
      const MapOp& map_fun,
      const ReduceOp& reduce_fun,
      const size_t out_ix) const {
    ET_CHECK_MSG(
        plan_.get_input_tensor().numel() > 0, "Input tensor must be nonempty");

    const size_t init_index =
        get_init_index(plan_.get_input_tensor(), plan_.get_dim_list(), out_ix);

    const CTYPE_IN* const in_data =
        plan_.get_input_tensor().const_data_ptr<CTYPE_IN>();
    CTYPE_OUT acc_val = map_fun(in_data[init_index]);

    if (plan_.get_input_tensor().numel() == 1) {
      return acc_val;
    }

    plan_.execute(
        [&acc_val, reduce_fun, map_fun, in_data](const size_t in_ix) {
          acc_val = reduce_fun(map_fun(in_data[in_ix]), acc_val);
        },
        out_ix);
    return acc_val;
  }

 private:
  ApplyOverDimListPlan plan_;
};

/**
 * Useful to reduce a tensor `in` over a given list of dimensions `dim_list`
 * for the output element at index `out_ix`, first applying the map `map_fun`
 * to each element of `in`, which should have the signature:
 * `CTYPE_OUT map_fun(CTYPE_IN v)`
 * and then reducing using `reduce_fun`, which should have the signature:
 * `CTYPE_OUT reduce_fun(CTYPE_OUT v, CTYPE_OUT acc)`
 *
 * Common usage:
 *
 * CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
 * for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
 *   out_data[out_ix] = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
 *       [](CTYPE_IN v) {
 *         // map operation on `v`, outputs `outv`
 *       },
 *       [](CTYPE_OUT outv, CTYPE_OUT acc) {
 *         // reduce operation on `acc` using `v`, outputs `acc`
 *       in,
 *       dim_list,
 *       out_ix);
 * }
 */
template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename MapOp,
    typename ReduceOp>
CTYPE_OUT map_reduce_over_dim_list(
    const MapOp& map_fun,
    const ReduceOp& reduce_fun,
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    const size_t out_ix) {
  MapReduceOverDimListPlan plan(in, dim_list);
  return plan.execute<CTYPE_IN, CTYPE_OUT>(map_fun, reduce_fun, out_ix);
}

/**
 * Useful to reduce a tensor `in` over a dimension `dim` for the output element
 * at index `out_ix` using the reduce function `reduce_fun`, which should have
 * the following signature:
 * `CTYPE reduce_fun(CTYPE val, long ix, CTYPE acc_val, long acc_ix)`
 *
 * Common usage:
 *
 * CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
 * for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
 *   out_data[out_ix] = reduce_over_dim<CTYPE>(
 *       [](CTYPE val, long ix, CTYPE acc_val, long acc_ix) {
 *         // reduce operation on `acc_val` and `acc_ix` using `val` and `ix`,
 *         // outputs {`acc_val`, `acc_ix`} pair
 *       },
 *       in,
 *       dim_list,
 *       out_ix);
 * }
 */
template <typename CTYPE, typename ReduceOp>
std::tuple<CTYPE, long> reduce_over_dim(
    const ReduceOp& reduce_fun,
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    const size_t out_ix) {
  return map_reduce_over_dim<CTYPE, CTYPE>(
      [](CTYPE v) { return v; }, reduce_fun, in, dim, out_ix);
}

/**
 * Execution plan for repeated reduce_over_dim_list with the same
 * function, input tensor, and dim_list but varying out_ix.
 */
class ReduceOverDimListPlan {
 public:
  ReduceOverDimListPlan(
      const executorch::aten::Tensor& in,
      const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list)
      : plan_(in, dim_list) {}

  template <typename CTYPE, typename ReduceOp>
  CTYPE execute(const ReduceOp& reduce_fun, const size_t out_ix) {
    return plan_.execute<CTYPE, CTYPE>(
        [](CTYPE v) { return v; }, reduce_fun, out_ix);
  }

 private:
  MapReduceOverDimListPlan plan_;
};

/**
 * Useful to reduce a tensor `in` over a given list of dimensions `dim_list`
 * for the output element at index `out_ix` using the reduce function
 * `reduce_fun`, which should have the following signature:
 * `CTYPE reduce_fun(CTYPE v, CTYPE acc)`
 *
 * Common usage:
 *
 * CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
 * for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
 *   out_data[out_ix] = reduce_over_dim_list<CTYPE>(
 *       [](CTYPE v, CTYPE acc) {
 *         // reduce operation on `acc` using `v`, outputs `acc`
 *       },
 *       in,
 *       dim_list,
 *       out_ix);
 * }
 */
template <typename CTYPE, typename ReduceOp>
CTYPE reduce_over_dim_list(
    const ReduceOp& reduce_fun,
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    const size_t out_ix) {
  ReduceOverDimListPlan plan(in, dim_list);
  return plan.execute<CTYPE>(reduce_fun, out_ix);
}

//
// Compute reduced out tensor size and dim
//

size_t compute_reduced_out_size(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    bool keepdim,
    executorch::aten::SizesType* sizes_arr);

size_t compute_reduced_out_size(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    executorch::aten::SizesType* sizes_arr);

inline ssize_t compute_reduced_out_dim(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    bool keepdim) {
  return (
      keepdim                                ? in.dim()
          : dim.has_value() && in.dim() != 0 ? in.dim() - 1
                                             : 0);
}

inline ssize_t compute_reduced_out_dim(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    bool keepdim) {
  return (
      keepdim ? in.dim()
          : dim_list.has_value() && dim_list.value().size() != 0 &&
              in.dim() != 0

          ? in.dim() - dim_list.value().size()
          : 0);
}

//
// Resize out tensor of reduction op
//

Error resize_reduction_out(
    const executorch::aten::Tensor& in,
    const std::optional<int64_t>& dim,
    bool keepdim,
    executorch::aten::Tensor& out);

Error resize_reduction_out(
    const executorch::aten::Tensor& in,
    const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    executorch::aten::Tensor& out);

// Resolve ambiguity between the above two overloads -- ArrayRef and
// optional are both implicitly constructible from int64_t.
inline Error resize_reduction_out(
    const executorch::aten::Tensor& in,
    int64_t dim,
    bool keepdim,
    executorch::aten::Tensor& out) {
  return resize_reduction_out(in, std::optional<int64_t>(dim), keepdim, out);
}

#ifndef USE_ATEN_LIB
bool check_reduction_args(
    const Tensor& in,
    const optional<ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out);

bool check_reduction_args_single_dim(
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out,
    bool allow_empty_dim = false);

bool check_mean_dim_args(
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out);

bool check_amin_amax_args(
    const Tensor& in,
    ArrayRef<int64_t> dim_list,
    bool keepdim,
    Tensor& out);

bool check_argmin_argmax_args(
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    Tensor& out);

bool check_min_max_args(
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_indices);

bool check_prod_out_args(
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out);

#endif

/**
 * parallel_for wrapper for reductions that call reduce_over_dim or
 * map_reduce_over_dim for each output element. Automatically
 * calculates appropriate grain size.
 */
template <typename Func>
[[nodiscard]] bool parallel_for_each_reduce_over_dim_output_index(
    const Tensor& in,
    std::optional<int64_t> dim,
    const Tensor& out,
    const Func& func) {
#ifdef ET_USE_THREADPOOL
  const ssize_t reduction_size = get_reduced_dim_product(in, dim);
  const auto grain_size = std::max(
      static_cast<ssize_t>(1),
      static_cast<ssize_t>(executorch::extension::internal::GRAIN_SIZE) /
          reduction_size);
#else // ET_USE_THREADPOOL
  const auto grain_size = 1;
#endif // ET_USE_THREADPOOL
  return executorch::extension::parallel_for(0, out.numel(), grain_size, func);
}

/**
 * parallel_for wrapper for reductions that call reduce_over_dim_list or
 * map_reduce_over_dim_list for each output element. Automatically
 * calculates appropriate grain size.
 */
template <typename Func>
[[nodiscard]] bool parallel_for_each_reduce_over_dim_list_output_index(
    const Tensor& in,
    std::optional<ArrayRef<int64_t>> dim_list,
    const Tensor& out,
    const Func& func) {
#ifdef ET_USE_THREADPOOL
  const ssize_t reduction_size = get_reduced_dim_product(in, dim_list);
  const auto grain_size = reduction_size == 0
      ? 1
      : std::max(
            static_cast<ssize_t>(1),
            static_cast<ssize_t>(executorch::extension::internal::GRAIN_SIZE) /
                reduction_size);
#else // ET_USE_THREADPOOL
  const auto grain_size = 1;
#endif // ET_USE_THREADPOOL
  return executorch::extension::parallel_for(0, out.numel(), grain_size, func);
}

} // namespace executor
} // namespace torch
