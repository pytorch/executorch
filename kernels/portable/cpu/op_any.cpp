/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <optional>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& any_all_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "any.all_out";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_TWO_TYPES(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, [&] {
      const auto data_in = in.const_data_ptr<CTYPE_IN>();
      auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
      data_out[0] = static_cast<CTYPE_OUT>(false);
      for (const auto i : c10::irange(in.numel())) {
        if (static_cast<bool>(data_in[i])) {
          data_out[0] = static_cast<CTYPE_OUT>(true);
          break;
        }
      }
    });
  });

  return out;
}

Tensor& any_dims_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    std::optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out),
      InvalidArgument,
      out);

  if (dim_list.has_value() && dim_list.value().empty()) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
  } else {
    ET_KERNEL_CHECK(
        ctx,
        resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "any.dims_out";

  const bool in_not_empty = in.numel() > 0;
  std::optional<MapReduceOverDimListPlan> plan;
  if ((!dim_list.has_value() || !dim_list.value().empty()) && in_not_empty) {
    plan.emplace(in, dim_list);
  }
  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_TWO_TYPES(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      if (dim_list.has_value() && dim_list.value().empty()) {
        const CTYPE_IN* in_data = in.const_data_ptr<CTYPE_IN>();
        for (const auto out_ix : c10::irange(out.numel())) {
          out_data[out_ix] =
              static_cast<CTYPE_OUT>(static_cast<bool>(in_data[out_ix]));
        }
      } else {
        const bool success =
            parallel_for_each_reduce_over_dim_list_output_index(
                in, dim_list, out, [&](const auto begin, const auto end) {
                  for (const auto out_ix : c10::irange(begin, end)) {
                    bool any = false;
                    if (plan.has_value()) {
                      any = plan->execute<CTYPE_IN, bool>(
                          [](CTYPE_IN v) { return static_cast<bool>(v); },
                          [](bool outv, bool acc) { return acc || outv; },
                          out_ix);
                    }
                    out_data[out_ix] = static_cast<CTYPE_OUT>(any);
                  }
                });
        ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
      }
    });
  });

  return out;
}

Tensor& any_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args_single_dim(
          in, dim, keepdim, {}, out, /*allow_empty_dim*/ true),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "any.out";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_TWO_TYPES(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      const bool success = parallel_for_each_reduce_over_dim_output_index(
          in, dim, out, [&](const auto begin, const auto end) {
            for (const auto out_ix : c10::irange(begin, end)) {
              CTYPE_OUT any = false;
              if (in.numel() > 0) {
                std::tuple<CTYPE_OUT, long> acc =
                    map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>(
                        [](CTYPE_IN v) { return static_cast<bool>(v); },
                        [](bool outv, long, bool acc, long) {
                          return std::tuple<bool, long>{acc || outv, 0};
                        },
                        in,
                        dim,
                        out_ix);
                any = std::get<0>(acc);
              }
              out_data[out_ix] = any;
            }
          });
      ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
