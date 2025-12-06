/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

namespace torch {
namespace executor {
namespace native {

Tensor& opt_where_out(
    KernelRuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out.scalar_type(), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(cond, a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "where.self_out";

  if (a.scalar_type() == b.scalar_type() &&
      a.scalar_type() == out.scalar_type() && a.scalar_type() == compute_type &&
      // Using a Byte tensor for cond has been deprecated for a long time.
      cond.scalar_type() == ScalarType::Bool) {
    auto out_numel = out.numel();
    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      const CTYPE_COMPUTE* const data_a = a.const_data_ptr<CTYPE_COMPUTE>();
      const CTYPE_COMPUTE* const data_b = b.const_data_ptr<CTYPE_COMPUTE>();
      const bool* const data_cond = cond.const_data_ptr<bool>();
      CTYPE_COMPUTE* const data_out = out.mutable_data_ptr<CTYPE_COMPUTE>();
      executorch::extension::parallel_for(
          0,
          out_numel,
          ::executorch::extension::internal::GRAIN_SIZE,
          [&](const auto begin, const auto end) {
            auto range = BroadcastIndexesRange<3>(out, a, b, cond);
            auto begin_it = range.begin();
            begin_it += begin;
            for (; (*begin_it)[0] < end; ++begin_it) {
              const auto [out_index, a_index, b_index, cond_index] = *begin_it;
              data_out[out_index] =
                  data_cond[cond_index] ? data_a[a_index] : data_b[b_index];
            }
          });
    });
  } else {
    // Fall back for mixed dtype to keep code size and compile time
    // reasonable.
    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      utils::apply_tritensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
          [](const CTYPE_COMPUTE val_a,
             const CTYPE_COMPUTE val_b,
             const CTYPE_COMPUTE val_c) { return val_c ? val_a : val_b; },
          ctx,
          a,
          utils::SupportedTensorDtypes::REALHBBF16,
          b,
          utils::SupportedTensorDtypes::REALHBBF16,
          cond,
          utils::SupportedTensorDtypes::BOOL_OR_BYTE,
          out,
          utils::SupportedTensorDtypes::SAME_AS_COMMON);
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
