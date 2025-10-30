/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/cpu/binary_ops.h>

namespace torch::executor::internal {
std::optional<BroadcastElementwisePlan> plan_broadcast_elementwise(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    const ElementwiseOptimizedPath selected_optimized_path) {
  BroadcastElementwisePlan plan;
  if ((selected_optimized_path ==
       ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments) ||
      (selected_optimized_path ==
       ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments)) {
    plan.lhs = &b;
    plan.rhs = &a;
  } else {
    // Catch failure to update logic when adding new broadcasting possibility.
    ET_DCHECK(
        (selected_optimized_path ==
         ElementwiseOptimizedPath::kBroadcast2dBy1d) ||
        (selected_optimized_path ==
         ElementwiseOptimizedPath::kBroadcastNdByNd));
    plan.lhs = &a;
    plan.rhs = &b;
  }
  auto error = resize_tensor(out, plan.lhs->sizes());
  ET_KERNEL_CHECK_MSG(
      ctx,
      error == Error::Ok,
      InvalidArgument,
      std::nullopt,
      "Failed to resize output tensor.");
  plan.outer_size = 1;
  if ((selected_optimized_path == ElementwiseOptimizedPath::kBroadcastNdByNd) ||
      (selected_optimized_path ==
       ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments)) {
    int32_t broadcast_dim = internal::get_broadcast_dim(*plan.lhs, *plan.rhs);
    int32_t broadcast_dim_lhs = plan.lhs->dim() + broadcast_dim;
    auto normalized_tensor_size_lhs =
        get_normalized_tensor_size(*plan.lhs, broadcast_dim_lhs);
    plan.outer_size = normalized_tensor_size_lhs[0];
    plan.broadcast_size = normalized_tensor_size_lhs[1];
    plan.inner_size = normalized_tensor_size_lhs[2];
  } else {
    plan.broadcast_size = plan.lhs->sizes()[plan.lhs->dim() - 2];
    plan.inner_size = plan.lhs->sizes()[plan.lhs->dim() - 1];
  }
  return plan;
}
} // namespace torch::executor::internal
