/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/select_copy_util.h>
#include <executorch/kernels/portable/cpu/util/sort_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using namespace exec_aten;

Tensor& nms_out(
    RuntimeContext& ctx,
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      dets.dim() == 2,
      InvalidArgument,
      out,
      "boxes should be a 2d tensor, got %zd",
      dets.dim());
  ET_KERNEL_CHECK_MSG(
      ctx,
      dets.size(1) == 4,
      InvalidArgument,
      out,
      "boxes should have 4 elements in dimension 1, got %zd",
      dets.size(1));
  ET_KERNEL_CHECK_MSG(
      ctx,
      scores.dim() == 1,
      InvalidArgument,
      out,
      "scores should be a 1d tensor, got %zd",
      scores.dim());
  ET_KERNEL_CHECK_MSG(
      ctx,
      dets.size(0) == scores.size(0),
      InvalidArgument,
      out,
      "boxes and scores should have same number of elements in dimension 0, got %zd and %zd",
      dets.size(0),
      scores.size(0));
  ET_KERNEL_CHECK_MSG(
      ctx,
      dets.scalar_type() == scores.scalar_type(),
      InvalidArgument,
      out,
      "boxes and scores should have same type, got %s and %s",
      toString(dets.scalar_type()),
      toString(scores.scalar_type()));
  ET_KERNEL_CHECK_MSG(
      ctx,
      dets.scalar_type() == ScalarType::Float,
      InvalidArgument,
      out,
      "dets should have type float, got %s",
      toString(dets.scalar_type()));

  if (dets.numel() == 0) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        resize_tensor(out, {0}) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor.");
  }

  ArrayRef<exec_aten::SizesType> sizes = {dets.sizes()[0]};
  ArrayRef<exec_aten::StridesType> strides = {dets.strides()[0]};
  ArrayRef<exec_aten::DimOrderType> dim_orders = {dets.dim_order()[0]};

  Tensor x1_t = make_tensor(sizes, dim_orders, strides, dets.scalar_type());
  Tensor y1_t = make_tensor(sizes, dim_orders, strides, dets.scalar_type());
  Tensor x2_t = make_tensor(sizes, dim_orders, strides, dets.scalar_type());
  Tensor y2_t = make_tensor(sizes, dim_orders, strides, dets.scalar_type());

  select_copy_util(dets, 1, 0, x1_t);
  select_copy_util(dets, 1, 1, y1_t);
  select_copy_util(dets, 1, 2, x2_t);
  select_copy_util(dets, 1, 3, y2_t);

  Tensor x_diff = make_tensor(
      x1_t.sizes(), x1_t.dim_order(), x1_t.strides(), x1_t.scalar_type());
  Tensor y_diff = make_tensor(
      y1_t.sizes(), y1_t.dim_order(), y1_t.strides(), y1_t.scalar_type());
  Tensor areas_t = make_tensor(
      y1_t.sizes(), y1_t.dim_order(), y1_t.strides(), y1_t.scalar_type());

  apply_binary_elementwise_fn<float, float, float>(
      [](const float val_a, const float val_b) { return val_a - val_b; },
      x2_t,
      x1_t,
      x_diff);

  apply_binary_elementwise_fn<float, float, float>(
      [](const float val_a, const float val_b) { return val_a - val_b; },
      y2_t,
      y1_t,
      y_diff);

  apply_binary_elementwise_fn<float, float, float>(
      [](const float val_a, const float val_b) { return val_a * val_b; },
      x_diff,
      y_diff,
      areas_t);

  free_tensor(x_diff);
  free_tensor(y_diff);

  Tensor sorted_tensor = make_tensor(
      scores.sizes(),
      scores.dim_order(),
      scores.strides(),
      scores.scalar_type());
  Tensor sorted_indices = make_tensor(
      scores.sizes(), scores.dim_order(), scores.strides(), ScalarType::Long);
  Error error = sort_tensor(scores, sorted_tensor, sorted_indices, true);
  ET_KERNEL_CHECK_MSG(
      ctx,
      error == Error::Ok,
      InvalidArgument,
      out,
      "Failed to sort scores tensor.");

  auto ndets = dets.size(0);
  Tensor suppressed_t = make_tensor(
      {ndets}, {dets.dim_order()[0]}, {dets.strides()[0]}, ScalarType::Byte);
  std::memset(
      suppressed_t.mutable_data_ptr<uint8_t>(), 0, suppressed_t.nbytes());
  std::memset(out.mutable_data_ptr<int64_t>(), 0, out.nbytes());

  auto suppressed = suppressed_t.mutable_data_ptr<uint8_t>();
  auto keep = out.mutable_data_ptr<int64_t>();
  auto order = sorted_indices.const_data_ptr<int64_t>();
  auto x1 = x1_t.const_data_ptr<float>();
  auto y1 = y1_t.const_data_ptr<float>();
  auto x2 = x2_t.const_data_ptr<float>();
  auto y2 = y2_t.const_data_ptr<float>();
  auto areas = areas_t.const_data_ptr<float>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<float>(0), xx2 - xx1);
      auto h = std::max(static_cast<float>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  free_tensor(x1_t);
  free_tensor(y1_t);
  free_tensor(x2_t);
  free_tensor(y2_t);
  free_tensor(areas_t);
  free_tensor(sorted_tensor);
  free_tensor(sorted_indices);
  free_tensor(suppressed_t);

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, {num_to_keep}) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
