// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& gt_tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "gt", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "gt", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, common_type, ctx, "gt", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "gt", CTYPE_OUT, [&]() {
          apply_binary_elementwise_fn(
              [](const CTYPE_A val_a, const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                bool value = a_casted > b_casted;
                return static_cast<CTYPE_OUT>(value);
              },
              a,
              a.const_data_ptr<CTYPE_A>(),
              b,
              b.const_data_ptr<CTYPE_B>(),
              out,
              out.mutable_data_ptr<CTYPE_OUT>());
        });
      });
    });
  });

  return out;
}

Tensor& gt_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "gt", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "gt", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, common_type, ctx, "gt", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "gt", CTYPE_OUT, [&]() {
          CTYPE_B val_b = 0;
          ET_EXTRACT_SCALAR(b, val_b);
          apply_unary_map_fn(
              [val_b](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                bool value = a_casted > b_casted;
                return static_cast<CTYPE_OUT>(value);
              },
              a.const_data_ptr<CTYPE_A>(),
              out.mutable_data_ptr<CTYPE_OUT>(),
              out.numel());
        });
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
