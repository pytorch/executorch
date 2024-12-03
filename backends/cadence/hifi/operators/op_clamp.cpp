/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdbool.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;
using Tensor = exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using executorch::runtime::canCast;
using executorch::runtime::isFloatingType;
using executorch::runtime::isIntegralType;
using executorch::runtime::promoteTypes;
using torch::executor::apply_ternary_elementwise_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::max_override;
using torch::executor::native::utils::min_override;
using torch::executor::native::utils::promote_type_with_scalar;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

namespace {

template <typename CTYPE_VAL, typename CTYPE_OUT, typename CTYPE_CAST>
/** Check if val, when cast to CTYPE_CAST, is not in the range of CTYPE_OUT */
bool is_out_of_bounds(CTYPE_VAL val) {
  const CTYPE_CAST val_cast = static_cast<CTYPE_CAST>(val);
  return val_cast < std::numeric_limits<CTYPE_OUT>::lowest() ||
      val_cast > std::numeric_limits<CTYPE_OUT>::max();
}

__ET_NODISCARD bool check_bounds(
    const Scalar& val_scalar,
    const ScalarType& val_type,
    const ScalarType& out_type,
    const char* val_name) {
  auto is_valid = true;

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, "clamp.out", CTYPE_VAL, [&]() {
    CTYPE_VAL val = 0;
    extract_scalar(val_scalar, &val);
    if (isIntegralType(out_type, /*includeBool=*/false)) {
      ET_SWITCH_INT_TYPES(out_type, ctx, "clamp.out", CTYPE_OUT, [&]() {
        if (is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, long>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    } else if (isFloatingType(out_type)) {
      ET_SWITCH_FLOATH_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
        if (std::isfinite(val) &&
            is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, double>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    }
  });

  return is_valid;
}

} // namespace

Tensor& clamp_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Scalar>& min_opt,
    const exec_aten::optional<Scalar>& max_opt,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = in_type;
  ScalarType max_type = in_type;
  ScalarType common_type = in_type;
  ScalarType out_type = out.scalar_type();

  bool has_min = min_opt.has_value();
  if (has_min) {
    min_type = get_scalar_dtype(min_opt.value());
    common_type = promote_type_with_scalar(common_type, min_opt.value());
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(min_opt.value(), min_type, out_type, "minimum"),
        InvalidArgument,
        out);
  }
  bool has_max = max_opt.has_value();
  if (has_max) {
    max_type = get_scalar_dtype(max_opt.value());
    common_type = promote_type_with_scalar(common_type, max_opt.value());
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(max_opt.value(), max_type, out_type, "maximum"),
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  ET_SWITCH_REALH_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
    // Extract optional min value
    CTYPE_OUT min = 0;
    if (has_min) {
      ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "clamp", CTYPE_MIN, [&]() {
        CTYPE_MIN min_val = 0;
        extract_scalar(min_opt.value(), &min_val);
        min = static_cast<CTYPE_OUT>(min_val);
      });
    }

    // Extract optional max value
    CTYPE_OUT max = 0;
    if (has_max) {
      ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "clamp", CTYPE_MAX, [&]() {
        CTYPE_MAX max_val = 0;
        extract_scalar(max_opt.value(), &max_val);
        max = static_cast<CTYPE_OUT>(max_val);
      });
    }

    ET_SWITCH_REALHB_TYPES(in_type, ctx, "clamp", CTYPE_IN, [&]() {
      torch::executor::apply_unary_map_fn(
          [has_min, min, has_max, max](const CTYPE_IN val_in) {
            CTYPE_OUT val_out = static_cast<CTYPE_OUT>(val_in);
            if (has_min) {
              val_out = max_override(val_out, min);
            }
            if (has_max) {
              val_out = min_override(val_out, max);
            }
            return val_out;
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });

  return out;
}

Tensor& clamp_tensor_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Tensor>& min_opt,
    const exec_aten::optional<Tensor>& max_opt,
    Tensor& out) {
  (void)ctx;

  bool has_min = min_opt.has_value();
  bool has_max = max_opt.has_value();

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  const Tensor& min = has_min ? min_opt.value() : in;
  const Tensor& max = has_max ? max_opt.value() : in;

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(in, min, max, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr int kNnlibMaxDim =
      4; /*fallback to not optimised if broadcast and dim > 4 */

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = min.scalar_type();
  ScalarType max_type = max.scalar_type();
  ScalarType common_type = in_type;
  ScalarType out_type = out.scalar_type();

  if (has_min) {
    common_type = promoteTypes(common_type, min_type, /*half_to_float*/ true);
  }
  if (has_max) {
    common_type = promoteTypes(common_type, max_type, /*half_to_float*/ true);
  }

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  bool in_is_broadcasted = !out.sizes().equals(in.sizes());
  bool min_is_broadcasted = !out.sizes().equals(min.sizes());
  bool max_is_broadcasted = !out.sizes().equals(max.sizes());
  bool broadcast =
      (in_is_broadcasted || min_is_broadcasted || max_is_broadcasted);

  int max_dim = in.dim() > min.dim() ? in.dim() : min.dim();
  max_dim = max.dim() > max_dim ? max.dim() : max_dim;
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  bool optimized = true;
  bool fall_back = false;
  if ((in_type != ScalarType::Float) || (min_type != ScalarType::Float) ||
      (max_type != ScalarType::Float))
    optimized = false;
  if ((broadcast == true) && (max_dim > kNnlibMaxDim))
    optimized = false;

  if (optimized) {
    if (!has_min) {
      const float* const max_data = max.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int max_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          max_shape[i] = 1;
        }

        int max_dim = max.dim(), inp_dim = in.dim(), out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_max = kNnlibMaxDim - max_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++) {
          out_shape[i + off_o] = out.size(i);
        }
        for (int i = 0; i < max_dim; i++) {
          max_shape[i + off_max] = max.size(i);
        }
        for (int i = 0; i < inp_dim; i++) {
          inp_shape[i + off_inp] = in.size(i);
        }

        WORD32 ret_val = xa_nn_elm_minimum_broadcast_4D_f32xf32_f32(
            out_data, out_shape, inp_data, inp_shape, max_data, max_shape);

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      } else {
        WORD32 ret_val = xa_nn_elm_minimum_f32xf32_f32(
            out_data, inp_data, max_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    } else if (!has_max) {
      const float* const min_data = min.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast == 1) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int min_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          min_shape[i] = 1;
        }

        int min_dim = min.dim(), max_dim = max.dim(), inp_dim = in.dim(),
            out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_min = kNnlibMaxDim - min_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++)
          out_shape[i + off_o] = out.size(i);
        for (int i = 0; i < min_dim; i++)
          min_shape[i + off_min] = min.size(i);
        for (int i = 0; i < inp_dim; i++)
          inp_shape[i + off_inp] = in.size(i);
        WORD32 ret_val = xa_nn_elm_maximum_broadcast_4D_f32xf32_f32(
            out_data, out_shape, inp_data, inp_shape, min_data, min_shape);

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      } else {
        WORD32 ret_val = xa_nn_elm_maximum_f32xf32_f32(
            out_data, inp_data, min_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    } else {
      const float* const min_data = min.const_data_ptr<float>();
      const float* const max_data = max.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast == 1) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int min_shape[kNnlibMaxDim];
        int max_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          min_shape[i] = 1;
          max_shape[i] = 1;
        }

        int min_dim = min.dim(), max_dim = max.dim(), inp_dim = in.dim(),
            out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_min = kNnlibMaxDim - min_dim;
        int off_max = kNnlibMaxDim - max_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++)
          out_shape[i + off_o] = out.size(i);
        for (int i = 0; i < min_dim; i++)
          min_shape[i + off_min] = min.size(i);

        for (int i = 0; i < max_dim; i++)
          max_shape[i + off_max] = max.size(i);

        for (int i = 0; i < inp_dim; i++)
          inp_shape[i + off_inp] = in.size(i);

        if (inp_shape[0] != out_shape[0] || inp_shape[1] != out_shape[1] ||
            inp_shape[2] != out_shape[2] || inp_shape[3] != out_shape[3]) {
          void* p_scratch = (void*)kernels::allocate_temp_memory(
              ctx,
              (out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]) *
                  sizeof(int));
          const FLOAT32* p_brd_cond = (const FLOAT32*)p_scratch;
          xa_nn_broadcast_32_32(
              (WORD32*)p_brd_cond, out_shape, (WORD32*)inp_data, inp_shape, 4);

          for (int i = 0; i < 4; i++) {
            inp_shape[i] = out_shape[i];
          }

          WORD32 ret_val = xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
              out_data,
              out_shape,
              p_brd_cond,
              inp_shape,
              min_data,
              min_shape,
              max_data,
              max_shape);

          ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

        } else {
          WORD32 ret_val = xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
              out_data,
              out_shape,
              inp_data,
              inp_shape,
              min_data,
              min_shape,
              max_data,
              max_shape);

          ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
        }
      } else {
        WORD32 ret_val = xa_nn_elm_clamp_f32xf32xf32_f32(
            out_data, inp_data, min_data, max_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    }
    return out;
  }

  constexpr auto name = "clamp.Tensor_out";

  ET_SWITCH_REALHB_TYPES(in_type, ctx, name, CTYPE_IN, [&]() {
    ET_SWITCH_REALHB_TYPES(min_type, ctx, name, CTYPE_MIN, [&]() {
      ET_SWITCH_REALHB_TYPES(max_type, ctx, name, CTYPE_MAX, [&]() {
        ET_SWITCH_REALHB_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
          apply_ternary_elementwise_fn<
              CTYPE_IN,
              CTYPE_MIN,
              CTYPE_MAX,
              CTYPE_OUT>(
              [has_min, has_max](
                  const CTYPE_IN val_in,
                  const CTYPE_MIN val_min,
                  const CTYPE_MAX val_max) {
                CTYPE_OUT val_out = static_cast<CTYPE_OUT>(val_in);
                if (has_min) {
                  val_out =
                      max_override(val_out, static_cast<CTYPE_OUT>(val_min));
                }
                if (has_max) {
                  val_out =
                      min_override(val_out, static_cast<CTYPE_OUT>(val_max));
                }
                return val_out;
              },
              in,
              min,
              max,
              out);
        });
      });
    });
  });
  return out;
}
} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
