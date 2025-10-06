/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/fusion_g3/operators/xt_macros.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::canCast;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

namespace impl {
namespace G3 {
namespace native {

namespace {

template <typename CTYPE_VAL, typename CTYPE_OUT, typename CTYPE_CAST>
/** Check if val, when cast to CTYPE_CAST, is not in the range of CTYPE_OUT */
bool is_out_of_bounds(CTYPE_VAL val) {
  const CTYPE_CAST val_cast = static_cast<CTYPE_CAST>(val);
  return val_cast < std::numeric_limits<CTYPE_OUT>::lowest() ||
      val_cast > std::numeric_limits<CTYPE_OUT>::max();
}

ET_NODISCARD bool check_bounds(
    KernelRuntimeContext& ctx,
    const Scalar& val_scalar,
    const ScalarType& val_type,
    const ScalarType& out_type,
    const char* val_name) {
  auto is_valid = true;

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, "clamp.out", CTYPE_VAL, [&]() {
    CTYPE_VAL val = 0;
    torch::executor::native::utils::extract_scalar(val_scalar, &val);
    if (executorch::runtime::isIntegralType(out_type, /*includeBool=*/false)) {
      ET_SWITCH_INT_TYPES(out_type, ctx, "clamp.out", CTYPE_OUT, [&]() {
        if (is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, long>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    } else if (executorch::runtime::isFloatingType(out_type)) {
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
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const optional<Scalar>& min_opt,
    const optional<Scalar>& max_opt,
    Tensor& out) {
  bool has_min = min_opt.has_value();
  bool has_max = max_opt.has_value();

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  // Input Dtypes
  ScalarType in_type = in.scalar_type();
  ScalarType min_type = has_min
      ? torch::executor::native::utils::get_scalar_dtype(min_opt.value())
      : in_type;
  ScalarType max_type = has_max
      ? torch::executor::native::utils::get_scalar_dtype(max_opt.value())
      : in_type;
  ScalarType out_type = out.scalar_type();

  // Check Scalar Bounds
  if (has_min) {
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(ctx, min_opt.value(), min_type, out_type, "minimum"),
        InvalidArgument,
        out);
  }
  if (has_max) {
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(ctx, max_opt.value(), max_type, out_type, "maximum"),
        InvalidArgument,
        out);
  }

#ifdef OP_ARG_CHECK
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out);
#endif

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "clamp.out";

  bool optimized = true;

  if (!(((in_type == ScalarType::Float) || (in_type == ScalarType::Short) ||
         (in_type == ScalarType::Char) || (in_type == ScalarType::Byte)) &&
        (in_type == out_type))) {
    optimized = false;
  }

  if (has_max) {
    if ((max_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Short)) {
      optimized = false;
    }

    if ((max_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Char)) {
      optimized = false;
    }

    if ((max_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Byte)) {
      optimized = false;
    }
  }

  if (has_min) {
    if ((min_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Short)) {
      optimized = false;
    }

    if ((min_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Char)) {
      optimized = false;
    }

    if ((min_opt.value().isFloatingPoint()) &&
        (in.scalar_type() == ScalarType::Byte)) {
      optimized = false;
    }
  }

  if ((in_type == ScalarType::Float) && (optimized)) {
    const float* const inp1_data = in.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();
    float min_val, max_val;

    if (!has_min) {
      min_val = std::numeric_limits<float>::lowest();
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    } else if (!has_max) {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      max_val = std::numeric_limits<float>::max();
    } else {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    }

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_clamp_scalar_f32_f32,
        out_data,
        inp1_data,
        min_val,
        max_val,
        out.numel());
  } else if ((in_type == ScalarType::Short) && (optimized)) {
    const signed short* const inp1_data = in.const_data_ptr<signed short>();
    signed short* const out_data = out.mutable_data_ptr<signed short>();
    signed short min_val, max_val;

    if (!has_min) {
      min_val = std::numeric_limits<int16_t>::lowest();
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    } else if (!has_max) {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      max_val = std::numeric_limits<int16_t>::max();
    } else {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    }

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_clamp_scalar_16_16,
        out_data,
        inp1_data,
        min_val,
        max_val,
        out.numel());
  } else if ((in_type == ScalarType::Char) && (optimized)) {
    const signed char* const inp1_data = in.const_data_ptr<signed char>();
    signed char* const out_data = out.mutable_data_ptr<signed char>();
    signed char min_val, max_val;

    if (!has_min) {
      min_val = std::numeric_limits<int8_t>::lowest();
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    } else if (!has_max) {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      max_val = std::numeric_limits<int8_t>::max();
    } else {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    }

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_clamp_scalar_8_8,
        out_data,
        inp1_data,
        min_val,
        max_val,
        out.numel());
  } else if ((in_type == ScalarType::Byte) && (optimized)) {
    const unsigned char* const inp1_data = in.const_data_ptr<unsigned char>();
    unsigned char* const out_data = out.mutable_data_ptr<unsigned char>();
    unsigned char min_val, max_val;

    if (!has_min) {
      min_val = std::numeric_limits<uint8_t>::lowest();
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    } else if (!has_max) {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      max_val = std::numeric_limits<uint8_t>::max();
    } else {
      torch::executor::native::utils::extract_scalar(min_opt.value(), &min_val);
      torch::executor::native::utils::extract_scalar(max_opt.value(), &max_val);
    }

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_clamp_scalar_8u_8u,
        out_data,
        inp1_data,
        min_val,
        max_val,
        out.numel());
  } else {
    // Common Dtype
    ScalarType common_type = in_type;
    if (has_min) {
      common_type = torch::executor::native::utils::promote_type_with_scalar(
          common_type, min_opt.value());
    }
    if (has_max) {
      common_type = torch::executor::native::utils::promote_type_with_scalar(
          common_type, max_opt.value());
    }

    // Check Common Dtype
    ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

    // Compute Dtype
    ScalarType compute_type =
        torch::executor::native::utils::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      torch::executor::native::utils::
          apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
              [has_min, min_opt, has_max, max_opt](const CTYPE_COMPUTE val_in) {
                CTYPE_COMPUTE val_out = val_in;
                if (has_min) {
                  val_out = torch::executor::native::utils::max_override(
                      val_out,
                      torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(
                          min_opt.value()));
                }
                if (has_max) {
                  val_out = torch::executor::native::utils::min_override(
                      val_out,
                      torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(
                          max_opt.value()));
                }
                return val_out;
              },
              ctx,
              in,
              torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
              out,
              torch::executor::native::utils::SupportedTensorDtypes::
                  SAME_AS_COMMON);
    });
  }

  return out;
}

Tensor& clamp_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const optional<Tensor>& min_opt,
    const optional<Tensor>& max_opt,
    Tensor& out) {
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

#ifdef OP_ARG_CHECK
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, min, max, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(in, min, max, out) ==
          Error::Ok,
      InvalidArgument,
      out);
#endif

  static constexpr const char op_name[] = "clamp.Tensor_out";

  int kTensorDimensionLimit = 5;

  int inp_shape[kTensorDimensionLimit];
  int min_shape[kTensorDimensionLimit];
  int max_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  bool broadcast = false;

  int max1_dim = min.dim() > max.dim() ? min.dim() : max.dim();
  int max2_dim = in.dim() > out.dim() ? in.dim() : out.dim();
  int max_dim = max1_dim > max2_dim ? max1_dim : max2_dim;

  bool optimized = true;

  for (int i = 0; i < max_dim; i++) {
    out_shape[i] = 1;
    inp_shape[i] = 1;
    min_shape[i] = 1;
    max_shape[i] = 1;
  }

  int offset_out = max_dim - out.dim();
  int offset_inp = max_dim - in.dim();
  int offset_min = max_dim - min.dim();
  int offset_max = max_dim - max.dim();

  for (int i = 0; i < out.dim(); i++) {
    out_shape[i + offset_out] = out.size(i);
  }
  for (int i = 0; i < in.dim(); i++) {
    inp_shape[i + offset_inp] = in.size(i);
  }
  if (has_min) {
    for (int i = 0; i < min.dim(); i++) {
      min_shape[i + offset_min] = min.size(i);
    }
  }
  if (has_max) {
    for (int i = 0; i < max.dim(); i++) {
      max_shape[i + offset_max] = max.size(i);
    }
  }

  /*find broadcast*/
  for (int i = 0; i < max_dim; i++) {
    if (((inp_shape[i]) != (out_shape[i])) ||
        ((min_shape[i]) != (out_shape[i])) ||
        ((max_shape[i]) != (out_shape[i]))) {
      broadcast = true;
    }
  }

  if (((broadcast) && (max_dim > kTensorDimensionLimit)) ||
      (!(((in.scalar_type() == ScalarType::Float) ||
          (in.scalar_type() == ScalarType::Short) ||
          (in.scalar_type() == ScalarType::Char) ||
          (in.scalar_type() == ScalarType::Byte)) &&
         (in.scalar_type() == min.scalar_type()) &&
         (in.scalar_type() == max.scalar_type()) &&
         (in.scalar_type() == out.scalar_type())))) {
    optimized = false;
  }

  if ((in.scalar_type() == ScalarType::Float) && (optimized)) {
    const float* const inp1_data = in.const_data_ptr<float>();
    const float* min_data = min.const_data_ptr<float>();
    const float* max_data = max.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();
    float lowest_val, highest_val;

    if (broadcast || !has_min || !has_max) {
      if (!has_min) {
        lowest_val = std::numeric_limits<float>::lowest();
        min_data = &lowest_val;
      }

      if (!has_max) {
        highest_val = std::numeric_limits<float>::max();
        max_data = &highest_val;
      }

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_broadcast_5D_f32_f32,
          out_data,
          out_shape,
          inp1_data,
          inp_shape,
          min_data,
          min_shape,
          max_data,
          max_shape,
          max_dim);

    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_f32_f32,
          out_data,
          inp1_data,
          min_data,
          max_data,
          out.numel());
    }
  } else if ((in.scalar_type() == ScalarType::Short) && (optimized)) {
    const signed short* const inp1_data = in.const_data_ptr<signed short>();
    const signed short* min_data = min.const_data_ptr<signed short>();
    const signed short* max_data = max.const_data_ptr<signed short>();
    signed short* const out_data = out.mutable_data_ptr<signed short>();
    signed short lowest_val, highest_val;

    if (broadcast || !has_min || !has_max) {
      if (!has_min) {
        lowest_val = std::numeric_limits<signed short>::lowest();
        min_data = &lowest_val;
      }

      if (!has_max) {
        highest_val = std::numeric_limits<signed short>::max();
        max_data = &highest_val;
      }

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_broadcast_5D_16_16,
          out_data,
          out_shape,
          inp1_data,
          inp_shape,
          min_data,
          min_shape,
          max_data,
          max_shape,
          max_dim);

    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_16_16,
          out_data,
          inp1_data,
          min_data,
          max_data,
          out.numel());
    }
  } else if ((in.scalar_type() == ScalarType::Char) && (optimized)) {
    const signed char* const inp1_data = in.const_data_ptr<signed char>();
    const signed char* min_data = min.const_data_ptr<signed char>();
    const signed char* max_data = max.const_data_ptr<signed char>();
    signed char* const out_data = out.mutable_data_ptr<signed char>();
    signed char lowest_val, highest_val;

    if (broadcast || !has_min || !has_max) {
      if (!has_min) {
        lowest_val = std::numeric_limits<signed char>::lowest();
        min_data = &lowest_val;
      }

      if (!has_max) {
        highest_val = std::numeric_limits<signed char>::max();
        max_data = &highest_val;
      }

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_broadcast_5D_8_8,
          out_data,
          out_shape,
          inp1_data,
          inp_shape,
          min_data,
          min_shape,
          max_data,
          max_shape,
          max_dim);

    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_8_8,
          out_data,
          inp1_data,
          min_data,
          max_data,
          out.numel());
    }
  } else if ((in.scalar_type() == ScalarType::Byte) && (optimized)) {
    const unsigned char* const inp1_data = in.const_data_ptr<unsigned char>();
    const unsigned char* min_data = min.const_data_ptr<unsigned char>();
    const unsigned char* max_data = max.const_data_ptr<unsigned char>();
    unsigned char* const out_data = out.mutable_data_ptr<unsigned char>();
    unsigned char lowest_val, highest_val;

    if (broadcast || !has_min || !has_max) {
      if (!has_min) {
        lowest_val = std::numeric_limits<unsigned char>::lowest();
        min_data = &lowest_val;
      }

      if (!has_max) {
        highest_val = std::numeric_limits<unsigned char>::max();
        max_data = &highest_val;
      }

      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_broadcast_5D_8u_8u,
          out_data,
          out_shape,
          inp1_data,
          inp_shape,
          min_data,
          min_shape,
          max_data,
          max_shape,
          max_dim);

    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_clamp_8u_8u,
          out_data,
          inp1_data,
          min_data,
          max_data,
          out.numel());
    }
  } else {
    // Common Dtype
    ScalarType common_type = in.scalar_type();

    // Check Common Dtype
    ET_KERNEL_CHECK(
        ctx, canCast(common_type, out.scalar_type()), InvalidArgument, out);

    if (has_min) {
      common_type =
          executorch::runtime::promoteTypes(common_type, min.scalar_type());
    }
    if (has_max) {
      common_type =
          executorch::runtime::promoteTypes(common_type, max.scalar_type());
    }

    // Compute Dtype
    ScalarType compute_type =
        torch::executor::native::utils::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      torch::executor::native::utils::apply_tritensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name>(
          [has_min, has_max](
              const CTYPE_COMPUTE val_in,
              const CTYPE_COMPUTE val_min,
              const CTYPE_COMPUTE val_max) {
            CTYPE_COMPUTE val_out = val_in;
            if (has_min) {
              val_out = torch::executor::native::utils::max_override(
                  val_out, val_min);
            }
            if (has_max) {
              val_out = torch::executor::native::utils::min_override(
                  val_out, val_max);
            }
            return val_out;
          },
          ctx,
          in,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          min,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          max,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          out,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16);
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
