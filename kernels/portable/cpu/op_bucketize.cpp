/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;
using Tensor = executorch::aten::Tensor;

namespace {

inline bool check_bucketize_args(
    const Tensor& /* self */,
    const Tensor& boundaries,
    bool out_int32,
    const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(boundaries.dim() == 1);
  if (out_int32) {
    ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == ScalarType::Int);
  } else {
    ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == ScalarType::Long);
  }
  return true;
}

template <typename T>
inline bool is_nan_val(T val) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(val);
  } else if constexpr (
      std::is_same_v<T, torch::executor::Half> ||
      std::is_same_v<T, torch::executor::BFloat16>) {
    return std::isnan(static_cast<float>(val));
  } else {
    return false;
  }
}

template <typename T>
inline double to_double_val(T val) {
  if constexpr (
      std::is_same_v<T, torch::executor::Half> ||
      std::is_same_v<T, torch::executor::BFloat16>) {
    return static_cast<double>(static_cast<float>(val));
  } else {
    return static_cast<double>(val);
  }
}

template <typename IN_T, typename BOUND_T, typename OUT_T>
void bucketize_kernel(
    const Tensor& self,
    const Tensor& boundaries,
    bool right,
    Tensor& out) {
  const IN_T* const self_data = self.const_data_ptr<IN_T>();
  const BOUND_T* const boundaries_data = boundaries.const_data_ptr<BOUND_T>();
  OUT_T* const out_data = out.mutable_data_ptr<OUT_T>();

  const int64_t num_boundaries = boundaries.numel();
  const size_t num_elements = self.numel();

  if (num_boundaries == 0) {
    for (size_t i = 0; i < num_elements; ++i) {
      out_data[i] = 0;
    }
    return;
  }

  for (size_t i = 0; i < num_elements; ++i) {
    const IN_T val = self_data[i];
    if (is_nan_val(val)) {
      out_data[i] = static_cast<OUT_T>(num_boundaries);
      continue;
    }

    if constexpr (std::is_same_v<IN_T, BOUND_T>) {
      if (right) {
        auto it = std::lower_bound(
            boundaries_data,
            boundaries_data + num_boundaries,
            val,
            [](const BOUND_T& bound, const IN_T& v) { return bound < v; });
        out_data[i] = static_cast<OUT_T>(it - boundaries_data);
      } else {
        auto it = std::upper_bound(
            boundaries_data,
            boundaries_data + num_boundaries,
            val,
            [](const IN_T& v, const BOUND_T& bound) { return v < bound; });
        out_data[i] = static_cast<OUT_T>(it - boundaries_data);
      }
    } else {
      const double dval = to_double_val(val);
      if (right) {
        auto it = std::lower_bound(
            boundaries_data,
            boundaries_data + num_boundaries,
            dval,
            [](const BOUND_T& bound, double v) {
              return to_double_val(bound) < v;
            });
        out_data[i] = static_cast<OUT_T>(it - boundaries_data);
      } else {
        auto it = std::upper_bound(
            boundaries_data,
            boundaries_data + num_boundaries,
            dval,
            [](double v, const BOUND_T& bound) {
              return v < to_double_val(bound);
            });
        out_data[i] = static_cast<OUT_T>(it - boundaries_data);
      }
    }
  }
}

template <typename BOUND_T, typename OUT_T>
void bucketize_scalar_kernel(
    const Scalar& self,
    const Tensor& boundaries,
    bool right,
    Tensor& out) {
  const BOUND_T* const boundaries_data = boundaries.const_data_ptr<BOUND_T>();
  OUT_T* const out_data = out.mutable_data_ptr<OUT_T>();

  const int64_t num_boundaries = boundaries.numel();
  if (num_boundaries == 0) {
    out_data[0] = 0;
    return;
  }

  double val = 0.0;
  if (self.isFloatingPoint()) {
    val = self.to<double>();
  } else if (self.isIntegral(false)) {
    val = static_cast<double>(self.to<int64_t>());
  } else if (self.isBoolean()) {
    val = static_cast<double>(self.to<bool>());
  }

  if (std::isnan(val)) {
    out_data[0] = static_cast<OUT_T>(num_boundaries);
    return;
  }

  if (right) {
    auto it = std::lower_bound(
        boundaries_data,
        boundaries_data + num_boundaries,
        val,
        [](const BOUND_T& bound, double v) {
          return to_double_val(bound) < v;
        });
    out_data[0] = static_cast<OUT_T>(it - boundaries_data);
  } else {
    auto it = std::upper_bound(
        boundaries_data,
        boundaries_data + num_boundaries,
        val,
        [](double v, const BOUND_T& bound) {
          return v < to_double_val(bound);
        });
    out_data[0] = static_cast<OUT_T>(it - boundaries_data);
  }
}

} // namespace

Tensor& bucketize_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(op_name, "bucketize.Tensor_out");

  ET_KERNEL_CHECK(
      ctx,
      check_bucketize_args(self, boundaries, out_int32, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  if (self.numel() == 0) {
    return out;
  }

  ET_SWITCH_REALHBF16_TYPES(self.scalar_type(), ctx, op_name, IN_T, [&]() {
    ET_SWITCH_REALHBF16_TYPES(
        boundaries.scalar_type(), ctx, op_name, BOUND_T, [&]() {
          if (out_int32) {
            bucketize_kernel<IN_T, BOUND_T, int32_t>(
                self, boundaries, right, out);
          } else {
            bucketize_kernel<IN_T, BOUND_T, int64_t>(
                self, boundaries, right, out);
          }
        });
  });

  return out;
}

Tensor& bucketize_Scalar_out(
    KernelRuntimeContext& ctx,
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(op_name, "bucketize.Scalar_out");

  if (out_int32) {
    ET_KERNEL_CHECK(
        ctx, out.scalar_type() == ScalarType::Int, InvalidArgument, out);
  } else {
    ET_KERNEL_CHECK(
        ctx, out.scalar_type() == ScalarType::Long, InvalidArgument, out);
  }

  ET_KERNEL_CHECK(
      ctx, boundaries.dim() == 1, InvalidArgument, out);

  // Resize out to 0-dim scalar tensor
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ET_SWITCH_REALHBF16_TYPES(
      boundaries.scalar_type(), ctx, op_name, BOUND_T, [&]() {
        if (out_int32) {
          bucketize_scalar_kernel<BOUND_T, int32_t>(
              self, boundaries, right, out);
        } else {
          bucketize_scalar_kernel<BOUND_T, int64_t>(
              self, boundaries, right, out);
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
