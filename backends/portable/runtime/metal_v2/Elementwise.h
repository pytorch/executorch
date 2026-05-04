/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Elementwise op layout classification.

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::etensor::Tensor;

// Classifies an elementwise binary op's input layout to pick the fastest
// kernel specialization. Mirrors mlx::core::BinaryOpType.
enum class ElementwiseVariant {
  ScalarScalar, // both inputs are 1-element
  ScalarVector, // a is scalar, b is contiguous vector
  VectorScalar, // a is contiguous vector, b is scalar
  VectorVector, // both inputs are same shape and contiguous
  General,      // arbitrary strides / broadcast required
};

inline const char* variantPrefix(ElementwiseVariant v) {
  switch (v) {
    case ElementwiseVariant::ScalarScalar: return "ss";
    case ElementwiseVariant::ScalarVector: return "sv";
    case ElementwiseVariant::VectorScalar: return "vs";
    case ElementwiseVariant::VectorVector: return "vv";
    case ElementwiseVariant::General:      return "g";
  }
  return "g";
}

// True if all dims of `t` have stride matching a packed row-major
// (innermost-fastest) layout. Equivalent to MLX's `flags().row_contiguous`.
inline bool isRowContiguous(const Tensor& t) {
  auto sizes = t.sizes();
  auto strides = t.strides();
  if (sizes.size() != strides.size()) return false;
  if (sizes.empty()) return true;
  int64_t expected = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    if (sizes[i] == 1) continue; // size-1 dim's stride is irrelevant
    if (strides[i] != expected) return false;
    expected *= sizes[i];
  }
  return true;
}

inline bool sameShape(const Tensor& a, const Tensor& b) {
  auto as = a.sizes();
  auto bs = b.sizes();
  if (as.size() != bs.size()) return false;
  for (size_t i = 0; i < as.size(); ++i) {
    if (as[i] != bs[i]) return false;
  }
  return true;
}

inline ElementwiseVariant classifyBinary(const Tensor& a, const Tensor& b) {
  bool a_scalar = (a.numel() == 1);
  bool b_scalar = (b.numel() == 1);
  if (a_scalar && b_scalar) return ElementwiseVariant::ScalarScalar;
  if (a_scalar && isRowContiguous(b)) return ElementwiseVariant::ScalarVector;
  if (b_scalar && isRowContiguous(a)) return ElementwiseVariant::VectorScalar;
  if (sameShape(a, b) && isRowContiguous(a) && isRowContiguous(b)) {
    return ElementwiseVariant::VectorVector;
  }
  return ElementwiseVariant::General;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
