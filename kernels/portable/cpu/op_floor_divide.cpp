// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {
/**
 * Element-wise floor_division of `a` and `b`, overwriting `out`.
 * Python's __floordiv__ operator is more complicated than just floor(a / b).
 * It aims to maintain the property: a == (a // b) * b + remainder(a, b)
 * which can otherwise fail due to rounding errors in the remainder.
 * So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
 * With some additional fix-ups added to the result.
 *
 * Assumes that the tensors are contiguous, are the same shape, and have the
 * same dtype. CTYPE should be the C type (like `float` or `int`) that matches
 * the dtype of the tensors.
 */

template <typename CTYPE>
CTYPE floor_divide(CTYPE a, CTYPE b) {
  if constexpr (std::is_integral_v<CTYPE>) {
    const auto quot = a / b;
    if (std::signbit(a) == std::signbit(b)) {
      return quot;
    }
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  } else {
    const auto mod = std::fmod(a, b);
    auto div = (a - mod) / b;
    if ((mod != 0) && std::signbit(b) != std::signbit(mod)) {
      return div - 1;
    }
    return div;
  }
}
template <class CTYPE>
void floor_divide_tensors(const Tensor& a, const Tensor& b, Tensor& out) {
  ET_DCHECK(a.numel() == b.numel() && b.numel() == out.numel());
  const size_t n = a.numel();
  const auto data_a = a.data_ptr<CTYPE>();
  const auto data_b = b.data_ptr<CTYPE>();
  auto data_out = out.data_ptr<CTYPE>();
  for (auto i = 0; i < n; ++i) {
    data_out[i] = floor_divide(data_a[i], data_b[i]);
  }
}

} // namespace

/**
 * Element-wise floor divisions of `a` and `b`, overwriting `out`.
 *
 * Asserts that all tensors have the same dtype and shape.
 *
 * floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& floor_divide_out(
    RuntimeContext& context,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out);

// helper for generating the cases for different data types
#define FLOOR_DIV_TENSORS(ctype, dtype)     \
  case ScalarType::dtype:                   \
    floor_divide_tensors<ctype>(a, b, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES(FLOOR_DIV_TENSORS)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", a.scalar_type());
  }

#undef FLOOR_DIV_TENSORS

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
