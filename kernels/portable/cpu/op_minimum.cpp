// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <class T>
const T& min(const T& a, const T& b) {
  return (b < a) ? b : a;
}

template <class CTYPE>
void minimum_out_helper(const Tensor& a, const Tensor& b, Tensor& out) {
  const size_t a_numel = a.numel();
  const size_t b_numel = b.numel();

  const auto data_a = a.data_ptr<CTYPE>();
  const auto data_b = b.data_ptr<CTYPE>();
  const auto data_out = out.data_ptr<CTYPE>();

  if (a_numel == b_numel) {
    ET_CHECK_SAME_SHAPE3(a, b, out);

    for (size_t i = 0; i < a_numel; ++i) {
      data_out[i] = min(data_a[i], data_b[i]);
    }
  } else if (a_numel == 1) {
    ET_CHECK_SAME_SHAPE2(b, out);
    auto singleton_el = data_a[0];

    for (size_t i = 0; i < b_numel; ++i) {
      data_out[i] = min(singleton_el, data_b[i]);
    }
  } else if (b_numel == 1) {
    ET_CHECK_SAME_SHAPE2(a, out);
    auto singleton_el = data_b[0];

    for (size_t i = 0; i < a_numel; ++i) {
      data_out[i] = min(data_a[i], singleton_el);
    }
  } else {
    ET_CHECK_MSG(
        false,
        "Mismatched dimension a: %zu, b: %zu, out: %zu",
        a_numel,
        b_numel,
        out.numel());
  }
}

} // namespace

/**
 * Given input tensors a and b, does element-wise minimum operation and saves
 * the result in out tensor, and also returns the same tensor.
 *
 * 1) If a, b and out have the same shape, then the operation does element-wise,
 * minimum operation. out(x) = min(a(x), b(x)).
 *
 * 2) If either a or b is of size 1x1, then it compares this singleton element
 * with every element in the other tensor, and saves the output in out.
 * For example, if a is a singleton tensor, then out(x) = min(a(0), b(x)).
 *
 * Asserts that the all dtypes must be the same.
 */
Tensor& minimum_out(
    RuntimeContext& context,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)context;
  ET_CHECK_SAME_DTYPE3(a, b, out);

#define MINIMUM_TENSOR(ctype, dtype)      \
  case ScalarType::dtype:                 \
    minimum_out_helper<ctype>(a, b, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES(MINIMUM_TENSOR)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", a.scalar_type());
  }
#undef MINIMUM_TENSOR
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
