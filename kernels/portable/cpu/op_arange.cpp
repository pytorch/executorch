// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

namespace {

template <class CTYPE>
void check_end(const Scalar end) {
  CTYPE end_v = 0;
  bool ok = utils::extract_scalar(end, &end_v);
  ET_CHECK_MSG(ok, "Invalid alpha value: wrong type or out of range");
  ET_CHECK_MSG(end_v >= 0, "end shall be larger than or equal to 0\n");
}

// end here is non-negative scalar, so we can floor it by casting it to int.
template <class CTYPE>
int64_t floor_scalar_to_nearest_int(const Scalar end) {
  CTYPE end_v = 0;
  bool ok = utils::extract_scalar(end, &end_v);
  ET_CHECK_MSG(end_v >= 0, "Input end should be non-negative.");
  ET_CHECK_MSG(ok, "Invalid end value: wrong type or out of range");
  return static_cast<int64_t>(end_v);
}

void check_precondition(const Scalar end, Tensor& out) {
// Check the type consistency between scalar end and tensor out.
// They should be in floating point or integer simultaneously.
#define CHECK_FLOAT_TENSOR(ctype, dtype)                           \
  case ScalarType::dtype:                                          \
    ET_CHECK_MSG(                                                  \
        end.isFloatingPoint(),                                     \
        "end should have same type as out.dtype, but get \
         non-floating point end and a floating point out tensor"); \
    break;

#define CHECK_INT_TENSOR(ctype, dtype)              \
  case ScalarType::dtype:                           \
    ET_CHECK_MSG(                                   \
        end.isIntegral(true),                       \
        "end should have same type as out, \
        but get non-int end and a int out tensor"); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_FLOAT_TYPES(CHECK_FLOAT_TENSOR);
    ET_FORALL_INT_TYPES_AND(Bool, CHECK_INT_TENSOR);
    default:
      ET_CHECK_MSG(
          false,
          "out tensor should be in floating point or int dtype, but get %hhd",
          out.scalar_type());
  }

#undef CHECK_FLOAT_TENSOR
#undef CHECK_INT_TENSOR

  ET_CHECK_MSG(
      out.sizes().size() == 1,
      "out should be a 1-d tensor, but got a %zu-d tensor",
      out.sizes().size());

  // Check if out size matches end.

  // Set includeBool = false here because the following extract_scalar for int
  // use includeBool = False. Have deal with boolean type separately.
  if (end.isIntegral(false)) {
    check_end<int64_t>(end);
  } else if (end.isFloatingPoint()) {
    check_end<double>(end);
  } else if (end.isBoolean()) {
    check_end<bool>(end);
  } else {
    ET_CHECK_MSG(
        false, "Unexepcted type of end. Should be floating point or int type");
  }
};

template <class CTYPE>
Tensor& set_arange_value(const size_t out_length, Tensor& out) {
  auto out_data = out.data_ptr<CTYPE>();
  for (size_t i = 0; i < out_length; i++) {
    out_data[i] = static_cast<CTYPE>(i);
  }
  return out;
}

} // namespace

/*
 * Fill out tensor using arange(0, end)
 *
 * arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& arange_out(RuntimeContext& context, const Scalar& end, Tensor& out) {
  check_precondition(end, out);

  int64_t end_floor = 0;
  if (end.isIntegral(false)) {
    end_floor = floor_scalar_to_nearest_int<int64_t>(end);
  } else if (end.isFloatingPoint()) {
    end_floor = floor_scalar_to_nearest_int<double>(end);
  } else if (end.isBoolean()) {
    end_floor = floor_scalar_to_nearest_int<bool>(end);
  } else {
    ET_CHECK_MSG(false, "Unhandled scalar type");
  }

  Tensor::SizesType out_target_length =
      static_cast<Tensor::SizesType>(end_floor);
  Error status = resize_tensor(out, {&out_target_length, 1});
  ET_CHECK_MSG(status == Error::Ok, "resize_tensor fails\n");

#define SET_ARANGE_VALUE_TO_TENSOR(ctype, dtype)   \
  case ScalarType::dtype:                          \
    out = set_arange_value<ctype>(end_floor, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, SET_ARANGE_VALUE_TO_TENSOR)
    default:
      ET_CHECK_MSG(
          false,
          "out tensor should be in floating point or int dtype, but get %hhd",
          out.scalar_type());
  }
#undef SET_ARANGE_VALUE_TO_TENSOR

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
