/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array> // std::array
#include <cinttypes> // PRId64
#include <cmath>
#include <cstddef> // size_t
#include <limits>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/compiler.h>

/// All assertion messages should begin with this prefix.
#define ET_TENSOR_CHECK_PREFIX__ "Tensors do not match"
#define ET_MIN2(a, b) (std::min(a, b))
#define ET_MIN3(a, b, c) (std::min(a, std::min(b, c)))

#define ET_NORMALIZE_IX(IX, UPPER_BOUND) IX < 0 ? IX + UPPER_BOUND : IX

#define ET_CHECK_VALID_IX(IX, UPPER_BOUND)                  \
  ET_CHECK_MSG(                                             \
      IX >= -static_cast<int64_t>(UPPER_BOUND) &&           \
          IX < static_cast<int64_t>(UPPER_BOUND),           \
      "index %" PRId64 " must be within range [-%zd, %zd)", \
      IX,                                                   \
      UPPER_BOUND,                                          \
      UPPER_BOUND)

#define ET_CHECK_VALID_DIM(DIM, UPPER_BOUND)              \
  ET_CHECK_MSG(                                           \
      DIM >= -static_cast<int64_t>(UPPER_BOUND) &&        \
          DIM < static_cast<int64_t>(UPPER_BOUND),        \
      "dim %" PRId64 " must be within range [-%zd, %zd)", \
      DIM,                                                \
      UPPER_BOUND,                                        \
      UPPER_BOUND)

#define ET_CHECK_NON_ZERO_DIM_SIZE(DIM, T)           \
  const size_t udim = ET_NORMALIZE_IX(DIM, T.dim()); \
  ET_CHECK_MSG(                                      \
      T.size(udim) != 0, "Expected dim %zd to have non-zero size.", udim);

/**
 * Asserts that all tensors have the same shape.
 * This also handles a edge case where there is only one element in all the
 * tensors being compared but the number of dimensions >= 0. In the for loop
 * iterating over the dimensions we make sure that we pick the smallest
 * dimension of all the tensors as the upper bound for the for loop.
 */
#define ET_CHECK_SAME_SHAPE2(a__, b__)                                    \
  ({                                                                      \
    const size_t a_numel__ = (a__).numel();                               \
    const size_t b_numel__ = (b__).numel();                               \
    const size_t a_dim__ = (a__).dim();                                   \
    const size_t b_dim__ = (b__).dim();                                   \
    ET_CHECK_MSG(                                                         \
        a_numel__ == b_numel__ &&                                         \
            ((a_numel__ == 1 && b_numel__ == 1) || (a_dim__ == b_dim__)), \
        ET_TENSOR_CHECK_PREFIX__ ": numel={%zu, %zu}, dim={%zu, %zu}",    \
        a_numel__,                                                        \
        b_numel__,                                                        \
        a_dim__,                                                          \
        b_dim__);                                                         \
    for (size_t dim__ = 0; dim__ < ET_MIN2(a_dim__, b_dim__); ++dim__) {  \
      size_t a_size__ = (a__).size(dim__);                                \
      size_t b_size__ = (b__).size(dim__);                                \
      ET_CHECK_MSG(                                                       \
          a_size__ == b_size__,                                           \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu}",           \
          dim__,                                                          \
          a_size__,                                                       \
          b_size__);                                                      \
    }                                                                     \
  })

#define ET_CHECK_SAME_SHAPE3(a__, b__, c__)                            \
  ({                                                                   \
    const size_t a_numel__ = (a__).numel();                            \
    const size_t b_numel__ = (b__).numel();                            \
    const size_t c_numel__ = (c__).numel();                            \
    const size_t a_dim__ = (a__).dim();                                \
    const size_t b_dim__ = (b__).dim();                                \
    const size_t c_dim__ = (c__).dim();                                \
    ET_CHECK_MSG(                                                      \
        a_numel__ == b_numel__ && b_numel__ == c_numel__ &&            \
            ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1) ||   \
             a_dim__ == b_dim__ && b_dim__ == c_dim__),                \
        ET_TENSOR_CHECK_PREFIX__                                       \
        ": numel={%zu, %zu, %zu}, dim={%zu, %zu, %zu}",                \
        a_numel__,                                                     \
        b_numel__,                                                     \
        c_numel__,                                                     \
        a_dim__,                                                       \
        b_dim__,                                                       \
        c_dim__);                                                      \
    for (size_t dim__ = 0; dim__ < ET_MIN3(a_dim__, b_dim__, c_dim__); \
         ++dim__) {                                                    \
      size_t a_size__ = (a__).size(dim__);                             \
      size_t b_size__ = (b__).size(dim__);                             \
      size_t c_size__ = (c__).size(dim__);                             \
      ET_CHECK_MSG(                                                    \
          a_size__ == b_size__ && b_size__ == c_size__,                \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu, %zu}",   \
          dim__,                                                       \
          a_size__,                                                    \
          b_size__,                                                    \
          c_size__);                                                   \
    }                                                                  \
  })

/// Asserts that all tensors have the same dtype.
#define ET_CHECK_SAME_DTYPE2(a__, b__)                                   \
  ({                                                                     \
    const ::executorch::aten::ScalarType a_type__ = (a__).scalar_type(); \
    const ::executorch::aten::ScalarType b_type__ = (b__).scalar_type(); \
    ET_CHECK_MSG(                                                        \
        a_type__ == b_type__,                                            \
        ET_TENSOR_CHECK_PREFIX__ ": dtype={%" PRId8 ", %" PRId8 "}",     \
        static_cast<int8_t>(a_type__),                                   \
        static_cast<int8_t>(b_type__));                                  \
  })

#define ET_CHECK_SAME_DTYPE3(a__, b__, c__)                                 \
  ({                                                                        \
    const ::executorch::aten::ScalarType a_type__ = (a__).scalar_type();    \
    const ::executorch::aten::ScalarType b_type__ = (b__).scalar_type();    \
    const ::executorch::aten::ScalarType c_type__ = (c__).scalar_type();    \
    ET_CHECK_MSG(                                                           \
        a_type__ == b_type__ && b_type__ == c_type__,                       \
        ET_TENSOR_CHECK_PREFIX__ ": dtype={%" PRId8 ", %" PRId8 ", %" PRId8 \
                                 "}",                                       \
        static_cast<int8_t>(a_type__),                                      \
        static_cast<int8_t>(b_type__),                                      \
        static_cast<int8_t>(c_type__));                                     \
  })

/**
 * Asserts that all tensors have the same shape and dtype.
 *
 * This macro should produce less code/data than calling the SHAPE and DTYPE
 * macros independently, because it only calls ET_CHECK_MSG once.
 */
#define ET_CHECK_SAME_SHAPE_AND_DTYPE2(a__, b__)                              \
  ({                                                                          \
    const size_t a_numel__ = (a__).numel();                                   \
    const size_t b_numel__ = (b__).numel();                                   \
    const size_t a_dim__ = (a__).dim();                                       \
    const size_t b_dim__ = (b__).dim();                                       \
    const ::executorch::aten::ScalarType a_type__ = (a__).scalar_type();      \
    const ::executorch::aten::ScalarType b_type__ = (b__).scalar_type();      \
                                                                              \
    ET_CHECK_MSG(                                                             \
        a_numel__ == b_numel__ &&                                             \
            ((a_numel__ == 1 && b_numel__ == 1) || a_dim__ == b_dim__) &&     \
            a_type__ == b_type__,                                             \
        ET_TENSOR_CHECK_PREFIX__                                              \
        ": numel={%zu, %zu}, dim={%zu, %zu}, dtype={%" PRId8 ", %" PRId8 "}", \
        a_numel__,                                                            \
        b_numel__,                                                            \
        a_dim__,                                                              \
        b_dim__,                                                              \
        static_cast<int8_t>(a_type__),                                        \
        static_cast<int8_t>(b_type__));                                       \
    for (size_t dim__ = 0; dim__ < ET_MIN2(a_dim__, b_dim__); ++dim__) {      \
      size_t a_size__ = (a__).size(dim__);                                    \
      size_t b_size__ = (b__).size(dim__);                                    \
      ET_CHECK_MSG(                                                           \
          a_size__ == b_size__,                                               \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu}",               \
          dim__,                                                              \
          a_size__,                                                           \
          b_size__);                                                          \
    }                                                                         \
  })

#define ET_CHECK_SAME_SHAPE_AND_DTYPE3(a__, b__, c__)                    \
  ({                                                                     \
    const size_t a_numel__ = (a__).numel();                              \
    const size_t b_numel__ = (b__).numel();                              \
    const size_t c_numel__ = (c__).numel();                              \
    const size_t a_dim__ = (a__).dim();                                  \
    const size_t b_dim__ = (b__).dim();                                  \
    const size_t c_dim__ = (c__).dim();                                  \
    const ::executorch::aten::ScalarType a_type__ = (a__).scalar_type(); \
    const ::executorch::aten::ScalarType b_type__ = (b__).scalar_type(); \
    const ::executorch::aten::ScalarType c_type__ = (c__).scalar_type(); \
                                                                         \
    ET_CHECK_MSG(                                                        \
        a_numel__ == b_numel__ && b_numel__ == c_numel__ &&              \
            ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1) ||     \
             (a_dim__ == b_dim__ && b_dim__ == c_dim__)) &&              \
            a_type__ == b_type__ && b_type__ == c_type__,                \
        ET_TENSOR_CHECK_PREFIX__                                         \
        ": numel={%zu, %zu, %zu}, dim={%zu, %zu, %zu}, "                 \
        "dtype={%" PRId8 ", %" PRId8 ", %" PRId8 "}",                    \
        a_numel__,                                                       \
        b_numel__,                                                       \
        c_numel__,                                                       \
        a_dim__,                                                         \
        b_dim__,                                                         \
        c_dim__,                                                         \
        static_cast<int8_t>(a_type__),                                   \
        static_cast<int8_t>(b_type__),                                   \
        static_cast<int8_t>(c_type__));                                  \
    for (size_t dim__ = 0; dim__ < ET_MIN3(a_dim__, b_dim__, c_dim__);   \
         ++dim__) {                                                      \
      size_t a_size__ = (a__).size(dim__);                               \
      size_t b_size__ = (b__).size(dim__);                               \
      size_t c_size__ = (c__).size(dim__);                               \
      ET_CHECK_MSG(                                                      \
          a_size__ == b_size__ && b_size__ == c_size__,                  \
          ET_TENSOR_CHECK_PREFIX__ " at size(%zu): {%zu, %zu, %zu}",     \
          dim__,                                                         \
          a_size__,                                                      \
          b_size__,                                                      \
          c_size__);                                                     \
    }                                                                    \
  })

/**
 * Assert that the input tensor is contiguous tensor.
 */
#define ET_CHECK_CONTIGUOUS(a__)                                              \
  ({                                                                          \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>         \
        strides = a__.strides();                                              \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType> sizes = \
        a__.sizes();                                                          \
    ET_CHECK_MSG(                                                             \
        strides[strides.size() - 1] == 1,                                     \
        "The stride of the last dimension shall be 1 for contiguous tensor, " \
        "not %d",                                                             \
        strides[strides.size() - 1]);                                         \
    for (size_t i = strides.size() - 1; i > 0; i--) {                         \
      ET_CHECK_MSG(                                                           \
          strides[i - 1] == strides[i] * sizes[i],                            \
          "The stride of the %zu-th dimension shall equal to "                \
          "strides[%zu] * sizes[%zu], now is %d and %d",                      \
          i - 1,                                                              \
          i,                                                                  \
          i,                                                                  \
          strides[i - 1],                                                     \
          strides[i] * sizes[i]);                                             \
    }                                                                         \
  })

/**
 * Assert the input two tensors share same strides.
 * Noted that this function does not make any check or promise on the contiguity
 * of any input tensors.
 */
#define ET_CHECK_SAME_STRIDES2(a__, b__)                                       \
  ({                                                                           \
    ET_CHECK_MSG(                                                              \
        a__.dim() == b__.dim(),                                                \
        "Two tensors shall have same number of strides, but not %zu and %zu.", \
        a__.dim(),                                                             \
        b__.dim());                                                            \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>          \
        a_strides = a__.strides();                                             \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>          \
        b_strides = b__.strides();                                             \
    for (size_t i = 0; i < a__.dim(); i++) {                                   \
      ET_CHECK_MSG(                                                            \
          a_strides[i] == b_strides[i],                                        \
          "a.strides()[%zu] shall equal to b.strides()[%zu], "                 \
          "but now is %d and %d.",                                             \
          i,                                                                   \
          i,                                                                   \
          (int32_t)a_strides[i],                                               \
          (int32_t)b_strides[i]);                                              \
    }                                                                          \
  })

/**
 * Assert the input three tensors share same strides.
 * Noted that this function does not make any check or promise on the contiguity
 * of any input tensors.
 */
#define ET_CHECK_SAME_STRIDES3(a__, b__, c__)                           \
  ({                                                                    \
    ET_CHECK_MSG(                                                       \
        a__.dim() == b__.dim() && b__.dim() == c__.dim(),               \
        "Three tensors shall have same number of strides, "             \
        "but not %zu, %zu and %zu.",                                    \
        a__.dim(),                                                      \
        b__.dim(),                                                      \
        c__.dim());                                                     \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>   \
        a_strides = a__.strides();                                      \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>   \
        b_strides = b__.strides();                                      \
    const ::executorch::aten::ArrayRef<executorch::aten::StridesType>   \
        c_strides = c__.strides();                                      \
    for (size_t i = 0; i < a__.dim(); i++) {                            \
      ET_CHECK_MSG(                                                     \
          a_strides[i] == b_strides[i] && b_strides[i] == c_strides[i], \
          "a_strides[%zu], b_strides[%zu] and c_strides[%zu] "          \
          "shall share same value, but now is %d, %d and %d",           \
          i,                                                            \
          i,                                                            \
          i,                                                            \
          (int32_t)a_strides[i],                                        \
          (int32_t)b_strides[i],                                        \
          (int32_t)c_strides[i]);                                       \
    }                                                                   \
  })

#define ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(t__)           \
  ({                                                             \
    ET_CHECK_MSG(                                                \
        is_contiguous_dim_order(                                 \
            t__.dim_order().data(), t__.dim_order().size()) ||   \
            is_channels_last_dim_order(                          \
                t__.dim_order().data(), t__.dim_order().size()), \
        "Tensor must have default or channels last dim order");  \
  })

/**
 * A convenience macro to be used in utility functions that check whether input
 * tensor(s) are valid, which are expected to return a boolean. Checks whether
 * `cond` is true; if not, log the failed check and return false.
 *
 * @param[in] cond the condition to check
 */
#define ET_LOG_AND_RETURN_IF_FALSE(cond)           \
  do {                                             \
    if (!(cond)) {                                 \
      ET_LOG(Error, "Check failed (%s): ", #cond); \
      return false;                                \
    }                                              \
  } while (false)

/**
 * A convenience macro to be used in utility functions that check whether input
 * tensor(s) are valid, which are expected to return a boolean. Checks whether
 * `cond` is true; if not, log the failed check with `message` and return false.
 *
 * @param[in] cond the condition to check
 * @param[in] message an additional message to log with `cond`
 */
#define ET_LOG_MSG_AND_RETURN_IF_FALSE(cond, message, ...)                \
  do {                                                                    \
    if (!(cond)) {                                                        \
      ET_LOG(Error, "Check failed (%s): " message, #cond, ##__VA_ARGS__); \
      return false;                                                       \
    }                                                                     \
  } while (false)

/**
 * If `cond` is false, log `cond` and return from the kernel with a failure
 * state set.
 *
 * @param[in] context the runtime context
 * @param[in] cond the condition to check
 * @param[in] error torch::executor::Error enum value (e.g `InvalidArgument`)
 * @param[in] retval return value of the kernel to allow for early exit
 */
#define ET_KERNEL_CHECK(context, cond, error, retval) \
  do {                                                \
    if (!(cond)) {                                    \
      ET_LOG(Error, "Check failed (%s): ", #cond);    \
      context.fail(torch::executor::Error::error);    \
      return retval;                                  \
    }                                                 \
  } while (false)

/**
 * If `cond` is false, log `message` and return from the kernel with a failure
 * state set.
 *
 * @param[in] context the runtime context
 * @param[in] cond the condition to check
 * @param[in] error torch::executor::Error enum value (e.g `InvalidArgument`)
 * @param[in] retval return value of the kernel to allow for early exit
 */
#define ET_KERNEL_CHECK_MSG(context, cond, error, retval, message, ...)   \
  do {                                                                    \
    if (!(cond)) {                                                        \
      ET_LOG(Error, "Check failed (%s): " message, #cond, ##__VA_ARGS__); \
      context.fail(torch::executor::Error::error);                        \
      return retval;                                                      \
    }                                                                     \
  } while (false)

/**
 * Convenience macro to extract a scalar tensor into a value
 */
#define ET_EXTRACT_SCALAR_TENSOR(scalar_tensor, out_val) \
  ET_CHECK_MSG(                                          \
      extract_scalar_tensor(scalar_tensor, &out_val),    \
      #scalar_tensor " could not be extracted: wrong type or out of range");

namespace executorch {
namespace runtime {

//
// Utility functions for checking tensor attributes
//
//

/*
 * Returns true if the given dimension value is between -upper_bound and
 * upper_bound - 1, inclusive.
 */
inline bool dim_is_valid(int64_t dim, int64_t upper_bound) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim >= -upper_bound && dim < upper_bound,
      "Dimension %" PRId64
      " is out of range. Dimension should be between %" PRId64 " and %" PRId64
      ", inclusive.",
      dim,
      -upper_bound,
      upper_bound - 1);

  return true;
}

/*
 * Returns the tensor's number of dimensions, except when the tensor is zero
 * dimensional. In this case, it returns 1. This is used to properly handle
 * the zero dimensional tensors in some kernels, that treat them as 1D tensors
 * with a single element.
 */
inline ssize_t nonzero_dim(const executorch::aten::Tensor& tensor) {
  return tensor.dim() == 0 ? 1 : tensor.dim();
}

/*
 * Returns the size along a dimension dim, except when the tensor is zero
 * dimensional. In this case, it returns 1. This is used to properly handle
 * the zero dimensional tensors in some kernels, that treat them as 1D tensors
 * with a single element.
 */
inline ssize_t nonempty_size(
    const executorch::aten::Tensor& tensor,
    ssize_t dim) {
  return tensor.dim() == 0 ? 1 : tensor.size(dim);
}

inline bool tensor_can_cast_to(
    executorch::aten::Tensor a,
    executorch::aten::ScalarType dtype) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::canCast(a.scalar_type(), dtype),
      "Tensor of dtype %s cannot cast to dtype %s",
      torch::executor::toString(a.scalar_type()),
      torch::executor::toString(dtype));

  return true;
}

inline bool tensor_is_bool_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      t.scalar_type() == executorch::aten::ScalarType::Bool,
      "Expected to find bool type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_integral_type(
    executorch::aten::Tensor t,
    bool includeBool = false) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isIntegralType(t.scalar_type(), includeBool),
      "Expected to find a integral type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_floating_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isFloatingType(t.scalar_type()),
      "Expected to find a floating type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_real_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isRealType(t.scalar_type()),
      "Expected to find a real type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_realh_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isRealHType(t.scalar_type()),
      "Expected to find a real type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_realhb_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isRealHBType(t.scalar_type()),
      "Expected to find a real type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_realhbbf16_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      executorch::runtime::isRealHBBF16Type(t.scalar_type()),
      "Expected to find a real type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_complex_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isComplexType(t.scalar_type()),
      "Expected to find a complex type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensor_is_bits_type(executorch::aten::Tensor t) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      torch::executor::isBitsType(t.scalar_type()),
      "Expected to find a bits type, but tensor has type %s",
      torch::executor::toString(t.scalar_type()));

  return true;
}

inline bool tensors_have_same_dtype(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      a.scalar_type() == b.scalar_type(),
      ET_TENSOR_CHECK_PREFIX__ ": dtype={%s, %s}",
      torch::executor::toString(a.scalar_type()),
      torch::executor::toString(b.scalar_type()));
  return true;
}

inline bool tensors_have_same_dtype(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b,
    executorch::aten::Tensor c) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      a.scalar_type() == b.scalar_type() && b.scalar_type() == c.scalar_type(),
      ET_TENSOR_CHECK_PREFIX__ ": dtype={%s, %s, %s}",
      torch::executor::toString(a.scalar_type()),
      torch::executor::toString(b.scalar_type()),
      torch::executor::toString(c.scalar_type()));
  return true;
}

inline bool tensor_is_rank(executorch::aten::Tensor t, size_t rank) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      t.dim() == rank,
      "Expected tensor.dim() to be %zu, but got %zu",
      static_cast<size_t>(rank),
      static_cast<size_t>(t.dim()));

  return true;
}

inline bool tensor_has_rank_greater_or_equal_to(
    executorch::aten::Tensor t,
    size_t rank) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      t.dim() >= rank,
      "Expected tensor.dim() to be >= %zu, but got %zu",
      static_cast<size_t>(rank),
      static_cast<size_t>(t.dim()));

  return true;
}

inline bool tensor_has_rank_smaller_or_equal_to(
    executorch::aten::Tensor t,
    size_t rank) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      t.dim() <= rank,
      "Expected tensor.dim() to be <= %zu, but got %zu",
      static_cast<size_t>(rank),
      static_cast<size_t>(t.dim()));

  return true;
}

inline bool tensor_has_dim(executorch::aten::Tensor t, int64_t d) {
  if (t.dim() == 0) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        d == 0 || d == -1,
        "dim must be 0 or -1 for 0-dim tensor, got %" PRId64,
        d);
  } else {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        d > 0 ? d < t.dim() : t.dim() + d >= 0,
        "%zu-dim tensor does not have dim at index %zu",
        static_cast<size_t>(t.dim()),
        static_cast<size_t>(d));
  }
  return true;
}

inline bool tensor_has_non_empty_dim(executorch::aten::Tensor t, int64_t d) {
  const size_t udim = ET_NORMALIZE_IX(d, t.dim());
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(t, d));
  ET_LOG_AND_RETURN_IF_FALSE(t.size(udim) != 0);
  return true;
}

inline bool
tensor_dim_has_index(executorch::aten::Tensor t, int64_t d, int64_t ix) {
  // Indexing ops don't support zero-dim tensors
  ET_CHECK(t.dim() != 0);
  if (d < 0) {
    d += t.dim();
  }
  // Dimension must have been already checked by tensor_has_dim
  ET_CHECK(d >= 0 && d < t.dim());

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      ix >= -t.size(d) && ix < t.size(d),
      "index %" PRId64 " out of range [-%zu,%zu) at dimension %" PRId64 ")",
      ix,
      static_cast<size_t>(t.size(d)),
      static_cast<size_t>(t.size(d)),
      d);
  return true;
}

inline bool tensors_have_same_size_at_dims(
    executorch::aten::Tensor a,
    size_t dim_a,
    executorch::aten::Tensor b,
    size_t dim_b) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim_a < a.dim(),
      "Cannot retrieve dim %zu from tensor with dim %zu",
      static_cast<size_t>(dim_a),
      static_cast<size_t>(a.dim()));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim_b < b.dim(),
      "Cannot retrieve dim %zu from tensor with dim %zu",
      static_cast<size_t>(dim_b),
      static_cast<size_t>(b.dim()));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      a.size(dim_a) == b.size(dim_b),
      ET_TENSOR_CHECK_PREFIX__
      ": a.size(%zu) = %zu does not match b.size(%zu) = %zu",
      static_cast<size_t>(dim_a),
      static_cast<size_t>(a.size(dim_a)),
      static_cast<size_t>(dim_b),
      static_cast<size_t>(b.size(dim_b)));

  return true;
}

inline bool tensors_have_same_shape(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b) {
  if (a.numel() == 1 && b.numel() == 1) {
    // PyTorch operators treat all scalar tensors as the same shape even if
    // they have different dims.
    return true;
  }
  if (!(a.sizes() == b.sizes() && a.numel() == b.numel())) {
    ET_LOG(
        Error,
        ET_TENSOR_CHECK_PREFIX__ ": numel=(%zu,  %zu), dim=(%zu, %zu)",
        static_cast<size_t>(a.numel()),
        static_cast<size_t>(b.numel()),
        static_cast<size_t>(a.dim()),
        static_cast<size_t>(b.dim()));
    for (size_t d = 0; d < ET_MIN2(a.dim(), b.dim()); ++d) {
      ET_LOG(
          Error,
          "    size(%zu): (%zu, %zu)",
          static_cast<size_t>(d),
          static_cast<size_t>(a.size(d)),
          static_cast<size_t>(b.size(d)));
    }

    return false;
  }

  return true;
}

inline bool tensors_have_same_shape(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b,
    executorch::aten::Tensor c) {
  if (a.numel() == 1 && b.numel() == 1 && c.numel() == 1) {
    // PyTorch operators treat all scalar tensors as the same shape even if
    // they have different dims.
    return true;
  }
  bool cond1 = (a.sizes() == b.sizes()) && (a.numel() == b.numel());
  bool cond2 = (b.sizes() == c.sizes()) && (b.numel() == c.numel());

  if (!(cond1 && cond2)) {
    ET_LOG(
        Error,
        ET_TENSOR_CHECK_PREFIX__ ": numel=(%zu, %zu, %zu), dim=(%zu, %zu, %zu)",
        static_cast<size_t>(a.numel()),
        static_cast<size_t>(b.numel()),
        static_cast<size_t>(c.numel()),
        static_cast<size_t>(a.dim()),
        static_cast<size_t>(b.dim()),
        static_cast<size_t>(c.dim()));
    for (size_t d = 0; d < ET_MIN3(a.dim(), b.dim(), c.dim()); ++d) {
      ET_LOG(
          Error,
          "    size(%zu): (%zu, %zu, %zu)",
          static_cast<size_t>(d),
          static_cast<size_t>(a.size(d)),
          static_cast<size_t>(b.size(d)),
          static_cast<size_t>(c.size(d)));
    }

    return false;
  }

  return true;
}

inline bool tensors_have_same_shape_and_dtype(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b) {
  return tensors_have_same_shape(a, b) && tensors_have_same_dtype(a, b);
}

inline bool tensors_have_same_shape_and_dtype(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b,
    executorch::aten::Tensor c) {
  return tensors_have_same_shape(a, b, c) && tensors_have_same_dtype(a, b, c);
}

inline bool tensor_has_expected_size(
    executorch::aten::Tensor a,
    executorch::aten::ArrayRef<executorch::aten::SizesType> expected_sizes) {
  if (!(a.sizes() == expected_sizes)) {
    ET_LOG(
        Error,
        ET_TENSOR_CHECK_PREFIX__ ": dim=(%zu, %zu)",
        static_cast<size_t>(a.dim()),
        static_cast<size_t>(expected_sizes.size()));
    size_t a_dim = static_cast<size_t>(a.dim());
    size_t expected_dim = static_cast<size_t>(expected_sizes.size());
    for (size_t d = 0; d < ET_MIN2(a_dim, expected_dim); ++d) {
      ET_LOG(
          Error,
          "    size(%zu): (%zu, %zu)",
          static_cast<size_t>(d),
          static_cast<size_t>(a.size(d)),
          static_cast<size_t>(expected_sizes[d]));
    }

    return false;
  }
  return true;
}

inline bool tensors_have_same_strides(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b) {
  if (a.strides() != b.strides()) {
    ET_LOG(
        Error,
        ET_TENSOR_CHECK_PREFIX__ ": dim=(%zu, %zu)",
        static_cast<size_t>(a.dim()),
        static_cast<size_t>(b.dim()));
    for (size_t d = 0; d < ET_MIN2(a.dim(), b.dim()); ++d) {
      ET_LOG(
          Error,
          "    stride(%zu): (%zu, %zu)",
          static_cast<size_t>(d),
          static_cast<size_t>(a.strides()[d]),
          static_cast<size_t>(b.strides()[d]));
    }

    return false;
  }
  return true;
}

inline bool tensors_have_same_strides(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b,
    executorch::aten::Tensor c) {
  if (!(a.strides() == b.strides() && b.strides() == c.strides())) {
    ET_LOG(
        Error,
        ET_TENSOR_CHECK_PREFIX__ ": dim=(%zu, %zu, %zu)",
        static_cast<size_t>(a.dim()),
        static_cast<size_t>(b.dim()),
        static_cast<size_t>(c.dim()));
    for (size_t d = 0; d < ET_MIN3(a.dim(), b.dim(), c.dim()); ++d) {
      ET_LOG(
          Error,
          "    stride(%zu): (%zu, %zu, %zu)",
          static_cast<size_t>(d),
          static_cast<size_t>(a.strides()[d]),
          static_cast<size_t>(b.strides()[d]),
          static_cast<size_t>(c.strides()[d]));
    }

    return false;
  }
  return true;
}

inline bool tensor_is_contiguous(executorch::aten::Tensor t) {
  const auto strides = t.strides();
  const auto sizes = t.sizes();
  // If tensor is 0-dim (i.e. a scalar tensor) it is contiguous
  if (strides.size() == 0) {
    return true;
  }
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      strides[strides.size() - 1] == 1,
      "Tensor is not contiguous; the stride of the last dimension must be 1, "
      "but got %zu",
      static_cast<size_t>(strides[strides.size() - 1]));
  for (int i = strides.size() - 1; i > 0; --i) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        strides[i - 1] == strides[i] * sizes[i],
        "Tensor is not contiguous; the stride of dim %zu should be equal to "
        "strides[%zu] * sizes[%zu] = %zu, but found %zu",
        static_cast<size_t>(i - 1),
        static_cast<size_t>(i),
        static_cast<size_t>(i),
        static_cast<size_t>(strides[i] * sizes[i]),
        static_cast<size_t>(strides[i - 1]));
  }
  return true;
}

inline bool tensors_have_same_rank(
    executorch::aten::Tensor a,
    executorch::aten::Tensor b) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      a.dim() == b.dim(),
      ET_TENSOR_CHECK_PREFIX__ ": rank={%zd, %zd}",
      ssize_t(a.dim()),
      ssize_t(b.dim()));
  return true;
}

inline bool tensor_is_scalar(executorch::aten::Tensor t) {
  return t.dim() == 0 && t.numel() == 1;
}

/**
 * The expected output size may not be the existing size of any inputs and
 * outputs if the operator supports both broadcast and dynamic shape.
 * Therefore such operators needs extra space to store the calculated expected
 * output size. such dynamic allocation is troublesome in executorch so we can
 * just hard code a static value of a relatively small value because users
 * don't create high dimensional tensors.
 */
constexpr size_t kTensorDimensionLimit = 16;

/// Returns the product of dim[0:dim), not including dim.
inline size_t getLeadingDims(
    const executorch::aten::Tensor& tensor,
    int64_t dim) {
  ET_CHECK_MSG(
      dim >= 0 && dim <= tensor.dim(),
      "Ending dimension %" PRId64
      " should be in the range [0, tensor.dim() %zd].",
      dim,
      ssize_t(tensor.dim()));
  size_t dims = 1;
  for (size_t i = 0; i < dim; ++i) {
    dims *= static_cast<size_t>(tensor.size(i));
  }
  return dims;
}

/// Returns the product of dim[dim+1:].
inline size_t getTrailingDims(
    const executorch::aten::Tensor& tensor,
    int64_t dim) {
  ET_CHECK_MSG(
      dim >= -1 && dim < tensor.dim(),
      "Starting dimension %" PRId64
      " should be in the range [-1, tensor.dim() -1 %zd).",
      dim,
      ssize_t(tensor.dim()));
  size_t dims = 1;
  for (size_t i = dim + 1; i < tensor.dim(); ++i) {
    dims *= static_cast<size_t>(tensor.size(i));
  }
  return dims;
}

/**
 * Given a N-dimensional tensor coordinate, return a linear index that can be
 * used to access the corresponding element in the tensor's data buffer.
 *
 * @param[in] tensor The tensor that will be indexed
 * @param[in] coordinate A n-dimensional array representing the coordinate to
 * index. It is assumed that the array has kTensorDimensionLimit elements.
 * @param[out] index The linear index to element at the specified coordinate
 * in the tensor.
 */
inline size_t coordinateToIndex(
    const executorch::aten::Tensor& tensor,
    const size_t* const coordinate) {
  size_t index = 0;
  for (int d = 0; d < tensor.dim(); ++d) {
    index += coordinate[d] * getTrailingDims(tensor, d);
  }
  return index;
}

/**
 * Produce a memoized array for use with repeated calls to
 * coordinateToIndexWithTrailingDimsMemo, which will be faster than
 * repeated calls to coordinateToIndex.
 */
inline void memoizeTrailingDims(
    const executorch::aten::Tensor& tensor,
    size_t trailing_dims_memo[kTensorDimensionLimit]) {
  const auto tensorDim = tensor.dim();
  size_t dims = 1;
  for (int ii = tensorDim - 1; ii >= 0; --ii) {
    trailing_dims_memo[ii] = dims;
    dims *= static_cast<size_t>(tensor.size(ii));
  }
}

/**
 * Like coordinateToIndex, but faster for repeated calls with the same
 * tensor. trailing_dims_memo must be produced by a call to
 * memoizeTrailingDims.
 */
inline size_t coordinateToIndexWithTrailingDimsMemo(
    const executorch::aten::Tensor& tensor,
    const size_t* const coordinate,
    const size_t trailing_dims_memo[kTensorDimensionLimit]) {
  size_t index = 0;
  for (int d = 0; d < tensor.dim(); ++d) {
    index += coordinate[d] * trailing_dims_memo[d];
  }
  return index;
}

/**
 * Given the linear index return the N-dimensional tensor coordinate. This is
 * the inverse operation of coordinateToIndex.
 *
 * @param[in] tensor The tensor that will be indexed
 * @param[in] index The linear index to element at the specified coordinate in
 * the tensor.
 * @param[out] coordinate A n-dimensional array representing the coordinate to
 * index. It is assumed that the array has kTensorDimensionLimit elements.
 * @returns void
 */
inline void indexToCoordinate(
    const executorch::aten::Tensor& tensor,
    size_t index,
    size_t* coordinate) {
  ET_CHECK(index < tensor.numel());
  for (auto i = 0; i < tensor.dim(); ++i) {
    auto dim = tensor.dim() - 1 - i;
    size_t dim_size = tensor.size(dim);
    coordinate[dim] = index % dim_size;
    index /= dim_size;
  }
}

/**
 * Extracts an integer value from a scalar Tensor.
 *
 * @param[in] tensor The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that
 * value. `false` if a value could not be extracted: either it was not an
 * integer Scalar Tensor, or the value of that Scalar Tensor could not be
 * represented by INT_T.
 */
template <
    typename INT_T,
    typename std::enable_if<
        std::is_integral<INT_T>::value && !std::is_same<INT_T, bool>::value,
        bool>::type = true>
bool extract_scalar_tensor(executorch::aten::Tensor tensor, INT_T* out_val) {
  if (tensor.numel() != 1) {
    return false;
  }
#define CASE_INT_DTYPE(TENSOR_CTYPE, TENSOR_DTYPE)                     \
  case executorch::aten::ScalarType::TENSOR_DTYPE: {                   \
    const TENSOR_CTYPE val = tensor.const_data_ptr<TENSOR_CTYPE>()[0]; \
    if (val < std::numeric_limits<INT_T>::lowest() ||                  \
        val > std::numeric_limits<INT_T>::max()) {                     \
      return false;                                                    \
    }                                                                  \
    *out_val = static_cast<INT_T>(val);                                \
    return true;                                                       \
  }

  switch (tensor.scalar_type()) {
    ET_FORALL_INT_TYPES(CASE_INT_DTYPE);
    default:
      return false;
  }
#undef CASE_INT_DTYPE
}

/**
 * Extracts a floating point value from a scalar Tensor.
 *
 * @param[in] tensor The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that
 * value. `false` if a value could not be extracted: either it was not a
 * floating point Scalar Tensor, or the value of that Scalar Tensor could not
 * be represented by FLOAT_T.
 */
template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
bool extract_scalar_tensor(executorch::aten::Tensor tensor, FLOAT_T* out_val) {
  if (tensor.numel() != 1) {
    return false;
  }
#define CASE_REAL_DTYPE(TENSOR_CTYPE, TENSOR_DTYPE)                    \
  case executorch::aten::ScalarType::TENSOR_DTYPE: {                   \
    /* ET_FORALL_REAL_TYPES guarantees TENSOR_CTYPE is a real type. */ \
    double val =                                                       \
        static_cast<double>(tensor.const_data_ptr<TENSOR_CTYPE>()[0]); \
    if (std::isfinite(val) &&                                          \
        (val < std::numeric_limits<FLOAT_T>::lowest() ||               \
         val > std::numeric_limits<FLOAT_T>::max())) {                 \
      return false;                                                    \
    }                                                                  \
    *out_val = static_cast<FLOAT_T>(val);                              \
    return true;                                                       \
  }

  switch (tensor.scalar_type()) {
    ET_FORALL_REAL_TYPES(CASE_REAL_DTYPE);
    default:
      return false;
  }
#undef CASE_REAL_DTYPE
}

/**
 * Extracts a boolean value from a Scalar.
 *
 * @param[in] scalar The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that
 * value. `false` if a value could not be extracted, i.e. not a boolean
 */
template <
    typename BOOL_T,
    typename std::enable_if<std::is_same<BOOL_T, bool>::value, bool>::type =
        true>
bool extract_scalar_tensor(executorch::aten::Tensor tensor, BOOL_T* out_val) {
  if (tensor.scalar_type() != executorch::aten::ScalarType::Bool) {
    return false;
  }
  if (tensor.numel() != 1) {
    return false;
  }

  bool val = tensor.const_data_ptr<bool>()[0];

  *out_val = static_cast<BOOL_T>(val);

  return true;
}

/// These APIs should not be used outside of Executor.cpp.
namespace internal {
/**
 * Share t_src's data_ptr with t_dst.
 */
ET_NODISCARD Error share_tensor_data(
    const executorch::aten::Tensor& t_dst,
    const executorch::aten::Tensor& t_src);

/**
 * Copy t_src's data_ptr to t_dst.
 */
ET_NODISCARD Error copy_tensor_data(
    const executorch::aten::Tensor& t_dst,
    const executorch::aten::Tensor& t_src);

/**
 * Set the data_ptr of t to buffer.
 */
ET_NODISCARD Error set_tensor_data(
    const executorch::aten::Tensor& t,
    void* buffer,
    size_t buffer_size);

/**
 * Reset tensor's data_ptr, clear all the storage for at::Tensor.
 */
void reset_data_ptr(const executorch::aten::Tensor& tensor);

/**
 * Resize tensor impl
 */
ET_NODISCARD Error resize_tensor_impl(
    executorch::aten::TensorImpl* impl,
    executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes);

} // namespace internal

/**
 * Resize a tensor to new_sizes, rank must stay the same. Currently does not
 * expand the tensor if new size exceeds the current capacity. Currently
 * fails an ET_CHECK if the tensor cannot be resized.
 *
 * WARNING: Placeholder API until discussion around runtime context is
 * settled, will likely move to be a class method on a TensorResizer object
 * passed in through runtimeContext.
 */
ET_NODISCARD inline Error resize_tensor(
    executorch::aten::Tensor t,
    executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes) {
  return internal::resize_tensor_impl(t.unsafeGetTensorImpl(), new_sizes);
}

/**
 * Resize a tensor to new_sizes, rank must stay the same. Currently does not
 * expand the tensor if new size exceeds the current capacity. Currently
 * fails an ET_CHECK if the tensor cannot be resized.
 *
 * WARNING: Placeholder API until discussion around runtime context is
 * settled, will likely move to be a class method on a TensorResizer object
 * passed in through runtimeContext.
 */
template <
    typename T,
    typename std::enable_if<
        !std::is_same<executorch::aten::SizesType, T>::value,
        int>::type = 0>
ET_NODISCARD inline Error resize_tensor(
    executorch::aten::Tensor t,
    executorch::aten::ArrayRef<T> new_sizes) {
  // Need to cast the input array to an array of Tensor::SizesType
  std::array<executorch::aten::SizesType, kTensorDimensionLimit>
      new_sizes_casted{};
  size_t new_sizes_ndim = new_sizes.size();
  for (size_t i = 0; i < new_sizes_ndim; ++i) {
    new_sizes_casted[i] =
        static_cast<executorch::aten::SizesType>(new_sizes[i]);
  }

  return internal::resize_tensor_impl(
      t.unsafeGetTensorImpl(), {new_sizes_casted.data(), new_sizes_ndim});
}

/// DEPRECATED: Use `resize_tensor()` instead, which can fail non-fatally.
ET_DEPRECATED inline void resize(
    executorch::aten::Tensor t,
    executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes) {
  Error err = resize_tensor(t, new_sizes);
  ET_CHECK_MSG(
      err == Error::Ok, "Could not resize Tensor; see logs for details");
}
/**
 * Get dim_order of a Tensor and write it to out_dim_order.
 * @param tensor The tensor where we want to get dim order from.
 * @param out_dim_order Pointing to an array of DimOrderType where we write
 * dim order into it.
 * @param out_dim_order_size Size of the DimOrderType array.
 */
ET_NODISCARD Error get_dim_order(
    const executorch::aten::Tensor& tensor,
    executorch::aten::DimOrderType* out_dim_order,
    size_t out_dim_order_size);

/**
 * Checks whether a tensor has a valid dim order. If the dim order could not
 * be determined, then this function returns false by default.
 */
bool tensor_has_valid_dim_order(executorch::aten::Tensor t);

/**
 * Checks whether a tensor has either the default of channels last dim order.
 * If the dim order could not be determined, then this function returns false
 * by default.
 */
bool tensor_is_default_or_channels_last_dim_order(executorch::aten::Tensor t);

/**
 * Checks whether a tensor has the default dimension order.
 * Logs an error message if the tensor does not meet the expected criteria.
 *
 * @param t The tensor to check the dimension order of.
 * @return True if the tensor has the default dimension order, false otherwise.
 */
bool tensor_is_default_dim_order(executorch::aten::Tensor t);

/**
 * Checks whether a tensor has the channels last dimension order.
 * Logs an error message if the tensor does not meet the expected criteria.
 *
 * @param t The tensor to check the dimension order of.
 * @return True if the tensor has the channels last dimension order, false
 * otherwise.
 */
bool tensor_is_channels_last_dim_order(executorch::aten::Tensor t);

/**
 * Asserts that four tensors have the same dim_order
 *
 * Note that this macro only tests dim order, but not others like actual data,
 * sizes, etc.
 *
 */
bool tensors_have_same_dim_order(
    const executorch::aten::ArrayRef<executorch::aten::Tensor> tensor_list);

/**
 * Asserts that two tensors have the same dim_order
 *
 * Note that this macro only tests dim order, but not others like actual data,
 * sizes, etc.
 */

inline bool tensors_have_same_dim_order(
    const executorch::aten::Tensor& a,
    const executorch::aten::Tensor& b) {
  executorch::aten::Tensor tensor_list[2] = {a, b};
  return tensors_have_same_dim_order(tensor_list);
}

/**
 * Asserts that three tensors have the same dim_order
 *
 * Note that this macro only tests dim order, but not others like actual data,
 * sizes, etc.
 *
 */

inline bool tensors_have_same_dim_order(
    const executorch::aten::Tensor& a,
    const executorch::aten::Tensor& b,
    const executorch::aten::Tensor& c) {
  executorch::aten::Tensor tensor_list[3] = {a, b, c};
  return tensors_have_same_dim_order(tensor_list);
}

/**
 * Asserts that four tensors have the same dim_order
 *
 * Note that this macro only tests dim order, but not others like actual data,
 * sizes, etc.
 *
 */

inline bool tensors_have_same_dim_order(
    const executorch::aten::Tensor& a,
    const executorch::aten::Tensor& b,
    const executorch::aten::Tensor& c,
    const executorch::aten::Tensor& d) {
  executorch::aten::Tensor tensor_list[4] = {a, b, c, d};
  return tensors_have_same_dim_order(tensor_list);
}

/**
 * Given an n-dimensional coordinate array and an array of tensor strides,
 * calculates the linear index that can be used to retrieve the value at the
 * given coordinates.
 * @param coordinate Pointer to the array of coordinates.
 * @param strides Pointer to the array of strides.
 * @param ndim Number of dimensions in the tensor.
 */
inline size_t calculate_linear_index(
    const executorch::aten::SizesType* coordinate,
    const executorch::aten::StridesType* strides,
    const size_t ndim) {
  size_t index = 0;
  for (size_t i = 0; i < ndim; i++) {
    index += coordinate[i] * strides[i];
  }
  return index;
}

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::calculate_linear_index;
using ::executorch::runtime::coordinateToIndex;
using ::executorch::runtime::dim_is_valid;
using ::executorch::runtime::extract_scalar_tensor;
using ::executorch::runtime::get_dim_order;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::getTrailingDims;
using ::executorch::runtime::indexToCoordinate;
using ::executorch::runtime::kTensorDimensionLimit;
using ::executorch::runtime::nonempty_size;
using ::executorch::runtime::nonzero_dim;
using ::executorch::runtime::resize;
using ::executorch::runtime::resize_tensor;
using ::executorch::runtime::tensor_can_cast_to;
using ::executorch::runtime::tensor_dim_has_index;
using ::executorch::runtime::tensor_has_dim;
using ::executorch::runtime::tensor_has_expected_size;
using ::executorch::runtime::tensor_has_non_empty_dim;
using ::executorch::runtime::tensor_has_rank_greater_or_equal_to;
using ::executorch::runtime::tensor_has_rank_smaller_or_equal_to;
using ::executorch::runtime::tensor_has_valid_dim_order;
using ::executorch::runtime::tensor_is_bits_type;
using ::executorch::runtime::tensor_is_bool_type;
using ::executorch::runtime::tensor_is_complex_type;
using ::executorch::runtime::tensor_is_contiguous;
using ::executorch::runtime::tensor_is_default_dim_order;
using ::executorch::runtime::tensor_is_default_or_channels_last_dim_order;
using ::executorch::runtime::tensor_is_floating_type;
using ::executorch::runtime::tensor_is_integral_type;
using ::executorch::runtime::tensor_is_rank;
using ::executorch::runtime::tensor_is_real_type;
using ::executorch::runtime::tensor_is_realh_type;
using ::executorch::runtime::tensor_is_realhb_type;
using ::executorch::runtime::tensor_is_scalar;
using ::executorch::runtime::tensors_have_same_dim_order;
using ::executorch::runtime::tensors_have_same_dtype;
using ::executorch::runtime::tensors_have_same_rank;
using ::executorch::runtime::tensors_have_same_shape;
using ::executorch::runtime::tensors_have_same_shape_and_dtype;
using ::executorch::runtime::tensors_have_same_size_at_dims;
using ::executorch::runtime::tensors_have_same_strides;
namespace internal {
using ::executorch::runtime::internal::copy_tensor_data;
using ::executorch::runtime::internal::reset_data_ptr;
using ::executorch::runtime::internal::resize_tensor_impl;
using ::executorch::runtime::internal::set_tensor_data;
using ::executorch::runtime::internal::share_tensor_data;
} // namespace internal
} // namespace executor
} // namespace torch
