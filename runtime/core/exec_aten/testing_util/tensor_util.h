/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <gmock/gmock.h> // For MATCHER_P

namespace executorch {
namespace runtime {
namespace testing {

namespace internal {
constexpr double kDefaultRtol = 1e-5;
constexpr double kDefaultAtol = 1e-8;
} // namespace internal

/**
 *  Returns true if the tensors are of the same shape and dtype, and if all
 * elements are close to each other.
 *
 * TODO(T132992348): This function will currently fail an ET_CHECK if the
 * strides of the tensors are not identical. Add support for comparing
 * tensors with different strides.
 *
 * Note that gtest users can write `EXPECT_THAT(tensor1, IsCloseTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsCloseTo(tensor2)))`, or use the helper macros
 * `EXPECT_TENSOR_CLOSE()` and `EXPECT_TENSOR_NOT_CLOSE()`.
 *
 * For exact equality, use `EXPECT_THAT(tensor1, IsEqualTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsEqualTo(tensor2)))`, or the helper macros
 * `EXPECT_TENSOR_EQ()` and `EXPECT_TENSOR_NE()`.
 *
 * An element A is close to B when one is true:
 *
 * (1) A is equal to B.
 * (2) A and B are both NaN, are both -infinity, or are both +infinity.
 * (3) The error abs(A - B) is finite and less than the max error
 *     (atol + abs(rtol * B)).
 *
 * If both rtol/atol are zero, this function checks for exact equality.
 *
 * NOTE: rtol/atol are ignored and treated as zero for non-floating-point
 * dtypes.
 *
 * @param[in] a The first tensor to compare.
 * @param[in] b The second tensor to compare.
 * @param[in] rtol Relative tolerance; see note above.
 * @param[in] atol Absolute tolerance; see note above.
 * @retval true All corresponding elements of the two tensors are within the
 *     specified tolerance of each other.
 * @retval false One or more corresponding elements of the two tensors are
 *     outside of the specified tolerance of each other.
 */
bool tensors_are_close(
    const exec_aten::Tensor& a,
    const exec_aten::Tensor& b,
    double rtol = internal::kDefaultRtol,
    double atol = internal::kDefaultAtol);

/**
 * Returns true if the tensors are of the same numel and dtype, and if all
 * elements are close to each other. The tensor shapes do not need to be same.
 *
 * Note that gtest users can write `EXPECT_THAT(tensor1,
 * IsDataCloseTo(tensor2))` or `EXPECT_THAT(tensor1,
 * Not(IsDataCloseTo(tensor2)))`, or use the helper macros
 * `EXPECT_TENSOR_DATA_CLOSE()` and `EXPECT_TENSOR_DATA_NOT_CLOSE()`.
 *
 * For exact equality, use `EXPECT_THAT(tensor1, IsDataEqualTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsDataEqualTo(tensor2)))`, or the helper macros
 * `EXPECT_TENSOR_DATA_EQ()` and `EXPECT_TENSOR_DATA_NE()`.
 *
 * The defination of an element A is close to B is in the comment of the
 * function `tensors_are_close`
 *
 * @param[in] a The first tensor to compare.
 * @param[in] b The second tensor to compare.
 * @param[in] rtol Relative tolerance; see note above.
 * @param[in] atol Absolute tolerance; see note above.
 * @retval true All corresponding elements of the two tensors are within the
 *     specified tolerance of each other.
 * @retval false One or more corresponding elements of the two tensors are
 *     outside of the specified tolerance of each other.
 */
bool tensor_data_is_close(
    const exec_aten::Tensor& a,
    const exec_aten::Tensor& b,
    double rtol = internal::kDefaultRtol,
    double atol = internal::kDefaultAtol);

/**
 * Returns true if the two lists are of the same length, and
 * tensor_data_is_close(tensors_a[i], tensors_b[i], rtol, atol) is true for all
 * i.
 */
bool tensor_lists_are_close(
    const exec_aten::Tensor* tensors_a,
    size_t num_tensors_a,
    const exec_aten::Tensor* tensors_b,
    size_t num_tensors_b,
    double rtol = internal::kDefaultRtol,
    double atol = internal::kDefaultAtol);

/**
 * Lets gtest users write `EXPECT_THAT(tensor1, IsCloseTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsCloseTo(tensor2)))`.
 *
 * See also `EXPECT_TENSOR_CLOSE()` and `EXPECT_TENSOR_NOT_CLOSE()`.
 */
MATCHER_P(IsCloseTo, other, "") {
  return tensors_are_close(arg, other);
}

/**
 * Lets gtest users write
 * `EXPECT_THAT(tensor1, IsCloseToWithTol(tensor2, rtol, atol))`
 * or `EXPECT_THAT(tensor1, Not(IsCloseToWithTol(tensor2, rtol, atol)))`.
 *
 * See also `EXPECT_TENSOR_CLOSE_WITH_TOL()` and
 * `EXPECT_TENSOR_NOT_CLOSE_WITH_TOL()`.
 */
MATCHER_P3(IsCloseToWithTol, other, rtol, atol, "") {
  return tensors_are_close(arg, other, rtol, atol);
}

/**
 * Lets gtest users write `EXPECT_THAT(tensor1, IsEqualTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsEqualTo(tensor2)))`.
 *
 * See also `EXPECT_TENSOR_EQ()` and `EXPECT_TENSOR_NE()`.
 */
MATCHER_P(IsEqualTo, other, "") {
  return tensors_are_close(arg, other, /*rtol=*/0, /*atol=*/0);
}

/**
 * Lets gtest users write `EXPECT_THAT(tensor1, IsDataCloseTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsDataCloseTo(tensor2)))`.
 *
 * See also `EXPECT_TENSOR_DATA_CLOSE()` and `EXPECT_TENSOR_DATA_NOT_CLOSE()`.
 */
MATCHER_P(IsDataCloseTo, other, "") {
  return tensor_data_is_close(arg, other);
}

/**
 * Lets gtest users write
 * `EXPECT_THAT(tensor1, IsDataCloseToWithTol(tensor2, rtol, atol))`
 * or `EXPECT_THAT(tensor1, Not(IsDataCloseToWithTol(tensor2, rtol, atol)))`.
 *
 * See also `EXPECT_TENSOR_CLOSE_WITH_TOL()` and
 * `EXPECT_TENSOR_NOT_CLOSE_WITH_TOL()`.
 */
MATCHER_P3(IsDataCloseToWithTol, other, rtol, atol, "") {
  return tensor_data_is_close(arg, other, rtol, atol);
}

/**
 * Lets gtest users write `EXPECT_THAT(tensor1, IsDataEqualTo(tensor2))` or
 * `EXPECT_THAT(tensor1, Not(IsDataEqualTo(tensor2)))`.
 *
 * See also `EXPECT_TENSOR_DATA_EQ()` and `EXPECT_TENSOR_DATA_NE()`.
 */
MATCHER_P(IsDataEqualTo, other, "") {
  return tensor_data_is_close(arg, other, /*rtol=*/0, /*atol=*/0);
}

/**
 * Lets gtest users write `EXPECT_THAT(tensor_list1,
 * IsListCloseTo(tensor_list2))` or `EXPECT_THAT(tensor_list1,
 * Not(IsListCloseTo(tensor_list2)))`.
 *
 * The lists can be any container of Tensor that supports ::data() and ::size().
 *
 * See also `EXPECT_TENSOR_LISTS_CLOSE()` and `EXPECT_TENSOR_LISTS_NOT_CLOSE()`.
 */
MATCHER_P(IsListCloseTo, other, "") {
  return tensor_lists_are_close(
      arg.data(), arg.size(), other.data(), other.size());
}

/**
 * Lets gtest users write `EXPECT_THAT(tensor_list1,
 * IsListEqualTo(tensor_list2))` or `EXPECT_THAT(tensor_list1,
 * Not(IsListEqualTo(tensor_list2)))`.
 *
 * The lists can be any container of Tensor that supports ::data() and ::size().
 *
 * See also `EXPECT_TENSOR_LISTS_EQ()` and `EXPECT_TENSOR_LISTS_NE()`.
 */
MATCHER_P(IsListEqualTo, other, "") {
  return tensor_lists_are_close(
      arg.data(),
      arg.size(),
      other.data(),
      other.size(),
      /*rtol=*/0,
      /*atol=*/0);
}

/*
 * NOTE: Although it would be nice to make `EXPECT_EQ(t1, t2)` and friends work,
 * that would require implementing `bool operator==(Tensor, Tensor)`.
 *
 * at::Tensor implements `Tensor operator==(Tensor, Tensor)`, returning an
 * element-by-element comparison. This causes an ambiguous conflict with the
 * `bool`-returning operator.
 */
#define EXPECT_TENSOR_EQ(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsEqualTo(t2))
#define EXPECT_TENSOR_NE(t1, t2) \
  EXPECT_THAT((t1), ::testing::Not(executorch::runtime::testing::IsEqualTo(t2)))
#define ASSERT_TENSOR_EQ(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsEqualTo(t2))
#define ASSERT_TENSOR_NE(t1, t2) \
  ASSERT_THAT((t1), ::testing::Not(executorch::runtime::testing::IsEqualTo(t2)))

#define EXPECT_TENSOR_CLOSE(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsCloseTo(t2))
#define EXPECT_TENSOR_NOT_CLOSE(t1, t2) \
  EXPECT_THAT((t1), ::testing::Not(executorch::runtime::testing::IsCloseTo(t2)))
#define ASSERT_TENSOR_CLOSE(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsCloseTo(t2))
#define ASSERT_TENSOR_NOT_CLOSE(t1, t2) \
  ASSERT_THAT((t1), ::testing::Not(executorch::runtime::testing::IsCloseTo(t2)))

#define EXPECT_TENSOR_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  EXPECT_THAT(                                           \
      (t1), ::executorch::runtime::testing::IsCloseToWithTol(t2, rtol, atol))
#define EXPECT_TENSOR_NOT_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  EXPECT_THAT(                                               \
      (t1),                                                  \
      ::testing::Not(                                        \
          executorch::runtime::testing::IsCloseToWithTol(t2, rtol, atol)))
#define ASSERT_TENSOR_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  ASSERT_THAT(                                           \
      (t1), ::executorch::runtime::testing::IsCloseToWithTol(t2, rtol, atol))
#define ASSERT_TENSOR_NOT_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  ASSERT_THAT(                                               \
      (t1),                                                  \
      ::testing::Not(                                        \
          executorch::runtime::testing::IsCloseToWithTol(t2, rtol, atol)))

#define EXPECT_TENSOR_DATA_EQ(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsDataEqualTo(t2))
#define EXPECT_TENSOR_DATA_NE(t1, t2) \
  EXPECT_THAT(                        \
      (t1), ::testing::Not(executorch::runtime::testing::IsDataEqualTo(t2)))
#define ASSERT_TENSOR_DATA_EQ(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsDataEqualTo(t2))
#define ASSERT_TENSOR_DATA_NE(t1, t2) \
  ASSERT_THAT(                        \
      (t1), ::testing::Not(executorch::runtime::testing::IsDataEqualTo(t2)))

#define EXPECT_TENSOR_DATA_CLOSE(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsDataCloseTo(t2))
#define EXPECT_TENSOR_DATA_NOT_CLOSE(t1, t2) \
  EXPECT_THAT(                               \
      (t1), ::testing::Not(executorch::runtime::testing::IsDataCloseTo(t2)))
#define ASSERT_TENSOR_DATA_CLOSE(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsDataCloseTo(t2))
#define ASSERT_TENSOR_DATA_NOT_CLOSE(t1, t2) \
  ASSERT_THAT(                               \
      (t1), ::testing::Not(executorch::runtime::testing::IsDataCloseTo(t2)))

#define EXPECT_TENSOR_DATA_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  EXPECT_THAT(                                                \
      (t1),                                                   \
      ::executorch::runtime::testing::IsDataCloseToWithTol(t2, rtol, atol))
#define EXPECT_TENSOR_DATA_NOT_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  EXPECT_THAT(                                                    \
      (t1),                                                       \
      ::testing::Not(                                             \
          executorch::runtime::testing::IsDataCloseToWithTol(t2, rtol, atol)))
#define ASSERT_TENSOR_DATA_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  ASSERT_THAT(                                                \
      (t1),                                                   \
      ::executorch::runtime::testing::IsDataCloseToWithTol(t2, rtol, atol))
#define ASSERT_TENSOR_DATA_NOT_CLOSE_WITH_TOL(t1, t2, rtol, atol) \
  ASSERT_THAT(                                                    \
      (t1),                                                       \
      ::testing::Not(                                             \
          executorch::runtime::testing::IsDataCloseToWithTol(t2, rtol, atol)))

/*
 * Helpers for comparing lists of Tensors.
 */

#define EXPECT_TENSOR_LISTS_EQ(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsListEqualTo(t2))
#define EXPECT_TENSOR_LISTS_NE(t1, t2) \
  EXPECT_THAT(                         \
      (t1), ::testing::Not(executorch::runtime::testing::IsListEqualTo(t2)))
#define ASSERT_TENSOR_LISTS_EQ(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsListEqualTo(t2))
#define ASSERT_TENSOR_LISTS_NE(t1, t2) \
  ASSERT_THAT(                         \
      (t1), ::testing::Not(executorch::runtime::testing::IsListEqualTo(t2)))

#define EXPECT_TENSOR_LISTS_CLOSE(t1, t2) \
  EXPECT_THAT((t1), ::executorch::runtime::testing::IsListCloseTo(t2))
#define EXPECT_TENSOR_LISTS_NOT_CLOSE(t1, t2) \
  EXPECT_THAT(                                \
      (t1), ::testing::Not(executorch::runtime::testing::IsListCloseTo(t2)))
#define ASSERT_TENSOR_LISTS_CLOSE(t1, t2) \
  ASSERT_THAT((t1), ::executorch::runtime::testing::IsListCloseTo(t2))
#define ASSERT_TENSOR_LISTS_NOT_CLOSE(t1, t2) \
  ASSERT_THAT(                                \
      (t1), ::testing::Not(executorch::runtime::testing::IsListCloseTo(t2)))

} // namespace testing
} // namespace runtime
} // namespace executorch

// ATen already defines operator<<() for Tensor and ScalarType.
#ifndef USE_ATEN_LIB

/*
 * These functions must be declared in the original namespaces of their
 * associated types so that C++ can find them.
 */
namespace torch {
namespace executor {

/**
 * Prints the ScalarType to the stream as a human-readable string.
 *
 * See also executorch::runtime::toString(ScalarType t) in ScalarTypeUtil.h.
 */
std::ostream& operator<<(std::ostream& os, const exec_aten::ScalarType& t);

/**
 * Prints the Tensor to the stream as a human-readable string.
 */
std::ostream& operator<<(std::ostream& os, const exec_aten::Tensor& t);

} // namespace executor
} // namespace torch

#endif // !USE_ATEN_LIB

namespace torch {
namespace executor {
namespace testing {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::testing::IsCloseTo;
using ::executorch::runtime::testing::IsCloseToWithTol;
using ::executorch::runtime::testing::IsDataCloseTo;
using ::executorch::runtime::testing::IsDataCloseToWithTol;
using ::executorch::runtime::testing::IsDataEqualTo;
using ::executorch::runtime::testing::IsEqualTo;
using ::executorch::runtime::testing::IsListCloseTo;
using ::executorch::runtime::testing::IsListEqualTo;
using ::executorch::runtime::testing::tensor_data_is_close;
using ::executorch::runtime::testing::tensor_lists_are_close;
using ::executorch::runtime::testing::tensors_are_close;
} // namespace testing
} // namespace executor
} // namespace torch
