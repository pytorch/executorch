/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/generic/operators/op_where_scalar.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's.
class GenericWhereScalarTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Bool> tf_bool_;
  TensorFactory<ScalarType::Float> tf_;
};

TEST_F(GenericWhereScalarTest, SelectsPerElement) {
  const Tensor condition =
      tf_bool_.make({2, 3}, {true, false, true, false, false, true});
  Tensor out = tf_.zeros({2, 3});

  where_Scalar_out(context_, condition, 1.5, -2.25, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 3}, {1.5, -2.25, 1.5, -2.25, -2.25, 1.5}));
}

// The scalars arrive as double and the output is float.
TEST_F(GenericWhereScalarTest, NarrowsTheScalarsToFloat) {
  const Tensor condition = tf_bool_.make({2}, {true, false});
  Tensor out = tf_.zeros({2});

  where_Scalar_out(context_, condition, 0.1, 1e-50, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2}, {0.1f, 0.0f}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
