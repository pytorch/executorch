/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/times.h>
#include <xtensa/sim.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/backends/cadence/hifi/operators/operators.h>

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {
namespace {

using ::executorch::aten::ArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class HiFiCatTest : public OperatorTest {
 public:
 protected:
  Tensor& cat_out(ArrayRef<Tensor> tensors, int64_t dim, Tensor& out) {
    return ::cadence::impl::HiFi::native::cat_out(context_, tensors, dim, out);
  }
};

TEST_F(HiFiCatTest, FloatCatDim0Test) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor b = tf.make({1, 3}, {7.0, 8.0, 9.0});
  Tensor c = tf.make({2, 3}, {10.0, 11.0, 12.0, 13.0, 14.0, 15.0});

  Tensor expected = tf.make(
      {5, 3},
      {1.0,
       2.0,
       3.0,
       4.0,
       5.0,
       6.0,
       7.0,
       8.0,
       9.0,
       10.0,
       11.0,
       12.0,
       13.0,
       14.0,
       15.0});

  Tensor out = tf.zeros({5, 3});
  std::vector<Tensor> tensors = {a, b, c};

  cat_out(ArrayRef<Tensor>(tensors.data(), tensors.size()), 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiCatTest, FloatCatDim1Test) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  Tensor b = tf.make({2, 1}, {5.0, 6.0});
  Tensor c = tf.make({2, 3}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  Tensor expected = tf.make(
      {2, 6}, {1.0, 2.0, 5.0, 7.0, 8.0, 9.0, 3.0, 4.0, 6.0, 10.0, 11.0, 12.0});

  Tensor out = tf.zeros({2, 6});
  std::vector<Tensor> tensors = {a, b, c};

  cat_out(ArrayRef<Tensor>(tensors.data(), tensors.size()), 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiCatTest, IntCatDim0Test) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = tf.make({1, 3}, {7, 8, 9});

  Tensor expected = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  Tensor out = tf.zeros({3, 3});
  std::vector<Tensor> tensors = {a, b};
  cat_out(ArrayRef<Tensor>(tensors.data(), tensors.size()), 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiCatTest, SingleTensorTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor expected = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  Tensor out = tf.zeros({2, 3});
  std::vector<Tensor> tensors = {a};
  cat_out(ArrayRef<Tensor>(tensors.data(), tensors.size()), 0, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiCatTest, ThreeDimensionalCatTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  Tensor b = tf.make({2, 2, 1}, {9.0, 10.0, 11.0, 12.0});

  Tensor expected = tf.make(
      {2, 2, 3},
      {1.0, 2.0, 9.0, 3.0, 4.0, 10.0, 5.0, 6.0, 11.0, 7.0, 8.0, 12.0});

  Tensor out = tf.zeros({2, 2, 3});
  std::vector<Tensor> tensors = {a, b};

  cat_out(ArrayRef<Tensor>(tensors.data(), tensors.size()), 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
