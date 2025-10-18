/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/times.h>
#include <xtensa/sim.h>

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

namespace impl {
namespace G3 {
namespace native {
namespace {

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class FusionG3OperatorTest : public OperatorTest {
 public:
 protected:
  Tensor&
  add_out(const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out) {
    return impl::G3::native::add_out(context_, a, b, alpha, out);
  }
};

TEST_F(FusionG3OperatorTest, TwoDimFloatTensorAddTest) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{2, 2};
  Tensor out = tf.zeros(sizes);

  // Add two 2x2 tensors.
  add_out(tf.make(sizes, {1, 2, 3, 4}), tf.make(sizes, {2, 2, 2, 2}), 1, out);

  EXPECT_TENSOR_EQ(out, tf.make(sizes, {3, 4, 5, 6}));
}

TEST_F(FusionG3OperatorTest, TensorScalarAddTest) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{2, 2};
  Tensor out = tf.zeros(sizes);

  // Add 2x2 tensor with scalar.
  add_out(tf.make(sizes, {1, 2, 3, 4}), tf.make({1}, {2}), 1, out);

  EXPECT_TENSOR_EQ(out, tf.make(sizes, {3, 4, 5, 6}));
}

TEST_F(FusionG3OperatorTest, AddWithBroadcastTest) {
  TensorFactory<ScalarType::Float> tf;
  // Broadcast add.
  const std::vector<int> size_a{1, 3, 2, 4}, size_b{2, 4};
  Tensor out = tf.zeros(size_a);

  add_out(tf.ones(size_a), tf.ones(size_b), 1, out);

  EXPECT_TENSOR_EQ(out, tf.full(size_a, 2));
}

TEST_F(FusionG3OperatorTest, KernelCheckTest) {
  TensorFactory<ScalarType::Float> tf;
  // Broadcast add.
  const std::vector<TensorImpl::SizesType> sizeOfA{1, 3, 2, 4}, sizeOfB{2, 4};
  const Tensor b = tf.ones(sizeOfB);
  Tensor out = tf.zeros(sizeOfA);
  // Create a null tensor to force kernel check failure.
  TensorImpl nullTensorImpl(
      b.scalar_type(),
      b.dim(),
      const_cast<TensorImpl::SizesType*>(b.sizes().data()),
      // Use nullptr to force kernel check failure.
      /*data=*/nullptr,
      const_cast<TensorImpl::DimOrderType*>(b.dim_order().data()));
  Tensor nullTensor(&nullTensorImpl);

  ET_EXPECT_KERNEL_FAILURE(
      context_, add_out(tf.ones(sizeOfA), nullTensor, 1, out));
}

} // namespace
} // namespace native
} // namespace G3
} // namespace impl
