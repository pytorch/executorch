/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpRandTest : public OperatorTest {
 protected:
  void op_rand_out(const IntArrayRef sizes, Tensor& out) {
    torch::executor::aten::rand_outf(context_, sizes, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_rand(std::vector<int64_t>& sizes) {
    TensorFactory<DTYPE> tf;

    // Tensor factory wants int32 scales, op kernel wants int64.
    std::vector<int32_t> sizes_i32;
    std::transform(
        sizes.begin(),
        sizes.end(),
        std::back_inserter(sizes_i32),
        [](int64_t s) { return static_cast<int32_t>(s); });
    Tensor out = tf.zeros(sizes_i32);

    IntArrayRef sizes_ref(sizes.data(), sizes.size());
    op_rand_out(sizes_ref, out);

    // Check mean and standard deviation. To avoid flaky CI, test pretty
    // loosely.
    auto out_data = out.const_data_ptr<CTYPE>();
    double mean =
        std::accumulate(
            out_data,
            out_data + out.numel(),
            0.0,
            [](double acc, CTYPE n) { return acc + static_cast<double>(n); }) /
        out.numel();
    double var = std::accumulate(
                     out_data,
                     out_data + out.numel(),
                     0.0,
                     [=](double acc, CTYPE n) {
                       return acc + std::pow(static_cast<double>(n) - mean, 2);
                     }) /
        out.numel();
    auto stdev = std::sqrt(var);

    // These are very rough thresholds. A better test implementation would
    // probably do a proper statistical test to compare the generated empirical
    // data to the reference distribution, but this should do.

    // Expected mean is 0.5
    EXPECT_NEAR(mean, 0.5, 5.0 / std::sqrt(out.numel()));
    // Expected stdev is 1/sqrt(12) ~= 0.289
    EXPECT_NEAR(stdev, 1.0 / std::sqrt(12), 0.1);
    EXPECT_GT(stdev, 0);
  }
};

TEST_F(OpRandTest, SmokeTest) {
  std::vector<int64_t> sizes = {2, 3, 4, 128};

#define TEST_ENTRY(ctype, dtype) test_rand<ctype, ScalarType::dtype>(sizes);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpRandTest, Rank) {
  std::vector<int64_t> sizes = {1024};

  for (int64_t i = 0; i < 4; i++) {
    sizes.push_back(i + 1);
    test_rand<float, executorch::aten::ScalarType::Float>(sizes);
  }
}
