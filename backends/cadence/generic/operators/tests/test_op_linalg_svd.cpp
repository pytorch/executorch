/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <vector>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <executorch/backends/cadence/generic/operators/op_linalg_svd.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

constexpr float kTol = 1e-4;

// The singular values are unique, so the test compares them with
// torch.linalg.svdvals. The signs of the singular vectors are not, so it checks
// that U and Vh are orthogonal and reconstruct A.
class GenericLinalgSvdTest : public OperatorTest {
 protected:
  void expect_valid_svd(
      const Tensor& a,
      const Tensor& u,
      const Tensor& s,
      const Tensor& vh,
      const std::vector<float>& expected_s) {
    const ssize_t batch = a.numel() / 9;
    for (ssize_t b = 0; b < batch; ++b) {
      const float* A = a.const_data_ptr<float>() + 9 * b;
      const float* U = u.const_data_ptr<float>() + 9 * b;
      const float* S = s.const_data_ptr<float>() + 3 * b;
      const float* V = vh.const_data_ptr<float>() + 9 * b;
      const float scale = std::max(1.0f, S[0]);
      for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(S[k], expected_s[3 * b + k], kTol * scale) << "batch " << b;
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          // U * diag(S) * Vh reconstructs A.
          float usv = 0, utu = 0, vvt = 0;
          for (int k = 0; k < 3; ++k) {
            usv += U[i * 3 + k] * S[k] * V[k * 3 + j];
            utu += U[k * 3 + i] * U[k * 3 + j];
            vvt += V[i * 3 + k] * V[j * 3 + k];
          }
          const float eye = i == j ? 1.0f : 0.0f;
          EXPECT_NEAR(usv, A[i * 3 + j], kTol * scale) << "batch " << b;
          // U and Vh are orthogonal.
          EXPECT_NEAR(utu, eye, kTol) << "batch " << b;
          EXPECT_NEAR(vvt, eye, kTol) << "batch " << b;
        }
      }
    }
  }

  TensorFactory<ScalarType::Float> tf_;
};

// A general matrix and a rank-deficient one, as a batch of two.
TEST_F(GenericLinalgSvdTest, DecomposesABatch) {
  const Tensor a = tf_.make(
      {2, 3, 3},
      {4.0,
       -2.0,
       1.0,
       3.0,
       6.0,
       -4.0,
       2.0,
       1.0,
       8.0,
       1.0,
       2.0,
       3.0,
       2.0,
       4.0,
       6.0,
       -1.0,
       0.5,
       2.0});
  Tensor u = tf_.zeros({2, 3, 3});
  Tensor s = tf_.zeros({2, 3});
  Tensor vh = tf_.zeros({2, 3, 3});

  linalg_svd_out(context_, a, false, true, std::nullopt, u, s, vh);

  // The second matrix has rank 2.
  expect_valid_svd(
      a, u, s, vh, {9.385550, 6.763656, 4.142996, 8.524661, 1.606288, 0.0});
}

TEST_F(GenericLinalgSvdTest, RejectsNonFloatInput) {
  TensorFactory<ScalarType::Double> tf_double;
  const Tensor a = tf_double.ones({3, 3});
  Tensor u = tf_.zeros({3, 3});
  Tensor s = tf_.zeros({3});
  Tensor vh = tf_.zeros({3, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      linalg_svd_out(context_, a, false, true, std::nullopt, u, s, vh));
}

TEST_F(GenericLinalgSvdTest, RejectsMatricesThatAreNot3x3) {
  const Tensor a = tf_.ones({2, 2});
  Tensor u = tf_.zeros({2, 2});
  Tensor s = tf_.zeros({2});
  Tensor vh = tf_.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      linalg_svd_out(context_, a, false, true, std::nullopt, u, s, vh));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
