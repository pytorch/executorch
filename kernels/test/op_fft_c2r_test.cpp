/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;

class OpFftC2rOutTest : public OperatorTest {
 protected:
  Tensor& op_fft_c2r_out(
      const Tensor& in,
      IntArrayRef dim,
      int64_t normalization,
      int64_t last_dim_size,
      Tensor& out) {
    return torch::executor::aten::_fft_c2r_outf(
        context_, in, dim, normalization, last_dim_size, out);
  }

  template <
      class CTYPE_OUT,
      executorch::aten::ScalarType DTYPE_OUT,
      bool expect_failure = false>
  void test_dtype(int64_t norm, int64_t dim = 0) {
    TensorFactory<DTYPE_OUT> tf_out;
    constexpr auto DTYPE_IN = executorch::runtime::toComplexType(DTYPE_OUT);
    TensorFactory<DTYPE_IN> tf_in;

    using CTYPE_IN =
        typename executorch::runtime::ScalarTypeToCppType<DTYPE_IN>::type;

    std::vector<CTYPE_IN> input_data = {
        CTYPE_IN{24, 4},
        CTYPE_IN{4, -8},
        CTYPE_IN{0, 4},

        CTYPE_IN{8, -16},
        CTYPE_IN{-4, 0},
        CTYPE_IN{0, 32},

        CTYPE_IN{12, 0},
        CTYPE_IN{0, 4},
        CTYPE_IN{-8, 4},

        CTYPE_IN{0, 8},
        CTYPE_IN{-4, 8},
        CTYPE_IN{8, 0},
    };

    Tensor in = tf_in.make({4, 3}, input_data);
    Tensor out = tf_out.full({4, 3}, 0);

    int64_t last_dim_size =
        (dim >= 0 && dim < out.dim()) ? out.sizes()[dim] : 0;
    op_fft_c2r_out(in, {dim}, norm, last_dim_size, out);

    double norm_factor = 1;
    if (norm == 1) {
      norm_factor = 2;
    } else if (norm == 2) {
      norm_factor = 4;
    }
    std::vector<CTYPE_OUT> expected_data = {
        52., -4., -8., 44., 4., -56., 20., 12., -8., -20., 4., 72.};
    for (auto& elem : expected_data) {
      elem /= norm_factor;
    }
    Tensor expected = tf_out.make({4, 3}, expected_data);

    if (!expect_failure) {
      EXPECT_TENSOR_CLOSE(out, expected);
    }
  }

  template <class CTYPE_OUT, executorch::aten::ScalarType DTYPE_OUT>
  void test_dtype_multiple_axes() {
    TensorFactory<DTYPE_OUT> tf_out;
    constexpr auto DTYPE_IN = executorch::runtime::toComplexType(DTYPE_OUT);
    TensorFactory<DTYPE_IN> tf_in;

    using CTYPE_IN =
        typename executorch::runtime::ScalarTypeToCppType<DTYPE_IN>::type;

    std::vector<CTYPE_IN> input_data = {
        CTYPE_IN{16, 4},
        CTYPE_IN{4, -8},
        CTYPE_IN{0, 4},

        CTYPE_IN{8, -16},
        CTYPE_IN{-4, 0},
        CTYPE_IN{0, 36},

        CTYPE_IN{32, 0},
        CTYPE_IN{0, 4},
        CTYPE_IN{-8, 4},

        CTYPE_IN{0, 8},
        CTYPE_IN{-4, 8},
        CTYPE_IN{8, 0},
    };

    Tensor in = tf_in.make({4, 3}, input_data);
    Tensor out = tf_out.full({4, 4}, 0);

    int64_t last_dim_size = out.sizes()[0];
    std::array<int64_t, 2> dim = {0, 1};
    op_fft_c2r_out(in, dim, 1, last_dim_size, out);

    std::vector<CTYPE_OUT> expected_data = {
        12.,
        12.,
        16.,
        16.,
        1.,
        15.,
        -11.,
        3.,
        12.,
        20.,
        0.,
        8.,
        -1.,
        -15.,
        3.,
        -27.};
    Tensor expected = tf_out.make({4, 4}, expected_data);
    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpFftC2rOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype)           \
  test_dtype<ctype, ScalarType::dtype>(0); \
  test_dtype<ctype, ScalarType::dtype>(1); \
  test_dtype<ctype, ScalarType::dtype>(2);
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFftC2rOutTest, MultipleDims) {
#define TEST_ENTRY(ctype, dtype) \
  test_dtype_multiple_axes<ctype, ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFftC2rOutTest, InvalidNorm) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen MKL path does not validate norm";
    return;
  }
  auto invalid_norm = [this](int64_t norm) {
    test_dtype<float, ScalarType::Float, /* expect_failure = */ true>(norm);
  };
  ET_EXPECT_KERNEL_FAILURE(context_, invalid_norm(3));
  ET_EXPECT_KERNEL_FAILURE(context_, invalid_norm(4));
  ET_EXPECT_KERNEL_FAILURE(context_, invalid_norm(-1));
  ET_EXPECT_KERNEL_FAILURE(context_, invalid_norm(9999999));
}

TEST_F(OpFftC2rOutTest, InvalidDim) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen fails UBSAN";
    return;
  }
  auto negative_dim = [this]() {
    test_dtype<float, ScalarType::Float, /* expect_failure = */ true>(0, -1);
    test_dtype<float, ScalarType::Float, /* expect_failure = */ true>(0, 3);
    test_dtype<float, ScalarType::Float, /* expect_failure = */ true>(0, 9001);
  };
  ET_EXPECT_KERNEL_FAILURE(context_, negative_dim());
}
