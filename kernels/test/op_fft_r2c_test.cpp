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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;

class OpFftR2cOutTest : public OperatorTest {
 protected:
  Tensor& op_fft_r2c_out(
      const Tensor& in,
      IntArrayRef dim,
      int64_t normalization,
      bool onesided,
      Tensor& out) {
    return torch::executor::aten::_fft_r2c_outf(
        context_, in, dim, normalization, onesided, out);
  }

  template <
      class CTYPE,
      executorch::aten::ScalarType DTYPE,
      bool expect_failure = false>
  void test_dtype(int64_t norm, int64_t dim = 1, bool onesided = true) {
    TensorFactory<DTYPE> tf;
    constexpr auto DTYPE_OUT = executorch::runtime::toComplexType(DTYPE);
    TensorFactory<DTYPE_OUT> tf_out;

    using CTYPE_OUT =
        typename executorch::runtime::ScalarTypeToCppType<DTYPE_OUT>::type;

    Tensor in = tf.make({2, 4}, {0, 1, 2, 3, 0, 1, 2, 3});
    Tensor out = tf_out.full({2, 3}, CTYPE_OUT{0, 0});

    op_fft_r2c_out(in, {dim}, norm, onesided, out);

    double norm_factor = 1;
    if (norm == 1) {
      norm_factor = 2;
    } else if (norm == 2) {
      norm_factor = 4;
    }
    std::vector<CTYPE_OUT> expected_data = {
        CTYPE_OUT{6, 0},
        CTYPE_OUT{-2, 2},
        CTYPE_OUT{-2, 0},
        CTYPE_OUT{6, 0},
        CTYPE_OUT{-2, 2},
        CTYPE_OUT{-2, 0}};
    for (auto& elem : expected_data) {
      elem.real_ /= norm_factor;
      elem.imag_ /= norm_factor;
    }
    Tensor expected = tf_out.make({2, 3}, expected_data);

    if (!expect_failure) {
      EXPECT_TENSOR_CLOSE(out, expected);
    }
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  void test_dtype_multiple_axes(bool onesided = true) {
    TensorFactory<DTYPE> tf;
    constexpr auto DTYPE_OUT = executorch::runtime::toComplexType(DTYPE);
    TensorFactory<DTYPE_OUT> tf_out;

    using CTYPE_OUT =
        typename executorch::runtime::ScalarTypeToCppType<DTYPE_OUT>::type;

    Tensor in =
        tf.make({4, 4}, {0, 1, 2, 3, 3, 2, 1, 0, 2, 3, 0, 1, 1, 2, 3, 0});
    Tensor out = tf_out.full({4, 3}, CTYPE_OUT{0, 0});

    std::array<int64_t, 2> dim = {0, 1};
    op_fft_r2c_out(in, dim, 0, onesided, out);

    std::vector<CTYPE_OUT> expected_data = {
        CTYPE_OUT{24, 0},
        CTYPE_OUT{0, -4},
        CTYPE_OUT{0, 0},

        CTYPE_OUT{0, 0},
        CTYPE_OUT{-4, 0},
        CTYPE_OUT{0, 0},

        CTYPE_OUT{0, 0},
        CTYPE_OUT{0, 4},
        CTYPE_OUT{-8, 0},

        CTYPE_OUT{0, 0},
        CTYPE_OUT{-4, 8},
        CTYPE_OUT{0, 0},
    };
    Tensor expected = tf_out.make({4, 3}, expected_data);

    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpFftR2cOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype)           \
  test_dtype<ctype, ScalarType::dtype>(0); \
  test_dtype<ctype, ScalarType::dtype>(1); \
  test_dtype<ctype, ScalarType::dtype>(2);
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFftR2cOutTest, MultipleDims) {
#define TEST_ENTRY(ctype, dtype) \
  test_dtype_multiple_axes<ctype, ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFftR2cOutTest, InvalidNorm) {
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

TEST_F(OpFftR2cOutTest, InvalidDim) {
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

// TODO: support this and patch test accordingly!
TEST_F(OpFftR2cOutTest, TwoSidedIsNotSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen supports two-sided";
    return;
  }
  auto twosided = [this]() {
    test_dtype<double, ScalarType::Double, /* expect_failure = */ true>(
        0, 1, false);
  };
  ET_EXPECT_KERNEL_FAILURE(context_, twosided());
}
