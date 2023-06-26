// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/core/kernel_types/util/ScalarTypeUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& scalar_tensor_out(const Scalar& s, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::scalar_tensor_outf(context, s, out);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_0d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

#define GENERATE_TEST_0D(ctype, dtype)                      \
  TEST(OpScalarTensorOutKernelTest, dtype##TensorsDim0) {   \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(4); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(8); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(9); \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST_0D)

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_1d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_2d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1, 1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_3d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1, 1, 1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

#define GENERATE_TEST(ctype, dtype)                                    \
  TEST(OpScalarTensorOutKernelTest, dtype##Tensors) {                  \
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) { \
      GTEST_SKIP() << "ATen kernel resizes output to shape {}";        \
    }                                                                  \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(7);            \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST)

TEST(OpScalarTensorOutKernelTest, InvalidOutShapeFails) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel will reshape output";
  }

  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> sizes{1, 2, 1};

  Tensor out = tf.ones(sizes);
  ET_EXPECT_KERNEL_FAILURE(scalar_tensor_out(7, out));
}
