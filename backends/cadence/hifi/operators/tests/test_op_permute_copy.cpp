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

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class HiFiPermuteCopyTest : public OperatorTest {
 public:
 protected:
  Tensor& permute_copy_out(const Tensor& in, IntArrayRef dims, Tensor& out) {
    return ::cadence::impl::HiFi::native::permute_copy_out(
        context_, in, dims, out);
  }
};

TEST_F(HiFiPermuteCopyTest, FloatPermute2DTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor expected = tf.make({3, 2}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, IntPermute2DTest) {
  TensorFactory<ScalarType::Int> tf;
  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor expected = tf.make({3, 2}, {1, 4, 2, 5, 3, 6});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, Int8Permute2DTest) {
  TensorFactory<ScalarType::Char> tf;
  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor expected = tf.make({3, 2}, {1, 4, 2, 5, 3, 6});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, UInt8Permute2DTest) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor expected = tf.make({3, 2}, {1, 4, 2, 5, 3, 6});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, DoublePermute2DTest) {
  TensorFactory<ScalarType::Double> tf;
  Tensor in = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor expected = tf.make({3, 2}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, Long8Permute2DTest) {
  TensorFactory<ScalarType::Long> tf;
  Tensor in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor expected = tf.make({3, 2}, {1, 4, 2, 5, 3, 6});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, BoolPermute2DTest) {
  TensorFactory<ScalarType::Bool> tf;
  Tensor in = tf.make({2, 3}, {true, false, true, false, true, false});
  Tensor expected = tf.make({3, 2}, {true, false, false, true, true, false});

  Tensor out = tf.zeros({3, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, Float3DPermuteTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  Tensor expected =
      tf.make({2, 2, 2}, {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0});

  Tensor out = tf.zeros({2, 2, 2});
  std::vector<int64_t> dims = {2, 0, 1};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, Float4DPermuteTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({1, 2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  Tensor expected =
      tf.make({2, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

  Tensor out = tf.zeros({2, 1, 2, 2});
  std::vector<int64_t> dims = {1, 0, 2, 3};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, IdentityPermuteTest) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor expected = tf.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  Tensor out = tf.zeros({2, 3});
  std::vector<int64_t> dims = {0, 1};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, LargeTensorPermuteTest) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> input_data;
  for (int i = 0; i < 60; ++i) {
    input_data.push_back(static_cast<float>(i + 1));
  }
  Tensor in = tf.make({3, 4, 5}, input_data);

  // Permute: [3, 4, 5] -> [5, 3, 4] with dims [2, 0, 1]
  std::vector<float> expected_data(60);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        int old_idx = i * 20 + j * 5 + k;
        int new_idx = k * 12 + i * 4 + j;
        expected_data[new_idx] = static_cast<float>(old_idx + 1);
      }
    }
  }

  Tensor expected = tf.make({5, 3, 4}, expected_data);
  Tensor out = tf.zeros({5, 3, 4});
  std::vector<int64_t> dims = {2, 0, 1};

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiPermuteCopyTest, HighDimPermuteTest) {
  TensorFactory<ScalarType::Double> tf;
  std::vector<int32_t> shape = {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0};
  Tensor in = tf.make(shape, input_data);

  // Simple transpose: swap first and last dimension
  std::vector<int64_t> dims(16);
  for (int i = 0; i < 16; ++i) {
    dims[i] = i;
  }
  std::swap(dims[0], dims[15]);
  Tensor out = tf.zeros(shape);

  permute_copy_out(in, IntArrayRef(dims.data(), dims.size()), out);
  EXPECT_DOUBLE_EQ(out.const_data_ptr<double>()[0], 1.0);
  EXPECT_DOUBLE_EQ(out.const_data_ptr<double>()[1], 3.0);
  EXPECT_DOUBLE_EQ(out.const_data_ptr<double>()[2], 2.0);
  EXPECT_DOUBLE_EQ(out.const_data_ptr<double>()[3], 4.0);
}

TEST_F(HiFiPermuteCopyTest, MixedDataTypesTest) {
  TensorFactory<ScalarType::Short> tf_short;
  Tensor in_short = tf_short.make({2, 2}, {1, 2, 3, 4});
  Tensor expected_short = tf_short.make({2, 2}, {1, 3, 2, 4});
  Tensor out_short = tf_short.zeros({2, 2});
  std::vector<int64_t> dims = {1, 0};

  permute_copy_out(in_short, IntArrayRef(dims.data(), dims.size()), out_short);
  EXPECT_TENSOR_EQ(out_short, expected_short);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
