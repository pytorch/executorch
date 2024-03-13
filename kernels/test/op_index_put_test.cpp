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
#include <sys/types.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

using OptTensorArrayRef = ArrayRef<optional<Tensor>>;

class OpIndexPutOutTest : public OperatorTest {
 protected:
  Tensor& op_index_put_out(
      const Tensor& input,
      OptTensorArrayRef indices,
      const Tensor& values,
      const bool accumulate,
      Tensor& out) {
#ifdef USE_ATEN_LIB
    c10::List<c10::optional<at::Tensor>> indices_list(indices);
    return torch::executor::aten::index_put_outf(
        context_, input, indices_list, values, accumulate, out);
#else
    return torch::executor::aten::index_put_outf(
        context_, input, indices, values, accumulate, out);
#endif
  }

  template <
      exec_aten::ScalarType INPUT_DTYPE,
      exec_aten::ScalarType INDICES_DTYPE>
  void test_dtype() {
    TensorFactory<INPUT_DTYPE> tf;
    TensorFactory<INDICES_DTYPE> tfl;
    TensorFactory<ScalarType::Bool> tfb;

    // clang-format off
    Tensor x = tf.make(
        {3, 2, 4},
        {
          // [0, :, :]
          1, 1, 1, 1, // [0, 0, :]
          0, 0, 0, 0, // [0, 1, :]

          // [1, :, :]
          1, 1, 1, 1, // [1, 0, :]
          0, 0, 0, 0, // [1, 1, :]

          // [2, :, :]
          1, 1, 1, 1, // [2, 0, :]
          0, 0, 0, 0, // [2, 1, :]
        });
    // clang-format on

    // First, index_put to make everything equal to 1

    // indices [0, 1, :], [1, 1, :], [2, 1, :]
    optional<Tensor> indices[] = {
        optional<Tensor>(tfl.make({1, 3}, {0, 1, 2})),
        optional<Tensor>(tfl.make({1, 3}, {1, 1, 1})),
    };
    // bool representation of the same index list
    optional<Tensor> indices_bool[] = {
        optional<Tensor>(tfb.make({3}, {true, true, true})),
        optional<Tensor>(tfb.make({2}, {false, true})),
    };

    Tensor values = tf.ones({3, 4});

    std::vector<int32_t> out_size{3, 2, 4};

    Tensor out = tf.zeros(out_size);
    Tensor ret =
        op_index_put_out(x, indices, values, /*accumulate=*/false, out);

    EXPECT_TENSOR_EQ(ret, out);
    EXPECT_TENSOR_EQ(ret, tf.ones(out_size));

    // Repeat the test with bool indices
    Tensor out_with_bool = tf.zeros(out_size);
    Tensor ret_with_bool = op_index_put_out(
        x, indices_bool, values, /*accumulate=*/false, out_with_bool);

    EXPECT_TENSOR_EQ(ret_with_bool, out_with_bool);
    EXPECT_TENSOR_EQ(ret_with_bool, tf.ones(out_size));

    // Then, index_put to make everything equal to 0

    // indices [0, 1, :], [1, 0, :], [2, 0, :]
    optional<Tensor> indices_alt[] = {
        optional<Tensor>(tfl.make({1, 3}, {0, 1, 2})),
        optional<Tensor>(tfl.make({1, 3}, {0, 0, 0})),
    };
    // bool representation of the same index list
    optional<Tensor> indices_alt_bool[] = {
        optional<Tensor>(tfb.make({3}, {true, true, true})),
        optional<Tensor>(tfb.make({2}, {true, false})),
    };

    Tensor values_alt = tf.zeros({3, 4});

    Tensor out_alt = tf.ones(out_size);
    Tensor ret_alt = op_index_put_out(
        x, indices_alt, values_alt, /*accumulate=*/false, out_alt);

    EXPECT_TENSOR_EQ(ret_alt, out_alt);
    EXPECT_TENSOR_EQ(ret_alt, tf.zeros(out_size));

    // Repeat the test with bool indices
    Tensor out_alt_with_bool = tf.ones(out_size);
    Tensor ret_alt_with_bool = op_index_put_out(
        x,
        indices_alt_bool,
        values_alt,
        /*accumulate=*/false,
        out_alt_with_bool);

    EXPECT_TENSOR_EQ(ret_alt_with_bool, out_alt_with_bool);
    EXPECT_TENSOR_EQ(ret_alt_with_bool, tf.zeros(out_size));
  }

  /* %python
  import torch
  torch.manual_seed(0)
  input = torch.rand(2, 3, 4)
  indices = [torch.tensor([1]), torch.tensor([0]), torch.tensor([1, 2])]
  values = torch.rand(2)
  accumulate = False
  expected = input.index_put(indices, values, accumulate=accumulate)

  index_put_template = f"""
    {declare_tensor_factory("ScalarType::Float", "tf")}
    {declare_tensor_factory("ScalarType::Long", "tf_indices")}

    {declare_tensor_make_t("input", "tf")}
    {declare_optional_tensor_list_make_t("indices", "tf_indices")}
    {declare_tensor_make_t("values", "tf")}
    {declare_tensor_make_t("expected", "tf")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out")}

    op_index_put_out(input, indices, values, $accumulate$, out);
    EXPECT_TENSOR_EQ(out, expected);"""
  */
  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(index_put_template) */

    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tf_indices;

    Tensor input = tf.make(
        {2, 3, 4},
        {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
         0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
         0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
    optional<Tensor> indices[] = {
        optional<Tensor>(tf_indices.make({1}, {1})),
        optional<Tensor>(tf_indices.make({1}, {0})),
        optional<Tensor>(tf_indices.make({2}, {1, 2}))};
    Tensor values = tf.make({2}, {0.41940832138061523, 0.5529070496559143});
    Tensor expected = tf.make(
        {2, 3, 4},
        {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
         0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.022325754165649414, 0.41940832138061523, 0.5529070496559143,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
         0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_index_put_out(input, indices, values, false, out);
    EXPECT_TENSOR_EQ(out, expected);
  }

  // Run the test by putting values into the selected elements
  void run_test_cases(
      const Tensor& x,
      OptTensorArrayRef indices,
      const Tensor& values,
      const Tensor& expected,
      const Tensor& expected_accum) {
    // Generated out tensor sharing same size and dtype with expected tensor
    TensorFactory<ScalarType::Double> tf;

    const std::vector<int32_t> out_size(
        expected.sizes().begin(), expected.sizes().end());
    Tensor out = tf.ones(out_size);

    Tensor ret =
        op_index_put_out(x, indices, values, /*accumulate=*/false, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected);

    Tensor out_accum = tf.ones(out_size);
    Tensor ret_accum =
        op_index_put_out(x, indices, values, /*accumulate=*/true, out_accum);
    EXPECT_TENSOR_EQ(out_accum, ret_accum);
    EXPECT_TENSOR_EQ(ret_accum, expected_accum);
  }
};

//
// Correctness Tests
//

TEST_F(OpIndexPutOutTest, IndexPutMask) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Bool> tfb;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor indices = tfb.make(
      {2, 3, 4},
      {
         // [0, :, :]
          true, false, false, false, // [0, 0, :]
         false, false,  true, false, // [0, 1, :]
         false, false, false, false, // [0, 2, :]

         // [1, :, :]
         false,  true, false, false, // [1, 0, :]
         false, false, false, false, // [1, 1, :]
         false, false,  true, false, // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor values = tf.make(
    {4},
    {10., 20., 30., 40.}
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
         10.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  20.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  30.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10.,  40., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
         11.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  27.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  28.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10.,  29., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, {indices}, values, expected, expected_accum);
}

TEST_F(OpIndexPutOutTest, IndexPutMaskBroadcast) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Bool> tfb;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the input value at indices
  // [1, 0, 1], [1, 0, 2]. This is expressed in various ways to test different
  // indexing expressions.

  // clang-format off
  Tensor indices = tfb.make(
      {2, 3, 4},
      {
         // [0, :, :]
          true, false, false, false, // [0, 0, :]
         false, false,  true, false, // [0, 1, :]
         false, false, false, false, // [0, 2, :]

         // [1, :, :]
         false,  true, false, false, // [1, 0, :]
         false, false, false, false, // [1, 1, :]
         false, false,  true, false, // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
         10.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  10.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  10.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10.,  10., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
         11.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  17.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,   8.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10.,  -1., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, {indices}, values, expected, expected_accum);
}

TEST_F(OpIndexPutOutTest, PutFrontDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Bool> tfb;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the input value at indices
  // [1, 0, 1], [1, 0, 2]. This is expressed in various ways to test different
  // indexing expressions.

  optional<Tensor> indices_long[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {1, 2}))};

  optional<Tensor> indices_int[] = {
      optional<Tensor>(tfi.make({1}, {1})),
      optional<Tensor>(tfi.make({1}, {0})),
      optional<Tensor>(tfi.make({2}, {1, 2}))};

  optional<Tensor> indices_negative[] = {
      optional<Tensor>(tfl.make({1}, {-1})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  optional<Tensor> indices_bool[] = {
      optional<Tensor>(tfb.make({2}, {false, true})),
      optional<Tensor>(tfb.make({3}, {true, false, false})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  optional<Tensor> indices_mixed[] = {
      optional<Tensor>(tfb.make({2}, {false, true})),
      optional<Tensor>(tfl.make({1}, {0})),
      optional<Tensor>(tfl.make({2}, {-3, -2}))};

  // clang-format off
  Tensor values = tf.make(
    {2},
    {10., 20.}
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  10.,  20.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,   8.,  17.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, indices_long, values, expected, expected_accum);
  run_test_cases(x, indices_int, values, expected, expected_accum);
  run_test_cases(x, indices_negative, values, expected, expected_accum);
  run_test_cases(x, indices_bool, values, expected, expected_accum);
  run_test_cases(x, indices_mixed, values, expected, expected_accum);
}

TEST_F(OpIndexPutOutTest, PutTwoValuesAtSameIndex) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the value at the same index
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1, 2}, {0, 0})),
      optional<Tensor>(tfl.make({1, 2}, {1, 1})),
      optional<Tensor>(tfl.make({1, 2}, {2, 2}))};

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.,}
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  10.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,  27.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, /*indices=*/indices, values, expected, expected_accum);
}

TEST_F(OpIndexPutOutTest, IndicesFewerThanInputDimSupported) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Bool> tfb;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the input value at indices
  // [1, 0, :], [1, 1, :]. This is expressed in various ways to test different
  // indexing expressions.

  optional<Tensor> indices_long[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({2}, {0, 1}))};

  optional<Tensor> indices_mixed[] = {
      optional<Tensor>(tfi.make({1}, {-1})),
      optional<Tensor>(tfb.make({3}, {true, true, false}))};

  // clang-format off
  Tensor values = tf.make(
    {2, 4},
    {
       10.,  20.,  30.,  40.,
      -10., -20., -30., -40.,
    }
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
          10.,  20.,  30.,  40., // [1, 0, :]
         -10., -20., -30., -40., // [1, 1, :]
          -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
           9.,  18.,  27.,  36., // [1, 0, :]
         -15., -26., -37., -48., // [1, 1, :]
          -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, indices_long, values, expected, expected_accum);
  run_test_cases(x, indices_mixed, values, expected, expected_accum);
}

TEST_F(OpIndexPutOutTest, IndicesFewerThanInputDimSupportedSameValue) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the input value at indices
  // [1, 0, :], [1, 1, :]
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({1}, {1})),
      optional<Tensor>(tfl.make({2}, {0, 1}))};

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         10.,  10.,  10.,  10., // [1, 0, :]
         10.,  10.,  10.,  10., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // clang-format off
  Tensor expected_accum = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
          9.,   8.,   7.,   6., // [1, 0, :]
          5.,   4.,   3.,   2., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  run_test_cases(x, /*indices=*/indices, values, expected, expected_accum);
}

//
// Test that all dtypes are supported
//

/**
 * Generic test for integral index lists
 */
TEST_F(OpIndexPutOutTest, AllDtypesSupportedForInput) {
#define TEST_ENTRY(ctype, dtype) \
  test_dtype<ScalarType::dtype, ScalarType::Long>();

  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);

#undef TEST_ENTRY
}

TEST_F(OpIndexPutOutTest, AllDtypesSupportedForIndicesList) {
  test_dtype<ScalarType::Float, ScalarType::Long>();
  test_dtype<ScalarType::Float, ScalarType::Int>();
}

//
// Death Tests
//

TEST_F(OpIndexPutOutTest, IndexOutOfBoundDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfl.make({1}, {5});

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, NegativeIndexOutOfBoundDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfl.make({1}, {-5});

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, TooManyBooleanIndexCountDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfb.make({3}, {true, true, false});

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, TooFewBooleanIndexCountDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({4});
  Tensor out = tf.zeros({4});
  Tensor index = tfb.make({1}, {true});

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  // ATen kernel will throw exception instead of death
  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, MismatchedIndexMaskDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor x = tf.ones({4, 4});
  Tensor out = tf.zeros({4, 4});
  Tensor index = tfb.ones({3, 3});

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  // ATen kernel will throw exception instead of death
  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, MismatchedOutputDtypesDies) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor x = tf_float.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_double.ones({1, 2, 2});
  Tensor index = tf_long.make({1}, {0});

  // clang-format off
  Tensor values = tf_float.make(
    {1},
    {10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, MismatchedValuesDtypesDies) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor x = tf_float.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});
  Tensor index = tf_long.make({1}, {0});

  // clang-format off
  Tensor values = tf_double.make(
    {1},
    {10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, ValuesSizeMismatchDimDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  Tensor index = tfl.make({1}, {1});

  Tensor out = tf.ones({2, 4, 7, 5});

  // clang-format off
  Tensor values = tf.make(
    {1, 2},
    {10., 10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(
          x, /*indices=*/{index}, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, InvalidIndicesDtypeDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Float> tff;

  Tensor x = tf.zeros({2, 4, 7, 5});
  // clang-format off
  optional<Tensor> indices[] = {
      optional<Tensor>(tff.make({3}, {1, 1, 1,})),
      optional<Tensor>(tff.make({2}, {1, 2}))};
  // clang-format on

  Tensor out = tf.ones({2, 4, 7, 5});

  // clang-format off
  Tensor values = tf.make(
    {1,},
    {10}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(x, indices, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, InvalidIndicesShapesDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  // clang-format off
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({3}, {1, 1, 1,})),
      optional<Tensor>(tfl.make({2}, {1, 2}))};

  Tensor out = tf.ones({2, 4, 7, 5});
  // clang-format on

  // clang-format off
  Tensor values = tf.make(
    {1, 2},
    {10., 10.}
  );
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE_WITH_MSG(
      context_,
      op_index_put_out(x, indices, values, /*accumulate=*/false, out),
      "");
}

TEST_F(OpIndexPutOutTest, NonLinearIndices) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({4, 4});
  // clang-format off
  optional<Tensor> indices[] = {
      optional<Tensor>(tfl.make({2, 2}, {1, 1, 1, 1,})),
      optional<Tensor>(tfl.make({1, 2}, {3, 0,}))};

  Tensor out = tf.ones({4, 4});
  // clang-format on

  // clang-format off
  Tensor values = tf.make(
    {1},
    {10.}
  );
  // clang-format on

  Tensor expected =
      tf.make({4, 4}, {0, 0, 0, 0, 10, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0});

  Tensor ret = op_index_put_out(x, indices, values, /*accumulate=*/false, out);

  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}

//
// Dynamic Shape Tests
//

TEST_F(OpIndexPutOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpIndexPutOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpIndexPutOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
