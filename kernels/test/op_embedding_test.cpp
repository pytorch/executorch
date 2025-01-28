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

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpEmbeddingOutTest : public OperatorTest {
 protected:
  Tensor& op_embedding_out(
      const Tensor& weight,
      const Tensor& indices,
      int64_t padding_idx,
      bool scale_grad_by_freq,
      bool sparse,
      Tensor& out) {
    return torch::executor::aten::embedding_outf(
        context_,
        weight,
        indices,
        padding_idx,
        scale_grad_by_freq,
        sparse,
        out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> tfl;
    // clang-format off
    Tensor weight = tf.make(
      {3, 2},
      {
        1, 2,
        3, 4,
        5, 6,
      });
    Tensor indices = tfl.make(
      {1, 2},
      {0, 2}
    );
    // clang-format on
    Tensor out = tf.zeros({1, 2, 2});
    Tensor actual = op_embedding_out(
        weight,
        indices,
        /*padding_idx=*/0,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false,
        out);

    Tensor expected = tf.make({1, 2, 2}, {1, 2, 5, 6});

    EXPECT_TENSOR_EQ(actual, out);
    EXPECT_TENSOR_EQ(out, expected);
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(embedding_template) */

    TensorFactory<ScalarType::Float> tf_weight;
    TensorFactory<ScalarType::Long> tf_indices;

    Tensor weight = tf_weight.make(
        {10, 3},
        {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
         0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
         0.9151939749717712,   0.39709991216659546, 0.8741558790206909,
         0.41940832138061523,  0.5529070496559143,  0.9527381062507629,
         0.036164820194244385, 0.1852310299873352,  0.37341737747192383});
    Tensor indices = tf_indices.make({2, 4}, {1, 2, 4, 5, 4, 3, 2, 9});
    Tensor expected = tf_weight.make(
        {2, 4, 3},
        {0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.036164820194244385, 0.1852310299873352,  0.37341737747192383});
    Tensor out = tf_weight.zeros(out_shape, dynamism);

    op_embedding_out(weight, indices, 0, false, false, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpEmbeddingOutTest, Smoke) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {2, 2},
    {
      1., 2.,
      0.5, 0.6,
    });
  // clang-format on
  Tensor out = tff.zeros({1, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({1}, {1});
  // clang-format on
  Tensor actual = op_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);
  // Embedding takes the ith entry in `weight` for i in `indices`. So out =
  // weight.index_select(indices.reshape(-1)), in this test, out = weight[1]
  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, tff.make({1, 2}, {0.5, 0.6}));
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpEmbeddingOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpEmbeddingOutTest, IndicesMultiDims) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({1, 2, 3, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({1, 2, 3}, {1, 0, 2, 3, 4, 0});
  // clang-format on
  Tensor actual = op_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);
  // clang-format off
  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, tff.make({1, 2, 3, 2}, {
      0.5, 0.6, // weight[1]
      1., 2.,   // weight[0]
      0.1, 0.2, // weight[2]
      3., 4.,   // weight[3]
      5., 6.,   // weight[4]
      1., 2.,   // weight[0]
  }));
  // clang-format on
}

TEST_F(OpEmbeddingOutTest, WeightWrongDimensionsDies) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {2, 2, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
    });
  // clang-format on
  Tensor out = tff.zeros({2, 2, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});
  // clang-format on
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_out(
          weight,
          indices,
          /*padding_idx=*/0,
          /*scale_grad_by_freq=*/false,
          /*sparse=*/false,
          out));
}

TEST_F(OpEmbeddingOutTest, WrongOutShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  auto wrong_outs = {
      tff.zeros({4, 3}), tff.zeros({4, 2}), tff.zeros({4, 2, 2})};

  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});

  for (auto wrong_out: wrong_outs) {
    // clang-format on
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_embedding_out(
            weight,
            indices,
            /*padding_idx=*/0,
            /*scale_grad_by_freq=*/false,
            /*sparse=*/false,
            wrong_out));
  }
}

TEST_F(OpEmbeddingOutTest, UnmatchedOutTypeDie) {
  TensorFactory<ScalarType::Float> tff;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });

  Tensor wrong_out = tfl.zeros({2, 2, 2});
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_out(
          weight,
          indices,
          /*padding_idx=*/0,
          /*scale_grad_by_freq=*/false,
          /*sparse=*/false,
          wrong_out));
}

TEST_F(OpEmbeddingOutTest, OutOfBoundIndicesDies) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({2, 2, 2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor neg_indices = tfl.make({2, 2}, {-1, 0, 2, 4});
  Tensor overflow_indices = tfl.make({2, 2}, {1, 0, 2, 8});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_out(
          weight,
          neg_indices,
          /*padding_idx=*/0,
          /*scale_grad_by_freq=*/false,
          /*sparse=*/false,
          out));

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_embedding_out(
          weight,
          overflow_indices,
          /*padding_idx=*/0,
          /*scale_grad_by_freq=*/false,
          /*sparse=*/false,
          out));
}

TEST_F(OpEmbeddingOutTest, EmptyWeightSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 0},
    {});
  // clang-format on
  Tensor out = tff.ones({2, 2, 0});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({2, 2}, {2, 0, 2, 4});

  Tensor actual = op_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(actual, tff.zeros({2, 2, 0}));
}

TEST_F(OpEmbeddingOutTest, ZeroDimIndicesSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({}, {3});

  // clang-format off
  Tensor expected = tff.make(
    {2},
    {3., 4.,}
  );
  // clang-format on

  Tensor actual = op_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpEmbeddingOutTest, EmptyDimIndicesSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({3, 0, 2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({3, 0}, {});

  // clang-format off
  Tensor expected = tff.make(
    {3, 0, 2},
    {}
  );
  // clang-format on

  Tensor actual = op_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
weight = torch.rand(10, 3)
indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
padding = 0
scale = False
sparse = False
expected = torch.nn.functional.embedding(
  indices, weight, padding_idx=padding, scale_grad_by_freq=scale, sparse=sparse
)
embedding_ template = f"""
  {declare_tensor_factory("ScalarType::Float", "tf_weight")}
  {declare_tensor_factory("ScalarType::Long", "tf_indices")}

  {declare_tensor_make_t("weight", "tf_weight")}
  {declare_tensor_make_t("indices", "tf_indices")}
  {declare_tensor_make_t("expected", "tf_weight")}
  {declare_tensor_zeros("out_shape, dynamism", "tf_weight", "out")}

  op_embedding_out(weight, indices, $padding$, $scale$, $sparse$, out);
  EXPECT_TENSOR_CLOSE(out, expected);""" */

TEST_F(OpEmbeddingOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 4, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpEmbeddingOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpEmbeddingOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
