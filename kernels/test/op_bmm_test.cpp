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

class OpBmmOutTest : public OperatorTest {
 protected:
  Tensor& op_bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    return torch::executor::aten::bmm_outf(context_, self, mat2, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Gives 4 * 2 * 3 = 24, shape (10, 3, 5)
    Tensor x = tf.full({10, 3, 4}, 2);
    Tensor y = tf.full({10, 4, 5}, 3);

    Tensor out = tf.zeros({10, 3, 5});
    op_bmm_out(x, y, out);

    Tensor expected = tf.full({10, 3, 5}, 24);

    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpBmmOutTest, OutputDim) {
  TensorFactory<ScalarType::Int> tf;

  // Two tensors with compatible dimensions: (10, 3, 4) and (10, 4, 5).
  Tensor x = tf.ones({10, 3, 4});
  Tensor y = tf.ones({10, 4, 5});

  // Output shape should be (10, 3, 5)
  Tensor out = tf.zeros({10, 3, 5});

  Tensor ret = op_bmm_out(x, y, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, filled with 4.
  Tensor expected = tf.full({10, 3, 5}, 4);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBmmOutTest, OutputDimFloat) {
  TensorFactory<ScalarType::Float> tf;

  // clang-format off
  Tensor x = tf.make(
      {2, 4, 5},
      {
        4., 3., 1., 1., 1.,
        3., 1., 4., 4., 2.,
        1., 1., 1., 3., 3.,
        4., 2., 2., 2., 3.,

        1., 3., 1., 4., 4.,
        1., 1., 2., 4., 3.,
        4., 3., 4., 1., 2.,
        1., 4., 4., 4., 4.,
      });
  // clang-format on

  // clang-format off
  Tensor y = tf.make(
      {2, 5, 3},
      {
        4., 4., 4.,
        2., 3., 1.,
        1., 4., 4.,
        3., 1., 2.,
        1., 4., 3.,

        1., 4., 4.,
        4., 4., 4.,
        2., 1., 4.,
        1., 4., 3.,
        1., 4., 4.,
      });
  // clang-format on

  // Output shape should be (10, 3, 5)
  Tensor out = tf.zeros({2, 4, 3});

  Tensor ret = op_bmm_out(x, y, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // clang-format off
  Tensor expected = tf.make(
      {2, 4, 3},
      {
        27., 34., 28.,
        32., 43., 43.,
        19., 26., 24.,
        31., 44., 39.,

        23., 49., 48.,
        16., 38., 40.,
        27., 44., 55.,
        33., 56., 64.,
      });
  // clang-format on

  EXPECT_TENSOR_EQ(out, expected);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpBmmOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpBmmOutTest, EmptyInputWithEmptyOutTensorPasses) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.full({2, 2, 2}, 3);
  Tensor y = tf.make({2, 2, 0}, {});

  // Make an empty out tensor and demonstrate that it's empty.
  Tensor out = tf.make({2, 2, 0}, {});

  EXPECT_EQ(out.numel(), 0);

  op_bmm_out(x, y, out);

  EXPECT_EQ(out.numel(), 0);
}

TEST_F(OpBmmOutTest, MismatchedDimensionsDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({2, 10, 3});

  // wrong_y has incompatible shape
  Tensor wrong_y = tf.ones({3, 7, 4});
  Tensor right_y = tf.ones({2, 3, 4});

  Tensor out = tf.ones({2, 10, 4});

  ET_EXPECT_KERNEL_FAILURE(context_, op_bmm_out(x, wrong_y, out));

  EXPECT_TENSOR_EQ(op_bmm_out(x, right_y, out), tf.full({2, 10, 4}, 3));
}

TEST_F(OpBmmOutTest, MismatchedDimensionSizeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimension size";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({2, 10, 3});

  Tensor y = tf.ones({2, 3, 4});

  // wrong_y has incompatible dim
  Tensor wrong_y = tf.ones({7, 4});
  Tensor right_y = tf.ones({2, 3, 4});

  // wrong_out has incompatible dim
  Tensor right_out = tf.ones({2, 10, 4});
  Tensor wrong_out = tf.ones({7, 5});

  ET_EXPECT_KERNEL_FAILURE(context_, op_bmm_out(x, right_y, wrong_out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_bmm_out(x, wrong_y, right_out));
}

TEST_F(OpBmmOutTest, WrongOutShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones({2, 10, 3});

  Tensor y = tf.ones({2, 3, 4});

  // wrong_out has incompatible shape
  Tensor right_out = tf.ones({2, 10, 4});
  Tensor wrong_out = tf.ones({3, 7, 5});

  ET_EXPECT_KERNEL_FAILURE(context_, op_bmm_out(x, y, wrong_out));

  EXPECT_TENSOR_EQ(op_bmm_out(x, y, right_out), tf.full({2, 10, 4}, 3));
}

TEST_F(OpBmmOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 3, 6},
      {0.7231091856956482,    0.7423362731933594,  0.5262957811355591,
       0.24365824460983276,   0.584592342376709,   0.033152639865875244,
       0.13871687650680542,   0.242235004901886,   0.815468966960907,
       0.793160617351532,     0.2782524824142456,  0.48195880651474,
       0.8197803497314453,    0.9970665574073792,  0.6984410881996155,
       0.5675464272499084,    0.8352431654930115,  0.2055988311767578,
       0.593172013759613,     0.11234724521636963, 0.1534569263458252,
       0.24170821905136108,   0.7262365221977234,  0.7010802030563354,
       0.2038237452507019,    0.6510535478591919,  0.7744860053062439,
       0.4368913173675537,    0.5190907716751099,  0.6158523559570312,
       0.8101882934570312,    0.9800970554351807,  0.1146882176399231,
       0.3167651295661926,    0.6965049505233765,  0.9142746925354004,
       0.9351036548614502,    0.9411783814430237,  0.5995072722434998,
       0.06520867347717285,   0.5459962487220764,  0.18719732761383057,
       0.03402292728424072,   0.944246232509613,   0.8801798820495605,
       0.0012360215187072754, 0.5935860276222229,  0.4157699942588806,
       0.41771942377090454,   0.2711215615272522,  0.6922780871391296,
       0.2038482427597046,    0.6832956671714783,  0.75285404920578});
  auto y = tf.make(
      {3, 6, 2},
      {0.8579357862472534,   0.6869555711746216,  0.0051323771476745605,
       0.17565155029296875,  0.7496575117111206,  0.6046506762504578,
       0.1099579930305481,   0.21209025382995605, 0.9703746438026428,
       0.8369089365005493,   0.28198742866516113, 0.3741576075553894,
       0.023700952529907227, 0.49101293087005615, 0.12347054481506348,
       0.11432164907455444,  0.4724501967430115,  0.5750725269317627,
       0.2952348589897156,   0.7966887950897217,  0.19573044776916504,
       0.9536850452423096,   0.8426499366760254,  0.07835853099822998,
       0.3755578398704529,   0.5225613117218018,  0.572950541973114,
       0.6185871362686157,   0.6962141394615173,  0.5299500823020935,
       0.25603562593460083,  0.7365944981575012,  0.020375549793243408,
       0.20364665985107422,  0.3748350739479065,  0.2564433217048645});
  Tensor expected_result = tf.make(
      {3, 3, 2},
      {1.6221470832824707,
       1.498693823814392,
       1.224705696105957,
       1.2123372554779053,
       2.1629090309143066,
       2.05692195892334,
       0.9047035574913025,
       1.3324503898620605,
       1.2006582021713257,
       1.5112680196762085,
       1.1946606636047363,
       1.5640640258789062,
       1.405808448791504,
       1.5957869291305542,
       1.3348338603973389,
       1.2967426776885986,
       1.1425018310546875,
       1.2352378368377686});

  Tensor out =
      tf.zeros({3, 3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_bmm_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpBmmOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 3, 6},
      {0.7231091856956482,    0.7423362731933594,  0.5262957811355591,
       0.24365824460983276,   0.584592342376709,   0.033152639865875244,
       0.13871687650680542,   0.242235004901886,   0.815468966960907,
       0.793160617351532,     0.2782524824142456,  0.48195880651474,
       0.8197803497314453,    0.9970665574073792,  0.6984410881996155,
       0.5675464272499084,    0.8352431654930115,  0.2055988311767578,
       0.593172013759613,     0.11234724521636963, 0.1534569263458252,
       0.24170821905136108,   0.7262365221977234,  0.7010802030563354,
       0.2038237452507019,    0.6510535478591919,  0.7744860053062439,
       0.4368913173675537,    0.5190907716751099,  0.6158523559570312,
       0.8101882934570312,    0.9800970554351807,  0.1146882176399231,
       0.3167651295661926,    0.6965049505233765,  0.9142746925354004,
       0.9351036548614502,    0.9411783814430237,  0.5995072722434998,
       0.06520867347717285,   0.5459962487220764,  0.18719732761383057,
       0.03402292728424072,   0.944246232509613,   0.8801798820495605,
       0.0012360215187072754, 0.5935860276222229,  0.4157699942588806,
       0.41771942377090454,   0.2711215615272522,  0.6922780871391296,
       0.2038482427597046,    0.6832956671714783,  0.75285404920578});
  auto y = tf.make(
      {3, 6, 2},
      {0.8579357862472534,   0.6869555711746216,  0.0051323771476745605,
       0.17565155029296875,  0.7496575117111206,  0.6046506762504578,
       0.1099579930305481,   0.21209025382995605, 0.9703746438026428,
       0.8369089365005493,   0.28198742866516113, 0.3741576075553894,
       0.023700952529907227, 0.49101293087005615, 0.12347054481506348,
       0.11432164907455444,  0.4724501967430115,  0.5750725269317627,
       0.2952348589897156,   0.7966887950897217,  0.19573044776916504,
       0.9536850452423096,   0.8426499366760254,  0.07835853099822998,
       0.3755578398704529,   0.5225613117218018,  0.572950541973114,
       0.6185871362686157,   0.6962141394615173,  0.5299500823020935,
       0.25603562593460083,  0.7365944981575012,  0.020375549793243408,
       0.20364665985107422,  0.3748350739479065,  0.2564433217048645});
  Tensor expected_result = tf.make(
      {3, 3, 2},
      {1.6221470832824707,
       1.498693823814392,
       1.224705696105957,
       1.2123372554779053,
       2.1629090309143066,
       2.05692195892334,
       0.9047035574913025,
       1.3324503898620605,
       1.2006582021713257,
       1.5112680196762085,
       1.1946606636047363,
       1.5640640258789062,
       1.405808448791504,
       1.5957869291305542,
       1.3348338603973389,
       1.2967426776885986,
       1.1425018310546875,
       1.2352378368377686});

  Tensor out =
      tf.zeros({6, 6, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_bmm_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpBmmOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 3, 6},
      {0.7231091856956482,    0.7423362731933594,  0.5262957811355591,
       0.24365824460983276,   0.584592342376709,   0.033152639865875244,
       0.13871687650680542,   0.242235004901886,   0.815468966960907,
       0.793160617351532,     0.2782524824142456,  0.48195880651474,
       0.8197803497314453,    0.9970665574073792,  0.6984410881996155,
       0.5675464272499084,    0.8352431654930115,  0.2055988311767578,
       0.593172013759613,     0.11234724521636963, 0.1534569263458252,
       0.24170821905136108,   0.7262365221977234,  0.7010802030563354,
       0.2038237452507019,    0.6510535478591919,  0.7744860053062439,
       0.4368913173675537,    0.5190907716751099,  0.6158523559570312,
       0.8101882934570312,    0.9800970554351807,  0.1146882176399231,
       0.3167651295661926,    0.6965049505233765,  0.9142746925354004,
       0.9351036548614502,    0.9411783814430237,  0.5995072722434998,
       0.06520867347717285,   0.5459962487220764,  0.18719732761383057,
       0.03402292728424072,   0.944246232509613,   0.8801798820495605,
       0.0012360215187072754, 0.5935860276222229,  0.4157699942588806,
       0.41771942377090454,   0.2711215615272522,  0.6922780871391296,
       0.2038482427597046,    0.6832956671714783,  0.75285404920578});
  auto y = tf.make(
      {3, 6, 2},
      {0.8579357862472534,   0.6869555711746216,  0.0051323771476745605,
       0.17565155029296875,  0.7496575117111206,  0.6046506762504578,
       0.1099579930305481,   0.21209025382995605, 0.9703746438026428,
       0.8369089365005493,   0.28198742866516113, 0.3741576075553894,
       0.023700952529907227, 0.49101293087005615, 0.12347054481506348,
       0.11432164907455444,  0.4724501967430115,  0.5750725269317627,
       0.2952348589897156,   0.7966887950897217,  0.19573044776916504,
       0.9536850452423096,   0.8426499366760254,  0.07835853099822998,
       0.3755578398704529,   0.5225613117218018,  0.572950541973114,
       0.6185871362686157,   0.6962141394615173,  0.5299500823020935,
       0.25603562593460083,  0.7365944981575012,  0.020375549793243408,
       0.20364665985107422,  0.3748350739479065,  0.2564433217048645});
  Tensor expected_result = tf.make(
      {3, 3, 2},
      {1.6221470832824707,
       1.498693823814392,
       1.224705696105957,
       1.2123372554779053,
       2.1629090309143066,
       2.05692195892334,
       0.9047035574913025,
       1.3324503898620605,
       1.2006582021713257,
       1.5112680196762085,
       1.1946606636047363,
       1.5640640258789062,
       1.405808448791504,
       1.5957869291305542,
       1.3348338603973389,
       1.2967426776885986,
       1.1425018310546875,
       1.2352378368377686});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_bmm_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
