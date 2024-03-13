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
#include <limits>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpAddmmOutTest : public OperatorTest {
 protected:
  Tensor& op_addmm_out(
      const Tensor& self,
      const Tensor& mat1,
      const Tensor& mat2,
      const Scalar& beta,
      const Scalar& alpha,
      Tensor& out) {
    return torch::executor::aten::addmm_outf(
        context_, self, mat1, mat2, beta, alpha, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
      if (DTYPE == ScalarType::Half) {
        GTEST_SKIP()
            << "skip Half because torch::executor::aten::mm_out does not support Half";
        return;
      }
    }

    // matmul gives 4 * 2 * 3 = 24, α * 24 = 48, 48 + β * self = 51
    Tensor self = tf.full({3, 5}, 1);
    Tensor x = tf.full({3, 4}, 2);
    Tensor y = tf.full({4, 5}, 3);

    // Output shape should be (3, 5)
    Tensor out = tf.zeros({3, 5});

    Scalar alpha = Scalar(2.0);
    Scalar beta = Scalar(3.0);

    op_addmm_out(self, x, y, beta, alpha, out);

    Tensor expected = tf.full({3, 5}, 51);

    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpAddmmOutTest, OutputDim) {
  TensorFactory<ScalarType::Int> tf;

  // 3 tensors with compatible dimensions: (3, 5), (3, 4) and (4, 5).
  Tensor self = tf.ones({3, 5});
  Tensor x = tf.ones({3, 4});
  Tensor y = tf.ones({4, 5});

  // Output shape should be (3, 5)
  Tensor out = tf.zeros({3, 5});

  Scalar alpha = Scalar(1);
  Scalar beta = Scalar(1);

  Tensor ret = op_addmm_out(self, x, y, beta, alpha, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, filled with 5.
  Tensor expected = tf.full({3, 5}, 5);

  EXPECT_TENSOR_EQ(out, expected);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpAddmmOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Half, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpAddmmOutTest, EmptyInputWithEmptyOutTensorPasses) {
  TensorFactory<ScalarType::Float> tf;

  // Empty input matrices
  Tensor self = tf.make({0, 0}, {});
  Tensor x = tf.make({0, 3}, {});
  Tensor y = tf.make({3, 0}, {});

  // Output matrix is also empty
  Tensor out = tf.make({0, 0}, {});

  Tensor expected = tf.make({0, 0}, {});

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(2), Scalar(3), out), expected);
}

TEST_F(OpAddmmOutTest, FloatTensorDtypeAndIntScalarTypePasses) {
  // case 1: Tensor dtype float, scalar type int
  TensorFactory<ScalarType::Float> tff;
  // matmul gives 4 * 2 * 3 = 24, α * 24 = 72, 72 + β * self = 74
  Tensor self = tff.full({3, 5}, 1);
  Tensor x = tff.full({3, 4}, 2);
  Tensor y = tff.full({4, 5}, 3);

  // Output shape should be (3, 5)
  Tensor out = tff.zeros({3, 5});

  Tensor expected = tff.full({3, 5}, 74);

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(2), Scalar(3), out), expected);
}

TEST_F(OpAddmmOutTest, IntTensorDtypeAndFloatScalarTypePasses) {
  // case 2: Tensor dtype int, scalar type loat
  TensorFactory<ScalarType::Int> tfi;
  // matmul gives 4 * 2 * 3 = 24, α * 24 = 72, 72 + β * self = 74
  Tensor self = tfi.full({3, 5}, 1);
  Tensor x = tfi.full({3, 4}, 2);
  Tensor y = tfi.full({4, 5}, 3);

  // Output shape should be (3, 5)
  Tensor out = tfi.zeros({3, 5});

  Tensor expected = tfi.full({3, 5}, 74);

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(2.0), Scalar(3.0), out), expected);
}

TEST_F(OpAddmmOutTest, InfinityTensorAndFloatScalarTypePasses) {
  // case 2: Tensor dtype int, scalar type loat
  TensorFactory<ScalarType::Float> tff;

  Tensor self = tff.full({3, 5}, std::numeric_limits<float>::infinity());
  Tensor x = tff.full({3, 4}, 2);
  Tensor y = tff.full({4, 5}, 3);

  // Output shape should be (3, 5)
  Tensor out = tff.zeros({3, 5});

  Tensor expected = tff.full({3, 5}, std::numeric_limits<float>::infinity());

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(2), Scalar(3), out), expected);
}

TEST_F(OpAddmmOutTest, MismatchedDimensionsDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor self = tf.full({2, 2}, 3);
  Tensor x = tf.full({2, 2}, 3);

  Tensor wrong_y = tf.full({3, 1}, 1);
  Tensor right_y = tf.full({2, 2}, 1);

  // Make an empty out tensor and demonstrate that it's empty.
  Tensor out = tf.full({2, 2}, 0);

  Tensor expected = tf.full({2, 2}, 9);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_addmm_out(self, x, wrong_y, Scalar(1), Scalar(1), out));

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, right_y, Scalar(1), Scalar(1), out), expected);
}

TEST_F(OpAddmmOutTest, MismatchedDimensionSizeDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.full({2, 2}, 3);
  Tensor x = tf.full({2, 2}, 3);

  // wrong_y has incompatible dim
  Tensor wrong_y = tf.full({2, 2, 2}, 1);
  Tensor right_y = tf.full({2, 2}, 1);

  // wrong_out has incompatible dim
  Tensor right_out = tf.ones({2, 2});
  Tensor wrong_out = tf.ones({2, 2, 3});

  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_addmm_out(self, x, right_y, Scalar(1), Scalar(1), wrong_out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_addmm_out(self, x, wrong_y, Scalar(1), Scalar(1), right_out));
}

TEST_F(OpAddmmOutTest, WrongOutShapeDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.ones({10, 4});
  Tensor x = tf.ones({10, 3});

  Tensor y = tf.ones({3, 4});

  // wrong_out has incompatible shape
  Tensor right_out = tf.ones({10, 4});
  Tensor wrong_out = tf.ones({7, 5});

  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_addmm_out(self, x, y, Scalar(1), Scalar(1), wrong_out));

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(1), Scalar(1), right_out),
      tf.full({10, 4}, 4));
}

TEST_F(OpAddmmOutTest, BroadcastTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor self = tf.make({1}, {1});
  Tensor x = tf.make({2, 2}, {1, 2, 3, 4});
  Tensor y = tf.make({2, 2}, {1, 2, 3, 4});

  Tensor out = tf.make({2, 2}, {0, 0, 0, 0});

  EXPECT_TENSOR_EQ(
      op_addmm_out(self, x, y, Scalar(1), Scalar(1), out),
      tf.make({2, 2}, {8, 11, 16, 23}));
}
TEST_F(OpAddmmOutTest, BroadcastDimSize1) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.9937992691993713, 0.7011417150497437});
  Tensor y = tf.make(
      {3, 6},
      {0.3271445035934448,
       0.4104803800582886,
       0.26973772048950195,
       0.29142987728118896,
       0.20096111297607422,
       0.7686975002288818,
       0.07416731119155884,
       0.276896595954895,
       0.43525755405426025,
       0.8261672854423523,
       0.22888076305389404,
       0.042113542556762695,
       0.8771350979804993,
       0.4088439345359802,
       0.0258103609085083,
       0.26305103302001953,
       0.6766068339347839,
       0.3576545715332031});
  Tensor z = tf.make(
      {6, 2},
      {0.5702318549156189,
       0.8886868953704834,
       0.8667161464691162,
       0.7151150107383728,
       0.19591552019119263,
       0.7918031811714172,
       0.8956874012947083,
       0.7162176966667175,
       0.34151601791381836,
       0.16078311204910278,
       0.6722156405448914,
       0.048251569271087646});
  Tensor expected_result = tf.make(
      {3, 2},
      {2.4353551864624023,
       1.7771198749542236,
       2.207819700241089,
       1.9402521848678589,
       2.5604825019836426,
       2.107893466949463});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpAddmmOutTest, BroadcastDimSizeMissing) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2}, {0.9937992691993713, 0.7011417150497437});
  Tensor y = tf.make(
      {3, 6},
      {0.3271445035934448,
       0.4104803800582886,
       0.26973772048950195,
       0.29142987728118896,
       0.20096111297607422,
       0.7686975002288818,
       0.07416731119155884,
       0.276896595954895,
       0.43525755405426025,
       0.8261672854423523,
       0.22888076305389404,
       0.042113542556762695,
       0.8771350979804993,
       0.4088439345359802,
       0.0258103609085083,
       0.26305103302001953,
       0.6766068339347839,
       0.3576545715332031});
  Tensor z = tf.make(
      {6, 2},
      {0.5702318549156189,
       0.8886868953704834,
       0.8667161464691162,
       0.7151150107383728,
       0.19591552019119263,
       0.7918031811714172,
       0.8956874012947083,
       0.7162176966667175,
       0.34151601791381836,
       0.16078311204910278,
       0.6722156405448914,
       0.048251569271087646});
  Tensor expected_result = tf.make(
      {3, 2},
      {2.4353551864624023,
       1.7771198749542236,
       2.207819700241089,
       1.9402521848678589,
       2.5604825019836426,
       2.107893466949463});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpAddmmOutTest, BroadcastDimSizeIsOne) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.9093303680419922, 0.37621551752090454});
  Tensor y = tf.make(
      {3, 6},
      {0.5741164088249207,
       0.3001101613044739,
       0.6543494462966919,
       0.8815506100654602,
       0.8948686122894287,
       0.3319156765937805,
       0.6683467030525208,
       0.37235790491104126,
       0.15439540147781372,
       0.05733710527420044,
       0.5467379093170166,
       0.9564069509506226,
       0.2915573716163635,
       0.5548340082168579,
       0.20116734504699707,
       0.8199875950813293,
       0.270835816860199,
       0.1414813995361328});
  Tensor z = tf.make(
      {6, 2},
      {0.6883938312530518,
       0.9387704133987427,
       0.6991894841194153,
       0.2945629954338074,
       0.48106586933135986,
       0.932110607624054,
       0.9461215138435364,
       0.7682468295097351,
       0.6223915219306946,
       0.0702824592590332,
       0.9750580787658691,
       0.05068659782409668});
  Tensor expected_result = tf.make(
      {3, 2},
      {3.5438172817230225,
       2.3704721927642822,
       3.0311243534088135,
       1.388188123703003,
       2.6770718097686768,
       1.6570236682891846});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpAddmmOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.5024666786193848,
       0.8311734795570374,
       0.17922323942184448,
       0.5711425542831421,
       0.23492926359176636,
       0.6693081259727478});
  Tensor y = tf.make(
      {3, 6},
      {0.8927820920944214,
       0.13490021228790283,
       0.49518370628356934,
       0.027777791023254395,
       0.7909245491027832,
       0.07999932765960693,
       0.9496669173240662,
       0.18807870149612427,
       0.44375330209732056,
       0.761903703212738,
       0.24175149202346802,
       0.31033122539520264,
       0.8609206080436707,
       0.1580638885498047,
       0.2585788369178772,
       0.4787442088127136,
       0.17180007696151733,
       0.2109091877937317});
  Tensor z = tf.make(
      {6, 2},
      {0.06361657381057739,
       0.8065286874771118,
       0.610871434211731,
       0.19808048009872437,
       0.7010428309440613,
       0.904334545135498,
       0.8460395932197571,
       0.34137529134750366,
       0.4836529493331909,
       0.2751874327659607,
       0.22036516666412354,
       0.742312490940094});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.4124772548675537,
       2.3122801780700684,
       1.495530605316162,
       2.3326172828674316,
       1.1021348237991333,
       1.9960856437683105});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpAddmmOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.5024666786193848,
       0.8311734795570374,
       0.17922323942184448,
       0.5711425542831421,
       0.23492926359176636,
       0.6693081259727478});
  Tensor y = tf.make(
      {3, 6},
      {0.8927820920944214,
       0.13490021228790283,
       0.49518370628356934,
       0.027777791023254395,
       0.7909245491027832,
       0.07999932765960693,
       0.9496669173240662,
       0.18807870149612427,
       0.44375330209732056,
       0.761903703212738,
       0.24175149202346802,
       0.31033122539520264,
       0.8609206080436707,
       0.1580638885498047,
       0.2585788369178772,
       0.4787442088127136,
       0.17180007696151733,
       0.2109091877937317});
  Tensor z = tf.make(
      {6, 2},
      {0.06361657381057739,
       0.8065286874771118,
       0.610871434211731,
       0.19808048009872437,
       0.7010428309440613,
       0.904334545135498,
       0.8460395932197571,
       0.34137529134750366,
       0.4836529493331909,
       0.2751874327659607,
       0.22036516666412354,
       0.742312490940094});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.4124772548675537,
       2.3122801780700684,
       1.495530605316162,
       2.3326172828674316,
       1.1021348237991333,
       1.9960856437683105});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpAddmmOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.754013180732727,
       0.16418755054473877,
       0.8077310919761658,
       0.7187556624412537,
       0.0470539927482605,
       0.2438456416130066});
  Tensor y = tf.make(
      {3, 6},
      {0.5899912118911743,
       0.5052928328514099,
       0.13990312814712524,
       0.22438400983810425,
       0.1697748899459839,
       0.6022286415100098,
       0.08701932430267334,
       0.7246091961860657,
       0.44388288259506226,
       0.9451560974121094,
       0.8658323884010315,
       0.781434953212738,
       0.02855396270751953,
       0.49756181240081787,
       0.506054699420929,
       0.12560266256332397,
       0.7099084854125977,
       0.04813879728317261});
  Tensor z = tf.make(
      {6, 2},
      {0.19827371835708618,
       0.486919641494751,
       0.7659645080566406,
       0.7863746285438538,
       0.032599568367004395,
       0.8414170145988464,
       0.7014893293380737,
       0.2445545196533203,
       0.07429623603820801,
       0.12777382135391235,
       0.39169949293136597,
       0.80079185962677});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.6684993505477905,
       1.5253589153289795,
       2.427912712097168,
       2.6719717979431152,
       0.6100357174873352,
       1.2347958087921143});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_addmm_out(x, y, z, Scalar(1), Scalar(1), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
