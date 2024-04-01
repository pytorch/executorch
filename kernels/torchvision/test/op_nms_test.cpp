/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/torchvision/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <limits>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::optional;
using exec_aten::RuntimeContext;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

using torch::executor::native::nms_out;

TEST(OpDequantizeOutTest, NonWholeNumbers) {
  RuntimeContext ctx{};
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;
  Tensor boxes =
      tf_float.make({5, 4}, {0.0, 0.0, 0.5, 0.5, 0.2, 0.2, 0.6, 0.6, 0.4, 0.4,
                             0.9, 0.9, 0.6, 0.6, 0.8, 0.8, 0.1, 0.1, 0.4, 0.4});
  Tensor scores = tf_float.make({5}, {0.9, 0.8, 0.7, 0.6, 0.5});
  double iou_threshold = 0.5;
  Tensor out =
      tf_long.zeros({5}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor expected = tf_long.make({5}, {0, 1, 2, 3, 4});

  nms_out(ctx, boxes, scores, iou_threshold, out);

  EXPECT_TENSOR_EQ(out, expected);

  scores = tf_float.make({5}, {0.3, 0.7, 0.2, 0.6, 0.5});
  expected = tf_long.make({5}, {1, 3, 4, 0, 2});
  nms_out(ctx, boxes, scores, iou_threshold, out);

  EXPECT_TENSOR_EQ(out, expected);

  // Test a case where the output tensor will have to be resized.
  iou_threshold = 0.1;
  expected = tf_long.make({2}, {1, 3});
  nms_out(ctx, boxes, scores, iou_threshold, out);

  EXPECT_TENSOR_EQ(out, expected);
}
