/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_cdist_forward_out(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    optional<int64_t> compute_mode,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::_cdist_forward_outf(
      context, x1, x2, p, compute_mode, out);
}

class OpCdistForwardOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpCdistForwardOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor x1 =
      tfFloat.make({2, 1, 4, 3}, {0,  1, 2, 3, 5,  4, 3, -3, 7, 1, 6,  2,
                                  -1, 5, 1, 1, -2, 1, 5, 4,  3, 2, -1, 5});
  Tensor x2 = tfFloat.make(
      {1, 2, 5, 3}, {0, 1, 2, 3,  5, -3, 7, 1,  6, 2, -1, 5, 1, 1,  -2,
                     4, 3, 2, -1, 5, 1,  1, -2, 1, 5, 4,  3, 2, -1, 5});
  optional<int64_t> compute_mode = optional<int64_t>();

  Tensor out = tfFloat.zeros({2, 2, 4, 5});

  Tensor l0 = tfFloat.make(
      {2, 2, 4, 5},
      {0., 3., 2., 3., 2., 3., 1., 3., 3., 3., 3., 2., 3., 3., 3., 2.,
       3., 3., 3., 2., 2., 3., 3., 3., 3., 3., 2., 3., 3., 3., 3., 3.,
       3., 3., 3., 2., 3., 2., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3.,
       3., 2., 3., 3., 3., 3., 3., 3., 3., 3., 0., 3., 3., 0., 2., 3.,
       3., 3., 2., 0., 3., 3., 3., 3., 3., 0., 3., 3., 3., 3., 3., 0.});
  op_cdist_forward_out(x1, x2, 0.0, compute_mode, out);
  EXPECT_TENSOR_CLOSE(out, l0);

  Tensor l1 = tfFloat.make(
      {2, 2, 4, 5},
      {0.,  12., 11., 7.,  5.,  9.,  7.,  10., 8.,  12., 12., 18., 9.,  5.,
       15., 6.,  8.,  15., 11., 9.,  6.,  6.,  5.,  9.,  7.,  5.,  7.,  12.,
       4.,  8.,  12., 18., 9.,  13., 5.,  6.,  4.,  9.,  7.,  11., 6.,  8.,
       17., 13., 9.,  5.,  13., 14., 6.,  6.,  9.,  9.,  8.,  10., 12., 7.,
       15., 8.,  0.,  10., 8.,  0.,  9.,  9.,  13., 9.,  9.,  0.,  12., 6.,
       3.,  9.,  12., 0.,  10., 9.,  13., 6.,  10., 0.});
  op_cdist_forward_out(x1, x2, 1.0, compute_mode, out);
  EXPECT_TENSOR_CLOSE(out, l1);

  Tensor l2 = tfFloat.make(
      {2, 2, 4, 5},
      {0.00000000, 7.07106781,  8.06225777,  4.12310553, 4.12310553,
       5.38516474, 7.00000000,  6.00000000,  6.16441393, 7.48331499,
       7.07106781, 12.80624866, 5.74456263,  3.00000000, 10.04987526,
       5.09901953, 5.47722578,  8.77496433,  7.68114567, 6.40312433,
       4.47213602, 4.24264050,  3.31662488,  5.91608000, 4.12310553,
       3.00000000, 5.00000000,  7.87400770,  2.44948983, 6.16441393,
       7.87400770, 10.77032948, 6.40312433,  8.30662346, 3.00000000,
       4.24264050, 2.44948983,  8.06225777,  4.58257580, 7.68114567,
       4.24264050, 5.65685415,  10.24695110, 7.81024981, 5.38516474,
       3.31662488, 8.30662346,  8.36660004,  4.24264050, 4.24264050,
       5.91608000, 6.40312433,  4.69041586,  6.16441393, 7.07106781,
       4.12310553, 10.04987526, 5.47722578,  0.00000000, 7.34846926,
       5.47722578, 0.00000000,  7.28010988,  6.40312433, 7.81024981,
       5.91608000, 7.28010988,  0.00000000,  7.48331499, 4.24264050,
       1.73205078, 6.40312433,  7.48331499,  0.00000000, 6.16441393,
       5.38516474, 7.81024981,  4.24264050,  6.16441393, 0.00000000});
  op_cdist_forward_out(x1, x2, 2.0, compute_mode, out);
  EXPECT_TENSOR_CLOSE(out, l2);

  Tensor l3 = tfFloat.make(
      {2, 2, 4, 5},
      {0.00000000, 6.00000000, 7.41079521, 3.50339794, 4.02072573, 4.62606478,
       7.00000000, 5.14256334, 6.01846170, 6.60385466, 6.00000000, 11.47758675,
       5.05277443, 2.57128167, 9.28704357, 5.01329803, 5.11722994, 7.39863634,
       7.18551636, 5.73879337, 4.16016769, 4.04124022, 3.07231688, 5.34848118,
       3.50339794, 2.57128167, 4.49794149, 7.23042679, 2.15443468, 6.01846170,
       6.99319077, 9.25212955, 6.08220196, 7.45903587, 2.57128167, 3.77976322,
       2.15443468, 8.00520515, 4.17933941, 7.18551636, 4.04124022, 5.03968430,
       8.88326645, 6.74599648, 4.62606478, 3.07231688, 7.45903587, 7.16609573,
       4.04124022, 3.77976322, 5.34848118, 6.08220196, 3.95789170, 5.42883539,
       6.00000000, 3.50339794, 9.00000000, 5.11722994, 0.00000000, 7.06069660,
       5.11722994, 0.00000000, 7.05400419, 6.08220196, 6.74599648, 5.34848118,
       7.05400419, 0.00000000, 6.60385466, 4.04124022, 1.44224954, 6.08220196,
       6.60385466, 0.00000000, 5.42883539, 4.62606478, 6.74599648, 4.04124022,
       5.42883539, 0.00000000});
  op_cdist_forward_out(x1, x2, 3.0, compute_mode, out);
  EXPECT_TENSOR_CLOSE(out, l3);

  Tensor linf = tfFloat.make(
      {2, 2, 4, 5},
      {0., 5., 7., 3., 4., 4., 7., 4., 6., 6., 5., 10., 4., 2., 9., 5.,
       5., 6., 7., 5., 4., 4., 3., 5., 3., 2., 4., 7.,  2., 6., 6., 8.,
       6., 7., 2., 3., 2., 8., 4., 7., 4., 4., 8., 6.,  4., 3., 7., 6.,
       4., 3., 5., 6., 3., 5., 5., 3., 8., 5., 0., 7.,  5., 0., 7., 6.,
       6., 5., 7., 0., 6., 4., 1., 6., 6., 0., 5., 4.,  6., 4., 5., 0.});
  op_cdist_forward_out(x1, x2, INFINITY, compute_mode, out);
  EXPECT_TENSOR_CLOSE(out, linf);
}
