/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/UnaryUfuncRealHBToFloatHTest.h>

#include <gtest/gtest.h>

#include <cmath>

using exec_aten::Tensor;
class OpAcosOutTest : public torch::executor::testing::UnaryUfuncRealHBToFloatHTest {
 protected:
  Tensor& op_out(const Tensor& self, Tensor& out) override {
    return torch::executor::aten::acos_outf(context_, self, out);
  }

  double op_reference(double x) const override {
    return std::acos(x);
  }
};

IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST(OpAcosOutTest)
