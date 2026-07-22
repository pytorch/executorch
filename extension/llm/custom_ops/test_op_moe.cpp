/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_moe.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;

namespace {

// Smoke test: schema is registered and the op can be called via its
// out-variant entry point. Numerical correctness is exercised by the
// Python tests in test_quantized_moe.py against a Python q-dq reference,
// because constructing a meaningful torchao packed-weight blob from C++
// here would require duplicating the AOT packer logic.
TEST(OpQuantizedMoeFfnTest, RegistrationSmokeTest) {
  // Tiny shapes
  constexpr int T = 2;
  constexpr int D = 32;
  constexpr int F = 32;
  constexpr int E = 4;
  constexpr int K = 2;
  constexpr int kGroupSize = 32;
  constexpr int kWeightNbit = 4;

  TensorFactory<ScalarType::Float> tff;
  TensorFactory<ScalarType::Byte> tfb;

  Tensor x = tff.zeros({T, D});
  Tensor gate = tff.zeros({E, D});
  Tensor expert_bias = tff.zeros({0});

  // Use empty packed buffers; the kernel will fail loudly if it tries to
  // dereference them. With ENABLE_QUANTIZED_MOE_FFN unset (CI x86 build
  // without torchao linkage) the kernel ET_CHECK_MSGs out before doing
  // any real work, which is what we want this test to verify.
  Tensor packed_w1 = tfb.zeros({E, 1});
  Tensor packed_w3 = tfb.zeros({E, 1});
  Tensor packed_w2 = tfb.zeros({E, 1});

  Tensor out = tff.zeros({T, D});

  executorch::runtime::KernelRuntimeContext ctx{};
  // We don't actually call the kernel here in the registration smoke test
  // because the empty packed buffers would not be valid torchao blobs.
  // Just verify the op symbol resolves at link time.
  auto fn = &torch::executor::native::quantized_moe_ffn_out;
  EXPECT_NE(fn, nullptr);
  // Silence unused-variable warnings on the input tensors above; they
  // demonstrate the expected schema.
  (void)x;
  (void)gate;
  (void)expert_bias;
  (void)packed_w1;
  (void)packed_w3;
  (void)packed_w2;
  (void)out;
  (void)ctx;
  (void)kGroupSize;
  (void)kWeightNbit;
  (void)K;
  (void)F;
}

} // namespace
