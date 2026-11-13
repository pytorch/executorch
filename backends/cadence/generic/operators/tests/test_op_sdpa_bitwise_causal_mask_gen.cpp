/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_causal_mask_gen.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's. Each row packs key_length
// bits LSB first, 8 keys per byte; a set bit masks a key, so position p leaves
// keys 0..p visible.
class GenericSdpaBitwiseCausalMaskGenTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Byte> tf_byte_;
  TensorFactory<ScalarType::Int> tf_int_;
  TensorFactory<ScalarType::Long> tf_long_;
};

// Position 20 is past the 16 keys, so its row shows them all.
TEST_F(GenericSdpaBitwiseCausalMaskGenTest, Int32Positions) {
  Tensor out = tf_byte_.zeros({4, 2});

  sdpa_bitwise_causal_mask_gen_out(
      context_, tf_int_.make({4}, {0, 3, 9, 20}), 16, out);

  EXPECT_TENSOR_EQ(
      out, tf_byte_.make({4, 2}, {254, 255, 240, 255, 0, 252, 0, 0}));
}

// Positions 7 and 8 straddle the byte boundary.
TEST_F(GenericSdpaBitwiseCausalMaskGenTest, Int64Positions) {
  Tensor out = tf_byte_.zeros({2, 2});

  sdpa_bitwise_causal_mask_gen_out(
      context_, tf_long_.make({2}, {7, 8}), 16, out);

  EXPECT_TENSOR_EQ(out, tf_byte_.make({2, 2}, {0, 255, 0, 254}));
}

TEST_F(GenericSdpaBitwiseCausalMaskGenTest, RejectsKeyLengthNotAMultipleOf8) {
  Tensor out = tf_byte_.zeros({1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      sdpa_bitwise_causal_mask_gen_out(
          context_, tf_int_.make({1}, {0}), 12, out));
}

TEST_F(GenericSdpaBitwiseCausalMaskGenTest, RejectsANegativePosition) {
  Tensor out = tf_byte_.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      sdpa_bitwise_causal_mask_gen_out(
          context_, tf_int_.make({2}, {1, -1}), 16, out));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
