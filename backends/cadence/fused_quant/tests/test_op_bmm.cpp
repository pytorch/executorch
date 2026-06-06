/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_bmm.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

using executorch::aten::optional;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;

namespace {

optional<Tensor> none_tensor() {
  return optional<Tensor>();
}

optional<int64_t> none_axis() {
  return optional<int64_t>();
}

} // namespace

class FusedQuantBmmTest : public OperatorTest {};

// All quantized: int8 × int8 → int8 (per-tensor)
TEST_F(FusedQuantBmmTest, AllQuantizedPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2,2]: identity matrix {{1,0},{0,1}} quantized as int8
  // other [1,2,2]: {{1,2},{3,4}} quantized as int8
  const std::vector<int> inp_sizes{1, 2, 2};
  const std::vector<int> other_sizes{1, 2, 2};
  const std::vector<int> out_sizes{1, 2, 2};

  // scale=0.5, zp=0: int8 value v maps to v * 0.5
  // identity: {1,0,0,1} -> int8 {2,0,0,2}
  Tensor inp = tf_int8.make(inp_sizes, {2, 0, 0, 2});
  // {{1,2},{3,4}} -> int8 {2,4,6,8}
  Tensor other = tf_int8.make(other_sizes, {2, 4, 6, 8});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(out_sizes);

  // dequant inp: {{1,0},{0,1}}
  // dequant other: {{1,2},{3,4}}
  // bmm: I * {{1,2},{3,4}} = {{1,2},{3,4}}
  // requant (scale=0.5, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(out_sizes, {2, 4, 6, 8}));
}

// float × float → int8
TEST_F(FusedQuantBmmTest, FloatInputsQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> inp_sizes{1, 2, 2};
  const std::vector<int> other_sizes{1, 2, 2};
  const std::vector<int> out_sizes{1, 2, 2};

  // identity
  Tensor inp = tf_float.make(inp_sizes, {1.0, 0.0, 0.0, 1.0});
  Tensor other = tf_float.make(other_sizes, {1.0, 2.0, 3.0, 4.0});

  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(out_sizes);

  // bmm: I * {{1,2},{3,4}} = {{1,2},{3,4}}
  // requant (scale=0.5, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(out_sizes, {2, 4, 6, 8}));
}

// int8 × int8 → float
TEST_F(FusedQuantBmmTest, QuantizedInputsFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> inp_sizes{1, 2, 2};
  const std::vector<int> other_sizes{1, 2, 2};
  const std::vector<int> out_sizes{1, 2, 2};

  Tensor inp = tf_int8.make(inp_sizes, {2, 0, 0, 2});
  Tensor other = tf_int8.make(other_sizes, {2, 4, 6, 8});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(out_sizes);

  // dequant inp: {{1,0},{0,1}}
  // dequant other: {{1,2},{3,4}}
  // bmm: {{1,2},{3,4}}
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make(out_sizes, {1.0, 2.0, 3.0, 4.0}));
}

// int8 × float → int8
TEST_F(FusedQuantBmmTest, QuantizedInpFloatOther) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> inp_sizes{1, 2, 2};
  const std::vector<int> other_sizes{1, 2, 2};
  const std::vector<int> out_sizes{1, 2, 2};

  Tensor inp = tf_int8.make(inp_sizes, {2, 0, 0, 2});
  Tensor other = tf_float.make(other_sizes, {1.0, 2.0, 3.0, 4.0});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(out_sizes);

  // dequant inp: {{1,0},{0,1}}
  // bmm: I * {{1,2},{3,4}} = {{1,2},{3,4}}
  // requant (scale=0.5, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(out_sizes, {2, 4, 6, 8}));
}

// Non-zero zero_point
TEST_F(FusedQuantBmmTest, NonZeroZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> inp_sizes{1, 2, 2};
  const std::vector<int> other_sizes{1, 2, 2};
  const std::vector<int> out_sizes{1, 2, 2};

  // scale=0.25, zp=2: int8 value v maps to (v - 2) * 0.25
  // inp: {{1,0.5},{0.5,1}} -> int8: (1/0.25)+2=6, (0.5/0.25)+2=4, 4, 6
  Tensor inp = tf_int8.make(inp_sizes, {6, 4, 4, 6});
  // other: {{1,2},{0,1}} -> int8: (1/0.25)+2=6, (2/0.25)+2=10, (0/0.25)+2=2,
  // (1/0.25)+2=6
  Tensor other = tf_int8.make(other_sizes, {6, 10, 2, 6});

  Tensor inp_scale = tf_float.make({1}, {0.25});
  Tensor inp_zp = tf_long.make({1}, {2});
  Tensor other_scale = tf_float.make({1}, {0.25});
  Tensor other_zp = tf_long.make({1}, {2});
  // out: scale=0.5, zp=1 -> float f maps to round(f / 0.5) + 1
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {1});

  Tensor out = tf_int8.zeros(out_sizes);

  // dequant inp: (6-2)*0.25=1, (4-2)*0.25=0.5, (4-2)*0.25=0.5, (6-2)*0.25=1
  //   -> {{1, 0.5}, {0.5, 1}}
  // dequant other: (6-2)*0.25=1, (10-2)*0.25=2, (2-2)*0.25=0, (6-2)*0.25=1
  //   -> {{1, 2}, {0, 1}}
  // bmm: {{1*1+0.5*0, 1*2+0.5*1}, {0.5*1+1*0, 0.5*2+1*1}}
  //    = {{1, 2.5}, {0.5, 2}}
  // requant (scale=0.5, zp=1):
  //   round(1/0.5)+1=3, round(2.5/0.5)+1=6,
  //   round(0.5/0.5)+1=2, round(2/0.5)+1=5
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(out_sizes, {3, 6, 2, 5}));
}

// batch=2, verify both batch elements
TEST_F(FusedQuantBmmTest, LargerBatch) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [2,2,2]: two identity matrices
  // other [2,2,2]: batch 0 = {{1,2},{3,4}}, batch 1 = {{5,6},{7,8}}
  const std::vector<int> inp_sizes{2, 2, 2};
  const std::vector<int> other_sizes{2, 2, 2};
  const std::vector<int> out_sizes{2, 2, 2};

  // scale=0.5, zp=0: two identity matrices as int8
  Tensor inp = tf_int8.make(inp_sizes, {2, 0, 0, 2, 2, 0, 0, 2});
  Tensor other = tf_int8.make(other_sizes, {2, 4, 6, 8, 10, 12, 14, 16});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(out_sizes);

  // dequant inp: two identity matrices {{1,0},{0,1}}, {{1,0},{0,1}}
  // dequant other: {{1,2},{3,4}}, {{5,6},{7,8}}
  // bmm batch 0: I * {{1,2},{3,4}} = {{1,2},{3,4}}
  // bmm batch 1: I * {{5,6},{7,8}} = {{5,6},{7,8}}
  // requant (scale=0.5, zp=0): {2,4,6,8, 10,12,14,16}
  cadence::fused_quant::native::bmm_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(out_sizes, {2, 4, 6, 8, 10, 12, 14, 16}));
}
