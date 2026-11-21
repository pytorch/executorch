/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_sdpa_row_exact.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::StridesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::testing::TensorFactory;
using torch::executor::native::SdpaRowExactScope;

namespace {

// 4 query heads share 2 KV heads (GQA).
constexpr int32_t kHeads = 4;
constexpr int32_t kKvHeads = 2;
constexpr size_t kAlignment = 64;

struct Case {
  int32_t head_dim;
  int32_t cache_len;
  int32_t rows;
  int64_t start_pos;
};

// Small cases, plus model-sized heads over a cache longer than one 512-key
// block.
const std::vector<Case> kCases = {
    {16, 96, 2, 0},
    {16, 96, 5, 7},
    {16, 96, 16, 33},
    {64, 700, 16, 640},
    {128, 700, 3, 500},
};

template <typename T>
std::vector<T> random_values(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<T> values(n);
  for (auto& value : values) {
    value = static_cast<T>(dist(gen));
  }
  return values;
}

// A [1, rows, kHeads, width] tensor in 64-byte aligned storage. The expected
// result of row r comes from a second buffer with the same layout, so both
// calls see row r at the same address alignment: on x86-64 Linux the float
// path goes through MKL, whose results depend on buffer alignment.
template <typename T>
class RowsBuffer {
 public:
  RowsBuffer(int32_t rows, int32_t width, const std::vector<T>& values)
      : rows_(rows), width_(width), data_(allocate(values.size())) {
    std::memcpy(data_.get(), values.data(), values.size() * sizeof(T));
  }

  Tensor all() {
    return view(0, rows_);
  }
  Tensor row(int32_t r) {
    return view(r, 1);
  }
  T* data() {
    return data_.get();
  }
  const T* row_data(int32_t r) const {
    return data_.get() + r * kHeads * width_;
  }
  size_t row_bytes() const {
    return kHeads * width_ * sizeof(T);
  }

 private:
  struct View {
    std::array<SizesType, 4> sizes;
    std::array<StridesType, 4> strides;
    std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
    std::optional<TensorImpl> impl;
  };

  static std::unique_ptr<T, decltype(&std::free)> allocate(size_t n) {
    const size_t bytes =
        (n * sizeof(T) + kAlignment - 1) / kAlignment * kAlignment;
    return {static_cast<T*>(std::aligned_alloc(kAlignment, bytes)), &std::free};
  }

  Tensor view(int32_t first, int32_t n) {
    View& v = views_.emplace_back();
    v.sizes = {1, n, kHeads, width_};
    v.strides = {n * kHeads * width_, kHeads * width_, width_, 1};
    v.impl.emplace(
        executorch::runtime::CppTypeToScalarType<T>::value,
        4,
        v.sizes.data(),
        data_.get() + first * kHeads * width_,
        v.dim_order.data(),
        v.strides.data());
    return Tensor(&*v.impl);
  }

  int32_t rows_;
  int32_t width_;
  std::unique_ptr<T, decltype(&std::free)> data_;
  std::deque<View> views_;
};

// The query side of one call: q and, for int8, its quantization parameters.
template <typename T>
struct Query {
  RowsBuffer<T> q;
  std::optional<RowsBuffer<int8_t>> zero_points;
  std::optional<RowsBuffer<float>> scales;

  std::optional<Tensor> zero_points_of(std::optional<int32_t> row) {
    if (!zero_points) {
      return std::nullopt;
    }
    return row ? zero_points->row(*row) : zero_points->all();
  }
  std::optional<Tensor> scales_of(std::optional<int32_t> row) {
    if (!scales) {
      return std::nullopt;
    }
    return row ? scales->row(*row) : scales->all();
  }
};

// Runs `call(q, zero_points, scales, start_pos, out)` once on all rows in
// row-exact mode, and once per row on a second, identical set of buffers in
// the default mode, and compares the bytes of every row.
template <typename T, typename OutT, typename MakeQuery, typename Call>
void expect_rows_match_one_row_calls(
    const Case& c,
    MakeQuery make_query,
    Call call) {
  const std::vector<OutT> zeros(c.rows * kHeads * c.head_dim);
  Query<T> all_query = make_query();
  RowsBuffer<OutT> all_out(c.rows, c.head_dim, zeros);
  {
    const SdpaRowExactScope row_exact;
    call(
        all_query.q.all(),
        all_query.zero_points_of(std::nullopt),
        all_query.scales_of(std::nullopt),
        c.start_pos,
        all_out.all());
  }
  Query<T> row_query = make_query();
  RowsBuffer<OutT> row_out(c.rows, c.head_dim, zeros);
  for (int32_t row = 0; row < c.rows; ++row) {
    call(
        row_query.q.row(row),
        row_query.zero_points_of(row),
        row_query.scales_of(row),
        c.start_pos + row,
        row_out.row(row));
    EXPECT_EQ(
        std::memcmp(
            all_out.row_data(row), row_out.row_data(row), all_out.row_bytes()),
        0)
        << "head_dim " << c.head_dim << " rows " << c.rows << " row " << row;
  }
}

template <ScalarType DTYPE>
void expect_float_rows_match(const Case& c) {
  using T = typename TensorFactory<DTYPE>::ctype;
  TensorFactory<DTYPE> tf;
  const std::vector<int32_t> kv_sizes = {1, c.cache_len, kKvHeads, c.head_dim};
  const int32_t kv_numel = c.cache_len * kKvHeads * c.head_dim;
  const Tensor k = tf.make(kv_sizes, random_values<T>(kv_numel, 2, -1, 1));
  const Tensor v = tf.make(kv_sizes, random_values<T>(kv_numel, 3, -1, 1));
  const auto q = random_values<T>(c.rows * kHeads * c.head_dim, 1, -1, 1);
  expect_rows_match_one_row_calls<T, T>(
      c,
      [&] { return Query<T>{RowsBuffer<T>(c.rows, c.head_dim, q)}; },
      [&](Tensor q_in, auto, auto, int64_t pos, Tensor out) {
        KernelRuntimeContext ctx{};
        torch::executor::native::custom_sdpa_out(
            ctx, q_in, k, v, pos, {}, 0.0, true, {}, out);
        ASSERT_EQ(ctx.failure_state(), Error::Ok);
      });
}

void expect_int8_rows_match(const Case& c) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  const int32_t kv_rows = c.cache_len * kKvHeads;
  const std::vector<int32_t> kv_sizes = {1, c.cache_len, kKvHeads, c.head_dim};
  const std::vector<int32_t> kv_param_sizes = {1, c.cache_len, kKvHeads, 1};
  const Tensor k = tf_int8.make(
      kv_sizes, random_values<int8_t>(kv_rows * c.head_dim, 7, -127, 127));
  const Tensor v = tf_int8.make(
      kv_sizes, random_values<int8_t>(kv_rows * c.head_dim, 8, -127, 127));
  const Tensor k_zero_points =
      tf_int8.make(kv_param_sizes, random_values<int8_t>(kv_rows, 9, -8, 8));
  const Tensor k_scales = tf_float.make(
      kv_param_sizes, random_values<float>(kv_rows, 10, 0.005, 0.05));
  const Tensor v_zero_points =
      tf_int8.make(kv_param_sizes, random_values<int8_t>(kv_rows, 11, -8, 8));
  const Tensor v_scales = tf_float.make(
      kv_param_sizes, random_values<float>(kv_rows, 12, 0.005, 0.05));
  const auto q =
      random_values<int8_t>(c.rows * kHeads * c.head_dim, 4, -127, 127);
  const auto q_zero_points = random_values<int8_t>(c.rows * kHeads, 5, -8, 8);
  const auto q_scales = random_values<float>(c.rows * kHeads, 6, 0.005, 0.05);
  expect_rows_match_one_row_calls<int8_t, float>(
      c,
      [&] {
        return Query<int8_t>{
            RowsBuffer<int8_t>(c.rows, c.head_dim, q),
            RowsBuffer<int8_t>(c.rows, 1, q_zero_points),
            RowsBuffer<float>(c.rows, 1, q_scales)};
      },
      [&](Tensor q_in,
          std::optional<Tensor> zero_points,
          std::optional<Tensor> scales,
          int64_t pos,
          Tensor out) {
        KernelRuntimeContext ctx{};
        torch::executor::native::custom_quantized_sdpa_out(
            ctx,
            q_in,
            k,
            v,
            pos,
            {},
            0.0,
            true,
            {},
            zero_points,
            scales,
            k_zero_points,
            k_scales,
            v_zero_points,
            v_scales,
            false,
            out);
        ASSERT_EQ(ctx.failure_state(), Error::Ok);
      });
}

} // namespace

TEST(OpSdpaRowExactTest, ScopeTurnsTheModeOnAndRestoresIt) {
  EXPECT_FALSE(torch::executor::native::sdpa_row_exact_enabled());
  {
    const SdpaRowExactScope outer;
    {
      const SdpaRowExactScope inner(false);
      EXPECT_FALSE(torch::executor::native::sdpa_row_exact_enabled());
    }
    EXPECT_TRUE(torch::executor::native::sdpa_row_exact_enabled());
  }
  EXPECT_FALSE(torch::executor::native::sdpa_row_exact_enabled());
}

TEST(OpSdpaRowExactTest, FloatRowsMatchOneRowCalls) {
  for (const Case& c : kCases) {
    expect_float_rows_match<ScalarType::Float>(c);
  }
}

TEST(OpSdpaRowExactTest, HalfAndBFloat16RowsMatchOneRowCalls) {
  for (const Case& c : kCases) {
    expect_float_rows_match<ScalarType::Half>(c);
    expect_float_rows_match<ScalarType::BFloat16>(c);
  }
}

// The default path ignores q quantization parameters for a float query; so
// must row-exact mode, even when they are incomplete.
TEST(OpSdpaRowExactTest, FloatQueryIgnoresQuantParams) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Char> tf_int8;
  const Case c = kCases[1];
  const std::vector<int32_t> kv_sizes = {1, c.cache_len, kKvHeads, c.head_dim};
  const int32_t kv_numel = c.cache_len * kKvHeads * c.head_dim;
  const Tensor k = tf.make(kv_sizes, random_values<float>(kv_numel, 2, -1, 1));
  const Tensor v = tf.make(kv_sizes, random_values<float>(kv_numel, 3, -1, 1));
  const Tensor unused = tf.zeros({1, c.cache_len, kKvHeads, 1});
  const auto q = random_values<float>(c.rows * kHeads * c.head_dim, 1, -1, 1);
  const std::vector<int8_t> zero_points(c.rows * kHeads);
  expect_rows_match_one_row_calls<float, float>(
      c,
      [&] {
        return Query<float>{
            RowsBuffer<float>(c.rows, c.head_dim, q),
            RowsBuffer<int8_t>(c.rows, 1, zero_points)};
      },
      [&](Tensor q_in,
          std::optional<Tensor> q_zero_points,
          auto,
          int64_t pos,
          Tensor out) {
        KernelRuntimeContext ctx{};
        torch::executor::native::custom_quantized_sdpa_out(
            ctx,
            q_in,
            k,
            v,
            pos,
            {},
            0.0,
            true,
            {},
            q_zero_points,
            {},
            unused,
            unused,
            unused,
            unused,
            false,
            out);
        ASSERT_EQ(ctx.failure_state(), Error::Ok);
      });
}

// With the sequence at dim 2, a float query keeps the default path in
// row-exact mode, also when its last rows reach the end of the cache.
TEST(OpSdpaRowExactTest, FloatQueryWithSeqAtDim2KeepsTheDefaultPath) {
  TensorFactory<ScalarType::Float> tf;
  constexpr int32_t kHeadDim = 16;
  constexpr int32_t kCacheLen = 96;
  constexpr int32_t kRows = 5;
  // The default path covers start_pos + kHeads keys in this layout.
  constexpr int64_t kStartPos = kCacheLen - kHeads;
  constexpr bool kSeqAtDim2 = true;
  const int32_t kv_numel = kKvHeads * kCacheLen * kHeadDim;
  const Tensor k = tf.make(
      {1, kKvHeads, kCacheLen, kHeadDim},
      random_values<float>(kv_numel, 2, -1, 1));
  const Tensor v = tf.make(
      {1, kKvHeads, kCacheLen, kHeadDim},
      random_values<float>(kv_numel, 3, -1, 1));
  const auto q_values =
      random_values<float>(kHeads * kRows * kHeadDim, 1, -1, 1);
  const auto run = [&](bool row_exact) {
    // Both runs see 64-byte aligned query and output storage (MKL results
    // depend on alignment on x86-64 Linux).
    RowsBuffer<float> q_buf(kRows, kHeadDim, q_values);
    RowsBuffer<float> out_buf(
        kRows, kHeadDim, std::vector<float>(q_values.size()));
    std::array<SizesType, 4> sizes = {1, kHeads, kRows, kHeadDim};
    std::array<StridesType, 4> strides = {
        kHeads * kRows * kHeadDim, kRows * kHeadDim, kHeadDim, 1};
    std::array<DimOrderType, 4> dim_order = {0, 1, 2, 3};
    TensorImpl q_impl(
        ScalarType::Float,
        4,
        sizes.data(),
        q_buf.data(),
        dim_order.data(),
        strides.data());
    TensorImpl out_impl(
        ScalarType::Float,
        4,
        sizes.data(),
        out_buf.data(),
        dim_order.data(),
        strides.data());
    Tensor out(&out_impl);
    const SdpaRowExactScope scope(row_exact);
    KernelRuntimeContext ctx{};
    torch::executor::native::custom_quantized_sdpa_out(
        ctx,
        Tensor(&q_impl),
        k,
        v,
        kStartPos,
        {},
        0.0,
        true,
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        kSeqAtDim2,
        out);
    EXPECT_EQ(ctx.failure_state(), Error::Ok);
    return std::vector<float>(
        out_buf.row_data(0), out_buf.row_data(0) + q_values.size());
  };
  EXPECT_EQ(run(true), run(false));
}

TEST(OpSdpaRowExactTest, Int8RowsMatchOneRowCalls) {
  for (const Case& c : kCases) {
    expect_int8_rows_match(c);
  }
}
