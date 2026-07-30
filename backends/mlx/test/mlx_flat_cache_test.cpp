/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op-level test for the off-graph flat KV cache (MLXFlatCache / FlatPool).
//
// Drives MLXFlatCache::update_and_fetch directly (no interpreter / .pte), then
// attends the returned AttendSpec exactly as the op handler does, and checks
// the result against MLX SDPA over the full K/V history the cache should have
// assembled -- verifying the plan/write/read window and mask kind across
// prefill (Causal) and decode (None), plus the capacity-reject, storage-dtype,
// and required-dtype paths.
//
// Must run on Apple Silicon: MLX SDPA needs the Metal backend.

#include "MLXFlatCache.h"

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace ::executorch::backends::mlx;
namespace cache = ::executorch::extension::llm::cache;
using ::mlx::core::array;

namespace {

// Max absolute difference within tolerance. Computed in float32: item<float>()
// reads sizeof(float) bytes, so calling it on an fp16 scalar misreads the
// buffer.
bool allclose(const array& a, const array& b, float atol) {
  using namespace ::mlx::core;
  array m = max(abs(subtract(astype(a, float32), astype(b, float32))));
  eval(m);
  return m.item<float>() <= atol;
}

cache::CacheConfig flat_config(
    int capacity,
    int n_layers,
    int n_kv_heads,
    int head_dim,
    std::optional<int> kv_dtype = std::nullopt) {
  cache::CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.n_layers = n_layers;
  cfg.layers = {cache::LayerConfig{
      cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0},
      n_kv_heads,
      head_dim}};
  cfg.kv_dtype = kv_dtype;
  return cfg;
}

class MLXFlatCacheTest : public ::testing::Test {
 protected:
  const int H = 2;
  const int D = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  ::mlx::core::StreamOrDevice s = {};

  array randn(int T, ::mlx::core::Dtype dt) {
    return ::mlx::core::random::normal(
        to_shape(std::vector<int>{1, H, T, D}), dt);
  }
};

// Prefill: T=4 at position 0 -> Causal (lower-right aligned).
TEST_F(MLXFlatCacheTest, PrefillIsCausal) {
  using namespace ::mlx::core;
  MLXFlatCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array q0 = randn(T0, float16);
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);

  AttendSpec spec0 = c.update_and_fetch(0, /*position=*/0, k0, v0, s);
  EXPECT_EQ(spec0.kind, AttendSpec::Mask::Causal);

  array cand0 = fast::scaled_dot_product_attention(
      q0,
      spec0.K,
      spec0.V,
      scale,
      std::string("causal"),
      std::nullopt,
      std::nullopt,
      s);
  array ref0 = fast::scaled_dot_product_attention(
      q0, k0, v0, scale, std::string("causal"), std::nullopt, std::nullopt, s);
  EXPECT_TRUE(allclose(cand0, ref0, 1e-2f));
}

// Decode: after a T=4 prefill, a single query at position 4 -> None and must
// attend the full assembled history.
TEST_F(MLXFlatCacheTest, DecodeAttendsFullHistory) {
  using namespace ::mlx::core;
  MLXFlatCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k0 = randn(T0, float16);
  array v0 = randn(T0, float16);
  c.update_and_fetch(0, /*position=*/0, k0, v0, s); // prefill

  array q1 = randn(1, float16);
  array k1 = randn(1, float16);
  array v1 = randn(1, float16);
  AttendSpec spec1 = c.update_and_fetch(0, /*position=*/T0, k1, v1, s);
  EXPECT_EQ(spec1.kind, AttendSpec::Mask::None);

  array cand1 = fast::scaled_dot_product_attention(
      q1,
      spec1.K,
      spec1.V,
      scale,
      std::string(""),
      std::nullopt,
      std::nullopt,
      s);
  array Khist = concatenate(std::vector<array>{k0, k1}, 2, s);
  array Vhist = concatenate(std::vector<array>{v0, v1}, 2, s);
  array ref1 = fast::scaled_dot_product_attention(
      q1, Khist, Vhist, scale, std::string(""), std::nullopt, std::nullopt, s);
  EXPECT_TRUE(allclose(cand1, ref1, 1e-2f));
}

// A step past capacity is rejected (plan returns nullopt).
TEST_F(MLXFlatCacheTest, StepPastCapacityThrows) {
  using namespace ::mlx::core;
  MLXFlatCache c(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  array kx = randn(1, float16);
  EXPECT_ANY_THROW(c.update_and_fetch(0, /*position=*/32, kx, kx, s));
}

// Storage dtype != compute: fp32 input, fp16 storage. The cache casts on write,
// so the read-back K/V are exactly the fp16 of the input.
TEST_F(MLXFlatCacheTest, StorageDtypeDiffersCastsOnWrite) {
  using namespace ::mlx::core;
  MLXFlatCache c16(flat_config(
      /*capacity=*/32,
      /*n_layers=*/1,
      H,
      D,
      static_cast<int>(ScalarType::Half)));
  const int T0 = 4;
  array k2 = randn(T0, float32);
  array v2 = randn(T0, float32);
  AttendSpec spec2 = c16.update_and_fetch(0, /*position=*/0, k2, v2, s);
  EXPECT_EQ(spec2.K.dtype(), float16);
  EXPECT_EQ(spec2.V.dtype(), float16);
  EXPECT_TRUE(allclose(spec2.K, astype(k2, float16, s), 0.0f));
  EXPECT_TRUE(allclose(spec2.V, astype(v2, float16, s), 0.0f));
}

// The MLX cache requires an explicit storage dtype.
TEST_F(MLXFlatCacheTest, UnsetKvDtypeThrows) {
  cache::CacheConfig cfg = flat_config(/*capacity=*/32, /*n_layers=*/1, H, D);
  EXPECT_ANY_THROW(MLXFlatCache{cfg});
}

} // namespace
