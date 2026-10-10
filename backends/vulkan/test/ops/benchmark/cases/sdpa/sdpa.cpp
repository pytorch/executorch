// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/sdpa/sdpa.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace sdpa {

TestCase create_sdpa_test_case(
    const SDPAConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    const std::string& impl) {
  TestCase test_case;

  const bool is_perf = config.context_len > kRefContextLenLimit;
  const std::string prefix = is_perf ? "PERF" : "ACCU";
  const std::string storage_str = repr_str(storage_type, utils::kWidthPacked);
  const std::string dtype_str = dtype_short(dtype);

  const std::string shape = "D" + std::to_string(config.head_dim) + " H" +
      std::to_string(config.n_heads) + " Hkv" +
      std::to_string(config.n_kv_heads) + " S" +
      std::to_string(config.seq_len) + " C" +
      std::to_string(config.context_len);

  const std::string suffix =
      "[" + config.model + " " + config.regime + " " + impl + "]";

  test_case.set_name(make_test_label(
      prefix, dtype_str, dtype_str, shape, storage_str, suffix));
  test_case.set_operator_name("test_etvk.test_sdpa.default");

  // q: [1, S, n_heads, head_dim]
  ValueSpec q(
      {1, config.seq_len, config.n_heads, config.head_dim},
      dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  // k_cache / v_cache: [1, context_len, n_kv_heads, head_dim]
  ValueSpec k_cache(
      {1, config.context_len, config.n_kv_heads, config.head_dim},
      dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);
  ValueSpec v_cache(
      {1, config.context_len, config.n_kv_heads, config.head_dim},
      dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  ValueSpec impl_selector = ValueSpec::make_string(impl);

  // out: [1, S, n_heads, head_dim]
  ValueSpec output(
      {1, config.seq_len, config.n_heads, config.head_dim},
      dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  test_case.add_input_spec(q);
  test_case.add_input_spec(k_cache);
  test_case.add_input_spec(v_cache);
  test_case.add_input_spec(impl_selector);
  test_case.add_output_spec(output);

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-2f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  return test_case;
}

// FLOPs: QK (2*S*C*D) + AV (2*S*C*D) per head, summed over heads. Softmax
// is negligible. Uses the causal-average context (~C/2) is ignored; report
// full-C dense FLOPs as an upper bound proxy.
int64_t sdpa_flop_calculator(const TestCase& test_case) {
  const auto q_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto k_sizes = test_case.inputs()[1].get_tensor_sizes();
  const int64_t S = q_sizes[1];
  const int64_t H = q_sizes[2];
  const int64_t D = q_sizes[3];
  const int64_t C = k_sizes[1];
  return 4 * H * S * C * D;
}

static std::vector<float> as_float_data(const ValueSpec& spec) {
  if (spec.dtype == vkapi::kFloat) {
    return spec.get_float_data();
  }
  if (spec.dtype == vkapi::kHalf) {
    const auto& half_bits = spec.get_half_data();
    std::vector<float> out(half_bits.size());
    for (size_t i = 0; i < half_bits.size(); ++i) {
      out[i] = half_to_float(half_bits[i]);
    }
    return out;
  }
  throw std::invalid_argument("as_float_data: unsupported dtype");
}

// Reference: causal SDPA over the KV cache.
//   q:[1,S,H,D], k/v cache:[1,C,Hkv,D], input_pos = C - S.
//   For query row s (absolute position input_pos + s), attends to cache
//   positions [0, input_pos + s]. GQA: head h maps to kv head h / (H/Hkv).
void sdpa_reference_impl(TestCase& test_case) {
  const auto& q = test_case.inputs()[0];
  const auto& k = test_case.inputs()[1];
  const auto& v = test_case.inputs()[2];

  const auto q_sizes = q.get_tensor_sizes();
  const auto k_sizes = k.get_tensor_sizes();

  const int64_t S = q_sizes[1];
  const int64_t H = q_sizes[2];
  const int64_t D = q_sizes[3];
  const int64_t C = k_sizes[1];
  const int64_t Hkv = k_sizes[2];

  if (C > kRefContextLenLimit) {
    throw std::invalid_argument("sdpa reference: perf shape, skipping");
  }

  const int64_t input_pos = C - S;
  const int64_t heads_per_kv = H / Hkv;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  const auto q_data = as_float_data(q);
  const auto k_data = as_float_data(k);
  const auto v_data = as_float_data(v);

  ValueSpec& output = test_case.outputs()[0];
  auto& ref = output.get_ref_float_data();
  ref.assign(S * H * D, 0.0f);

  // Index helpers (contiguous WHCN-flattened as [1, dim1, dim2, dim3]).
  auto q_idx = [&](int64_t s, int64_t h, int64_t d) {
    return (s * H + h) * D + d;
  };
  auto kv_idx = [&](int64_t c, int64_t hk, int64_t d) {
    return (c * Hkv + hk) * D + d;
  };

  std::vector<float> scores(C);
  for (int64_t s = 0; s < S; ++s) {
    const int64_t attend_len = input_pos + s + 1; // causal
    for (int64_t h = 0; h < H; ++h) {
      const int64_t hk = h / heads_per_kv;

      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t c = 0; c < attend_len; ++c) {
        float dot = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
          dot += q_data[q_idx(s, h, d)] * k_data[kv_idx(c, hk, d)];
        }
        dot *= scale;
        scores[c] = dot;
        max_score = std::max(max_score, dot);
      }

      float denom = 0.0f;
      for (int64_t c = 0; c < attend_len; ++c) {
        scores[c] = std::exp(scores[c] - max_score);
        denom += scores[c];
      }

      for (int64_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int64_t c = 0; c < attend_len; ++c) {
          acc += scores[c] * v_data[kv_idx(c, hk, d)];
        }
        ref[q_idx(s, h, d)] = acc / denom;
      }
    }
  }
}

} // namespace sdpa
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
