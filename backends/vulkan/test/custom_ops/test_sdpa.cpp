// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

// Correctness is only checked for these small shapes; larger perf shapes throw
// std::invalid_argument from the reference (framework marks them SKIPPED) to
// avoid an O(H*S*context*D) CPU reference on the large perf matrix.
static constexpr int64_t kRefContextLenLimit = 256;

// When true (env SDPA_NO_CHAIN=1), each graph.execute() dispatches the SDPA op
// exactly once (op_invocations_per_execute=1), disabling the framework's
// probe-then-scale chaining. Chaining stacks many back-to-back QK/softmax/AV
// triples in one command buffer; consecutive triples pipeline on the GPU so
// their timestamp windows overlap, which inflates/misattributes per-dispatch
// durations (notably the AV dispatch) at the large chained_dispatches factors
// picked for the cheap small-context decode cases. One-invocation-per-execute
// removes the inter-invocation overlap; stability comes from warmup + a large
// median-of-N instead.
static bool no_chain_mode() {
  const char* v = std::getenv("SDPA_NO_CHAIN");
  return v != nullptr && v[0] == '1';
}

// When true (env SDPA_DECODE_ONLY=1), generate only the 9 decode cases.
static bool decode_only_mode() {
  const char* v = std::getenv("SDPA_DECODE_ONLY");
  return v != nullptr && v[0] == '1';
}

// LLM SDPA (llama.custom_sdpa) shape:
//   q:       [1, S,           n_heads,    head_dim]  (DHSB, width-packed)
//   k/v cache:[1, context_len, n_kv_heads, head_dim]
struct SDPAConfig {
  int64_t head_dim;
  int64_t n_heads;
  int64_t n_kv_heads;
  int64_t seq_len; // S: query tokens (1 for decode, >1 for prefill)
  int64_t context_len; // total KV length (kv_len)
  std::string model; // label only
  std::string regime; // "decode" / "prefill", label only
};

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

static TestCase create_sdpa_test_case(
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

  if (no_chain_mode()) {
    test_case.set_op_invocations_per_execute(1);
  }

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-2f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  return test_case;
}

// Reference: causal SDPA over the KV cache.
//   q:[1,S,H,D], k/v cache:[1,C,Hkv,D], input_pos = C - S.
//   For query row s (absolute position input_pos + s), attends to cache
//   positions [0, input_pos + s]. GQA: head h maps to kv head h / (H/Hkv).
static void sdpa_reference_impl(TestCase& test_case) {
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

// FLOPs: QK (2*S*C*D) + AV (2*S*C*D) per head, summed over heads. Softmax
// is negligible. Uses the causal-average context (~C/2) is ignored; report
// full-C dense FLOPs as an upper bound proxy.
static int64_t sdpa_flop_calculator(const TestCase& test_case) {
  const auto q_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto k_sizes = test_case.inputs()[1].get_tensor_sizes();
  const int64_t S = q_sizes[1];
  const int64_t H = q_sizes[2];
  const int64_t D = q_sizes[3];
  const int64_t C = k_sizes[1];
  return 4 * H * S * C * D;
}

static std::vector<TestCase> generate_sdpa_test_cases() {
  std::vector<TestCase> test_cases;

  struct ModelDims {
    std::string name;
    int64_t head_dim;
    int64_t n_heads;
    int64_t n_kv_heads;
  };

  const std::vector<ModelDims> models = {
      {"Llama-3.2-1B", 64, 32, 8},
      {"Qwen3-0.6B", 128, 16, 8},
      {"Phi-4-mini", 128, 24, 8},
  };

  // Decode: S=1, sweep context_len.
  const std::vector<int64_t> decode_context_lens = {512, 1024, 4096};
  // Prefill: S == context_len.
  const std::vector<int64_t> prefill_seq_lens = {128, 512};

  // Perf runs use fp16 texture (matches the LLM decode/prefill production
  // path). A couple of small ACCU shapes validate correctness in fp32.
  const auto dtype = vkapi::kHalf;
  const auto storage = utils::kTexture3D;

  // Decode (S==1) picks a coop AV shader; exercise both the GQA-reuse variant
  // and the per-query-head variant for every decode case. Prefill (tiled) is
  // unaffected by the selector, so it runs a single case.
  const std::vector<std::string> decode_impls = {"gqa", "non_gqa"};

  for (const auto& m : models) {
    for (int64_t c : decode_context_lens) {
      SDPAConfig cfg;
      cfg.head_dim = m.head_dim;
      cfg.n_heads = m.n_heads;
      cfg.n_kv_heads = m.n_kv_heads;
      cfg.seq_len = 1;
      cfg.context_len = c;
      cfg.model = m.name;
      cfg.regime = "decode";
      for (const auto& impl : decode_impls) {
        test_cases.push_back(create_sdpa_test_case(cfg, dtype, storage, impl));
      }
    }
    if (decode_only_mode()) {
      continue;
    }
    for (int64_t s : prefill_seq_lens) {
      SDPAConfig cfg;
      cfg.head_dim = m.head_dim;
      cfg.n_heads = m.n_heads;
      cfg.n_kv_heads = m.n_kv_heads;
      cfg.seq_len = s;
      cfg.context_len = s;
      cfg.model = m.name;
      cfg.regime = "prefill";
      test_cases.push_back(
          create_sdpa_test_case(cfg, dtype, storage, "default"));
    }
  }

  if (decode_only_mode()) {
    return test_cases;
  }

  // Small ACCU correctness cases (fp32), decode + prefill. Texture is the
  // production LLM path; buffer is also validated for decode to guard the
  // attn_weights S/context alignment (a decode-shaped buffer allocation has no
  // headroom for the shaders' align_up_4 stride unless padded — see sdpa_impl).
  {
    // Cover D=64 (D4=16) and D=128 (D4=32) with the vendor-default GQA and the
    // per-query-head shaders, across texture + buffer.
    const std::vector<SDPAConfig> decs = {
        {64, 8, 2, 1, 32, "accu", "decode"},
        {128, 8, 2, 1, 32, "accu_d128", "decode"},
    };
    for (const auto& dec : decs) {
      for (const auto& storage : {utils::kTexture3D, utils::kBuffer}) {
        for (const auto& impl : decode_impls) {
          test_cases.push_back(
              create_sdpa_test_case(dec, vkapi::kFloat, storage, impl));
        }
      }
    }

    // Force the head_dim output-tiled GQA variant (Adreno-only in production)
    // so its wg x-collapse and the partial_n_tile tail get deterministic
    // coverage on any device: D=64/128 give even D4 (fast path); D=4 gives D4=1
    // (odd), exercising the partial-tile checked load.
    const std::vector<SDPAConfig> tile2_decs = {
        {64, 8, 2, 1, 32, "accu_tile2", "decode"},
        {128, 8, 2, 1, 32, "accu_tile2_d128", "decode"},
        {4, 8, 2, 1, 32, "accu_tile2_d4", "decode"},
    };
    for (const auto& dec : tile2_decs) {
      for (const auto& storage : {utils::kTexture3D, utils::kBuffer}) {
        test_cases.push_back(
            create_sdpa_test_case(dec, vkapi::kFloat, storage, "gqa_tile2"));
      }
    }

    SDPAConfig pre{64, 8, 2, 16, 16, "accu", "prefill"};
    test_cases.push_back(create_sdpa_test_case(
        pre, vkapi::kFloat, utils::kTexture3D, "default"));
  }

  return test_cases;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "SDPA (llama.custom_sdpa) Benchmark" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = sdpa_reference_impl;

  const bool decode_only = decode_only_mode();
  const int warmup_runs = decode_only ? 10 : 3;
  const int benchmark_runs = decode_only ? 30 : 10;

  auto results = execute_test_cases(
      generate_sdpa_test_cases,
      sdpa_flop_calculator,
      "SDPA",
      warmup_runs,
      benchmark_runs,
      ref_fn);

  return 0;
}
