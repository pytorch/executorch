// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/sdpa/sdpa.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace sdpa {

namespace {

struct ModelDims {
  std::string name;
  int64_t head_dim;
  int64_t n_heads;
  int64_t n_kv_heads;
};

const std::vector<ModelDims> kModels = {
    {"Llama-3.2-1B", 64, 32, 8},
    {"Qwen3-0.6B", 128, 16, 8},
    {"Phi-4-mini", 128, 24, 8},
};

// fp32 cases: each config yields one case per storage type x impl.
struct SDPACaseGroup {
  std::vector<SDPAConfig> configs;
  std::vector<utils::StorageType> storage_types;
  std::vector<std::string> impls;
};

struct SDPASetOptions {
  // Swept over every model in kModels (fp16, texture). Each decode context
  // length yields an S=1 case per decode impl; each prefill sequence length
  // yields one S == context_len case.
  std::vector<int64_t> decode_context_lens = {};
  std::vector<std::string> decode_impls = {};
  std::vector<int64_t> prefill_seq_lens = {};
  // Appended after the model sweep.
  std::vector<SDPACaseGroup> accu_cases = {};
};

std::vector<TestCase> generate_test_cases(const SDPASetOptions& options) {
  std::vector<TestCase> test_cases;

  // Perf runs use fp16 texture (matches the LLM decode/prefill production
  // path). A couple of small ACCU shapes validate correctness in fp32.
  const auto dtype = vkapi::kHalf;
  const auto storage = utils::kTexture3D;

  for (const auto& m : kModels) {
    for (int64_t c : options.decode_context_lens) {
      SDPAConfig cfg;
      cfg.head_dim = m.head_dim;
      cfg.n_heads = m.n_heads;
      cfg.n_kv_heads = m.n_kv_heads;
      cfg.seq_len = 1;
      cfg.context_len = c;
      cfg.model = m.name;
      cfg.regime = "decode";
      for (const auto& impl : options.decode_impls) {
        test_cases.push_back(create_sdpa_test_case(cfg, dtype, storage, impl));
      }
    }
    for (int64_t s : options.prefill_seq_lens) {
      SDPAConfig cfg;
      cfg.head_dim = m.head_dim;
      cfg.n_heads = m.n_heads;
      cfg.n_kv_heads = m.n_kv_heads;
      cfg.seq_len = s;
      cfg.context_len = s;
      cfg.model = m.name;
      cfg.regime = "prefill";
      // Prefill (tiled) is unaffected by the impl selector, so it runs a
      // single case.
      test_cases.push_back(
          create_sdpa_test_case(cfg, dtype, storage, "default"));
    }
  }

  for (const auto& group : options.accu_cases) {
    for (const auto& cfg : group.configs) {
      for (const auto& st : group.storage_types) {
        for (const auto& impl : group.impls) {
          test_cases.push_back(
              create_sdpa_test_case(cfg, vkapi::kFloat, st, impl));
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("sdpa", "correctness") {
  return {
      generate_test_cases({
          // Prefill: S == context_len.
          .prefill_seq_lens = {128},
          // Small ACCU correctness cases (fp32), decode + prefill. Texture is
          // the production LLM path; buffer is also validated for decode to
          // guard the attn_weights S/context alignment (a decode-shaped buffer
          // allocation has no headroom for the shaders' align_up_4 stride
          // unless padded — see sdpa_impl).
          .accu_cases =
              {
                  // Cover D=64 (D4=16) and D=128 (D4=32) with the
                  // vendor-default GQA and the per-query-head shaders, across
                  // texture + buffer.
                  {.configs =
                       {
                           {64, 8, 2, 1, 32, "accu", "decode"},
                           {128, 8, 2, 1, 32, "accu_d128", "decode"},
                       },
                   .storage_types = {utils::kTexture3D, utils::kBuffer},
                   .impls = {"gqa", "non_gqa"}},
                  // Force the head_dim output-tiled GQA variant (Adreno-only
                  // in production) so its wg x-collapse and the
                  // partial_n_tile tail get deterministic coverage on any
                  // device: D=64/128 give even D4 (fast path); D=4 gives D4=1
                  // (odd), exercising the partial-tile checked load.
                  {.configs =
                       {
                           {64, 8, 2, 1, 32, "accu_tile2", "decode"},
                           {128, 8, 2, 1, 32, "accu_tile2_d128", "decode"},
                           {4, 8, 2, 1, 32, "accu_tile2_d4", "decode"},
                       },
                   .storage_types = {utils::kTexture3D, utils::kBuffer},
                   .impls = {"gqa_tile2"}},
                  {.configs = {{64, 8, 2, 16, 16, "accu", "prefill"}},
                   .storage_types = {utils::kTexture3D},
                   .impls = {"default"}},
              },
      }),
      sdpa_reference_impl,
      sdpa_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

REGISTER_TEST_CASE_SET("sdpa", "decode") {
  return {
      generate_test_cases({
          // Decode: S=1, sweep context_len.
          .decode_context_lens = {512, 1024, 4096},
          // Decode (S==1) picks a coop AV shader; exercise both the GQA-reuse
          // variant and the per-query-head variant for every decode case.
          .decode_impls = {"gqa", "non_gqa"},
      }),
      sdpa_reference_impl,
      sdpa_flop_calculator,
      /*warmup_runs=*/10,
      /*benchmark_runs=*/30};
}

REGISTER_TEST_CASE_SET("sdpa", "performance") {
  return {
      generate_test_cases({
          // Prefill: S == context_len.
          .prefill_seq_lens = {512},
      }),
      sdpa_reference_impl,
      sdpa_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

} // namespace sdpa
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
