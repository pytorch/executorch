// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q4gsw_linear/q4gsw_linear.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

namespace {

// POSITIVE well-conditioned data (no fp16 cancellation): activations are
// multiples of 1/16 in [0.5,1.375]; int4 nibbles in {9..14} (-> weight +1..+6)
// / int8 weights in {1..6}. For dq8ca the per-row activation scale is forced to
// 1/16 with zp=0 so the dynamic int8 quant round-trip is EXACT in both fp16 and
// fp32 (quantized values 8..22) and the fp32 reference is valid. fp16~=fp32
// throughout, so a tight tolerance validates shader structure (catches
// zero-subtile bugs) while ignoring benign fp16 noise.
void set_well_conditioned_data(TestCase& t, const CoopmatLinearConfig& cfg) {
  const bool dq = is_dq8ca(cfg.op_name);
  const bool four = is_4bit(cfg.op_name);
  auto& hin = t.inputs()[0].get_half_data();
  for (size_t i = 0; i < hin.size(); ++i) {
    hin[i] = float_to_half(0.5f + 0.125f * float(i % 8));
  }
  const size_t w_idx = dq ? 3 : 1;
  if (four) {
    auto& wq = t.inputs()[w_idx].get_uint8_data();
    const std::array<uint8_t, 6> kPos = {0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE};
    for (size_t i = 0; i < wq.size(); ++i) {
      wq[i] = kPos[i % 6];
    }
  } else {
    auto& wq = t.inputs()[w_idx].get_int8_data();
    for (size_t i = 0; i < wq.size(); ++i) {
      wq[i] = int8_t(1 + (i % 6));
    }
  }
  if (dq) {
    auto& hs = t.inputs()[1].get_half_data();
    std::fill(hs.begin(), hs.end(), float_to_half(0.0625f));
    auto& zp = t.inputs()[2].get_int8_data();
    std::fill(zp.begin(), zp.end(), int8_t(0));
    // weights were overwritten above -> recompute the sums
    if (four) {
      compute_weight_sums_4bit_grouped(
          t.inputs()[4],
          t.inputs()[w_idx],
          cfg.K / cfg.group_size,
          cfg.N,
          cfg.group_size);
    } else {
      compute_weight_sums(t.inputs()[4], t.inputs()[w_idx], cfg.N, cfg.K);
    }
  }
  t.set_abs_tolerance(0.5f);
  t.set_rel_tolerance(0.05f);
}

std::vector<TestCase> generate_coopmat_test_cases(
    const std::vector<CoopmatLinearConfig>& shapes,
    const std::vector<std::string>& ops,
    const std::vector<utils::StorageType>& storage_types,
    bool well_conditioned_data) {
  std::vector<TestCase> cases;
  for (const auto& op : ops) {
    for (const auto& shape : shapes) {
      CoopmatLinearConfig cfg{shape.M, shape.K, shape.N, shape.group_size, op};
      for (auto st : storage_types) {
        TestCase t = make_case(cfg, st);
        if (well_conditioned_data) {
          set_well_conditioned_data(t, cfg);
        }
        cases.push_back(std::move(t));
      }
    }
  }
  return cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "coopmat_correctness") {
  // Correctness: small aligned {64,128,64} cases for ALL FOUR ops; the buffer
  // case fires the coopmat shader, validated by bench_reference (the
  // performance cases are skipped by it). Texture3D = tiled, Buffer = coopmat.
  // Shapes align to BOTH coopmat geometries (64x64x32 legacy, 128x128x16
  // double-buffered); the second shape dispatches a multi-workgroup grid for
  // both, covering the gl_WorkGroupID-derived tile offsets in the store
  // address math.
  return {
      generate_coopmat_test_cases(
          {
              {64, 128, 64, 64, ""},
              {128, 256, 128, 64, ""},
              {128, 128, 128, 64, ""},
              {256, 256, 256, 64, ""},
              // Discriminators for the tiled-texture cube-shape failure:
              {128, 128, 256, 64, ""}, // M == K only
              {256, 128, 128, 64, ""}, // K == N only
              {64, 128, 256, 64, ""}, // K > M, K < N
              {256, 128, 64, 64, ""}, // K < M, K > N
          },
          /*ops=*/{"linear_q4gsw", "linear_dq8ca_q4gsw"},
          /*storage_types=*/{utils::kTexture3D, utils::kBuffer},
          /*well_conditioned_data=*/true),
      bench_reference,
      flop_calc,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/5};
}

namespace {

constexpr int64_t kM = 1024;
constexpr int64_t kGroup = 128;

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "coopmat_performance") {
  return {
      generate_coopmat_test_cases(
          {
              // Llama 3.1 8B linear weight shapes (K,N) at prefill M (multiple
              // of 64 so coopmat fires).
              {kM, 4096, 4096, kGroup, ""}, // q_proj / o_proj
              {kM, 4096, 1024, kGroup, ""}, // k_proj / v_proj (GQA)
              {kM, 4096, 14336, kGroup, ""}, // gate_proj / up_proj
              {kM, 14336, 4096, kGroup, ""}, // down_proj
          },
          /*ops=*/{"linear_q4gsw", "linear_dq8ca_q4gsw"},
          // Texture3D selects the tiled shader, Buffer the coopmat shader.
          /*storage_types=*/{utils::kTexture3D, utils::kBuffer},
          /*well_conditioned_data=*/false),
      bench_reference,
      flop_calc,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/5};
}

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
