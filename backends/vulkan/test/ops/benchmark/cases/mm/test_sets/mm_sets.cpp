// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/mm/mm.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace mm {

namespace {

struct MmShape {
  int64_t B, M, K, N;
};

// Coopmat shader requires M%64==0, N%64==0, K%32==0 (no partial-tile or K-tail
// handling). The force-dispatch selector sweep runs only for shapes that
// satisfy alignment, on buffer storage only (coopmat requires buffer outputs).
bool coopmat_shape_eligible(const MmShape& s) {
  return s.B == 0 && s.M % 64 == 0 && s.N % 64 == 0 && s.K % 32 == 0;
}

std::vector<TestCase> generate_mm_test_cases(
    const std::vector<MmShape>& shapes,
    const std::vector<vkapi::ScalarType>& dtypes,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<std::string>& coopmat_sweep_selectors) {
  std::vector<TestCase> test_cases;

  // The "coopmat" forced-dispatch sweep skips the runtime eligibility check
  // (it's intentionally bypassing the gate to exercise the shader). Skip
  // generating those cases on adapters that can't actually run the shader,
  // or pipeline creation/dispatch will fail.
  const auto* adapter = api::context()->adapter_ptr();
  const bool coopmat_runnable =
      adapter->supports_cooperative_matrix() && adapter->subgroup_size() == 64;
  std::vector<std::string> sweep_selectors;
  for (const auto& sel : coopmat_sweep_selectors) {
    if (sel != "coopmat" || coopmat_runnable) {
      sweep_selectors.push_back(sel);
    }
  }

  for (const auto& s : shapes) {
    bool is_batched = s.B > 0;

    MmConfig dynamic_cfg{s.B, s.M, s.K, s.N, false, false, false};
    MmConfig const_cfg{s.B, s.M, s.K, s.N, false, false, true};

    for (auto dtype : dtypes) {
      for (auto st : storage_types) {
        test_cases.push_back(
            create_mm_test_case(dynamic_cfg, dtype, st, utils::kWidthPacked));
        test_cases.push_back(
            create_mm_test_case(const_cfg, dtype, st, utils::kWidthPacked));
      }

      // Coopmat A/B sweep: only on aligned shapes + buffer storage.
      if (coopmat_shape_eligible(s)) {
        for (const auto& sel : sweep_selectors) {
          MmConfig dyn = dynamic_cfg;
          dyn.impl_selector = sel;
          test_cases.push_back(create_mm_test_case(
              dyn, dtype, utils::kBuffer, utils::kWidthPacked));
          MmConfig con = const_cfg;
          con.impl_selector = sel;
          test_cases.push_back(create_mm_test_case(
              con, dtype, utils::kBuffer, utils::kWidthPacked));
        }
      }

      if (!is_batched) {
        MmConfig addmm_cfg{s.B, s.M, s.K, s.N, true, false, false};
        MmConfig addmm_const_cfg{s.B, s.M, s.K, s.N, true, false, true};
        MmConfig linear_cfg{s.B, s.M, s.K, s.N, false, true, false};
        MmConfig linear_bias_cfg{s.B, s.M, s.K, s.N, true, true, false};

        for (auto st : storage_types) {
          test_cases.push_back(
              create_mm_test_case(addmm_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(create_mm_test_case(
              addmm_const_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(
              create_mm_test_case(linear_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(create_mm_test_case(
              linear_bias_cfg, dtype, st, utils::kWidthPacked));
        }

        // Coopmat A/B sweep on linear paths too (only aligned shapes).
        if (coopmat_shape_eligible(s)) {
          for (const auto& sel : sweep_selectors) {
            MmConfig lin = linear_cfg;
            lin.impl_selector = sel;
            test_cases.push_back(create_mm_test_case(
                lin, dtype, utils::kBuffer, utils::kWidthPacked));
            MmConfig lin_bias = linear_bias_cfg;
            lin_bias.impl_selector = sel;
            test_cases.push_back(create_mm_test_case(
                lin_bias, dtype, utils::kBuffer, utils::kWidthPacked));
          }
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("mm", "correctness") {
  const std::vector<MmShape> shapes = {
      // Accuracy shapes (float)
      {0, 64, 128, 64},
      {0, 128, 256, 128},
      {0, 32, 64, 256},
      {1, 64, 128, 64},
      {1, 4, 32, 16},
      // Non-multiple-of-4 accuracy shapes (exercises scalar shader fallback)
      {0, 57, 131, 43},
      {0, 33, 67, 91},
      {1, 19, 53, 37},
      {0, 64, 128, 47}, // only N unaligned
      {0, 64, 47, 128}, // only K unaligned
  };
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat};
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<std::string> coopmat_sweep_selectors = {"tiled", "coopmat"};

  return {
      generate_mm_test_cases(
          shapes, dtypes, storage_types, coopmat_sweep_selectors),
      reference_impl,
      mm_flop_calculator};
}

REGISTER_TEST_CASE_SET("mm", "performance") {
  const std::vector<MmShape> shapes = {
      // Performance shapes (half)
      {0, 4096, 1024, 256},
      {0, 4096, 256, 128},
      {0, 4096, 128, 256},
      {1, 4096, 256, 1024},
      {1, 4096, 256, 128},
      {1, 256, 4096, 64},
      {0, 4096, 64, 128},
  };
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<std::string> coopmat_sweep_selectors = {"tiled", "coopmat"};

  return {
      generate_mm_test_cases(
          shapes, dtypes, storage_types, coopmat_sweep_selectors),
      reference_impl,
      mm_flop_calculator};
}

} // namespace mm
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
