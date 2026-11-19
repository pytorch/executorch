/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/torch_rng.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

using executorch::aten::BFloat16;
using executorch::aten::Half;
using executorch::extension::diffusers::TorchGenerator;
using json = nlohmann::json;

namespace {

json load_goldens() {
  std::ifstream file(std::getenv("DIFFUSERS_GOLDENS_PATH"));
  return json::parse(file);
}

// Sign-magnitude float bits as an integer whose consecutive values are
// adjacent floats, so the difference of two of them counts ulps.
int64_t ordered(uint64_t bits, int width) {
  const uint64_t sign = uint64_t{1} << (width - 1);
  const auto magnitude = static_cast<int64_t>(bits & (sign - 1));
  return (bits & sign) ? -magnitude : magnitude;
}

std::vector<int64_t>
draw(TorchGenerator& generator, const std::string& dtype, size_t numel) {
  std::vector<int64_t> values;
  if (dtype == "float32") {
    for (float value : generator.randn<float>(numel)) {
      uint32_t raw;
      std::memcpy(&raw, &value, sizeof(raw));
      values.push_back(ordered(raw, 32));
    }
  } else if (dtype == "float16") {
    const auto draws = generator.randn<Half>(numel);
    std::transform(
        draws.begin(), draws.end(), std::back_inserter(values), [](Half value) {
          return ordered(value.x, 16);
        });
  } else {
    const auto draws = generator.randn<BFloat16>(numel);
    std::transform(
        draws.begin(),
        draws.end(),
        std::back_inserter(values),
        [](BFloat16 value) { return ordered(value.x, 16); });
  }
  return values;
}

} // namespace

// Within two ulps. sin and cos come from the platform's math library and can
// differ from torch's build in the last bit: an unoptimized build calls sinf
// and cosf, while optimized builds, like torch's, call one fused sincos.
// Multiplying by the Box-Muller radius can turn that one ulp into two.
TEST(TorchGeneratorTest, MatchesTorchRandn) {
  const auto goldens = load_goldens();
  for (const auto& rng_case : goldens.at("rng")) {
    TorchGenerator generator(rng_case.at("seed").get<uint64_t>());
    for (const auto& draw_case : rng_case.at("draws")) {
      const auto dtype = draw_case.at("dtype").get<std::string>();
      const int width = dtype == "float32" ? 32 : 16;
      const auto bits = draw_case.at("bits").get<std::vector<int64_t>>();
      const auto values = draw(generator, dtype, bits.size());
      for (size_t i = 0; i < bits.size(); ++i) {
        const auto expected = ordered(
            static_cast<uint64_t>(bits[i]) & ((uint64_t{1} << width) - 1),
            width);
        EXPECT_LE(std::llabs(values[i] - expected), 2)
            << "seed " << rng_case.at("seed") << ", " << dtype << " x"
            << bits.size() << " at " << i;
      }
    }
  }
}
