// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstdio>
#include <exception>
#include <memory>
#include <string_view>

#include <executorch/backends/native/runtime/api/Model.h>
#include <executorch/backends/native/runtime/api/Session.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    return 1;
  }
  try {
    std::unique_ptr<ptn::EngineHost> host;
    if (std::string_view(argv[2]) == "vulkan") {
      host = ptn::VulkanEngineHost::create();
    } else {
      host = ptn::GgmlEngineHost::create();
    }
    const auto model = ptn::Model::load_file(argv[1]);
    auto session = model.create_session(*host);
    const std::array<int64_t, 2> shape{1, 6};
    const std::array<int64_t, 2> strides{6, 1};
    const std::array<float, 6> input{1, 2, 3, 4, 5, 6};
    std::array<float, 6> output{};
    const std::array inputs{ptn::ConstTensorView(
        ptn::kFloat, shape, strides, input.data(), sizeof(input))};
    std::array outputs{ptn::MutableTensorView(
        ptn::kFloat, shape, strides, output.data(), sizeof(output))};
    session.run("forward", inputs, outputs);
    for (size_t i = 0; i < output.size(); ++i) {
      if (output[i] != 2 * input[i]) {
        std::fprintf(stderr, "output mismatch at %zu: %g\n", i, output[i]);
        return 1;
      }
    }
    std::printf("%s: PASS\n", host->name().c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
