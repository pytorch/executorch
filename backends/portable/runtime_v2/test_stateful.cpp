/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Integration test: stateful model with a mutable buffer.
 *
 * Loads /tmp/stateful_add_v2.pte (a model with `register_buffer("acc",
 * zeros(2,3))` that does `self.acc.add_(x); return self.acc.clone()`)
 * and runs forward(ones(2,3)) three times against the SAME loaded
 * delegate, expecting the accumulator to grow:
 *
 *   call 1 → output = ones(2,3)
 *   call 2 → output = twos(2,3)
 *   call 3 → output = threes(2,3)
 *
 * This exercises the IR's mutable-buffer concept. The current Graph
 * adapter has `num_mutable_buffer_ids() = 0` (stub), so this test
 * surfaces what happens end-to-end when the executor doesn't know to
 * preserve the mutable buffer's state across calls.
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

constexpr const char* kDefaultModelPath = "/tmp/stateful_add_v2.pte";

bool run_one(Module& mod, int call_idx, float expected) {
  std::vector<float> data(6, 1.0f);
  std::vector<::executorch::aten::SizesType> sizes = {2, 3};
  auto input_tensor = from_blob(
      data.data(), sizes, ::executorch::aten::ScalarType::Float);

  auto outputs_result = mod.execute("forward", {EValue(input_tensor)});
  if (!outputs_result.ok()) {
    std::fprintf(stderr, "  call %d: execute() failed: 0x%x\n",
                 call_idx, static_cast<unsigned>(outputs_result.error()));
    return false;
  }
  const auto& outputs = outputs_result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    std::fprintf(stderr, "  call %d: no tensor output\n", call_idx);
    return false;
  }
  const auto& out_t = outputs[0].toTensor();
  auto out_sizes = out_t.sizes();
  bool shape_ok =
      (out_sizes.size() == 2 && out_sizes[0] == 2 && out_sizes[1] == 3);

  const float* odata = out_t.const_data_ptr<float>();
  float max_abs_err = 0.0f;
  for (size_t i = 0; i < 6; ++i) {
    float err = std::fabs(odata[i] - expected);
    if (err > max_abs_err) max_abs_err = err;
  }
  bool values_ok = (max_abs_err < 1e-5f);

  std::printf("  call %d: out=[%.1f %.1f %.1f; %.1f %.1f %.1f] "
              "expected=%.1f  %s%s\n",
              call_idx,
              odata[0], odata[1], odata[2],
              odata[3], odata[4], odata[5],
              expected,
              shape_ok ? "shape ✓" : "shape ✗",
              values_ok ? " values ✓" : " values ✗");
  return shape_ok && values_ok;
}

}  // namespace

int main(int argc, char** argv) {
  ::executorch::runtime::runtime_init();

  const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
  std::printf("Loading model: %s\n", model_path);

  Module mod(model_path);
  auto load_err = mod.load_forward();
  if (load_err != Error::Ok) {
    std::fprintf(stderr, "load_forward failed: 0x%x\n",
                 static_cast<unsigned>(load_err));
    return 1;
  }

  std::printf("\nStateful execute() — accumulator should grow:\n");
  bool all_ok = true;
  for (int call = 1; call <= 3; ++call) {
    float expected = static_cast<float>(call);
    if (!run_one(mod, call, expected)) all_ok = false;
  }

  std::printf("\n%s\n",
              all_ok ? "PASS \u2014 mutable buffer state preserved"
                     : "FAIL \u2014 mutable buffer state NOT preserved correctly");
  return all_ok ? 0 : 1;
}
