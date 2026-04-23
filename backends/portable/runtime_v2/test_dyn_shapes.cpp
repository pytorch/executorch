/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Integration test: batch-varying execute() calls.
 *
 * Loads /tmp/dyn_linear_v2.pte (a Linear model with dynamic batch dim
 * in [1, 8], traced at batch=3) and runs forward() with batch sizes
 * {1, 3, 5, 8} in succession against the SAME loaded delegate, to
 * verify true runtime-varying dynamic-shape behavior across our v2
 * runtime — including:
 *
 *   - Buffer max-shape allocation (sized once at init for batch=8).
 *   - bind_inputs accepting a smaller-than-max input shape.
 *   - Per-execute reset of transient bindings.
 *   - Cross-runtime TransferStep (cpu → metal for permute_copy
 *     output) propagating the actual batch each call.
 *   - MetalInstance kernels reading the actual shape via
 *     TensorImpl.sizes() and producing correctly-shaped output.
 *
 * Expected output for each batch B: a [B, 5] tensor of zeros, since
 * x = ones(B,4); weight=full(5,4,0.25); bias=full(5,-1).
 *   x @ weight.T + bias  =  4*1*0.25 + (-1)  =  0.
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

constexpr const char* kDefaultModelPath = "/tmp/dyn_linear_v2.pte";

bool run_one(Module& mod, int32_t batch) {
  // Build a [batch, 4] float input filled with 1.0. The data buffer
  // must outlive the execute() call.
  std::vector<float> data(static_cast<size_t>(batch) * 4, 1.0f);
  std::vector<::executorch::aten::SizesType> sizes = {batch, 4};

  auto input_tensor = from_blob(
      data.data(), sizes, ::executorch::aten::ScalarType::Float);

  auto outputs_result = mod.execute("forward", {EValue(input_tensor)});
  if (!outputs_result.ok()) {
    std::fprintf(stderr, "  batch=%d: execute() failed: 0x%x\n",
                 batch, static_cast<unsigned>(outputs_result.error()));
    return false;
  }

  const auto& outputs = outputs_result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    std::fprintf(stderr, "  batch=%d: no tensor output\n", batch);
    return false;
  }
  const auto& out_t = outputs[0].toTensor();

  auto out_sizes = out_t.sizes();
  std::printf("  batch=%d: output sizes=[", batch);
  for (size_t i = 0; i < out_sizes.size(); ++i) {
    std::printf("%s%d", i ? ", " : "",
                static_cast<int>(out_sizes[i]));
  }
  std::printf("]");

  // Verify: shape should be [batch, 5].
  bool shape_ok = (out_sizes.size() == 2 && out_sizes[0] == batch &&
                   out_sizes[1] == 5);

  // Verify: every element should be 0.0.
  const float* odata = out_t.const_data_ptr<float>();
  size_t numel = static_cast<size_t>(batch) * 5;
  bool values_ok = true;
  float max_abs = 0.0f;
  for (size_t i = 0; i < numel; ++i) {
    float a = std::fabs(odata[i]);
    if (a > max_abs) max_abs = a;
    if (a > 1e-5f) values_ok = false;
  }

  std::printf(", max_abs=%g  %s%s\n", max_abs,
              shape_ok ? "shape ✓" : "shape ✗",
              values_ok ? " values ✓" : " values ✗");
  return shape_ok && values_ok;
}

}  // namespace

int main(int argc, char** argv) {
  ::executorch::runtime::runtime_init();

  const char* model_path =
      (argc > 1) ? argv[1] : kDefaultModelPath;
  std::printf("Loading model: %s\n", model_path);

  Module mod(model_path);
  auto load_err = mod.load_forward();
  if (load_err != Error::Ok) {
    std::fprintf(stderr, "load_forward failed: 0x%x\n",
                 static_cast<unsigned>(load_err));
    return 1;
  }

  std::printf("\nBatch-varying execute() across one loaded delegate:\n");
  bool all_ok = true;
  for (int32_t b : {1, 3, 5, 8}) {
    if (!run_one(mod, b)) all_ok = false;
  }

  std::printf("\n%s\n", all_ok ? "PASS \u2014 all batch sizes correct"
                                : "FAIL \u2014 at least one batch produced wrong output");
  return all_ok ? 0 : 1;
}
