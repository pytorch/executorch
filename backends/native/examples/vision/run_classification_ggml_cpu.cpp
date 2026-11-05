// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/vision/ClassificationRunner.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <gflags/gflags.h>

DEFINE_int32(
    cpu_threads,
    0,
    "ggml CPU threads; zero keeps the backend default.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Run an image-classification .ptn with the native ggml CPU engine.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return ptn::run_classification(
      [] {
        ptn::GgmlEngineOptions options;
        options.cpu_threads = FLAGS_cpu_threads;
        return ptn::GgmlEngineHost::create(options);
      },
      [](const ptn::EngineContext& context) {
        const auto& ggml = dynamic_cast<const ptn::GgmlEngineContext&>(context);
        return ptn::ConstantBytes{
            std::nullopt, ggml.materialized_constant_bytes()};
      });
}
