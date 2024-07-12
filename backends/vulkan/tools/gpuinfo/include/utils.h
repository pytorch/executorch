/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

using namespace vkcompute;
using namespace api;

#define QP context()->querypool()

auto benchmark_on_gpu(
    std::string shader_id,
    uint32_t niter,
    std::function<void()> encode_kernel) {
  auto fence = context()->fences().get_fence();

  for (int i = 0; i < niter; ++i) {
    encode_kernel();
  };

  context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
  QP.extract_results();
  uint64_t count = QP.get_mean_shader_ns(shader_id);
  QP.reset_state();
  context()->flush();

  return count / 1000.f;
}

void ensure_min_niter(
    double min_time_us,
    uint32_t& niter,
    std::function<double()> run) {
  const uint32_t DEFAULT_NITER = 100;
  niter = DEFAULT_NITER;
  for (uint32_t i = 0; i < 100; ++i) {
    double t = run();
    if (t > min_time_us * 0.99) {
      return;
    }
    niter = uint32_t(niter * min_time_us / t);
  }
}
