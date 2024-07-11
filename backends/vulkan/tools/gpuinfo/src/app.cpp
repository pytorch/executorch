/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>
#include <iostream>

#include "stats.h"
#include "utils.h"

void reg_count() {
  const uint32_t NREG_MIN = 1;
  const uint32_t NREG_MAX = 512;
  const uint32_t NREG_STEP = 1;

  const double COMPENSATE = 0.01;
  const double THRESHOLD = 3;

  uint32_t NITER;

  auto bench = [&](uint32_t nthread, uint32_t ngrp, uint32_t nreg) {
    size_t len = sizeof(float);
    StorageBuffer buffer(context(), vkapi::kFloat, len);
    ParamsBuffer params(context(), int32_t(len));
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "reg_count_" + std::to_string(nreg);

    auto time = benchmark_on_gpu(shader_name, 100, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          {nthread, ngrp, 1},
          {nthread, 1, 1},
          {SV(NITER)},
          VK_NULL_HANDLE,
          0,
          buffer.buffer(),
          params.buffer());
    });
    return time;
  };

  std::cout << "Calculating NITER..." << std::endl;
  ensure_min_niter(1000, NITER, [&]() { return bench(1, 1, NREG_MIN); });
  std::cout << "NITER," << NITER << std::endl;

  uint32_t nreg_max;

  DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
  uint32_t nreg = NREG_MIN;
  for (; nreg <= NREG_MAX; nreg += NREG_STEP) {
    double time = bench(1, 1, nreg);
    std::cout << "Testing nreg=\t" << nreg << "\tTime=\t" << time << std::endl;
    if (dj.push(time)) {
      nreg -= NREG_STEP;
      nreg_max = nreg;
      break;
    }
  }
  if (nreg >= NREG_MAX) {
    std::cout << "Unable to conclude a maximal register count" << std::endl;
    nreg_max = NREG_STEP;
  } else {
    std::cout << nreg_max << " registers are available at most" << std::endl;
  }
}

int main(int argc, const char** argv) {
  context()->initialize_querypool();

  reg_count();

  return 0;
}
