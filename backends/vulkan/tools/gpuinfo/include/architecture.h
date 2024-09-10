/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include "app.h"
#include "stats.h"
#include "utils.h"

using namespace vkapi;

namespace gpuinfo {

void reg_count(const App& app) {
  if (!app.enabled("reg_count")) {
    std::cout << "Skipped Register Count" << std::endl;
    return;
  }

  std::cout << std::endl;
  std::cout << "------ Register Count ------" << std::endl;
  const uint32_t NREG_MIN = 1;
  const uint32_t NREG_MAX = 512;
  const uint32_t NREG_STEP = 1;

  const double COMPENSATE = app.get_config("reg_count", "compensate");
  const double THRESHOLD = app.get_config("reg_count", "threshold");

  const uint32_t NGRP_MIN = 1;
  const uint32_t NGRP_MAX = 64;
  const uint32_t NGRP_STEP = 1;

  uint32_t NITER;

  auto bench = [&](uint32_t ngrp, uint32_t nreg) {
    StagingBuffer buffer(context(), vkapi::kFloat, 1);
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "reg_count_" + std::to_string(nreg);

    auto time = benchmark_on_gpu(shader_name, 30, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          {1, ngrp, 1},
          {1, 1, 1},
          {SV(NITER)},
          VK_NULL_HANDLE,
          0,
          buffer.buffer());
    });
    return time;
  };

  ensure_min_niter(1000, NITER, [&]() { return bench(1, NREG_MIN); });

  uint32_t nreg_max;

  DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
  uint32_t nreg = NREG_MIN;
  for (; nreg <= NREG_MAX; nreg += NREG_STEP) {
    double time = bench(1, nreg);
    std::cout << "Testing nreg=\t" << nreg << "\tTime=\t" << time << "\tus"
              << std::endl;
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

  auto find_ngrp_by_nreg = [&](const uint32_t nreg) {
    DtJumpFinder<3> dj(COMPENSATE, THRESHOLD);
    for (auto ngrp = NGRP_MIN; ngrp <= NGRP_MAX; ngrp += NGRP_STEP) {
      auto time = bench(ngrp, nreg);
      std::cout << "Testing occupation (nreg=\t" << nreg << "\t); ngrp=\t"
                << ngrp << "\t, time=\t" << time << "\tus" << std::endl;

      if (dj.push(time)) {
        ngrp -= NGRP_STEP;
        std::cout << "Using " << nreg << " registers can have " << ngrp
                  << " concurrent single-thread workgroups" << std::endl;
        return ngrp;
      }
    }
    std::cout
        << "Unable to conclude a maximum number of concurrent single-thread workgroups when "
        << nreg << " registers are occupied" << std::endl;
    return (uint32_t)1;
  };

  uint32_t ngrp_full, ngrp_half;
  ngrp_full = find_ngrp_by_nreg(nreg_max);
  ngrp_half = find_ngrp_by_nreg(nreg_max / 2);

  std::string reg_ty;

  if (ngrp_full * 1.5 < ngrp_half) {
    std::cout << "All physical threads in an sm share " << nreg_max
              << " registers" << std::endl;
    reg_ty = "Pooled";

  } else {
    std::cout << "Each physical thread has " << nreg_max << " registers"
              << std::endl;
    reg_ty = "Dedicated";
  }

  std::cout << std::endl << std::endl;
  std::cout << "MaxRegisters," << nreg_max << std::endl;
  std::cout << "ConcurrentWorkgroupsFullReg," << ngrp_full << std::endl;
  std::cout << "ConcurrentWorkgroupsHalfReg," << ngrp_half << std::endl;
  std::cout << "RegisterType," << reg_ty << std::endl;
}

// Warp size is a difficult metric to obtain because the hardware limitations
// do not always coincide with the way the SM divides the workload. For
// instance, the hardware can have a warp size of 64 threads, but an SM might
// be able to simulate concurrency of 128 threads with a single scheduler.

// Because of this, it is important to measure the warp size different ways,
// that can evidence both the physical limitations of the hardware, and the
// actual behavior of the driver.

// Additionally,the SM can behave in two different ways when the assigned
// workload is smaller than the warp size.

// In Case 1, like ARM Mali, the SM can assign dummy workloads to fill empty
// threads and maintain a uniform workload.

// In Case 2, like in Adreno, the driver might decide to pack multiple works
// together and dispatch them at once.
void warp_size(const App& app, const bool verbose = false) {
  if (!app.enabled("warp_size")) {
    std::cout << "Skipped Warp Size" << std::endl;
    return;
  }

  std::cout << "\n------ Warp Size ------" << std::endl;

  // Method A: Stress test with a kernel that uses complex ALU operations like
  // integer division to avoid latency hiding. Increase the number of threads
  // until a jump in latency is detected.

  // This timing-based method helps us identify physical warp sizes. It also
  // helps with Case 2, when threads of multiple warps are managed by the same
  // scheduler at the same time.
  const double COMPENSATE = app.get_config("warp_size", "compensate");
  const double THRESHOLD = app.get_config("warp_size", "threshold");

  uint32_t NITER;

  auto bench = [&](uint32_t nthread) {
    StagingBuffer out_buf(context(), vkapi::kInt, app.nthread_logic);
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "warp_size_physical";

    auto time = benchmark_on_gpu(shader_name, 10, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          // Large number of work groups selected to potentially saturate all
          // ALUs and thus have a better baseline for comparison.
          {nthread, 1024, 1},
          {nthread, 1, 1},
          {SV(NITER)},
          VK_NULL_HANDLE,
          0,
          out_buf.buffer());
    });

    return time;
  };

  ensure_min_niter(1000, NITER, [&]() { return bench(1); });

  uint32_t warp_size = app.subgroup_size;
  DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);

  // We increase the number of threads until we hit a jump in the data.
  uint32_t nthread = 1;
  for (; nthread <= app.nthread_logic; ++nthread) {
    double time = bench(nthread);
    std::cout << "nthread=\t" << nthread << "\t(\t" << time << "\tus)"
              << std::endl;
    if (dj.push(time)) {
      warp_size = nthread - 1;
      break;
    }
  }
  if (nthread >= app.nthread_logic) {
    std::cout
        << "Unable to conclude a physical warp size. Assuming warp_size == subgroup_size"
        << std::endl;
  }

  // Method B: Let all the threads in a warp race and atomically fetch-add
  // a counter, then store the counter values to the output buffer in the
  // scheduling order of these threads. If all the order numbers follow an
  // ascending order, then the threads are likely executing within a warp.
  // Threads in different warps are not managed by the same scheduler, so they
  // would race for a same ID out of order, unaware of each other.

  // This method evidences the actual driver behavior when running
  // concurrency, regardless of the physical limitations of the hardware.

  // Likewise, this method helps us identify warp sizes when the SM
  // sub-divides its ALUs into independent groups, like the three execution
  // engines in a Mali G76 core. It helps warp-probing in Case 1 because it
  // doesn't depend on kernel timing, so the extra wait time doesn't lead to
  // inaccuracy.
  auto bench_sm = [&](uint32_t nthread) {
    StagingBuffer out_buf(context(), vkapi::kInt, app.nthread_logic);
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "warp_size_scheduler";

    benchmark_on_gpu(shader_name, 1, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          {nthread, 1, 1},
          {nthread, 1, 1},
          {},
          VK_NULL_HANDLE,
          0,
          out_buf.buffer());
    });

    std::vector<int32_t> data(app.nthread_logic);
    out_buf.copy_to(data.data(), out_buf.nbytes());

    if (verbose) {
      std::stringstream ss;
      for (auto j = 0; j < nthread; ++j) {
        ss << data[j] << " ";
      }
      std::cout << ss.str() << std::endl;
    }

    // Check until which point is the data in ascending order.
    int32_t last = -1;
    int32_t j = 0;
    for (; j < nthread; ++j) {
      if (last >= data[j]) {
        break;
      }
      last = data[j];
    }

    return j;
  };

  // Test increasing sizes until the data is no longer in ascending order.
  uint32_t warp_size_scheduler = warp_size;
  int i = 1;
  for (; i <= app.nthread_logic; ++i) {
    uint32_t nascend = bench_sm(i);
    if (nascend != i) {
      warp_size_scheduler = nascend;
      break;
    }
  }
  if (i > app.nthread_logic) {
    std::cout << "Unable to conclude an SM Warp Size." << std::endl;
  }

  std::cout << "PhysicalWarpSize," << warp_size << std::endl;
  std::cout << "SMWarpSize," << warp_size_scheduler << std::endl;
}
}; // namespace gpuinfo
