/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "app.h"
#include "stats.h"
#include "utils.h"

using namespace vkapi;

namespace gpuinfo {

void buf_cacheline_size(const App& app) {
  if (!app.enabled("buf_cacheline_size")) {
    std::cout << "Skipped Buffer Cacheline Size" << std::endl;
    return;
  }

  std::cout << std::endl;
  std::cout << "------ Buffer Cacheline Size ------" << std::endl;

  const double COMPENSATE = app.get_config("buf_cacheline_size", "compensate");
  const double THRESHOLD = app.get_config("buf_cacheline_size", "threshold");

  const uint32_t PITCH = app.buf_cache_size / app.nthread_logic;
  const uint32_t BUF_SIZE = app.buf_cache_size;
  const uint32_t MAX_STRIDE = PITCH;

  uint32_t NITER;

  auto bench = [&](int stride) {
    StagingBuffer in_buf(context(), vkapi::kFloat, BUF_SIZE);
    StagingBuffer out_buf(context(), vkapi::kFloat, 1);
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "buf_cacheline_size";

    auto time = benchmark_on_gpu(shader_name, 100, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          {app.nthread_logic, 1, 1},
          {app.nthread_logic, 1, 1},
          {SV(NITER), SV(stride), SV(PITCH)},
          VK_NULL_HANDLE,
          0,
          in_buf.buffer(),
          out_buf.buffer());
    });
    return time;
  };

  ensure_min_niter(1000, NITER, [&]() { return bench(1); });

  uint32_t cacheline_size;

  DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
  uint32_t stride = 1;
  for (; stride <= MAX_STRIDE; ++stride) {
    double time = bench(stride);
    std::cout << "Testing stride=\t" << stride << "\t, time=\t" << time
              << std::endl;

    if (dj.push(time)) {
      cacheline_size = stride * sizeof(float);
      break;
    }
  }
  if (stride >= MAX_STRIDE) {
    std::cout << "Unable to conclude a top level buffer cacheline size."
              << std::endl;
    cacheline_size = MAX_STRIDE * sizeof(float);
  }

  std::cout << "BufTopLevelCachelineSize," << cacheline_size << std::endl;
}

void _bandwidth(
    const App& app,
    const std::string memtype,
    const uint32_t range) {
  auto memtype_lower = memtype;
  std::transform(
      memtype_lower.begin(),
      memtype_lower.end(),
      memtype_lower.begin(),
      [](unsigned char c) { return std::tolower(c); });

  auto test_name = memtype_lower + "_bandwidth";

  // Cache lines flushed
  const uint32_t NFLUSH = app.get_config(test_name, "nflush");
  // Number of loop unrolls. Changing this value requires an equal change in
  // buf_bandwidth.yaml
  const uint32_t NUNROLL = app.get_config(test_name, "nunroll");
  // Number of iterations. Increasing this value reduces noise in exchange for
  // higher latency.
  const uint32_t NITER = app.get_config(test_name, "niter");
  // Vector dimensions (vec4)
  const uint32_t VEC_WIDTH = 4;
  const uint32_t VEC_SIZE = VEC_WIDTH * sizeof(float);
  // Number of vectors that fit in the selected memory space
  const uint32_t NVEC = range / VEC_SIZE;
  // Number of memory reads per thread
  const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;
  // Number of threads needed to read al l vectors
  // The thread count doesn't divide by thread workload in shared memory
  // because of the limited memory size.
  const uint32_t NTHREAD = memtype == "Shared" ? NVEC : NVEC / NREAD_PER_THREAD;
  // Occupy all threads
  const uint32_t local_x = app.nthread_logic;
  // Ensure that global is a multiple of local, and distribute across all SMs
  const uint32_t global_x =
      (NTHREAD / local_x * local_x) * app.sm_count * NFLUSH;

  auto bench = [&](uint32_t access_size) {
    // Number of vectors that fit in this iteration
    const uint32_t nvec_access = access_size / VEC_SIZE;

    // The address mask works as a modulo because x % 2^n == x & (2^n - 1).
    // This will help us limit address accessing to a specific set of unique
    // addresses depending on the access size we want to measure.
    const uint32_t addr_mask = nvec_access - 1;

    // This is to distribute the accesses to unique addresses across the
    // workgroups, once the size of the access excedes the workgroup width.
    const uint32_t workgroup_width = local_x * NITER * NUNROLL;

    StagingBuffer in_buf(context(), vkapi::kFloat, range / sizeof(float));
    StagingBuffer out_buf(
        context(), vkapi::kFloat, VEC_WIDTH * app.nthread_logic);
    vkapi::PipelineBarrier pipeline_barrier{};

    auto shader_name = "buf_bandwidth_" + memtype_lower;

    auto time = benchmark_on_gpu(shader_name, 10, [&]() {
      context()->submit_compute_job(
          VK_KERNEL_FROM_STR(shader_name),
          pipeline_barrier,
          {global_x, 1, 1},
          {local_x, 1, 1},
          {SV(NITER),
           SV(nvec_access),
           SV(local_x),
           SV(addr_mask),
           SV(workgroup_width)},
          VK_NULL_HANDLE,
          0,
          in_buf.buffer(),
          out_buf.buffer());
    });

    const uint32_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
    auto gbps = SIZE_TRANS * 1e-3 / time;
    std::cout << memtype << " bandwidth accessing \t" << access_size
              << "\tB unique data is \t" << gbps << " \tgbps (\t" << time
              << "\tus)" << std::endl;
    return gbps;
  };

  double max_bandwidth = 0;
  double min_bandwidth = DBL_MAX;
  for (uint32_t access_size = VEC_SIZE; access_size < range; access_size *= 2) {
    double gbps = bench(access_size);
    max_bandwidth = std::max(gbps, max_bandwidth);
    min_bandwidth = std::min(gbps, min_bandwidth);
  }

  std::cout << "Max" << memtype << "Bandwidth (GB/s)," << max_bandwidth
            << std::endl;
  std::cout << "Min" << memtype << "Bandwidth (GB/s)," << min_bandwidth
            << std::endl;
}

void buf_bandwidth(const App& app) {
  if (!app.enabled("buffer_bandwidth")) {
    std::cout << "Skipped Memory Bandwidth" << std::endl;
    return;
  }

  std::cout << "\n------ Memory Bandwidth ------" << std::endl;
  // Maximum memory space read - 128MB
  // For regular devices, bandwidth plateaus at less memory than this, so more
  // is not needed.
  const uint32_t RANGE = app.get_config("buffer_bandwidth", "range");
  _bandwidth(app, "Buffer", RANGE);
}

void ubo_bandwidth(const App& app) {
  if (!app.enabled("ubo_bandwidth")) {
    std::cout << "Skipped UBO Bandwidth" << std::endl;
    return;
  }

  std::cout << "\n------ UBO Bandwidth ------" << std::endl;
  const uint32_t RANGE = app.get_config("ubo_bandwidth", "range");
  _bandwidth(app, "UBO", RANGE);
}

void shared_mem_bandwidth(const App& app) {
  if (!app.enabled("shared_bandwidth")) {
    std::cout << "Skipped Shared Memory Bandwidth" << std::endl;
    return;
  }

  std::cout << "\n------ Shared Bandwidth ------" << std::endl;
  const uint32_t RANGE = app.max_shared_mem_size;
  _bandwidth(app, "Shared", RANGE);
}
} // namespace gpuinfo
