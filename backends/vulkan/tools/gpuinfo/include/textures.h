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

namespace gpuinfo {

// Textures are drastically different from buffers in terms of data layout.
// While buffers are a contiguous range of memory, textures are opaque objects
// defined by the vendor and it is possible that nearby points of data are not
// neighboring in memory. Likewise, data points are accessed in
// multi-dimensional patches instead of simple lines. This makes the stride
// method for figuring out the cache line size not applicable. To go around
// this, this experiment runs an increasing amount of threads accessing
// different datapoints in the texture and measures latency. If the cache line
// is big enough to contain all requested data for the amount of threads,
// latency will be low. When there are more threads and hence more data than
// what a single cache line can handle, a second line must be fetched,
// increasing latency in a measurable way.
void tex_cacheline_concurr(const App& app) {
  if (!app.enabled("tex_cacheline_concurr")) {
    std::cout << "Skipped Texture Cacheline Optimal Concurrency" << std::endl;
    return;
  }

  const uint32_t TEXEL_WIDTH = 4;
  const uint32_t TEXEL_SIZE = sizeof(float) * TEXEL_WIDTH;

  const double COMPENSATE =
      app.get_config("tex_cacheline_concurr", "compensate");
  const double THRESHOLD = app.get_config("tex_cacheline_concurr", "threshold");

  for (int dim = 0; dim < 3; ++dim) {
    std::cout << std::endl;
    std::cout << "------ Texture Cacheline Optimal Concurrency (dim = " << dim
              << ") ------" << std::endl;

    uint32_t NITER;

    const uint32_t IMG_OTHER_EDGE = dim == 0 ? app.max_tex_width
        : dim == 1                           ? app.max_tex_height
                                             : app.max_tex_depth;

    const uint32_t MAX_NTHREAD = std::min(app.nthread_logic, IMG_OTHER_EDGE);

    auto bench = [&](uint32_t nthread) {
      std::vector<int64_t> sizes_whd = {
          app.max_tex_width, app.max_tex_height, app.max_tex_depth};

      auto sizes_nchw = whd_to_nchw(sizes_whd);

      vTensor in_tensor =
          api::vTensor(api::context(), sizes_nchw, vkapi::kFloat);

      StagingBuffer out_buf(context(), vkapi::kFloat, TEXEL_WIDTH);

      vkapi::PipelineBarrier pipeline_barrier{};

      auto shader_name = "tex_cacheline_concurr_" + std::to_string(dim);

      auto time = benchmark_on_gpu(shader_name, 100, [&]() {
        context()->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            pipeline_barrier,
            {nthread, 1, 1},
            {nthread, 1, 1},
            {SV(NITER)},
            VK_NULL_HANDLE,
            0,
            in_tensor.image(),
            out_buf.buffer());
      });
      return time;
    };

    ensure_min_niter(1000, NITER, [&]() { return bench(1); });

    DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
    uint32_t nthread = 1;
    for (; nthread <= MAX_NTHREAD; ++nthread) {
      double time = bench(nthread);
      std::cout << "Testing nthread=\t" << nthread << "\t, time=\t" << time
                << std::endl;

      if (dj.push(time)) {
        auto max_concurrency = nthread - 1;
        std::cout << "TextureCachelineConcurrencyDim" << dim << " (B),"
                  << max_concurrency * TEXEL_SIZE << std::endl;
        break;
      }
    }
    if (nthread >= MAX_NTHREAD) {
      std::cout
          << "Unable to conclude an optimal texture cacheline concurrency for dim "
          << dim << std::endl;
    };
  }

  // TODO: Use concurrency information to obtain the cache line size for
  // textures as done in https://fburl.com/98xiou3g
}

void tex_bandwidth(const App& app) {
  if (!app.enabled("tex_bandwidth")) {
    std::cout << "Skipped Texture Bandwidth" << std::endl;
    return;
  }

  for (int dim = 0; dim < 3; dim++) {
    std::cout << "\n------ Texture Bandwidth (Dim = " << dim << ") ------"
              << std::endl;
    const uint32_t MAX_SIZE = dim == 0 ? app.max_tex_width
        : dim == 1                     ? app.max_tex_height
                                       : app.max_tex_depth;

    // rgba, float
    const uint32_t VEC_WIDTH = 4;
    const uint32_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    const uint32_t NVEC = MAX_SIZE;

    const uint32_t RANGE = NVEC * VEC_SIZE;

    // Cache lines flushed
    const uint32_t NFLUSH = app.get_config("tex_bandwidth", "nflush");
    // Number of loop unrolls. Changing this value requires an equal change in
    // tex_bandwidth.yaml
    const uint32_t NUNROLL = app.get_config("tex_bandwidth", "nunroll");
    // Number of iterations. Increasing this value reduces noise in exchange
    // for higher latency.
    const uint32_t NITER = app.get_config("tex_bandwidth", "niter");
    // Number of memory reads per thread
    const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;
    // Number of threads needed to read all texells
    const uint32_t NTHREAD = NVEC;
    // Occupy all threads
    const uint32_t local_x = app.nthread_logic;
    // Ensure that global is a multiple of local, and distribute across all
    // SMs
    const uint32_t global_x =
        (NTHREAD / local_x * local_x) * app.sm_count * NFLUSH;

    auto shader_name = "tex_bandwidth_" + std::to_string(dim);

    std::vector<int64_t> sizes_whd = {MAX_SIZE, 1, 1};
    if (dim == 1) {
      sizes_whd = {1, MAX_SIZE, 1};
    } else if (dim == 2) {
      sizes_whd = {1, 1, MAX_SIZE};
    }
    auto sizes_nchw = whd_to_nchw(sizes_whd);

    vTensor in_tensor = api::vTensor(api::context(), sizes_nchw, vkapi::kFloat);

    auto bench = [&](uint32_t access_size, uint32_t dim) {
      // Number of texels that fit in this iteration
      const uint32_t ntexel_access = access_size / VEC_SIZE;

      // The address mask works as a modulo because x % 2^n == x & (2^n - 1).
      // This will help us limit address accessing to a specific set of unique
      // addresses depending on the access size we want to measure.
      const uint32_t addr_mask = ntexel_access - 1;

      // This is to distribute the accesses to unique addresses across the
      // workgroups, once the size of the access excedes the workgroup width.
      const uint32_t workgroup_width = local_x * NITER * NUNROLL;

      StagingBuffer out_buf(
          context(), vkapi::kFloat, VEC_WIDTH * app.nthread_logic);
      vkapi::PipelineBarrier pipeline_barrier{};

      auto time = benchmark_on_gpu(shader_name, 10, [&]() {
        context()->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            pipeline_barrier,
            {global_x, 1, 1},
            {local_x, 1, 1},
            {SV(NITER),
             SV(ntexel_access),
             SV(local_x),
             SV(addr_mask),
             SV(workgroup_width)},
            VK_NULL_HANDLE,
            0,
            in_tensor.image(),
            out_buf.buffer());
      });

      const uint32_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
      double gbps = SIZE_TRANS * 1e-3 / time;
      std::cout << "Texture bandwidth accessing \t" << access_size
                << "\tB unique data is \t" << gbps << " \tgbps (\t" << time
                << "\tus)" << std::endl;
      return gbps;
    };

    double max_bandwidth = 0;
    double min_bandwidth = DBL_MAX;
    for (uint32_t access_size = VEC_SIZE; access_size < RANGE;
         access_size *= 2) {
      double gbps = bench(access_size, dim);
      max_bandwidth = std::max(gbps, max_bandwidth);
      min_bandwidth = std::min(gbps, min_bandwidth);
    }

    std::cout << "MaxTextureBandwidthDim" << dim << "(GB/s)," << max_bandwidth
              << std::endl;
    std::cout << "MinTextureBandwidthDim" << dim << "(GB/s)," << min_bandwidth
              << std::endl;
  }
}
} // namespace gpuinfo
