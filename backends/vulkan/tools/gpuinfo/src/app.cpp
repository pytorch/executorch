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

using namespace vkapi;

class App {
 private:
  size_t buf_cache_size_;
  uint32_t sm_count_;
  uint32_t nthread_logic_;

 public:
  App() {
    context()->initialize_querypool();

    std::cout << context()->adapter_ptr()->stringize() << std::endl
              << std::endl;

    auto cl_device = get_cl_device();

    sm_count_ = cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    nthread_logic_ = cl_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    buf_cache_size_ = cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

    std::cout << std::endl;
    std::cout << "SM count," << sm_count_ << std::endl;
    std::cout << "Logic Thread Count," << nthread_logic_ << std::endl;
    std::cout << "Cache Size," << buf_cache_size_ << std::endl;
  }

  void reg_count() {
    std::cout << std::endl;
    std::cout << "------ Register Count ------" << std::endl;
    const uint32_t NREG_MIN = 1;
    const uint32_t NREG_MAX = 512;
    const uint32_t NREG_STEP = 1;

    // TODO: Make these values configurable
    const double COMPENSATE = 0.01;
    const double THRESHOLD = 3;

    const uint32_t NGRP_MIN = 1;
    const uint32_t NGRP_MAX = 64;
    const uint32_t NGRP_STEP = 1;

    uint32_t NITER;

    auto bench = [&](uint32_t ngrp, uint32_t nreg) {
      size_t len = sizeof(float);
      StorageBuffer buffer(context(), vkapi::kFloat, len);
      ParamsBuffer params(context(), int32_t(len));
      vkapi::PipelineBarrier pipeline_barrier{};

      auto shader_name = "reg_count_" + std::to_string(nreg);

      auto time = benchmark_on_gpu(shader_name, 100, [&]() {
        context()->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            pipeline_barrier,
            {1, ngrp, 1},
            {1, 1, 1},
            {SV(NITER)},
            VK_NULL_HANDLE,
            0,
            buffer.buffer(),
            params.buffer());
      });
      return time;
    };

    std::cout << "Calculating NITER..." << std::endl;
    ensure_min_niter(1000, NITER, [&]() { return bench(1, NREG_MIN); });
    std::cout << "NITER," << NITER << std::endl;

    uint32_t nreg_max;

    DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
    uint32_t nreg = NREG_MIN;
    for (; nreg <= NREG_MAX; nreg += NREG_STEP) {
      double time = bench(1, nreg);
      std::cout << "Testing nreg=\t" << nreg << "\tTime=\t" << time
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
      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
      for (auto ngrp = NGRP_MIN; ngrp <= NGRP_MAX; ngrp += NGRP_STEP) {
        auto time = bench(ngrp, nreg);
        std::cout << "Testing occupation (nreg=" << nreg << "); ngrp=" << ngrp
                  << ", time=" << time << " us" << std::endl;

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
    std::cout << "NITER," << NITER << std::endl;
    std::cout << "Max registers," << nreg_max << std::endl;
    std::cout << "Concurrent full single thread workgroups," << ngrp_full
              << std::endl;
    std::cout << "Concurrent half single thread workgroups," << ngrp_half
              << std::endl;
    std::cout << "Register type," << reg_ty << std::endl;
  }

  void buf_cacheline_size() {
    std::cout << std::endl;
    std::cout << "------ Buffer Cacheline Size ------" << std::endl;

    // TODO: Make these values configurable
    const double COMPENSATE = 0.01;
    const double THRESHOLD = 10;

    const uint32_t PITCH = buf_cache_size_ / nthread_logic_;
    const uint32_t BUF_SIZE = buf_cache_size_;
    const uint32_t MAX_STRIDE = PITCH;

    uint32_t NITER;

    auto bench = [&](int stride) {
      size_t len = sizeof(float);
      StorageBuffer in_buf(context(), vkapi::kFloat, BUF_SIZE);
      StorageBuffer out_buf(context(), vkapi::kFloat, len);
      vkapi::PipelineBarrier pipeline_barrier{};

      auto shader_name = "buf_cacheline_size";

      auto time = benchmark_on_gpu(shader_name, 100, [&]() {
        context()->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            pipeline_barrier,
            {nthread_logic_, 1, 1},
            {nthread_logic_, 1, 1},
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
      cacheline_size = MAX_STRIDE;
    }

    std::cout << "BufTopLevelCachelineSize," << cacheline_size << std::endl;
  }
};

int main(int argc, const char** argv) {
  App app;

  // TODO: Allow user to skip tests
  app.reg_count();
  app.buf_cacheline_size();

  return 0;
}
