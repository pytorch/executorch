/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION CL_TARGET_OPENCL_VERSION
#include <CL/opencl.hpp>

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

std::vector<int64_t> whd_to_nchw(std::vector<int64_t> sizes) {
  const int64_t W = sizes[0];
  const int64_t H = sizes[1];
  const int64_t D = sizes[2];

  // Channels-packed: {W, H, D} = {W, H, (C / 4) * N}
  return {1, D * 4, H, W};
}

cl_platform_id get_cl_platform_id() {
  cl_uint nplatform_id;
  clGetPlatformIDs(0, nullptr, &nplatform_id);
  std::vector<cl_platform_id> platform_ids;
  platform_ids.resize(nplatform_id);
  clGetPlatformIDs(nplatform_id, platform_ids.data(), nullptr);
  return platform_ids[0];
}

cl_device_id get_cl_dev_id(cl_platform_id platform_id) {
  cl_uint ndev_id;
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev_id);
  std::vector<cl_device_id> dev_ids;
  dev_ids.resize(ndev_id);
  clGetDeviceIDs(
      platform_id, CL_DEVICE_TYPE_ALL, ndev_id, dev_ids.data(), nullptr);
  return dev_ids[0];
}

cl::Device get_cl_device() {
  auto platform_id = get_cl_platform_id();
  auto dev_id = get_cl_dev_id(platform_id);
  cl::Device dev(dev_id);
  return dev;
}
