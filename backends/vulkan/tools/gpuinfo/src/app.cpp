/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>
#include <folly/json.h>
#include <fstream>
#include <iostream>

#include "stats.h"
#include "utils.h"

using namespace vkapi;

class App {
 private:
  size_t buf_cache_size_;
  uint32_t max_shared_mem_size_;
  uint32_t sm_count_;
  uint32_t nthread_logic_;
  uint32_t subgroup_size_;
  uint32_t max_tex_width_;
  uint32_t max_tex_height_;
  uint32_t max_tex_depth_;
  folly::dynamic config_;

  std::vector<int64_t> _whd_to_nchw(std::vector<int64_t> sizes) {
    const int64_t W = sizes[0];
    const int64_t H = sizes[1];
    const int64_t D = sizes[2];

    // Channels-packed: {W, H, D} = {W, H, (C / 4) * N}
    return {1, D * 4, H, W};
  }

  float _get_config(const std::string& test, const std::string& key) {
    if (config_[test].empty()) {
      throw std::runtime_error("Missing config for " + test);
    }

    if (!config_[test][key].isNumber()) {
      throw std::runtime_error(
          "Config for " + test + "." + key + " is not a number");
    }

    float value;
    if (config_[test][key].isDouble()) {
      value = config_[test][key].getDouble();
    } else {
      value = config_[test][key].getInt();
    }

    std::cout << "Read value for " << test << "." << key << " = " << value
              << std::endl;
    return value;
  }

  bool _enabled(const std::string& test) {
    if (config_.empty() || config_[test].empty() ||
        !config_[test]["enabled"].isBool()) {
      return true;
    }
    return config_[test]["enabled"].getBool();
  }

 public:
  App() {
    context()->initialize_querypool();

    std::cout << context()->adapter_ptr()->stringize() << std::endl
              << std::endl;

    auto cl_device = get_cl_device();

    sm_count_ = cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    nthread_logic_ = cl_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    buf_cache_size_ = cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    max_shared_mem_size_ = cl_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    max_tex_width_ = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
    max_tex_height_ = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
    max_tex_depth_ = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    VkPhysicalDeviceProperties2 props2{};

    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    subgroup_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    vkGetPhysicalDeviceProperties2(
        context()->adapter_ptr()->physical_handle(), &props2);
    subgroup_size_ = subgroup_props.subgroupSize;

    std::cout << std::endl;
    std::cout << "SM count," << sm_count_ << std::endl;
    std::cout << "Logic Thread Count," << nthread_logic_ << std::endl;
    std::cout << "Cache Size," << buf_cache_size_ << std::endl;
    std::cout << "Shared Memory Size," << max_shared_mem_size_ << std::endl;
    std::cout << "SubGroup Size," << subgroup_size_ << std::endl;
    std::cout << "MaxTexWidth," << max_tex_width_ << std::endl;
    std::cout << "MaxTexHeight," << max_tex_height_ << std::endl;
    std::cout << "MaxTexDepth," << max_tex_depth_ << std::endl;
  }

  void load_config(std::string file_path) {
    std::ifstream file(file_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string json_str = buffer.str();
    if (json_str.empty()) {
      throw std::runtime_error(
          "Failed to read config file from " + file_path + ".");
    }
    config_ = folly::parseJson(json_str);
  }

  void reg_count() {
    if (!_enabled("reg_count")) {
      std::cout << "Skipped Register Count" << std::endl;
      return;
    }

    std::cout << std::endl;
    std::cout << "------ Register Count ------" << std::endl;
    const uint32_t NREG_MIN = 1;
    const uint32_t NREG_MAX = 512;
    const uint32_t NREG_STEP = 1;

    const double COMPENSATE = _get_config("reg_count", "compensate");
    const double THRESHOLD = _get_config("reg_count", "threshold");

    const uint32_t NGRP_MIN = 1;
    const uint32_t NGRP_MAX = 64;
    const uint32_t NGRP_STEP = 1;

    uint32_t NITER;

    auto bench = [&](uint32_t ngrp, uint32_t nreg) {
      StorageBuffer buffer(context(), vkapi::kFloat, 1);
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
            buffer.buffer());
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
    if (!_enabled("buf_cacheline_size")) {
      std::cout << "Skipped Buffer Cacheline Size" << std::endl;
      return;
    }

    std::cout << std::endl;
    std::cout << "------ Buffer Cacheline Size ------" << std::endl;

    const double COMPENSATE = _get_config("buf_cacheline_size", "compensate");
    const double THRESHOLD = _get_config("buf_cacheline_size", "threshold");

    const uint32_t PITCH = buf_cache_size_ / nthread_logic_;
    const uint32_t BUF_SIZE = buf_cache_size_;
    const uint32_t MAX_STRIDE = PITCH;

    uint32_t NITER;

    auto bench = [&](int stride) {
      StorageBuffer in_buf(context(), vkapi::kFloat, BUF_SIZE);
      StorageBuffer out_buf(context(), vkapi::kFloat, 1);
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

 private:
  void _bandwidth(std::string memtype, uint32_t range) {
    auto memtype_lower = memtype;
    std::transform(
        memtype_lower.begin(),
        memtype_lower.end(),
        memtype_lower.begin(),
        [](unsigned char c) { return std::tolower(c); });

    auto test_name = memtype_lower + "_bandwidth";

    // Cache lines flushed
    const uint32_t NFLUSH = _get_config(test_name, "nflush");
    // Number of loop unrolls. Changing this value requires an equal change in
    // buf_bandwidth.yaml
    const uint32_t NUNROLL = _get_config(test_name, "nunroll");
    // Number of iterations. Increasing this value reduces noise in exchange for
    // higher latency.
    const uint32_t NITER = _get_config(test_name, "niter");
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
    const uint32_t NTHREAD =
        memtype == "Shared" ? NVEC : NVEC / NREAD_PER_THREAD;
    // Occupy all threads
    const uint32_t local_x = nthread_logic_;
    // Ensure that global is a multiple of local, and distribute across all SMs
    const uint32_t global_x =
        (NTHREAD / local_x * local_x) * sm_count_ * NFLUSH;

    auto bench = [&](uint32_t access_size) {
      // Number of vectors that fit in this iteration
      const uint32_t nvec_access = access_size / VEC_SIZE;

      StorageBuffer in_buf(context(), vkapi::kFloat, range / sizeof(float));
      StorageBuffer out_buf(
          context(), vkapi::kFloat, VEC_WIDTH * nthread_logic_);
      vkapi::PipelineBarrier pipeline_barrier{};

      auto shader_name = "buf_bandwidth_" + memtype_lower;

      auto time = benchmark_on_gpu(shader_name, 10, [&]() {
        context()->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            pipeline_barrier,
            {global_x, 1, 1},
            {local_x, 1, 1},
            {SV(NITER), SV(nvec_access), SV(local_x)},
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
    for (uint32_t access_size = VEC_SIZE; access_size < range;
         access_size *= 2) {
      double gbps = bench(access_size);
      max_bandwidth = std::max(gbps, max_bandwidth);
      min_bandwidth = std::min(gbps, min_bandwidth);
    }

    std::cout << "Max" << memtype << "Bandwidth (GB/s)," << max_bandwidth
              << std::endl;
    std::cout << "Min" << memtype << "Bandwidth (GB/s)," << min_bandwidth
              << std::endl;
  }

 public:
  void buf_bandwidth() {
    if (!_enabled("buffer_bandwidth")) {
      std::cout << "Skipped Memory Bandwidth" << std::endl;
      return;
    }

    std::cout << "\n------ Memory Bandwidth ------" << std::endl;
    // Maximum memory space read - 128MB
    // For regular devices, bandwidth plateaus at less memory than this, so more
    // is not needed.
    const uint32_t RANGE = _get_config("buffer_bandwidth", "range");
    _bandwidth("Buffer", RANGE);
  }

  void ubo_bandwidth() {
    if (!_enabled("ubo_bandwidth")) {
      std::cout << "Skipped UBO Bandwidth" << std::endl;
      return;
    }

    std::cout << "\n------ UBO Bandwidth ------" << std::endl;
    const uint32_t RANGE = _get_config("ubo_bandwidth", "range");
    _bandwidth("UBO", RANGE);
  }

  void shared_mem_bandwidth() {
    if (!_enabled("shared_mem_bandwidth")) {
      std::cout << "Skipped Shared Memory Bandwidth" << std::endl;
      return;
    }

    std::cout << "\n------ Shared Bandwidth ------" << std::endl;
    const uint32_t RANGE = max_shared_mem_size_;
    _bandwidth("Shared", RANGE);
  }

  void tex_bandwidth() {
    if (!_enabled("tex_bandwidth")) {
      std::cout << "Skipped Texture Bandwidth" << std::endl;
      return;
    }

    for (int dim = 0; dim < 3; dim++) {
      std::cout << "\n------ Texture Bandwidth (Dim = " << dim << ") ------"
                << std::endl;
      const uint32_t MAX_SIZE = dim == 0 ? max_tex_width_
          : dim == 1                     ? max_tex_height_
                                         : max_tex_depth_;

      // rgba, float
      const uint32_t VEC_WIDTH = 4;
      const uint32_t VEC_SIZE = VEC_WIDTH * sizeof(float);
      const uint32_t NVEC = MAX_SIZE;

      const uint32_t RANGE = NVEC * VEC_SIZE;

      // Cache lines flushed
      const uint32_t NFLUSH = _get_config("tex_bandwidth", "nflush");
      // Number of loop unrolls. Changing this value requires an equal change in
      // tex_bandwidth.yaml
      const uint32_t NUNROLL = _get_config("tex_bandwidth", "nunroll");
      // Number of iterations. Increasing this value reduces noise in exchange
      // for higher latency.
      const uint32_t NITER = _get_config("tex_bandwidth", "niter");
      // Number of memory reads per thread
      const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;
      // Number of threads needed to read all texells
      const uint32_t NTHREAD = NVEC;
      // Occupy all threads
      const uint32_t local_x = nthread_logic_;
      // Ensure that global is a multiple of local, and distribute across all
      // SMs
      const uint32_t global_x =
          (NTHREAD / local_x * local_x) * sm_count_ * NFLUSH;

      auto shader_name = "tex_bandwidth_" + std::to_string(dim);

      std::vector<int64_t> sizes_whd = {MAX_SIZE, 1, 1};
      if (dim == 1) {
        sizes_whd = {1, MAX_SIZE, 1};
      } else if (dim == 2) {
        sizes_whd = {1, 1, MAX_SIZE};
      }
      auto sizes_nchw = _whd_to_nchw(sizes_whd);

      vTensor in_tensor =
          api::vTensor(api::context(), sizes_nchw, vkapi::kFloat);

      auto bench = [&](uint32_t access_size, uint32_t dim) {
        // Number of texels that fit in this iteration
        const uint32_t ntexel_access = access_size / VEC_SIZE;

        StorageBuffer out_buf(
            context(), vkapi::kFloat, VEC_WIDTH * nthread_logic_);
        vkapi::PipelineBarrier pipeline_barrier{};

        auto time = benchmark_on_gpu(shader_name, 10, [&]() {
          context()->submit_compute_job(
              VK_KERNEL_FROM_STR(shader_name),
              pipeline_barrier,
              {global_x, 1, 1},
              {local_x, 1, 1},
              {SV(NITER), SV(ntexel_access), SV(local_x), SV(dim)},
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
  void warp_size(bool verbose = false) {
    if (!_enabled("warp_size")) {
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
    const double COMPENSATE = _get_config("warp_size", "compensate");
    const double THRESHOLD = _get_config("warp_size", "threshold");

    uint32_t NITER;

    auto bench = [&](uint32_t nthread) {
      StorageBuffer out_buf(context(), vkapi::kInt, nthread_logic_);
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

    uint32_t warp_size = subgroup_size_;
    DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);

    // We increase the number of threads until we hit a jump in the data.
    uint32_t nthread = 1;
    for (; nthread <= nthread_logic_; ++nthread) {
      double time = bench(nthread);
      std::cout << "nthread=\t" << nthread << "\t(\t" << time << "\tus)"
                << std::endl;
      if (dj.push(time)) {
        warp_size = nthread - 1;
        break;
      }
    }
    if (nthread >= nthread_logic_) {
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
      StorageBuffer out_buf(context(), vkapi::kInt, nthread_logic_);
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

      std::vector<int32_t> data(nthread_logic_);
      copy_staging_to_ptr(out_buf, data.data(), out_buf.nbytes());

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
    for (; i <= nthread_logic_; ++i) {
      uint32_t nascend = bench_sm(i);
      if (nascend != i) {
        warp_size_scheduler = nascend;
        break;
      }
    }
    if (i > nthread_logic_) {
      std::cout << "Unable to conclude an SM Warp Size." << std::endl;
    }

    std::cout << "PhysicalWarpSize," << warp_size << std::endl;
    std::cout << "SMWarpSize," << warp_size_scheduler << std::endl;
  }
};

int main(int argc, const char** argv) {
  App app;

  std::string file_path = "config.json";
  if (argc > 1) {
    file_path = argv[1];
  };
  app.load_config(file_path);

  app.reg_count();
  app.buf_cacheline_size();
  app.buf_bandwidth();
  app.ubo_bandwidth();
  app.shared_mem_bandwidth();
  app.warp_size();
  app.tex_bandwidth();

  return 0;
}
