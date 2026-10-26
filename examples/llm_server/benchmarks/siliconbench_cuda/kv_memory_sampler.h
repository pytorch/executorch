#pragma once

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace siliconbench {

class KVMemorySampler final {
 public:
  KVMemorySampler(
      executorch::extension::llm::batching::Executor& executor,
      const std::string& backend,
      bool dflash) {
    const auto* value = std::getenv("ET_CUDA_BENCH_MEMORY_TRACE");
    if (value == nullptr || std::string(value) != "1") {
      return;
    }
    if (backend != "cuda" || dflash) {
      throw std::invalid_argument("Memory sampler requires solo CUDA");
    }
    auto& cuda = static_cast<
        executorch::backends::cuda::batching::CudaExecutor&>(executor);
    thread_ = std::thread([this, &cuda]() {
      const auto device_status = cudaSetDevice(0);
      for (;;) {
        const auto begin = std::chrono::steady_clock::now();
        const auto kv = cuda.kv_metrics(); // The cache owns the snapshot mutex.
        nlohmann::json sample = {
            {"pid", getpid()},
            {"physical_columns", kv.physical_columns},
            {"declared_columns", kv.declared_columns},
            {"live_sequences", kv.live_sequences},
            {"logical_length", kv.logical_length},
            {"max_full_capacity", kv.flat_capacity},
            {"live_KV_bytes", kv.allocated_bytes},
            {"row_growth_count", kv.growth_count},
            {"column_resize_count", kv.column_resize_count},
            {"cuda_set_device_status", static_cast<int>(device_status)},
        };
        const auto pool = executorch::extension::cuda::CudaAllocator::pool_for_device(0);
        sample["allocator_pool_available"] = pool != nullptr;
        if (pool != nullptr && device_status == cudaSuccess) {
          uint64_t used = 0, reserved = 0, used_high = 0, reserved_high = 0;
          const auto a = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
          const auto b = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
          const auto c = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high);
          const auto d = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_high);
          sample["allocator_query_status"] = {static_cast<int>(a), static_cast<int>(b),
                                              static_cast<int>(c), static_cast<int>(d)};
          if (a == cudaSuccess && b == cudaSuccess && c == cudaSuccess && d == cudaSuccess) {
            sample["allocator_used_bytes"] = used;
            sample["allocator_reserved_bytes"] = reserved;
            sample["allocator_used_high_bytes"] = used_high;
            sample["allocator_reserved_high_bytes"] = reserved_high;
          }
        }
        const auto end = std::chrono::steady_clock::now();
        sample["monotonic_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
        sample["query_begin_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count();
        sample["query_duration_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        const std::string line = "KV_MEM_SAMPLE:" + sample.dump() + "\n";
        std::fwrite(line.data(), 1, line.size(), stderr);
        std::fflush(stderr);
        std::unique_lock<std::mutex> lock(mutex_);
        if (condition_.wait_for(lock, std::chrono::milliseconds(250), [this]() { return stop_; })) {
          return;
        }
      }
    });
  }

  ~KVMemorySampler() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  KVMemorySampler(const KVMemorySampler&) = delete;
  KVMemorySampler& operator=(const KVMemorySampler&) = delete;

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_{false};
  std::thread thread_;
};

} // namespace siliconbench
