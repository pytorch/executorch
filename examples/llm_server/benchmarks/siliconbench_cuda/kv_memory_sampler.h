#pragma once

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
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

// Older retained SDKs have no private KV allocator. Preserve their behavior
// while exposing both components when the allocation-only candidate is used.
struct EmptySequenceMemoryStatistics {
  bool available{false};
  bool query_ok{true};
  uint64_t used_bytes{0}, reserved_bytes{0};
  uint64_t used_high_bytes{0}, reserved_high_bytes{0};
};
template <class Pool>
auto private_sequence_statistics(int device, int)
    -> decltype(Pool::sequence_memory_pool_statistics(device)) {
  return Pool::sequence_memory_pool_statistics(device);
}
template <class Pool>
EmptySequenceMemoryStatistics private_sequence_statistics(int, long) {
  return {};
}

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
        const auto private_kv = private_sequence_statistics<
            executorch::backends::cuda::CudaKVPool>(0, 0);
        sample["private_allocator_pool_available"] = private_kv.available;
        sample["private_allocator_query_ok"] = private_kv.query_ok;
        sample["private_allocator_used_bytes"] = private_kv.used_bytes;
        sample["private_allocator_reserved_bytes"] = private_kv.reserved_bytes;
        sample["private_allocator_used_high_bytes"] = private_kv.used_high_bytes;
        sample["private_allocator_reserved_high_bytes"] = private_kv.reserved_high_bytes;
        const auto pool = executorch::extension::cuda::CudaAllocator::pool_for_device(0);
        sample["shared_allocator_pool_available"] = pool != nullptr;
        sample["allocator_pool_available"] = pool != nullptr || private_kv.available;
        if (device_status == cudaSuccess) {
          uint64_t used = 0, reserved = 0, used_high = 0, reserved_high = 0;
          cudaError_t a = cudaSuccess, b = cudaSuccess, c = cudaSuccess, d = cudaSuccess;
          if (pool != nullptr) {
            a = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
            b = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
            c = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high);
            d = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_high);
          }
          sample["allocator_query_status"] = {static_cast<int>(a), static_cast<int>(b),
                                              static_cast<int>(c), static_cast<int>(d)};
          if (sample["allocator_pool_available"].get<bool>() && private_kv.query_ok &&
              a == cudaSuccess && b == cudaSuccess && c == cudaSuccess && d == cudaSuccess) {
            sample["shared_allocator_used_bytes"] = used;
            sample["shared_allocator_reserved_bytes"] = reserved;
            sample["shared_allocator_used_high_bytes"] = used_high;
            sample["shared_allocator_reserved_high_bytes"] = reserved_high;
            sample["allocator_used_bytes"] = used + private_kv.used_bytes;
            sample["allocator_reserved_bytes"] = reserved + private_kv.reserved_bytes;
            sample["allocator_used_high_bytes"] = used_high + private_kv.used_high_bytes;
            sample["allocator_reserved_high_bytes"] = reserved_high + private_kv.reserved_high_bytes;
            sample["allocator_high_watermark_semantics"] =
                "Sum of component high watermarks; upper bound. Sample current values for peaks.";
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
