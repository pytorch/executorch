// Experimental shared-pool maintenance at an explicit idle boundary.
// Caller must admit no model execution concurrently with this function.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <executorch/extension/cuda/cuda_allocator.h>

namespace siliconbench {
struct IdleSharedWorkset {
  cudaError_t error{cudaSuccess};
  bool enabled{false};
  uint64_t used{0};
  uint64_t used_high{0};
  uint64_t reserved_before{0};
  uint64_t reserved_after{0};
  uint64_t keep{0};
};

inline IdleSharedWorkset refresh_idle_shared_workset() {
  IdleSharedWorkset result;
  const char* mode = std::getenv("ET_CUDA_SHARED_IDLE_REFRESH");
  result.enabled = mode && mode[0] == '1' && mode[1] == '\0';
  int device = -1;
  result.error = cudaGetDevice(&device);
  if (result.error != cudaSuccess) return result;
  // Both arms fence; callers record this maintenance cost separately and
  // include the next cold/warm request in their performance comparison.
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) return result;
  auto pool = executorch::extension::cuda::CudaAllocator::pool_for_device(device);
  if (!pool) return result;
  result.error = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &result.used);
  if (result.error != cudaSuccess) return result;
  result.error = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &result.used_high);
  if (result.error != cudaSuccess) return result;
  result.error = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &result.reserved_before);
  if (result.error != cudaSuccess) return result;
  const uint64_t floor = std::max(result.used, result.used_high);
  constexpr uint64_t allowance = 256ULL << 20;
  result.keep = floor > std::numeric_limits<uint64_t>::max() - allowance
      ? std::numeric_limits<uint64_t>::max() : floor + allowance;
  if (result.enabled && result.reserved_before > result.keep) {
    result.error = cudaMemPoolTrimTo(pool, result.keep);
    if (result.error != cudaSuccess) return result;
  }
  result.error = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &result.reserved_after);
  return result;
}
} // namespace siliconbench
