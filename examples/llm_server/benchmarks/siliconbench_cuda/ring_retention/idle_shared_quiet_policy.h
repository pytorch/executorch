// A stricter request-idle policy; next-request cold allocation must be measured.
#pragma once
#include "idle_shared_workset.h"

namespace siliconbench {
inline IdleSharedWorkset refresh_idle_shared_selected_workset(bool keep_high) {
  if (keep_high) return refresh_idle_shared_workset();
  IdleSharedWorkset r;
  const char* mode=std::getenv("ET_CUDA_SHARED_IDLE_REFRESH");
  r.enabled=mode&&mode[0]=='1'&&mode[1]=='\0';
  int device=-1;r.error=cudaGetDevice(&device);
  if(r.error!=cudaSuccess)return r;
  r.error=cudaDeviceSynchronize();if(r.error!=cudaSuccess)return r;
  auto pool=executorch::extension::cuda::CudaAllocator::pool_for_device(device);
  if(!pool)return r;
  r.error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&r.used);
  if(r.error!=cudaSuccess)return r;
  r.error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemHigh,&r.used_high);
  if(r.error!=cudaSuccess)return r;
  r.error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&r.reserved_before);
  if(r.error!=cudaSuccess)return r;
  constexpr uint64_t allowance=256ULL<<20;
  r.keep=r.used>std::numeric_limits<uint64_t>::max()-allowance?std::numeric_limits<uint64_t>::max():r.used+allowance;
  if(r.enabled&&r.reserved_before>r.keep){r.error=cudaMemPoolTrimTo(pool,r.keep);if(r.error!=cudaSuccess)return r;}
  r.error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&r.reserved_after);
  return r;
}
} // namespace siliconbench
