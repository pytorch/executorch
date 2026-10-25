// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license in the repository LICENSE file.
// Isolated KV growth diagnostic. Full-size mode reserves placeholder bytes;
// it is not a loaded model, model correctness, or inference performance test.
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cu = executorch::backends::cuda;
namespace aoti = executorch::backends::aoti;
using executorch::runtime::Error;
using executorch::extension::cuda::CudaAllocator;
using json = nlohmann::json;

static void cuda_ok(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static Error empty_count(aoti::AOTInductorModelContainerHandle, size_t* n) {
  *n = 0; return Error::Ok;
}
static Error empty_name(aoti::AOTInductorModelContainerHandle, size_t, const char**) {
  return Error::InvalidArgument;
}
static Error empty_pairs(aoti::AOTInductorModelContainerHandle,
                        const aoti::AOTInductorConstantMapEntry*, size_t, bool, bool) {
  return Error::Ok;
}
static json memory() {
  size_t free = 0, total = 0;
  cuda_ok(cudaMemGetInfo(&free, &total));
  json r = {{"device_used_bytes", total-free}, {"device_total_bytes", total}};
  auto pool = CudaAllocator::pool_for_device(0);
  if (pool) {
    for (const auto& x : std::vector<std::pair<const char*, cudaMemPoolAttr>>{
        {"pool_used", cudaMemPoolAttrUsedMemCurrent},
        {"pool_reserved", cudaMemPoolAttrReservedMemCurrent},
        {"pool_used_high", cudaMemPoolAttrUsedMemHigh},
        {"pool_reserved_high", cudaMemPoolAttrReservedMemHigh}}) {
      uint64_t bytes = 0;
      cuda_ok(cudaMemPoolGetAttribute(pool, x.second, &bytes));
      r[x.first] = bytes;
    }
  }
  return r;
}
static unsigned char pattern(size_t layer, bool value) {
  return static_cast<unsigned char>(1 + (layer*2 + value)%251);
}
static size_t check(cu::CudaKVPool& pool, const std::vector<cu::CudaKVPool::Layer>& layers,
                    int64_t live, bool full_check) {
  size_t checked = 0;
  for (size_t i = 0; i < layers.size(); ++i) {
    size_t bytes = std::min<int64_t>(live, layers[i].declared_rows)
        * layers[i].n_kv_heads * layers[i].head_dim * 2;
    size_t part = full_check ? bytes : std::min<size_t>(bytes, 32768);
    for (bool v : {false, true}) {
      auto* data = static_cast<unsigned char*>(pool.layer_data(i, v));
      for (size_t offset : full_check ? std::vector<size_t>{0}
                    : std::vector<size_t>{0, (bytes-part)/2, bytes-part}) {
        std::vector<unsigned char> host(part);
        cuda_ok(cudaMemcpy(host.data(), data+offset, part, cudaMemcpyDeviceToHost));
        if (!std::all_of(host.begin(), host.end(), [i,v](unsigned char c) {
              return c == pattern(i,v); })) throw std::runtime_error("KV prefix changed");
        checked += part;
      }
    }
  }
  return checked;
}
static void fill(cu::CudaKVPool& pool, const std::vector<cu::CudaKVPool::Layer>& layers,
                 int64_t live, cudaStream_t stream) {
  for (size_t i = 0; i < layers.size(); ++i) {
    size_t bytes = std::min<int64_t>(live, layers[i].declared_rows)
        * layers[i].n_kv_heads * layers[i].head_dim * 2;
    for (bool v : {false, true})
      cuda_ok(cudaMemsetAsync(pool.layer_data(i,v), pattern(i,v), bytes, stream));
  }
  cuda_ok(cudaStreamSynchronize(stream));
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: probe mini|full baseline|candidate");
    bool full = std::string(argv[1]) == "full";
    bool candidate = std::string(argv[2]) == "candidate";
    if (!full && std::string(argv[1]) != "mini") throw std::runtime_error("bad scope");
    if (!candidate && std::string(argv[2]) != "baseline") throw std::runtime_error("bad arm");
    cuda_ok(cudaSetDevice(0));
    cudaStream_t stream = nullptr; cuda_ok(cudaStreamCreate(&stream));
    void* placeholder = nullptr;
    // Disclosed headroom control approximates model plus runtime residency.
    const uint64_t placeholder_bytes = full ? 17ULL*(1ULL<<29) : 0;
    if (placeholder_bytes) cuda_ok(cudaMalloc(&placeholder, placeholder_bytes));
    const int columns = 19;
    std::vector<cu::CudaKVPool::Layer> layers;
    for (int i = 0; i < 48; ++i) {
      bool ring = i%6 != 5;  // Exactly40 ring and8 flat layers, like Gemma.
      layers.push_back({ring ? 8 : 1, full ? (ring ? 256 : 512) : 8,
        columns*(ring ? (full ? 2047 : 65) : (full ? 131072 : 512)), true});
    }
    json stages = json::array(); size_t checked = 0; bool failed = false;
    {
      cu::CudaDelegateHandle handle;
      handle.get_num_constants = empty_count;
      handle.get_constant_name = empty_name;
      handle.get_constant_original_fqn = empty_name;
      handle.update_user_managed_constant_buffer_pairs = empty_pairs;
      cu::CudaKVPool pool(layers, {}, aoti::slim::c10::ScalarType::BFloat16,
                         columns*(full ? 512 : 16));
      // Empty descriptors set the device for proper teardown; no AOTI model.
      auto noted = pool.note_handle(&handle);
      if (!noted.ok()) throw std::runtime_error("cannot initialize pool metadata");
      int64_t live = 0;
      std::vector<int> ends = full ? std::vector<int>{1008,2016,3024,6048,12000}
                                  : std::vector<int>{32,64,96,200};
      for (int end : ends) {
        std::vector<void*> old;
        for (size_t i=0; i<layers.size(); ++i) old.push_back(pool.layer_data(i,false));
        int64_t old_rows = pool.rows();
        Error e = pool.prepare(columns*end, live, stream);
        if (e != Error::Ok) {
          (void)cudaGetLastError(); cuda_ok(cudaStreamSynchronize(stream));
          bool pointers_unchanged = true;
          for (size_t i=0; i<layers.size(); ++i)
            pointers_unchanged &= old[i] == pool.layer_data(i,false);
          checked += check(pool, layers, live, !full);
          stages.push_back({{"end",end}, {"error",static_cast<int>(e)},
              {"rows",pool.rows()}, {"live_bytes",pool.allocated_bytes()},
              {"rollback_pointers_unchanged",pointers_unchanged}, {"memory",memory()}});
          if (e != Error::MemoryAllocationFailed || pool.rows()!=old_rows || !pointers_unchanged)
            throw std::runtime_error("growth failure broke rollback or was not allocation OOM");
          if (candidate || !full) throw std::runtime_error("candidate or mini growth failed");
          if (pool.prepare(old_rows, live, stream) != Error::Ok)
            throw std::runtime_error("old capacity became unusable after OOM");
          failed = true; break;
        }
        cuda_ok(cudaStreamSynchronize(stream));
        if (live) checked += check(pool, layers, live, !full);
        fill(pool, layers, columns*end, stream);
        bool ring_same = old[0] && old[0] == pool.layer_data(0,false);
        if (candidate && end == (full ? 3024 : 96) && !ring_same)
          throw std::runtime_error("near-cap ring reallocated at following growth");
        stages.push_back({{"end",end}, {"error",0}, {"rows",pool.rows()},
            {"live_bytes",pool.allocated_bytes()}, {"first_ring_pointer_unchanged",ring_same},
            {"memory",memory()}});
        live = columns*end;
        if (pool.mark_step_done()!=Error::Ok) throw std::runtime_error("step event failed");
      }
    }
    if (placeholder) cuda_ok(cudaFree(placeholder));
    cuda_ok(cudaStreamDestroy(stream));
    std::cout << json({{"scope",full?"full-placeholder":"mini"},
      {"arm",candidate?"candidate":"baseline"}, {"stages",stages},
      {"checked_bytes",checked}, {"prefix_preserved",true},
      {"full_buffers_checked",!full}, {"placeholder_bytes",placeholder_bytes},
      {"baseline_expected_OOM_observed",failed}, {"model_loaded",false},
      {"actual_model_correctness_proven",false}}).dump() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl; return 1;
  }
}
