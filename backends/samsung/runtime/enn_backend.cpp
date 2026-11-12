/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/backends/samsung/runtime/logging.h>
#include <executorch/backends/samsung/runtime/profile.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/profiler.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace torch {
namespace executor {

using executorch::runtime::NamedDataMap;

const char ennWeightBufferKey[] = "enn_weight_buffer_hash_id";

namespace {
// Methods sharing weights (e.g. prefill and decode) reference the same buffer,
// so it is refcounted and released when the last delegate using it is
// destroyed.
struct SharedWeight {
  FreeableBuffer buffer;
  int32_t refs = 0;
};

std::mutex g_weight_cache_mutex;
std::unordered_map<std::string, SharedWeight> g_weight_cache;

Result<const FreeableBuffer*> acquire_weight(
    const NamedDataMap* named_data_map,
    const std::string& key) {
  std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
  auto it = g_weight_cache.find(key);
  if (it == g_weight_cache.end()) {
    ET_CHECK_OR_RETURN_ERROR(
        named_data_map != nullptr,
        InvalidArgument,
        "Weight-sharing delegate needs a NamedDataMap (.ptd) for key %s",
        key.c_str());
    Result<FreeableBuffer> loaded = named_data_map->get_data(key.c_str());
    ET_CHECK_OK_OR_RETURN_ERROR(
        loaded.error(), "Failed to load shared weight %s", key.c_str());
    it = g_weight_cache.emplace(key, SharedWeight{std::move(loaded.get()), 0})
             .first;
  }
  it->second.refs++;
  return &it->second.buffer;
}

void release_weight(const std::string& key) {
  std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
  auto it = g_weight_cache.find(key);
  if (it != g_weight_cache.end() && --it->second.refs == 0) {
    g_weight_cache.erase(it);
  }
}
} // namespace

class EnnBackend final : public PyTorchBackendInterface {
 public:
  ~EnnBackend() = default;

  bool is_available() const override {
    return true;
  }

  std::string decode_spec_value(const CompileSpec& spec) const {
    const char* char_ptr = static_cast<const char*>(spec.value.buffer);
    return std::string(char_ptr, spec.value.nbytes);
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    EXYNOS_ATRACE_NAME_LINE("backend_init");
    MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
    auto executor = runtime_allocator->allocateInstance<enn::EnnExecutor>();
    ET_CHECK_OR_RETURN_ERROR(
        executor != nullptr,
        MemoryAllocationFailed,
        "Failed to allocate EnnExecutor");
    new (executor) enn::EnnExecutor();
    const char* binary_buf_addr =
        reinterpret_cast<const char*>(processed->data());
    size_t buf_size = processed->size();

    auto it = std::find_if(
        compile_specs.begin(),
        compile_specs.end(),
        [](const CompileSpec& spec) {
          return std::strcmp(spec.key, ennWeightBufferKey) == 0;
        });

    Error err = Error::Ok;
    if (it != compile_specs.end()) {
      std::string weight_key = decode_spec_value(*it);
      auto weight = acquire_weight(context.get_named_data_map(), weight_key);
      if (!weight.ok()) {
        executor->~EnnExecutor();
        return weight.error();
      }
      executor->set_weight_key(weight_key);
      ET_LOG(
          Info,
          "EnnBackend, [%s] shared weight %s size(%zu), addr(%p)",
          context.get_method_name(),
          weight_key.c_str(),
          weight.get()->size(),
          weight.get()->data());
      err = executor->initialize(
          binary_buf_addr,
          buf_size,
          reinterpret_cast<const char*>(weight.get()->data()),
          weight.get()->size(),
          true);
    } else {
      err = executor->initialize(binary_buf_addr, buf_size);
    }
    if (err != Error::Ok) {
      ENN_LOG_ERROR("Exynos backend initialize failed.");
      destroy(executor);
      return err;
    }

    return executor;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    EXYNOS_ATRACE_NAME_LINE("backend_execute");
    auto executor = static_cast<enn::EnnExecutor*>(handle);
    std::vector<enn::DataBuffer> inputs;
    std::vector<enn::DataBuffer> outputs;
    for (int32_t index = 0;
         index < executor->getInputSize() + executor->getOutputSize();
         index++) {
      ET_CHECK_OR_RETURN_ERROR(
          args[index]->isTensor(),
          InvalidArgument,
          "Expected argument to delegate at index %u to be a Tensor, but got %" PRIu32,
          index,
          static_cast<uint32_t>(args[index]->tag));
      Tensor* tensor = &args[index]->toTensor();
      enn::DataBuffer data_buffer = {
          .buf_ptr_ = tensor->mutable_data_ptr<void*>(),
          .size_ = tensor->nbytes()};
      if (index < executor->getInputSize()) {
        inputs.push_back(data_buffer);
      } else {
        outputs.push_back(data_buffer);
      }
    }
    Error err = executor->eval(inputs, outputs);
    return err;
  }

  void destroy(DelegateHandle* handle) const override {
    EXYNOS_ATRACE_NAME_LINE("backend_destroy");
    if (handle != nullptr) {
      auto executor = static_cast<enn::EnnExecutor*>(handle);
      std::string weight_key = executor->get_weight_key();
      executor->~EnnExecutor();
      if (!weight_key.empty()) {
        release_weight(weight_key);
      }
    }
  }
}; // namespace executor

namespace {
auto cls = EnnBackend();
Backend backend{"EnnBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace executor
} // namespace torch
