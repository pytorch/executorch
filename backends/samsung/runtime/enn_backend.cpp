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

#include <memory>
#include <queue>
#include <vector>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace torch {
namespace executor {

using executorch::runtime::NamedDataMap;

const char ennWeightBufferKey[] = "enn_weight_buffer_hash_id";

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
    const char* binary_buf_addr =
        reinterpret_cast<const char*>(processed->data());
    size_t buf_size = processed->size();
    const NamedDataMap* named_data_map = context.get_named_data_map();

    auto it = std::find_if(
        compile_specs.begin(),
        compile_specs.end(),
        [](const CompileSpec& spec) {
          return std::strcmp(spec.key, ennWeightBufferKey) == 0;
        });

    if (it != compile_specs.end()) {
      std::string weight_key = decode_spec_value(*it);
      Result<FreeableBuffer> weight =
          named_data_map->get_data(weight_key.c_str());
      const char* weight_buf_addr =
          reinterpret_cast<const char*>(weight->data());
      size_t weight_size = weight->size();

      ET_LOG(
          Info, "EnnBackend, [%s] Weight Buffer Enabled for %s size(%ld), addr(%p)",
                context.get_method_name(),
                weight_key.c_str(), weight_size, weight_buf_addr);

      FreeableBuffer* wgt = new FreeableBuffer(std::move(weight.get()));

      Error err = executor->initialize(
          binary_buf_addr, buf_size, weight_buf_addr, weight_size, true);
      if (err != Error::Ok) {
        ENN_LOG_ERROR("Exynos backend initialize failed.");
        executor->~EnnExecutor();
        return err;
      }
    } else {
      Error err = executor->initialize(binary_buf_addr, buf_size);
      if (err != Error::Ok) {
        ENN_LOG_ERROR("Exynos backend initialize failed.");
        executor->~EnnExecutor();
        return err;
      }
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
      executor->~EnnExecutor();
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
