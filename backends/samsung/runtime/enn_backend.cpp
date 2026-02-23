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
#include <executorch/runtime/platform/profiler.h>

#include <memory>
#include <vector>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace torch {
namespace executor {

class EnnBackend final : public PyTorchBackendInterface {
 public:
  ~EnnBackend() = default;

  bool is_available() const override {
    return true;
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
    Error err = executor->initialize(binary_buf_addr, buf_size);
    if (err != Error::Ok) {
      ENN_LOG_ERROR("Exynos backend initialize failed.");
      executor->~EnnExecutor();
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
