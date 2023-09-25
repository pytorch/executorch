/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_QNN_EXECUTORCH_BACKEND_H_
#define EXECUTORCH_QNN_EXECUTORCH_QNN_EXECUTORCH_BACKEND_H_

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

namespace torch {
namespace executor {

class QnnExecuTorchBackend final : public PyTorchBackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  Result<DelegateHandle*> init(
      BackendInitContext& context, FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override;

  Error execute(__ET_UNUSED BackendExecutionContext& context,
                DelegateHandle* handle, EValue** args) const override;

  void destroy(DelegateHandle* handle) const override;

  bool is_available() const override;
};

namespace {
auto cls = QnnExecuTorchBackend();
Backend backend{"QnnBackend", &cls};
static auto success_with_compiler = register_backend(backend);
}  // namespace

}  // namespace executor
}  // namespace torch
#endif  // EXECUTORCH_QNN_EXECUTORCH_QNN_EXECUTORCH_BACKEND_H_
