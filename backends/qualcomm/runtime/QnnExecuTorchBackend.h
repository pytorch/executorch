/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

namespace torch {
namespace executor {

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override;

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override;

  void destroy(DelegateHandle* handle) const override;

  bool is_available() const override;
};

} // namespace executor
} // namespace torch
