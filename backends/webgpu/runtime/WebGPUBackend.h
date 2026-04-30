/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/backend/interface.h>

namespace executorch {
namespace backends {
namespace webgpu {

class WebGPUBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~WebGPUBackend() override = default;

  bool is_available() const override;

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override;

  executorch::runtime::Error execute(
      executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::Span<executorch::runtime::EValue*> args)
      const override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace webgpu
} // namespace backends
} // namespace executorch
