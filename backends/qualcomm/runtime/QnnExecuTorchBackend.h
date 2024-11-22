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

#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override;

  executorch::runtime::Error execute(
      ET_UNUSED executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::EValue** args) const override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  void add_cached_delegate(
      const std::string& signature,
      executorch::runtime::DelegateHandle* handle) const;
  void erase_cached_delegate(executorch::runtime::DelegateHandle* handle) const;

  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, executorch::runtime::DelegateHandle*>
      delegate_map_;
  mutable std::unordered_map<executorch::runtime::DelegateHandle*, std::string>
      delegate_map_rev_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
