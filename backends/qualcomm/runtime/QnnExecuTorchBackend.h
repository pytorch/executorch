/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
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
      executorch::runtime::Span<executorch::runtime::EValue*> args)
      const override;

  ET_NODISCARD executorch::runtime::Error set_option(
      executorch::runtime::BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  executorch::runtime::Error get_option(
      executorch::runtime::BackendOptionContext& context,
      executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  void add_cached_delegate(
      const std::int64_t& signature,
      executorch::runtime::DelegateHandle* handle) const;
  void erase_cached_delegate(executorch::runtime::DelegateHandle* handle) const;

  mutable std::mutex mutex_;
  mutable std::mutex runtime_option_mutex_;
  mutable std::unordered_map<int64_t, executorch::runtime::DelegateHandle*>
      delegate_map_;
  mutable std::unordered_map<executorch::runtime::DelegateHandle*, std::int64_t>
      delegate_map_rev_;

  RuntimeOption qnn_runtime_log_level_{false, 0};
  RuntimeOption qnn_runtime_performance_mode_{false, 0};
  RuntimeOption qnn_runtime_profile_level_{false, 0};
};

} // namespace qnn
} // namespace backends
} // namespace executorch
