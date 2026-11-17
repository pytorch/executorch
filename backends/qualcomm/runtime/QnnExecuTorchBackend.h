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

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend() override = default;

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
  friend class QnnExecuTorchBackendTest;

  // A context binary signature plus the runtime allocator that the methods of
  // one Module share. Other Modules can be destroyed independently, so they
  // must never receive this Module's QnnManager.
  using DelegateKey =
      std::pair<std::int64_t, const executorch::runtime::MemoryAllocator*>;

  struct SpillFillGroup {
    std::mutex mutex;
    // HTP requires the first context to outlive every other group member.
    std::shared_ptr<executorch::runtime::DelegateHandle> owner;
  };
  using SpillFillKey =
      std::pair<const executorch::runtime::MemoryAllocator*, uint64_t>;

  struct DelegateEntry {
    executorch::runtime::DelegateHandle* handle;
    size_t users;
    // Destroy each member context before releasing the group's first context.
    std::shared_ptr<SpillFillGroup> group;
    std::shared_ptr<executorch::runtime::DelegateHandle> manager;
  };

  std::shared_ptr<SpillFillGroup> get_spill_fill_group(
      const SpillFillKey& key) const;
  std::optional<executorch::runtime::DelegateHandle*> get_cached_delegate(
      const DelegateKey& key) const;
  executorch::runtime::DelegateHandle* add_cached_delegate(
      const DelegateKey& key,
      executorch::runtime::DelegateHandle* handle,
      std::shared_ptr<executorch::runtime::DelegateHandle> manager = nullptr,
      std::shared_ptr<SpillFillGroup> group = nullptr) const;
  bool is_cached_delegate(executorch::runtime::DelegateHandle* handle) const;
  std::optional<DelegateEntry> erase_cached_delegate(
      executorch::runtime::DelegateHandle* handle) const;

  mutable std::mutex mutex_;
  mutable std::mutex runtime_option_mutex_;
  mutable std::map<DelegateKey, DelegateEntry> delegate_map_;
  mutable std::map<SpillFillKey, std::weak_ptr<SpillFillGroup>>
      spill_fill_groups_;
  mutable std::unordered_map<executorch::runtime::DelegateHandle*, DelegateKey>
      delegate_map_rev_;

  RuntimeOption qnn_runtime_log_level_{false, 0};
  RuntimeOption qnn_runtime_performance_mode_{false, 0};
  RuntimeOption qnn_runtime_profile_level_{false, 0};
  RuntimeOption qnn_runtime_lpai_fps_{false, 0};
  RuntimeOption qnn_runtime_lpai_ftrt_ratio_{false, 0};
  RuntimeOption qnn_runtime_lpai_client_perf_type_{false, 0};
  RuntimeOption qnn_runtime_lpai_affinity_{false, 0};
  RuntimeOption qnn_runtime_lpai_core_selection_{false, 0};
  RuntimeOption qnn_runtime_heap_profiling_path_{false, {}};
};

} // namespace qnn
} // namespace backends
} // namespace executorch
