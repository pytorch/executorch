/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "HAP_farf.h"

#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include "qnn_executorch.h"

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchIdlWrapper {
 public:
  QnnExecuTorchIdlWrapper(const char* pte_path, const int method_index);
  executorch::runtime::Error execute_all(
      const char* input_list_path,
      const char* output_folder_path,
      int& num_inferences_performed,
      double& total_execute_interval,
      double& total_read_file_interval,
      double& total_save_file_interval);

  executorch::runtime::Error enable_intermediate_tensor_dump(
      const int debug_buffer_size);
  executorch::runtime::Error dump_etdp(const char* etdump_path);
  executorch::runtime::Error dump_intermediate_tensor(
      const char* debug_output_path);

 private:
  uint8_t method_allocator_pool_[4 * 1024U * 1024U];

  executorch::etdump::ETDumpGen etdump_gen_;
  void* debug_buffer_ = nullptr;
  std::string etdump_path_;
  size_t debug_buffer_size_;

  std::unique_ptr<executorch::extension::FileDataLoader> loader_;
  std::unique_ptr<executorch::runtime::Program> program_;
  std::unique_ptr<executorch::runtime::MethodMeta> method_meta_;
  std::unique_ptr<executorch::runtime::MemoryAllocator> memory_allocator_;
  std::unique_ptr<executorch::runtime::HierarchicalAllocator> planned_memory_;
  std::unique_ptr<executorch::runtime::MemoryManager> memory_manager_;
  std::unique_ptr<executorch::runtime::Method> method_;

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers_;
  std::vector<executorch::runtime::Span<uint8_t>> planned_spans_;
  std::vector<std::vector<uint8_t>> input_tensors_;
  std::vector<std::vector<uint8_t>> output_tensors_;
  std::vector<executorch::aten::TensorImpl> input_tensor_impls_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
