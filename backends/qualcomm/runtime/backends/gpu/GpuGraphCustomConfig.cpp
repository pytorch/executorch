/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuGraphCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

GpuGraphCustomConfig::GpuGraphCustomConfig(
    const QnnExecuTorchGpuBackendOptions* gpu_options)
    : gpu_options_(gpu_options) {}

QnnGpuGraph_CustomConfig_t* GpuGraphCustomConfig::AllocGraphCustomConfig() {
  gpu_graph_config_.emplace_back(
      std::make_unique<QnnGpuGraph_CustomConfig_t>());
  return gpu_graph_config_.back().get();
}

std::vector<QnnGraph_CustomConfig_t>
GpuGraphCustomConfig::CreateGraphCustomConfig() {
  std::vector<QnnGraph_CustomConfig_t> ret;
  QnnGpuGraph_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->precision =
      static_cast<QnnGpu_Precision_t>(gpu_options_->precision());
  p_custom_config->disableMemoryOptimizations =
      !gpu_options_->use_memory_optimizations();
  p_custom_config->disableNodeOptimizations =
      !gpu_options_->use_node_optimizations();
  p_custom_config->disableQueueRecording = !gpu_options_->use_queue_recording();
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
