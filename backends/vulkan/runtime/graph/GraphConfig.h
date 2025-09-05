/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace vkcompute {

struct GraphConfig final {
  api::ContextConfig context_config;

  // Creating a descriptor pool with exactly the number of descriptors tallied
  // by iterating through the shader layouts of shaders used in the graph risks
  // the descriptor pool running out of memory, therefore apply a safety factor
  // to descriptor counts when creating the descriptor pool to mitigate this
  // risk.
  float descriptor_pool_safety_factor;

  bool enable_storage_type_override;
  utils::StorageType storage_type_override;

  bool enable_memory_layout_override;
  utils::GPUMemoryLayout memory_layout_override;

  bool enable_querypool;

  bool enable_local_wg_size_override;
  utils::uvec3 local_wg_size_override;

  // Whether or not the ComputeGraph should expect input shapes to be dynamic
  bool expect_dynamic_shapes;

  // Execution properties that determine specifics re: how command buffer
  // submission is handled, etc. 0 means this field is not set.

  // During prepacking, once this threshold is reached, submit the current
  // command buffer for execution. This allows the work to be distributed over
  // multiple command buffer submissions, which can improve model load
  // performance and prevent crashes when loading large models.
  size_t prepack_threshold_nbytes = 0;
  // Threshold used for the first command buffer submission during prepacking.
  // This can be set to be lower than prepack_submission_threshold_nbytes to
  // submit a command buffer for execution earlier which can improve performance
  // by taking more advantage of parallelism between the CPU and GPU.
  size_t prepack_initial_threshold_nbytes = 0;

  // During execute, once this node count is reached, submit the current
  // command buffer for execution. This allows the work to be distributed over
  // multiple command buffer submissions, which can improve execution
  // performance.
  size_t execute_threshold_node_count = 0;
  // Execute node count used for the first command buffer submission during
  // execute. This can be set to be lower than execute_threshold_nbytes to
  // submit a command buffer for execution earlier which can improve performance
  // by taking more advantage of parallelism between the CPU and GPU.
  size_t execute_initial_threshold_node_count = 0;

  // If this number is greater than 0 then, during execute create at most this
  // many command buffers.
  size_t execute_max_cmds = 0;

  vkapi::Adapter* external_adapter;

  // Generate a default graph config with pre-configured settings
  explicit GraphConfig();

  void set_storage_type_override(utils::StorageType storage_type);
  void set_memory_layout_override(utils::GPUMemoryLayout memory_layout);
  void set_local_wg_size_override(const utils::uvec3& local_wg_size);
};

} // namespace vkcompute
