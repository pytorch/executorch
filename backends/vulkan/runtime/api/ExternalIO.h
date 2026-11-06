/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <cstddef>
#include <map>
#include <utility>

namespace vkcompute {
namespace api {

#ifdef VK_ANDROID_external_memory_android_hardware_buffer
/*
 * Zero-copy I/O for the Vulkan delegate. Registers a BLOB AHardwareBuffer as
 * the staging memory of graph input or output `index` (the delegate's argument
 * order); a null `ahb` removes the registration. The next Vulkan delegate
 * instance initialized (i.e. the next method loaded) takes every registration
 * and clears the registry at the start of its init, so registrations never
 * carry over to later loads, even when that load fails. If the program loaded
 * has no Vulkan-delegated method, call clear_external_io_buffers(). That graph reads the input from, or writes
 * the output to, the AHardwareBuffer on the GPU in
 * the staging layout (contiguous, staging dtype), and skip the host copy for
 * it: the ExecuTorch tensor of that argument is neither read nor written.
 * Another API that imports the same AHardwareBuffer (e.g. GLES through
 * GL_EXT_external_buffer) can then produce the input or consume the output
 * without a CPU round trip. The caller keeps the AHardwareBuffer alive from
 * this call until the load returns; the graph holds its own reference after
 * that. The caller orders its own GPU work around execute(), which
 * returns after the delegate's GPU work completes.
 *
 * An input and an output may share one AHardwareBuffer (e.g. a recurrent
 * state): inputs are staged at the start of execute() and outputs at the end.
 *
 * A graph with data-dependent shapes reads input staging on the CPU and
 * rejects external I/O; init then fails with Error::InvalidArgument, as it does
 * for a buffer that cannot be imported or is smaller than the tensor.
 */
void set_external_io_buffer(bool is_input, size_t index, AHardwareBuffer* ahb);

// Used by the delegate: returns every registration, keyed by
// (is_input, index), and clears the registry.
std::map<std::pair<bool, size_t>, AHardwareBuffer*> take_external_io_buffers();

void clear_external_io_buffers();
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

} // namespace api
} // namespace vkcompute
