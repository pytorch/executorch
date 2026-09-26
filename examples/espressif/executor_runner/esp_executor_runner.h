/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Public API for the ESP32 ExecuTorch executor runner.
 *
 * Provides a simple interface to load a model once and run repeated inferences
 * on dynamically generated input data:
 *
 *   et_runner_init();
 *
 *   // For each inference:
 *   et_runner_set_input(0, my_input_data, my_input_bytes);
 *   et_runner_execute();
 *   et_runner_get_output(0, out_buf, out_buf_bytes, &num_elements);
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the runner: load the model, allocate memory pools, and prepare
 * the inference method. Must be called once before any other et_runner_*
 * function.
 *
 * @returns true on success, false on failure.
 */
bool et_runner_init(void);

/**
 * Copy raw data into the input tensor at the given index.
 *
 * The runner must already be initialized with et_runner_init(). The data's
 * layout (dtype and shape) must match the model's expected input tensor.
 *
 * @param input_idx  Zero-based index of the input tensor to set.
 * @param data       Pointer to the source data in host memory.
 * @param num_bytes  Number of bytes to copy. Must not exceed the tensor's
 *                   total byte size (element_size * num_elements).
 * @returns true on success, false on failure.
 */
bool et_runner_set_input(size_t input_idx, const void* data, size_t num_bytes);

/**
 * Execute one forward pass of the model.
 *
 * Must be called after et_runner_init(). Call et_runner_set_input() before
 * this if you want to provide custom input data. Results are available via
 * et_runner_get_output() after this call returns successfully.
 *
 * @returns true on success, false on failure.
 */
bool et_runner_execute(void);

/**
 * Copy the output tensor data at the given index into a caller-provided buffer.
 *
 * Must be called after a successful et_runner_execute().
 *
 * @param output_idx       Zero-based index of the output tensor to read.
 * @param buffer           Caller-allocated destination buffer.
 * @param buffer_bytes     Size of the destination buffer in bytes. Must be
 *                         >= the output tensor's total byte size.
 * @param out_num_elements If non-NULL, set to the number of elements in the
 *                         output tensor (not bytes).
 * @returns true on success, false on failure.
 */
bool et_runner_get_output(
    size_t output_idx,
    void* buffer,
    size_t buffer_bytes,
    size_t* out_num_elements);

/**
 * Returns the number of input tensors expected by the loaded model.
 * Returns 0 if the runner is not yet initialized.
 */
size_t et_runner_inputs_size(void);

/**
 * Returns the number of output tensors produced by the loaded model.
 * Returns 0 if the runner is not yet initialized.
 */
size_t et_runner_outputs_size(void);

#ifdef __cplusplus
} // extern "C"
#endif
