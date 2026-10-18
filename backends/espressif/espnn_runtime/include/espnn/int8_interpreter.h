/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <espnn/xts3_graph.h>

namespace executorch {
namespace backends {
namespace espressif {

/// Which ESP-NN kernel family runs the graph.
enum class KernelMode : uint8_t {
  /// Chip-optimized ESP-NN kernels (SIMD on the ESP32-S3).
  OPTIMIZED = 0,
  /// ESP-NN's portable ANSI C kernels, used to cross-check OPTIMIZED.
  REFERENCE = 1,
};

/// Options for Int8Interpreter::load().
struct Int8RuntimeOptions {
  /// Kernel family used by invoke().
  KernelMode kernel_mode = KernelMode::OPTIMIZED;
  /// Copy weights from memory-mapped flash into PSRAM when they fit.
  bool copy_constants_to_psram = false;
  /// Place the activation arena in internal RAM when it fits, else PSRAM.
  bool prefer_internal_arena = true;
  /// Split large convolutions across both cores.
  bool dual_core = false;
  /// Internal RAM left free when deciding whether the arena fits there.
  size_t internal_reserve_bytes = 48 * 1024;
};

/**
 * Shape and quantization of a graph input or output. Rank-4 tensors are
 * described in NCHW terms but stored on the device as NHWC.
 */
struct Int8TensorInfo {
  uint32_t tensor_id;
  uint8_t rank;
  int32_t shape[4];
  int32_t height;
  int32_t width;
  int32_t channels;
  size_t bytes;
  float scale;
  int32_t zero_point;
};

/**
 * Runs an XTS3 int8 graph (the EspNNBackend delegate payload) with ESP-NN.
 *
 * The interpreter does not copy the graph: the blob passed to load() must
 * stay valid until reset() or destruction. Activations live in one arena
 * planned at load time, so invoke() does not allocate.
 */
class Int8Interpreter {
 public:
  Int8Interpreter() = default;
  ~Int8Interpreter();
  Int8Interpreter(const Int8Interpreter&) = delete;
  Int8Interpreter& operator=(const Int8Interpreter&) = delete;

  /**
   * Validates the graph, plans activation memory and allocates the arena,
   * scratch buffers and requantization tables.
   *
   * @param[in] blob XTS3 graph, kept by reference.
   * @param[in] size Size of @p blob in bytes.
   * @param[in] options Placement and kernel options.
   * @retval true The graph is ready for invoke().
   * @retval false The graph is malformed or memory could not be allocated;
   *     the reason is printed.
   */
  bool load(
      const uint8_t* blob,
      size_t size,
      const Int8RuntimeOptions& options = {});

  /// Frees everything allocated by load(). The interpreter can be reused.
  void reset();

  /**
   * Runs every operation once on the current input data.
   *
   * @retval false The graph is not loaded or an operation failed.
   */
  bool invoke();

  /// Number of graph inputs.
  size_t input_count() const {
    return input_count_;
  }
  /// Number of graph outputs.
  size_t output_count() const {
    return output_count_;
  }
  /// Describes input @p index. @retval false @p index is out of range.
  bool input_info(size_t index, Int8TensorInfo& info) const;
  /// Describes output @p index. @retval false @p index is out of range.
  bool output_info(size_t index, Int8TensorInfo& info) const;
  /// Device-layout (NHWC) int8 buffer of input @p index, or nullptr.
  int8_t* input_data(size_t index);
  /// Device-layout (NHWC) int8 buffer of output @p index, or nullptr.
  const int8_t* output_data(size_t index) const;
  /// Raw data of any graph tensor, constant or activation, or nullptr.
  const int8_t* tensor_data(uint32_t tensor_id) const;

  /**
   * Quantizes float NCHW values into input @p index.
   *
   * @param[in] count Number of values; must equal the input's size.
   */
  bool quantize_input_nchw(size_t index, const float* values, size_t count);

  /**
   * Normalizes interleaved 8-bit RGB pixels with (x / 255 - mean) / stddev
   * and quantizes them into a 3-channel input of the same width and height.
   */
  bool set_input_rgb888(
      size_t index,
      const uint8_t* rgb,
      int32_t width,
      int32_t height,
      const float mean[3],
      const float stddev[3]);

  /// Dequantizes output @p index in device layout into @p values.
  bool dequantize_output(size_t index, float* values, size_t count) const;

  /// Index of the largest int8 value of output @p index, or -1.
  int32_t output_argmax(size_t index) const;

  /// Switches kernels for later invoke() calls without reloading.
  void set_kernel_mode(KernelMode mode) {
    options_.kernel_mode = mode;
  }
  KernelMode kernel_mode() const {
    return options_.kernel_mode;
  }
  const GraphView& graph() const {
    return graph_;
  }
  /// Duration of the last invoke() in microseconds.
  int64_t last_invoke_us() const {
    return last_invoke_us_;
  }
  /// Duration of operation @p op during the last invoke(), or 0.
  int64_t op_time_us(uint32_t op) const;
  /// Size of the activation arena.
  size_t arena_bytes() const {
    return arena_size_;
  }
  /// Whether the arena was placed in internal RAM rather than PSRAM.
  bool arena_in_internal_ram() const {
    return arena_internal_;
  }
  /// Size of the shared ESP-NN scratch buffer.
  size_t scratch_bytes() const {
    return scratch_size_;
  }
  /// Size of the precomputed requantization tables.
  size_t parameter_bytes() const {
    return params_size_;
  }
  /// Whether weights were copied into PSRAM by load().
  bool constants_copied() const {
    return blob_copy_ != nullptr;
  }
  /// Prints tensor, operation and memory statistics.
  void print_summary() const;
  /// Prints per-operation times of the last invoke().
  void print_profile() const;

 private:
  struct OpState;

  bool plan_memory();
  bool prepare_operations();
  void rebase_constants(const uint8_t* old_base, size_t size);
  bool tensor_info(uint32_t tensor_id, Int8TensorInfo& info) const;
  int8_t* activation(uint32_t tensor_id) const;
  const int8_t* operand(uint32_t tensor_id) const;
  void run_generic_conv(const OpState& s, const int8_t* input, int8_t* output)
      const;
  void run_dilated_by_phases(
      const OpState& s,
      const int8_t* input,
      int8_t* output) const;
  void run_copy(const OpState& s, int8_t* output) const;
  void run_upsample(const OpState& s, const int8_t* input, int8_t* output)
      const;
  void run_transposed_conv(
      const OpState& s,
      const int8_t* input,
      int8_t* output) const;
  bool run_operation(uint32_t index);

  Int8RuntimeOptions options_{};
  GraphView graph_{};
  bool loaded_ = false;
  uint8_t* blob_copy_ = nullptr;
  TensorRecord* tensors_ = nullptr;
  OperationRecord* operations_ = nullptr;
  OpState* op_states_ = nullptr;
  size_t* tensor_offsets_ = nullptr;
  uint8_t* arena_ = nullptr;
  size_t arena_size_ = 0;
  bool arena_internal_ = false;
  void* scratch_ = nullptr;
  size_t scratch_size_ = 0;
  int8_t* phase_buffer_ = nullptr;
  size_t phase_bytes_ = 0;
  int32_t* params_ = nullptr;
  size_t params_size_ = 0;
  uint32_t inputs_[MAX_GRAPH_IO]{};
  uint32_t outputs_[MAX_GRAPH_IO]{};
  size_t input_count_ = 0;
  size_t output_count_ = 0;
  int64_t last_invoke_us_ = 0;
};

} // namespace espressif
} // namespace backends
} // namespace executorch
