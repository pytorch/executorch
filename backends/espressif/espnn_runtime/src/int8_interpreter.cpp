/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <espnn/int8_interpreter.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include <esp_heap_caps.h>
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace executorch {
namespace backends {
namespace espressif {
namespace {

constexpr size_t ARENA_ALIGNMENT = 16;
constexpr int64_t YIELD_INTERVAL_US = 1000000;

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

void* allocate_preferring_internal(
    size_t bytes,
    bool prefer_internal,
    size_t reserve,
    bool* internal) {
  if (bytes == 0)
    return nullptr;
  if (prefer_internal) {
    const size_t largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (largest > bytes + reserve) {
      void* memory = heap_caps_aligned_alloc(
          ARENA_ALIGNMENT, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      if (memory != nullptr) {
        if (internal != nullptr)
          *internal = true;
        return memory;
      }
    }
  }
  void* memory = heap_caps_aligned_alloc(
      ARENA_ALIGNMENT, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (memory != nullptr) {
    if (internal != nullptr)
      *internal = false;
    return memory;
  }
  memory = heap_caps_aligned_alloc(
      ARENA_ALIGNMENT, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (internal != nullptr)
    *internal = memory != nullptr;
  return memory;
}

int8_t clamp_to_int8(int32_t value) {
  if (value < -128)
    return -128;
  if (value > 127)
    return 127;
  return static_cast<int8_t>(value);
}

int32_t rounding_high_mul(int32_t a, int32_t b) {
  if (a == b && a == INT32_MIN)
    return INT32_MAX;
  int64_t nudge = 1 << 30;
  if ((a < 0) ^ (b < 0))
    nudge = 1 - nudge;
  const int64_t product = static_cast<int64_t>(a) * b + nudge;
  const int64_t to_add = product < 0 ? (1LL << 31) - 1 : 0;
  return static_cast<int32_t>((product + to_add) >> 31);
}

int32_t rounding_shift_right(int32_t value, int32_t exponent) {
  const int32_t mask = static_cast<int32_t>((1LL << exponent) - 1);
  const int32_t remainder = value & mask;
  int32_t result = value >> exponent;
  const int32_t threshold = (mask >> 1) + (result < 0 ? 1 : 0);
  if (remainder > threshold)
    result += 1;
  return result;
}

int32_t requantize(int32_t value, int32_t multiplier, int32_t shift) {
  const int32_t left = shift > 0 ? shift : 0;
  const int32_t right = shift > 0 ? 0 : -shift;
  const int32_t shifted =
      static_cast<int32_t>(static_cast<uint32_t>(value) << left);
  return rounding_shift_right(rounding_high_mul(shifted, multiplier), right);
}

int8_t quantize_value(float value, float inverse_scale, int32_t zero_point) {
  const float scaled = value * inverse_scale;
  if (!std::isfinite(scaled))
    return clamp_to_int8(scaled > 0.0f ? 127 : -128);
  if (scaled > 1.0e6f)
    return 127;
  if (scaled < -1.0e6f)
    return -128;
  return clamp_to_int8(
      static_cast<int32_t>(std::nearbyintf(scaled)) + zero_point);
}

} // namespace

struct Int8Interpreter::OpState {
  OpCode opcode;
  uint32_t input0;
  uint32_t input1;
  uint32_t output;
  uint8_t input_count;
  uint32_t inputs[6];
  int32_t pool[6];
  const int8_t* weights;
  const int32_t* bias;
  data_dims_t input_dims;
  data_dims_t filter_dims;
  data_dims_t output_dims;
  conv_params_t conv;
  dw_conv_params_t dw;
  quant_data_t quant;
  int32_t* multipliers;
  int32_t* shifts;
  Int8AddParams add;
  int32_t value0;
  int32_t value1;
  int32_t input_zero_point;
  int32_t output_zero_point;
  uint32_t elements;
  int32_t rows;
  int32_t cols;
  bool transpose;
  bool generic;
  const uint8_t* table;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t batches;
  int64_t total_us;
  int64_t last_us;
  uint32_t calls;
};

Int8Interpreter::~Int8Interpreter() {
  reset();
}

void Int8Interpreter::reset() {
  if (arena_ != nullptr)
    heap_caps_free(arena_);
  if (scratch_ != nullptr)
    heap_caps_free(scratch_);
  if (phase_buffer_ != nullptr)
    heap_caps_free(phase_buffer_);
  phase_buffer_ = nullptr;
  phase_bytes_ = 0;
  if (params_ != nullptr)
    heap_caps_free(params_);
  if (blob_copy_ != nullptr)
    heap_caps_free(blob_copy_);
  if (tensors_ != nullptr)
    heap_caps_free(tensors_);
  if (operations_ != nullptr)
    heap_caps_free(operations_);
  if (op_states_ != nullptr)
    heap_caps_free(op_states_);
  if (tensor_offsets_ != nullptr)
    heap_caps_free(tensor_offsets_);
  arena_ = nullptr;
  scratch_ = nullptr;
  params_ = nullptr;
  blob_copy_ = nullptr;
  tensors_ = nullptr;
  operations_ = nullptr;
  op_states_ = nullptr;
  tensor_offsets_ = nullptr;
  arena_size_ = 0;
  scratch_size_ = 0;
  params_size_ = 0;
  input_count_ = 0;
  output_count_ = 0;
  loaded_ = false;
  graph_ = {};
}

bool Int8Interpreter::load(
    const uint8_t* blob,
    size_t size,
    const Int8RuntimeOptions& options) {
  reset();
  options_ = options;
  if (!parse_graph(blob, size, graph_)) {
    std::printf("ERROR: int8 runtime rejected the model blob\n");
    return false;
  }
  if (!is_int8_graph_version(graph_.header.version)) {
    std::printf(
        "ERROR: model is XTS3 v%u; the int8 runtime requires v%u or v%u\n",
        static_cast<unsigned>(graph_.header.version),
        static_cast<unsigned>(GRAPH_ESP_NN_INT8_VERSION),
        static_cast<unsigned>(GRAPH_ESP_NN_INT8_IO_VERSION));
    return false;
  }
  const bool misaligned =
      (reinterpret_cast<uintptr_t>(graph_.constant_data) & 15U) != 0;
  if (misaligned) {
    blob_copy_ = static_cast<uint8_t*>(allocate_preferring_internal(
        size, false, options_.internal_reserve_bytes, nullptr));
    if (blob_copy_ == nullptr) {
      std::printf(
          "ERROR: unable to copy %u-byte misaligned model into RAM\n",
          static_cast<unsigned>(size));
      return false;
    }
    std::memcpy(blob_copy_, blob, size);
    if (!parse_graph(blob_copy_, size, graph_))
      return false;
    std::printf(
        "WARNING: model constants were not 16-byte aligned; copied to RAM\n");
  }

  const uint32_t tensor_count = graph_.header.num_tensors;
  const uint32_t op_count = graph_.header.num_operations;
  tensors_ = static_cast<TensorRecord*>(
      heap_caps_calloc(tensor_count, sizeof(TensorRecord), MALLOC_CAP_8BIT));
  operations_ = static_cast<OperationRecord*>(
      heap_caps_calloc(op_count, sizeof(OperationRecord), MALLOC_CAP_8BIT));
  op_states_ = static_cast<OpState*>(
      heap_caps_calloc(op_count, sizeof(OpState), MALLOC_CAP_8BIT));
  tensor_offsets_ = static_cast<size_t*>(
      heap_caps_calloc(tensor_count, sizeof(size_t), MALLOC_CAP_8BIT));
  if (tensors_ == nullptr || operations_ == nullptr || op_states_ == nullptr ||
      tensor_offsets_ == nullptr) {
    std::printf("ERROR: unable to allocate interpreter tables\n");
    reset();
    return false;
  }
  for (uint32_t i = 0; i < tensor_count; ++i)
    read_tensor(graph_, i, tensors_[i]);
  input_count_ = graph_input_count(graph_);
  output_count_ = graph_output_count(graph_);
  if (input_count_ > MAX_GRAPH_IO || output_count_ > MAX_GRAPH_IO) {
    std::printf(
        "ERROR: model has more than %u inputs or outputs\n",
        static_cast<unsigned>(MAX_GRAPH_IO));
    reset();
    return false;
  }
  for (uint32_t i = 0; i < input_count_; ++i)
    inputs_[i] = graph_input_id(graph_, i);
  for (uint32_t i = 0; i < output_count_; ++i)
    outputs_[i] = graph_output_id(graph_, i);
  for (uint32_t i = 0; i < op_count; ++i)
    read_operation(graph_, i, operations_[i]);

  if (!plan_memory() || !prepare_operations()) {
    reset();
    return false;
  }
  if (options_.copy_constants_to_psram && blob_copy_ == nullptr &&
      heap_caps_get_total_size(MALLOC_CAP_SPIRAM) != 0) {
    const size_t largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (largest >= size + 64 * 1024) {
      blob_copy_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(
          ARENA_ALIGNMENT, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    if (blob_copy_ == nullptr) {
      std::printf(
          "WARNING: not enough PSRAM to copy %u model bytes; reading weights from flash\n",
          static_cast<unsigned>(size));
    } else {
      std::memcpy(blob_copy_, graph_.data, size);
      const uint8_t* old_base = graph_.data;
      if (!parse_graph(blob_copy_, size, graph_)) {
        reset();
        return false;
      }
      rebase_constants(old_base, size);
    }
  }
  if (options_.dual_core) {
#if ESP_NN_DUAL_CORE_SUPPORTED
    esp_nn_dual_core_enable();
#else
    std::printf("WARNING: dual-core ESP-NN is not supported on this target\n");
#endif
  }
  loaded_ = true;
  return true;
}

bool Int8Interpreter::plan_memory() {
  const uint32_t tensor_count = graph_.header.num_tensors;
  const int32_t op_count = static_cast<int32_t>(graph_.header.num_operations);
  int32_t* first = static_cast<int32_t*>(
      heap_caps_calloc(tensor_count, sizeof(int32_t), MALLOC_CAP_8BIT));
  int32_t* last = static_cast<int32_t*>(
      heap_caps_calloc(tensor_count, sizeof(int32_t), MALLOC_CAP_8BIT));
  size_t* bytes = static_cast<size_t*>(
      heap_caps_calloc(tensor_count, sizeof(size_t), MALLOC_CAP_8BIT));
  bool* planned = static_cast<bool*>(
      heap_caps_calloc(tensor_count, sizeof(bool), MALLOC_CAP_8BIT));
  struct PlanBuffers {
    void* buffers[4];
    ~PlanBuffers() {
      for (void* buffer : buffers) {
        if (buffer != nullptr)
          heap_caps_free(buffer);
      }
    }
  } plan_buffers{{first, last, bytes, planned}};
  if (first == nullptr || last == nullptr || bytes == nullptr ||
      planned == nullptr) {
    std::printf("ERROR: unable to allocate memory-planner tables\n");
    return false;
  }
  for (uint32_t i = 0; i < tensor_count; ++i) {
    first[i] =
        (tensors_[i].flags & TENSOR_FLAG_GRAPH_INPUT) != 0 ? -1 : INT32_MAX;
    last[i] =
        (tensors_[i].flags & TENSOR_FLAG_GRAPH_OUTPUT) != 0 ? op_count : -1;
    bytes[i] = align_up(tensor_data_size(tensors_[i]), ARENA_ALIGNMENT);
    tensor_offsets_[i] = SIZE_MAX;
  }
  for (int32_t op = 0; op < op_count; ++op) {
    const OperationRecord& record = operations_[op];
    for (uint8_t j = 0; j < record.num_inputs; ++j) {
      const uint32_t id = record.inputs[j];
      if ((tensors_[id].flags & TENSOR_FLAG_CONSTANT) == 0 && last[id] < op)
        last[id] = op;
    }
    for (uint8_t j = 0; j < record.num_outputs; ++j) {
      const uint32_t id = record.outputs[j];
      if (first[id] > op)
        first[id] = op;
      if (last[id] < op)
        last[id] = op;
    }
  }

  size_t arena_size = 0;
  for (uint32_t step = 0; step < tensor_count; ++step) {
    int32_t best = -1;
    for (uint32_t i = 0; i < tensor_count; ++i) {
      if (planned[i] || (tensors_[i].flags & TENSOR_FLAG_CONSTANT) != 0)
        continue;
      if (first[i] == INT32_MAX)
        continue;
      if (best < 0 || bytes[i] > bytes[best])
        best = static_cast<int32_t>(i);
    }
    if (best < 0)
      break;
    planned[best] = true;
    size_t candidate = 0;
    bool moved = true;
    while (moved) {
      moved = false;
      for (uint32_t j = 0; j < tensor_count; ++j) {
        if (j == static_cast<uint32_t>(best) || tensor_offsets_[j] == SIZE_MAX)
          continue;
        const bool lifetime_overlap =
            first[j] <= last[best] && first[best] <= last[j];
        if (!lifetime_overlap)
          continue;
        const size_t other_begin = tensor_offsets_[j];
        const size_t other_end = other_begin + bytes[j];
        if (candidate < other_end && other_begin < candidate + bytes[best]) {
          candidate = other_end;
          moved = true;
        }
      }
    }
    tensor_offsets_[best] = candidate;
    if (candidate + bytes[best] > arena_size)
      arena_size = candidate + bytes[best];
  }
  for (uint32_t i = 0; i < tensor_count; ++i) {
    if ((tensors_[i].flags & TENSOR_FLAG_CONSTANT) == 0 &&
        tensor_offsets_[i] == SIZE_MAX) {
      std::printf(
          "ERROR: tensor %u is never produced or consumed\n",
          static_cast<unsigned>(i));
      return false;
    }
  }

  arena_size_ = arena_size;
  arena_ = static_cast<uint8_t*>(allocate_preferring_internal(
      arena_size_,
      options_.prefer_internal_arena,
      options_.internal_reserve_bytes,
      &arena_internal_));
  if (arena_ == nullptr) {
    std::printf(
        "ERROR: unable to allocate %u-byte activation arena\n",
        static_cast<unsigned>(arena_size_));
    return false;
  }
  std::memset(arena_, 0, arena_size_);
  return true;
}

bool Int8Interpreter::prepare_operations() {
  const uint32_t op_count = graph_.header.num_operations;
  size_t param_words = 0;
  for (uint32_t i = 0; i < op_count; ++i) {
    const OperationRecord& op = operations_[i];
    const OpCode code = static_cast<OpCode>(op.opcode);
    if (code == OpCode::CONV2D || code == OpCode::DEPTHWISE_CONV2D ||
        code == OpCode::LINEAR || code == OpCode::TRANSPOSE_CONV2D) {
      param_words += 2U * tensor_element_count(tensors_[op.inputs[2]]);
    } else if (code == OpCode::SOFTMAX) {
      param_words += 256U;
    }
  }
  params_size_ = param_words * sizeof(int32_t);
  if (params_size_ != 0) {
    params_ = static_cast<int32_t*>(allocate_preferring_internal(
        params_size_, true, options_.internal_reserve_bytes, nullptr));
    if (params_ == nullptr) {
      std::printf("ERROR: unable to allocate requantization tables\n");
      return false;
    }
  }

  int32_t* cursor = params_;
  size_t scratch_size = 0;
  for (uint32_t i = 0; i < op_count; ++i) {
    const OperationRecord& op = operations_[i];
    OpState& state = op_states_[i];
    state = {};
    state.opcode = static_cast<OpCode>(op.opcode);
    state.input0 = op.inputs[0];
    state.input1 = op.num_inputs > 1 ? op.inputs[1] : 0;
    state.output = op.outputs[0];
    const TensorRecord& input = tensors_[op.inputs[0]];
    const TensorRecord& output = tensors_[op.outputs[0]];
    state.input_zero_point = input.quant_zero_point;
    state.output_zero_point = output.quant_zero_point;
    state.elements = static_cast<uint32_t>(tensor_element_count(output));
    const uint8_t* p = graph_.parameter_data + op.parameter_offset;

    switch (state.opcode) {
      case OpCode::CONV2D:
      case OpCode::DEPTHWISE_CONV2D: {
        const TensorRecord& weight = tensors_[op.inputs[1]];
        const TensorRecord& bias = tensors_[op.inputs[2]];
        const uint32_t channels = static_cast<uint32_t>(output.shape[1]);
        Int8ConvParams params{};
        if (!read_int8_conv_params(graph_, op, channels, params))
          return false;
        state.weights = reinterpret_cast<const int8_t*>(
            tensor_constant_data(graph_, weight));
        state.bias = reinterpret_cast<const int32_t*>(
            tensor_constant_data(graph_, bias));
        state.multipliers = cursor;
        state.shifts = cursor + channels;
        for (uint32_t c = 0; c < channels; ++c) {
          state.multipliers[c] = read_le_i32(params.multipliers + c * 4);
          state.shifts[c] = read_le_i32(params.shifts + c * 4);
        }
        cursor += 2U * channels;
        state.quant.mult = state.multipliers;
        state.quant.shift = state.shifts;
        state.input_dims = {input.shape[3], input.shape[2], input.shape[1], 1};
        state.output_dims = {
            output.shape[3], output.shape[2], output.shape[1], 1};
        state.dilation_h = params.dilation_h;
        state.dilation_w = params.dilation_w;
        state.generic = params.dilation_h != 1 || params.dilation_w != 1;
        state.pool[0] = params.kernel_h;
        state.pool[1] = params.kernel_w;
        state.pool[2] = params.stride_h;
        state.pool[3] = params.stride_w;
        state.pool[4] = params.pad_h;
        state.pool[5] = params.pad_w;
        state.value0 = params.activation_min;
        state.value1 = params.activation_max;
        state.rows = params.groups;
        if (state.generic) {
          if (params.stride_h != 1 || params.stride_w != 1)
            break;
          state.transpose = true;
          const int32_t phase_h =
              (output.shape[2] + params.dilation_h - 1) / params.dilation_h;
          const int32_t phase_w =
              (output.shape[3] + params.dilation_w - 1) / params.dilation_w;
          data_dims_t tile_in{
              phase_w + params.kernel_w - 1,
              phase_h + params.kernel_h - 1,
              input.shape[1],
              1};
          data_dims_t tile_out{phase_w, phase_h, output.shape[1], 1};
          const size_t tile_bytes = static_cast<size_t>(tile_in.width) *
                  tile_in.height * tile_in.channels +
              static_cast<size_t>(phase_w) * phase_h * output.shape[1];
          if (tile_bytes > phase_bytes_)
            phase_bytes_ = tile_bytes;
          if (state.opcode == OpCode::CONV2D) {
            state.filter_dims = {
                params.kernel_w, params.kernel_h, weight.shape[1], 1};
            state.conv.in_offset = -input.quant_zero_point;
            state.conv.out_offset = output.quant_zero_point;
            state.conv.stride = {1, 1};
            state.conv.padding = {0, 0};
            state.conv.dilation = {1, 1};
            state.conv.activation = {
                params.activation_min, params.activation_max};
            const int needed = esp_nn_get_conv_scratch_size(
                &tile_in, &state.filter_dims, &tile_out, &state.conv);
            if (needed > 0 && static_cast<size_t>(needed) > scratch_size)
              scratch_size = needed;
          } else {
            state.filter_dims = {
                params.kernel_w, params.kernel_h, output.shape[1], 1};
            state.dw.in_offset = -input.quant_zero_point;
            state.dw.out_offset = output.quant_zero_point;
            state.dw.ch_mult = output.shape[1] / input.shape[1];
            state.dw.stride = {1, 1};
            state.dw.padding = {0, 0};
            state.dw.dilation = {1, 1};
            state.dw.activation = {
                params.activation_min, params.activation_max};
            const int needed = esp_nn_get_depthwise_conv_scratch_size(
                &tile_in, &state.filter_dims, &tile_out, &state.dw);
            if (needed > 0 && static_cast<size_t>(needed) > scratch_size)
              scratch_size = needed;
          }
          break;
        }
        if (state.opcode == OpCode::CONV2D) {
          state.filter_dims = {
              params.kernel_w, params.kernel_h, weight.shape[1], 1};
          state.conv.in_offset = -input.quant_zero_point;
          state.conv.out_offset = output.quant_zero_point;
          state.conv.stride = {params.stride_w, params.stride_h};
          state.conv.padding = {params.pad_w, params.pad_h};
          state.conv.dilation = {1, 1};
          state.conv.activation = {
              params.activation_min, params.activation_max};
          const int needed = esp_nn_get_conv_scratch_size(
              &state.input_dims,
              &state.filter_dims,
              &state.output_dims,
              &state.conv);
          if (needed > 0 && static_cast<size_t>(needed) > scratch_size)
            scratch_size = needed;
        } else {
          state.filter_dims = {
              params.kernel_w, params.kernel_h, output.shape[1], 1};
          state.dw.in_offset = -input.quant_zero_point;
          state.dw.out_offset = output.quant_zero_point;
          state.dw.ch_mult = output.shape[1] / input.shape[1];
          state.dw.stride = {params.stride_w, params.stride_h};
          state.dw.padding = {params.pad_w, params.pad_h};
          state.dw.dilation = {1, 1};
          state.dw.activation = {params.activation_min, params.activation_max};
          const int needed = esp_nn_get_depthwise_conv_scratch_size(
              &state.input_dims,
              &state.filter_dims,
              &state.output_dims,
              &state.dw);
          if (needed > 0 && static_cast<size_t>(needed) > scratch_size)
            scratch_size = needed;
        }
        break;
      }
      case OpCode::RELU6:
        state.value0 = read_le_i32(p);
        state.value1 = read_le_i32(p + 4);
        break;
      case OpCode::ADD:
        if (!read_int8_add_params(graph_, op, state.add))
          return false;
        state.value0 = tensors_[op.inputs[1]].quant_zero_point;
        break;
      case OpCode::COPY:
        state.input_count = op.num_inputs;
        for (uint8_t k = 0; k < op.num_inputs; ++k)
          state.inputs[k] = op.inputs[k];
        state.table = p;
        break;
      case OpCode::UPSAMPLE:
        state.table = p;
        state.input_dims = {input.shape[3], input.shape[2], input.shape[1], 1};
        state.output_dims = {
            output.shape[3], output.shape[2], output.shape[1], 1};
        state.value0 = read_le_i32(p);
        break;
      case OpCode::TRANSPOSE_CONV2D: {
        const TensorRecord& weight = tensors_[op.inputs[1]];
        const TensorRecord& bias = tensors_[op.inputs[2]];
        const uint32_t channels = static_cast<uint32_t>(output.shape[1]);
        state.weights = reinterpret_cast<const int8_t*>(
            tensor_constant_data(graph_, weight));
        state.bias = reinterpret_cast<const int32_t*>(
            tensor_constant_data(graph_, bias));
        for (int k = 0; k < 6; ++k)
          state.pool[k] = read_le_i32(p + 4 * k);
        state.dilation_h = read_le_i32(p + 32);
        state.dilation_w = read_le_i32(p + 36);
        state.value0 = read_le_i32(p + 44);
        state.value1 = read_le_i32(p + 48);
        state.multipliers = cursor;
        state.shifts = cursor + channels;
        const uint8_t* multipliers = p + TRANSPOSE_CONV_PARAMS_SIZE;
        const uint8_t* shifts = multipliers + channels * 4U;
        for (uint32_t c = 0; c < channels; ++c) {
          state.multipliers[c] = read_le_i32(multipliers + c * 4);
          state.shifts[c] = read_le_i32(shifts + c * 4);
        }
        cursor += 2U * channels;
        state.input_dims = {input.shape[3], input.shape[2], input.shape[1], 1};
        state.output_dims = {
            output.shape[3], output.shape[2], output.shape[1], 1};
        break;
      }
      case OpCode::AVGPOOL2D: {
        state.value0 = read_le_i32(p);
        state.value1 = read_le_i32(p + 4);
        state.input_dims = {input.shape[3], input.shape[2], input.shape[1], 1};
        const int32_t needed = esp_nn_get_mean_scratch_size(
            input.shape[2], input.shape[3], input.shape[1]);
        if (needed > 0 && static_cast<size_t>(needed) > scratch_size)
          scratch_size = needed;
        break;
      }
      case OpCode::MAX_POOL2D:
      case OpCode::AVG_POOL2D:
        for (int k = 0; k < 6; ++k)
          state.pool[k] = p[k];
        state.value0 = read_le_i32(p + 8);
        state.value1 = read_le_i32(p + 12);
        state.input_dims = {input.shape[3], input.shape[2], input.shape[1], 1};
        state.output_dims = {
            output.shape[3], output.shape[2], output.shape[1], 1};
        break;
      case OpCode::LUT:
        state.weights = reinterpret_cast<const int8_t*>(p);
        break;
      case OpCode::MUL:
        state.value0 = tensors_[op.inputs[1]].quant_zero_point;
        state.add.output_mult = read_le_i32(p);
        state.add.output_shift = read_le_i32(p + 4);
        state.add.activation_min = read_le_i32(p + 8);
        state.add.activation_max = read_le_i32(p + 12);
        state.transpose = read_le_i32(p + 16) == MUL_MODE_CHANNEL_BROADCAST;
        state.cols = input.rank == 4 ? input.shape[1]
                                     : static_cast<int32_t>(state.elements);
        state.rows = static_cast<int32_t>(state.elements) / state.cols;
        break;
      case OpCode::SOFTMAX:
        state.multipliers = cursor;
        for (int k = 0; k < 256; ++k)
          state.multipliers[k] = static_cast<int32_t>(read_le_u32(p + 4 * k));
        cursor += 256;
        state.cols = input.shape[1];
        state.rows = static_cast<int32_t>(state.elements) / state.cols;
        break;
      case OpCode::CONCAT:
        state.input_count = op.num_inputs;
        for (uint8_t k = 0; k < op.num_inputs; ++k)
          state.inputs[k] = op.inputs[k];
        state.rows = output.rank == 4 ? output.shape[2] * output.shape[3] : 1;
        state.cols = output.shape[1];
        break;
      case OpCode::FLATTEN:
        state.transpose =
            input.rank == 4 && input.shape[2] * input.shape[3] > 1;
        state.input_dims = input.rank == 4
            ? data_dims_t{input.shape[3], input.shape[2], input.shape[1], 1}
            : data_dims_t{
                  1, 1, static_cast<int32_t>(tensor_element_count(input)), 1};
        break;
      case OpCode::LINEAR: {
        const TensorRecord& weight = tensors_[op.inputs[1]];
        const TensorRecord& bias = tensors_[op.inputs[2]];
        state.rows = weight.shape[0];
        state.cols = weight.shape[1];
        state.batches = static_cast<int32_t>(
            tensor_element_count(input) / static_cast<size_t>(weight.shape[1]));
        state.weights = reinterpret_cast<const int8_t*>(
            tensor_constant_data(graph_, weight));
        state.bias = reinterpret_cast<const int32_t*>(
            tensor_constant_data(graph_, bias));
        state.value0 = read_le_i32(p);
        state.value1 = read_le_i32(p + 4);
        state.multipliers = cursor;
        state.shifts = cursor + state.rows;
        const uint8_t* multipliers = p + INT8_PAIR_PARAMS_SIZE;
        const uint8_t* shifts =
            multipliers + static_cast<size_t>(state.rows) * 4U;
        for (int32_t r = 0; r < state.rows; ++r) {
          state.multipliers[r] = read_le_i32(multipliers + r * 4);
          state.shifts[r] = read_le_i32(shifts + r * 4);
        }
        cursor += 2 * state.rows;
        break;
      }
      default:
        return false;
    }
  }

  if (phase_bytes_ != 0) {
    phase_buffer_ = static_cast<int8_t*>(allocate_preferring_internal(
        phase_bytes_, true, options_.internal_reserve_bytes, nullptr));
    if (phase_buffer_ == nullptr) {
      std::printf(
          "ERROR: unable to allocate %u-byte dilated-convolution tile buffer\n",
          static_cast<unsigned>(phase_bytes_));
      return false;
    }
  }
  scratch_size_ = scratch_size;
  if (scratch_size_ != 0) {
    scratch_ = allocate_preferring_internal(
        scratch_size_, true, options_.internal_reserve_bytes, nullptr);
    if (scratch_ == nullptr) {
      std::printf(
          "ERROR: unable to allocate %u-byte ESP-NN scratch buffer\n",
          static_cast<unsigned>(scratch_size_));
      return false;
    }
  }
  return true;
}

void Int8Interpreter::rebase_constants(const uint8_t* old_base, size_t size) {
  const uint8_t* old_end = old_base + size;
  auto rebase = [&](const void* pointer) -> const uint8_t* {
    const uint8_t* bytes = static_cast<const uint8_t*>(pointer);
    if (bytes >= old_base && bytes < old_end)
      return graph_.data + (bytes - old_base);
    return bytes;
  };
  for (uint32_t i = 0; i < graph_.header.num_operations; ++i) {
    OpState& state = op_states_[i];
    if (state.weights != nullptr)
      state.weights = reinterpret_cast<const int8_t*>(rebase(state.weights));
    if (state.bias != nullptr)
      state.bias = reinterpret_cast<const int32_t*>(rebase(state.bias));
    if (state.table != nullptr)
      state.table = rebase(state.table);
  }
}

int8_t* Int8Interpreter::activation(uint32_t tensor_id) const {
  return reinterpret_cast<int8_t*>(arena_ + tensor_offsets_[tensor_id]);
}

const int8_t* Int8Interpreter::operand(uint32_t tensor_id) const {
  if ((tensors_[tensor_id].flags & TENSOR_FLAG_CONSTANT) != 0) {
    return reinterpret_cast<const int8_t*>(
        tensor_constant_data(graph_, tensors_[tensor_id]));
  }
  return activation(tensor_id);
}

const int8_t* Int8Interpreter::tensor_data(uint32_t tensor_id) const {
  if (!loaded_ || tensor_id >= graph_.header.num_tensors)
    return nullptr;
  if ((tensors_[tensor_id].flags & TENSOR_FLAG_CONSTANT) != 0) {
    return reinterpret_cast<const int8_t*>(
        tensor_constant_data(graph_, tensors_[tensor_id]));
  }
  return activation(tensor_id);
}

bool Int8Interpreter::tensor_info(uint32_t tensor_id, Int8TensorInfo& info)
    const {
  if (!loaded_ || tensor_id >= graph_.header.num_tensors)
    return false;
  const TensorRecord& tensor = tensors_[tensor_id];
  info = {};
  info.tensor_id = tensor_id;
  info.rank = tensor.rank;
  std::memcpy(info.shape, tensor.shape, sizeof(info.shape));
  info.bytes = tensor_data_size(tensor);
  info.scale = tensor.quant_scale;
  info.zero_point = tensor.quant_zero_point;
  if (tensor.rank == 4) {
    info.height = tensor.shape[2];
    info.width = tensor.shape[3];
    info.channels = tensor.shape[1];
  } else {
    info.height = 1;
    info.width = 1;
    info.channels = static_cast<int32_t>(tensor_element_count(tensor));
  }
  return true;
}

bool Int8Interpreter::input_info(size_t index, Int8TensorInfo& info) const {
  return index < input_count_ && tensor_info(inputs_[index], info);
}

bool Int8Interpreter::output_info(size_t index, Int8TensorInfo& info) const {
  return index < output_count_ && tensor_info(outputs_[index], info);
}

int8_t* Int8Interpreter::input_data(size_t index) {
  return loaded_ && index < input_count_ ? activation(inputs_[index]) : nullptr;
}

const int8_t* Int8Interpreter::output_data(size_t index) const {
  return loaded_ && index < output_count_ ? activation(outputs_[index])
                                          : nullptr;
}

bool Int8Interpreter::quantize_input_nchw(
    size_t index,
    const float* values,
    size_t count) {
  Int8TensorInfo info{};
  if (values == nullptr || !input_info(index, info) || count != info.bytes)
    return false;
  const float inverse =
      static_cast<float>(1.0 / static_cast<double>(info.scale));
  int8_t* destination = input_data(index);
  const size_t plane = static_cast<size_t>(info.height) * info.width;
  for (int32_t c = 0; c < info.channels; ++c) {
    for (size_t s = 0; s < plane; ++s) {
      destination[s * info.channels + c] =
          quantize_value(values[c * plane + s], inverse, info.zero_point);
    }
  }
  return true;
}

bool Int8Interpreter::set_input_rgb888(
    size_t index,
    const uint8_t* rgb,
    int32_t width,
    int32_t height,
    const float mean[3],
    const float stddev[3]) {
  Int8TensorInfo info{};
  if (rgb == nullptr || !input_info(index, info) || info.channels != 3 ||
      info.width != width || info.height != height)
    return false;
  const float inverse =
      static_cast<float>(1.0 / static_cast<double>(info.scale));
  int8_t lut[3][256];
  for (int c = 0; c < 3; ++c) {
    for (int v = 0; v < 256; ++v) {
      const float normalized =
          (static_cast<float>(v) / 255.0f - mean[c]) / stddev[c];
      lut[c][v] = quantize_value(normalized, inverse, info.zero_point);
    }
  }
  int8_t* destination = input_data(index);
  const size_t pixels = static_cast<size_t>(width) * height;
  for (size_t i = 0; i < pixels; ++i) {
    destination[i * 3 + 0] = lut[0][rgb[i * 3 + 0]];
    destination[i * 3 + 1] = lut[1][rgb[i * 3 + 1]];
    destination[i * 3 + 2] = lut[2][rgb[i * 3 + 2]];
  }
  return true;
}

bool Int8Interpreter::dequantize_output(
    size_t index,
    float* values,
    size_t count) const {
  Int8TensorInfo info{};
  if (values == nullptr || !output_info(index, info) || count != info.bytes)
    return false;
  const int8_t* source = output_data(index);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>(source[i] - info.zero_point) * info.scale;
  }
  return true;
}

int32_t Int8Interpreter::output_argmax(size_t index) const {
  Int8TensorInfo info{};
  if (!output_info(index, info))
    return -1;
  const int8_t* source = output_data(index);
  int32_t best = 0;
  for (size_t i = 1; i < info.bytes; ++i) {
    if (source[i] > source[best])
      best = static_cast<int32_t>(i);
  }
  return best;
}

int64_t Int8Interpreter::op_time_us(uint32_t op) const {
  return loaded_ && op < graph_.header.num_operations ? op_states_[op].last_us
                                                      : 0;
}

bool Int8Interpreter::run_operation(uint32_t index) {
  OpState& s = op_states_[index];
  const bool reference = options_.kernel_mode == KernelMode::REFERENCE;
  const int8_t* input = activation(s.input0);
  int8_t* output = activation(s.output);
  switch (s.opcode) {
    case OpCode::CONV2D:
      if (s.generic) {
        run_generic_conv(s, input, output);
        return true;
      }
      if (reference) {
        esp_nn_conv_s8_ansi(
            &s.input_dims,
            input,
            &s.filter_dims,
            s.weights,
            s.bias,
            &s.output_dims,
            output,
            &s.conv,
            &s.quant);
      } else {
        esp_nn_conv_s8(
            &s.input_dims,
            input,
            &s.filter_dims,
            s.weights,
            s.bias,
            &s.output_dims,
            output,
            &s.conv,
            &s.quant);
      }
      return true;
    case OpCode::DEPTHWISE_CONV2D:
      if (s.generic) {
        run_generic_conv(s, input, output);
        return true;
      }
      if (reference) {
        esp_nn_depthwise_conv_s8_ansi(
            &s.input_dims,
            input,
            &s.filter_dims,
            s.weights,
            s.bias,
            &s.output_dims,
            output,
            &s.dw,
            &s.quant);
      } else {
        esp_nn_depthwise_conv_s8(
            &s.input_dims,
            input,
            &s.filter_dims,
            s.weights,
            s.bias,
            &s.output_dims,
            output,
            &s.dw,
            &s.quant);
      }
      return true;
    case OpCode::RELU6:
      for (uint32_t i = 0; i < s.elements; ++i) {
        int32_t value = input[i];
        if (value < s.value0)
          value = s.value0;
        if (value > s.value1)
          value = s.value1;
        output[i] = static_cast<int8_t>(value);
      }
      return true;
    case OpCode::ADD: {
      input = operand(s.input0);
      const int8_t* rhs = operand(s.input1);
      if (reference) {
        esp_nn_add_elementwise_s8_ansi(
            input,
            rhs,
            -s.input_zero_point,
            -s.value0,
            s.add.input1_mult,
            s.add.input2_mult,
            s.add.input1_shift,
            s.add.input2_shift,
            s.add.left_shift,
            output,
            s.output_zero_point,
            s.add.output_mult,
            s.add.output_shift,
            s.add.activation_min,
            s.add.activation_max,
            static_cast<int32_t>(s.elements));
      } else {
        esp_nn_add_elementwise_s8(
            input,
            rhs,
            -s.input_zero_point,
            -s.value0,
            s.add.input1_mult,
            s.add.input2_mult,
            s.add.input1_shift,
            s.add.input2_shift,
            s.add.left_shift,
            output,
            s.output_zero_point,
            s.add.output_mult,
            s.add.output_shift,
            s.add.activation_min,
            s.add.activation_max,
            static_cast<int32_t>(s.elements));
      }
      return true;
    }
    case OpCode::AVGPOOL2D:
      if (reference) {
        esp_nn_mean_nhwc_s8_ansi(
            input,
            output,
            s.input_dims.height,
            s.input_dims.width,
            s.input_dims.channels,
            s.input_zero_point,
            s.output_zero_point,
            s.value0,
            s.value1);
      } else {
        esp_nn_mean_nhwc_s8(
            input,
            output,
            s.input_dims.height,
            s.input_dims.width,
            s.input_dims.channels,
            s.input_zero_point,
            s.output_zero_point,
            s.value0,
            s.value1);
      }
      return true;
    case OpCode::FLATTEN:
      if (s.transpose) {
        const int32_t plane = s.input_dims.height * s.input_dims.width;
        const int32_t channels = s.input_dims.channels;
        for (int32_t c = 0; c < channels; ++c) {
          for (int32_t p = 0; p < plane; ++p) {
            output[c * plane + p] = input[p * channels + c];
          }
        }
      } else {
        std::memcpy(output, input, s.elements);
      }
      return true;
    case OpCode::MAX_POOL2D:
    case OpCode::AVG_POOL2D: {
      const uint16_t in_w = static_cast<uint16_t>(s.input_dims.width);
      const uint16_t in_h = static_cast<uint16_t>(s.input_dims.height);
      const uint16_t out_w = static_cast<uint16_t>(s.output_dims.width);
      const uint16_t out_h = static_cast<uint16_t>(s.output_dims.height);
      const uint16_t channels = static_cast<uint16_t>(s.input_dims.channels);
      const uint16_t kh = s.pool[0], kw = s.pool[1], sh = s.pool[2],
                     sw = s.pool[3];
      const uint16_t ph = s.pool[4], pw = s.pool[5];
      if (s.opcode == OpCode::MAX_POOL2D) {
        if (reference) {
          esp_nn_max_pool_s8_ansi(
              input,
              in_w,
              in_h,
              output,
              out_w,
              out_h,
              sw,
              sh,
              kw,
              kh,
              pw,
              ph,
              s.value0,
              s.value1,
              channels);
        } else {
          esp_nn_max_pool_s8(
              input,
              in_w,
              in_h,
              output,
              out_w,
              out_h,
              sw,
              sh,
              kw,
              kh,
              pw,
              ph,
              s.value0,
              s.value1,
              channels);
        }
      } else if (reference) {
        esp_nn_avg_pool_s8_ansi(
            input,
            in_w,
            in_h,
            output,
            out_w,
            out_h,
            sw,
            sh,
            kw,
            kh,
            pw,
            ph,
            s.value0,
            s.value1,
            channels);
      } else {
        esp_nn_avg_pool_s8(
            input,
            in_w,
            in_h,
            output,
            out_w,
            out_h,
            sw,
            sh,
            kw,
            kh,
            pw,
            ph,
            s.value0,
            s.value1,
            channels);
      }
      return true;
    }
    case OpCode::LUT: {
      const uint8_t* table = reinterpret_cast<const uint8_t*>(s.weights) + 128;
      const uint32_t words = s.elements / 4;
      const uint32_t* source = reinterpret_cast<const uint32_t*>(input);
      uint32_t* destination = reinterpret_cast<uint32_t*>(output);
      for (uint32_t w = 0; w < words; ++w) {
        const uint32_t packed = source[w];
        destination[w] =
            static_cast<uint32_t>(table[static_cast<int8_t>(packed)]) |
            (static_cast<uint32_t>(table[static_cast<int8_t>(packed >> 8)])
             << 8) |
            (static_cast<uint32_t>(table[static_cast<int8_t>(packed >> 16)])
             << 16) |
            (static_cast<uint32_t>(table[static_cast<int8_t>(packed >> 24)])
             << 24);
      }
      for (uint32_t i = words * 4; i < s.elements; ++i) {
        output[i] = static_cast<int8_t>(table[input[i]]);
      }
      return true;
    }
    case OpCode::MUL: {
      input = operand(s.input0);
      const int8_t* rhs = operand(s.input1);
      if (s.transpose) {
        if (reference) {
          esp_nn_mul_broadcast_channel_s8_ansi(
              input,
              rhs,
              -s.input_zero_point,
              -s.value0,
              output,
              s.output_zero_point,
              s.add.output_mult,
              s.add.output_shift,
              s.add.activation_min,
              s.add.activation_max,
              s.rows,
              s.cols);
        } else {
          esp_nn_mul_broadcast_channel_s8(
              input,
              rhs,
              -s.input_zero_point,
              -s.value0,
              output,
              s.output_zero_point,
              s.add.output_mult,
              s.add.output_shift,
              s.add.activation_min,
              s.add.activation_max,
              s.rows,
              s.cols);
        }
      } else if (reference) {
        esp_nn_mul_elementwise_s8_ansi(
            input,
            rhs,
            -s.input_zero_point,
            -s.value0,
            output,
            s.output_zero_point,
            s.add.output_mult,
            s.add.output_shift,
            s.add.activation_min,
            s.add.activation_max,
            static_cast<int32_t>(s.elements));
      } else {
        esp_nn_mul_elementwise_s8(
            input,
            rhs,
            -s.input_zero_point,
            -s.value0,
            output,
            s.output_zero_point,
            s.add.output_mult,
            s.add.output_shift,
            s.add.activation_min,
            s.add.activation_max,
            static_cast<int32_t>(s.elements));
      }
      return true;
    }
    case OpCode::SOFTMAX: {
      const uint32_t* table = reinterpret_cast<const uint32_t*>(s.multipliers);
      for (int32_t r = 0; r < s.rows; ++r) {
        const int8_t* row = input + r * s.cols;
        int8_t* destination = output + r * s.cols;
        int32_t maximum = row[0];
        for (int32_t c = 1; c < s.cols; ++c) {
          if (row[c] > maximum)
            maximum = row[c];
        }
        uint64_t total = 0;
        for (int32_t c = 0; c < s.cols; ++c)
          total += table[maximum - row[c]];
        for (int32_t c = 0; c < s.cols; ++c) {
          const uint64_t numerator = 512ULL * table[maximum - row[c]] + total;
          const int32_t value =
              static_cast<int32_t>(numerator / (2ULL * total)) +
              SOFTMAX_OUTPUT_ZERO_POINT;
          destination[c] = static_cast<int8_t>(value > 127 ? 127 : value);
        }
      }
      return true;
    }
    case OpCode::CONCAT: {
      int32_t offset = 0;
      for (uint8_t k = 0; k < s.input_count; ++k) {
        const int32_t channels = tensors_[s.inputs[k]].shape[1];
        const int8_t* source = activation(s.inputs[k]);
        for (int32_t r = 0; r < s.rows; ++r) {
          std::memcpy(
              output + r * s.cols + offset, source + r * channels, channels);
        }
        offset += channels;
      }
      return true;
    }
    case OpCode::LINEAR:
      for (int32_t b = 0; b < s.batches; ++b) {
        const int8_t* row = input + static_cast<size_t>(b) * s.cols;
        int8_t* destination = output + static_cast<size_t>(b) * s.rows;
        if (reference) {
          esp_nn_fully_connected_per_ch_s8_ansi(
              row,
              -s.input_zero_point,
              static_cast<uint16_t>(s.cols),
              s.weights,
              0,
              s.bias,
              destination,
              static_cast<uint16_t>(s.rows),
              s.output_zero_point,
              s.shifts,
              s.multipliers,
              s.value0,
              s.value1);
        } else {
          esp_nn_fully_connected_per_ch_s8(
              row,
              -s.input_zero_point,
              static_cast<uint16_t>(s.cols),
              s.weights,
              0,
              s.bias,
              destination,
              static_cast<uint16_t>(s.rows),
              s.output_zero_point,
              s.shifts,
              s.multipliers,
              s.value0,
              s.value1);
        }
      }
      return true;
    case OpCode::COPY:
      run_copy(s, output);
      return true;
    case OpCode::UPSAMPLE:
      run_upsample(s, input, output);
      return true;
    case OpCode::TRANSPOSE_CONV2D:
      run_transposed_conv(s, input, output);
      return true;
    default:
      return false;
  }
}

void Int8Interpreter::run_dilated_by_phases(
    const OpState& s,
    const int8_t* input,
    int8_t* output) const {
  const int32_t in_h = s.input_dims.height;
  const int32_t in_w = s.input_dims.width;
  const int32_t in_c = s.input_dims.channels;
  const int32_t out_h = s.output_dims.height;
  const int32_t out_w = s.output_dims.width;
  const int32_t out_c = s.output_dims.channels;
  const int32_t kh = s.pool[0], kw = s.pool[1], ph = s.pool[4], pw = s.pool[5];
  const int32_t dh = s.dilation_h, dw = s.dilation_w;
  const bool reference = options_.kernel_mode == KernelMode::REFERENCE;
  const int8_t pad_value = clamp_to_int8(s.input_zero_point);
  for (int32_t qy = 0; qy < dh; ++qy) {
    const int32_t rows = (out_h - qy + dh - 1) / dh;
    if (rows <= 0)
      continue;
    const int32_t ty = qy - ph;
    const int32_t ry = ((ty % dh) + dh) % dh;
    const int32_t by = (ty - ry) / dh;
    for (int32_t qx = 0; qx < dw; ++qx) {
      const int32_t cols = (out_w - qx + dw - 1) / dw;
      if (cols <= 0)
        continue;
      const int32_t tx = qx - pw;
      const int32_t rx = ((tx % dw) + dw) % dw;
      const int32_t bx = (tx - rx) / dw;
      const int32_t tile_h = rows + kh - 1;
      const int32_t tile_w = cols + kw - 1;
      int8_t* tile = phase_buffer_;
      int8_t* result =
          phase_buffer_ + static_cast<size_t>(tile_h) * tile_w * in_c;
      for (int32_t i = 0; i < tile_h; ++i) {
        const int32_t iy = ry + dh * (by + i);
        int8_t* row = tile + static_cast<size_t>(i) * tile_w * in_c;
        for (int32_t j = 0; j < tile_w; ++j) {
          const int32_t ix = rx + dw * (bx + j);
          int8_t* pixel = row + static_cast<size_t>(j) * in_c;
          if (iy < 0 || iy >= in_h || ix < 0 || ix >= in_w) {
            std::memset(pixel, pad_value, in_c);
          } else {
            std::memcpy(
                pixel,
                input + (static_cast<size_t>(iy) * in_w + ix) * in_c,
                in_c);
          }
        }
      }
      data_dims_t tile_dims{tile_w, tile_h, in_c, 1};
      data_dims_t result_dims{cols, rows, out_c, 1};
      if (s.opcode == OpCode::CONV2D) {
        if (reference) {
          esp_nn_conv_s8_ansi(
              &tile_dims,
              tile,
              &s.filter_dims,
              s.weights,
              s.bias,
              &result_dims,
              result,
              &s.conv,
              &s.quant);
        } else {
          esp_nn_conv_s8(
              &tile_dims,
              tile,
              &s.filter_dims,
              s.weights,
              s.bias,
              &result_dims,
              result,
              &s.conv,
              &s.quant);
        }
      } else if (reference) {
        esp_nn_depthwise_conv_s8_ansi(
            &tile_dims,
            tile,
            &s.filter_dims,
            s.weights,
            s.bias,
            &result_dims,
            result,
            &s.dw,
            &s.quant);
      } else {
        esp_nn_depthwise_conv_s8(
            &tile_dims,
            tile,
            &s.filter_dims,
            s.weights,
            s.bias,
            &result_dims,
            result,
            &s.dw,
            &s.quant);
      }
      for (int32_t i = 0; i < rows; ++i) {
        const int32_t oy = qy + dh * i;
        for (int32_t j = 0; j < cols; ++j) {
          const int32_t ox = qx + dw * j;
          std::memcpy(
              output + (static_cast<size_t>(oy) * out_w + ox) * out_c,
              result + (static_cast<size_t>(i) * cols + j) * out_c,
              out_c);
        }
      }
    }
  }
}

void Int8Interpreter::run_generic_conv(
    const OpState& s,
    const int8_t* input,
    int8_t* output) const {
  if (s.transpose) {
    run_dilated_by_phases(s, input, output);
    return;
  }
  const int32_t in_h = s.input_dims.height;
  const int32_t in_w = s.input_dims.width;
  const int32_t in_c = s.input_dims.channels;
  const int32_t out_h = s.output_dims.height;
  const int32_t out_w = s.output_dims.width;
  const int32_t out_c = s.output_dims.channels;
  const int32_t kh = s.pool[0], kw = s.pool[1], sh = s.pool[2], sw = s.pool[3];
  const int32_t ph = s.pool[4], pw = s.pool[5];
  const int32_t dh = s.dilation_h, dw = s.dilation_w;
  const bool depthwise = s.opcode == OpCode::DEPTHWISE_CONV2D;
  const int32_t groups = depthwise ? in_c : s.rows;
  const int32_t group_in = depthwise ? 1 : in_c / groups;
  const int32_t group_out = out_c / groups;
  const int32_t in_offset = -s.input_zero_point;
  for (int32_t oy = 0; oy < out_h; ++oy) {
    for (int32_t ox = 0; ox < out_w; ++ox) {
      int8_t* destination =
          output + (static_cast<size_t>(oy) * out_w + ox) * out_c;
      for (int32_t oc = 0; oc < out_c; ++oc) {
        const int32_t g = oc / group_out;
        int32_t acc = s.bias != nullptr ? s.bias[oc] : 0;
        for (int32_t ky = 0; ky < kh; ++ky) {
          const int32_t iy = oy * sh - ph + ky * dh;
          if (iy < 0 || iy >= in_h)
            continue;
          for (int32_t kx = 0; kx < kw; ++kx) {
            const int32_t ix = ox * sw - pw + kx * dw;
            if (ix < 0 || ix >= in_w)
              continue;
            const int8_t* pixel =
                input + (static_cast<size_t>(iy) * in_w + ix) * in_c;
            if (depthwise) {
              const int32_t ic = oc / (out_c / in_c);
              acc += (pixel[ic] + in_offset) *
                  s.weights[(static_cast<size_t>(ky) * kw + kx) * out_c + oc];
            } else {
              const int8_t* filter = s.weights +
                  ((static_cast<size_t>(oc) * kh + ky) * kw + kx) * group_in;
              const int8_t* source = pixel + g * group_in;
              for (int32_t ic = 0; ic < group_in; ++ic) {
                acc += (source[ic] + in_offset) * filter[ic];
              }
            }
          }
        }
        int32_t value = requantize(acc, s.multipliers[oc], s.shifts[oc]) +
            s.output_zero_point;
        if (value < s.value0)
          value = s.value0;
        if (value > s.value1)
          value = s.value1;
        destination[oc] = static_cast<int8_t>(value);
      }
    }
  }
}

void Int8Interpreter::run_copy(const OpState& s, int8_t* output) const {
  const uint8_t* p = s.table;
  const int32_t segments = read_le_i32(p);
  const int32_t flags = read_le_i32(p + 4);
  if ((flags & COPY_FLAG_FILL) != 0) {
    std::memset(
        output, read_le_i32(p + 8), tensor_data_size(tensors_[s.output]));
  }
  for (int32_t k = 0; k < segments; ++k) {
    const uint8_t* q =
        p + COPY_HEADER_SIZE + static_cast<size_t>(k) * COPY_SEGMENT_SIZE;
    const int8_t* source = operand(s.inputs[read_le_i32(q)]);
    const int32_t rank = read_le_i32(q + 4);
    const int32_t in_base = read_le_i32(q + 8);
    const int32_t out_base = read_le_i32(q + 12);
    int32_t dims[COPY_MAX_RANK];
    int32_t in_strides[COPY_MAX_RANK];
    int32_t out_strides[COPY_MAX_RANK];
    for (int32_t d = 0; d < static_cast<int32_t>(COPY_MAX_RANK); ++d) {
      const int32_t from = d - (static_cast<int32_t>(COPY_MAX_RANK) - rank);
      dims[d] = from >= 0 ? read_le_i32(q + 16 + 4 * from) : 1;
      in_strides[d] = from >= 0 ? read_le_i32(q + 40 + 4 * from) : 0;
      out_strides[d] = from >= 0 ? read_le_i32(q + 64 + 4 * from) : 0;
    }
    const bool requant = read_le_i32(q + 88) != 0;
    const int32_t multiplier = read_le_i32(q + 92);
    const int32_t shift = read_le_i32(q + 96);
    const int32_t in_zp = read_le_i32(q + 100);
    const int32_t out_zp = read_le_i32(q + 104);
    const int32_t inner = dims[5];
    const bool contiguous =
        in_strides[5] == 1 && out_strides[5] == 1 && !requant;
    for (int32_t a = 0; a < dims[0]; ++a)
      for (int32_t b = 0; b < dims[1]; ++b)
        for (int32_t c = 0; c < dims[2]; ++c)
          for (int32_t d = 0; d < dims[3]; ++d)
            for (int32_t e = 0; e < dims[4]; ++e) {
              const int32_t in_index = in_base + a * in_strides[0] +
                  b * in_strides[1] + c * in_strides[2] + d * in_strides[3] +
                  e * in_strides[4];
              const int32_t out_index = out_base + a * out_strides[0] +
                  b * out_strides[1] + c * out_strides[2] + d * out_strides[3] +
                  e * out_strides[4];
              if (contiguous) {
                std::memcpy(output + out_index, source + in_index, inner);
                continue;
              }
              const int8_t* from = source + in_index;
              int8_t* to = output + out_index;
              for (int32_t f = 0; f < inner; ++f) {
                int32_t value = from[f * in_strides[5]];
                if (requant)
                  value = clamp_to_int8(
                      requantize(value - in_zp, multiplier, shift) + out_zp);
                to[f * out_strides[5]] = static_cast<int8_t>(value);
              }
            }
  }
}

void Int8Interpreter::run_upsample(
    const OpState& s,
    const int8_t* input,
    int8_t* output) const {
  const int32_t in_w = s.input_dims.width;
  const int32_t channels = s.input_dims.channels;
  const int32_t out_h = s.output_dims.height;
  const int32_t out_w = s.output_dims.width;
  const uint8_t* y_table = s.table + UPSAMPLE_HEADER_SIZE;
  const uint8_t* x_table = y_table + 12U * static_cast<size_t>(out_h);
  const int32_t one = 1 << UPSAMPLE_WEIGHT_BITS;
  const int32_t half = 1 << (2 * UPSAMPLE_WEIGHT_BITS - 1);
  const bool nearest = s.value0 == UPSAMPLE_MODE_NEAREST;
  for (int32_t oy = 0; oy < out_h; ++oy) {
    const int32_t y0 = read_le_i32(y_table + 12 * oy);
    const int32_t y1 = read_le_i32(y_table + 12 * oy + 4);
    const int32_t wy = read_le_i32(y_table + 12 * oy + 8);
    const int8_t* row0 = input + static_cast<size_t>(y0) * in_w * channels;
    const int8_t* row1 = input + static_cast<size_t>(y1) * in_w * channels;
    int8_t* destination = output + static_cast<size_t>(oy) * out_w * channels;
    for (int32_t ox = 0; ox < out_w; ++ox) {
      const int32_t x0 = read_le_i32(x_table + 12 * ox);
      const int32_t x1 = read_le_i32(x_table + 12 * ox + 4);
      const int32_t wx = read_le_i32(x_table + 12 * ox + 8);
      int8_t* pixel = destination + static_cast<size_t>(ox) * channels;
      if (nearest) {
        std::memcpy(pixel, row0 + static_cast<size_t>(x0) * channels, channels);
        continue;
      }
      const int8_t* a = row0 + static_cast<size_t>(x0) * channels;
      const int8_t* b = row0 + static_cast<size_t>(x1) * channels;
      const int8_t* c = row1 + static_cast<size_t>(x0) * channels;
      const int8_t* d = row1 + static_cast<size_t>(x1) * channels;
      for (int32_t ch = 0; ch < channels; ++ch) {
        const int32_t top = a[ch] * (one - wx) + b[ch] * wx;
        const int32_t bottom = c[ch] * (one - wx) + d[ch] * wx;
        const int32_t total = top * (one - wy) + bottom * wy + half;
        pixel[ch] = clamp_to_int8(total >> (2 * UPSAMPLE_WEIGHT_BITS));
      }
    }
  }
}

void Int8Interpreter::run_transposed_conv(
    const OpState& s,
    const int8_t* input,
    int8_t* output) const {
  const int32_t in_h = s.input_dims.height;
  const int32_t in_w = s.input_dims.width;
  const int32_t in_c = s.input_dims.channels;
  const int32_t out_h = s.output_dims.height;
  const int32_t out_w = s.output_dims.width;
  const int32_t out_c = s.output_dims.channels;
  const int32_t kh = s.pool[0], kw = s.pool[1], sh = s.pool[2], sw = s.pool[3];
  const int32_t ph = s.pool[4], pw = s.pool[5];
  const int32_t dh = s.dilation_h, dw = s.dilation_w;
  const int32_t in_offset = -s.input_zero_point;
  for (int32_t oy = 0; oy < out_h; ++oy) {
    for (int32_t ox = 0; ox < out_w; ++ox) {
      int8_t* destination =
          output + (static_cast<size_t>(oy) * out_w + ox) * out_c;
      for (int32_t oc = 0; oc < out_c; ++oc) {
        int32_t acc = s.bias[oc];
        for (int32_t ky = 0; ky < kh; ++ky) {
          const int32_t ty = oy + ph - ky * dh;
          if (ty < 0 || ty % sh != 0)
            continue;
          const int32_t iy = ty / sh;
          if (iy >= in_h)
            continue;
          for (int32_t kx = 0; kx < kw; ++kx) {
            const int32_t tx = ox + pw - kx * dw;
            if (tx < 0 || tx % sw != 0)
              continue;
            const int32_t ix = tx / sw;
            if (ix >= in_w)
              continue;
            const int8_t* pixel =
                input + (static_cast<size_t>(iy) * in_w + ix) * in_c;
            const int8_t* filter = s.weights +
                ((static_cast<size_t>(oc) * kh + ky) * kw + kx) * in_c;
            for (int32_t ic = 0; ic < in_c; ++ic)
              acc += (pixel[ic] + in_offset) * filter[ic];
          }
        }
        int32_t value = requantize(acc, s.multipliers[oc], s.shifts[oc]) +
            s.output_zero_point;
        if (value < s.value0)
          value = s.value0;
        if (value > s.value1)
          value = s.value1;
        destination[oc] = static_cast<int8_t>(value);
      }
    }
  }
}

bool Int8Interpreter::invoke() {
  if (!loaded_)
    return false;
  if (scratch_ != nullptr) {
    esp_nn_set_conv_scratch_buf(scratch_);
    esp_nn_set_depthwise_conv_scratch_buf(scratch_);
    esp_nn_set_mean_scratch_buf(scratch_);
  }
  int64_t compute_us = 0;
  int64_t last_yield = esp_timer_get_time();
  for (uint32_t i = 0; i < graph_.header.num_operations; ++i) {
    const int64_t started = esp_timer_get_time();
    if (!run_operation(i)) {
      std::printf(
          "ERROR: operation %u (%s) failed\n",
          static_cast<unsigned>(i),
          opcode_name(op_states_[i].opcode));
      return false;
    }
    const int64_t finished = esp_timer_get_time();
    OpState& state = op_states_[i];
    state.last_us = finished - started;
    state.total_us += state.last_us;
    ++state.calls;
    compute_us += state.last_us;
    if (finished - last_yield > YIELD_INTERVAL_US) {
      vTaskDelay(1);
      last_yield = esp_timer_get_time();
    }
  }
  last_invoke_us_ = compute_us;
  return true;
}

void Int8Interpreter::print_summary() const {
  if (!loaded_)
    return;
  std::printf(
      "Int8 model: %u tensors, %u operations, %u constant bytes\n",
      static_cast<unsigned>(graph_.header.num_tensors),
      static_cast<unsigned>(graph_.header.num_operations),
      static_cast<unsigned>(graph_.header.constant_size));
  std::printf(
      "Activation arena: %u bytes in %s\n",
      static_cast<unsigned>(arena_size_),
      arena_internal_ ? "internal RAM" : "PSRAM");
  std::printf(
      "ESP-NN scratch: %u bytes; requant tables: %u bytes; constants %s\n",
      static_cast<unsigned>(scratch_size_),
      static_cast<unsigned>(params_size_),
      blob_copy_ != nullptr ? "copied to RAM" : "read in place from flash");
  for (size_t i = 0; i < input_count_; ++i) {
    Int8TensorInfo info{};
    input_info(i, info);
    std::printf(
        "Input %u: tensor %u HWC=%ldx%ldx%ld scale=%.9g zero_point=%ld\n",
        static_cast<unsigned>(i),
        static_cast<unsigned>(info.tensor_id),
        static_cast<long>(info.height),
        static_cast<long>(info.width),
        static_cast<long>(info.channels),
        info.scale,
        static_cast<long>(info.zero_point));
  }
  for (size_t i = 0; i < output_count_; ++i) {
    Int8TensorInfo info{};
    output_info(i, info);
    std::printf(
        "Output %u: tensor %u elements=%u scale=%.9g zero_point=%ld\n",
        static_cast<unsigned>(i),
        static_cast<unsigned>(info.tensor_id),
        static_cast<unsigned>(info.bytes),
        info.scale,
        static_cast<long>(info.zero_point));
  }
}

void Int8Interpreter::print_profile() const {
  if (!loaded_)
    return;
  int64_t per_type[OPCODE_COUNT]{};
  uint32_t count_per_type[OPCODE_COUNT]{};
  for (uint32_t i = 0; i < graph_.header.num_operations; ++i) {
    const OpState& s = op_states_[i];
    const unsigned type = static_cast<unsigned>(s.opcode);
    if (type < OPCODE_COUNT) {
      per_type[type] += s.last_us;
      ++count_per_type[type];
    }
    std::printf(
        "  op %2u %-16s out=%ldx%ldx%ld %8lld us\n",
        static_cast<unsigned>(i),
        opcode_name(s.opcode),
        static_cast<long>(
            tensors_[s.output].rank == 4 ? tensors_[s.output].shape[2] : 1),
        static_cast<long>(
            tensors_[s.output].rank == 4 ? tensors_[s.output].shape[3] : 1),
        static_cast<long>(
            tensors_[s.output].rank == 4 ? tensors_[s.output].shape[1]
                                         : static_cast<int32_t>(s.elements)),
        static_cast<long long>(s.last_us));
  }
  for (unsigned t = 0; t < OPCODE_COUNT; ++t) {
    if (count_per_type[t] == 0)
      continue;
    std::printf(
        "  %-16s x%-3u %8lld us\n",
        opcode_name(static_cast<OpCode>(t)),
        static_cast<unsigned>(count_per_type[t]),
        static_cast<long long>(per_type[t]));
  }
}

} // namespace espressif
} // namespace backends
} // namespace executorch
