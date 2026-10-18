/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <espnn/xts3_graph.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace executorch {
namespace backends {
namespace espressif {
namespace {

uint16_t read_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
      (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

int32_t read_i32(const uint8_t* p) {
  return static_cast<int32_t>(read_u32(p));
}

bool range_valid(size_t offset, size_t length, size_t total) {
  return offset <= total && length <= total - offset;
}

bool valid_dtype(uint8_t dtype) {
  return dtype >= static_cast<uint8_t>(DataType::FLOAT32) &&
      dtype <= static_cast<uint8_t>(DataType::INT32);
}

bool valid_opcode(uint8_t opcode, uint32_t version) {
  if (version == GRAPH_ESP_NN_INT8_IO_VERSION) {
    return opcode <= static_cast<uint8_t>(OpCode::TRANSPOSE_CONV2D);
  }
  return opcode <= static_cast<uint8_t>(OpCode::SOFTMAX);
}

bool valid_tensor_reference(const GraphView& graph, uint32_t id) {
  return id < graph.header.num_tensors;
}

bool same_shape(const TensorRecord& a, const TensorRecord& b) {
  return a.rank == b.rank &&
      std::memcmp(a.shape, b.shape, sizeof(a.shape)) == 0;
}

bool same_quantization(const TensorRecord& a, const TensorRecord& b) {
  return a.quant_scale == b.quant_scale &&
      a.quant_zero_point == b.quant_zero_point;
}

bool valid_requantization(int32_t multiplier, int32_t shift) {
  return multiplier > 0 && shift >= -31 && shift <= 30;
}

bool valid_activation_range(int32_t low, int32_t high) {
  return low >= -128 && high <= 127 && low <= high;
}

bool is_int8_activation(const TensorRecord& tensor) {
  return tensor.dtype == static_cast<uint8_t>(DataType::INT8) &&
      (tensor.flags & TENSOR_FLAG_CONSTANT) == 0 &&
      (tensor.flags & TENSOR_FLAG_QUANTIZED) != 0;
}

bool is_int8_value(const TensorRecord& tensor) {
  return tensor.dtype == static_cast<uint8_t>(DataType::INT8) &&
      (tensor.flags & TENSOR_FLAG_QUANTIZED) != 0;
}

bool valid_layout(const TensorRecord& tensor) {
  return tensor.rank >= 1 && tensor.rank <= 4 &&
      (tensor.rank != 4 || tensor.shape[0] == 1);
}

bool is_constant_of(const TensorRecord& tensor, DataType dtype) {
  return tensor.dtype == static_cast<uint8_t>(dtype) &&
      (tensor.flags & TENSOR_FLAG_CONSTANT) != 0 &&
      (tensor.flags & TENSOR_FLAG_QUANTIZED) == 0;
}

bool validate_int8_conv(
    const GraphView& graph,
    const OperationRecord& op,
    bool depthwise) {
  if (op.num_inputs != 3 || op.num_outputs != 1)
    return false;
  TensorRecord input{};
  TensorRecord weight{};
  TensorRecord bias{};
  TensorRecord output{};
  if (!read_tensor(graph, op.inputs[0], input) ||
      !read_tensor(graph, op.inputs[1], weight) ||
      !read_tensor(graph, op.inputs[2], bias) ||
      !read_tensor(graph, op.outputs[0], output))
    return false;
  if (!is_int8_activation(input) || !is_int8_activation(output) ||
      !is_constant_of(weight, DataType::INT8) ||
      !is_constant_of(bias, DataType::INT32) || input.rank != 4 ||
      weight.rank != 4 || bias.rank != 1 || output.rank != 4 ||
      input.shape[0] != 1 || output.shape[0] != 1 ||
      weight.shape[0] != output.shape[1] || bias.shape[0] != output.shape[1])
    return false;

  const uint32_t channels = static_cast<uint32_t>(output.shape[1]);
  Int8ConvParams params{};
  if (!read_int8_conv_params(graph, op, channels, params))
    return false;
  if (params.dilation_h <= 0 || params.dilation_w <= 0)
    return false;
  if (graph.header.version == GRAPH_ESP_NN_INT8_VERSION &&
      (params.dilation_h != 1 || params.dilation_w != 1))
    return false;
  const int32_t effective_h = params.dilation_h * (params.kernel_h - 1) + 1;
  const int32_t effective_w = params.dilation_w * (params.kernel_w - 1) + 1;

  if (depthwise) {
    if (params.groups != input.shape[1] || weight.shape[1] != 1 ||
        output.shape[1] % input.shape[1] != 0)
      return false;
  } else if (
      params.groups <= 0 ||
      static_cast<int64_t>(weight.shape[1]) * params.groups != input.shape[1] ||
      output.shape[1] % params.groups != 0) {
    return false;
  }
  if (params.kernel_h != weight.shape[2] ||
      params.kernel_w != weight.shape[3] || params.stride_h <= 0 ||
      params.stride_w <= 0 || params.pad_h >= effective_h ||
      params.pad_w >= effective_w ||
      !valid_activation_range(params.activation_min, params.activation_max))
    return false;

  const int64_t expected_h =
      (static_cast<int64_t>(input.shape[2]) + 2 * params.pad_h - effective_h) /
          params.stride_h +
      1;
  const int64_t expected_w =
      (static_cast<int64_t>(input.shape[3]) + 2 * params.pad_w - effective_w) /
          params.stride_w +
      1;
  if (output.shape[2] != expected_h || output.shape[3] != expected_w)
    return false;
  if (input.shape[1] > 65535 || input.shape[2] > 65535 ||
      input.shape[3] > 65535 || output.shape[1] > 65535)
    return false;
  for (uint32_t i = 0; i < channels; ++i) {
    if (!valid_requantization(
            read_i32(params.multipliers + i * 4),
            read_i32(params.shifts + i * 4)))
      return false;
  }
  return true;
}

bool copy_extent_valid(
    int32_t rank,
    const int32_t* dims,
    const int32_t* strides,
    int32_t offset,
    size_t limit) {
  int64_t low = offset;
  int64_t high = offset;
  for (int32_t d = 0; d < rank; ++d) {
    if (dims[d] <= 0)
      return false;
    const int64_t span = static_cast<int64_t>(dims[d] - 1) * strides[d];
    if (span < 0)
      low += span;
    else
      high += span;
  }
  return low >= 0 && high < static_cast<int64_t>(limit);
}

bool validate_copy(const GraphView& graph, const OperationRecord& op) {
  TensorRecord out{};
  if (op.num_outputs != 1 || op.parameter_size < COPY_HEADER_SIZE ||
      !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(out) ||
      !valid_layout(out))
    return false;
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  const int32_t segments = read_i32(p);
  const int32_t flags = read_i32(p + 4);
  const int32_t fill = read_i32(p + 8);
  if (segments <= 0 || (flags & ~COPY_FLAG_FILL) != 0 || fill < -128 ||
      fill > 127 ||
      op.parameter_size !=
          COPY_HEADER_SIZE +
              static_cast<size_t>(segments) * COPY_SEGMENT_SIZE) {
    return false;
  }
  TensorRecord inputs[6]{};
  for (uint8_t j = 0; j < op.num_inputs; ++j) {
    if (!read_tensor(graph, op.inputs[j], inputs[j]) ||
        !is_int8_value(inputs[j]) || !valid_layout(inputs[j]))
      return false;
  }
  const size_t out_count = tensor_element_count(out);
  for (int32_t k = 0; k < segments; ++k) {
    const uint8_t* q =
        p + COPY_HEADER_SIZE + static_cast<size_t>(k) * COPY_SEGMENT_SIZE;
    const int32_t index = read_i32(q);
    const int32_t rank = read_i32(q + 4);
    const int32_t in_offset = read_i32(q + 8);
    const int32_t out_offset = read_i32(q + 12);
    int32_t dims[COPY_MAX_RANK];
    int32_t in_strides[COPY_MAX_RANK];
    int32_t out_strides[COPY_MAX_RANK];
    for (size_t d = 0; d < COPY_MAX_RANK; ++d) {
      dims[d] = read_i32(q + 16 + 4 * d);
      in_strides[d] = read_i32(q + 40 + 4 * d);
      out_strides[d] = read_i32(q + 64 + 4 * d);
    }
    const int32_t requant = read_i32(q + 88);
    if (index < 0 || index >= op.num_inputs || rank <= 0 ||
        rank > static_cast<int32_t>(COPY_MAX_RANK) ||
        (requant != 0 && requant != 1))
      return false;
    const TensorRecord& source = inputs[index];
    if (!copy_extent_valid(
            rank, dims, in_strides, in_offset, tensor_element_count(source)) ||
        !copy_extent_valid(rank, dims, out_strides, out_offset, out_count))
      return false;
    if (requant != 0) {
      if (!valid_requantization(read_i32(q + 92), read_i32(q + 96)) ||
          read_i32(q + 100) != source.quant_zero_point ||
          read_i32(q + 104) != out.quant_zero_point)
        return false;
    } else if (!same_quantization(source, out)) {
      return false;
    }
  }
  return true;
}

bool validate_upsample(const GraphView& graph, const OperationRecord& op) {
  TensorRecord a{};
  TensorRecord out{};
  if (op.num_inputs != 1 || op.num_outputs != 1 ||
      op.parameter_size < UPSAMPLE_HEADER_SIZE ||
      !read_tensor(graph, op.inputs[0], a) ||
      !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(a) ||
      !is_int8_activation(out) || !same_quantization(a, out) || a.rank != 4 ||
      out.rank != 4 || a.shape[0] != 1 || out.shape[0] != 1 ||
      a.shape[1] != out.shape[1])
    return false;
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  const int32_t mode = read_i32(p);
  const int32_t in_h = read_i32(p + 4);
  const int32_t in_w = read_i32(p + 8);
  const int32_t out_h = read_i32(p + 12);
  const int32_t out_w = read_i32(p + 16);
  if ((mode != UPSAMPLE_MODE_NEAREST && mode != UPSAMPLE_MODE_BILINEAR) ||
      in_h != a.shape[2] || in_w != a.shape[3] || out_h != out.shape[2] ||
      out_w != out.shape[3] ||
      op.parameter_size !=
          UPSAMPLE_HEADER_SIZE + 12U * static_cast<size_t>(out_h + out_w)) {
    return false;
  }
  const int32_t one = 1 << UPSAMPLE_WEIGHT_BITS;
  const uint8_t* table = p + UPSAMPLE_HEADER_SIZE;
  for (int32_t i = 0; i < out_h + out_w; ++i) {
    const int32_t limit = i < out_h ? in_h : in_w;
    const int32_t i0 = read_i32(table + 12 * i);
    const int32_t i1 = read_i32(table + 12 * i + 4);
    const int32_t weight = read_i32(table + 12 * i + 8);
    if (i0 < 0 || i0 >= limit || i1 < 0 || i1 >= limit || weight < 0 ||
        weight > one)
      return false;
  }
  return true;
}

bool validate_transposed_conv(
    const GraphView& graph,
    const OperationRecord& op) {
  TensorRecord input{};
  TensorRecord weight{};
  TensorRecord bias{};
  TensorRecord output{};
  if (op.num_inputs != 3 || op.num_outputs != 1 ||
      !read_tensor(graph, op.inputs[0], input) ||
      !read_tensor(graph, op.inputs[1], weight) ||
      !read_tensor(graph, op.inputs[2], bias) ||
      !read_tensor(graph, op.outputs[0], output) ||
      !is_int8_activation(input) || !is_int8_activation(output) ||
      !is_constant_of(weight, DataType::INT8) ||
      !is_constant_of(bias, DataType::INT32) || input.rank != 4 ||
      output.rank != 4 || weight.rank != 4 || bias.rank != 1 ||
      input.shape[0] != 1 || output.shape[0] != 1 ||
      weight.shape[0] != output.shape[1] || weight.shape[3] != input.shape[1] ||
      bias.shape[0] != output.shape[1])
    return false;
  const size_t channels = static_cast<size_t>(output.shape[1]);
  if (op.parameter_size != TRANSPOSE_CONV_PARAMS_SIZE + channels * 8U)
    return false;
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  int32_t f[13];
  for (int k = 0; k < 13; ++k)
    f[k] = read_i32(p + 4 * k);
  const int32_t kh = f[0], kw = f[1], sh = f[2], sw = f[3], ph = f[4],
                pw = f[5];
  const int32_t oph = f[6], opw = f[7], dh = f[8], dw = f[9], groups = f[10];
  if (kh != weight.shape[1] || kw != weight.shape[2] || sh <= 0 || sw <= 0 ||
      ph < 0 || pw < 0 || oph < 0 || opw < 0 || dh <= 0 || dw <= 0 ||
      groups != 1 || !valid_activation_range(f[11], f[12]))
    return false;
  const int64_t expected_h = static_cast<int64_t>(input.shape[2] - 1) * sh -
      2 * ph + static_cast<int64_t>(dh) * (kh - 1) + oph + 1;
  const int64_t expected_w = static_cast<int64_t>(input.shape[3] - 1) * sw -
      2 * pw + static_cast<int64_t>(dw) * (kw - 1) + opw + 1;
  if (output.shape[2] != expected_h || output.shape[3] != expected_w)
    return false;
  const uint8_t* multipliers = p + TRANSPOSE_CONV_PARAMS_SIZE;
  const uint8_t* shifts = multipliers + channels * 4U;
  for (size_t c = 0; c < channels; ++c) {
    if (!valid_requantization(
            read_i32(multipliers + 4 * c), read_i32(shifts + 4 * c)))
      return false;
  }
  return true;
}

bool validate_int8_operation(
    const GraphView& graph,
    const OperationRecord& op) {
  TensorRecord a{};
  TensorRecord b{};
  TensorRecord out{};
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  switch (static_cast<OpCode>(op.opcode)) {
    case OpCode::CONV2D:
      return validate_int8_conv(graph, op, false);
    case OpCode::DEPTHWISE_CONV2D:
      return validate_int8_conv(graph, op, true);
    case OpCode::RELU6:
      return op.num_inputs == 1 && op.num_outputs == 1 &&
          op.parameter_size == INT8_PAIR_PARAMS_SIZE &&
          read_tensor(graph, op.inputs[0], a) &&
          read_tensor(graph, op.outputs[0], out) && is_int8_activation(a) &&
          is_int8_activation(out) && same_shape(a, out) &&
          same_quantization(a, out) &&
          valid_activation_range(read_i32(p), read_i32(p + 4));
    case OpCode::ADD: {
      Int8AddParams params{};
      return op.num_inputs == 2 && op.num_outputs == 1 &&
          read_tensor(graph, op.inputs[0], a) &&
          read_tensor(graph, op.inputs[1], b) &&
          read_tensor(graph, op.outputs[0], out) && is_int8_value(a) &&
          is_int8_value(b) && is_int8_activation(out) && same_shape(a, b) &&
          same_shape(a, out) && read_int8_add_params(graph, op, params) &&
          params.input1_mult > 0 && params.input2_mult > 0 &&
          params.output_mult > 0 && params.input1_shift <= 0 &&
          params.input1_shift >= -31 && params.input2_shift <= 0 &&
          params.input2_shift >= -31 && params.output_shift <= 0 &&
          params.output_shift >= -31 && params.left_shift >= 0 &&
          params.left_shift <= 22 &&
          valid_activation_range(params.activation_min, params.activation_max);
    }
    case OpCode::AVGPOOL2D:
      return op.num_inputs == 1 && op.num_outputs == 1 &&
          op.parameter_size == INT8_PAIR_PARAMS_SIZE &&
          read_tensor(graph, op.inputs[0], a) &&
          read_tensor(graph, op.outputs[0], out) && is_int8_activation(a) &&
          is_int8_activation(out) && a.rank == 4 && out.rank == 4 &&
          a.shape[0] == 1 && out.shape[0] == 1 && a.shape[1] == out.shape[1] &&
          out.shape[2] == 1 && out.shape[3] == 1 &&
          static_cast<int64_t>(a.shape[2]) * a.shape[3] * 255 < INT32_MAX &&
          valid_requantization(read_i32(p), read_i32(p + 4));
    case OpCode::FLATTEN:
      return op.num_inputs == 1 && op.num_outputs == 1 &&
          op.parameter_size == 0 && read_tensor(graph, op.inputs[0], a) &&
          read_tensor(graph, op.outputs[0], out) && is_int8_activation(a) &&
          is_int8_activation(out) && same_quantization(a, out) &&
          out.rank == 2 && out.shape[0] == 1 &&
          (a.rank != 4 || a.shape[0] == 1) &&
          tensor_element_count(a) == tensor_element_count(out);
    case OpCode::MAX_POOL2D:
    case OpCode::AVG_POOL2D: {
      if (op.num_inputs != 1 || op.num_outputs != 1 ||
          op.parameter_size != INT8_POOL_PARAMS_SIZE ||
          !read_tensor(graph, op.inputs[0], a) ||
          !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(a) ||
          !is_int8_activation(out) || !same_quantization(a, out) ||
          a.rank != 4 || out.rank != 4 || a.shape[0] != 1 ||
          out.shape[0] != 1 || a.shape[1] != out.shape[1] ||
          a.shape[1] > 65535 || a.shape[2] > 65535 || a.shape[3] > 65535)
        return false;
      const int32_t kh = p[0];
      const int32_t kw = p[1];
      const int32_t sh = p[2];
      const int32_t sw = p[3];
      const int32_t ph = p[4];
      const int32_t pw = p[5];
      if (kh == 0 || kw == 0 || sh == 0 || sw == 0 || p[6] != 1 || p[7] != 1 ||
          2 * ph > kh || 2 * pw > kw ||
          !valid_activation_range(read_i32(p + 8), read_i32(p + 12)))
        return false;
      return out.shape[2] == (a.shape[2] + 2 * ph - kh) / sh + 1 &&
          out.shape[3] == (a.shape[3] + 2 * pw - kw) / sw + 1;
    }
    case OpCode::LUT:
      return op.num_inputs == 1 && op.num_outputs == 1 &&
          op.parameter_size == INT8_LUT_PARAMS_SIZE &&
          read_tensor(graph, op.inputs[0], a) &&
          read_tensor(graph, op.outputs[0], out) && is_int8_activation(a) &&
          is_int8_activation(out) && same_shape(a, out);
    case OpCode::MUL: {
      if (op.num_inputs != 2 || op.num_outputs != 1 ||
          op.parameter_size != INT8_MUL_PARAMS_SIZE ||
          !read_tensor(graph, op.inputs[0], a) ||
          !read_tensor(graph, op.inputs[1], b) ||
          !read_tensor(graph, op.outputs[0], out) || !is_int8_value(a) ||
          !is_int8_value(b) || !is_int8_activation(out) ||
          !same_shape(a, out) ||
          !valid_requantization(read_i32(p), read_i32(p + 4)) ||
          !valid_activation_range(read_i32(p + 8), read_i32(p + 12)))
        return false;
      const int32_t mode = read_i32(p + 16);
      if (mode == MUL_MODE_ELEMENTWISE)
        return same_shape(a, b);
      return mode == MUL_MODE_CHANNEL_BROADCAST && a.rank == 4 && b.rank == 4 &&
          a.shape[0] == 1 && b.shape[0] == 1 && b.shape[1] == a.shape[1] &&
          b.shape[2] == 1 && b.shape[3] == 1;
    }
    case OpCode::SOFTMAX: {
      if (op.num_inputs != 1 || op.num_outputs != 1 ||
          op.parameter_size != INT8_SOFTMAX_PARAMS_SIZE ||
          !read_tensor(graph, op.inputs[0], a) ||
          !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(a) ||
          !is_int8_activation(out) || !same_shape(a, out) ||
          (a.rank != 2 && a.rank != 4) || a.shape[0] != 1 ||
          out.quant_zero_point != SOFTMAX_OUTPUT_ZERO_POINT ||
          std::fabs(out.quant_scale - 1.0f / 256.0f) > 1e-9f)
        return false;
      uint32_t previous = read_u32(p);
      if (previous == 0 || previous > (1U << 24))
        return false;
      for (size_t k = 1; k < 256; ++k) {
        const uint32_t value = read_u32(p + 4 * k);
        if (value > previous)
          return false;
        previous = value;
      }
      return true;
    }
    case OpCode::CONCAT: {
      if (op.num_inputs < 2 || op.num_outputs != 1 || op.parameter_size != 0 ||
          !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(out) ||
          (out.rank != 2 && out.rank != 4) || out.shape[0] != 1)
        return false;
      int64_t channels = 0;
      for (uint8_t j = 0; j < op.num_inputs; ++j) {
        if (!read_tensor(graph, op.inputs[j], a) || !is_int8_activation(a) ||
            !same_quantization(a, out) || a.rank != out.rank || a.shape[0] != 1)
          return false;
        for (uint8_t d = 2; d < out.rank; ++d) {
          if (a.shape[d] != out.shape[d])
            return false;
        }
        channels += a.shape[1];
      }
      return channels == out.shape[1];
    }
    case OpCode::LINEAR: {
      TensorRecord bias{};
      if (op.num_inputs != 3 || op.num_outputs != 1 ||
          !read_tensor(graph, op.inputs[0], a) ||
          !read_tensor(graph, op.inputs[1], b) ||
          !read_tensor(graph, op.inputs[2], bias) ||
          !read_tensor(graph, op.outputs[0], out) || !is_int8_activation(a) ||
          !is_int8_activation(out) || !is_constant_of(b, DataType::INT8) ||
          !is_constant_of(bias, DataType::INT32) || b.rank != 2 ||
          bias.rank != 1 || bias.shape[0] != b.shape[0] || b.shape[0] > 65535 ||
          b.shape[1] > 65535 || a.rank == 4 || out.rank == 4 ||
          tensor_element_count(a) % static_cast<size_t>(b.shape[1]) != 0 ||
          tensor_element_count(out) !=
              tensor_element_count(a) / static_cast<size_t>(b.shape[1]) *
                  static_cast<size_t>(b.shape[0]) ||
          op.parameter_size !=
              INT8_PAIR_PARAMS_SIZE + static_cast<size_t>(b.shape[0]) * 8U ||
          !valid_activation_range(read_i32(p), read_i32(p + 4)))
        return false;
      const uint8_t* multipliers = p + INT8_PAIR_PARAMS_SIZE;
      const uint8_t* shifts =
          multipliers + static_cast<size_t>(b.shape[0]) * 4U;
      for (int32_t i = 0; i < b.shape[0]; ++i) {
        if (!valid_requantization(
                read_i32(multipliers + i * 4), read_i32(shifts + i * 4)))
          return false;
      }
      return true;
    }
    case OpCode::COPY:
      return validate_copy(graph, op);
    case OpCode::UPSAMPLE:
      return validate_upsample(graph, op);
    case OpCode::TRANSPOSE_CONV2D:
      return validate_transposed_conv(graph, op);
  }
  return false;
}

bool validate_int8_dataflow(const GraphView& graph) {
  bool produced[MAX_GRAPH_TENSORS]{};
  bool copy_produced[MAX_GRAPH_TENSORS]{};
  uint32_t input_count = 0;
  uint32_t output_count = 0;
  for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
    TensorRecord tensor{};
    read_tensor(graph, i, tensor);
    if ((tensor.flags & TENSOR_FLAG_GRAPH_INPUT) != 0) {
      produced[i] = true;
      ++input_count;
    }
    if ((tensor.flags & TENSOR_FLAG_GRAPH_OUTPUT) != 0)
      ++output_count;
  }
  if (graph.header.version == GRAPH_ESP_NN_INT8_IO_VERSION) {
    bool listed_input[MAX_GRAPH_TENSORS]{};
    bool listed_output[MAX_GRAPH_TENSORS]{};
    for (uint32_t i = 0; i < graph.num_inputs; ++i) {
      const uint32_t id = graph_input_id(graph, i);
      TensorRecord tensor{};
      if (!read_tensor(graph, id, tensor) ||
          (tensor.flags & TENSOR_FLAG_GRAPH_INPUT) == 0) {
        std::printf(
            "ERROR: graph IO table lists input %u without the input flag\n",
            static_cast<unsigned>(id));
        return false;
      }
      listed_input[id] = true;
    }
    for (uint32_t i = 0; i < graph.num_outputs; ++i) {
      const uint32_t id = graph_output_id(graph, i);
      TensorRecord tensor{};
      if (!read_tensor(graph, id, tensor) ||
          (tensor.flags & TENSOR_FLAG_GRAPH_OUTPUT) == 0) {
        std::printf(
            "ERROR: graph IO table lists output %u without the output flag\n",
            static_cast<unsigned>(id));
        return false;
      }
      listed_output[id] = true;
    }
    for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
      TensorRecord tensor{};
      read_tensor(graph, i, tensor);
      if (((tensor.flags & TENSOR_FLAG_GRAPH_INPUT) != 0 && !listed_input[i]) ||
          ((tensor.flags & TENSOR_FLAG_GRAPH_OUTPUT) != 0 &&
           !listed_output[i])) {
        std::printf(
            "ERROR: tensor %u is flagged as graph IO but missing from the IO table\n",
            static_cast<unsigned>(i));
        return false;
      }
    }
  }
  if (input_count == 0 || output_count == 0) {
    std::printf(
        "ERROR: int8 graph must declare at least one input and one output\n");
    return false;
  }
  for (uint32_t i = 0; i < graph.header.num_operations; ++i) {
    OperationRecord op{};
    read_operation(graph, i, op);
    for (uint8_t j = 0; j < op.num_inputs; ++j) {
      TensorRecord tensor{};
      read_tensor(graph, op.inputs[j], tensor);
      if ((tensor.flags & TENSOR_FLAG_CONSTANT) == 0 &&
          !produced[op.inputs[j]]) {
        std::printf(
            "ERROR: operation %u reads tensor %u before it is produced\n",
            static_cast<unsigned>(i),
            static_cast<unsigned>(op.inputs[j]));
        return false;
      }
    }
    const bool is_copy = static_cast<OpCode>(op.opcode) == OpCode::COPY;
    for (uint8_t j = 0; j < op.num_outputs; ++j) {
      const uint32_t id = op.outputs[j];
      if (produced[id] && !(is_copy && copy_produced[id])) {
        std::printf(
            "ERROR: tensor %u has more than one producer\n",
            static_cast<unsigned>(id));
        return false;
      }
      produced[id] = true;
      copy_produced[id] = is_copy;
    }
  }
  for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
    TensorRecord tensor{};
    read_tensor(graph, i, tensor);
    if ((tensor.flags & TENSOR_FLAG_GRAPH_OUTPUT) != 0 && !produced[i]) {
      std::printf(
          "ERROR: graph output tensor %u is never produced\n",
          static_cast<unsigned>(i));
      return false;
    }
  }
  return true;
}

} // namespace

uint32_t read_le_u32(const uint8_t* p) {
  return read_u32(p);
}

int32_t read_le_i32(const uint8_t* p) {
  return read_i32(p);
}

bool is_int8_graph_version(uint32_t version) {
  return version == GRAPH_ESP_NN_INT8_VERSION ||
      version == GRAPH_ESP_NN_INT8_IO_VERSION;
}

namespace {

uint32_t flagged_tensor(const GraphView& graph, uint8_t flag, uint32_t index) {
  uint32_t seen = 0;
  for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
    TensorRecord tensor{};
    read_tensor(graph, i, tensor);
    if ((tensor.flags & flag) == 0)
      continue;
    if (seen == index)
      return i;
    ++seen;
  }
  return UINT32_MAX;
}

uint32_t flagged_count(const GraphView& graph, uint8_t flag) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
    TensorRecord tensor{};
    read_tensor(graph, i, tensor);
    if ((tensor.flags & flag) != 0)
      ++count;
  }
  return count;
}

} // namespace

uint32_t graph_input_count(const GraphView& graph) {
  if (graph.io_ids != nullptr)
    return graph.num_inputs;
  return flagged_count(graph, TENSOR_FLAG_GRAPH_INPUT);
}

uint32_t graph_output_count(const GraphView& graph) {
  if (graph.io_ids != nullptr)
    return graph.num_outputs;
  return flagged_count(graph, TENSOR_FLAG_GRAPH_OUTPUT);
}

uint32_t graph_input_id(const GraphView& graph, uint32_t index) {
  if (graph.io_ids != nullptr) {
    return index < graph.num_inputs ? read_u32(graph.io_ids + 4 * index)
                                    : UINT32_MAX;
  }
  return flagged_tensor(graph, TENSOR_FLAG_GRAPH_INPUT, index);
}

uint32_t graph_output_id(const GraphView& graph, uint32_t index) {
  if (graph.io_ids != nullptr) {
    return index < graph.num_outputs
        ? read_u32(graph.io_ids + 4 * (graph.num_inputs + index))
        : UINT32_MAX;
  }
  return flagged_tensor(graph, TENSOR_FLAG_GRAPH_OUTPUT, index);
}

bool read_int8_conv_params(
    const GraphView& graph,
    const OperationRecord& op,
    uint32_t channels,
    Int8ConvParams& params) {
  const size_t expected =
      INT8_CONV_BASE_PARAMS_SIZE + static_cast<size_t>(channels) * 8U;
  if (op.parameter_size != expected)
    return false;
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  params.kernel_h = p[0];
  params.kernel_w = p[1];
  params.stride_h = p[2];
  params.stride_w = p[3];
  params.pad_h = p[4];
  params.pad_w = p[5];
  params.dilation_h = p[6];
  params.dilation_w = p[7];
  params.groups = read_i32(p + 8);
  params.activation_min = read_i32(p + 12);
  params.activation_max = read_i32(p + 16);
  params.multipliers = p + INT8_CONV_BASE_PARAMS_SIZE;
  params.shifts = params.multipliers + static_cast<size_t>(channels) * 4U;
  return true;
}

bool read_int8_add_params(
    const GraphView& graph,
    const OperationRecord& op,
    Int8AddParams& params) {
  if (op.parameter_size != INT8_ADD_PARAMS_SIZE)
    return false;
  const uint8_t* p = graph.parameter_data + op.parameter_offset;
  params.input1_mult = read_i32(p);
  params.input1_shift = read_i32(p + 4);
  params.input2_mult = read_i32(p + 8);
  params.input2_shift = read_i32(p + 12);
  params.left_shift = read_i32(p + 16);
  params.output_mult = read_i32(p + 20);
  params.output_shift = read_i32(p + 24);
  params.activation_min = read_i32(p + 28);
  params.activation_max = read_i32(p + 32);
  return true;
}

size_t tensor_element_count(const TensorRecord& tensor) {
  size_t count = 1;
  for (uint8_t i = 0; i < tensor.rank; ++i) {
    if (tensor.shape[i] <= 0 ||
        static_cast<size_t>(tensor.shape[i]) > SIZE_MAX / count) {
      return 0;
    }
    count *= static_cast<size_t>(tensor.shape[i]);
  }
  return count;
}

size_t tensor_data_size(const TensorRecord& tensor) {
  const size_t count = tensor_element_count(tensor);
  size_t element_size = 0;
  switch (static_cast<DataType>(tensor.dtype)) {
    case DataType::FLOAT32:
    case DataType::INT32:
      element_size = 4;
      break;
    case DataType::INT8:
    case DataType::UINT8:
      element_size = 1;
      break;
    default:
      return 0;
  }
  if (count == 0 || count > SIZE_MAX / element_size) {
    return 0;
  }
  return count * element_size;
}

bool read_tensor(const GraphView& graph, uint32_t index, TensorRecord& tensor) {
  if (index >= graph.header.num_tensors)
    return false;
  const uint8_t* p = graph.data + graph.tensor_table_offset +
      static_cast<size_t>(index) * graph.tensor_record_size;
  tensor.tensor_id = read_u32(p);
  tensor.constant_offset = read_u32(p + 4);
  tensor.dtype = p[8];
  tensor.flags = p[9];
  tensor.rank = p[11] == 0 ? p[10] : 0xFF;
  for (size_t i = 0; i < 4; ++i)
    tensor.shape[i] = read_i32(p + 12 + 4 * i);
  const uint32_t scale_bits = read_u32(p + 28);
  std::memcpy(&tensor.quant_scale, &scale_bits, sizeof(scale_bits));
  tensor.quant_zero_point = read_i32(p + 32);
  return true;
}

bool read_operation(
    const GraphView& graph,
    uint32_t index,
    OperationRecord& operation) {
  if (index >= graph.header.num_operations)
    return false;
  const uint8_t* p = graph.data + graph.operation_table_offset +
      static_cast<size_t>(index) * OPERATION_RECORD_SIZE;
  operation.op_id = p[0];
  operation.opcode = p[1];
  operation.num_inputs = p[2];
  operation.num_outputs = p[3];
  for (size_t i = 0; i < 6; ++i)
    operation.inputs[i] = read_u32(p + 4 + 4 * i);
  for (size_t i = 0; i < 2; ++i)
    operation.outputs[i] = read_u32(p + 28 + 4 * i);
  operation.parameter_offset = read_u32(p + 36);
  operation.parameter_size = read_u32(p + 40);
  return true;
}

const uint8_t* tensor_constant_data(
    const GraphView& graph,
    const TensorRecord& tensor) {
  return graph.constant_data + tensor.constant_offset;
}

bool parse_graph(const uint8_t* data, size_t size, GraphView& graph) {
  graph = {};
  if (data == nullptr || size < GRAPH_HEADER_SIZE) {
    std::printf("ERROR: graph is null or shorter than the 22-byte header\n");
    return false;
  }
  if (std::memcmp(data, "XTS3", 4) != 0) {
    std::printf("ERROR: invalid graph magic\n");
    return false;
  }
  graph.header.magic[0] = 'X';
  graph.header.magic[1] = 'T';
  graph.header.magic[2] = 'S';
  graph.header.magic[3] = '3';
  graph.header.version = read_u16(data + 4);
  graph.header.num_tensors = read_u32(data + 6);
  graph.header.num_operations = read_u32(data + 10);
  graph.header.parameter_size = read_u32(data + 14);
  graph.header.constant_size = read_u32(data + 18);
  if (!is_int8_graph_version(graph.header.version) ||
      graph.header.num_tensors == 0 ||
      graph.header.num_tensors > MAX_GRAPH_TENSORS ||
      graph.header.num_operations == 0 ||
      graph.header.num_operations > MAX_GRAPH_OPERATIONS) {
    std::printf("ERROR: unsupported version or graph record counts\n");
    return false;
  }

  graph.data = data;
  graph.size = size;
  graph.tensor_table_offset = GRAPH_HEADER_SIZE;
  if (graph.header.version == GRAPH_ESP_NN_INT8_IO_VERSION) {
    if (!range_valid(GRAPH_HEADER_SIZE, GRAPH_IO_TABLE_HEADER_SIZE, size))
      return false;
    graph.num_inputs = read_u16(data + GRAPH_HEADER_SIZE);
    graph.num_outputs = read_u16(data + GRAPH_HEADER_SIZE + 2);
    const size_t io_bytes =
        4U * (static_cast<size_t>(graph.num_inputs) + graph.num_outputs);
    if (graph.num_inputs == 0 || graph.num_outputs == 0 ||
        graph.num_inputs > MAX_GRAPH_IO || graph.num_outputs > MAX_GRAPH_IO ||
        !range_valid(
            GRAPH_HEADER_SIZE + GRAPH_IO_TABLE_HEADER_SIZE, io_bytes, size)) {
      std::printf("ERROR: graph IO table is invalid\n");
      return false;
    }
    graph.io_ids = data + GRAPH_HEADER_SIZE + GRAPH_IO_TABLE_HEADER_SIZE;
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(graph.num_inputs) + graph.num_outputs;
         ++i) {
      if (read_u32(graph.io_ids + 4 * i) >= graph.header.num_tensors)
        return false;
    }
    graph.tensor_table_offset =
        GRAPH_HEADER_SIZE + GRAPH_IO_TABLE_HEADER_SIZE + io_bytes;
  }
  graph.tensor_record_size = TENSOR_RECORD_SIZE;
  const size_t tensor_bytes =
      static_cast<size_t>(graph.header.num_tensors) * graph.tensor_record_size;
  if (!range_valid(graph.tensor_table_offset, tensor_bytes, size))
    return false;
  graph.operation_table_offset = graph.tensor_table_offset + tensor_bytes;
  const size_t operation_bytes =
      static_cast<size_t>(graph.header.num_operations) * OPERATION_RECORD_SIZE;
  if (!range_valid(graph.operation_table_offset, operation_bytes, size))
    return false;
  graph.parameter_offset = graph.operation_table_offset + operation_bytes;
  if (!range_valid(graph.parameter_offset, graph.header.parameter_size, size))
    return false;
  graph.constant_offset = graph.parameter_offset + graph.header.parameter_size;
  if (!range_valid(graph.constant_offset, graph.header.constant_size, size) ||
      graph.constant_offset + graph.header.constant_size != size) {
    std::printf("ERROR: graph sections exceed blob or leave trailing bytes\n");
    return false;
  }
  if ((graph.constant_offset % INT8_PARAMETER_ALIGNMENT) != 0) {
    std::printf(
        "ERROR: int8 constant region is not 16-byte aligned within the blob\n");
    return false;
  }
  graph.parameter_data = data + graph.parameter_offset;
  graph.constant_data = data + graph.constant_offset;

  const unsigned allowed_flags = 0x0FU;
  for (uint32_t i = 0; i < graph.header.num_tensors; ++i) {
    TensorRecord tensor{};
    read_tensor(graph, i, tensor);
    if (tensor.tensor_id != i || !valid_dtype(tensor.dtype) ||
        tensor.rank > 4 || (tensor.flags & ~allowed_flags) != 0) {
      std::printf(
          "ERROR: tensor record %u has invalid ID, dtype, rank, or flags\n",
          static_cast<unsigned>(i));
      return false;
    }
    for (uint8_t d = 0; d < tensor.rank; ++d) {
      if (tensor.shape[d] <= 0)
        return false;
    }
    for (uint8_t d = tensor.rank; d < 4; ++d) {
      if (tensor.shape[d] != 0)
        return false;
    }
    const bool is_constant = (tensor.flags & TENSOR_FLAG_CONSTANT) != 0;
    const bool has_quantization = (tensor.flags & TENSOR_FLAG_QUANTIZED) != 0;
    const uint8_t quantized_dtype = static_cast<uint8_t>(DataType::INT8);
    if ((has_quantization &&
         (tensor.dtype != quantized_dtype ||
          !std::isfinite(tensor.quant_scale) || tensor.quant_scale <= 0.0f ||
          tensor.quant_zero_point < -128 || tensor.quant_zero_point > 127)) ||
        (!has_quantization &&
         (tensor.quant_scale != 0.0f || tensor.quant_zero_point != 0))) {
      std::printf(
          "ERROR: tensor %u quantization metadata is invalid\n",
          static_cast<unsigned>(i));
      return false;
    }
    const bool quantized_constant_allowed =
        graph.header.version == GRAPH_ESP_NN_INT8_IO_VERSION &&
        tensor.dtype == static_cast<uint8_t>(DataType::INT8);
    if ((is_constant
             ? ((tensor.dtype != static_cast<uint8_t>(DataType::INT8) &&
                 tensor.dtype != static_cast<uint8_t>(DataType::INT32)) ||
                (has_quantization && !quantized_constant_allowed) ||
                (tensor.flags &
                 (TENSOR_FLAG_GRAPH_INPUT | TENSOR_FLAG_GRAPH_OUTPUT)) != 0)
             : (tensor.dtype != static_cast<uint8_t>(DataType::INT8) ||
                !has_quantization))) {
      std::printf(
          "ERROR: tensor %u is not a supported native int8 tensor\n",
          static_cast<unsigned>(i));
      return false;
    }
    const size_t data_size = tensor_data_size(tensor);
    if (data_size == 0 ||
        (is_constant &&
         !range_valid(
             tensor.constant_offset, data_size, graph.header.constant_size)) ||
        (!is_constant && tensor.constant_offset != 0)) {
      std::printf(
          "ERROR: tensor %u constant data range or shape is invalid\n",
          static_cast<unsigned>(i));
      return false;
    }
  }

  size_t expected_parameter_offset = 0;
  for (uint32_t i = 0; i < graph.header.num_operations; ++i) {
    OperationRecord op{};
    read_operation(graph, i, op);
    if (op.op_id != (i & 0xFFU) ||
        !valid_opcode(op.opcode, graph.header.version) || op.num_inputs == 0 ||
        op.num_inputs > 6 || op.num_outputs == 0 || op.num_outputs > 2 ||
        op.parameter_offset != expected_parameter_offset ||
        !range_valid(
            op.parameter_offset,
            op.parameter_size,
            graph.header.parameter_size)) {
      std::printf(
          "ERROR: operation record %u is invalid\n", static_cast<unsigned>(i));
      return false;
    }
    for (uint8_t j = 0; j < op.num_inputs; ++j)
      if (!valid_tensor_reference(graph, op.inputs[j]))
        return false;
    for (uint8_t j = 0; j < op.num_outputs; ++j)
      if (!valid_tensor_reference(graph, op.outputs[j]))
        return false;
    for (uint8_t j = op.num_inputs; j < 6; ++j)
      if (op.inputs[j] != 0)
        return false;
    for (uint8_t j = op.num_outputs; j < 2; ++j)
      if (op.outputs[j] != 0)
        return false;
    if (!validate_int8_operation(graph, op)) {
      std::printf(
          "ERROR: operation %u (%s) metadata or parameter block is invalid\n",
          static_cast<unsigned>(i),
          opcode_name(static_cast<OpCode>(op.opcode)));
      return false;
    }
    expected_parameter_offset += op.parameter_size;
  }
  const size_t parameter_padding =
      graph.header.parameter_size - expected_parameter_offset;
  if (expected_parameter_offset > graph.header.parameter_size ||
      parameter_padding >= INT8_PARAMETER_ALIGNMENT) {
    std::printf(
        "ERROR: parameter region size does not match operation records\n");
    return false;
  }
  if (!validate_int8_dataflow(graph))
    return false;
  return true;
}

const char* opcode_name(OpCode opcode) {
  switch (opcode) {
    case OpCode::CONV2D:
      return "CONV2D";
    case OpCode::DEPTHWISE_CONV2D:
      return "DEPTHWISE_CONV2D";
    case OpCode::RELU6:
      return "RELU6";
    case OpCode::ADD:
      return "ADD";
    case OpCode::AVGPOOL2D:
      return "AVGPOOL2D";
    case OpCode::LINEAR:
      return "LINEAR";
    case OpCode::FLATTEN:
      return "FLATTEN";
    case OpCode::MAX_POOL2D:
      return "MAX_POOL2D";
    case OpCode::AVG_POOL2D:
      return "AVG_POOL2D";
    case OpCode::CONCAT:
      return "CONCAT";
    case OpCode::LUT:
      return "LUT";
    case OpCode::MUL:
      return "MUL";
    case OpCode::SOFTMAX:
      return "SOFTMAX";
    case OpCode::COPY:
      return "COPY";
    case OpCode::UPSAMPLE:
      return "UPSAMPLE";
    case OpCode::TRANSPOSE_CONV2D:
      return "TRANSPOSE_CONV2D";
    default:
      return "UNKNOWN";
  }
}

} // namespace espressif
} // namespace backends
} // namespace executorch
