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

/**
 * @file
 * Parser for XTS3, the int8 graph format produced by EspNNBackend.preprocess()
 * (backends/espressif/serialization/xts3.py). All multi-byte fields are
 * little-endian. A graph is a header, an optional input/output table
 * (version 6), fixed-size tensor and operation records, a parameter region
 * and a constant region. parse_graph() bounds-checks every region and record
 * before any accessor reads it.
 */

namespace executorch {
namespace backends {
namespace espressif {

/// Graph versions accepted by this runtime; 6 adds the input/output table.
static constexpr uint32_t GRAPH_ESP_NN_INT8_VERSION = 5;
static constexpr uint32_t GRAPH_ESP_NN_INT8_IO_VERSION = 6;
static constexpr size_t GRAPH_HEADER_SIZE = 22;
static constexpr size_t GRAPH_IO_TABLE_HEADER_SIZE = 4;
static constexpr size_t MAX_GRAPH_IO = 16;
static constexpr size_t TENSOR_RECORD_SIZE = 36;
static constexpr size_t OPERATION_RECORD_SIZE = 44;
static constexpr size_t MAX_GRAPH_TENSORS = 1024;
static constexpr size_t MAX_GRAPH_OPERATIONS = 1024;

/// TensorRecord::flags bits.
static constexpr uint8_t TENSOR_FLAG_CONSTANT = 1U << 0;
static constexpr uint8_t TENSOR_FLAG_QUANTIZED = 1U << 1;
static constexpr uint8_t TENSOR_FLAG_GRAPH_INPUT = 1U << 2;
static constexpr uint8_t TENSOR_FLAG_GRAPH_OUTPUT = 1U << 3;

/// Serialized parameter block sizes and modes, mirrored in xts3.py.
static constexpr size_t INT8_CONV_BASE_PARAMS_SIZE = 20;
static constexpr size_t INT8_ADD_PARAMS_SIZE = 36;
static constexpr size_t INT8_PAIR_PARAMS_SIZE = 8;
static constexpr size_t INT8_POOL_PARAMS_SIZE = 16;
static constexpr size_t INT8_LUT_PARAMS_SIZE = 256;
static constexpr size_t INT8_MUL_PARAMS_SIZE = 20;
static constexpr size_t INT8_SOFTMAX_PARAMS_SIZE = 1024;
static constexpr int32_t MUL_MODE_ELEMENTWISE = 0;
static constexpr int32_t MUL_MODE_CHANNEL_BROADCAST = 1;
static constexpr int32_t SOFTMAX_OUTPUT_ZERO_POINT = -128;
static constexpr size_t INT8_PARAMETER_ALIGNMENT = 16;
static constexpr size_t COPY_HEADER_SIZE = 12;
static constexpr size_t COPY_MAX_RANK = 6;
static constexpr size_t COPY_SEGMENT_SIZE = 108;
static constexpr int32_t COPY_FLAG_FILL = 1;
static constexpr size_t UPSAMPLE_HEADER_SIZE = 20;
static constexpr int32_t UPSAMPLE_MODE_NEAREST = 0;
static constexpr int32_t UPSAMPLE_MODE_BILINEAR = 1;
static constexpr int32_t UPSAMPLE_WEIGHT_BITS = 10;
/// Largest output height or width an UPSAMPLE operation may declare.
static constexpr int32_t UPSAMPLE_MAX_EXTENT = 65535;
static constexpr size_t TRANSPOSE_CONV_PARAMS_SIZE = 52;

/// Operation types; values are part of the file format.
enum class OpCode : uint8_t {
  CONV2D = 0,
  DEPTHWISE_CONV2D = 1,
  RELU6 = 2,
  ADD = 3,
  AVGPOOL2D = 4,
  LINEAR = 5,
  FLATTEN = 6,
  MAX_POOL2D = 7,
  AVG_POOL2D = 8,
  CONCAT = 9,
  LUT = 10,
  MUL = 11,
  SOFTMAX = 12,
  COPY = 13,
  UPSAMPLE = 14,
  TRANSPOSE_CONV2D = 15,
};

static constexpr uint8_t OPCODE_COUNT = 16;

/// Tensor element types; delegated graphs only use INT8 and INT32.
enum class DataType : uint8_t {
  UNKNOWN = 0,
  FLOAT32 = 1,
  INT8 = 2,
  UINT8 = 3,
  INT32 = 4,
};

/// Fixed graph header.
struct GraphHeader {
  char magic[4];
  uint16_t version;
  uint32_t num_tensors;
  uint32_t num_operations;
  uint32_t parameter_size;
  uint32_t constant_size;
};

/// One tensor; shape is NCHW for rank 4, stored on the device as NHWC.
struct TensorRecord {
  uint32_t tensor_id;
  uint32_t constant_offset;
  uint8_t dtype;
  uint8_t flags;
  uint8_t rank;
  int32_t shape[4];
  float quant_scale;
  int32_t quant_zero_point;
};

/// One operation with its tensor IDs and parameter block.
struct OperationRecord {
  uint8_t op_id;
  uint8_t opcode;
  uint8_t num_inputs;
  uint8_t num_outputs;
  uint32_t inputs[6];
  uint32_t outputs[2];
  uint32_t parameter_offset;
  uint32_t parameter_size;
};

/// Validated view of a graph; points into the caller's buffer.
struct GraphView {
  const uint8_t* data;
  size_t size;
  size_t tensor_record_size;
  GraphHeader header;
  size_t tensor_table_offset;
  size_t operation_table_offset;
  size_t parameter_offset;
  size_t constant_offset;
  const uint8_t* parameter_data;
  const uint8_t* constant_data;
  uint16_t num_inputs;
  uint16_t num_outputs;
  const uint8_t* io_ids;
};

/// Decoded convolution parameters; multipliers and shifts are per channel.
struct Int8ConvParams {
  int32_t kernel_h;
  int32_t kernel_w;
  int32_t stride_h;
  int32_t stride_w;
  int32_t pad_h;
  int32_t pad_w;
  int32_t groups;
  int32_t activation_min;
  int32_t activation_max;
  const uint8_t* multipliers;
  const uint8_t* shifts;
  int32_t dilation_h;
  int32_t dilation_w;
};

/// Decoded requantization parameters of an elementwise add.
struct Int8AddParams {
  int32_t input1_mult;
  int32_t input1_shift;
  int32_t input2_mult;
  int32_t input2_shift;
  int32_t left_shift;
  int32_t output_mult;
  int32_t output_shift;
  int32_t activation_min;
  int32_t activation_max;
};

/// Reads a little-endian 32-bit value from unaligned memory.
uint32_t read_le_u32(const uint8_t* p);
/// Reads a little-endian 32-bit signed value from unaligned memory.
int32_t read_le_i32(const uint8_t* p);

/**
 * Validates a serialized graph and fills @p graph.
 *
 * Checks the magic, version, region bounds, every tensor and operation
 * record, and each operation's parameter block against its tensor shapes.
 *
 * @param[in] data Serialized graph; must outlive @p graph.
 * @param[in] size Size of @p data in bytes.
 * @param[out] graph View used by the other functions.
 * @retval false The graph is malformed; the reason is printed.
 */
bool parse_graph(const uint8_t* data, size_t size, GraphView& graph);
/// Whether @p version is an int8 graph version this runtime can execute.
bool is_int8_graph_version(uint32_t version);
/// Number of graph inputs.
uint32_t graph_input_count(const GraphView& graph);
/// Number of graph outputs.
uint32_t graph_output_count(const GraphView& graph);
/// Tensor ID of input @p index, or UINT32_MAX if out of range.
uint32_t graph_input_id(const GraphView& graph, uint32_t index);
/// Tensor ID of output @p index, or UINT32_MAX if out of range.
uint32_t graph_output_id(const GraphView& graph, uint32_t index);
/// Decodes tensor record @p index. @retval false Out of range.
bool read_tensor(const GraphView& graph, uint32_t index, TensorRecord& tensor);
/// Decodes operation record @p index. @retval false Out of range.
bool read_operation(
    const GraphView& graph,
    uint32_t index,
    OperationRecord& operation);
/// Decodes a convolution's parameter block for @p channels output channels.
bool read_int8_conv_params(
    const GraphView& graph,
    const OperationRecord& op,
    uint32_t channels,
    Int8ConvParams& params);
/// Decodes an elementwise add's parameter block.
bool read_int8_add_params(
    const GraphView& graph,
    const OperationRecord& op,
    Int8AddParams& params);
/// Number of elements described by the tensor's shape.
size_t tensor_element_count(const TensorRecord& tensor);
/// Size of the tensor's data in bytes.
size_t tensor_data_size(const TensorRecord& tensor);
/// Constant data of @p tensor; only meaningful for TENSOR_FLAG_CONSTANT
/// tensors, whose offsets parse_graph() has checked.
const uint8_t* tensor_constant_data(
    const GraphView& graph,
    const TensorRecord& tensor);
/// Printable operation name.
const char* opcode_name(OpCode opcode);

} // namespace espressif
} // namespace backends
} // namespace executorch
