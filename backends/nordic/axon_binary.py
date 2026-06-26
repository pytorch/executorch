# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON intermediate binary file builder.

Creates the binary file that Nordic's AXON compiler lib reads as input.
Uses cffi to create C structs from nrf_axon_nn_compiler_types.h,
guaranteeing exact binary compatibility.

Binary format::

    [header struct: nrf_axon_nn_model_desc_hdr_s]  (offsets to sections)
    [model name string]
    [meta info: nrf_axon_nn_model_meta_info_s]
    [layer descs: nrf_axon_nn_model_layer_desc_s[]]
    [constants: weights, biases, multipliers, shifts]
    [compilation options: nrf_axon_nn_model_compilation_options_s]
    [title string: "AXON_INTERMEDIATE_REPRESENTATION_FILE"]
    [version: uint32]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from cffi import FFI

from .axon_types import AxonLayer

logger = logging.getLogger(__name__)

# Binary format constants
BINARY_TITLE = "AXON_INTERMEDIATE_REPRESENTATION_FILE"
VERSION_MAJOR = 0
VERSION_MINOR = 17
VERSION_PATCH = 0
MODEL_BIN_VERSION = (VERSION_MAJOR << 16) + (VERSION_MINOR << 8) + VERSION_PATCH


def _create_ffi() -> FFI:
    """Create cffi FFI with AXON compiler structs defined manually.

    We define the structs explicitly rather than parsing the header file,
    because cffi can't handle all the preprocessor macros and enum patterns
    in nrf_axon_nn_compiler_types.h.
    """
    ffi = FFI()
    ffi.cdef("""
        typedef struct {
            uint16_t height;
            uint16_t width;
            uint16_t channel_cnt;
            int32_t byte_width;  /* nrf_axon_nn_byte_width_e: 1=INT8, 2=INT16, 4=INT32 */
        } nrf_axon_nn_compiler_model_layer_dimensions_s;

        typedef struct {
            uint32_t begin[3];
            uint32_t end[3];
            uint32_t strides[3];
        } nrf_axon_nn_compiler_strided_slice_parameters_s;

        typedef struct {
            uint8_t input_id_cnt;
            int16_t input_ids[4];
            int32_t nn_operation;  /* nrf_axon_nn_op_e */
            nrf_axon_nn_compiler_model_layer_dimensions_s input_dimensions[4];
            nrf_axon_nn_compiler_model_layer_dimensions_s filter_dimensions;
            nrf_axon_nn_compiler_model_layer_dimensions_s output_dimensions;
            uint8_t concatenate_axis;
            uint8_t stride_x;
            uint8_t stride_y;
            uint8_t dilation_x;
            uint8_t dilation_y;
            int8_t input_zero_point;
            int8_t output_zero_point;
            uint64_t bias_prime;         /* offset into consts */
            uint64_t output_multipliers; /* offset into consts */
            uint64_t scale_shifts;       /* offset into consts */
            uint16_t scale_shift_cnt;
            int32_t activation_function; /* nrf_axon_nn_activation_function_e */
            uint8_t pad_left;
            uint8_t pad_right;
            uint8_t pad_top;
            uint8_t pad_bottom;
            uint64_t filter;             /* offset into consts */
            uint32_t cpu_op_additional_attributes_count;
            uint64_t cpu_op_additional_attributes; /* offset into consts */
        } nrf_axon_nn_model_layer_desc_s;

        typedef struct {
            uint32_t offset;
            uint32_t length;
        } nrf_axon_nn_model_bin_item_s;

        typedef struct {
            uint32_t mult;
            uint8_t round;
            int8_t zero_point;
        } nrf_axon_nn_model_quant_paramters_s;

        typedef struct {
            nrf_axon_nn_model_bin_item_s model_name;
            nrf_axon_nn_model_bin_item_s model_labels;
            uint32_t model_layer_cnt;
            nrf_axon_nn_model_quant_paramters_s input_quant;
            nrf_axon_nn_model_quant_paramters_s output_dequant;
        } nrf_axon_nn_model_meta_info_s;

        typedef struct {
            nrf_axon_nn_model_bin_item_s title;
            nrf_axon_nn_model_bin_item_s version;
            nrf_axon_nn_model_bin_item_s meta;
            nrf_axon_nn_model_bin_item_s layers;
            nrf_axon_nn_model_bin_item_s consts;
            nrf_axon_nn_model_bin_item_s compilation_option;
        } nrf_axon_nn_model_desc_hdr_s;

        typedef struct {
            uint32_t interlayer_buffer_size;
            uint32_t psum_buffer_size;
            uint32_t header_file_test_vector_cnt;
            int32_t convolution_2d_setting;
            int32_t log_level;
            int32_t psum_buffer_placement;
        } nrf_axon_nn_model_compilation_options_s;
    """)
    return ffi


class AxonBinaryBuilder:
    """Builds the AXON intermediate binary file using cffi structs."""

    def __init__(self, compiler_types_hdr_path: str | None = None):
        self.ffi = _create_ffi()
        self._data = bytearray()
        self._header_size = self.ffi.sizeof("nrf_axon_nn_model_desc_hdr_s")

    def build(
        self,
        layers: list[AxonLayer],
        model_name: str = "model",
        interlayer_buffer_size: int = 125000,
        psum_buffer_size: int = 4096,
        input_quant_mult: int = 1,
        input_quant_round: int = 0,
        input_quant_zp: int = 0,
        output_quant_mult: int = 1,
        output_quant_round: int = 0,
        output_quant_zp: int = 0,
    ) -> bytes:
        """Build the complete intermediate binary.

        Returns:
            bytes: The binary file contents ready for the compiler lib.
        """
        self._data = bytearray()
        header = self.ffi.new("nrf_axon_nn_model_desc_hdr_s *")

        # 1. Model name string
        model_name_bytes = model_name.lower().encode("utf-8") + b"\x00"
        model_name_item = self._append_data(model_name_bytes)

        # 2. Meta info
        meta = self.ffi.new("nrf_axon_nn_model_meta_info_s *")
        meta.model_name.offset = model_name_item[0]
        meta.model_name.length = model_name_item[1]
        meta.model_labels.offset = 0
        meta.model_labels.length = 0
        meta.model_layer_cnt = len(layers)
        meta.input_quant.mult = input_quant_mult
        meta.input_quant.round = input_quant_round
        meta.input_quant.zero_point = input_quant_zp
        meta.output_dequant.mult = output_quant_mult
        meta.output_dequant.round = output_quant_round
        meta.output_dequant.zero_point = output_quant_zp
        meta_item = self._append_struct(meta, "nrf_axon_nn_model_meta_info_s")

        # 3. Layer descriptors + constants
        consts_data = bytearray()
        layer_structs_data = bytearray()

        for i, layer in enumerate(layers):
            layer_struct = self.ffi.new("nrf_axon_nn_model_layer_desc_s *")

            # Input IDs
            layer_struct.input_id_cnt = len(layer.input_ids)
            for j in range(min(len(layer.input_ids), 4)):
                layer_struct.input_ids[j] = layer.input_ids[j]

            # Operation
            layer_struct.nn_operation = layer.operation

            # Input dimensions
            for j in range(min(len(layer.input_dimensions), 4)):
                d = layer.input_dimensions[j]
                layer_struct.input_dimensions[j].height = d.height
                layer_struct.input_dimensions[j].width = d.width
                layer_struct.input_dimensions[j].channel_cnt = d.channel_cnt
                layer_struct.input_dimensions[j].byte_width = d.byte_width

            # Filter dimensions
            layer_struct.filter_dimensions.height = layer.filter_dimensions.height
            layer_struct.filter_dimensions.width = layer.filter_dimensions.width
            layer_struct.filter_dimensions.channel_cnt = layer.filter_dimensions.channel_cnt
            layer_struct.filter_dimensions.byte_width = layer.filter_dimensions.byte_width

            # Output dimensions
            layer_struct.output_dimensions.height = layer.output_dimensions.height
            layer_struct.output_dimensions.width = layer.output_dimensions.width
            layer_struct.output_dimensions.channel_cnt = layer.output_dimensions.channel_cnt
            layer_struct.output_dimensions.byte_width = layer.output_dimensions.byte_width

            # Stride, dilation
            layer_struct.stride_x = layer.stride_x
            layer_struct.stride_y = layer.stride_y
            layer_struct.dilation_x = layer.dilation_x
            layer_struct.dilation_y = layer.dilation_y

            # Zero points
            layer_struct.input_zero_point = layer.input_zero_point
            layer_struct.output_zero_point = layer.output_zero_point
            logger.debug(f"  Binary layer {i}: in_zp={layer.input_zero_point} out_zp={layer.output_zero_point}")

            # Activation
            layer_struct.activation_function = layer.activation

            # Padding
            layer_struct.pad_left = layer.pad_left
            layer_struct.pad_right = layer.pad_right
            layer_struct.pad_top = layer.pad_top
            layer_struct.pad_bottom = layer.pad_bottom

            # Constants — store as offsets into the consts section.
            # Unused offsets must be 0xFFFFFFFFFFFFFFFF (sentinel),
            # not 0 (which points to filter data and corrupts compilation).
            if layer.filter_data:
                layer_struct.filter = len(consts_data)
                consts_data.extend(layer.filter_data)
                self._pad_to_4(consts_data)
            else:
                layer_struct.filter = 0xFFFFFFFFFFFFFFFF

            if layer.bias_data:
                layer_struct.bias_prime = len(consts_data)
                consts_data.extend(layer.bias_data)
                self._pad_to_4(consts_data)
            else:
                layer_struct.bias_prime = 0xFFFFFFFFFFFFFFFF

            from .axon_types import AxonOp

            # Determine number of output channels
            n_out_ch = layer.output_dimensions.channel_cnt
            if n_out_ch <= 1:
                n_out_ch = max(n_out_ch, layer.output_dimensions.width)
                n_out_ch = max(n_out_ch, layer.filter_dimensions.height)

            needs_per_ch_mult = layer.operation in (
                AxonOp.FULLY_CONNECTED, AxonOp.CONV2D,
                AxonOp.DEPTHWISE_CONV2D, AxonOp.POINTWISE_CONV2D,
            )

            if layer.multiplier_data:
                layer_struct.output_multipliers = len(consts_data)
                mult_arr = np.frombuffer(layer.multiplier_data, dtype=np.int32)
                if needs_per_ch_mult and len(mult_arr) < n_out_ch:
                    mult_arr = np.tile(mult_arr, (n_out_ch + len(mult_arr) - 1) // len(mult_arr))[:n_out_ch]
                consts_data.extend(mult_arr.tobytes())
                self._pad_to_4(consts_data)

            if layer.shift_data:
                layer_struct.scale_shifts = len(consts_data)
                consts_data.extend(layer.shift_data)
                self._pad_to_4(consts_data)

            layer_struct.scale_shift_cnt = layer.scale_shift_cnt

            layer_struct.cpu_op_additional_attributes_count = 0
            layer_struct.cpu_op_additional_attributes = 0xFFFFFFFFFFFFFFFF

            # Serialize layer struct
            layer_bin = bytes(self.ffi.buffer(layer_struct))
            layer_structs_data.extend(layer_bin)

        # Append layers section
        layers_item = self._append_data(bytes(layer_structs_data))

        # Append constants section
        consts_item = self._append_data(bytes(consts_data))

        # 4. Compilation options
        options = self.ffi.new("nrf_axon_nn_model_compilation_options_s *")
        options.interlayer_buffer_size = interlayer_buffer_size
        options.psum_buffer_size = psum_buffer_size
        options.header_file_test_vector_cnt = 0
        options.convolution_2d_setting = 0
        options.log_level = 0
        options.psum_buffer_placement = 0  # INTERLAYER_BUFFER (Nordic's default)
        options_item = self._append_struct(options, "nrf_axon_nn_model_compilation_options_s")

        # 5. Title string
        title_bytes = BINARY_TITLE.encode("utf-8") + b"\x00"
        title_item = self._append_data(title_bytes)

        # 6. Version (uint32)
        version_bytes = np.array([MODEL_BIN_VERSION], dtype=np.uint32).tobytes()
        version_item = self._append_data(version_bytes)

        # Fill header with offsets
        header.title.offset = title_item[0]
        header.title.length = title_item[1]
        header.version.offset = version_item[0]
        header.version.length = version_item[1]
        header.meta.offset = meta_item[0]
        header.meta.length = meta_item[1]
        header.layers.offset = layers_item[0]
        header.layers.length = layers_item[1]
        header.consts.offset = consts_item[0]
        header.consts.length = consts_item[1]
        header.compilation_option.offset = options_item[0]
        header.compilation_option.length = options_item[1]

        # Assemble: header + data
        header_bin = bytes(self.ffi.buffer(header))
        return header_bin + bytes(self._data)

    def _append_data(self, data: bytes) -> tuple[int, int]:
        """Append data to the binary, return (offset, length)."""
        offset = self._header_size + len(self._data)
        length = len(data)
        self._data.extend(data)
        self._pad_to_4(self._data)
        return (offset, length)

    def _append_struct(self, struct_ptr, struct_type: str) -> tuple[int, int]:
        """Serialize a cffi struct and append to binary."""
        struct_bytes = bytes(self.ffi.buffer(struct_ptr))
        return self._append_data(struct_bytes)

    @staticmethod
    def _pad_to_4(buf: bytearray):
        """Pad bytearray to 4-byte alignment."""
        while len(buf) % 4 != 0:
            buf.append(0)
