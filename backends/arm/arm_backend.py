# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#

import logging
import operator
import os
import struct
import subprocess
import tempfile
from typing import final, List

import numpy as np

import serializer.tosa_serializer as ts

import torch
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)

from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaOp
from torch._export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase

from . import tosa_mapping, tosa_quant_utils

# TOSA backend debug functionality
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


def dbg_node(node):
    # Debug output of node information
    logger.info("OP")
    logger.info(f"  op is {node.op}")
    logger.info(f"  name is {node.name}")
    logger.info(f"  node target is {node.target}")
    logger.info(f"  node args is {node.args}")
    logger.info(f"  node kwargs is {node.kwargs}")
    logger.info("  node.meta = ")
    for k, v in node.meta.items():
        logger.info(f"    '{k}' = {v}")
        if type([]) == type(v):
            for i in v:
                logger.info(f"      {i} ")


class TOSASupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.clone.default,
            operator.getitem,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        ]
        return supported


def attr_torch_to_tosa(op, node):
    if TosaOp.Op().MATMUL == op:
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(0, 0)
        return attr
    if TosaOp.Op().MUL == op:
        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(0)
        return attr
    return None


@final
class ArmPartitioner(Partitioner):
    compile_spec = []

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(ArmBackend.__name__, self.compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("ArmPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TOSASupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


# Output TOSA flatbuffer and test harness file
def dbg_tosa_dump(tosa_fb, path):
    filename = "output.tosa"

    logger.info(f"Emitting debug output to {path}")

    os.makedirs(path, exist_ok=True)

    fb = tosa_fb.serialize()
    js = tosa_fb.writeJson(filename)

    with open(path + filename, "wb") as f:
        f.write(fb)

    with open(path + "desc.json", "w") as f:
        f.write(js)


# Pack either input or output tensor block, compose the related arrays into
# per-io structs to simplify runtime use.
def vela_bin_pack_io(prefix, data):
    ios = struct.pack("<i", len(data[prefix + "_shape"]))
    for i in range(len(data[prefix + "_shape"])):
        io_shape = data[prefix + "_shape"][i]
        io_elem_size = data[prefix + "_elem_size"][i]
        io_offset = data[prefix + "_offset"][i]
        io_region = data[prefix + "_region"][i]
        assert len(io_shape) <= 4
        inp_pad = io_shape.tolist() + [0] * (4 - len(io_shape))
        io_struct = struct.pack(
            "<iiiiiii", *inp_pad, io_elem_size, io_offset, io_region
        )
        ios += io_struct
    return ios


# Output via Vela to binary stream for ArmBackendEthosU
# WARNING: Do not change this without changing VelaBinStream.cpp as that
#          function consumes this format and the two need to align.
def vela_compile(tosa_fb):
    with tempfile.TemporaryDirectory() as tmpdir:
        tosaname = "out.tosa"
        flatbuffer = tosa_fb.serialize()
        with open(os.path.join(tmpdir, tosaname), "wb") as f:
            f.write(flatbuffer)

        # invoke vela
        vela_command = (
            f"cd {tmpdir}; vela --accelerator-config ethos-u55-128 {tosaname}"
        )
        subprocess.run([vela_command], shell=True, check=True)

        np_path = os.path.join(tmpdir, "output", "out_sg0_vela.npz")
        blocks = b""

        with np.load(np_path, allow_pickle=False) as data:
            # Construct our modified output_blocks with data in a form easily
            # digested on the device side
            bin_blocks = {"vela_bin_stream": b""}

            # copy command data through unmodified
            bin_blocks["cmd_data"] = data["cmd_data"].tobytes()

            # copy weight data through unmodified
            bin_blocks["weight_data"] = data["weight_data"].tobytes()

            # Add a block for scratch, inputs and outputs;  scratch shape is a 1 element
            # array giving us size in bytes so extract this and add a block of 0's.
            # Currently we preallocated this on the host to provide SRAM for computation.
            if not isinstance(data["scratch_shape"][0]), np.int64):
                raise RuntimeError("Expected scratch to be int64")
            block_length = int(data["scratch_shape"][0])
            bin_blocks["scratch_data"] = b"\x00" * block_length

            # Capture inputs and outputs
            bin_blocks["inputs"] = vela_bin_pack_io("input", data)
            bin_blocks["outputs"] = vela_bin_pack_io("output", data)

            bin_blocks["vela_end_stream"] = b""

            # Emit the NPZ regions as:
            #  - 16 byte block name null terminated string (padded to 16 if name shorter)
            #  - 4 bytes of int32 block length and 12 bytes of 0's
            #  - block data (padded to 16 byte alignment at end)
            # Repeat for all blocks
            for key in bin_blocks.keys():
                block_name = bytes(key, "utf8")[:15]
                block_name = block_name + b"\x00" * (16 - len(block_name))

                # We need the acual unpadded block lengths for hw setup
                block_length = struct.pack("<iiii", len(bin_blocks[key]), 0, 0, 0)

                # Pad block data to multiple of 16 bytes
                block_data = bin_blocks[key]
                block_data = block_data + b"\x00" * (15 - (len(block_data) - 1) % 16)

                block = block_name + block_length + block_data
                blocks = blocks + block

        return blocks


def dbg_fail(node, tosa_fb, path):
    dbg_tosa_dump(tosa_fb, path)
    logger.warn("Internal error due to poorly handled node:")
    dbg_node(node)
    logger.warn(f"Debug output captured in '{path}'.")
    raise RuntimeError("TOSA Internal Error on node, enable logging for further info")


# Helper function to match TOSA's broadcasting rank requirement
# Ref: TOSA 0.80.0 specification - 1.9.3. Data Layouts from
# https://www.mlplatform.org/tosa/tosa_spec.html
def promote_shape(tosa_fb, arg, promoted_shape, out_dtype):
    assert np.prod(arg.shape) == np.prod(promoted_shape), "Incompatible promoted shape"
    reshape_res = tosa_fb.addIntermediate(promoted_shape, out_dtype)
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(promoted_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [arg.name], [reshape_res.name], attr)
    return reshape_res


# Helper transpose function to match TOSA's shape requirements
# E.g., TOSA 0.80.0 specification - 2.3.3 CONV2D shapes:
# https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d
def transpose_helper(tosa_fb, input, new_order, out_dtype):
    # Check new_order's length is equal to input rank
    assert len(input.shape) == len(new_order), "Wrong shape order length"

    # Check no duplications
    assert len(set(new_order)) == len(new_order), "Contain duplicated dim numbers"

    # Check all dims are valid
    for idx in new_order:
        if idx < 0:
            assert True, "Negative dim number"
        elif idx >= len(input.shape):
            assert True, "Dim is greater than input rank"

    input_shape_transpoed = [input.shape[i] for i in new_order]
    attr = ts.TosaSerializerAttribute()
    attr.TransposeAttribute(new_order)
    input_transposed = tosa_fb.addIntermediate(input_shape_transpoed, out_dtype)
    tosa_fb.addOperator(
        TosaOp.Op().TRANSPOSE, [input.name], [input_transposed.name], attr
    )
    return input_transposed


def broadcastShapes(shape1, shape2):
    assert len(shape1) == len(shape2), "broadcastShape::shapes must have same ranks"

    need_broadcasting = False
    for val1, val2 in zip(shape1, shape2):
        if val1 != val2:
            need_broadcasting = True
    if not need_broadcasting:
        return shape1

    broadcasted_shape = list(shape1)
    shape2 = list(shape2)
    for idx, _ in enumerate(broadcasted_shape):
        if broadcasted_shape[idx] == 1:
            broadcasted_shape[idx] = shape2[idx]
        else:
            assert not (
                shape2[idx] != 1 and shape2[idx] != broadcasted_shape[idx]
            ), "broadcastShape::broadcast shape mismatch"

    return broadcasted_shape


def getNodeArgs(node):
    return [tosa_mapping.TosaArg(arg) for arg in node.args]


def getQuantNodeArgs(node):
    quant_args = [tosa_mapping.TosaArg(arg) for arg in node.args]
    # Return the scale and zp
    return quant_args[1].number, quant_args[2].number


@final
class ArmBackend(BackendDetails):
    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info("ArmBackend::preprocess")

        # if a debug/test build capture output files from TOSA stage
        path = None
        debug_output = False
        output_format = "vela"
        for spec in compile_spec:
            if spec.key == "debug_tosa_path":
                path = spec.value.decode()
                debug_output = True
            if spec.key == "output_format":
                output_format = spec.value.decode()

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.
        tosa_fb = ts.TosaSerializer(path)

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                # Unpack arguments and convert
                inputs = []
                for arg in node.args:
                    inputs.append(tosa_mapping.TosaArg(arg))

                # Convert output (this node itself)
                outp = tosa_mapping.TosaArg(node)

                is_quant_node = tosa_quant_utils.isQuantNode(node)

                if is_quant_node:
                    tosa_fb.currRegion.currBasicBlock.addTensor(
                        outp.name, outp.shape, ts.DType.INT8
                    )
                else:
                    tosa_fb.currRegion.currBasicBlock.addTensor(
                        outp.name, outp.shape, outp.dtype
                    )

                op = tosa_mapping.op(node.target)
                attr = attr_torch_to_tosa(op, node)

                # Node walker TODO: refactor this to use visitor pattern
                if exir_ops.edge.aten.add.Tensor == node.target:
                    if is_quant_node:
                        # Single input or not
                        if len(node.all_input_nodes) == 1:
                            input_node_A = node.all_input_nodes[0]
                            input_node_B = node.all_input_nodes[0]
                        else:
                            input_node_A, input_node_B = node.all_input_nodes

                        # Get input scale_factor and zero_points for A, B
                        input_A, input_A_scale, input_A_zp, _, _, _ = getNodeArgs(
                            input_node_A
                        )
                        input_B, input_B_scale, input_B_zp, _, _, _ = getNodeArgs(
                            input_node_B
                        )

                        max_scale_2x = 2.0 * max(
                            input_A_scale.number, input_B_scale.number
                        )
                        inputA_rescale_scale = input_A_scale.number / max_scale_2x
                        inputB_rescale_scale = input_B_scale.number / max_scale_2x

                        input_A_rescaled_to_int32 = (
                            tosa_quant_utils.buildRescaleToInt32(
                                tosa_fb,
                                input_A,
                                input_A_zp.number,
                                inputA_rescale_scale,
                            )
                        )

                        input_B_rescaled_to_int32 = (
                            tosa_quant_utils.buildRescaleToInt32(
                                tosa_fb,
                                input_B,
                                input_B_zp.number,
                                inputB_rescale_scale,
                            )
                        )

                        ## Do the INT32 Add
                        broadcasted_shape = broadcastShapes(
                            input_A.shape, input_B.shape
                        )
                        add_res = tosa_fb.addIntermediate(
                            broadcasted_shape, ts.DType.INT32
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().ADD,
                            [
                                input_A_rescaled_to_int32.name,
                                input_B_rescaled_to_int32.name,
                            ],
                            [add_res.name],
                            None,
                        )

                        # Output
                        output_node = list(node.users)[0]
                        _, output_scale, output_zp, _, _, _ = getNodeArgs(output_node)
                        output_rescale_scale = max_scale_2x / (output_scale.number)

                        # Rescale Back to INT8
                        tosa_quant_utils.buildRescaleFromInt32(
                            tosa_fb,
                            add_res.name,
                            outp.name,
                            output_zp.number,
                            output_rescale_scale,
                        )
                    else:
                        # FP32 Add lowering
                        tosa_fb.addOperator(
                            op, [inputs[0].name, inputs[1].name], [outp.name], attr
                        )
                elif exir_ops.edge.aten.addmm.default == node.target:
                    bias, input, weight = inputs

                    output_dtype = ts.DType.INT8 if is_quant_node else outp.dtype

                    # Reshape input, weight, bias tensors
                    input_reshape_res = promote_shape(
                        tosa_fb, input, (1,) + input.shape, output_dtype
                    )
                    weight_reshape_res = promote_shape(
                        tosa_fb, weight, (1,) + weight.shape, output_dtype
                    )

                    bias_dtype = ts.DType.INT32 if is_quant_node else outp.dtype
                    bias_reshape_res = promote_shape(
                        tosa_fb,
                        bias,
                        (
                            1,
                            1,
                        )
                        + bias.shape,
                        bias_dtype,
                    )

                    # Add dummy batch 1 to mm_shape
                    mm_shape = (1, input.shape[0], weight.shape[1])
                    # Define Intermediate tensor for MatMul res
                    mm_res = tosa_fb.addIntermediate(
                        mm_shape, ts.DType.INT32 if is_quant_node else output_dtype
                    )

                    # Add MatMulOp
                    attr_matmul = ts.TosaSerializerAttribute()
                    a_zp, b_zp = (-128, 0) if is_quant_node else (0, 0)
                    attr_matmul.MatMulAttribute(a_zp, b_zp)
                    tosa_fb.addOperator(
                        TosaOp.Op().MATMUL,
                        [input_reshape_res.name, weight_reshape_res.name],
                        [mm_res.name],
                        attr_matmul,
                    )

                    # Add AddOp
                    add_res = tosa_fb.addIntermediate(
                        mm_shape, ts.DType.INT32 if is_quant_node else output_dtype
                    )

                    tosa_fb.addOperator(
                        TosaOp.Op().ADD,
                        [bias_reshape_res.name, mm_res.name],
                        [add_res.name],
                        None,
                    )

                    if is_quant_node:
                        # Read inputs' parent nodes
                        #
                        _, input_node, weight_node = node.all_input_nodes
                        input_scale, _ = getQuantNodeArgs(input_node)
                        weight_node_q_node = weight_node.all_input_nodes[0]
                        weight_scale, _ = getQuantNodeArgs(weight_node_q_node)

                        consumer_node = list(node.users)[0]
                        consumer_node_scale, consumer_node_node_zp = getQuantNodeArgs(
                            consumer_node
                        )

                        output_rescale_scale = (
                            input_scale * weight_scale
                        ) / consumer_node_scale
                        (
                            multiplier_output,
                            shift_output,
                        ) = tosa_quant_utils.computeMultiplierAndShift(
                            output_rescale_scale
                        )

                        attr_rescale_output = ts.TosaSerializerAttribute()
                        attr_rescale_output.RescaleAttribute(
                            input_zp=0,
                            output_zp=consumer_node_node_zp,
                            multiplier=[multiplier_output],
                            shift=[shift_output],
                            scale32=True,
                            double_round=True,
                            per_channel=False,
                        )
                        add_res_int8 = tosa_fb.addIntermediate(mm_shape, ts.DType.INT8)
                        tosa_fb.addOperator(
                            TosaOp.Op().RESCALE,
                            [add_res.name],
                            [add_res_int8.name],
                            attr_rescale_output,
                        )
                    # Reshape final result to original shape
                    attr_out = ts.TosaSerializerAttribute()
                    attr_out.ReshapeAttribute(outp.shape)
                    tosa_fb.addOperator(
                        TosaOp.Op().RESHAPE,
                        [add_res_int8.name if is_quant_node else add_res.name],
                        [outp.name],
                        attr_out,
                    )
                elif exir_ops.edge.aten.permute_copy.default == node.target:
                    attr = ts.TosaSerializerAttribute()
                    attr.TransposeAttribute(inputs[1].special)
                    tosa_fb.addOperator(
                        TosaOp.Op().TRANSPOSE, [inputs[0].name], [outp.name], attr
                    )
                elif exir_ops.edge.aten.hardtanh.default == node.target:
                    attr = ts.TosaSerializerAttribute()
                    attr.ClampAttribute(
                        tosa_fb.builder,
                        int(inputs[1].number),
                        int(inputs[2].number),
                        inputs[1].number,
                        inputs[2].number,
                    )
                    tosa_fb.addOperator(
                        TosaOp.Op().CLAMP, [inputs[0].name], [outp.name], attr
                    )
                elif exir_ops.edge.aten.convolution.default == node.target:
                    input, weight, bias, stride, pad, dilation, _, _, group = inputs

                    # Currently only int8 is supported in quantized types.
                    actual_out_type = ts.DType.INT8 if is_quant_node else outp.dtype

                    ## Transpose input tensor to NHWC_Order for TOSA
                    NHWC_Order = [0, 2, 3, 1]
                    input_transposed = transpose_helper(
                        tosa_fb, input, NHWC_Order, actual_out_type
                    )

                    # Get the attributes of convolution.
                    attr = ts.TosaSerializerAttribute()
                    pad_attr = [val for val in pad.special for _ in (0, 1)]
                    stride_attr = stride.special
                    dilation_attr = dilation.special
                    attr.ConvAttribute(pad_attr, stride_attr, dilation_attr, 0, 0)

                    # Non-bias case.
                    if len(node.all_input_nodes) == 2:
                        # Create a zero bias tensor if not presented
                        out_channels = weight.shape[0]
                        bias_name = "bias" + node.name.split("default", 1)[1]
                        bias = tosa_fb.addConst(
                            [out_channels],
                            ts.DType.INT32 if is_quant_node else outp.dtype,
                            [0] * out_channels,
                            name=bias_name,
                        )

                    if group.number > 1:
                        assert (
                            is_quant_node is False
                        ), "quantized depthwise convolution is not supported yet in BI mode"

                        # Transpose weight to [KH, KW, C, M]
                        weight_HWCM_Order = [2, 3, 0, 1]
                        weight_transposed = transpose_helper(
                            tosa_fb, weight, weight_HWCM_Order, outp.dtype
                        )

                        ## TOSA output shape is [N, H, W, C*M]
                        NHWO_Order = [0, 2, 3, 1]
                        out_shape_TOSA_Depthwise_CONV2D = [
                            outp.shape[i] for i in NHWO_Order
                        ]

                        conv2d_res = tosa_fb.addIntermediate(
                            out_shape_TOSA_Depthwise_CONV2D, outp.dtype
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().DEPTHWISE_CONV2D,
                            [
                                input_transposed.name,
                                weight_transposed.name,
                                bias.name,
                            ],
                            [conv2d_res.name],
                            attr,
                        )
                    else:
                        # TODO: Transpose the weight AoT
                        # Transpose weight to [OC, H, W, IC]
                        weight_CHWC_Order = [0, 2, 3, 1]
                        weight_transposed = transpose_helper(
                            tosa_fb, weight, weight_CHWC_Order, actual_out_type
                        )

                        ## TOSA output shape is [NHWO]
                        NHWO_Order = [0, 2, 3, 1]
                        out_shape_TOSA_CONV2D = [outp.shape[i] for i in NHWO_Order]

                        # The output type is int32 when input type is int8.
                        conv2d_res = tosa_fb.addIntermediate(
                            out_shape_TOSA_CONV2D,
                            ts.DType.INT32 if is_quant_node else outp.dtype,
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().CONV2D,
                            [
                                input_transposed.name,
                                weight_transposed.name,
                                bias.name,
                            ],
                            [conv2d_res.name],
                            attr,
                        )

                    ## Torch output shape is [NOHW]
                    NOHW_Order = [0, 3, 1, 2]
                    attr_output_transpose = ts.TosaSerializerAttribute()
                    attr_output_transpose.TransposeAttribute(NOHW_Order)

                    # For quantized convolution, rescale the output value back to the same
                    # integer value domain of the next op. Otherwise return float32 output.
                    if is_quant_node:
                        # Get scale_factor from input, weight, and output.
                        _, input_scale, _, _, _, _ = getNodeArgs(node.args[0])
                        _, weight_scale, _, _, _, _ = getNodeArgs(node.args[1])
                        _, output_scale, _, _, _, _ = getNodeArgs(list(node.users)[0])

                        conv2d_res = tosa_quant_utils.buildRescaleOpConvOutput(
                            tosa_fb,
                            conv2d_res,
                            actual_out_type,
                            input_scale,
                            weight_scale,
                            output_scale,
                        )

                    tosa_fb.addOperator(
                        TosaOp.Op().TRANSPOSE,
                        [conv2d_res.name],
                        [outp.name],
                        attr_output_transpose,
                    )
                elif exir_ops.edge.aten.div.Tensor == node.target:
                    # Div is implemented as x/y = x*1/y
                    recip = tosa_fb.addIntermediate(inputs[1].shape, inputs[1].dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().RECIPROCAL, [inputs[1].name], [recip.name]
                    )

                    attr = ts.TosaSerializerAttribute()
                    attr.MulAttribute(0)
                    tosa_fb.addOperator(
                        TosaOp.Op().MUL,
                        [inputs[0].name, recip.name],
                        [outp.name],
                        attr,
                    )
                elif (
                    exir_ops.edge.aten._native_batch_norm_legit_no_training.default
                    == node.target
                ):
                    # Decompose batch norm into sequence
                    (
                        activations,
                        _,
                        _,
                        running_mean,
                        running_var,
                        momentum,
                        epsilon,
                    ) = inputs

                    input_dtype = activations.dtype
                    input_shape = activations.shape

                    assert (
                        0.1 == momentum.number
                    ), "Expected 0.1 momentum, not currently encoded into TOSA"

                    # %op1 = tosa.SUB(%x, %bmean)
                    # %op2 = tosa.ADD(%variance, %epsilon_const)
                    # %op3 = tosa.RSQRT(%op2)
                    # %op4 = tosa.MUL(%op1, %op3)
                    # %op5 = tosa.MUL(%op4, %weight)
                    # %output = tosa.ADD(%op5, %bias)

                    # Reshape mean to match rank of activations
                    mean_reshaped_res = promote_shape(
                        tosa_fb,
                        running_mean,
                        (1,)
                        + running_mean.shape
                        + (
                            1,
                            1,
                        ),
                        input_dtype,
                    )

                    # Subtract mean
                    int1 = tosa_fb.addIntermediate(input_shape, input_dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().SUB,
                        [activations.name, mean_reshaped_res.name],
                        [int1.name],
                    )
                    # Adding eplison to variance
                    epsilon_const = tosa_fb.addConst([1], input_dtype, [epsilon.number])
                    int2 = tosa_fb.addIntermediate(running_var.shape, input_dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().ADD,
                        [running_var.name, epsilon_const.name],
                        [int2.name],
                    )
                    # Push downward the variance
                    int3 = tosa_fb.addIntermediate(running_var.shape, input_dtype)
                    tosa_fb.addOperator(TosaOp.Op().RSQRT, [int2.name], [int3.name])

                    # Reshape variable to match rank of activations
                    var_reshaped_res = promote_shape(
                        tosa_fb,
                        int3,
                        (1,)
                        + running_var.shape
                        + (
                            1,
                            1,
                        ),
                        input_dtype,
                    )

                    # Multiple shifted activations with reciprocal variance
                    # int4 = tosa_fb.addIntermediate( input_shape, input_dtype )
                    tosa_fb.addOperator(
                        TosaOp.Op().MUL,
                        [int1.name, var_reshaped_res.name],
                        [outp.name],
                        attr_torch_to_tosa(TosaOp.Op().MUL, node),
                    )
                elif exir_ops.edge.aten.avg_pool2d.default == node.target:
                    input_tensor = inputs[0]
                    kernel_size_list = inputs[1].special
                    stride_size_list = inputs[2].special
                    try:
                        pad_size_list = inputs[3].special
                    except IndexError:
                        pad_size_list = [0, 0, 0, 0]

                    attr = ts.TosaSerializerAttribute()
                    attr.PoolAttribute(
                        kernel=kernel_size_list,
                        stride=stride_size_list,
                        pad=pad_size_list,
                        input_zp=0,
                        output_zp=0,
                        accum_dtype=8,
                    )  # FP32 accum type

                    # Torch's input is [N,C,H,W], TOSA is [N, H, W, C],
                    # Transpose to align with TOSA
                    NHWC_Order = [0, 2, 3, 1]
                    input_transposed = transpose_helper(
                        tosa_fb, input_tensor, NHWC_Order, outp.dtype
                    )

                    avg_pool2d_res_shape = [outp.shape[i] for i in NHWC_Order]
                    avg_pool2d_res = tosa_fb.addIntermediate(
                        avg_pool2d_res_shape, outp.dtype
                    )
                    tosa_fb.addOperator(
                        TosaOp.Op().AVG_POOL2D,
                        [input_transposed.name],
                        [avg_pool2d_res.name],
                        attr,
                    )

                    # TOSA is [N, H, W, C], Transpose back to Torch's [N, C, H, W]
                    NCHW_Order = [0, 3, 1, 2]
                    attr_output_transpose = ts.TosaSerializerAttribute()
                    attr_output_transpose.TransposeAttribute(NCHW_Order)
                    tosa_fb.addOperator(
                        TosaOp.Op().TRANSPOSE,
                        [avg_pool2d_res.name],
                        [outp.name],
                        attr_output_transpose,
                    )
                elif exir_ops.edge.aten._softmax.default == node.target:
                    input_name = inputs[0].name
                    input_shape = inputs[0].shape
                    dim_value = inputs[1].number

                    ## softmax = exp(logits - max(logits)) / reduce_sum(exp(logits - max(logits)), -1)
                    # FP32
                    # reduce_max_res = reducemax(logits)
                    # sub_res = sub(inputs, reduce_max_res)
                    # exp_res = exp(sub_res)
                    # reduce_sum_res = reduce_sum(exp_res, -1)
                    # inverted_reduce_sum = reciprocal(reduce_sum_res)
                    # output = mul(exp_res, inverted_reduce_sum)

                    # Max_Reduction
                    attr_axis = ts.TosaSerializerAttribute()
                    attr_axis.AxisAttribute(axis=dim_value)
                    reduced_shape = list(input_shape)
                    reduced_shape[dim_value] = 1
                    reduce_max_res = tosa_fb.addIntermediate(reduced_shape, outp.dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().REDUCE_MAX,
                        [input_name],
                        [reduce_max_res.name],
                        attr_axis,
                    )

                    # Subtract max from logits
                    sub_res = tosa_fb.addIntermediate(input_shape, outp.dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().SUB,
                        [input_name, reduce_max_res.name],
                        [sub_res.name],
                    )

                    # Raise the subtraction results to exponent
                    exp_res = tosa_fb.addIntermediate(input_shape, outp.dtype)
                    tosa_fb.addOperator(TosaOp.Op().EXP, [sub_res.name], [exp_res.name])

                    # Reduce_sum of the calculated exponent value
                    reduce_sum_res = tosa_fb.addIntermediate(reduced_shape, outp.dtype)
                    tosa_fb.addOperator(
                        TosaOp.Op().REDUCE_SUM,
                        [exp_res.name],
                        [reduce_sum_res.name],
                        attr_axis,
                    )

                    # Invert the reduce_sum
                    inverted_reduce_sum = tosa_fb.addIntermediate(
                        reduced_shape, outp.dtype
                    )
                    tosa_fb.addOperator(
                        TosaOp.Op().RECIPROCAL,
                        [reduce_sum_res.name],
                        [inverted_reduce_sum.name],
                    )

                    # Multiply two parts to get the final results
                    attr_mul = ts.TosaSerializerAttribute()
                    attr_mul.MulAttribute(0)
                    tosa_fb.addOperator(
                        TosaOp.Op().MUL,
                        [exp_res.name, inverted_reduce_sum.name],
                        [outp.name],
                        attr_mul,
                    )
                elif exir_ops.edge.aten.view_copy.default == node.target:
                    attr = ts.TosaSerializerAttribute()
                    new_shape = inputs[1].special
                    attr.ReshapeAttribute(new_shape)
                    tosa_fb.addOperator(
                        TosaOp.Op().RESHAPE, [inputs[0].name], [outp.name], attr
                    )
                elif node.target in [
                    operator.getitem,
                    tosa_quant_utils.q_op,
                    tosa_quant_utils.dq_op,
                    exir_ops.edge.aten.clone.default,
                ]:
                    item_name = inputs[0].name
                    ## Simply add an identityOp
                    tosa_fb.addOperator(TosaOp.Op().IDENTITY, [item_name], [outp.name])
                else:
                    raise RuntimeError(f"Unknown operator {node.target}")

                continue

            elif node.op == "placeholder":
                assert (
                    node.name == node.target
                ), "Expect placeholder name and target to match"
                assert 0 == len(node.args), "Can't handle default input values"

                # TODO: this may fail on int64 constant input
                inputs = [tosa_mapping.TosaArg(node)]
                out = node.name

                if out in edge_program.graph_signature.inputs_to_parameters:
                    parameter_name = edge_program.graph_signature.inputs_to_parameters[
                        node.name
                    ]
                    p_data = edge_program.state_dict[parameter_name]

                    assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
                    parameter_values = p_data.detach().numpy()

                    # Check if they're for quantized nodes
                    consumer_node = list(node.users)[0]
                    if consumer_node.target in tosa_quant_utils.dq_q_ops:
                        _, weight_node_scale, weight_node_zp, _, _, _ = getNodeArgs(
                            consumer_node
                        )

                        parameter_values_quantized = (
                            (parameter_values / weight_node_scale.number)
                            + weight_node_zp.number
                        ).astype(np.int8)
                        tosa_fb.addConst(
                            inputs[0].shape,
                            ts.DType.INT8,
                            parameter_values_quantized,
                            name=out,
                        )
                    elif (
                        consumer_node.target == exir_ops.edge.aten.addmm.default
                        and list(consumer_node.users)[0].target == tosa_quant_utils.q_op
                    ):
                        (
                            _,
                            input_node,
                            weight_node_permuted,
                        ) = consumer_node.all_input_nodes
                        weight_node = weight_node_permuted.all_input_nodes[0]

                        input_node_scale, _ = getQuantNodeArgs(input_node)
                        weight_node_scale, weight_node_zp = getQuantNodeArgs(
                            weight_node
                        )

                        parameter_values_quantized = (
                            parameter_values / (input_node_scale * weight_node_scale)
                        ).astype(np.int32)

                        tosa_fb.addConst(
                            inputs[0].shape,
                            ts.DType.INT32,
                            parameter_values_quantized,
                            name=out,
                        )
                    elif (
                        consumer_node.target == exir_ops.edge.aten.convolution.default
                        and list(consumer_node.users)[0].target == tosa_quant_utils.q_op
                    ):
                        (
                            input_node,
                            weight_node,
                            bias_node,
                        ) = consumer_node.all_input_nodes

                        input_node_scale, _ = getQuantNodeArgs(input_node)
                        weight_node_scale, _ = getQuantNodeArgs(weight_node)

                        bias_scales = input_node_scale * weight_node_scale
                        parameter_values_quantized = (
                            parameter_values / bias_scales
                        ).astype(np.int32)

                        tosa_fb.addConst(
                            inputs[0].shape,
                            ts.DType.INT32,
                            parameter_values_quantized,
                            name=out,
                        )
                    else:
                        tosa_fb.addConst(
                            inputs[0].shape, inputs[0].dtype, parameter_values, name=out
                        )

                elif out in edge_program.graph_signature.inputs_to_buffers:
                    parameter_name = edge_program.graph_signature.inputs_to_buffers[
                        node.name
                    ]
                    p_data = edge_program.state_dict[parameter_name]

                    assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
                    buffer_values = p_data.detach().numpy()
                    tosa_fb.addConst(
                        inputs[0].shape, inputs[0].dtype, buffer_values, name=out
                    )
                else:
                    tensor = ts.TosaSerializerTensor(
                        inputs[0].name,
                        inputs[0].shape,
                        ts.DType.INT8
                        if tosa_quant_utils.isQuantArg(node)
                        else inputs[0].dtype,
                        data=None,
                        placeholderFilename=inputs[0].name + ".npy",
                    )
                    tosa_fb.addInputTensor(tensor)
                continue

            elif node.op == "output":
                for output in node.args[0]:
                    tosa_fb.addOutputTensor(
                        tosa_fb.currRegion.currBasicBlock.tensors[output.name]
                    )
                continue

            else:
                # This will only happen if an unpartitioned graph is passed without
                # any checking of compatibility.
                dbg_fail(node, tosa_fb, path)

        if debug_output is True:
            dbg_tosa_dump(tosa_fb, path)

        # Serialize and return the program. While we have always produced TOSA
        # output as an intermediate, some flows compile to device binaries in
        # preprocess and some consume TOSA fb directly.
        if output_format == "vela":
            # Emit vela_bin_stream format
            binary = vela_compile(tosa_fb)
        elif output_format == "tosa":
            # Emit TOSA flatbuffer
            binary = bytes(tosa_fb.serialize())
        else:
            raise RuntimeError(f"Unknown format {output_format}")

        return PreprocessResult(processed_bytes=binary)
