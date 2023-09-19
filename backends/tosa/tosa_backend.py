#
# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause
#

#
# Main implementation of partition and preprocess
#

import logging
import operator
import os
import tempfile

from typing import final, List

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

from . import tosa_mapping

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
class TosaPartitioner(Partitioner):
    compile_spec = []

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(TosaBackend.__name__, self.compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("TosaPartitioner::partition")
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

    f = open(path + filename, "wb")
    f.write(fb)
    f.close()

    f = open(path + "desc.json", "w")
    f.write(js)
    f.close()


def dbg_fail(node, tosa_fb, path):
    dbg_tosa_dump(tosa_fb, path)
    logger.warn("Internal error due to poorly handled node:")
    dbg_node(node)
    logger.warn(f"Debug output captured in '{path}'.")
    raise RuntimeError("TOSA Internal Error on node, enable logging for further info")


@final
class TosaBackend(BackendDetails):
    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> bytes:
        logger.info("TosaBackend::preprocess")

        # if a debug/test build capture output files from TOSA stage
        path = None
        debug_output = False
        for spec in compile_spec:
            if spec.key == "debug_tosa_path":
                path = spec.value.decode()
                debug_output = True

        # in non debug builds we still pass files to vela
        if path is None:
            path = tempfile.mkdtemp(prefix="arm_tosa_")

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

                # All paths have a single output
                tosa_fb.currRegion.currBasicBlock.addTensor(
                    outp.name, outp.shape, outp.dtype
                )

                op = tosa_mapping.op(node.target)
                attr = attr_torch_to_tosa(op, node)

                if op:
                    # a simple 1:1 mapping of operator taking 2 tensor arguments
                    assert len(inputs) == 2
                    assert inputs[0].dtype == outp.dtype
                    assert inputs[1].dtype == outp.dtype
                    tosa_fb.addOperator(
                        op, [inputs[0].name, inputs[1].name], [outp.name], attr
                    )
                else:
                    # A more complex mapping of operator
                    if exir_ops.edge.aten.addmm.default == node.target:
                        input = inputs[1]
                        weight = inputs[2]
                        bias = inputs[0]

                        # Reshape input tensor
                        # TODO: move shape compatibility promotion to function
                        # Many TOSA ops require a shape including a batch size so we make the implicit
                        # batch size from the edge graph explicit in TOSA
                        input_reshape_res = tosa_fb.addIntermediate(
                            (1,) + input.shape, outp.dtype
                        )
                        attr_input = ts.TosaSerializerAttribute()
                        attr_input.ReshapeAttribute((1,) + input.shape)
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [input.name],
                            [input_reshape_res.name],
                            attr_input,
                        )

                        # Reshape weight tensor
                        weight_reshape_res = tosa_fb.addIntermediate(
                            (1,) + weight.shape, outp.dtype
                        )
                        attr_weight = ts.TosaSerializerAttribute()
                        attr_weight.ReshapeAttribute((1,) + weight.shape)
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [weight.name],
                            [weight_reshape_res.name],
                            attr_weight,
                        )

                        # Reshape bias tensor
                        bias_reshape_res = tosa_fb.addIntermediate(
                            (
                                1,
                                1,
                            )
                            + bias.shape,
                            outp.dtype,
                        )
                        attr_bias = ts.TosaSerializerAttribute()
                        attr_bias.ReshapeAttribute(
                            (
                                1,
                                1,
                            )
                            + bias.shape
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [bias.name],
                            [bias_reshape_res.name],
                            attr_bias,
                        )

                        # Add dummy batch 1 to mm_shape
                        mm_shape = (1, input.shape[0], weight.shape[1])
                        # Define Intermediate tensor for MatMul res
                        mm_res = tosa_fb.addIntermediate(mm_shape, outp.dtype)

                        # Add MatMulOp
                        tosa_fb.addOperator(
                            TosaOp.Op().MATMUL,
                            [input_reshape_res.name, weight_reshape_res.name],
                            [mm_res.name],
                            attr_torch_to_tosa(TosaOp.Op().MATMUL, node),
                        )

                        # Add AddOp
                        add_res = tosa_fb.addIntermediate(mm_shape, outp.dtype)
                        tosa_fb.addOperator(
                            TosaOp.Op().ADD,
                            [bias_reshape_res.name, mm_res.name],
                            [add_res.name],
                            None,
                        )

                        # Reshape final result to original shape
                        attr_out = ts.TosaSerializerAttribute()
                        attr_out.ReshapeAttribute(outp.shape)
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE, [add_res.name], [outp.name], attr_out
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
                            int(inputs[1].threshold),
                            int(inputs[2].threshold),
                            inputs[1].threshold,
                            inputs[2].threshold,
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().CLAMP, [inputs[0].name], [outp.name], attr
                        )
                    elif exir_ops.edge.aten.convolution.default == node.target:
                        ## RESHAPE input tensor to NHWC_Order = [0, 2, 3, 1]
                        NHWC_Order = [0, 2, 3, 1]
                        attr_input_reshape = ts.TosaSerializerAttribute()
                        input_shape_NHWC = [inputs[0].shape[i] for i in NHWC_Order]
                        attr_input_reshape.ReshapeAttribute(input_shape_NHWC)
                        input_reshaped = tosa_fb.addIntermediate(
                            input_shape_NHWC, outp.dtype
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [inputs[0].name],
                            [input_reshaped.name],
                            attr_input_reshape,
                        )

                        ## CONV2DOp
                        attr = ts.TosaSerializerAttribute()
                        # PAD
                        pad_attr = [val for val in inputs[4].special for _ in (0, 1)]
                        # Stride
                        stride_attr = inputs[3].special
                        # Dilation
                        dilation_attr = inputs[5].special
                        attr.ConvAttribute(pad_attr, stride_attr, dilation_attr, 0, 0)

                        ## TOSA output shape is [NHWO] (num_batch, height, width, num_output)
                        NHWO_Order = [0, 2, 3, 1]
                        out_shape_TOSA_CONV2D = [outp.shape[i] for i in NHWO_Order]
                        conv2d_res = tosa_fb.addIntermediate(
                            out_shape_TOSA_CONV2D, outp.dtype
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().CONV2D,
                            [input_reshaped.name, inputs[1].name, inputs[2].name],
                            [conv2d_res.name],
                            attr,
                        )

                        ## Torch output shape is [NOHW]
                        NOHW_Order = [0, 3, 1, 2]
                        attr_output_transpose = ts.TosaSerializerAttribute()
                        attr_output_transpose.TransposeAttribute(NOHW_Order)
                        tosa_fb.addOperator(
                            TosaOp.Op().TRANSPOSE,
                            [conv2d_res.name],
                            [outp.name],
                            attr_output_transpose,
                        )
                    elif exir_ops.edge.aten.div.Tensor == node.target:
                        # Div is implemented as x/y = x*1/y
                        recip = tosa_fb.addIntermediate(
                            inputs[1].shape, inputs[1].dtype
                        )
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
                            0.1 == momentum.threshold
                        ), "Expected 0.1 momentum, not currently encoded into TOSA"

                        # %op1 = tosa.SUB(%x, %bmean)
                        # %op2 = tosa.ADD(%variance, %epsilon_const)
                        # %op3 = tosa.RSQRT(%op2)
                        # %op4 = tosa.MUL(%op1, %op3)
                        # %op5 = tosa.MUL(%op4, %weight)
                        # %output = tosa.ADD(%op5, %bias)

                        # Reshape mean to match rank of activations
                        mean_reshaped_res = tosa_fb.addIntermediate(
                            (1,)
                            + running_mean.shape
                            + (
                                1,
                                1,
                            ),
                            input_dtype,
                        )
                        attr_mean = ts.TosaSerializerAttribute()
                        attr_mean.ReshapeAttribute(
                            (1,)
                            + running_mean.shape
                            + (
                                1,
                                1,
                            )
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [running_mean.name],
                            [mean_reshaped_res.name],
                            attr_mean,
                        )

                        # Subtract mean
                        int1 = tosa_fb.addIntermediate(input_shape, input_dtype)
                        tosa_fb.addOperator(
                            TosaOp.Op().SUB,
                            [activations.name, mean_reshaped_res.name],
                            [int1.name],
                        )
                        # Adding eplison to variance
                        epsilon_const = tosa_fb.addConst(
                            [1], input_dtype, [epsilon.threshold]
                        )
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
                        var_reshaped_res = tosa_fb.addIntermediate(
                            (1,)
                            + running_var.shape
                            + (
                                1,
                                1,
                            ),
                            input_dtype,
                        )
                        attr_var = ts.TosaSerializerAttribute()
                        attr_var.ReshapeAttribute(
                            (1,)
                            + running_var.shape
                            + (
                                1,
                                1,
                            )
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().RESHAPE,
                            [int3.name],
                            [var_reshaped_res.name],
                            attr_var,
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
                        attr_input_transpose = ts.TosaSerializerAttribute()
                        attr_input_transpose.TransposeAttribute(NHWC_Order)

                        transeposed_input_shape = [
                            input_tensor.shape[i] for i in NHWC_Order
                        ]
                        input_transposed = tosa_fb.addIntermediate(
                            transeposed_input_shape, outp.dtype
                        )
                        tosa_fb.addOperator(
                            TosaOp.Op().TRANSPOSE,
                            [input_tensor.name],
                            [input_transposed.name],
                            attr_input_transpose,
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
                    elif operator.getitem == node.target:
                        item_name = inputs[0].name
                        ## Simply add an identityOp
                        tosa_fb.addOperator(
                            TosaOp.Op().IDENTITY, [item_name], [outp.name]
                        )
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
                    weight_values = p_data.detach().numpy()
                    tosa_fb.addConst(
                        inputs[0].shape, inputs[0].dtype, weight_values, name=out
                    )
                elif out in edge_program.graph_signature.inputs_to_buffers:
                    parameter_name = edge_program.graph_signature.inputs_to_buffers[
                        node.name
                    ]
                    p_data = edge_program.state_dict[parameter_name]

                    assert isinstance(p_data, torch.Tensor), "Expect Attr to be tensor"
                    weight_values = p_data.detach().numpy()
                    tosa_fb.addConst(
                        inputs[0].shape, inputs[0].dtype, weight_values, name=out
                    )
                else:
                    # Input argument
                    tensor = ts.TosaSerializerTensor(
                        inputs[0].name,
                        inputs[0].shape,
                        inputs[0].dtype,
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

        # Serialize and return the tosa flatbuffer
        fb = tosa_fb.serialize()
        return PreprocessResult(processed_bytes=bytes(fb))
