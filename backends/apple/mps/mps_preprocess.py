#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import Any, Dict, final, List, Optional, Union

import torch

from executorch.backends.apple.mps.utils.graph_bindings import graph_bindings
from executorch.backends.apple.mps.utils.mps_utils import get_mps_data_type

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.exported_program import ExportedProgram
from torch._subclasses import FakeTensor


def get_param_from_node(
    node: torch.fx.Node, edge_program: ExportedProgram
) -> Optional[torch.nn.Parameter]:
    """
    Returns the parameter associated with the given node in the edge program.
    Returns None if the node is not a parameter within the edge_program
    """
    if node.name in edge_program.graph_signature.inputs_to_parameters:
        parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
        return edge_program.state_dict[parameter_name]
    elif node.name in edge_program.graph_signature.inputs_to_buffers:
        buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
        return edge_program.state_dict[buffer_name]
    return None


def create_mpsgraph_constant_tensor(tensor: torch.Tensor, mpsGraph):
    if tensor.dim() == 0:
        return mpsGraph.constant(tensor.item(), get_mps_data_type(tensor.dtype))
    else:
        return mpsGraph.constantTensor(tensor, get_mps_data_type(tensor.dtype))


@final
class MPSBackend(BackendDetails):
    @staticmethod
    def fetch_attr(node: torch.fx.Node, edge_program: ExportedProgram):
        attr_itr = edge_program

        attr_itr = getattr(node.graph.owning_module, node.target)
        return attr_itr

    @staticmethod
    def eval_shape(node):
        def eval_expr(symint: Union[int, torch.SymInt, FakeTensor]) -> Optional[int]:
            if isinstance(symint, int):
                return symint

            return None

        """
      Evaluate the shape of a node.

      symint can of of type `SymInt`, `FakeTensor`, a `List[Union[FakeTensor, SymInt]]`, or `None`
      """
        if isinstance(node, FakeTensor):
            return node.shape

        new_shape = []
        for _, s in enumerate(node):
            new_shape.append(eval_expr(s))
        return new_shape

    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        # C++ MPSGraph bindings.
        mpsGraph = graph_bindings.MPSGraphModule()

        unaryOps = {
            exir_ops.edge.aten.exp.default: mpsGraph.exp,
            exir_ops.edge.aten.exp2.default: mpsGraph.exp2,
            exir_ops.edge.aten.reciprocal.default: mpsGraph.reciprocal,
            exir_ops.edge.aten.sqrt.default: mpsGraph.sqrt,
            exir_ops.edge.aten.neg.default: mpsGraph.neg,
            exir_ops.edge.aten.log.default: mpsGraph.log,
            exir_ops.edge.aten.log10.default: mpsGraph.log10,
            exir_ops.edge.aten.log2.default: mpsGraph.log2,
            exir_ops.edge.aten.erf.default: mpsGraph.erf,
            exir_ops.edge.aten.floor.default: mpsGraph.floor,
            exir_ops.edge.aten.ceil.default: mpsGraph.ceil,
            exir_ops.edge.aten.rsqrt.default: mpsGraph.rsqrt,
            exir_ops.edge.aten.sigmoid.default: mpsGraph.sigmoid,
            exir_ops.edge.aten.sin.default: mpsGraph.sin,
            exir_ops.edge.aten.sign.default: mpsGraph.sign,
            exir_ops.edge.aten.cos.default: mpsGraph.cos,
            exir_ops.edge.aten.tan.default: mpsGraph.tan,
            exir_ops.edge.aten.abs.default: mpsGraph.abs,
            exir_ops.edge.aten.asin.default: mpsGraph.asin,
            exir_ops.edge.aten.acos.default: mpsGraph.acos,
            exir_ops.edge.aten.atan.default: mpsGraph.atan,
            exir_ops.edge.aten.sinh.default: mpsGraph.sinh,
            exir_ops.edge.aten.cosh.default: mpsGraph.cosh,
            exir_ops.edge.aten.tanh.default: mpsGraph.tanh,
            exir_ops.edge.aten.asinh.default: mpsGraph.asinh,
            exir_ops.edge.aten.acosh.default: mpsGraph.acosh,
            exir_ops.edge.aten.atanh.default: mpsGraph.atanh,
            exir_ops.edge.aten.bitwise_not.default: mpsGraph.bitwise_not,
            exir_ops.edge.aten.isnan.default: mpsGraph.isnan,
            exir_ops.edge.aten.isinf.default: mpsGraph.isinf,
            exir_ops.edge.aten.round.default: mpsGraph.round,
        }

        binaryOps = {
            exir_ops.edge.aten.mm.default: mpsGraph.mm,
            exir_ops.edge.aten.bmm.default: mpsGraph.bmm,
            exir_ops.edge.aten.mul.Tensor: mpsGraph.mul,
            exir_ops.edge.aten.div.Tensor: mpsGraph.div,
            exir_ops.edge.aten.div.Tensor_mode: mpsGraph.div,
            exir_ops.edge.aten.floor_divide.default: mpsGraph.floor_divide,
            exir_ops.edge.aten.fmod.Tensor: mpsGraph.fmod,
            exir_ops.edge.aten.remainder.Tensor: mpsGraph.remainder,
            exir_ops.edge.aten.bitwise_and.Tensor: mpsGraph.bitwise_and,
            exir_ops.edge.aten.bitwise_or.Tensor: mpsGraph.bitwise_or,
            exir_ops.edge.aten.bitwise_xor.Tensor: mpsGraph.bitwise_xor,
            exir_ops.edge.aten.eq.Tensor: mpsGraph.eq,
            exir_ops.edge.aten.ne.Tensor: mpsGraph.ne,
            exir_ops.edge.aten.ge.Tensor: mpsGraph.ge,
            exir_ops.edge.aten.gt.Tensor: mpsGraph.gt,
            exir_ops.edge.aten.le.Tensor: mpsGraph.le,
            exir_ops.edge.aten.lt.Tensor: mpsGraph.lt,
            exir_ops.edge.aten.pow.Tensor_Tensor: mpsGraph.pow,
            exir_ops.edge.aten.minimum.default: mpsGraph.minimum,
        }

        binaryOpsWithScalar = {
            exir_ops.edge.aten.mul.Scalar: mpsGraph.mulWithScalar,
            exir_ops.edge.aten.remainder.Scalar: mpsGraph.remainder,
            exir_ops.edge.aten.eq.Scalar: mpsGraph.eq,
            exir_ops.edge.aten.ne.Scalar: mpsGraph.ne,
            exir_ops.edge.aten.ge.Scalar: mpsGraph.ge,
            exir_ops.edge.aten.gt.Scalar: mpsGraph.gt,
            exir_ops.edge.aten.le.Scalar: mpsGraph.le,
            exir_ops.edge.aten.lt.Scalar: mpsGraph.lt,
            exir_ops.edge.aten.bitwise_and.Scalar: mpsGraph.bitwise_and,
            exir_ops.edge.aten.bitwise_or.Scalar: mpsGraph.bitwise_or,
            exir_ops.edge.aten.bitwise_xor.Scalar: mpsGraph.bitwise_xor,
            exir_ops.edge.aten.pow.Tensor_Scalar: mpsGraph.pow,
        }

        # `graph_nodes` dictionary is made out of <key> : <MPSGraphTensor*>
        graphNodes: Dict[str, Any] = {}

        for node in edge_program.graph.nodes:
            if node.op == "get_attr":
                attr = MPSBackend.fetch_attr(node, edge_program)
                graphNodes[node.name] = create_mpsgraph_constant_tensor(
                    tensor=attr, mpsGraph=mpsGraph
                )

            # Handle inputs to the graph.
            elif node.op == "placeholder":
                # Check if this is a lifted parameter / buffer
                # If so, bundle the constants in the graph instead of creating placeholders
                lifted_param_or_buffer = get_param_from_node(node, edge_program)
                if lifted_param_or_buffer is not None:
                    graphNodes[node.name] = create_mpsgraph_constant_tensor(
                        tensor=lifted_param_or_buffer, mpsGraph=mpsGraph
                    )
                else:
                    if node.meta["val"] is None:
                        continue
                    shape = MPSBackend.eval_shape(node.meta["val"])
                    if shape is None:
                        graphNodes[node.name] = mpsGraph.mpsGraphUnrankedPlaceHolder(
                            get_mps_data_type(node.meta["val"].dtype)
                        )
                    else:
                        graphNodes[node.name] = mpsGraph.mpsGraphRankedPlaceHolder(
                            get_mps_data_type(node.meta["val"].dtype), shape
                        )

            # Handle `call_function` calls.
            elif node.op == "call_function":
                if node.target == exir_ops.edge.aten.mm.default:
                    graphNodes[node.name] = mpsGraph.mm(
                        graphNodes[node.args[0].name], graphNodes[node.args[1].name]
                    )
                elif node.target == exir_ops.edge.aten.bmm.default:
                    graphNodes[node.name] = mpsGraph.bmm(
                        graphNodes[node.args[0].name], graphNodes[node.args[1].name]
                    )
                elif node.target == exir_ops.edge.aten.add.Tensor:
                    alpha = 1.0
                    if node.kwargs and node.kwargs["alpha"] is not None:
                        alpha = node.kwargs["alpha"]
                    graphNodes[node.name] = mpsGraph.add(
                        graphNodes[node.args[0].name],
                        graphNodes[node.args[1].name],
                        alpha,
                    )
                elif node.target == exir_ops.edge.aten.add.Scalar:
                    graphNodes[node.name] = mpsGraph.add(
                        graphNodes[node.args[0].name], node.args[1]
                    )
                elif node.target == exir_ops.edge.aten.sub.Tensor:
                    alpha = 1.0
                    if node.kwargs and node.kwargs["alpha"] is not None:
                        alpha = node.kwargs["alpha"]
                    graphNodes[node.name] = mpsGraph.sub(
                        graphNodes[node.args[0].name],
                        graphNodes[node.args[1].name],
                        alpha,
                    )
                elif node.target == exir_ops.edge.aten.sub.Scalar:
                    graphNodes[node.name] = mpsGraph.sub(
                        graphNodes[node.args[0].name], node.args[1]
                    )
                elif node.target == exir_ops.edge.aten.mul.Tensor:
                    graphNodes[node.name] = mpsGraph.mul(
                        graphNodes[node.args[0].name], graphNodes[node.args[1].name]
                    )
                elif node.target == exir_ops.edge.aten.mul.Scalar:
                    graphNodes[node.name] = mpsGraph.mulWithScalar(
                        graphNodes[node.args[0].name], node.args[1]
                    )
                elif node.target in binaryOps:
                    graphNodes[node.name] = binaryOps[node.target](
                        graphNodes[node.args[0].name], graphNodes[node.args[1].name]
                    )
                elif node.target in binaryOpsWithScalar:
                    graphNodes[node.name] = binaryOpsWithScalar[node.target](
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.full.default:
                    if len(node.args) < 2:
                        raise AssertionError(
                            "Full op requires at least size & fill_value args"
                        )
                    dtype = get_mps_data_type(torch.float32)
                    if len(node.args) >= 3:
                        dtype = get_mps_data_type(node.args[2])
                    if len(node.args) >= 4:
                        raise AssertionError("Unexpected number of input parameters")
                    graphNodes[node.name] = mpsGraph.full(
                        node.args[0], node.args[1], dtype
                    )

                elif node.target == exir_ops.edge.aten.full_like.default:
                    if len(node.args) < 2:
                        raise AssertionError("Too few input parameters")
                    graphNodes[node.name] = mpsGraph.full_like(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.convolution.default:
                    from typing import cast

                    input_node = cast(torch.fx.Node, node.args[0]).meta["val"]
                    weight_node = cast(torch.fx.Node, node.args[1]).meta["val"]
                    groups = int(node.args[8])

                    # Convolution is depthwise if groups = input channels and output channel
                    # is a positive multiple of input channels
                    is_depthwise_conv = (groups > 1 and weight_node.size(1) == 1) and (
                        input_node.dim() >= 4 and weight_node.dim() >= 4
                    )

                    if node.args[2] is None:
                        graphNodes[node.name] = mpsGraph.conv2D(
                            graphNodes[node.args[0].name],
                            graphNodes[node.args[1].name],
                            None,
                            node.args[3],
                            node.args[4],
                            node.args[5],
                            node.args[6],
                            node.args[7],
                            node.args[8],
                            is_depthwise_conv,
                        )
                    else:
                        graphNodes[node.name] = mpsGraph.conv2D(
                            graphNodes[node.args[0].name],
                            graphNodes[node.args[1].name],
                            graphNodes[node.args[2].name],
                            node.args[3],
                            node.args[4],
                            node.args[5],
                            node.args[6],
                            node.args[7],
                            node.args[8],
                            is_depthwise_conv,
                        )

                elif node.target == exir_ops.edge.aten.max_pool2d_with_indices.default:
                    n_args = len(node.args)
                    if n_args > 6:
                        raise AssertionError("Unexpected number of input parameters")

                    padding = [0, 0]
                    dilation = [1, 1]
                    ceil_mode = False
                    if n_args >= 4:
                        padding = node.args[3]
                    if n_args >= 5:
                        dilation = node.args[4]
                    if n_args == 6:
                        ceil_mode = node.args[5]

                    graphNodes[node.name] = mpsGraph.maxPool2DWithIndices(
                        graphNodes[node.args[0].name],
                        node.args[1],
                        node.args[2],
                        padding,
                        dilation,
                        ceil_mode,
                    )

                elif node.target == exir_ops.edge.aten.avg_pool2d.default:
                    stride = node.args[1]
                    padding = [0, 0]
                    ceil_mode = False
                    count_include_pad = True
                    divisor_override = None

                    n_args = len(node.args)
                    if n_args >= 3:
                        stride = node.args[2]
                    if n_args >= 4:
                        padding = node.args[3]
                    if n_args >= 5:
                        ceil_mode = node.args[4]
                    if n_args >= 6:
                        count_include_pad = node.args[5]
                    if n_args == 7:
                        divisor_override = node.args[6]
                    if n_args > 7:
                        raise AssertionError("Unexpected number of arguments")

                    graphNodes[node.name] = mpsGraph.avgPool2D(
                        graphNodes[node.args[0].name],
                        node.args[1],
                        stride,
                        padding,
                        ceil_mode,
                        count_include_pad,
                        divisor_override,
                    )

                elif (
                    node.target
                    == exir_ops.edge.aten._native_batch_norm_legit_no_training.default
                ):
                    graphNodes[node.name] = mpsGraph.batchNorm(
                        graphNodes[node.args[0].name],
                        graphNodes[node.args[1].name],
                        graphNodes[node.args[2].name],
                        graphNodes[node.args[3].name],
                        graphNodes[node.args[4].name],
                        node.args[5],
                        node.args[6],
                    )

                elif node.target == exir_ops.edge.aten.native_layer_norm.default:
                    graphNodes[node.name] = mpsGraph.layerNorm(
                        graphNodes[node.args[0].name],
                        node.args[1],
                        graphNodes[node.args[2].name],
                        graphNodes[node.args[3].name],
                        node.args[4],
                    )

                elif node.target == exir_ops.edge.aten.hardtanh.default:
                    graphNodes[node.name] = mpsGraph.hardTanh(
                        graphNodes[node.args[0].name], node.args[1], node.args[2]
                    )

                elif node.target == exir_ops.edge.aten.relu.default:
                    graphNodes[node.name] = mpsGraph.relu(graphNodes[node.args[0].name])

                elif node.target == exir_ops.edge.aten.leaky_relu.default:
                    n_args = len(node.args)
                    if n_args > 2:
                        raise AssertionError("Unexpected number of input parameters")

                    negative_slope = 0.01
                    if n_args == 2:
                        negative_slope = node.args[1]
                    graphNodes[node.name] = mpsGraph.leaky_relu(
                        graphNodes[node.args[0].name], negative_slope
                    )

                elif node.target == exir_ops.edge.aten.gelu.default:
                    approximate = "none"
                    if len(node.args) > 1:
                        approximate = node.args[1]
                    graphNodes[node.name] = mpsGraph.gelu(
                        graphNodes[node.args[0].name], approximate
                    )

                elif node.target == exir_ops.edge.aten.glu.default:
                    dim = -1
                    if len(node.args) > 1:
                        dim = node.args[1]
                    graphNodes[node.name] = mpsGraph.glu(
                        graphNodes[node.args[0].name], dim
                    )

                elif node.target == exir_ops.edge.aten.index_select.default:
                    dim = node.args[1]
                    index = graphNodes[node.args[2].name]
                    graphNodes[node.name] = mpsGraph.index_select(
                        graphNodes[node.args[0].name], dim, index
                    )

                elif node.target == exir_ops.edge.aten._softmax.default:
                    graphNodes[node.name] = mpsGraph.softmax(
                        graphNodes[node.args[0].name], node.args[1], node.args[2]
                    )

                elif node.target == exir_ops.edge.aten._log_softmax.default:
                    graphNodes[node.name] = mpsGraph.log_softmax(
                        graphNodes[node.args[0].name], node.args[1], node.args[2]
                    )

                elif node.target == exir_ops.edge.aten._to_copy.default:
                    graphNodes[node.name] = mpsGraph.identity(
                        graphNodes[node.args[0].name]
                    )

                elif node.target == exir_ops.edge.aten.min.dim:
                    keep_dim = False
                    if len(node.args) == 3:
                        keep_dim = node.args[2]
                    graphNodes[node.name] = mpsGraph.minDim(
                        graphNodes[node.args[0].name], node.args[1], keep_dim
                    )

                elif node.target == exir_ops.edge.aten.max.dim:
                    keep_dim = False
                    if len(node.args) == 3:
                        keep_dim = node.args[2]
                    graphNodes[node.name] = mpsGraph.maxDim(
                        graphNodes[node.args[0].name], node.args[1], keep_dim
                    )

                elif node.target == exir_ops.edge.aten.amax.default:
                    if len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.amax(
                            graphNodes[node.args[0].name], node.args[1], False
                        )
                    elif len(node.args) == 3:
                        graphNodes[node.name] = mpsGraph.amax(
                            graphNodes[node.args[0].name], node.args[1], node.args[2]
                        )
                    else:
                        raise AssertionError("Unexpected number of input parameters")

                elif node.target == exir_ops.edge.aten.amin.default:
                    if len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.amin(
                            graphNodes[node.args[0].name], node.args[1], False
                        )
                    elif len(node.args) == 3:
                        graphNodes[node.name] = mpsGraph.amin(
                            graphNodes[node.args[0].name], node.args[1], node.args[2]
                        )
                    else:
                        raise AssertionError("Unexpected number of input parameters")

                elif node.target == exir_ops.edge.aten.argmax.default:
                    n_args = len(node.args)
                    if n_args == 1:
                        graphNodes[node.name] = mpsGraph.argmax(
                            graphNodes[node.args[0].name], 0, False, True
                        )
                    elif len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.argmax(
                            graphNodes[node.args[0].name], node.args[1], False, False
                        )
                    elif len(node.args) == 3:
                        graphNodes[node.name] = mpsGraph.argmax(
                            graphNodes[node.args[0].name],
                            node.args[1],
                            node.args[2],
                            False,
                        )
                    else:
                        raise AssertionError("Unexpected number of input parameters")

                elif node.target == exir_ops.edge.aten.argmin.default:
                    n_args = len(node.args)
                    if n_args == 1:
                        graphNodes[node.name] = mpsGraph.argmin(
                            graphNodes[node.args[0].name], 0, False, True
                        )
                    elif len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.argmin(
                            graphNodes[node.args[0].name], node.args[1], False, False
                        )
                    elif len(node.args) == 3:
                        graphNodes[node.name] = mpsGraph.argmin(
                            graphNodes[node.args[0].name],
                            node.args[1],
                            node.args[2],
                            False,
                        )
                    else:
                        raise AssertionError("Unexpected number of input parameters")

                elif node.target == exir_ops.edge.aten.mean.dim:
                    if len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.mean(
                            graphNodes[node.args[0].name], node.args[1], False
                        )
                    elif len(node.args) == 3:
                        graphNodes[node.name] = mpsGraph.mean(
                            graphNodes[node.args[0].name], node.args[1], node.args[2]
                        )
                    else:
                        raise AssertionError("Unexpected number of input parameters")

                elif node.target == exir_ops.edge.aten.pixel_shuffle.default:
                    torch._assert(
                        len(node.args) == 2, "Unexpected number of input parameters"
                    )
                    graphNodes[node.name] = mpsGraph.pixel_shuffle(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.split_with_sizes_copy.default:
                    dim = 0
                    torch._assert(
                        len(node.args) >= 2, "Unexpected number of input parameters"
                    )
                    split_sizes = node.args[1]
                    if len(node.args) >= 2:
                        dim = node.args[2]
                    graphNodes[node.name] = mpsGraph.split_size(
                        graphNodes[node.args[0].name], split_sizes, dim
                    )

                elif node.target == exir_ops.edge.aten.split_copy.Tensor:
                    dim = 0
                    torch._assert(
                        len(node.args) >= 2, "Unexpected number of input parameters"
                    )
                    split_sizes = node.args[1]
                    if len(node.args) >= 3:
                        dim = node.args[2]
                    graphNodes[node.name] = mpsGraph.split(
                        graphNodes[node.args[0].name], split_sizes, dim
                    )

                elif node.target == exir_ops.edge.aten.unbind_copy.int:
                    dim = 0
                    if len(node.args) >= 2:
                        dim = node.args[1]
                    graphNodes[node.name] = mpsGraph.unbind(
                        graphNodes[node.args[0].name], dim
                    )

                elif node.target == exir_ops.edge.aten.stack.default:
                    stackTensors = []
                    dim = 0
                    if len(node.args) > 1:
                        dim = node.args[1]
                    for inputTensor in node.args[0]:
                        stackTensors.append(graphNodes[inputTensor.name])
                    graphNodes[node.name] = mpsGraph.stack(dim, *stackTensors)

                elif node.target == exir_ops.edge.aten.cat.default:
                    catTensors = []
                    dim = 0
                    if len(node.args) > 1:
                        dim = node.args[1]
                    for inputTensor in node.args[0]:
                        catTensors.append(graphNodes[inputTensor.name])
                    graphNodes[node.name] = mpsGraph.cat(dim, *catTensors)

                elif node.target == exir_ops.edge.aten.slice_copy.Tensor:
                    dim = 0
                    step = 1
                    start = None
                    end = None
                    if len(node.args) >= 2:
                        dim = node.args[1]
                    if len(node.args) >= 4:
                        end = node.args[3]
                        start = node.args[2]
                    if len(node.args) >= 5:
                        step = node.args[4]
                    graphNodes[node.name] = mpsGraph.slice(
                        graphNodes[node.args[0].name], dim, start, end, step
                    )

                elif node.target == exir_ops.edge.aten.expand_copy.default:
                    graphNodes[node.name] = mpsGraph.expand(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.view_copy.default:
                    graphNodes[node.name] = mpsGraph.view(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.clone.default:
                    # Per exir documentation the executorch memory format and layout is still WIP
                    # So we'll assume the memory layout option of clone is to be ignored
                    # TODO: adjust once the memory format and layout have been finalized
                    graphNodes[node.name] = mpsGraph.identity(
                        graphNodes[node.args[0].name]
                    )

                elif node.target == exir_ops.edge.aten.select_copy.int:
                    idx = torch.sym_int(node.args[2])
                    graphNodes[node.name] = mpsGraph.select(
                        graphNodes[node.args[0].name], node.args[1], idx
                    )

                elif node.target == exir_ops.edge.aten.permute_copy.default:
                    graphNodes[node.name] = mpsGraph.permute(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.squeeze_copy.default:
                    graphNodes[node.name] = mpsGraph.squeeze(
                        graphNodes[node.args[0].name]
                    )

                elif node.target == exir_ops.edge.aten.squeeze_copy.dim:
                    graphNodes[node.name] = mpsGraph.squeeze(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.squeeze_copy.dims:
                    graphNodes[node.name] = mpsGraph.squeeze(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.unsqueeze_copy.default:
                    graphNodes[node.name] = mpsGraph.unsqueeze(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target == exir_ops.edge.aten.constant_pad_nd.default:
                    padding = node.args[1]
                    c_value = node.args[2]
                    graphNodes[node.name] = mpsGraph.constant_pad_nd(
                        graphNodes[node.args[0].name], padding, c_value
                    )

                elif node.target == exir_ops.edge.aten.addmm.default:
                    beta = 1.0
                    alpha = 1.0
                    if len(node.args) == 4:
                        beta = node.args[3]
                    if len(node.args) == 5:
                        alpha = node.args[4]
                    graphNodes[node.name] = mpsGraph.addmm(
                        graphNodes[node.args[0].name],
                        graphNodes[node.args[1].name],
                        graphNodes[node.args[2].name],
                        beta,
                        alpha,
                    )
                elif node.target == exir_ops.edge.aten.clamp.default:
                    min_value = 0.0
                    use_min = False
                    if len(node.args) >= 2 and node.args[1] is not None:
                        min_value = node.args[1]
                        use_min = True

                    max_value = 1.0
                    use_max = False
                    if len(node.args) >= 3 and node.args[2] is not None:
                        max_value = node.args[2]
                        use_max = True
                    graphNodes[node.name] = mpsGraph.clamp(
                        graphNodes[node.args[0].name],
                        min_value,
                        max_value,
                        use_min,
                        use_max,
                    )

                elif node.target == exir_ops.edge.aten.cumsum.default:
                    if len(node.args) != 2:
                        raise AssertionError("Unexpected number of input parameters")
                    graphNodes[node.name] = mpsGraph.cumsum(
                        graphNodes[node.args[0].name], node.args[1]
                    )

                elif node.target in unaryOps:
                    graphNodes[node.name] = unaryOps[node.target](
                        graphNodes[node.args[0].name]
                    )

                elif node.target == exir_ops.edge.aten.arange.start_step:
                    step = 1
                    if len(node.args) > 2 and node.args[2] is not None:
                        step = node.args[2]
                    dtype = get_mps_data_type(node.meta["val"].dtype)
                    shape = node.meta["val"].shape[0]
                    graphNodes[node.name] = mpsGraph.arange(
                        node.args[0], node.args[1], step, dtype, shape
                    )

                elif node.target == exir_ops.edge.aten.where.self:
                    graphNodes[node.name] = mpsGraph.where(
                        graphNodes[node.args[0].name],
                        graphNodes[node.args[1].name],
                        graphNodes[node.args[2].name],
                    )
                elif node.target == exir_ops.edge.aten.scalar_tensor.default:
                    graphNodes[node.name] = mpsGraph.scalar_out(
                        node.args[0], get_mps_data_type(node.meta["val"].dtype)
                    )

                elif node.target == exir_ops.edge.aten.empty.memory_format:
                    dtype = get_mps_data_type(torch.float32)
                    if len(node.args) >= 2:
                        dtype = get_mps_data_type(node.args[1])
                    graphNodes[node.name] = mpsGraph.empty(node.args[0], dtype)
                elif node.target == exir_ops.edge.aten.embedding.default:
                    if len(node.args) == 2:
                        graphNodes[node.name] = mpsGraph.index_select(
                            graphNodes[node.args[0].name],
                            0,
                            graphNodes[node.args[1].name],
                        )
                    elif len(node.args) > 2 and node.args[2] is not None:
                        r1 = mpsGraph.unsqueeze(
                            mpsGraph.ne(graphNodes[node.args[1].name], node.args[2]), -1
                        )
                        r2 = mpsGraph.index_select(
                            graphNodes[node.args[0].name],
                            0,
                            graphNodes[node.args[1].name],
                        )
                        graphNodes[node.name] = mpsGraph.where(
                            r1, r2, mpsGraph.full_like(r2, 0)
                        )
                # Cant check for target with getitem
                # Arg[0] target node, arg[1] target index
                elif "getitem" in node.name:
                    graphNodes[node.name] = graphNodes[node.args[0].name][node.args[1]]
                else:
                    raise AssertionError(f"Unknown op: {node.target}")

            # Handle `call_method` calls.
            elif node.op == "call_method":
                raise AssertionError("Not yet implemented")

            # Handle `call_method` calls.
            elif node.op == "call_module":
                raise AssertionError("Not yet implemented")

            # Handle output nodes in the graph.
            elif node.op == "output":
                output_nodes = []
                for i in range(len(node.args)):
                    for j in range(len(node.args[i])):
                        output_nodes.append(graphNodes[node.args[i][j].name])
                mpsGraph.set_outputs(*output_nodes)
            else:
                torch._assert(
                    False,
                    f"Unsupported operator: {node.op}, {node.name}, {node.target}",
                )

        mpsGraphExecutableBytes = mpsGraph.serialize()
        return PreprocessResult(processed_bytes=bytes(mpsGraphExecutableBytes))
