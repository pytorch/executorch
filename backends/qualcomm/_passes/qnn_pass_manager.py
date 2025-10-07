# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections import OrderedDict
from typing import Dict

from executorch.backends.qualcomm._passes import (
    AnnotateAdaptiveAvgPool1D,
    AnnotateQuantAttrs,
    AnnotateStack,
    AnnotateUnbind,
    CanonicalizeConv,
    ConvertBmmToMatmul,
    ConvertLinearToConv2d,
    ConvertSquareToPow,
    DecomposeAny,
    DecomposeBinaryAlpha,
    DecomposeCDist,
    DecomposeColIm,
    DecomposeEinsum,
    DecomposeExpM1,
    DecomposeLinalgVectorNorm,
    DecomposeMinMaxDim,
    DecomposeRoll,
    DecomposeSilu,
    DecomposeThreshold,
    DecomposeWrapWithAutocast,
    ExpandBroadcastTensorShape,
    FixedLinearKeepDim,
    FoldQDQ,
    FuseConsecutiveCast,
    FuseConsecutiveTranspose,
    I64toI32,
    InsertIOQDQ,
    InsertRequantize,
    InsertReshapeForReduceOps,
    LayoutTransform,
    LiftConstantScalarOperands,
    RecomposePixelUnshuffle,
    RecomposeRmsNorm,
    ReduceDynamicRange,
    Remove0DTensor,
    RemoveRedundancy,
    ReplaceArangeArgs,
    ReplaceInfValues,
    TagQuantIO,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.exir import ExportedProgram
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import (
    _get_updated_graph_signature,
    lift_constant_tensor_pass,
)
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import this_before_that_pass_constraint


def get_capture_program_passes():
    """
    Defines and returns the default ordered passes for the capture program.
    This function creates an OrderedDict containing a series of default passes.

    Returns:
        OrderedDict: An ordered dictionary containing all default passes along with their activation status and initialization parameters.
    """

    # The second value in each tuple in `default_passes_and_setting` indicates whether the corresponding pass is activated by default.
    # If a pass is activated, it will be executed by default.
    default_passes_and_setting = [
        (AnnotateAdaptiveAvgPool1D, True),
        (AnnotateQuantAttrs, True),
        (AnnotateStack, True),
        (AnnotateUnbind, True),
        (CanonicalizeConv, True),
        (ConvertBmmToMatmul, False),
        (DecomposeAny, True),
        (DecomposeColIm, True),
        (DecomposeMinMaxDim, True),
        (ExpandBroadcastTensorShape, False),
        (FixedLinearKeepDim, True),
        (FoldQDQ, True),
        (I64toI32, True),
        (LayoutTransform, True),
        (RecomposePixelUnshuffle, True),
        (RecomposeRmsNorm, True),
        (Remove0DTensor, True),
        (RemoveRedundancy, True),
        (TagQuantIO, False),
    ]

    passes = OrderedDict()
    for p, act in default_passes_and_setting:
        init_signature = inspect.signature(p.__init__)

        args_kwargs_defaults = {
            k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in init_signature.parameters.items()
            if k != "self"
        }

        passes[p] = {
            QCOM_PASS_ACTIVATE_KEY: act,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: args_kwargs_defaults,
        }

    return passes


class QnnPassManager(PassManager):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, graph_module: GraphModule):
        return self(graph_module).graph_module

    # TODO: Move these passes into qnn_partitioner and qnn_preprocess to
    # prevent users from needing to call custom APIs like capture_program
    def get_to_edge_transform_passes(
        self,
        exported_program: ExportedProgram,
        passes_job: OrderedDict = None,
        dep_table: Dict = None,
    ):
        # TODO: remove this workaround when target could be correctly detected
        from executorch.backends.qualcomm.builders import node_visitor
        from executorch.exir.dialects._ops import ops as exir_ops

        node_visitor.q_ops.add(exir_ops.edge.torchao.quantize_affine.default)
        node_visitor.dq_ops.add(exir_ops.edge.torchao.dequantize_affine.default)

        passes_job = (
            passes_job if passes_job is not None else get_capture_program_passes()
        )
        dep_table = (
            dep_table
            if dep_table is not None
            else get_passes_dependency_for_capture_program()
        )
        for that, these in dep_table.items():
            for this in these:
                self.add_constraint(this_before_that_pass_constraint(this, that))
        for p in passes_job:
            self.add_pass(p)
        self.solve_constraints()

        sorted_passes = self.passes
        self.passes = []
        for p in sorted_passes:
            if not passes_job[p][QCOM_PASS_ACTIVATE_KEY]:
                continue

            kwargs = passes_job[p][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
            if "edge_program" in kwargs:
                kwargs["edge_program"] = exported_program
            self.add_pass(p(**kwargs))
        return self.passes

    def transform_for_to_edge_pipeline(
        self,
        exported_program: ExportedProgram,
        passes_job: OrderedDict = None,
        dep_table: Dict = None,
    ):
        transform_passes = self.get_to_edge_transform_passes(
            exported_program, passes_job=passes_job, dep_table=dep_table
        )
        for p in transform_passes:
            p(exported_program.graph_module)
        exported_program._graph_signature = _get_updated_graph_signature(
            exported_program.graph_signature,
            exported_program.graph_module,
        )
        exported_program._validate()

        return exported_program

    # Before quantizer
    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        self.add_pass(RemoveRedundancy(quantization_capture=True))
        self.add_pass(ReduceDynamicRange())
        self.add_pass(RecomposePixelUnshuffle(quantization_capture=True))
        self.add_pass(RecomposeRmsNorm(quantization_capture=True))
        self.add_pass(ReplaceArangeArgs())
        self.add_pass(DecomposeBinaryAlpha())
        self.add_pass(DecomposeCDist())
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeRoll())
        self.add_pass(DecomposeSilu())
        self.add_pass(DecomposeThreshold())
        self.add_pass(DecomposeWrapWithAutocast())
        self.add_pass(DecomposeEinsum())
        self.add_pass(DecomposeExpM1())
        self.add_pass(DecomposeLinalgVectorNorm(quantization_capture=True))
        self.add_pass(ReplaceInfValues())
        self.add_pass(LiftConstantScalarOperands())
        self.add_pass(InsertReshapeForReduceOps())
        return self._transform(graph_module)

    def transform_for_export_pipeline(
        self, exported_program: ExportedProgram, convert_linear_to_conv2d: bool = False
    ):
        self.add_pass(DecomposeBinaryAlpha())
        self.add_pass(DecomposeCDist())
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeRoll())
        self.add_pass(DecomposeThreshold())
        self.add_pass(DecomposeLinalgVectorNorm(quantization_capture=True))
        self.add_pass(DecomposeExpM1())
        self.add_pass(DecomposeWrapWithAutocast())
        # this pass will rewrite state_dict, it needs to be accomplished before
        # to_edge_transform_and_lower
        self.add_pass(CanonicalizeConv(exported_program))
        if convert_linear_to_conv2d:
            self.add_pass(ConvertLinearToConv2d(exported_program))
        self.add_pass(ConvertSquareToPow())
        self.add_pass(LiftConstantScalarOperands())
        self.add_pass(InsertReshapeForReduceOps())
        self._transform(exported_program.graph_module)
        ep = lift_constant_tensor_pass(exported_program)
        return ep

    def transform_for_preprocess_pipeline(self, exported_program: ExportedProgram):
        self.add_pass(FoldQDQ(exported_program, force_fold=True))
        self.add_pass(InsertRequantize())
        self.add_pass(InsertIOQDQ(exported_program))
        self.add_pass(LayoutTransform(exported_program, insert_permute=True))
        self.add_pass(FuseConsecutiveCast())
        self.add_pass(FuseConsecutiveTranspose())
        self._transform(exported_program.graph_module)
        # Update inputs_to_buffers and buffers_to_mutate in graph signature for mutable buffer
        # Since I/O will be inserted Q/DQ, it results in failed to mapping output node names and buffer
        exported_program._graph_signature = _get_updated_graph_signature(
            exported_program.graph_signature,
            exported_program.graph_module,
        )
        return exported_program.graph_module
