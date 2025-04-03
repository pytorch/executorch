# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections import OrderedDict
from typing import Dict

from executorch.backends.qualcomm._passes import (
    AnnotateQuantAttrs,
    AnnotateStack,
    AnnotateUnbind,
    ConvertBmmToMatmul,
    ConvertConv1dToConv2d,
    DecomposeAny,
    DecomposeEinsum,
    DecomposeExpM1,
    DecomposeLinalgVectorNorm,
    DecomposeSilu,
    ExpandBroadcastTensorShape,
    FixedLinearKeepDim,
    FoldQDQ,
    FuseConsecutiveTranspose,
    I64toI32,
    InsertIOQDQ,
    InsertRequantize,
    LayoutTransform,
    LiftConstantScalarOperands,
    RecomposePixelUnshuffle,
    RecomposeRmsNorm,
    ReduceDynamicRange,
    RemoveRedundancy,
    ReplaceArangeArgs,
    ReplaceIndexPutInput,
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
        (AnnotateQuantAttrs, True),
        (AnnotateStack, False),
        (AnnotateUnbind, True),
        (ConvertBmmToMatmul, True),
        (ConvertConv1dToConv2d, True),
        (DecomposeAny, True),
        (ExpandBroadcastTensorShape, False),
        (FixedLinearKeepDim, True),
        (FoldQDQ, True),
        (I64toI32, True),
        (LayoutTransform, True),
        (RecomposePixelUnshuffle, True),
        (RecomposeRmsNorm, False),
        (RemoveRedundancy, True),
        (ReplaceIndexPutInput, True),
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
        from executorch.backends.qualcomm._passes import utils
        from executorch.exir.dialects._ops import ops as exir_ops

        utils.q_ops.add(exir_ops.edge.pt2e_quant.quantize_affine.default)
        utils.dq_ops.add(exir_ops.edge.pt2e_quant.dequantize_affine.default)

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

    def transform_for_export_pipeline(self, exported_program: ExportedProgram):
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeLinalgVectorNorm(quantization_capture=True))
        self.add_pass(DecomposeExpM1())
        self.add_pass(LiftConstantScalarOperands())
        self._transform(exported_program.graph_module)
        ep = lift_constant_tensor_pass(exported_program)
        return ep

    def transform_for_preprocess_pipeline(self, exported_program: ExportedProgram):
        self.add_pass(InsertRequantize())
        self.add_pass(InsertIOQDQ(exported_program))
        self.add_pass(LayoutTransform(exported_program, insert_permute=True))
        self.add_pass(FuseConsecutiveTranspose())
        return self._transform(exported_program.graph_module)

    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        self.add_pass(ReduceDynamicRange())
        self.add_pass(RecomposePixelUnshuffle(quantization_capture=True))
        self.add_pass(ReplaceArangeArgs())
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeSilu())
        self.add_pass(DecomposeEinsum())
        self.add_pass(DecomposeExpM1())
        self.add_pass(DecomposeLinalgVectorNorm(quantization_capture=True))
        self.add_pass(ReplaceInfValues())
        self.add_pass(LiftConstantScalarOperands())
        return self._transform(graph_module)
