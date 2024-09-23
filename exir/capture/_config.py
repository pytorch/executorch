# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.pass_manager import PassType
from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.tracer import ExirDynamoConfig
from torch.fx._compatibility import compatibility


@compatibility(is_backward_compatible=False)
@dataclass
class CaptureConfig:
    pt2_mode: bool = True
    enable_functionalization: bool = True
    enable_dynamic_shape: bool = False  # This flag does nothing if enable_aot is True
    enable_aot: bool = (
        False  # When it's true it implies automatic dynamic shapes via default dynamo config
    )
    _dynamo_config: "ExirDynamoConfig" = field(default_factory=ExirDynamoConfig)
    _unlift: bool = False  # This flag does nothing if enable_aot is False.
    _use_old_decomp_table: bool = False


@compatibility(is_backward_compatible=False)
@dataclass
class EdgeCompileConfig:
    # TODO(qihan): remove ability to opt out
    _check_ir_validity: bool = True
    # TODO(larryliu): remove this
    _use_edge_ops: bool = True
    # Allow core ATen ops check to be skipped for certain ops, but continue with the rest of the checks.
    _core_aten_ops_exception_list: List[torch._ops.OpOverload] = field(
        default_factory=list
    )
    _skip_type_promotion: bool = False
    # TODO(gasoonjia): remove this
    # TODO(T192537614): reenanle dim order as default
    _skip_dim_order: bool = True


@compatibility(is_backward_compatible=False)
@dataclass
class ExecutorchBackendConfig:
    passes: List[PassType] = field(default_factory=list)

    # A single memory planning pass can be defined for all the programs in the
    # EdgeProgramManager or can be defined per program.
    memory_planning_pass: Union[PassType, Dict[str, PassType]] = MemoryPlanningPass()
    to_out_var_pass: PassType = ToOutVarPass(ignore_to_out_var_failure=False)
    dynamic_memory_planning_mode: DynamicMemoryPlanningMode = (
        DynamicMemoryPlanningMode.UPPER_BOUND
    )
    emit_stacktrace: bool = False

    # Whether to move delegate data blobs from the Program into separate
    # segments, rather than encoding those blobs in the flatbuffer data.
    # This makes it possible to free those blobs at runtime.
    extract_delegate_segments: bool = True

    # When extracting segments, the starting offset of each segment will be
    # aligned to this value (in bytes). Must be a power of two.
    segment_alignment: int = 128

    # If provided, the minimum alignment of tensor buffers in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    constant_tensor_alignment: Optional[int] = None

    # If provided, the minimum alignment of delegate data in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    delegate_alignment: Optional[int] = None

    # A single sym shape eval pass can be defined for all the programs in the
    # EdgeProgramManager or can be defined per program.
    sym_shape_eval_pass: Union[PassType, Dict[str, PassType]] = (
        ConstraintBasedSymShapeEvalPass()
    )

    # If set to true, view_copy operations will be converted to lightweight
    # view operations in the ET runtime
    remove_view_copy: bool = True
