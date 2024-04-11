# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.pass_manager import PassType
from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from executorch.exir.passes.sym_shape_eval_pass import HintBasedSymShapeEvalPass
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
    _skip_type_promotion: bool = False
    # TODO(gasoonjia): set it as False by default, and remove it in the long term
    _skip_dim_order: bool = True


@compatibility(is_backward_compatible=False)
@dataclass
class ExecutorchBackendConfig:
    passes: List[PassType] = field(default_factory=list)
    memory_planning_pass: PassType = MemoryPlanningPass("greedy")
    to_out_var_pass: PassType = ToOutVarPass(ignore_to_out_var_failure=False)
    dynamic_memory_planning_mode: DynamicMemoryPlanningMode = (
        DynamicMemoryPlanningMode.UPPER_BOUND
    )
    emit_stacktrace: bool = False

    # Whether to move delegate data blobs from the Program into separate
    # segments, rather than encoding those blobs in the flatbuffer data.
    # This makes it possible to free those blobs at runtime.
    extract_delegate_segments: bool = False

    # Whether to extract constants from the Program into separate segments,
    # rather than encoding those constants in the flatbuffer data.
    # This reduces the memory overhead of creating the .pte file for models with
    # large constant data.
    extract_constant_segment: bool = True

    # When extracting segments, the starting offset of each segment will be
    # aligned to this value (in bytes). Must be a power of two.
    segment_alignment: int = 4096

    # If provided, the minimum alignment of tensor buffers in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    constant_tensor_alignment: Optional[int] = None

    # If provided, the minimum alignment of delegate data in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    delegate_alignment: Optional[int] = None
    sym_shape_eval_pass: PassType = HintBasedSymShapeEvalPass()

    # If set to true, view_copy operations will be converted to lightweight
    # view operations in the ET runtime
    remove_view_copy: bool = True
