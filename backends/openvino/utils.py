# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Any, Dict, Optional, Tuple, Union

import executorch.exir as exir

import torch
from executorch.exir import EdgeProgramManager
from executorch.exir.program._program import to_edge_with_preserved_ops
from executorch.exir.tracer import Value
from torch.export import ExportedProgram
from executorch.extension.export_util.utils import _to_core_aten

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
    _skip_dim_order=True,  # TODO(T189114319): Reuse dim order op after solving the ios oss issue
)

def nncf_core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=None,
    verbose=True,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge_with_preserved_ops(
        core_aten_exir_ep,
        constant_methods=edge_constant_methods,
        compile_config=edge_compile_config,
        preserve_ops=[torch.ops.aten.stack.default,],
    )
    if verbose:
        logging.info(f"Exported graph:\n{edge_manager.exported_program()}")
    return edge_manager

def nncf_export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    *,
    example_kwarg_inputs: Optional[Dict] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    strict=True,
    verbose=True,
) -> EdgeProgramManager:
    core_aten_ep = _to_core_aten(
        model,
        example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
        verbose=verbose,
    )
    return nncf_core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config, verbose=verbose
    )
