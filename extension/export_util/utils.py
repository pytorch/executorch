# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from typing import Any, Dict, Optional, Tuple, Union

import executorch.exir as exir

import torch
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.tracer import Value
from torch.export import export, export_for_training, ExportedProgram


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
    _skip_dim_order=True,  # TODO(T189114319): Reuse dim order op after solving the ios oss issue
)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    *,
    example_kwarg_inputs: Optional[Dict] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    strict=True,
    verbose=True,
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule) and not isinstance(
        model, torch.nn.Module
    ):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(
        model,
        example_inputs,
        example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
    )
    if verbose:
        logging.info(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=None,
    verbose=True,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep,
        constant_methods=edge_constant_methods,
        compile_config=edge_compile_config,
    )
    eg = torch.tensor([[2, 3, 4]], dtype=torch.int64)
    ip = torch.tensor([[0, 1, 2]], dtype=torch.long)
    em1 = edge_manager._edge_programs["forward"].module()(eg, input_pos=ip)
    eager = torch.load("/data/users/lfq/executorch/eager_res.pt")
    # breakpoint()
    assert torch.allclose(
        eager, em1
    )  # Fails to 100 lol. Passes if we do not decompose sdpa.
    if verbose:
        logging.info(f"Exported graph:\n{edge_manager.exported_program()}")
    return edge_manager


def export_to_edge(
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
    eg = torch.tensor([[2, 3, 4]], dtype=torch.int64)
    ip = torch.tensor([[0, 1, 2]], dtype=torch.long)
    x = model.forward(eg, input_pos=ip)
    eager = torch.load("/data/users/lfq/executorch/eager_res.pt")
    torch.allclose(x, eager)
    # breakpoint()  # same
    core_aten_ep = _to_core_aten(
        model,
        example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
        verbose=verbose,
    )
    # breakpoint()  # same
    y = core_aten_ep.module()(eg, input_pos=ip)
    torch.allclose(y, eager)
    return _core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config, verbose=verbose
    )


def export_to_exec_prog(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    *,
    example_kwarg_inputs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    backend_config=None,
    strict=True,
) -> ExecutorchProgramManager:
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    breakpoint()
    m = export_for_training(m, example_inputs).module()

    breakpoint()
    core_aten_ep = _to_core_aten(
        m,
        example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
    )

    breakpoint()
    edge_m = _core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config
    )

    breakpoint()
    exec_prog = edge_m.to_executorch(backend_config)

    breakpoint()
    return exec_prog


def save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> str:
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")

    try:
        # Write program to file.
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
        # Write data to file/s.
        prog.write_tensor_data_to_file(outdir=output_dir)
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

    return filename
