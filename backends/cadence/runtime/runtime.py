# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
import numbers
import tempfile
from typing import Any, Optional, Sequence, Union

import executorch.exir.schema as et_schema

import numpy as np
import torch

from executorch.backends.cadence.runtime import utils
from executorch.backends.cadence.runtime.etdump import CadenceETDump
from executorch.backends.cadence.runtime.executor import Executor
from executorch.exir import ExecutorchProgramManager
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import DataLocation

from numpy import ndarray

from torch.utils._pytree import TreeSpec


def get_op_names(program: et_schema.Program, execution_plan_id: int = 0) -> set[str]:
    """
    Get the list of operators from a Program
    """

    op_names = {
        f"{op.name}.{op.overload}"
        for op in program.execution_plan[execution_plan_id].operators
    }
    for delegate in program.execution_plan[execution_plan_id].delegates:
        logging.debug(f"Delegate: {delegate.id}")
        if delegate.id == "CadenceExecutorchBackend":
            assert delegate.processed.location == DataLocation.INLINE
            op_names |= get_op_names(
                deserialize_pte_binary(
                    program.backend_delegate_data[delegate.processed.index].data
                ).program
            )
    return op_names


# Run an ExecutorchProgram using the specified inputs and backend
def run(
    executorch_prog: ExecutorchProgramManager,
    inputs: Any,
    ref_outputs: Optional[Sequence[torch.Tensor]] = None,
    working_dir: Optional[str] = None,
) -> Any:
    # Get the Program
    program = executorch_prog.executorch_program
    out_spec = executorch_prog.exported_program().call_spec.out_spec
    # Run the program and return the outputs
    assert isinstance(
        program, et_schema.Program
    ), f"program must be Program. Got {type(program)} instead."

    if working_dir is None:
        working_dir = tempfile.mkdtemp(dir="/tmp")

    # initialize e2e Executor with executorch_cfg.
    executor = Executor(working_dir)

    # run Executor
    executor()

    etdump = CadenceETDump(output_dir=working_dir)
    outputs = etdump.get_outputs()

    # Print performance summary
    etdump.print_summary()

    assert isinstance(out_spec, TreeSpec)
    outputs = torch.utils._pytree.tree_unflatten(outputs, out_spec)

    return outputs


def compare(
    outputs: Any,
    ref_outputs: Any,
    name: str = "",
    eps_error: float = 1e-1,
    eps_warn: float = 1e-5,
) -> None:
    if isinstance(ref_outputs, dict):
        for k, v in outputs.items():
            compare(v, ref_outputs[k], f"{name}/{k}", eps_error, eps_warn)
        return

    if isinstance(ref_outputs, (list, tuple)):
        for i, (output, ref_output) in enumerate(zip(outputs, ref_outputs)):
            compare(output, ref_output, f"{name}/{i}", eps_error, eps_warn)
        return

    assert isinstance(ref_outputs, torch.Tensor), f"Got {type(ref_outputs)} instead."

    ref_outputs = to_nd_array(ref_outputs)
    outputs = to_nd_array(outputs)

    # compare
    rms = utils.rms(outputs, ref_outputs)
    norm_rms = utils.normalized_rms(outputs, ref_outputs)
    max_abs_diff = utils.max_abs_diff(outputs, ref_outputs)
    max_rel_diff = utils.max_rel_diff(outputs, ref_outputs)
    stats = (
        f"{rms = }, {norm_rms = }, {max_abs_diff = }, {max_rel_diff = :.2f}%, "
        f"{outputs.shape = }[{outputs.dtype}], {ref_outputs.shape = }[{ref_outputs.dtype}]"
    )

    if np.isnan(rms) or rms > eps_error:
        logging.error(f"\33[31m[Error]\33[0m Output {name} mismatched! {stats}")
        logging.error(f"Expected: {ref_outputs}\n")
        logging.error(f"Got instead: {outputs}\n")
        raise RuntimeError(f"\33[31m[Error]\33[0m Output {name} mismatched! {stats}")
    elif rms > eps_warn:
        logging.warning(f"\33[33m[Warning]\33[0m Output {name} mismatched!. {stats}")
    else:
        logging.info(f"\33[32m[Passed]\33[0m Output {name} matched. {stats}")


def run_and_compare(
    executorch_prog: ExecutorchProgramManager,
    inputs: Any,
    ref_outputs: Optional[Sequence[torch.Tensor]] = None,
    working_dir: Optional[str] = None,
    eps_error: float = 1e-1,
    eps_warn: float = 1e-5,
) -> Any:
    outputs = run(executorch_prog, inputs, ref_outputs, working_dir)
    compare(outputs, ref_outputs, eps_error=eps_error, eps_warn=eps_warn)


def to_nd_array(v: Union[bool, numbers.Number, ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v

    if isinstance(v, torch.Tensor):
        # If v was quantized, we compare its int representation.
        v = v.int_repr() if v.is_quantized else v
        return v.cpu().detach().numpy()

    if isinstance(v, (numbers.Number, bool)):
        return np.array([v])

    raise RuntimeError(f"Unknown type {type(v)}")
