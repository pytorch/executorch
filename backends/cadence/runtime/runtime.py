# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import numbers
import os
import tempfile
from typing import Any, Optional, Sequence, Tuple, Union

import executorch.exir.schema as et_schema

import numpy as np
import torch

from executorch.backends.cadence.runtime import utils
from executorch.backends.cadence.runtime.executor import Executor
from executorch.devtools import Inspector
from executorch.exir import ExecutorchProgramManager
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import DataLocation

from numpy import ndarray

from torch.utils._pytree import TreeSpec


class JarvisETDump:
    def __init__(self, output_dir: str) -> None:
        self.tensor_dump_dir: str = os.path.join(output_dir, "tensors")
        self.etdump_path: str = os.path.join(output_dir, "etdump.etdp")
        self.etrecord_path: Optional[str] = os.path.join(output_dir, "etrecord.bin")
        self.debug_buffer_path: Optional[str] = os.path.join(
            output_dir, "debug_output.bin"
        )

        if not os.path.exists(self.etdump_path):
            raise RuntimeError(f"{self.etdump_path} does not exist")
        # pyre-ignore[6]: os.path.exists expects str, but got Optional[str]
        if not os.path.exists(self.etrecord_path):
            logging.warning(
                "ETRecord not found, intermediate tensors will not be dumped"
            )
            self.etrecord_path = None
        # pyre-ignore[6]: os.path.exists expects str, but got Optional[str]
        if not os.path.exists(self.debug_buffer_path):
            logging.warning(
                "Debug buffer not found, intermediate tensors will not be dumped"
            )
            self.debug_buffer_path = None

        self.et_inspector: Inspector = Inspector(
            etdump_path=self.etdump_path,
            debug_buffer_path=self.debug_buffer_path,
            etrecord=self.etrecord_path,
        )

    def get_outputs(self, log_to_stdout: bool = False) -> Tuple[torch.Tensor]:
        output = [
            event_block.run_output
            for event_block in self.et_inspector.event_blocks
            if event_block.name == "Execute"
        ]
        logging.debug(f"[Jarvis][ETdump] output: {output}")
        return output[0]

    def print_event_block(self) -> None:
        logging.debug("[Jarvis][ETdump] data tabular:")
        if logging.getLogger().level <= logging.DEBUG:
            self.et_inspector.print_data_tabular()

    def print_event_data(self) -> None:
        logging.debug("[Jarvis][ETdump] event data ")
        for event_block in self.et_inspector.event_blocks:
            for event in event_block.events:
                logging.debug(event)

    def dump_intermediate_tensors(self) -> None:
        if self.etrecord_path is None:
            logging.info("[Jarvis][ETdump] Intermediate tensors not available")
            return

        logging.info(
            f"[Jarvis][ETdump] Dumping intermediate tensors to {self.tensor_dump_dir}"
        )
        os.makedirs(self.tensor_dump_dir, exist_ok=True)
        exec_blocks = [
            eb for eb in self.et_inspector.event_blocks if eb.name == "Execute"
        ]
        if len(exec_blocks) > 1:
            logging.warning(
                f'Found {len(exec_blocks)} "Execute" blocks, using the first one and ignoring the rest.'
            )
        block = exec_blocks[0]

        # OPERATOR_CALL events are duplicates that contain framework tax data. We don't need them
        op_events = [e for e in block.events if e.name != "OPERATOR_CALL"]
        torch.set_printoptions(profile="full")

        for event in op_events:
            instr_id = event._instruction_id
            if not event.debug_data:
                logging.debug(
                    f"Missing intermediate tensor data for {event.name} ({instr_id=})"
                )
                continue

            with open(f"{self.tensor_dump_dir}/{instr_id}.txt", "w") as f:
                for dd in event.debug_data:
                    f.write(f"{str(dd)}\n\n")
        torch.set_printoptions(profile="default")


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
                )
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

    # initialize Jarvis e2e Executor with executorch_cfg.
    executor = Executor(working_dir)

    # run Executor
    executor()

    etdump = JarvisETDump(output_dir=working_dir)
    outputs = etdump.get_outputs()

    assert isinstance(out_spec, TreeSpec)
    outputs = torch.utils._pytree.tree_unflatten(outputs, out_spec)

    return outputs


def compare(
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    outputs: Any,
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
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


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
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
