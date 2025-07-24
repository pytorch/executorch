# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import os
import tempfile

import uuid

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config

from executorch.devtools import BundledProgram, generate_etrecord
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.exir import to_edge

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,  # @manual
)
from torch.export import export


def _generate_new_paths():
    temp_dir = tempfile.mkdtemp()

    # Use uuid to generate unique filenames
    etrecord_filename = f"etrecord_{uuid.uuid4().hex}.bin"
    etdump_filename = f"etdump_{uuid.uuid4().hex}.etdp"
    debug_buffer_filename = f"debug_buffer_{uuid.uuid4().hex}.bin"
    etrecord_path = os.path.join(temp_dir, etrecord_filename)
    etdump_path = os.path.join(temp_dir, etdump_filename)
    debug_buffer_path = os.path.join(temp_dir, debug_buffer_filename)
    return etrecord_path, etdump_path, debug_buffer_path


def generate_etrecord_and_etdump(
    model,
    model_inputs,
    debug_buffer_size=1024 * 1024 * 1024,
    method_name="forward",
    num_test_cases=2,
    disturb=False,
):
    """
    Helper to generate ETRecord and ETDump (with debug buffer) for a model.

    Returns:
        Tuple of (etrecord_path, etdump_path, debug_buffer_path)
    """

    etrecord_path, etdump_path, debug_buffer_path = _generate_new_paths()

    aten_model = export(model, model_inputs, strict=True)

    edge_compile_config = get_xnnpack_edge_compile_config()

    edge_program_manager = to_edge(aten_model, compile_config=edge_compile_config)

    edge_program_manager_copy = copy.deepcopy(edge_program_manager)

    # Apply the disturbance if the flag is set
    if disturb:
        import torch

        for _, exported_program in edge_program_manager_copy._edge_programs.items():
            for module in exported_program.graph_module.modules():
                if not isinstance(module, torch.fx.GraphModule):
                    continue
                for node in module.graph.nodes:
                    if node.op == "call_function" and node.name == "aten_add_tensor":
                        node.target = torch.ops.aten.sub.Tensor
                module.recompile()
                module.graph.eliminate_dead_code()

    edge_program_manager = edge_program_manager.to_backend(XnnpackPartitioner())

    et_program_manager = edge_program_manager.to_executorch()

    method_graphs = {method_name: export(model, model_inputs, strict=True)}
    inputs = [list(model_inputs) for _ in range(num_test_cases)]
    method_test_suites = [
        MethodTestSuite(
            method_name=method_name,
            test_cases=[
                MethodTestCase(
                    inputs=inp, expected_outputs=getattr(model, method_name)(*inp)
                )
                for inp in inputs
            ],
        )
    ]
    executorch_program = (
        to_edge(method_graphs, compile_config=edge_compile_config)
        .to_backend(XnnpackPartitioner())
        .to_executorch()
    )
    bundled_program = BundledProgram(executorch_program, method_test_suites)

    # Generate ETRecord
    generate_etrecord(etrecord_path, edge_program_manager_copy, bundled_program)

    # Generate ETDump and debug buffer
    buff = et_program_manager.buffer
    executorch_module = _load_for_executorch_from_buffer(
        buff,
        enable_etdump=True,
        debug_buffer_size=debug_buffer_size,
    )
    executorch_module.run_method(method_name, tuple(model_inputs))
    executorch_module.write_etdump_result_to_file(etdump_path, debug_buffer_path)

    return etrecord_path, etdump_path, debug_buffer_path


from typing import Tuple

import pandas as pd
from executorch.devtools import Inspector


def check_numeric_gap(
    etdump_path: str,
    etrecord_path: str,
    debug_buffer_path: str,
    metric: str,
    max_allowed_gap: float,
) -> Tuple[bool, float]:
    """
    Create an Inspector and check if the maximum numeric gap for a given metric is less than the allowed threshold.
    Args:
        etdump_path: Path to the ETDump file.
        etrecord_path: Path to the ETRecord file.
        debug_buffer_path: Path to the debug buffer file.
        metric: The metric name to calculate the numeric gap for (e.g., "MSE").
        max_allowed_gap: The maximum allowed gap threshold.
    Returns:
        A tuple (is_within_threshold, max_gap) where:
        - is_within_threshold (bool): True if max gap < max_allowed_gap, else False.
        - max_gap (float): The maximum gap value found.
    """
    inspector = Inspector(
        etdump_path=etdump_path,
        etrecord=etrecord_path,
        debug_buffer_path=debug_buffer_path,
    )
    df: pd.DataFrame = inspector.calculate_numeric_gap(metric)
    max_gap = df["gap"].apply(lambda x: max(x) if isinstance(x, list) else x).max()
    is_within_threshold = max_gap < max_allowed_gap
    return is_within_threshold, max_gap


def check_disturbance(
    etdump_path: str,
    etrecord_path: str,
    debug_buffer_path: str,
    metric: str,
    row: int,
    max_allowed_gap: float,
    disturbance_threshold: float,
) -> bool:
    """
    Check if the given row in the DataFrame has a gap greater than the disturbance threshold.

    Args:
        etdump_path: Path to the ETDump file.
        etrecord_path: Path to the ETRecord file.
        debug_buffer_path: Path to the debug buffer file.
        metric: The metric name to calculate the numeric gap for (e.g., "MSE").
        disturbance_threshold: The threshold to detect a disturbance.
        max_allowed_gap: The maximum allowed gap threshold before the disturbance(row).
        row: The row number to check for a disturbance.
    """
    inspector = Inspector(
        etdump_path=etdump_path,
        etrecord=etrecord_path,
        debug_buffer_path=debug_buffer_path,
    )
    df: pd.DataFrame = inspector.calculate_numeric_gap(metric)

    # Get the maximum gap for the given row
    disturbance_row_gap = max(df.loc[row, "gap"])
    # Get the maximum gap for the rows before the given row
    if row > 0:
        before_disturbance_row_gap = max(df.loc[: row - 1, "gap"].apply(max))
    else:
        before_disturbance_row_gap = 0

    return (
        disturbance_row_gap > disturbance_threshold
        and before_disturbance_row_gap < max_allowed_gap
    )
