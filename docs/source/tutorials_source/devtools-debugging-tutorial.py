# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Using the ExecuTorch Developer Tools to Debug a Model
========================

**Author:** `ExecuTorch Team <https://github.com/pytorch/executorch>`__
"""

######################################################################
# The `ExecuTorch Developer Tools <../devtools-overview.html>`__ is a set of tools designed to
# provide users with the ability to profile, debug, and visualize ExecuTorch
# models.
#
# This tutorial will show a full end-to-end flow of how to utilize the Developer Tools to debug a model
# by detecting numerical discrepancies between the original PyTorch model and the ExecuTorch model.
# This is particularly useful when working with delegated models (e.g., XNNPACK) where numerical
# precision may differ. Specifically, it will:
#
# 1. Generate the artifacts consumed by the Developer Tools (`ETRecord <../etrecord.html>`__, `ETDump <../etdump.html>`__).
# 2. Create an Inspector class consuming these artifacts along with debug data.
# 3. Utilize the Inspector's ``calculate_numeric_gap`` method to identify numerical discrepancies.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you'll first need to
# `Set up your ExecuTorch environment <../getting-started-setup.html>`__.
#

######################################################################
# Generate ETRecord with Representative Inputs
# ---------------------------------------------
#
# The first step is to generate an ``ETRecord`` with representative inputs.
# ``ETRecord`` contains model graphs and metadata for linking runtime results
# to the eager model. For debugging, we also need to store representative
# inputs that will be used to calculate reference outputs.
#
# In this tutorial, an example model (shown below) is used to demonstrate.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.devtools import generate_etrecord

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
from torch.export import export, ExportedProgram


# Generate Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

aten_model: ExportedProgram = export(model, (torch.randn(1, 1, 32, 32),), strict=True)

edge_program_manager: EdgeProgramManager = to_edge(
    aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
)
edge_program_manager_copy = copy.deepcopy(edge_program_manager)
et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()


# Generate ETRecord
etrecord_path = "etrecord.bin"
generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)

######################################################################
#
# .. warning::
#    Users should do a deepcopy of the output of ``to_edge()`` and pass in the
#    deepcopy to the ``generate_etrecord`` API. This is needed because the
#    subsequent call, ``to_executorch()``, does an in-place mutation and will
#    lose debug data in the process.
#

######################################################################
# Generate BundledProgram
# -----------------------
#
# Next step is to generate a ``BundledProgram``. ``BundledProgram`` packages
# the model with sample inputs and expected outputs for testing and debugging.

import torch
from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.exir import to_edge
from torch.export import export

# Step 1: ExecuTorch Program Export
m_name = "forward"
method_graphs = {m_name: export(model, (torch.randn(1, 1, 32, 32),), strict=True)}

# Step 2: Construct Method Test Suites
inputs = [[torch.randn(1, 1, 32, 32)] for _ in range(2)]

method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(inputs=inp, expected_outputs=getattr(model, m_name)(*inp))
            for inp in inputs
        ],
    )
]

# Step 3: Generate BundledProgram
executorch_program = to_edge(method_graphs).to_executorch()
bundled_program = BundledProgram(executorch_program, method_test_suites)

# Step 4: Serialize BundledProgram to flatbuffer.
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
    bundled_program
)
save_path = "bundled_program.bp"
with open(save_path, "wb") as f:
    f.write(serialized_bundled_program)

######################################################################
# Generate ETDump with Debug Buffer
# ----------------------------------
#
# Next, we generate an ``ETDump`` along with a debug buffer. The debug buffer
# stores intermediate outputs from the runtime execution, which are essential
# for numerical discrepancy analysis.
#
# Use CMake (follow `these instructions <../runtime-build-and-cross-compilation.html#configure-the-cmake-build>`__ to set up cmake) to execute the Bundled Program with debug output enabled::
#
#       cd executorch
#       ./examples/devtools/build_example_runner.sh
#       cmake-out/examples/devtools/example_runner --bundled_program_path="bundled_program.bp" --debug_output_path="debug_output.bin"
#
# .. note::
#    The debug buffer size should be large enough to hold all intermediate
#    outputs. For large models, allocate sufficient buffer space.
#    If the buffer is too small, some intermediate outputs may be truncated.
#

######################################################################
# Creating an Inspector for Debugging
# ------------------------------------
#
# Create the ``Inspector`` by passing in the artifact paths, including
# the debug buffer path. The Inspector will use these to correlate
# runtime results with the original model and compute numerical gaps.
#
# Recall: An ``ETRecord`` is not required. If an ``ETRecord`` is not provided,
# the Inspector will show runtime results without operator correlation.

from executorch.devtools import Inspector

# sphinx_gallery_start_ignore
from unittest.mock import patch

inspector_patch_print = patch.object(Inspector, "print_data_tabular", return_value=None)
inspector_patch_print.start()
# sphinx_gallery_end_ignore
etrecord_path = "etrecord.bin"
etdump_path = "etdump.etdp"
debug_buffer_path = "debug_output.bin"

# sphinx_gallery_start_ignore
inspector_patch = patch.object(Inspector, "__init__", return_value=None)
inspector_patch.start()
# sphinx_gallery_end_ignore

inspector = Inspector(
    etdump_path=etdump_path,
    etrecord=etrecord_path,
    debug_buffer_path=debug_buffer_path,
)

# sphinx_gallery_start_ignore
inspector_patch.stop()
# sphinx_gallery_end_ignore

######################################################################
# Calculating Numerical Discrepancies
# ------------------------------------
#
# The ``calculate_numeric_gap`` method computes the numerical difference
# between the eager model outputs and the runtime outputs for each operator.
# This is invaluable for identifying where precision loss occurs, especially
# in delegated subgraphs.
#
# The method supports several metrics:
#
# - ``MSE``: Mean Squared Error
# - ``SNR``: Signal-to-Noise Ratio
#
# .. code-block:: python
#
#    import pandas as pd
#
#    pd.set_option("display.width", 100000)
#    pd.set_option("display.max_columns", None)
#
#    # Calculate numerical gap using Mean Squared Error
#    df = inspector.calculate_numeric_gap("MSE")
#
# The returned DataFrame contains columns for each operator including:
#
# - ``event_name``: The operator name
# - ``mse``: The Mean Squared Error between eager and runtime outputs
# - ``op_type``: Whether the operator is delegated or native
# - ``stack_traces``: Source code location (if ETRecord was provided)
#
# Example output:
#
# .. code-block:: text
#
#    |   event_name                    |   mse        | op_type   |
#    |---------------------------------|--------------|-----------|
#    | aten_linear_default_0           | 1.23e-06     | delegated |
#    | aten_add_tensor_1               | 2.45e-07     | delegated |
#    | aten_layer_norm_default_2       | 5.67e-05     | delegated |
#

######################################################################
# Analyzing Discrepancies
# -----------------------
#
# Once you have the numerical gaps, you can identify operators with
# significant discrepancies and investigate further.
#
# .. code-block:: python
#
#    # Find operators with the largest discrepancies
#    if df is not None:
#        # Sort by MSE to find the most problematic operators
#        df_sorted = df.sort_values(by="mse", ascending=False)
#
#        print("Top 5 operators with largest numerical discrepancies:")
#        print(df_sorted.head(5))
#
#        # Filter for operators with MSE above a threshold
#        threshold = 1e-4
#        problematic_ops = df[df["mse"] > threshold]
#        print(f"\nOperators with MSE > {threshold}:")
#        print(problematic_ops)
#

######################################################################
# Debugging Specific Operators
# ----------------------------
#
# For detailed debugging, you can examine the intermediate outputs
# of specific operators using the EventBlock and Event classes.
#
# .. code-block:: python
#
#    for event_block in inspector.event_blocks:
#        for event in event_block.events:
#            # Access debug data (intermediate outputs) for each event
#            if event.debug_data is not None:
#                print(f"Operator: {event.name}")
#                print(f"  Debug data shape: {event.debug_data.shape}")
#                print(f"  Debug data dtype: {event.debug_data.dtype}")
#
#                # If ETRecord was provided, you can also see stack traces
#                if event.stack_traces:
#                    print(f"  Stack trace: {event.stack_traces}")
#
#                if event.module_hierarchy:
#                    print(f"  Module hierarchy: {event.module_hierarchy}")
#

######################################################################
# Comparing with Reference Outputs
# ---------------------------------
#
# You can also use the ``compare_results`` utility to perform quality
# analysis between runtime outputs and reference outputs.
#
# .. code-block:: python
#
#    from executorch.devtools.inspector import compare_results
#
#    # Get runtime outputs and compare with eager model outputs
#    for event_block in inspector.event_blocks:
#        if event_block.name == "Execute":
#            # Get runtime output
#            runtime_output = event_block.run_output
#
#            # Compute reference output from eager model
#            with torch.no_grad():
#                ref_output = model(*model_inputs)
#
#            if runtime_output is not None:
#                # Compare results (set plot=True to generate visualizations)
#                comparison = compare_results(
#                    runtime_output,
#                    [ref_output],
#                    plot=False,
#                )
#                print("Comparison results:", comparison)
#

######################################################################
# Best Practices for Debugging
# ----------------------------
#
# 1. **Start with model-level outputs**: First verify that the final model
#    output matches the eager model within acceptable tolerance.
#
# 2. **Use appropriate metrics**: MSE is good for general comparison,
#    while SNR is better for understanding relative error magnitudes.
#
# 3. **Focus on delegated operators**: Numerical discrepancies are most
#    common in delegated operators due to different precision handling.
#
# 4. **Check quantized operators**: If your model uses quantization,
#    expect larger numerical gaps in quantized operators.
#
# 5. **Use stack traces**: With ETRecord, you can trace operators back
#    to the original PyTorch source code for easier debugging.
#

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned how to use the ExecuTorch Developer Tools
# to debug numerical discrepancies in models. The key steps are:
#
# 1. Generate ETRecord during model export
# 2. Execute the model with a debug buffer to capture intermediate outputs
# 3. Use the Inspector's ``calculate_numeric_gap`` method to identify discrepancies
# 4. Analyze and debug specific operators using the DataFrame and event APIs
#
# Links Mentioned
# ^^^^^^^^^^^^^^^
#
# - `ExecuTorch Developer Tools Overview <../devtools-overview.html>`__
# - `ETRecord <../etrecord.html>`__
# - `ETDump <../etdump.html>`__
# - `Inspector <../model-inspector.html>`__
# - `Model Debugging Guide <../model-debugging.html>`__
# - `Profiling Tutorial <devtools-integration-tutorial.html>`__
