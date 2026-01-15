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
# You'll also need to build the ExecuTorch runtime with event tracer enabled::
#
#       cd executorch
#       cmake -DEXECUTORCH_ENABLE_EVENT_TRACER=ON ...
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
# We use ``to_edge_transform_and_lower`` with ``generate_etrecord=True`` to
# automatically capture the ETRecord during the lowering process.
#
# In this tutorial, we use a Vision Transformer (ViT) model delegated to
# XNNPACK to demonstrate debugging a real-world delegated model.

import os
import tempfile

import torch
from torchvision import models

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.devtools import Inspector

from executorch.exir import (
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from torch.export import export, ExportedProgram


# Create a temporary directory for artifacts
temp_dir = tempfile.mkdtemp()

# Create Vision Transformer model
vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
model = vit.eval()
model_inputs = (torch.randn(1, 3, 224, 224),)

# Export the model
aten_model: ExportedProgram = export(model, model_inputs, strict=True)

# Lower to edge with XNNPACK delegation and generate ETRecord
edge_program_manager = to_edge_transform_and_lower(
    aten_model,
    partitioner=[XnnpackPartitioner()],
    compile_config=get_xnnpack_edge_compile_config(),
    generate_etrecord=True,
)

et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()

# Get the ETRecord
etrecord = et_program_manager.get_etrecord()

# Set the input for numerical discrepancy detection
etrecord.update_representative_inputs(model_inputs)

# Save to target location
etrecord_path = os.path.join(temp_dir, "etrecord.bin")
etrecord.save(etrecord_path)

# Save the PTE file
pte_path = os.path.join(temp_dir, "model.pte")
et_program_manager.save(pte_path)

# sphinx_gallery_start_ignore
from unittest.mock import patch

# sphinx_gallery_end_ignore

######################################################################
#
# .. note::
#    The ``update_representative_inputs`` method is crucial for debugging.
#    It stores the inputs that will be used to compute reference outputs
#    from the eager model, which are then compared against the runtime outputs.
#

######################################################################
# Generate ETDump with Debug Buffer
# ----------------------------------
#
# Next, we generate an ``ETDump`` along with a debug buffer. The debug buffer
# stores intermediate outputs from the runtime execution, which are essential
# for numerical discrepancy analysis.
#
# When loading the program, we enable ETDump generation and allocate a debug
# buffer to capture intermediate tensors.

from executorch.runtime import Method, Program, Runtime, Verification

# Generate ETDump and the debug data buffer
et_runtime: Runtime = Runtime.get()
program: Program = et_runtime.load_program(
    pte_path,
    verification=Verification.Minimal,
    enable_etdump=True,
    debug_buffer_size=1024 * 1024 * 1024,  # 1GB buffer for intermediate outputs
)

forward: Method = program.load_method("forward")
forward.execute(model_inputs)

etdump_path = os.path.join(temp_dir, "etdump.etdp")
debug_buffer_path = os.path.join(temp_dir, "debug_buffer.bin")
program.write_etdump_result_to_file(etdump_path, debug_buffer_path)

######################################################################
#
# .. warning::
#    The debug buffer size should be large enough to hold all intermediate
#    outputs. For large models like ViT, a 1GB buffer is recommended.
#    If the buffer is too small, some intermediate outputs may be truncated.
#

######################################################################
# Creating an Inspector for Debugging
# ------------------------------------
#
# Create the ``Inspector`` by passing in the artifact paths, including
# the debug buffer path. The Inspector will use these to correlate
# runtime results with the original model and compute numerical gaps.

# sphinx_gallery_start_ignore
inspector_patch_print = patch.object(Inspector, "print_data_tabular", return_value=None)
inspector_patch_print.start()
inspector_patch_calculate = patch.object(
    Inspector, "calculate_numeric_gap", return_value=None
)
inspector_patch_calculate.start()
# sphinx_gallery_end_ignore

inspector = Inspector(
    etdump_path=etdump_path,
    etrecord=etrecord_path,
    debug_buffer_path=debug_buffer_path,
)

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
# - ``MAE``: Mean Absolute Error
# - ``SNR``: Signal-to-Noise Ratio
# - ``COSINE``: Cosine Similarity

import pandas as pd

pd.set_option("display.width", 100000)
pd.set_option("display.max_columns", None)

# Calculate numerical gap using Mean Squared Error
df: pd.DataFrame = inspector.calculate_numeric_gap("MSE")

######################################################################
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

# Find operators with the largest discrepancies
if df is not None:
    # Sort by MSE to find the most problematic operators
    df_sorted = df.sort_values(by="mse", ascending=False)

    print("Top 5 operators with largest numerical discrepancies:")
    print(df_sorted.head(5))

    # Filter for operators with MSE above a threshold
    threshold = 1e-4
    problematic_ops = df[df["mse"] > threshold]
    print(f"\nOperators with MSE > {threshold}:")
    print(problematic_ops)

######################################################################
# Debugging Specific Operators
# ----------------------------
#
# For detailed debugging, you can examine the intermediate outputs
# of specific operators.

for event_block in inspector.event_blocks:
    for event in event_block.events:
        # Access debug data (intermediate outputs) for each event
        if event.debug_data is not None:
            print(f"Operator: {event.name}")
            print(f"  Debug data shape: {event.debug_data.shape}")
            print(f"  Debug data dtype: {event.debug_data.dtype}")

            # If ETRecord was provided, you can also see stack traces
            if event.stack_traces:
                print(f"  Stack trace: {event.stack_traces}")

            if event.module_hierarchy:
                print(f"  Module hierarchy: {event.module_hierarchy}")

######################################################################
# Comparing with Reference Outputs
# ---------------------------------
#
# You can also use the ``compare_results`` utility to perform quality
# analysis between runtime outputs and reference outputs.

from executorch.devtools.inspector import compare_results

# Get runtime outputs and compare with eager model outputs
for event_block in inspector.event_blocks:
    if event_block.name == "Execute":
        # Get runtime output
        runtime_output = event_block.run_output

        # Compute reference output from eager model
        with torch.no_grad():
            ref_output = model(*model_inputs)

        if runtime_output is not None:
            # Compare results (set plot=True to generate visualizations)
            comparison = compare_results(
                runtime_output,
                [ref_output],
                plot=False,  # Set to True in a notebook environment
            )
            print("Comparison results:", comparison)

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
# 1. Generate ETRecord with representative inputs using ``update_representative_inputs``
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
