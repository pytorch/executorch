# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Using the ExecuTorch Developer Tools to Debug a Model
========================
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
# 2. Run the model and compare final outputs between eager model and runtime.
# 3. If discrepancies exist, use the Inspector's ``calculate_numeric_gap`` method to identify operator-level issues.
#
# We provide two pipelines:
#
# - **Python Pipeline**: Export, run, and debug entirely in Python using the ExecuTorch Runtime API.
# - **CMake Pipeline**: Export in Python, run with CMake example runner, then analyze in Python.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you'll first need to
# `Set up your ExecuTorch environment <../getting-started-setup.html>`__.
#
# For the Python pipeline, you'll need the ExecuTorch Python runtime bindings.
# For the CMake pipeline, follow `these instructions <../runtime-build-and-cross-compilation.html#configure-the-cmake-build>`__ to set up CMake.
#

######################################################################
# Pipeline 1: Python Runtime (Recommended for Development)
# =========================================================
#
# This pipeline allows you to export, run, and debug your model entirely in Python,
# making it ideal for rapid iteration during development. We'll use a Vision Transformer
# (ViT) model delegated to XNNPACK as an example.

######################################################################
# Step 1: Export Model and Generate ETRecord
# ------------------------------------------
#
# First, we export the model and generate an ``ETRecord``. The ETRecord contains
# model graphs and metadata for linking runtime results to the eager model.
# We use ``to_edge_transform_and_lower`` with ``generate_etrecord=True`` to
# automatically capture the ETRecord during the lowering process.

import os
import tempfile

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config

from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower
from torch.export import export, ExportedProgram
from torchvision import models

# Create Vision Transformer model
vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
model = vit.eval()
model_inputs = (torch.randn(1, 3, 224, 224),)

temp_dir = tempfile.mkdtemp()

# Export and lower model to XNNPACK delegate
aten_model: ExportedProgram = export(model, model_inputs, strict=True)
edge_program_manager = to_edge_transform_and_lower(
    aten_model,
    partitioner=[XnnpackPartitioner()],
    compile_config=get_xnnpack_edge_compile_config(),
    generate_etrecord=True,
)

et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()

# Save the .pte file
pte_path = os.path.join(temp_dir, "model.pte")
et_program_manager.save(pte_path)

# Get and save ETRecord with representative inputs
etrecord = et_program_manager.get_etrecord()
etrecord.update_representative_inputs(model_inputs)
etrecord_path = os.path.join(temp_dir, "etrecord.bin")
etrecord.save(etrecord_path)

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
# Step 2: Run Model and Generate ETDump with Debug Buffer
# -------------------------------------------------------
#
# Next, we run the model using the ExecuTorch Python Runtime API with debug
# output enabled. The debug buffer captures intermediate outputs from the
# runtime execution.
#
# .. code-block:: python
#
#    from executorch.runtime import Method, Program, Runtime, Verification
#
#    # Load and run the model with debug output enabled
#    et_runtime: Runtime = Runtime.get()
#    program: Program = et_runtime.load_program(
#        pte_path,
#        verification=Verification.Minimal,
#        enable_etdump=True,
#        debug_buffer_size=1024 * 1024 * 1024,  # 1GB buffer
#    )
#
#    forward: Method = program.load_method("forward")
#    runtime_outputs = forward.execute(*model_inputs)
#
#    # Save ETDump and debug buffer
#    etdump_path = os.path.join(temp_dir, "etdump.etdp")
#    debug_buffer_path = os.path.join(temp_dir, "debug_buffer.bin")
#    program.write_etdump_result_to_file(etdump_path, debug_buffer_path)
#
# .. warning::
#    The debug buffer size should be large enough to hold all intermediate
#    outputs. For large models like ViT, a 1GB buffer is recommended.
#    If the buffer is too small, some intermediate outputs may be truncated.
#

######################################################################
# Step 3: Compare Final Outputs (Best Practice)
# ---------------------------------------------
#
# **Best Practice**: Before diving into operator-level debugging, first compare
# the final outputs between the eager model and the runtime model. This helps
# you quickly determine if there are any numerical issues worth investigating.
#
# .. code-block:: python
#
#    # Get eager model output
#    with torch.no_grad():
#        eager_output = model(*model_inputs)
#
#    # Compare with runtime output
#    if isinstance(runtime_outputs, (list, tuple)):
#        runtime_output = runtime_outputs[0]
#    else:
#        runtime_output = runtime_outputs
#
#    # Calculate MSE between eager and runtime outputs
#    mse = torch.mean((eager_output - runtime_output) ** 2).item()
#    print(f"Final output MSE: {mse}")
#
#    # Check if outputs are close enough
#    if torch.allclose(eager_output, runtime_output, rtol=1e-3, atol=1e-5):
#        print("Outputs match within tolerance!")
#    else:
#        print("Outputs differ - proceeding with operator-level analysis...")
#

######################################################################
# Step 4: Operator-Level Debugging with calculate_numeric_gap
# -----------------------------------------------------------
#
# If the final outputs show discrepancies, use the Inspector's ``calculate_numeric_gap``
# method to identify which operators are contributing to the numerical differences.
#
# .. code-block:: python
#
#    import pandas as pd
#    from executorch.devtools import Inspector
#
#    inspector = Inspector(
#        etdump_path=etdump_path,
#        etrecord=etrecord_path,
#        debug_buffer_path=debug_buffer_path,
#    )
#
#    pd.set_option("display.width", 100000)
#    pd.set_option("display.max_columns", None)
#
#    # Calculate numerical gap using Mean Squared Error
#    df: pd.DataFrame = inspector.calculate_numeric_gap("MSE")
#    print(df)
#
# The returned DataFrame contains columns for each operator including:
#
# - ``aot_ops``: The operators in the eager model graph
# - ``aot_intermediate_output``: Intermediate outputs from eager model
# - ``runtime_ops``: The operators executed at runtime (may show DELEGATE_CALL for delegated ops)
# - ``runtime_intermediate_output``: Intermediate outputs from runtime
# - ``gap``: The numerical gap (MSE) between eager and runtime outputs
#
# Example output:
#
# .. code-block:: text
#
#    |    | aot_ops                              | runtime_ops         | gap                      |
#    |----|--------------------------------------|---------------------|--------------------------|
#    | 0  | [conv2d]                             | [DELEGATE_CALL]     | [3.25e-15]               |
#    | 4  | [transpose, linear, unflatten, ...]  | [DELEGATE_CALL, ...]| [0.00010033142876115867] |
#    | 59 | [transpose_66, linear_44, ...]       | [DELEGATE_CALL, ...]| [0.02629170972698486]    |
#

######################################################################
# Step 5: Analyze and Identify Problematic Operators
# --------------------------------------------------
#
# Once you have the numerical gaps, identify operators with significant
# discrepancies for further investigation.
#
# .. code-block:: python
#
#    # Find operators with the largest discrepancies
#    df_sorted = df.sort_values(by="gap", ascending=False, key=lambda x: x.apply(lambda y: y[0] if isinstance(y, list) else y))
#
#    print("Top 5 operators with largest numerical discrepancies:")
#    print(df_sorted.head(5))
#
#    # Filter for operators with gap above a threshold
#    threshold = 1e-4
#    problematic_ops = df[df["gap"].apply(lambda x: x[0] > threshold if isinstance(x, list) else x > threshold)]
#    print(f"\nOperators with MSE > {threshold}:")
#    print(problematic_ops)
#
# Example output showing problematic operators in a ViT model:
#
# .. code-block:: text
#
#    Top 5 operators with largest numerical discrepancies:
#                                                  aot_ops           gap
#    59  [transpose_66, linear_44, unflatten_11, ...]  [0.02629170972698486]
#    24  [transpose_24, linear_16, unflatten_4, ...]   [0.010045093258604096]
#    29  [transpose_30, linear_20, unflatten_5, ...]   [0.008497326594593926]
#
#    Operators with MSE > 0.0001:
#    (12 operators found with gaps above threshold)
#
# In this example, we can see that the attention layers (transpose + linear + unflatten patterns)
# show the largest numerical discrepancies, which is expected behavior for delegated operators
# using different precision.

######################################################################
# Pipeline 2: CMake Runtime
# ==========================
#
# This pipeline is useful when you want to test your model with the native
# C++ runtime or on platforms where Python bindings are not available.
# We continue using the same ViT model from Pipeline 1.

######################################################################
# Step 1: Export Model and Generate ETRecord
# ------------------------------------------
#
# Same as Pipeline 1 - we reuse the model and export artifacts we already created.
# The key artifacts needed for the CMake pipeline are:
#
# - ``model.pte``: The ExecuTorch program file
# - ``etrecord.bin``: The ETRecord with representative inputs
#
# These were already generated in Pipeline 1's Step 1. If you're only using
# the CMake pipeline, use the same export code:
#
# .. code-block:: python
#
#    # Export and lower model (same as Pipeline 1)
#    aten_model = export(model, model_inputs, strict=True)
#    edge_program_manager = to_edge_transform_and_lower(
#        aten_model,
#        partitioner=[XnnpackPartitioner()],
#        compile_config=get_xnnpack_edge_compile_config(),
#        generate_etrecord=True,
#    )
#    et_program_manager = edge_program_manager.to_executorch()
#
#    # Save artifacts
#    et_program_manager.save(pte_path)
#    etrecord = et_program_manager.get_etrecord()
#    etrecord.update_representative_inputs(model_inputs)
#    etrecord.save(etrecord_path)
#

######################################################################
# Step 2: Create BundledProgram
# -----------------------------
#
# For the CMake pipeline, we create a ``BundledProgram`` that packages the model
# with sample inputs and expected outputs for testing. We reuse the
# ``et_program_manager`` from Step 1.

from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

# Construct Method Test Suites using the same model and inputs from Pipeline 1
m_name = "forward"
inputs = [model_inputs for _ in range(2)]

method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(inputs=inp, expected_outputs=model(*inp)) for inp in inputs
        ],
    )
]

# Generate BundledProgram using the existing et_program_manager
bundled_program = BundledProgram(et_program_manager, method_test_suites)

# Serialize BundledProgram to flatbuffer
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
    bundled_program
)
bundled_program_path = os.path.join(temp_dir, "bundled_program.bp")
with open(bundled_program_path, "wb") as f:
    f.write(serialized_bundled_program)

######################################################################
# Step 3: Run with CMake Example Runner
# -------------------------------------
#
# Build and run the example runner with debug output enabled::
#
#       cd executorch
#       ./examples/devtools/build_example_runner.sh
#       cmake-out/examples/devtools/example_runner --bundled_program_path="bundled_program.bp"
#
# Since the BundledProgram includes expected outputs from the eager model,
# the example runner will automatically compare runtime outputs against
# the reference outputs and report whether they match. This gives you
# immediate feedback on numerical accuracy.
#
# Example output:
#
# .. code-block:: text
#
#    I 00:00:00.123456 executorch:example_runner.cpp:123] Method forward: output 0 matches reference.
#
# If outputs don't match, you'll see:
#
# .. code-block:: text
#
#    W 00:00:00.123456 executorch:example_runner.cpp:123] Method forward: output 0 MISMATCH with reference!
#
# This will also generate:
#
# - ``etdump.etdp``: The ETDump file containing execution trace
# - ``debug_output.bin``: The debug buffer containing intermediate outputs

######################################################################
# Step 4: Analyze Results in Python
# ---------------------------------
#
# After running the model with the CMake runner, load the generated artifacts
# back into Python for analysis using the Inspector.

from executorch.devtools import Inspector

# sphinx_gallery_start_ignore
inspector_patch = patch.object(Inspector, "__init__", return_value=None)
inspector_patch.start()
# sphinx_gallery_end_ignore
etrecord_path = "etrecord.bin"
etdump_path = "etdump.etdp"
debug_buffer_path = "debug_output.bin"

inspector = Inspector(
    etdump_path=etdump_path,
    etrecord=etrecord_path,
    debug_buffer_path=debug_buffer_path,
)

# sphinx_gallery_start_ignore
inspector_patch.stop()
# sphinx_gallery_end_ignore

######################################################################
# Then use the same analysis techniques as in Pipeline 1:
#
# .. code-block:: python
#
#    import pandas as pd
#
#    # Calculate numerical gaps
#    df = inspector.calculate_numeric_gap("MSE")
#
#    # Find problematic operators
#    df_sorted = df.sort_values(by="gap", ascending=False,
#        key=lambda x: x.apply(lambda y: y[0] if isinstance(y, list) else y))
#    print("Top operators with largest gaps:")
#    print(df_sorted.head(5))
#

######################################################################
# Best Practices for Debugging
# ============================
#
# 1. **Start with final outputs**: Always compare the final model output first
#    before diving into operator-level analysis. This saves time if outputs match.
#
# 2. **Use appropriate thresholds**: Small numerical differences (< 1e-6) are
#    typically acceptable. Focus on operators with gaps > 1e-4.
#
# 3. **Focus on delegated operators**: Numerical discrepancies are most common
#    in delegated operators (shown as ``DELEGATE_CALL``) due to different
#    precision handling in delegate backends.
#
# 4. **Check accumulation patterns**: In transformer models, attention layers
#    often show larger gaps due to accumulated numerical differences across
#    many operations.
#
# 5. **Use stack traces**: With ETRecord, you can trace operators back to the
#    original PyTorch source code for easier debugging using
#    ``event.stack_traces`` and ``event.module_hierarchy``.
#

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned how to use the ExecuTorch Developer Tools
# to debug numerical discrepancies in models. The key workflow is:
#
# 1. Export the model with ETRecord generation enabled
# 2. Run the model with debug buffer enabled (Python or CMake)
# 3. **First** compare final outputs between eager and runtime models
# 4. **If issues found**, use ``calculate_numeric_gap`` for operator-level analysis
# 5. Identify and investigate operators with significant gaps
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
