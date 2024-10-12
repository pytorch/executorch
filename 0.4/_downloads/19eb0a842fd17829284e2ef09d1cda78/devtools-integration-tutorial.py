# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Using the ExecuTorch Developer Tools to Profile a Model
========================

**Author:** `Jack Khuu <https://github.com/Jack-Khuu>`__
"""

######################################################################
# The `ExecuTorch Developer Tools <../devtools-overview.html>`__ is a set of tools designed to
# provide users with the ability to profile, debug, and visualize ExecuTorch
# models.
#
# This tutorial will show a full end-to-end flow of how to utilize the Developer Tools to profile a model.
# Specifically, it will:
#
# 1. Generate the artifacts consumed by the Developer Tools (`ETRecord <../etrecord.html>`__, `ETDump <../etdump.html>`__).
# 2. Create an Inspector class consuming these artifacts.
# 3. Utilize the Inspector class to analyze the model profiling result.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you’ll first need to
# `Set up your ExecuTorch environment <../getting-started-setup.html>`__.
#

######################################################################
# Generate ETRecord (Optional)
# ----------------------------
#
# The first step is to generate an ``ETRecord``. ``ETRecord`` contains model
# graphs and metadata for linking runtime results (such as profiling) to
# the eager model. This is generated via ``executorch.devtools.generate_etrecord``.
#
# ``executorch.devtools.generate_etrecord`` takes in an output file path (str), the
# edge dialect model (``EdgeProgramManager``), the ExecuTorch dialect model
# (``ExecutorchProgramManager``), and an optional dictionary of additional models.
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
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

aten_model: ExportedProgram = export(
    model,
    (torch.randn(1, 1, 32, 32),),
)

edge_program_manager: EdgeProgramManager = to_edge(
    aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
)
edge_program_manager_copy = copy.deepcopy(edge_program_manager)
et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()


# Generate ETRecord
etrecord_path = "etrecord.bin"
generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)

# sphinx_gallery_start_ignore
from unittest.mock import patch

# sphinx_gallery_end_ignore

######################################################################
#
# .. warning::
#    Users should do a deepcopy of the output of ``to_edge()`` and pass in the
#    deepcopy to the ``generate_etrecord`` API. This is needed because the
#    subsequent call, ``to_executorch()``, does an in-place mutation and will
#    lose debug data in the process.
#

######################################################################
# Generate ETDump
# ---------------
#
# Next step is to generate an ``ETDump``. ``ETDump`` contains runtime results
# from executing a `Bundled Program Model <../bundled-io.html>`__.
#
# In this tutorial, a `Bundled Program` is created from the example model above.

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
method_graphs = {m_name: export(model, (torch.randn(1, 1, 32, 32),))}

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
# Use CMake (follow `these instructions <../runtime-build-and-cross-compilation.html#configure-the-cmake-build>`__ to set up cmake) to execute the Bundled Program to generate the ``ETDump``::
#
#       cd executorch
#       ./examples/devtools/build_example_runner.sh
#       cmake-out/examples/devtools/example_runner --bundled_program_path="bundled_program.bp"

######################################################################
# Creating an Inspector
# ---------------------
#
# Final step is to create the ``Inspector`` by passing in the artifact paths.
# Inspector takes the runtime results from ``ETDump`` and correlates them to
# the operators of the Edge Dialect Graph.
#
# Recall: An ``ETRecord`` is not required. If an ``ETRecord`` is not provided,
# the Inspector will show runtime results without operator correlation.
#
# To visualize all runtime events, call Inspector's ``print_data_tabular``.

from executorch.devtools import Inspector

# sphinx_gallery_start_ignore
inspector_patch = patch.object(Inspector, "__init__", return_value=None)
inspector_patch_print = patch.object(Inspector, "print_data_tabular", return_value="")
inspector_patch.start()
inspector_patch_print.start()
# sphinx_gallery_end_ignore
etrecord_path = "etrecord.bin"
etdump_path = "etdump.etdp"
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)
# sphinx_gallery_start_ignore
inspector.event_blocks = []
# sphinx_gallery_end_ignore
inspector.print_data_tabular()

# sphinx_gallery_start_ignore
inspector_patch.stop()
inspector_patch_print.stop()
# sphinx_gallery_end_ignore

######################################################################
# Analyzing with an Inspector
# ---------------------------
#
# ``Inspector`` provides 2 ways of accessing ingested information: `EventBlocks <../model-inspector#eventblock-class>`__
# and ``DataFrames``. These mediums give users the ability to perform custom
# analysis about their model performance.
#
# Below are examples usages, with both ``EventBlock`` and ``DataFrame`` approaches.

# Set Up
import pprint as pp

import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

######################################################################
# If a user wants the raw profiling results, they would do something similar to
# finding the raw runtime data of an ``addmm.out`` event.

for event_block in inspector.event_blocks:
    # Via EventBlocks
    for event in event_block.events:
        if event.name == "native_call_addmm.out":
            print(event.name, event.perf_data.raw)

    # Via Dataframe
    df = event_block.to_dataframe()
    df = df[df.event_name == "native_call_addmm.out"]
    print(df[["event_name", "raw"]])
    print()

######################################################################
# If a user wants to trace an operator back to their model code, they would do
# something similar to finding the module hierarchy and stack trace of the
# slowest ``convolution.out`` call.

for event_block in inspector.event_blocks:
    # Via EventBlocks
    slowest = None
    for event in event_block.events:
        if event.name == "native_call_convolution.out":
            if slowest is None or event.perf_data.p50 > slowest.perf_data.p50:
                slowest = event
    if slowest is not None:
        print(slowest.name)
        print()
        pp.pprint(slowest.stack_traces)
        print()
        pp.pprint(slowest.module_hierarchy)

    # Via Dataframe
    df = event_block.to_dataframe()
    df = df[df.event_name == "native_call_convolution.out"]
    if len(df) > 0:
        slowest = df.loc[df["p50"].idxmax()]
        print(slowest.event_name)
        print()
        pp.pprint(slowest.stack_traces)
        print()
        pp.pprint(slowest.module_hierarchy)

######################################################################
# If a user wants the total runtime of a module, they can use
# ``find_total_for_module``.

print(inspector.find_total_for_module("L__self__"))
print(inspector.find_total_for_module("L__self___conv2"))

######################################################################
# Note: ``find_total_for_module`` is a special first class method of
# `Inspector <../model-inspector.html>`__

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned about the steps required to consume an ExecuTorch
# model with the ExecuTorch Developer Tools. It also showed how to use the Inspector APIs
# to analyze the model run results.
#
# Links Mentioned
# ^^^^^^^^^^^^^^^
#
# - `ExecuTorch Developer Tools Overview <../devtools-overview.html>`__
# - `ETRecord <../etrecord.html>`__
# - `ETDump <../etdump.html>`__
# - `Inspector <../model-inspector.html>`__
