# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SDK Integration Tutorial
========================

**Author:** `Jack Khuu <https://github.com/Jack-Khuu>`__
"""

######################################################################
# The `ExecuTorch SDK <../sdk-overview.html>`__ is a set of tools designed to
# provide users with the ability to profile, debug, and visualize ExecuTorch
# models.
#
# This tutorial will show a full end-to-end flow of how to utilize the SDK.
# Specifically, it will:
#
# 1. Generate the artifacts consumed by the SDK (`ETRecord <../sdk-etrecord>`__, `ETDump <../sdk-etdump.html>`__).
# 2. Create an Inspector class consuming these artifacts.
# 3. Utilize the Inspector class to analyze the model.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you’ll need to install ExecuTorch.
#
# Set up a conda environment. To set up a conda environment in Google Colab::
#
#   !pip install -q condacolab
#   import condacolab
#   condacolab.install()
#
#   !conda create --name executorch python=3.10
#   !conda install -c conda-forge flatbuffers
#
# Install ExecuTorch from source. If cloning is failing on Google Colab, make
# sure Colab -> Setting -> Github -> Access Private Repo is checked::
#
#   !git clone https://{github_username}:{token}@github.com/pytorch/executorch.git
#   !cd executorch && bash ./install_requirements.sh

######################################################################
# Generate ETRecord (Optional)
# ----------------------------
#
# The first step is to generate an ``ETRecord``. ``ETRecord`` contains model
# graphs and metadata for linking runtime results (such as profiling) to
# the eager model. This is generated via ``executorch.sdk.generate_etrecord``.
#
# ``executorch.sdk.generate_etrecord`` takes in an output file path (str), the
# edge dialect model (``EdgeProgramManager``), the ExecuTorch dialect model
# (``ExecutorchProgramManager``), and an optional dictionary of additional models
#
# In this tutorial, the mobilenet v2 example model is used to demonstrate::
#
#   # Imports
#   import copy
#
#   import torch
#
#   from executorch.examples.models.mobilenet_v2 import MV2Model
#   from executorch.exir import (
#       EdgeCompileConfig,
#       EdgeProgramManager,
#       ExecutorchProgramManager,
#       to_edge,
#   )
#   from executorch.sdk import generate_etrecord
#   from torch.export import export, ExportedProgram
#
#   # Generate MV2 Model
#   model: torch.nn.Module = MV2Model()
#
#   aten_model: ExportedProgram = export(
#       model.get_eager_model().eval(),
#       model.get_example_inputs(),
#   )
#
#   edge_program_manager: EdgeProgramManager = to_edge(
#       aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
#   )
#   edge_program_manager_copy = copy.deepcopy(edge_program_manager)
#   et_program_manager: ExecutorchProgramManager = edge_program_manager_copy.to_executorch()
#
#
#   # Generate ETRecord
#   etrecord_path = "etrecord.bin"
#   generate_etrecord(etrecord_path, edge_program_manager, et_program_manager)
#
# .. warning::
#    Users should do a deepcopy of the output of to_edge() and pass in the
#    deepcopy to the generate_etrecord API. This is needed because the
#    subsequent call, to_executorch(), does an in-place mutation and will
#    lose debug data in the process.
#

######################################################################
# Generate ETDump
# ---------------
#
# Next step is to generate an ``ETDump``. ``ETDump`` contains runtime results
# from executing the model. To generate, simply pass the ExecuTorch model
# to the ``executor_runner``::
#
#   buck2 run executorch/examples/portable/scripts:export -- -m mv2
#   buck2 run @mode/opt -c executorch.event_tracer_enabled=true executorch/sdk/runners:executor_runner -- --model_path mv2.pte
#
# TODO: Add Instructions for CMake, when landed

######################################################################
# Creating an Inspector
# ---------------------
#
# Final step is to create the ``Inspector`` by passing in the artifact paths.
# Inspector takes the runtime results from ``ETDump`` and correlates them to
# the operators of the Edge Dialect Graph.
#
# Note: An ``ETRecord`` is not required. If an ``ETRecord`` is not provided,
# the Inspector will show runtime results without operator correlation.
#
# To visualize all runtime events, call Inspector's ``print_data_tabular``::
#
#   from executorch.sdk import Inspector
#
#   etdump_path = "etdump.etdp"
#   inspector = Inspector(etdump_path=etdump_path, etrecord_path=etrecord_path)
#   inspector.print_data_tabular()
#

######################################################################
# Analyzing with an Inspector
# ---------------------------
#
# ``Inspector`` provides 2 ways of accessing ingested information: `EventBlocks <../sdk-inspector.html>`__
# and ``DataFrames``. These mediums give users the ability to perform custom
# analysis about their model performance.
#
# Below are examples usages, with both ``EventBlock`` and ``DataFrame`` approaches::
#
#   # Set Up
#
#   import pprint as pp
#
#   import pandas as pd
#
#   pd.set_option("display.max_colwidth", None)
#   pd.set_option("display.max_columns", None)

######################################################################
# If a user wants the raw profiling results, they would do something similar to
# finding the raw runtime data of an ``addmm.out`` event::
#
#   for event_block in inspector.event_blocks:
#       # Via EventBlocks
#       for event in event_block.events:
#           if event.name == "native_call_addmm.out":
#               print(event.name, event.perf_data.raw)
#
#       # Via Dataframe
#       df = event_block.to_dataframe()
#       df = df[df.event_name == "native_call_addmm.out"]
#       print(df[["event_name', 'raw"]])
#       print()
#

######################################################################
# If a user wants to trace an operator back to their model code, they would do
# something similar to finding the module hierarchy and stack trace of the
# slowest ``convolution.out`` call::
#
#   for event_block in inspector.event_blocks:
#       # Via EventBlocks
#       slowest = None
#       for event in event_block.events:
#           if event.name == "native_call_convolution.out":
#               if slowest is None or event.perf_data.p50 > slowest.perf_data.p50:
#                   slowest = event
#       if slowest is not None:
#           print(slowest.name)
#           print()
#           pp.pprint(slowest.stack_traces)
#           print()
#           pp.pprint(slowest.module_hierarchy)
#
#       # Via Dataframe
#       df = event_block.to_dataframe()
#       df = df[df.event_name == "native_call_convolution.out"]
#       if len(df) > 0:
#           slowest = df.loc[df["p50"].idxmax()]
#           print(slowest.event_name)
#           print()
#           pp.pprint(slowest.stack_traces)
#           print()
#           pp.pprint(slowest.module_hierarchy)
#

######################################################################
# If a user wants the total runtime of a module, they can use
# ``find_total_for_module``::
#
#   print(inspector.find_total_for_module("L__self___features"))
#   print(inspector.find_total_for_module("L__self___features_14"))

######################################################################
# Note: ``find_total_for_module`` is a special first class method of
# `Inspector <../sdk-inspector.html>`__

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned about the steps required to consume an ExecuTorch
# model with the ExecuTorch SDK. It also showed how to use the Inspector APIs
# to analyze the model run results.
#
# Links Mentioned
# ^^^^^^^^^^^^^^^
#
# - `ExecuTorch SDK <../sdk-overview.html>`__
# - `ETRecord <../sdk-etrecord>`__
# - `ETDump <../sdk-etdump.html>`__
# - `Inspector <../sdk-inspector.html>`__
