# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Exporting to ExecuTorch Tutorial
================================

**Author:** `Angela Yi <https://github.com/angelayi>`__
"""

######################################################################
# ExecuTorch is a unified ML stack for lowering PyTorch models to edge devices.
# It introduces improved entry points to perform model, device, and/or use-case
# specific optimizations such as backend delegation, user-defined compiler
# transformations, default or user-defined memory planning, and more.
#
# At a high level, the workflow looks as follows:
#
# .. image:: ../executorch_stack.png
#   :width: 560
#
# In this tutorial, we will cover the APIs in the "Program preparation" steps to
# lower a PyTorch model to a format which can be loaded to device and run on the
# ExecuTorch runtime.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you’ll first need to
# `Set up your ExecuTorch environment <../getting-started-setup.html>`__.

######################################################################
# Exporting a Model
# -----------------
#
# Note: The Export APIs are still undergoing changes to align better with the
# longer term state of export. Please refer to this
# `issue <https://github.com/pytorch/executorch/issues/290>`__ for more details.
#
# The first step of lowering to ExecuTorch is to export the given model (any
# callable or ``torch.nn.Module``) to a graph representation. This is done via
# ``torch.export``, which takes in an ``torch.nn.Module``, a tuple of
# positional arguments, optionally a dictionary of keyword arguments (not shown
# in the example), and a list of dynamic shapes (covered later).

import torch
from torch.export import export, ExportedProgram


class SimpleConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        return self.relu(a)


example_args = (torch.randn(1, 3, 256, 256),)
aten_dialect: ExportedProgram = export(SimpleConv(), example_args)
print(aten_dialect)

######################################################################
# The output of ``torch.export.export`` is a fully flattened graph (meaning the
# graph does not contain any module hierarchy, except in the case of control
# flow operators). Additionally, the graph is purely functional, meaning it does
# not contain operations with side effects such as mutations or aliasing.
#
# More specifications about the result of ``torch.export`` can be found
# `here <https://pytorch.org/docs/main/export.html>`__ .
#
# The graph returned by ``torch.export`` only contains functional ATen operators
# (~2000 ops), which we will call the ``ATen Dialect``.

######################################################################
# Expressing Dynamism
# ^^^^^^^^^^^^^^^^^^^
#
# By default, the exporting flow will trace the program assuming that all input
# shapes are static, so if we run the program with inputs shapes that are
# different than the ones we used while tracing, we will run into an error:

import traceback as tb


class Basic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


example_args = (torch.randn(3, 3), torch.randn(3, 3))
aten_dialect: ExportedProgram = export(Basic(), example_args)

# Works correctly
print(aten_dialect.module()(torch.ones(3, 3), torch.ones(3, 3)))

# Errors
try:
    print(aten_dialect.module()(torch.ones(3, 2), torch.ones(3, 2)))
except Exception:
    tb.print_exc()

######################################################################
# To express that some input shapes are dynamic, we can insert dynamic
#  shapes to the exporting flow. This is done through the ``Dim`` API:

from torch.export import Dim


class Basic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


example_args = (torch.randn(3, 3), torch.randn(3, 3))
dim1_x = Dim("dim1_x", min=1, max=10)
dynamic_shapes = {"x": {1: dim1_x}, "y": {1: dim1_x}}
aten_dialect: ExportedProgram = export(
    Basic(), example_args, dynamic_shapes=dynamic_shapes
)
print(aten_dialect)

######################################################################
# Note that that the inputs ``arg0_1`` and ``arg1_1`` now have shapes (3, s0),
# with ``s0`` being a symbol representing that this dimension can be a range
# of values.
#
# Additionally, we can see in the **Range constraints** that value of ``s0`` has
# the range [1, 10], which was specified by our dynamic shapes.
#
# Now let's try running the model with different shapes:

# Works correctly
print(aten_dialect.module()(torch.ones(3, 3), torch.ones(3, 3)))
print(aten_dialect.module()(torch.ones(3, 2), torch.ones(3, 2)))

# Errors because it violates our constraint that input 0, dim 1 <= 10
try:
    print(aten_dialect.module()(torch.ones(3, 15), torch.ones(3, 15)))
except Exception:
    tb.print_exc()

# Errors because it violates our constraint that input 0, dim 1 == input 1, dim 1
try:
    print(aten_dialect.module()(torch.ones(3, 3), torch.ones(3, 2)))
except Exception:
    tb.print_exc()


######################################################################
# Addressing Untraceable Code
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As our goal is to capture the entire computational graph from a PyTorch
# program, we might ultimately run into untraceable parts of programs. To
# address these issues, the
# `torch.export documentation <https://pytorch.org/docs/main/export.html#limitations-of-torch-export>`__,
# or the
# `torch.export tutorial <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__
# would be the best place to look.

######################################################################
# Performing Quantization
# -----------------------
#
# To quantize a model, we first need to capture the graph with
# ``torch.export.export_for_training``, perform quantization, and then
# call ``torch.export``. ``torch.export.export_for_training`` returns a
# graph which contains ATen operators which are Autograd safe, meaning they are
# safe for eager-mode training, which is needed for quantization. We will call
# the graph at this level, the ``Pre-Autograd ATen Dialect`` graph.
#
# Compared to
# `FX Graph Mode Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`__,
# we will need to call two new APIs: ``prepare_pt2e`` and ``convert_pt2e``
# instead of ``prepare_fx`` and ``convert_fx``. It differs in that
# ``prepare_pt2e`` takes a backend-specific ``Quantizer`` as an argument, which
# will annotate the nodes in the graph with information needed to quantize the
# model properly for a specific backend.

from torch.export import export_for_training

example_args = (torch.randn(1, 3, 256, 256),)
pre_autograd_aten_dialect = export_for_training(SimpleConv(), example_args).module()
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
# calibrate with a sample dataset
converted_graph = convert_pt2e(prepared_graph)
print("Quantized Graph")
print(converted_graph)

aten_dialect: ExportedProgram = export(converted_graph, example_args)
print("ATen Dialect Graph")
print(aten_dialect)

######################################################################
# More information on how to quantize a model, and how a backend can implement a
# ``Quantizer`` can be found
# `here <https://pytorch.org/docs/main/quantization.html#prototype-pytorch-2-export-quantization>`__.

######################################################################
# Lowering to Edge Dialect
# ------------------------
#
# After exporting and lowering the graph to the ``ATen Dialect``, the next step
# is to lower to the ``Edge Dialect``, in which specializations that are useful
# for edge devices but not necessary for general (server) environments will be
# applied.
# Some of these specializations include:
#
# - DType specialization
# - Scalar to tensor conversion
# - Converting all ops to the ``executorch.exir.dialects.edge`` namespace.
#
# Note that this dialect is still backend (or target) agnostic.
#
# The lowering is done through the ``to_edge`` API.

from executorch.exir import EdgeProgramManager, to_edge

example_args = (torch.randn(1, 3, 256, 256),)
aten_dialect: ExportedProgram = export(SimpleConv(), example_args)

edge_program: EdgeProgramManager = to_edge(aten_dialect)
print("Edge Dialect Graph")
print(edge_program.exported_program())

######################################################################
# ``to_edge()`` returns an ``EdgeProgramManager`` object, which contains the
# exported programs which will be placed on this device. This data structure
# allows users to export multiple programs and combine them into one binary. If
# there is only one program, it will by default be saved to the name "forward".


class Encode(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.linear(x, torch.randn(5, 10))


class Decode(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.linear(x, torch.randn(10, 5))


encode_args = (torch.randn(1, 10),)
aten_encode: ExportedProgram = export(Encode(), encode_args)

decode_args = (torch.randn(1, 5),)
aten_decode: ExportedProgram = export(Decode(), decode_args)

edge_program: EdgeProgramManager = to_edge(
    {"encode": aten_encode, "decode": aten_decode}
)
for method in edge_program.methods:
    print(f"Edge Dialect graph of {method}")
    print(edge_program.exported_program(method))

######################################################################
# We can also run additional passes on the exported program through
# the ``transform`` API. An in-depth documentation on how to write
# transformations can be found
# `here <../compiler-custom-compiler-passes.html>`__.
#
# Note that since the graph is now in the Edge Dialect, all passes must also
# result in a valid Edge Dialect graph (specifically one thing to point out is
# that the operators are now in the ``executorch.exir.dialects.edge`` namespace,
# rather than the ``torch.ops.aten`` namespace.

example_args = (torch.randn(1, 3, 256, 256),)
aten_dialect: ExportedProgram = export(SimpleConv(), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
print("Edge Dialect Graph")
print(edge_program.exported_program())

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertReluToSigmoid(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op == exir_ops.edge.aten.relu.default:
            return super().call_operator(
                exir_ops.edge.aten.sigmoid.default, args, kwargs, meta
            )
        else:
            return super().call_operator(op, args, kwargs, meta)


transformed_edge_program = edge_program.transform((ConvertReluToSigmoid(),))
print("Transformed Edge Dialect Graph")
print(transformed_edge_program.exported_program())

######################################################################
# Note: if you see error like ``torch._export.verifier.SpecViolationError:
# Operator torch._ops.aten._native_batch_norm_legit_functional.default is not
# Aten Canonical``,
# please file an issue in https://github.com/pytorch/executorch/issues and we're happy to help!


######################################################################
# Delegating to a Backend
# -----------------------
#
# We can now delegate parts of the graph or the whole graph to a third-party
# backend through the ``to_backend`` API.  An in-depth documentation on the
# specifics of backend delegation, including how to delegate to a backend and
# how to implement a backend, can be found
# `here <../compiler-delegate-and-partitioner.html>`__.
#
# There are three ways for using this API:
#
# 1. We can lower the whole module.
# 2. We can take the lowered module, and insert it in another larger module.
# 3. We can partition the module into subgraphs that are lowerable, and then
#    lower those subgraphs to a backend.

######################################################################
# Lowering the Whole Module
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To lower an entire module, we can pass ``to_backend`` the backend name, the
# module to be lowered, and a list of compile specs to help the backend with the
# lowering process.


class LowerableModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


# Export and lower the module to Edge Dialect
example_args = (torch.ones(1),)
aten_dialect: ExportedProgram = export(LowerableModule(), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
to_be_lowered_module = edge_program.exported_program()

from executorch.exir.backend.backend_api import LoweredBackendModule, to_backend

# Import the backend
from executorch.exir.backend.test.backend_with_compiler_demo import (  # noqa
    BackendWithCompilerDemo,
)

# Lower the module
lowered_module: LoweredBackendModule = to_backend(
    "BackendWithCompilerDemo", to_be_lowered_module, []
)
print(lowered_module)
print(lowered_module.backend_id)
print(lowered_module.processed_bytes)
print(lowered_module.original_module)

# Serialize and save it to a file
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(lowered_module.buffer())

######################################################################
# In this call, ``to_backend`` will return a ``LoweredBackendModule``. Some
# important attributes of the ``LoweredBackendModule`` are:
#
# - ``backend_id``: The name of the backend this lowered module will run on in
#   the runtime
# - ``processed_bytes``: a binary blob which will tell the backend how to run
#   this program in the runtime
# - ``original_module``: the original exported module

######################################################################
# Compose the Lowered Module into Another Module
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In cases where we want to reuse this lowered module in multiple programs, we
# can compose this lowered module with another module.


class NotLowerableModule(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, a, b):
        return torch.add(torch.add(a, b), self.bias)


class ComposedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.non_lowerable = NotLowerableModule(torch.ones(1) * 0.3)
        self.lowerable = lowered_module

    def forward(self, x):
        a = self.lowerable(x)
        b = self.lowerable(a)
        ret = self.non_lowerable(a, b)
        return a, b, ret


example_args = (torch.ones(1),)
aten_dialect: ExportedProgram = export(ComposedModule(), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
exported_program = edge_program.exported_program()
print("Edge Dialect graph")
print(exported_program)
print("Lowered Module within the graph")
print(exported_program.graph_module.lowered_module_0.backend_id)
print(exported_program.graph_module.lowered_module_0.processed_bytes)
print(exported_program.graph_module.lowered_module_0.original_module)

######################################################################
# Notice that there is now a ``torch.ops.higher_order.executorch_call_delegate`` node in the
# graph, which is calling ``lowered_module_0``. Additionally, the contents of
# ``lowered_module_0`` are the same as the ``lowered_module`` we created
# previously.

######################################################################
# Partition and Lower Parts of a Module
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A separate lowering flow is to pass ``to_backend`` the module that we want to
# lower, and a backend-specific partitioner. ``to_backend`` will use the
# backend-specific partitioner to tag nodes in the module which are lowerable,
# partition those nodes into subgraphs, and then create a
# ``LoweredBackendModule`` for each of those subgraphs.


class Foo(torch.nn.Module):
    def forward(self, a, x, b):
        y = torch.mm(a, x)
        z = y + b
        a = z - a
        y = torch.mm(a, x)
        z = y + b
        return z


example_args = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
aten_dialect: ExportedProgram = export(Foo(), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
exported_program = edge_program.exported_program()
print("Edge Dialect graph")
print(exported_program)

from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo

delegated_program = to_backend(exported_program, AddMulPartitionerDemo())
print("Delegated program")
print(delegated_program)
print(delegated_program.graph_module.lowered_module_0.original_module)
print(delegated_program.graph_module.lowered_module_1.original_module)

######################################################################
# Notice that there are now 2 ``torch.ops.higher_order.executorch_call_delegate`` nodes in the
# graph, one containing the operations `add, mul` and the other containing the
# operations `mul, add`.
#
# Alternatively, a more cohesive API to lower parts of a module is to directly
# call ``to_backend`` on it:


class Foo(torch.nn.Module):
    def forward(self, a, x, b):
        y = torch.mm(a, x)
        z = y + b
        a = z - a
        y = torch.mm(a, x)
        z = y + b
        return z


example_args = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
aten_dialect: ExportedProgram = export(Foo(), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
exported_program = edge_program.exported_program()
delegated_program = edge_program.to_backend(AddMulPartitionerDemo())

print("Delegated program")
print(delegated_program.exported_program())

######################################################################
# Running User-Defined Passes and Memory Planning
# -----------------------------------------------
#
# As a final step of lowering, we can use the ``to_executorch()`` API to pass in
# backend-specific passes, such as replacing sets of operators with a custom
# backend operator, and a memory planning pass, to tell the runtime how to
# allocate memory ahead of time when running the program.
#
# A default memory planning pass is provided, but we can also choose a
# backend-specific memory planning pass if it exists. More information on
# writing a custom memory planning pass can be found
# `here <../compiler-memory-planning.html>`__

from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
from executorch.exir.passes import MemoryPlanningPass

executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
        memory_planning_pass=MemoryPlanningPass(),  # Default memory planning pass
    )
)

print("ExecuTorch Dialect")
print(executorch_program.exported_program())

import executorch.exir as exir

######################################################################
# Notice that in the graph we now see operators like ``torch.ops.aten.sub.out``
# and ``torch.ops.aten.div.out`` rather than ``torch.ops.aten.sub.Tensor`` and
# ``torch.ops.aten.div.Tensor``.
#
# This is because between running the backend passes and memory planning passes,
# to prepare the graph for memory planning, an out-variant pass is run on
# the graph to convert all of the operators to their out variants. Instead of
# allocating returned tensors in the kernel implementations, an operator's
# ``out`` variant will take in a prealloacated tensor to its out kwarg, and
# store the result there, making it easier for memory planners to do tensor
# lifetime analysis.
#
# We also insert ``alloc`` nodes into the graph containing calls to a special
# ``executorch.exir.memory.alloc`` operator. This tells us how much memory is
# needed to allocate each tensor output by the out-variant operator.
#

######################################################################
# Saving to a File
# ----------------
#
# Finally, we can save the ExecuTorch Program to a file and load it to a device
# to be run.
#
# Here is an example for an entire end-to-end workflow:

import torch
from torch.export import export, export_for_training, ExportedProgram


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)


example_args = (torch.randn(3, 4),)
pre_autograd_aten_dialect = export_for_training(M(), example_args).module()
# Optionally do quantization:
# pre_autograd_aten_dialect = convert_pt2e(prepare_pt2e(pre_autograd_aten_dialect, CustomBackendQuantizer))
aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)
# Optionally do delegation:
# edge_program = edge_program.to_backend(CustomBackendPartitioner)
executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we went over the APIs and steps required to lower a PyTorch
# program to a file that can be run on the ExecuTorch runtime.
#
# Links Mentioned
# ^^^^^^^^^^^^^^^
#
# - `torch.export Documentation <https://pytorch.org/docs/2.1/export.html>`__
# - `Quantization Documentation <https://pytorch.org/docs/main/quantization.html#prototype-pytorch-2-export-quantization>`__
# - `IR Spec <../ir-exir.html>`__
# - `Writing Compiler Passes + Partitioner Documentation <../compiler-custom-compiler-passes.html>`__
# - `Backend Delegation Documentation <../compiler-delegate-and-partitioner.html>`__
# - `Memory Planning Documentation <../compiler-memory-planning.html>`__
