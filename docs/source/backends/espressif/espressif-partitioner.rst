===============
Partitioner API
===============

``EspNNPartitioner`` delegates the int8-quantized regions of a graph to the ESP-NN backend. Each delegated region is serialized into a compact int8 graph that the device runtime executes with ESP-NN. Everything else stays in the program and runs on ExecuTorch's CPU kernels.

The partitioner expects a program quantized with ``EspNNQuantizer`` (see :doc:`espressif-quantization`). Operators that aren't quantized, aren't supported, or don't meet an operator's constraints stay on the CPU.

The usual entry point is ``to_espnn_edge``:

.. code-block:: python

   from executorch.backends.espressif import to_espnn_edge

   edge, report = to_espnn_edge(exported_program, allow_cpu_fallback=True)

It calls ``to_edge_transform_and_lower`` with an ``EspNNPartitioner``, the ``DuplicateDequantNodePass`` transform and ``EdgeCompileConfig(_check_ir_validity=False)``. It returns the ``EdgeProgramManager`` and the partitioner's report. To use the partitioner directly, pass the same transform pass, so that a dequantized tensor shared by several consumers can be delegated with each of them:

.. code-block:: python

   from executorch.backends.espressif import EspNNPartitioner
   from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
   from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
       DuplicateDequantNodePass,
   )

   partitioner = EspNNPartitioner()
   edge = to_edge_transform_and_lower(
       exported_program,
       transform_passes=[DuplicateDequantNodePass()],
       partitioner=[partitioner],
       compile_config=EdgeCompileConfig(_check_ir_validity=False),
   )
   report = partitioner.report

Options
-------

The `constructor <https://github.com/pytorch/executorch/blob/main/backends/espressif/partition/espnn_partitioner.py>`_ takes the following options:

- ``allow_cpu_fallback``: If true (the default), operators that can't be delegated stay in the program for the CPU kernels. If false, partitioning raises ``RuntimeError`` whenever any operator would run on the CPU, whether it is unsupported or its region could not be delegated. The error names the operators.
- ``compile_specs``: Compile specs forwarded to the backend. The only spec is ``espnn_format``, which the partitioner adds with the value ``int8`` when it is absent. ``EspNNBackend.preprocess`` rejects any other value.

The partitioner keeps the following operators whole during edge lowering when they are quantized, so that they reach the backend as single nodes: ``linear``, ``hardtanh``, ``max_pool2d``, ``flatten``, ``adaptive_avg_pool2d``, ``avg_pool2d``, ``hardswish``, ``hardsigmoid``, ``silu``, ``upsample_nearest2d`` and ``upsample_bilinear2d``. Float instances decompose as usual for the CPU kernels.

Lowering report
---------------

After partitioning, ``partitioner.report`` (the second value returned by ``to_espnn_edge``) has the following keys:

- ``partitions``: the number of delegated regions.
- ``rejected_partitions``: regions that were found but couldn't be delegated as a unit, for example because a boundary tensor wasn't quantized. Their operators run on the CPU.
- ``delegated_ops``: the number of operators inside delegates, not counting quantize and dequantize nodes.
- ``cpu_ops``: a list with one entry per operator left on the CPU. Each entry has ``name``, ``target`` and ``reason``, which explains why the operator wasn't delegated.

For example:

.. code-block:: python

   for op in report["cpu_ops"]:
       print(f"{op['target']:40s} {op['reason']}")

Common reasons are ``no int8 lowering for this operator``, a constraint from the table below, and ``data movement feeds an operator that runs on the CPU``. Data movement operators are delegated only when their consumers are delegated too.

Runtime limits
--------------

The lowering enforces the limits of the device runtime, so a graph that the device would reject fails at export time instead:

- at most 16 inputs and 16 outputs per delegate;
- at most 1024 tensors and 1024 operations per delegate;
- tensors of rank 4 or less, with batch size 1 for rank-4 tensors;
- every operation reads only delegate inputs, constants or tensors written by an earlier operation.

================
Operator Support
================

This section lists the operators the Espressif backend delegates. Operators are the building blocks of the ML model. See `IRs <https://docs.pytorch.org/docs/stable/torch.compiler_ir.html>`_ for more information on the PyTorch operator set.

All delegated operators compute in int8, with per-tensor activations and per-output-channel symmetric weights, as produced by ``EspNNQuantizer``. Shapes are static. Operators that aren't listed, float operators, and operators that don't meet their constraints run on ExecuTorch's CPU kernels.

.. csv-table:: Operator Support
   :file: op-support.csv
   :header-rows: 1
   :widths: 20 15 30 30
   :align: center
