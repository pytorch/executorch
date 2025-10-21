================
Operator Support
================

This page lists the operators currently supported by the Vulkan backend. The
source of truth for this information is `op_registry.py <https://github.com/pytorch/executorch/blob/main/backends/vulkan/op_registry.py>`_,
which is used by the Vulkan Partitioner to determine which operators should be
lowered to the Vulkan backend and additionally describes the capabilities of
each operator implementation.

If an operator used in your model is not in this list, feel free to create a
feature request on Github and we will do our best to add an implementation for
the operator.

The namespace of an operator describes where it originates from:

* **aten** - operators in this namespace correspond 1:1 to operators in PyTorch's
  `ATen library <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>`_.
  They all support fp16 and fp32 dtypes at a minimum.
* **dim_order_op** - these operators are inserted when lowering to ExecuTorch in
  order to manage optimal tensor memory layouts. They are typically removed,
  since the Vulkan backend manages optimal tensor representations internally.
* **llama** - custom ops targeted for LLM inference. These are typically inserted
  by model source transformations applied to a `nn.Module` and are not invoked
  directly by a PyTorch model.
* **operator** - these operators work with symbolic integers, which are also
  supported by the Vulkan backend.
* **quantized_decomposed** / **torchao** - these ops are introduced by quantization
  workflows (either torchao's `quantize_` API or the PT2E quantization flow).
  They typically represent quantizing/dequantizing a tensor, or choosing the
  quantization parameters for a tensor. In practice, most instances of these
  operators will be fused into a custom op in the **et_vk** namespace.
* **et_vk** - these are custom operators implemented only in the Vulkan backend.
  They typically represent quantized variants of **aten** operators, or fusions
  of common operator patterns. They are inserted by operator fusion graph passes
  when lowering to the Vulkan backend.

All operators support dynamic input shapes unless otherwise noted (i.e. "no
resize support"). The expectation is that over time, all operators will be able
to support dynamic shapes.

.. csv-table:: Vulkan Backend Operator Support
   :file: vulkan-op-support-table.csv
   :header-rows: 1
   :widths: 25 25 75
   :align: left
