===============
Partitioner API
===============

The Neutron partitioner API allows for configuration of the model delegation to Neutron. Passing an ``NeutronPartitioner`` instance with no additional parameters will run as much of the model as possible on the Neutron backend. This is the most common use-case.

It has the following arguments:

* `compile_spec` - list of key-value pairs defining compilation,
* `neutron_target_spec` - NeutronTargetSpec instance, initialized by SoC id, e.g. "imxrt700",
* `custom_delegation_options` - custom options for specifying node delegation,
* `preserve_ops` - list of aten operators to not be decomposed by ExecuTorch.

--------------------
Compile Spec Options
--------------------
To generate the Compile Spec for Neutron backend, you can use the `generate_neutron_compile_spec` function or directly the `NeutronCompileSpecBuilder().neutron_compile_spec()`
Following fields can be set:

* `config` - NXP platform defining the Neutron NPU configuration, e.g. "imxrt700".
* `extra_flags` - Extra flags for the Neutron compiler.
* `operators_not_to_delegate` - List of operators that will not be delegated.
* `use_neutron_for_format_conversion` - If True, let the eIQ Neutron NPU to handle conversion between channel-first (NCHW) and channel-last (NHWC) data formats. That is the Neutron backend will insert `Transpose` ops to ensure that the IO matches the executorch partition, which will be delegated to Neutron.
* `fetch_constants_to_sram`: If True, the Neutron Converter will insert microinstructions to prefetch weights from FLASH to SRAM. This should be used when the whole model does not fit into SRAM on Neutron-C devices, like i.MX RT700
* `dump_kernel_selection_code`: Whether Neutron converter dumps kernel selection code, which is used by the selective kernel registration, see :doc:`Neutron Firmware Kernel Selection support <nxp-kernel-selection.md>`.

-------------------------
Custom Delegation Options
-------------------------
By default the Neutron backend is defensive, what means it does not delegate operators which cannot be decided statically during partitioning. But as the model author you typically have insight into the model and so you can allow opportunistic delegation for some cases. For list of options, see
`CustomDelegationOptions <https://github.com/pytorch/executorch/blob/main/backends/nxp/backend/custom_delegation_options.py#L11>`_

================
Operator Support
================

Operators are the building blocks of the ML model. See `IRs <https://docs.pytorch.org/docs/stable/torch.compiler_ir.html>`_ for more information on the PyTorch operator set.

This section lists the Edge operators supported by the Neutron backend.
For detailed constraints of the operators see the ``is_supported`` / ``_is_supported_in_IR`` / ``_is_supported_on_target`` checks in the `Node converters <https://github.com/pytorch/executorch/blob/main/backends/nxp/backend/ir/converter/node_converter.py#L118>`_


.. csv-table:: Operator Support
   :file: op-support.csv
   :header-rows: 1
   :widths: 20 15 30 30
   :align: center