===============
Partitioner API
===============

The Neutron partitioner API allows for configuration of the model delegation to Neutron. Passing an ``NeutronPartitioner`` instance with no additional parameters will run as much of the model as possible on the Neutron backend. This is the most common use-case.

It has the following arguments:

* `compile_spec` - list of key-value pairs defining compilation:
* `custom_delegation_options` - custom options for specifying node delegation.

--------------------
Compile Spec Options
--------------------
To generate the Compile Spec for Neutron backend, you can use the `generate_neutron_compile_spec` function or directly the `NeutronCompileSpecBuilder().neutron_compile_spec()`
Following fields can be set:

* `config` - NXP platform defining the Neutron NPU configuration, e.g. "imxrt700".
* `neutron_converter_flavor` - Flavor of the neutron-converter module to use. Neutron-converter module named neutron_converter_SDK_25_06' has flavor 'SDK_25_06'. You shall set the flavour to the MCUXpresso SDK version you will use.
* `extra_flags` - Extra flags for the Neutron compiler.
* `operators_not_to_delegate` - List of operators that will not be delegated.

-------------------------
Custom Delegation Options
-------------------------
By default the Neutron backend is defensive, what means it does not delegate operators which cannot be decided statically during partitioning. But as the model author you typically have insight into the model and so you can allow opportunistic delegation for some cases. For list of options, see
`CustomDelegationOptions <https://github.com/pytorch/executorch/blob/release/1.0/backends/nxp/backend/custom_delegation_options.py#L11>`_

================
Operator Support
================

Operators are the building blocks of the ML model. See `IRs <https://docs.pytorch.org/docs/stable/torch.compiler_ir.html>`_ for more information on the PyTorch operator set.

This section lists the Edge operators supported by the Neutron backend.
For detailed constraints of the operators see the conditions in the ``is_supported_*`` functions in the `Node converters <https://github.com/pytorch/executorch/blob/release/1.0/backends/nxp/neutron_partitioner.py#L192>`_


.. csv-table:: Operator Support
   :file: op-support.csv
   :header-rows: 1
   :widths: 20 15 30 30
   :align: center