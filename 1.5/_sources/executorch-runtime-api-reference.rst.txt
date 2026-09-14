Runtime API Reference
================================

The ExecuTorch C++ API provides an on-device execution framework for exported PyTorch models.

For a tutorial style introduction to the runtime API, check out the
`using executorch with cpp tutorial <using-executorch-cpp.html>`__ and its `simplified <extension-module.html>`__ version.

For detailed information on how APIs evolve and the deprecation process, please refer to the `ExecuTorch API Life Cycle and Deprecation Policy <api-life-cycle.html>`__.

Model Loading and Execution
---------------------------

.. doxygenclass:: executorch::runtime::Program
  :members:

.. doxygenclass:: executorch::runtime::Method
  :members:

.. doxygenclass:: executorch::runtime::MethodMeta
  :members:

.. doxygenclass:: executorch::runtime::DataLoader
  :members:

.. doxygenclass:: executorch::runtime::MemoryAllocator
  :members:

.. doxygenclass:: executorch::runtime::HierarchicalAllocator
  :members:

.. doxygenclass:: executorch::runtime::MemoryManager
  :members:

Values
------

.. doxygenstruct:: executorch::runtime::EValue
  :members:

.. doxygenclass:: executorch::runtime::etensor::Tensor
  :members:

Module Extension
----------------

The Module extension provides a higher-level C++ facade for loading programs,
setting inputs and outputs, and executing methods with common runtime defaults.

.. doxygenclass:: executorch::extension::module::Module
  :members:

.. doxygenclass:: executorch::extension::bundled_module::BundledModule
  :members:

Tensor Extension
----------------

The Tensor extension provides managed tensor helpers for C++ applications that
need to create, alias, resize, or index tensors before passing them to runtime
APIs.

.. doxygennamespace:: executorch::extension
  :members:
  :content-only:
