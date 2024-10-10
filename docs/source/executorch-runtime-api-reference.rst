ExecuTorch Runtime API Reference
================================

The ExecuTorch C++ API provides an on-device execution framework for exported PyTorch models.

For a tutorial style introduction to the runtime API, check out the
`runtime tutorial <running-a-model-cpp-tutorial.html>`__ and its `simplified <extension-module.html>`__ version.

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
