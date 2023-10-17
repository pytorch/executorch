ExecuTorch Runtime API Reference
================================

The ExecuTorch C++ API provides an on-device execution framework for exported PyTorch models.

For a tutorial style introduction to the runtime API, check out the
`runtime api tutorial <running-a-model-cpp-tutorial.html>`__.

Model Loading and Execution
---------------------------

.. doxygenclass:: torch::executor::DataLoader
  :members:

.. doxygenclass:: torch::executor::MemoryAllocator
  :members:

.. doxygenclass:: torch::executor::HierarchicalAllocator
  :members:

.. doxygenclass:: torch::executor::MemoryManager
  :members:

.. doxygenclass:: torch::executor::Program
  :members:

.. doxygenclass:: torch::executor::Method
  :members:

.. doxygenclass:: torch::executor::MethodMeta
  :members:

Values
------

.. doxygenstruct:: torch::executor::EValue
  :members:

.. doxygenclass:: torch::executor::Tensor
  :members:
