ExecuTorch Runtime Python API Reference
----------------------------------
The Python ``executorch.runtime`` module wraps the C++ ExecuTorch runtime. It can load and execute serialized ``.pte`` program files: see the `Export to ExecuTorch Tutorial <tutorials/export-to-executorch-tutorial.html>`__ for how to convert a PyTorch ``nn.Module`` to an ExecuTorch ``.pte`` program file. Execution accepts and returns ``torch.Tensor`` values, making it a quick way to validate the correctness of the program.

For detailed information on how APIs evolve and the deprecation process, please refer to the `ExecuTorch API Life Cycle and Deprecation Policy <api-life-cycle.html>`__.

.. automodule:: executorch.runtime
.. autoclass:: Runtime
    :members: get, load_program

.. autoclass:: OperatorRegistry
    :members: operator_names

.. autoclass:: Program
    :members: method_names, load_method

.. autoclass:: Method
    :members: execute, metadata
