Export to ExecuTorch API Reference
----------------------------------

For detailed information on how APIs evolve and the deprecation process, please refer to the `ExecuTorch API Life Cycle and Deprecation Policy <api-life-cycle.html>`__.

.. automodule:: executorch.exir
.. autofunction:: to_edge

.. automodule:: executorch.exir
.. autofunction:: to_edge_transform_and_lower

.. autoclass:: EdgeProgramManager
    :members: methods, config_methods, exported_program, transform, to_backend, to_executorch

.. autoclass:: ExecutorchProgramManager
    :members: methods, config_methods, exported_program, buffer, debug_handle_map, dump_executorch_program

.. automodule:: executorch.exir.backend.backend_api
.. autofunction:: to_backend

.. autoclass:: LoweredBackendModule
    :members: backend_id, processed_bytes, compile_specs, original_module, buffer, program
