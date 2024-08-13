Export to ExecuTorch API Reference
----------------------------------

.. automodule:: executorch.exir
.. autofunction:: to_edge

.. autoclass:: EdgeProgramManager
    :members: methods, config_methods, exported_program, transform, to_backend, to_executorch

.. autoclass:: ExecutorchProgramManager
    :members: methods, config_methods, exported_program, buffer, debug_handle_map, dump_executorch_program

.. automodule:: executorch.exir.backend.backend_api
.. autofunction:: to_backend

.. autoclass:: LoweredBackendModule
    :members: backend_id, processed_bytes, compile_specs, original_module, buffer, program
