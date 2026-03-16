# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage:

.. code-block:: python

    from pathlib import Path

    import torch
    from executorch.runtime import Runtime, Program, Method

    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(Path("/tmp/program.pte"))
    print("Program methods:", program.method_names)
    forward: Method = program.load_method("forward")

    inputs = (torch.ones(2, 2), torch.ones(2, 2))
    outputs = forward.execute(inputs)
    print(f"Ran forward({inputs})")
    print(f"  outputs: {outputs}")

Example output:

.. code-block:: text

    Program methods: {'forward'}
    Ran forward((tensor([[1., 1.],
            [1., 1.]]), tensor([[1., 1.],
            [1., 1.]])))
      outputs: [tensor([[2., 2.],
            [2., 2.]])]

Example usage with ETDump generation:

Note: ETDump requires building ExecuTorch with event tracing enabled
(CMake option ``EXECUTORCH_ENABLE_EVENT_TRACER=ON``).

.. code-block:: python

    from pathlib import Path
    import os

    import torch
    from executorch.runtime import Runtime, Program, Method

    # Create program with etdump generation enabled
    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(
        Path("/tmp/program.pte"),
        enable_etdump=True,
        debug_buffer_size=int(1e7),  # 10MB buffer to capture all debug info
    )

    # Load method and execute
    forward: Method = program.load_method("forward")
    inputs = (torch.ones(2, 2), torch.ones(2, 2))
    outputs = forward.execute(inputs)

    # Write etdump result to file
    etdump_file = "/tmp/etdump_output.etdp"
    debug_file = "/tmp/debug_output.bin"
    program.write_etdump_result_to_file(etdump_file, debug_file)

    # Check that files were created
    print(f"ETDump file created: {os.path.exists(etdump_file)}")
    print(f"Debug file created: {os.path.exists(debug_file)}")
    print("Directory contents:", os.listdir("/tmp"))

Example output:

.. code-block:: text

    ETDump file created: True
    Debug file created: True
    Directory contents: ['program.pte', 'etdump_output.etdp', 'debug_output.bin']

Example usage with backend and operator introspection:

.. code-block:: python

    from executorch.runtime import Runtime

    runtime = Runtime.get()

    # Check available backends
    backends = runtime.backend_registry.registered_backend_names
    print(f"Available backends: {backends}")

    # Check if a specific backend is available
    if runtime.backend_registry.is_available("XnnpackBackend"):
        print("XNNPACK backend is available")

    # List all registered operators
    operators = runtime.operator_registry.operator_names
    print(f"Number of registered operators: {len(operators)}")

Example output:

.. code-block:: text

    Available backends: ['XnnpackBackend', ...]  # Depends on your build configuration
    XNNPACK backend is available
    Number of registered operators: 247  # Depends on linked kernels
"""

import functools
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Set, Union

try:
    from executorch.extension.pybindings.portable_lib import (  # type: ignore[import-not-found]
        ExecuTorchMethod,
        ExecuTorchProgram,
        MethodMeta,
        Verification,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Prebuilt <site-packages>/extension/pybindings/_portable_lib.so "
        "is not found. Please reinstall ExecuTorch from pip."
    ) from e


class Method:
    """An ExecuTorch method, loaded from a Program.
    This can be used to execute the method with inputs.
    """

    def __init__(self, method: ExecuTorchMethod) -> None:
        self._method = method

    def execute(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """Executes the method with the given inputs.

        Args:
            inputs: A sequence of input values, typically torch.Tensor objects.

        Returns:
            A list of output values, typically torch.Tensor objects.
        """
        return self._method(inputs)

    @property
    def metadata(self) -> MethodMeta:
        """Gets the metadata for the method.

        The metadata includes information about input and output specifications,
        such as tensor shapes, data types, and memory requirements.

        Returns:
            The MethodMeta object containing method specifications.
        """
        return self._method.method_meta()


class Program:
    """An ExecuTorch program, loaded from binary PTE data.

    This can be used to load the methods/models defined by the program.
    """

    def __init__(self, program: ExecuTorchProgram, data: Optional[bytes]) -> None:
        # Hold the data so the program is not freed.
        self._data = data
        self._program = program
        self._methods: Dict[str, Optional[Method]] = {}
        # The names of the methods are preemptively added to the dictionary,
        # but only map to None until they are loaded.
        for method_idx in range(self._program.num_methods()):
            self._methods[self._program.get_method_name(method_idx)] = None

    @property
    def method_names(self) -> Set[str]:
        """Returns method names of the Program as a set of strings."""
        return set(self._methods.keys())

    def load_method(self, name: str) -> Optional[Method]:
        """Loads a method from the program.

        Args:
            name: The name of the method to load.

        Returns:
            The loaded method.
        """

        method = self._methods[name]
        if method is None:
            method = Method(self._program.load_method(name))
            self._methods[name] = method
        return method

    def metadata(self, method_name: str) -> MethodMeta:
        """Gets the metadata for the specified method without loading it.

        Args:
            method_name: The name of the method.

        Returns:
            The metadata for the method, including input/output specifications.
        """
        return self._program.method_meta(method_name)

    def write_etdump_result_to_file(
        self, etdump_path: str, debug_buffer_path: str
    ) -> None:
        """Writes the etdump and debug result to a file.

        Args:
            etdump_path: The path to the etdump file.
            debug_buffer_path: The path to the debug buffer file.
        """
        self._program.write_etdump_result_to_file(etdump_path, debug_buffer_path)


class BackendRegistry:
    """The registry of backends that are available to the runtime."""

    def __init__(self, legacy_module: ModuleType) -> None:
        # TODO: Expose the kernel callables to Python.
        self._legacy_module = legacy_module

    @property
    def registered_backend_names(self) -> List[str]:
        """Returns the names of all registered backends as a list of strings."""
        return self._legacy_module._get_registered_backend_names()

    def is_available(self, backend_name: str) -> bool:
        """Checks if a specific backend is available in the runtime.

        Args:
            backend_name: The name of the backend to check (e.g., "XnnpackBackend").

        Returns:
            True if the backend is available, False otherwise.
        """
        return self._legacy_module._is_available(backend_name)


class OperatorRegistry:
    """The registry of operators that are available to the runtime."""

    def __init__(self, legacy_module: ModuleType) -> None:
        # TODO: Expose the kernel callables to Python.
        self._legacy_module = legacy_module

    @property
    def operator_names(self) -> Set[str]:
        """Returns the names of all registered operators as a set of strings."""
        return set(self._legacy_module._get_operator_names())


class Runtime:
    """An instance of the ExecuTorch runtime environment.

    This can be used to concurrently load and execute any number of ExecuTorch
    programs and methods.

    Attributes:
        backend_registry: Registry for querying available hardware backends.
        operator_registry: Registry for querying available operators/kernels.
    """

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get() -> "Runtime":
        """Gets the Runtime singleton."""
        import executorch.extension.pybindings.portable_lib as legacy_module  # type: ignore[import-not-found]

        return Runtime(legacy_module=legacy_module)

    def __init__(self, *, legacy_module: ModuleType) -> None:
        # Public attributes.
        self.backend_registry = BackendRegistry(legacy_module)
        self.operator_registry = OperatorRegistry(legacy_module)
        # Private attributes.
        self._legacy_module = legacy_module

    def load_program(
        self,
        data: Union[bytes, bytearray, BinaryIO, Path, str],
        *,
        verification: Verification = Verification.InternalConsistency,
        enable_etdump: bool = False,
        debug_buffer_size: int = 0,
    ) -> Program:
        """Loads an ExecuTorch program from a PTE binary.

        Args:
            data: The binary program data to load. Can be a file path (str or Path),
                bytes/bytearray, or a file-like object.
            verification: Level of program verification to perform (Minimal or InternalConsistency).
                Default is InternalConsistency.
            enable_etdump: If True, enables ETDump profiling for runtime performance analysis.
                Default is False.
            debug_buffer_size: Size of the debug buffer in bytes for ETDump data.
                Only used when enable_etdump=True. Default is 0.

        Returns:
            The loaded Program instance.
        """
        if isinstance(data, (Path, str)):
            p = self._legacy_module._load_program(
                str(data),
                enable_etdump=enable_etdump,
                debug_buffer_size=debug_buffer_size,
                program_verification=verification,
            )
            return Program(p, data=None)
        elif isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, bytearray):
            data_bytes = bytes(data)
        elif hasattr(data, "read"):
            # File-like object with read() method
            data_bytes = data.read()
        else:
            raise TypeError(
                f"Expected data to be bytes, bytearray, a path to a .pte file, or a file-like object, but got {type(data).__name__}."
            )
        p = self._legacy_module._load_program_from_buffer(
            data_bytes,
            enable_etdump=enable_etdump,
            debug_buffer_size=debug_buffer_size,
            program_verification=verification,
        )

        return Program(p, data=data_bytes)
