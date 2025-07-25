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
    from executorch.runtime import Verification, Runtime, Program, Method

    et_runtime: Runtime = Runtime.get()
    program: Program = et_runtime.load_program(
        Path("/tmp/program.pte"),
        verification=Verification.Minimal,
    )
    print("Program methods:", program.method_names)
    forward: Method = program.load_method("forward")

    inputs = (torch.ones(2, 2), torch.ones(2, 2))
    outputs = forward.execute(inputs)
    print(f"Ran forward({inputs})")
    print(f"  outputs: {outputs}")

Example output:

.. code-block:: text

    Program methods: ('forward', 'forward2')
    Ran forward((tensor([[1., 1.],
            [1., 1.]]), tensor([[1., 1.],
            [1., 1.]])))
      outputs: [tensor([[1., 1.],
            [1., 1.]])]
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
            inputs: The inputs to the method.

        Returns:
            The outputs of the method.
        """
        return self._method(inputs)

    @property
    def metadata(self) -> MethodMeta:
        """Gets the metadata for the method.

        Returns:
            The metadata for the method.
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
        """
        Returns method names of the `Program` as a set of strings.
        """
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
        """Gets the metadata for the specified method.

        Args:
            method_name: The name of the method.

        Returns:
            The outputs of the method.
        """
        return self._program.method_meta(method_name)


class BackendRegistry:
    """The registry of backends that are available to the runtime."""

    def __init__(self, legacy_module: ModuleType) -> None:
        # TODO: Expose the kernel callables to Python.
        self._legacy_module = legacy_module

    @property
    def registered_backend_names(self) -> List[str]:
        """
        Returns the names of all registered backends as a list of strings.
        """
        return self._legacy_module._get_registered_backend_names()

    def is_available(self, backend_name: str) -> bool:
        """
        Returns the names of all registered backends as a list of strings.
        """
        return self._legacy_module._is_available(backend_name)


class OperatorRegistry:
    """The registry of operators that are available to the runtime."""

    def __init__(self, legacy_module: ModuleType) -> None:
        # TODO: Expose the kernel callables to Python.
        self._legacy_module = legacy_module

    @property
    def operator_names(self) -> Set[str]:
        """
        Returns the names of all registered operators as a set of strings.
        """
        return set(self._legacy_module._get_operator_names())


class Runtime:
    """An instance of the ExecuTorch runtime environment.

    This can be used to concurrently load and execute any number of ExecuTorch
    programs and methods.
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
    ) -> Program:
        """Loads an ExecuTorch program from a PTE binary.

        Args:
            data: The binary program data to load; typically PTE data.
            verification: level of program verification to perform.

        Returns:
            The loaded program.
        """
        if isinstance(data, (Path, str)):
            p = self._legacy_module._load_program(
                str(data),
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=verification,
            )
            return Program(p, data=None)
        elif isinstance(data, BinaryIO):
            data_bytes = data.read()
        elif isinstance(data, bytearray):
            data_bytes = bytes(data)
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise TypeError(
                f"Expected data to be bytes, bytearray, a path to a .pte file, or a file-like object, but got {type(data).__name__}."
            )
        p = self._legacy_module._load_program_from_buffer(
            data_bytes,
            enable_etdump=False,
            debug_buffer_size=0,
            program_verification=verification,
        )

        return Program(p, data=data_bytes)
