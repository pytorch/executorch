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
from typing import Any, BinaryIO, Dict, Optional, Sequence, Set, Union

try:
    from executorch.extension.pybindings.portable_lib import (
        ExecuTorchModule,
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

    def __init__(self, method_name: str, module: ExecuTorchModule) -> None:
        # TODO: This class should be pybind to the C++ counterpart instead of hosting ExecuTorchModule.
        self._method_name = method_name
        self._module = module

    def execute(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """Executes the method with the given inputs.

        Args:
            inputs: The inputs to the method.

        Returns:
            The outputs of the method.
        """
        return self._module.run_method(self._method_name, inputs)

    @property
    def metadata(self) -> MethodMeta:
        """Gets the metadata for the method.

        Returns:
            The metadata for the method.
        """
        return self._module.method_meta(self._method_name)


class Program:
    """An ExecuTorch program, loaded from binary PTE data.

    This can be used to load the methods/models defined by the program.
    """

    def __init__(self, module: ExecuTorchModule, data: Optional[bytes]) -> None:
        # Hold the data so the program is not freed.
        self._data = data
        self._module = module
        self._methods: Dict[str, Method] = {}
        # ExecuTorchModule already pre-loads all Methods when created, so this
        # doesn't do any extra work. TODO: Don't load a given Method until
        # load_method() is called. Create a separate Method instance each time,
        # to allow multiple independent instances of the same model.
        for method_name in self._module.method_names():
            self._methods[method_name] = Method(method_name, self._module)

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
        return self._methods.get(name, None)


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
        import executorch.extension.pybindings.portable_lib as legacy_module

        return Runtime(legacy_module=legacy_module)

    def __init__(self, *, legacy_module: ModuleType) -> None:
        # Public attributes.
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
            m = self._legacy_module._load_for_executorch(
                str(data),
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=verification,
            )
            return Program(m, data=None)
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
        m = self._legacy_module._load_for_executorch_from_buffer(
            data_bytes,
            enable_etdump=False,
            debug_buffer_size=0,
            program_verification=verification,
        )

        return Program(m, data=data_bytes)
