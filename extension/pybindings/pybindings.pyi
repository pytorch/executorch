# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

from typing import Any, Dict, Enum, List, Optional, Sequence, Tuple

from executorch.exir._warnings import experimental

@experimental("This API is experimental and subject to change without notice.")
class Verification(Enum):
    """Verification maps C++ Program::Verification to Python.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    Minimal: ...
    InternalConsistency: ...

@experimental("This API is experimental and subject to change without notice.")
class ExecuTorchModule:
    """ExecuTorchModule is a Python wrapper around a C++ ExecuTorch program.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def __call__(self, inputs: Any, clone_outputs: bool = True) -> List[Any]: ...
    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def run_method(
        self,
        method_name: str,
        inputs: Sequence[Any],  # pyre-ignore[2]: "Any" in parameter type annotations.
        clone_outputs: bool = True,
    ) -> List[Any]: ...
    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def forward(
        self,
        inputs: Sequence[Any],  # pyre-ignore[2]: "Any" in parameter type annotations.
        clone_outputs: bool = True,
    ) -> List[Any]: ...
    # pyre-ignore[3]: "Any" in return type annotations.
    def plan_execute(self) -> List[Any]: ...
    # Bundled program methods.
    def load_bundled_input(
        self, bundle: BundledModule, method_name: str, testset_idx: int
    ) -> None: ...
    # pyre-ignore[3]: "Any" in return type annotations.
    def verify_result_with_bundled_expected_output(
        self,
        bundle: BundledModule,
        method_name: str,
        testset_idx: int,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> List[Any]: ...
    def has_etdump(self) -> bool: ...
    def write_etdump_result_to_file(
        self, path: str, debug_buffer_path: Optional[str] = None
    ) -> None: ...
    def method_meta(self, method_name: str) -> MethodMeta: ...
    def method_names(self) -> List[str]: ...

@experimental("This API is experimental and subject to change without notice.")
class BundledModule:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """

    ...

@experimental("This API is experimental and subject to change without notice.")
class TensorInfo:
    """Metadata about a tensor such as the shape and dtype.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    def sizes(self) -> Tuple[int, ...]:
        """Shape of the tensor as a tuple"""
        ...

    def dtype(self) -> int:
        """The data type of the elements inside the tensor.
        See documentation for ScalarType in executorch/runtime/core/portable_type/scalar_type.h
        for the values these integers can take."""
        ...

    def is_memory_planned(self) -> bool:
        """True if the tensor is already memory planned, meaning no allocation
        needs to be provided. False otherwise"""
        ...

    def nbytes(self) -> int:
        """Number of bytes in the tensor. Not the same as numel if the dtype is
        larger than 1 byte wide"""
        ...

    def __repr__(self) -> str: ...

@experimental("This API is experimental and subject to change without notice.")
class MethodMeta:
    """Metadata about a method such as the number of inputs and outputs.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    def name(self) -> str:
        """The name of the method, such as 'forward'"""
        ...

    def num_inputs(self) -> int:
        """The number of user inputs to the method. This does not include any
        internal buffers or weights, which don't need to be provided by the user"""
        ...

    def num_outputs(self) -> int:
        """The number of outputs from the method. This does not include any mutated
        internal buffers"""
        ...

    def num_attributes(self) -> int:
        """The number of attribute tensors from the method"""
        ...

    def input_tensor_meta(self, index: int) -> TensorInfo:
        """The tensor info for the 'index'th input. Index must be in the interval
        [0, num_inputs()). Raises an IndexError if the index is out of bounds"""
        ...

    def output_tensor_meta(self, index: int) -> TensorInfo:
        """The tensor info for the 'index'th output. Index must be in the interval
        [0, num_outputs()). Raises an IndexError if the index is out of bounds"""
        ...

    def attribute_tensor_meta(self, index: int) -> TensorInfo:
        """The tensor info for the 'index'th attribute. Index must be in the interval
        [0, num_attributes()). Raises an IndexError if the index is out of bounds"""
        ...

    def __repr__(self) -> str: ...

@experimental("This API is experimental and subject to change without notice.")
def _load_for_executorch(
    program_path: str,
    data_path: Optional[str] = None,
    enable_etdump: bool = False,
    debug_buffer_size: int = 0,
    program_verification: Verification = Verification.InternalConsistency,
) -> ExecuTorchModule:
    """Load an ExecuTorch Program from a file.

    .. warning::

        This API is experimental and subject to change without notice.

    Args:
        program_path: File path to the ExecuTorch program as a string.
        data_path: File path to a .ptd file containing data used by the program.
        enable_etdump: If true, enables an ETDump which can store profiling information.
            See documentation at https://pytorch.org/executorch/main/etdump
            for how to use it.
        debug_buffer_size: If non-zero, enables a debug buffer which can store
            intermediate results of each instruction in the ExecuTorch program.
            This is the fixed size of the buffer, if you have more intermediate
            result bytes than this allows, the execution will abort with a failed
            runtime check.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _load_for_executorch_from_buffer(
    buffer: bytes,
    data_map_buffer: Optional[bytes] = None,
    enable_etdump: bool = False,
    debug_buffer_size: int = 0,
    program_verification: Verification = Verification.InternalConsistency,
) -> ExecuTorchModule:
    """Same as _load_for_executorch, but takes a byte buffer instead of a file path.

    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _load_for_executorch_from_bundled_program(
    module: BundledModule,
    data_map_buffer: Optional[bytes] = None,
    enable_etdump: bool = False,
    debug_buffer_size: int = 0,
) -> ExecuTorchModule:
    """Same as _load_for_executorch, but takes a bundled program instead of a file path.

    See https://pytorch.org/executorch/main/bundled-io for documentation.

    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _load_bundled_program_from_buffer(
    buffer: bytes, non_const_pool_size: int = ...
) -> BundledModule:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _is_available(backend_name: str) -> bool:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _get_operator_names() -> List[str]:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _get_registered_backend_names() -> List[str]:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _create_profile_block(name: str) -> None:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _dump_profile_results() -> bytes:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _reset_profile_results() -> None:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _unsafe_reset_threadpool(num_threads: int) -> None:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _threadpool_get_thread_count() -> int:
    """
    .. warning::

        This API is experimental and subject to change without notice.
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
class FlatTensorDataMap:
    """FlatTensorDataMap loads external data from a .ptd file.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    def get_named_data_map(self) -> Any:
        """Get a pointer to the underlying NamedDataMap.

        Returns:
            A capsule containing a pointer to the internal NamedDataMap
            that can be passed to ExecuTorchProgram.load_method().

        Warning:
            The FlatTensorDataMap instance must outlive the returned capsule.
        """
        ...

@experimental("This API is experimental and subject to change without notice.")
def _load_flat_tensor_data_map(
    data_path: str,
) -> FlatTensorDataMap:
    """Load a flat tensor data map from a file.

    .. warning::

        This API is experimental and subject to change without notice.

    Args:
        data_path: Path to the .ptd file with external data.

    Returns:
        A FlatTensorDataMap instance that can be used with ExecuTorchProgram.load_method().
    """
    ...

@experimental("This API is experimental and subject to change without notice.")
def _load_flat_tensor_data_map_from_buffer(
    data_buffer: bytes,
) -> FlatTensorDataMap:
    """Load a flat tensor data map from a buffer.

    .. warning::

        This API is experimental and subject to change without notice.

    Args:
        data_buffer: Buffer holding a .ptd file with external data.

    Returns:
        A FlatTensorDataMap instance that can be used with ExecuTorchProgram.load_method().
    """
    ...
