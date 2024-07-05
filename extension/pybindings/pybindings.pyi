# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Dict, List, Optional, Sequence, Tuple

class ExecuTorchModule:
    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def __call__(self, inputs: Any) -> List[Any]: ...
    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def run_method(self, method_name: str, inputs: Sequence[Any]) -> List[Any]: ...
    # pyre-ignore[2, 3]: "Any" in parameter and return type annotations.
    def forward(self, inputs: Sequence[Any]) -> List[Any]: ...
    # Bundled program methods.
    def load_bundled_input(
        self, bundle: BundledModule, method_name: str, testset_idx: int
    ) -> None: ...
    def verify_result_with_bundled_expected_output(
        self,
        bundle: BundledModule,
        method_name: str,
        testset_idx: int,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None: ...
    def has_etdump(self) -> bool: ...
    def write_etdump_result_to_file(
        self, path: str, debug_buffer_path: Optional[str] = None
    ) -> None: ...

class BundledModule: ...

def _load_for_executorch(
    path: str, enable_etdump: bool = False, debug_buffer_size: int = 0
) -> ExecuTorchModule:
    """Load an ExecuTorch Program from a file.
    Args:
        path: File path to the ExecuTorch program as a string.
        enable_etdump: If true, enables an ETDump which can store profiling information.
            See documentation at https://pytorch.org/executorch/stable/sdk-etdump.html
            for how to use it.
        debug_buffer_size: If non-zero, enables a debug buffer which can store
            intermediate results of each instruction in the ExecuTorch program.
            This is the fixed size of the buffer, if you have more intermediate
            result bytes than this allows, the execution will abort with a failed
            runtime check.
    """
    ...

def _load_for_executorch_from_buffer(
    buffer: bytes, enable_etdump: bool = False, debug_buffer_size: int = 0
) -> ExecuTorchModule:
    """Same as _load_for_executorch, but takes a byte buffer instead of a file path."""
    ...

def _load_for_executorch_from_bundled_program(
    module: BundledModule, enable_etdump: bool = False, debug_buffer_size: int = 0
) -> ExecuTorchModule:
    """Same as _load_for_executorch, but takes a bundled program instead of a file path.
    See https://pytorch.org/executorch/stable/sdk-bundled-io.html for documentation."""
    ...

def _load_bundled_program_from_buffer(
    buffer: bytes, non_const_pool_size: int = ...
) -> BundledModule: ...
def _get_operator_names() -> List[str]: ...
def _create_profile_block(name: str) -> None: ...
def _dump_profile_results() -> bytes: ...
def _reset_profile_results() -> None: ...
