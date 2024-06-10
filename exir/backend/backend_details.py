# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, Union

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.runtime_info_schema import RuntimeInfo
from torch.export.exported_program import ExportedProgram


def enforcedmethod(func):
    func.__enforcedmethod__ = True
    return func


@dataclass
class PreprocessResult:
    processed_bytes: bytes = bytes()
    debug_handle_map: Optional[Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]] = (
        None
    )


"""
How to create a backend (for example, BackendWithCompilerDemo):
1. Create a python file, like backend_with_compiler_demo.py, with
a custom class BackendWithCompilerDemo, derived from BackendDetails.

How to use the backend (for example, BackendWithCompilerDemo):
2. Import this class, like
from executorch.exir.backend.backend_with_compiler_demo import BackendWithCompilerDemo
"""


class BackendDetails(ABC):
    """
    A base interface to lower the implementation to the according backend. With
    the decorators, this interface will be static, abstract and all inheritances are
    enforced to implement this method.

    Args:
        edge_program: The original exported program. It will not be modified in place.
        compile_specs: List of values needed for compilation

    Returns:
        PreprocessResult: It wraps the following information:
            processed_bytes -> bytes: A compiled blob - a binary that can run the desired program in the backend.
            debug_handle_map (Optional[Dict[int, Tuple[int]]]): For profiling purposes, a map from the node_id in the final graph (either EXIR or the user's self-defined IR)
            to debug handle id attached in the original exported program.
    """

    @staticmethod
    # all backends need to implement this method
    @enforcedmethod
    # it's a virtual method and inheritant class needs to implement the actual function
    @abstractmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        # Users should return a compiled blob - a binary that can run the desired
        # program in the backend.
        pass

    """
    Args:
        processed_bytes: The preprocessed binary result from preprocessed given the graph.
        compile_spec: The same compile spec given to preprocess.
        runtime_info: The runtime info provided by the backend runtime

    Returns:
        A boolean to indicated whether the preprocessed result is compatible with the backend runtime.
    """

    @staticmethod
    @abstractmethod
    def is_compatible(
        processed_bytes: bytes,
        compile_spec: List[CompileSpec],
        runtime_info: List[RuntimeInfo],
    ) -> bool:
        # Users should return if the preprocesssed_blob is compatible with the runtime info.
        pass
