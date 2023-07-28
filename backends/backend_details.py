# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from typing import List

from executorch.backends.compile_spec_schema import CompileSpec
from torch._export.exported_program import ExportedProgram


def enforcedmethod(func):
    func.__enforcedmethod__ = True
    return func


"""
How to create a backend (for example, BackendWithCompilerDemo):
1. Create a python file, like backend_with_compiler_demo.py, with
a custom class BackendWithCompilerDemo, derived from BackendDetails.

How to use the backend (for example, BackendWithCompilerDemo):
2. Import this class, like
from executorch.backends.backend_with_compiler_demo import BackendWithCompilerDemo
"""


class BackendDetails(ABC):
    """
    A base interface to lower the implementation to the according backend. With
    the decorators, this interface will be static, abstract and all inheritances are
    enforced to implement this method.

    Args:
        edge_program: The original exported program. It will not be modified in place.
        backend_debug_handle_generator: A callable to map a graph to a dictionary (key is node, value is id)
        compile_specs: List of values needed for compilation

    Returns:
        Bytes: A compiled blob - a binary that can run the desired program in the backend.
    """

    @staticmethod
    # all backends need to implement this method
    @enforcedmethod
    # it's a virtual method and inheritant class needs to implement the actual function
    @abstractmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        # Users should return a compiled blob - a binary that can run the desired
        # program in the backend.
        pass
