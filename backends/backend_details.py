from abc import ABC, abstractmethod

from typing import Callable, Dict, List

from executorch.backends.compile_spec_schema import CompileSpec

from executorch.exir import ExportGraphModule
from torch.fx.node import Node


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
        edge_ir_module: The original module. It will not be modified in place.
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
        edge_ir_module: ExportGraphModule,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        # Users should return a compiled blob - a binary that can run the desired
        # program in the backend.
        pass
