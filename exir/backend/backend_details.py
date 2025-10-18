# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple, Union

from executorch.exir._serialize._named_data_store import NamedDataStoreOutput

from executorch.exir.backend.compile_spec_schema import CompileSpec
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
    # Data Store output created from NamedDataStore.

    # Named Data store contains all the named data that is stored in the PTE file,
    # but retrieveable by delegates via the NamedDataMap at runtime.
    data_store_output: Optional[NamedDataStoreOutput] = None

    # Optional delegate-specific information that will be added to the
    # lowered_module.meta field in the graph, but not directly serialized
    # into the PTE file.
    _delegate_info_meta: Optional[Any] = None


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
        """
        Preprocesses an edge program and returns the preprocess result fo the given backend

        Args:
            edge_program: The original exported program. It will not be modified in place.
            compile_specs: List of values needed for compilation

        Returns:
            PreprocessResult: It wraps the following information:
                processed_bytes -> bytes: A compiled blob - a binary that can run the desired
                program in the backend.
                debug_handle_map (Optional[Dict[int, Tuple[int]]]): For profiling purposes, a
                map from the node_id  in the final graph (either EXIR or the user's self-defined
                IR) to debug handle id attached in the original exported program.
        """
        # Users should return a compiled blob - a binary that can run the desired
        # program in the backend.
        pass

    @classmethod
    def preprocess_multimethod(
        cls,
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, list[PreprocessResult]]:
        """
        Runs preprocess on all partitioned Edge Programs across multiple methods. This allows
        backends to share information across partitioned graphs. Backend can serialize shared
        data by putting the shared data into the data_store_output of the preprocess results.
        This will record the shared data used by that specific partition.

        Default implementation is running the existing preprocess implementation on all

        Args:
            edge_programs: Dictionary mapping the method name to a list of all the partitioned
                           edge_programs from that method to be lowered.
            compile_specs: Dictionary mapping the method name to a list of compile_specs. The
                           list of compile specs maps directly to the list of edge_programs for the
                           same given method name i.e. edge_program[method_name][i] --> compile_specs[method_name][i]

        Returns:
            Dictionary mapping the method name to a list of PreprocessResults. The list of
            PreprocessResults maps directly to the list of edge_programs for the same given
            method name. i.e. edge_program[method_name][i] --> result[method_name][i]


        """
        preprocess_results = {}
        for method_name, programs in edge_programs.items():
            assert (
                method_name in compile_specs
            ), f"Error: missing compile specs for {method_name}"
            compile_specs_for_method = compile_specs[method_name]
            assert len(compile_specs_for_method) == len(
                programs
            ), f"Error: method {method_name} has {len(programs)} partitions but only {len(compile_specs_for_method)}"
            results_for_method = []
            for program, compile_spec_for_program in zip(
                programs, compile_specs_for_method
            ):
                preprocess_result = cls.preprocess(program, compile_spec_for_program)
                results_for_method.append(preprocess_result)

            preprocess_results[method_name] = results_for_method

        return preprocess_results
