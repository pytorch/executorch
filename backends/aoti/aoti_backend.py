# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import torch
from executorch.backends.aoti.passes.replace_view_copy_with_view import (
    ReplaceViewCopyWithViewPass,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import ExportedProgram, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch.export.passes import move_to_device_pass


class COMPILE_SPEC_KEYS(Enum):
    METHOD_NAME = "method_name"


@experimental(
    "This API and all of aoti-driven backend related functionality are experimental."
)
class AotiBackend(ABC):
    """
    Base mixin class for AOTInductor-based backends.

    This class provides common functionality for compiling models using AOTInductor
    with different device targets (CUDA, Metal, etc.).

    This is a mixin class, not an actual backend object, for aoti-driven backends.
    Concrete backends (e.g., CudaBackend, MetalBackend) should inherit from both
    BackendDetails and AotiBackend to get the full functionality.
    """

    @classmethod
    @abstractmethod
    def get_device_name(cls) -> str:
        """Return the device name for this backend (e.g., 'cuda', 'metal')."""
        pass

    @classmethod
    @abstractmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        """Return the set of supported fallback kernels for this backend."""
        pass

    @classmethod
    @abstractmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        """Return the decomposition table for this backend."""
        pass

    @classmethod
    @abstractmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, typing.Any]:
        """Return the AOTInductor compilation options for this backend."""
        pass

    @classmethod
    @abstractmethod
    def get_custom_passes(cls) -> List[typing.Any]:
        """Return the list of custom passes to apply after ReplaceViewCopyWithViewPass and before decomposition."""
        pass

    @classmethod
    @contextlib.contextmanager
    def collect_unsupported_fallback_kernels(cls, missing_fallback_kernels: Set[str]):
        """
        Context manager to collect unsupported fallback kernels during compilation.
        Monitors both extern kernel calls and runtime lookup.
        """
        supported_kernels = cls.get_supported_fallback_kernels()

        original_generate_c_shim_extern_kernel_call = (
            CppWrapperCpu.generate_c_shim_extern_kernel_call
        )
        original_generate_fallback_kernel_with_runtime_lookup_aot = (
            CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot
        )

        def generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels(
            self,
            kernel: str,
            args: list[str],
            device: str,
            *,
            debug_args: Optional[list[str]] = None,
            debug_handle: Optional[int] = None,
        ):
            if kernel not in supported_kernels:
                missing_fallback_kernels.add(kernel)

            original_generate_c_shim_extern_kernel_call(
                self,
                kernel,
                args,
                device,
                debug_args=debug_args,
                debug_handle=debug_handle,
            )

        def generate_fallback_kernel_with_runtime_lookup_aot_and_collect_unsupported_kernels(
            self,
            op_overload,
            raw_args,
            output_args,
            raw_outputs,
        ):
            kernel_name = getattr(op_overload, "_name", str(op_overload))
            if kernel_name not in supported_kernels:
                missing_fallback_kernels.add(kernel_name)

            original_generate_fallback_kernel_with_runtime_lookup_aot(
                self, op_overload, raw_args, output_args, raw_outputs
            )

        CppWrapperCpu.generate_c_shim_extern_kernel_call = (
            generate_c_shim_extern_kernel_call_and_collect_unsupported_kernels
        )
        CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot = generate_fallback_kernel_with_runtime_lookup_aot_and_collect_unsupported_kernels

        try:
            yield
        finally:
            CppWrapperCpu.generate_c_shim_extern_kernel_call = (
                original_generate_c_shim_extern_kernel_call
            )
            CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot = (
                original_generate_fallback_kernel_with_runtime_lookup_aot
            )

    @classmethod
    def preprocess(
        cls,
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """
        Preprocess the edge program and compile it using AOTInductor.
        Weights are always separated from the SO file.
        """
        device_name = cls.get_device_name()
        decomposition_table = cls.get_decomposition_table()
        options = cls.get_aoti_compile_options(compile_specs)

        # Move the edge_program to the target device
        device_edge_program = move_to_device_pass(
            edge_program, device_name if device_name != "metal" else "mps"
        )

        # Replace view_copy with view
        ReplaceViewCopyWithViewPass()(device_edge_program.graph_module)

        # Apply custom backend-specific passes
        custom_passes = cls.get_custom_passes()
        for custom_pass in custom_passes:
            custom_pass(device_edge_program.graph_module)

        # Run decompositions if any
        if decomposition_table:
            device_edge_program = device_edge_program.run_decompositions(
                decomposition_table
            )

        edge_program_module = device_edge_program.module()

        # Grab all input placeholders from the graph
        user_input_names = device_edge_program.graph_signature.user_inputs
        user_input_placeholders = []
        for node in device_edge_program.graph.nodes:
            if node.op == "placeholder" and node.name in user_input_names:
                user_input_placeholders.append(node.meta["val"])

        # Track missing fallback kernels
        missing_fallback_kernels: Set[str] = set()

        # Compile with fallback kernel collection
        with cls.collect_unsupported_fallback_kernels(
            missing_fallback_kernels
        ), torch.no_grad():
            paths = torch._inductor.aot_compile(
                edge_program_module, tuple(user_input_placeholders), options=options
            )

            if len(missing_fallback_kernels) > 0:
                formatted_kernels = "\n  - ".join(sorted(missing_fallback_kernels))
                method_name = cls.method_name_from_compile_specs(compile_specs)
                raise RuntimeError(
                    f"Method {method_name} missing fallback kernels ({len(missing_fallback_kernels)} total):\n  - {formatted_kernels}\n"
                    "Please add them to the AOTI backend."
                )

        # Extract paths - weights are always separated
        so_path = None
        blob_path = None

        if isinstance(paths, list):
            for path in paths:
                if path.endswith(".wrapper.so"):
                    so_path = path
                elif path.endswith(".wrapper_weights.blob"):
                    blob_path = path
        else:
            so_path = paths

        if so_path is None or blob_path is None:
            raise RuntimeError(
                f"Could not find required files in compiled paths, got {paths}"
            )

        # Read SO file
        with open(so_path, "rb") as f:
            so_data = f.read()

        # Read weights blob
        with open(blob_path, "rb") as f:
            blob_data = f.read()

        # Create named data store
        named_data_store = NamedDataStore()
        method_name = cls.method_name_from_compile_specs(compile_specs)

        # Add SO and weights blob separately
        named_data_store.add_named_data(method_name + "_so_blob", so_data, 1, None)
        weights_blob_data_type = f"aoti_{device_name}_blob"
        named_data_store.add_named_data(
            method_name + "_weights_blob", blob_data, 1, weights_blob_data_type
        )

        # Clean up the generated files
        os.remove(so_path)
        os.remove(blob_path)

        return PreprocessResult(
            processed_bytes=b"",
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )

    @classmethod
    def generate_method_name_compile_spec(
        cls,
        method_name: str,
    ) -> CompileSpec:
        """
        Generate a CompileSpec for the given method name.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.METHOD_NAME.value,
            method_name.encode("utf-8"),
        )

    @classmethod
    def method_name_from_compile_specs(
        cls,
        compile_specs: List[CompileSpec],
    ) -> str:
        """
        Extract the method name from the compile specs.
        """
        for spec in compile_specs:
            if spec.key == COMPILE_SPEC_KEYS.METHOD_NAME.value:
                return spec.value.decode("utf-8")
        raise RuntimeError(
            f"Could not find method name in compile specs: {compile_specs}"
        )
