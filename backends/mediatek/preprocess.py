# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import collections
import contextlib
import struct

from typing import Dict, final, List

import mtk_converter
import mtk_neuron
import torch
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

SKIP_COMPILE_SPEC_KEYS = {"ImportForever"}
EXTRACT_SHARED_BLOB_KEY = "ExtractSharedBlobKey"
HEADER_SIZE = 13
HEADER_VERSION = 1
REQUIRED_COMPILE_SPEC_KEYS = {"platform-config"}
SUPPORTED_PLATFORM_CONFIGS = {"mt6989", "mt6991", "mt6993"}


def assert_default_dim_order(edge_graph_module: torch.fx.GraphModule) -> None:
    for node in edge_graph_module.graph.nodes:
        if node.op != "placeholder":
            continue

        # We expect the default dim order for all tensor-like inputs i.e. inputs, buffers, and params
        t = node.meta.get("val", None)
        if t is not None and getattr(t, "dim_order", None) is not None:
            default_dim_order = tuple(range(t.dim()))
            if t.dim_order() != default_dim_order:
                raise RuntimeError(
                    f"Neuropilot backend only supports contiguous memory format for inputs."
                    f"Expecting dim_order: {default_dim_order}, but got "
                    f"{node.meta['val'].dim_order()} for a placeholder node {node}."
                )


def _pack_header(num_inputs, num_outputs, model_bytes_size):
    header_bytes = struct.pack(
        "<BIII", HEADER_VERSION, num_inputs, num_outputs, model_bytes_size
    )
    assert len(header_bytes) == HEADER_SIZE
    return header_bytes


def _unpack_header(header_bytes):
    assert len(header_bytes) == HEADER_SIZE
    version, num_inputs, num_outputs, buffer_size = struct.unpack("<BIII", header_bytes)
    assert version == HEADER_VERSION
    return num_inputs, num_outputs, buffer_size


@final
class NeuropilotBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:

        # Validate CompileSpec settings
        compile_spec_keys = [spec.key for spec in module_compile_spec]
        if len(compile_spec_keys) != len(set(compile_spec_keys)):
            raise RuntimeError(
                "Unsupported duplicated keys in the CompileSpec settings."
            )
        if not REQUIRED_COMPILE_SPEC_KEYS.issubset(set(compile_spec_keys)):
            raise RuntimeError(
                "Following keys are required in the CompileSpec settings: {}."
                "".format(REQUIRED_COMPILE_SPEC_KEYS)
            )
        platform = [
            spec.value.decode("utf-8")
            for spec in module_compile_spec
            if spec.key == "platform-config"
        ][0]
        if platform not in SUPPORTED_PLATFORM_CONFIGS:
            raise ValueError(
                "Unsupported value of platform-config CompileSpec. Given {} but expected to be one "
                "of {}.".format(platform, SUPPORTED_PLATFORM_CONFIGS)
            )

        # Make sure all inputs are contiguous_format or NCHW or default dim order
        assert_default_dim_order(edge_program.graph_module)

        name_to_node_mappings = {node.name: node for node in edge_program.graph.nodes}
        input_names = edge_program.graph_signature.user_inputs
        output_names = edge_program.graph_signature.user_outputs
        fp_input_indices = [
            idx
            for idx, name in enumerate(input_names)
            if name_to_node_mappings[name].meta["val"].dtype == torch.float32
        ]
        fp_output_indices = [
            idx
            for idx, name in enumerate(output_names)
            if name_to_node_mappings[name].meta["val"].dtype == torch.float32
        ]

        compile_options = ["--relax-fp32", "--opt=3"]
        for spec in module_compile_spec:
            # Special compile spec handling
            if spec.key in SKIP_COMPILE_SPEC_KEYS:
                continue
            if spec.key == EXTRACT_SHARED_BLOB_KEY:
                compile_options.append("--dla-opt=0")
                continue

            # General compile spec handling
            if spec.value == b"":
                compile_options.append(f"--{spec.key}")
            else:
                value = spec.value.decode("utf-8")
                compile_options.append(f"--{spec.key}={value}")

        converter = mtk_converter.PyTorchV2Converter.from_exported_program(edge_program)
        converter.quantize = True
        converter.input_quantization_bitwidths = None
        converter.allow_missing_quantization_ranges = True
        converter.prepend_input_quantize_ops = True
        converter.prepend_input_quantize_ops_indices = fp_input_indices
        converter.append_output_dequantize_ops = True
        converter.append_output_dequantize_ops_indices = fp_output_indices
        with contextlib.redirect_stdout(None):
            mlir_str = converter.convert_to_mlir()
            model_bytes = mtk_neuron.compile(mlir_str, " ".join(compile_options))

        num_inputs = len(input_names)
        num_outputs = len(output_names)
        header_bytes = _pack_header(num_inputs, num_outputs, len(model_bytes))
        return PreprocessResult(processed_bytes=bytes(header_bytes + model_bytes))

    @classmethod
    def preprocess_multimethod(
        cls,
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, list[PreprocessResult]]:

        # Follow the default behavior of `preprocess_multimethod`
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

        # Try extract shared data blob if necessary
        infos_dict = collections.defaultdict(list)
        models_dict = collections.defaultdict(list)
        result_dict = collections.defaultdict(list)
        for method_name, method_results in preprocess_results.items():
            for idx, result in enumerate(method_results):
                shared_blob_key = None
                for spec in compile_specs[method_name][idx]:
                    if spec.key == EXTRACT_SHARED_BLOB_KEY:
                        shared_blob_key = spec.value.decode("utf-8")

                if shared_blob_key is None:
                    continue

                header_bytes = result.processed_bytes[:HEADER_SIZE]
                model_bytes = result.processed_bytes[HEADER_SIZE:]
                num_inputs, num_outputs, model_bytes_size = _unpack_header(header_bytes)
                assert len(model_bytes) == model_bytes_size
                infos_dict[shared_blob_key].append((num_inputs, num_outputs))
                models_dict[shared_blob_key].append(model_bytes)
                result_dict[shared_blob_key].append(result)

        data_store_output_dict = {}
        for key, models in models_dict.items():
            ndm = NamedDataStore()
            blob, new_models = mtk_neuron.extract_shared_data(
                models, options="-e union"
            )
            ndm.add_named_data(key, bytes(blob))
            data_store_output_dict[key] = ndm.get_named_data_store_output()
            models.clear()
            models.extend(new_models)

        for key, data_store_output in data_store_output_dict.items():
            for idx, (model_info, model_bytes) in enumerate(
                zip(infos_dict[key], models_dict[key])
            ):
                num_inputs, num_outputs = model_info
                header_bytes = _pack_header(num_inputs, num_outputs, len(model_bytes))
                result_dict[key][idx].data_store_output = data_store_output
                result_dict[key][idx].processed_bytes = bytes(
                    header_bytes + model_bytes
                )

        return preprocess_results
