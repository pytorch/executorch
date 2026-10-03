# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
from typing import Dict, final, List

import executorch.backends.samsung.python.PyEnnWrapperAdaptor as PyEnnWrapper
import torch
from executorch.backends.samsung._passes.enn_pass_manager import EnnPassManager
from executorch.backends.samsung.builders.node_visitor import get_node_visitors
from executorch.backends.samsung.serialization.compile_options import (
    ENN_COMPILE_OPTION_TITLE,
    ENN_COMPILE_WEIGHT_BUFFER_HASH_ID,
    get_weight_sharing_flag_from_compile_spec,
    WeightSharingFlag,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.samsung.utils.utils import get_compile_spec

from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from torch.export.exported_program import ExportedProgram


@final
class EnnBackend(BackendDetails):

    @staticmethod
    def _build_enn_graph_from_program(
        edge_program: ExportedProgram,
    ) -> bytes:
        """Build ENN graph from an exported program and return serialized buffer.
        Args:
            edge_program: The exported program to process.

        Returns:
            The serialized ENN graph buffer.
        """
        graph_module = EnnPassManager().transform_for_preprocess_pass(edge_program)
        assert graph_module is not None

        enn_graph = EnnGraph()
        # node visitors
        node_visitors = get_node_visitors(edge_program)

        vals_to_ids: Dict[torch.fx.Node, int] = {}
        placeholder_vistor = node_visitors["placeholder"]
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                logging.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    node_visitors[node.target.__name__].define_node(
                        node, enn_graph, vals_to_ids
                    )
                else:
                    raise RuntimeError(
                        f"{node.target.__name__}" " is not supported in ENN Delegate"
                    )
            elif node.op == "placeholder":
                logging.info(f"Visiting input of graph: {node}")
                placeholder_vistor.define_node(node, enn_graph, vals_to_ids)
            elif node.op in [
                "get_attr",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op}" " is not supported in ENN Delegate")

        # Compile Graph
        enn_graph.finish()
        ser_buf = enn_graph.serialize()
        return ser_buf

    @classmethod
    def preprocess_multimethod(
        cls,
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, list[PreprocessResult]]:
        named_data_store = NamedDataStore()
        enn_wrapper = PyEnnWrapper.EnnWrapper()
        enn_wrapper.Init()

        first_programs_len = len(next(iter(edge_programs.values())))
        assert all(
            len(programs) == first_programs_len for programs in edge_programs.values()
        ), "All subgraphs must have the same number of partitions!"

        preprocess_results = {}
        # Store GEN compile specs to set weight_buffer_hash_id_spec.value in USE phase
        gen_compile_specs = []
        for method_name, programs in edge_programs.items():
            assert (
                method_name in compile_specs
            ), f"Error: missing compile specs for {method_name}"
            compile_specs_for_method = compile_specs[method_name]
            assert len(compile_specs_for_method) == len(
                programs
            ), f"Error: method {method_name} has {len(programs)} partitions but only {len(compile_specs_for_method)}"
            preprocess_results_for_method = []
            for program, compile_spec_for_program in zip(
                programs, compile_specs_for_method
            ):
                option_spec = get_compile_spec(
                    compile_spec_for_program, ENN_COMPILE_OPTION_TITLE, required=True
                )
                enn_wrapper.SetOptions(option_spec.value)

                ser_buf = EnnBackend._build_enn_graph_from_program(program)
                enn_context_binary = enn_wrapper.Compile(ser_buf)
                assert enn_context_binary is not None and len(enn_context_binary) > 0

                weight_sharing_flag = get_weight_sharing_flag_from_compile_spec(
                    option_spec
                )
                shares_weight = (
                    weight_sharing_flag != WeightSharingFlag.WEIGHT_SHARING_NONE
                )
                weight_buffer_hash_id_spec = get_compile_spec(
                    compile_spec_for_program,
                    ENN_COMPILE_WEIGHT_BUFFER_HASH_ID,
                    required=shares_weight,
                )
                if weight_sharing_flag == WeightSharingFlag.WEIGHT_SHARING_GEN:
                    # GEN phase: don't get weights, just store compile spec for later
                    gen_compile_specs.append(weight_buffer_hash_id_spec)
                elif weight_sharing_flag == WeightSharingFlag.WEIGHT_SHARING_USE:
                    # USE phase: get weights, compute hash, set GEN and USE weight_buffer_hash_id_spec.value
                    assert (
                        len(gen_compile_specs) > 0
                    ), "No GEN compile specs available for WEIGHT_SHARING_USE!"
                    gen_weight_buffer_hash_id_spec = gen_compile_specs.pop(0)
                    use_weight_buffer_hash_id_spec = weight_buffer_hash_id_spec
                    enn_weight_binary_list = enn_wrapper.GetWeights()
                    assert (
                        len(enn_weight_binary_list) == 1
                    ), "Only support exactly one weight binary per subgraph for weight sharing!"
                    weight_hash = hashlib.sha256(
                        bytes(enn_weight_binary_list[0])
                    ).hexdigest()
                    # Set weight_buffer_hash_id_spec.value for both GEN and USE
                    gen_weight_buffer_hash_id_spec.value = weight_hash.encode("utf-8")
                    use_weight_buffer_hash_id_spec.value = weight_hash.encode("utf-8")
                    # Store weights in named_data_store
                    named_data_store.add_named_data(
                        weight_hash,
                        bytes(enn_weight_binary_list[0]),
                        external_tag="weights",
                    )
                else:  # WEIGHT_SHARING_NONE
                    pass
                preprocess_results_for_method.append(
                    PreprocessResult(
                        processed_bytes=bytes(enn_context_binary),
                        debug_handle_map={},
                    )
                )
            preprocess_results[method_name] = preprocess_results_for_method

        # Set the same data_store_output for all PreprocessResults
        for method_name in preprocess_results:
            for result in preprocess_results[method_name]:
                result.data_store_output = (
                    named_data_store.get_named_data_store_output()
                )

        enn_wrapper.Destroy()
        return preprocess_results
