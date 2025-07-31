# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict

import torch

from executorch import version as executorch_version
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import MemoryPlanningPass

from optimum.exporters.executorch.integrations import (
    ImageTextToTextExportableModule,
)

from packaging.version import parse
from tabulate import tabulate
from torch.export import ExportedProgram


class RemovePaddingIdxEmbeddingPass(ExportPass):
    """
    An ExportPass that removes the `padding_idx` keyword argument
    from all aten.embedding.default operator calls.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.embedding.default
            ):
                # node.args[2] is the padding_idx
                if len(node.args) == 3:
                    node.args = (node.args[0], node.args[1])
        graph_module.recompile()
        return PassResult(graph_module, True)


def export_to_executorch_with_xnnpack(
    model: ImageTextToTextExportableModule,
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to XNNPACK backend.

    This function also write metadata required by the ExecuTorch runtime to the model.

    Args:
        model (Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule, ImageTextToTextExportableModule]):
            The PyTorch model to be exported to ExecuTorch.
        **kwargs:
            Additional keyword arguments for recipe-specific configurations, e.g. export using different example inputs, or different compile/bechend configs.

    Returns:
        Dict[str, ExecutorchProgram]:
            A map of exported and optimized program for ExecuTorch.
            For encoder-decoder models or multimodal models, it may generate multiple programs.
    """

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
    ) -> Dict[str, ExecutorchProgram]:
        backend_config_dict = {
            "extract_delegate_segments": True,
            "memory_planning_pass": MemoryPlanningPass(alloc_graph_input=False),
        }
        if parse(executorch_version.__version__).base_version > "0.6.0":
            backend_config_dict["do_quant_fusion_and_const_prop"] = True
        pte_name = model.model.config.model_type
        logging.debug(f"\nExported program for {pte_name}.pte: {exported_programs}")
        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        ).to_executorch(
            config=ExecutorchBackendConfig(**backend_config_dict),
        )
        delegation_info = get_delegation_info(
            et_prog.exported_program(list(exported_programs.keys())[0]).graph_module
        )
        logging.debug(
            f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}"
        )
        logging.debug(
            f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
        )
        return {pte_name: et_prog}

    exported_progs = model.export()

    if (
        model.config._attn_implementation == "custom_sdpa"
        or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
    ):
        # Sanity check to make sure the exported program contains the custom sdpa operator.
        if not any(
            node.op == "call_function" and "custom_sdpa" in str(node.target)
            for exported_program in exported_progs.values()
            for node in exported_program.graph_module.graph.nodes
        ):
            raise ValueError("'custom_sdpa' not found in the graph.")

    return _lower_to_executorch(exported_progs, model.metadata)
