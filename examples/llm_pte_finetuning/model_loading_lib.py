# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Tuple

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.llm_pte_finetuning.training_lib import TrainingModule
from executorch.exir import EdgeCompileConfig, to_edge

from omegaconf import DictConfig
from torch.export import export, ExportedProgram
from torch.export.experimental import _export_forward_backward
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtune import config
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.training.precision import get_dtype, set_default_dtype
from torchtune.utils._device import get_device


def load_checkpoint(cfg: Any) -> Dict[str, Any]:  # pyre-ignore[2]
    """
    Extract the checkpoint state from file and validate. This includes the
    base model weights. If resume_from_checkpoint is True, this also includes
    the adapter weights and recipe state
    """
    checkpointer = config.instantiate(
        cfg.checkpointer,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
    )
    checkpoint_dict = checkpointer.load_checkpoint()
    return checkpoint_dict


def setup_model(
    cfg: DictConfig,
    base_model_state_dict: Dict[str, Any],
) -> torch.nn.Module:
    device = get_device(device=cfg.device)
    dtype = get_dtype(cfg.dtype, device=device)
    with set_default_dtype(dtype), device:
        model = config.instantiate(cfg.model)

    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)
    model.load_state_dict(base_model_state_dict, strict=False)
    return model


def export_model_lora_training(
    model: TrainingModule,
    example_args: Tuple[Any, ...],  # pyre-ignore[2]
    output_file: str,
) -> None:
    """
    Export model with LoRA model to executorch for training, only.
    """

    # 0. Mark the LoRA layers as trainable (requires_grad = True) in order
    # to just export the backwards pass for these layers later in the
    # export process.
    set_trainable_params(model, get_adapter_params(model))

    print("Exporting model with LoRA for training")
    # 1. torch.export: Defines the program with the ATen operator set.

    with sdpa_kernel([SDPBackend.MATH]):
        exported_graph: ExportedProgram = export(model, example_args, strict=False)
        print("Creating a joint forward-backwards graph for training")
        joint_graph = _export_forward_backward(exported_graph)
        ep = joint_graph

        # Currently there is no implementation of empty_permuted for edge dialect.
        # We manually make a pass to rewrite the empty_permuted to empty and permute.
        for node in ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.empty_permuted.out
            ):
                print("find empty_permute: ", node)
                empty_permuted_node = node
                with ep.graph.inserting_before(empty_permuted_node):
                    empty_node = ep.graph.create_node(
                        "call_function",
                        torch.ops.aten.empty.memory_format,
                        (node.args[0],),
                        empty_permuted_node.kwargs,
                    )
                    permute_node = ep.graph.create_node(
                        "call_function",
                        torch.ops.aten.permute,
                        (empty_node, node.args[1]),
                    )
                    for user in empty_permuted_node.users.copy():
                        user.replace_input_with(empty_permuted_node, permute_node)
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.empty_permuted.default
            ):
                print("find empty_permute default: ", node)
                empty_permuted_node = node
                with ep.graph.inserting_before(empty_permuted_node):
                    empty_node = ep.graph.create_node(
                        "call_function",
                        torch.ops.aten.empty.memory_format,
                        (node.args[0],),
                        empty_permuted_node.kwargs,
                    )
                    permute_node = ep.graph.create_node(
                        "call_function",
                        torch.ops.aten.permute.default,
                        (empty_node, node.args[1]),
                    )
                    for user in empty_permuted_node.users.copy():
                        user.replace_input_with(empty_permuted_node, permute_node)

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(
            joint_graph,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten.empty_permuted.default]
            ),
        )

        print(edge_program._edge_programs["forward"].graph_module)

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    edge_program = edge_program.to_backend(
        XnnpackPartitioner(force_fp32_dynamic_linear=True)
    )
    executorch_program = edge_program.to_executorch()

    print(executorch_program.exported_program().graph_signature)
    print(f"Saving to {output_file}")
    with open(output_file, "wb") as file:
        file.write(executorch_program.buffer)
