# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.import copy

import copy
from typing import Callable

import torch

from executorch.backends.cortex_m.passes import CortexMPassManager
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_helper import is_qdq_op
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.imxrt700cm.imxrt700cm_quantizer import IMXRT700CMQuantizer
from executorch.backends.nxp.neutron_partitioner import (
    NeutronPartitioner,
    NXP_DELEGATION_TAG,
)
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_random_calibration_inputs,
    ModelInputSpec,
    to_model_input_spec,
    to_quantized_edge_program,
)
from executorch.exir import EdgeProgramManager, to_edge
from executorch.exir.backend.partitioner import PartitionResult
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx import Node


def lower_to_imxrt700cm(
    model: torch.nn.Module,
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]],
    get_calibration_inputs_fn: Callable[
        [tuple[ModelInputSpec, ...]], list[tuple[torch.Tensor, ...]]
    ] = get_random_calibration_inputs,
    target: str = "imxrt700",
    remove_quant_io_ops: bool = False,
    custom_delegation_options: CustomDelegationOptions = CustomDelegationOptions(),  # noqa B008
    use_neutron_for_format_conversion: bool = True,
    use_quant_state_dict: bool = True,
    fetch_constants_to_sram: bool = False,
    dump_kernel_selection_code: bool = False,
) -> EdgeProgramManager:
    """Lower model to hybrid Neutron + Cortex-M backend.

    Pipeline:
    1. Identify nodes not supported by Neutron.
        1.1 Quantize a copy of the model with 1 dummy calibration tensor.
        1.2 Run NeutronPartitioner to mark supported nodes.
        1.3 Extract the names of the aten operator unsupported by neutron.
    2. Quantize using a hybrid quantizer, which applies NeutronQuantizer to some nodes, and CortexMQuantizer to others.
    3. Lower to edge using Neutron backend.
    4. Run Cortex-M passes on the edge program to replace leftover nodes with Cortex-M operators.

    NOTE: Some Cortex-M operators require the channels last dim order. So the provided `model` and `example_inputs`
          should use the channels last memory format for best results.

    TODO (Martin) The Cortex-M backend requires some aten nodes to be preserved in the edge program.
     This is not yet implemented. (EIEX-805)
    """
    input_spec = to_model_input_spec(input_spec)

    # Discover the names (stored in node.meta["torch_fn"][0]) of the aten operators which are not supported by Neutron.
    # The Cortex-M backend will be used for these if possible.
    cortex_m_designated_node_identifiers = _get_neutron_unsupported_node_identifiers(
        model,
        input_spec,
        target,
    )

    # Use the standard Neutron lowering pipeline
    edge_program_manager = to_quantized_edge_program(
        model,
        input_spec,
        get_calibration_inputs_fn=get_calibration_inputs_fn,
        target=target,
        remove_quant_io_ops=remove_quant_io_ops,
        custom_delegation_options=custom_delegation_options,
        get_quantizer_fn=lambda: IMXRT700CMQuantizer(
            NeutronTargetSpec(target), cortex_m_designated_node_identifiers
        ),
        use_neutron_for_format_conversion=use_neutron_for_format_conversion,
        use_quant_state_dict=use_quant_state_dict,
        fetch_constants_to_sram=fetch_constants_to_sram,
        dump_kernel_selection_code=dump_kernel_selection_code,
    )

    # Apply Cortex-M passes to replace the remaining nodes with Cortex-M variants where possible.
    pass_manager = CortexMPassManager(edge_program_manager.exported_program())
    edge_program_manager._edge_programs["forward"] = pass_manager.transform()

    return edge_program_manager


def get_non_delegated_nodes(partition_result: PartitionResult) -> list[Node]:
    """Return a list of nodes which were not marked by the NeutronPartitioner for delegation."""

    def _is_compute_op(node: Node) -> bool:
        # Return `True` for call functions which represent edge operators (not getitem or ExecutorchCallDelegate, ...)
        # Also exclude quantize/dequantize operations.
        return (
            node.op == "call_function"
            and isinstance(node.target, EdgeOpOverload)
            and not is_qdq_op(node)
        )

    return list(
        filter(
            lambda n: _is_compute_op(n) and NXP_DELEGATION_TAG not in n.meta,
            partition_result.tagged_exported_program.graph.nodes,
        )
    )


def _get_neutron_unsupported_node_identifiers(
    model: torch.nn.Module,
    input_spec: tuple[ModelInputSpec, ...],
    target: str,
    use_quant_state_dict: bool = False,
) -> set[str]:
    """Identify nodes not supported by Neutron.
        This is done by running quantization with dummy calibration data and then applying the NeutronPartitioner.

    :param model: Input model to analyze.
    :param input_spec: Tuple of objects containing information about the model inputs.
    :param target: Neutron target to use for quantization.
    :param use_quant_state_dict: If `True` the state dict from the quantized model will be used to assess operator
                                  support by the NeutronPartitioner.
    :return: Set of identifiers of nodes (stored in node.meta["torch_fn"][0]) which are not supported by Neutron.
    """
    example_inputs = tuple(
        torch.rand(spec.shape, dtype=spec.dtype) for spec in input_spec
    )
    exir_program_aten = torch.export.export(model, example_inputs, strict=True)

    # Make a deep copy for discovery
    copied_program = copy.deepcopy(exir_program_aten)

    # Use the example input to quantize the model. There is no need to have a representative calibration dataset.
    #  Instead, we want to use just 1 sample as we only care about speed and which nodes are supported.
    calibration_inputs = [example_inputs]

    # Quantize with NeutronQuantizer
    neutron_target_spec = NeutronTargetSpec(target)
    copied_program_quantized = calibrate_and_quantize(
        model=copied_program,
        calibration_inputs=calibration_inputs,
        quantizer=NeutronQuantizer(neutron_target_spec),
    )

    # Partition with Neutron
    compile_spec = generate_neutron_compile_spec(target)

    neutron_partitioner = NeutronPartitioner(
        compile_spec,
        neutron_target_spec,
        post_quantization_state_dict=(
            copied_program_quantized.state_dict() if use_quant_state_dict else None
        ),
    )

    epm = to_edge(
        torch.export.export(copied_program_quantized, example_inputs, strict=True)
    )
    epm = epm.transform(NeutronEdgePassManager())
    partition_result = neutron_partitioner.partition(epm.exported_program())

    # Identify edge compute operators which were not delegated to Neutron.
    non_delegated_edge_nodes = get_non_delegated_nodes(partition_result)

    # Extract the identifiers of these nodes, which can be used to identify the original aten nodes.
    non_delegated_node_identifiers = {
        n.meta["torch_fn"][0] for n in non_delegated_edge_nodes if "torch_fn" in n.meta
    }

    return non_delegated_node_identifiers
