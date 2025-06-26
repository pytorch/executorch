# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model


@dataclass
class NeutronNodeArtifacts:
    microcode: np.ndarray
    weights: np.ndarray
    kernels: np.ndarray


def extract_artifacts_from_neutron_node(
    tflite_flatbuffer_or_path: bytes | str,
) -> NeutronNodeArtifacts:
    """Extract the payload (microcode, weights, kernels) from the Neutron Node in the given TFLite model.
    The model can be provided as a binary flatbuffer, or a path to a `.tflite` model.
    """

    if isinstance(tflite_flatbuffer_or_path, str):
        with open(tflite_flatbuffer_or_path, "rb") as f:
            flatbuffer = f.read()
    else:
        flatbuffer = tflite_flatbuffer_or_path

    model = Model.GetRootAs(flatbuffer, 0)
    assert (
        model.SubgraphsLength() == 1
    ), f"The model has `{model.SubgraphsLength()}` SubGraphs instead of `1`."

    sub_graph = model.Subgraphs(0)

    if sub_graph.OperatorsLength() == 0:
        raise RuntimeError(
            "Model converted with neutron-converter has `0` operators instead of `1`."
        )
    elif sub_graph.OperatorsLength() > 1:
        builtin_operators_map: dict[int, str] = {
            y: x for x, y in BuiltinOperator.__dict__.items()
        }

        opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
        nodes = [sub_graph.Operators(i) for i in range(sub_graph.OperatorsLength())]
        ops_found = [
            builtin_operators_map[opcodes[node.OpcodeIndex()].BuiltinCode()]
            for node in nodes
        ]

        raise RuntimeError(
            f"Model converted with neutron-converter has `{sub_graph.OperatorsLength()}` operators "
            f'instead of `1`. Operators found: {", ".join(ops_found)}.'
        )

    neutron_node = None
    opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
    for i in range(sub_graph.OperatorsLength()):
        opcode = opcodes[sub_graph.Operators(i).OpcodeIndex()]
        if (
            opcode.BuiltinCode() == BuiltinOperator.CUSTOM
            and opcode.CustomCode() == b"NeutronGraph"
        ):
            # Found the NeutronNode.
            neutron_node = sub_graph.Operators(i)
            break

    if neutron_node is None:
        raise RuntimeError(
            "Model converted with neutron-converter does not contain a NeutronGraph node."
        )

    # The last 3 input tensors of the Neutron Node contain:
    #   1. Neutron Microcode
    #   2. Neutron Weights
    #   3. Neutron Kernels
    assert (
        neutron_node.InputsLength() >= 3
    ), f"The Neutron Node only has `{neutron_node.GetInputsLen()}` inputs. Expected at least `3`."
    microcode_idx, weights_idx, kernels_idx = neutron_node.InputsAsNumpy()[-3:]

    microcode_buffer_idx = sub_graph.Tensors(microcode_idx).Buffer()
    weights_buffer_idx = sub_graph.Tensors(weights_idx).Buffer()
    kernels_buffer_idx = sub_graph.Tensors(kernels_idx).Buffer()

    microcode = model.Buffers(microcode_buffer_idx).DataAsNumpy()
    weights = model.Buffers(weights_buffer_idx).DataAsNumpy()
    kernels = model.Buffers(kernels_buffer_idx).DataAsNumpy()

    assert (
        microcode.dtype == weights.dtype == kernels.dtype == np.dtype("uint8")
    ), "The Neutron Node uses unexpected data types."

    return NeutronNodeArtifacts(microcode, weights, kernels)
