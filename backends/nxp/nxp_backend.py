# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Main implementation of AoT flow to partition and preprocess for Neutron target
# backends.
#

import logging
import struct
from typing import final, List, Optional

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import Target
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.neutron_converter_manager import (
    NeutronConverterManager,
)
from executorch.backends.nxp.neutron_node_extraction import (
    extract_artifacts_from_neutron_node,
    NeutronNodeArtifacts,
)
from executorch.backends.nxp.neutron_pass_manager import NeutronPassManager
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch.export.exported_program import ExportedProgram


class NeutronCompileSpecBuilder:

    def __init__(self):
        self.config: Target = None
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.operators_not_to_delegate: List[str] = []
        self.neutron_converter_flavor = None

    def _replace_colons(self, operator: str) -> str:
        """
        Replace '::' with '_'
        """
        return operator.replace("::", "_")

    def neutron_compile_spec(
        self,
        config: str,
        neutron_converter_flavor: str,
        extra_flags: Optional[str] = None,
        operators_not_to_delegate: Optional[List[str]] = None,
    ):
        """
        Generate compile spec for Neutron NPU

        Args:
            config: Neutron accelerator configuration, e.g. "imxrt700"
            neutron_converter_flavor: Flavor of the neutron-converter module to use. Neutron-converter module named "
             "'neutron_converter_SDK_25_09' has flavor 'SDK_25_09'.
            extra_flags: Extra flags for the Neutron compiler
            operators_not_to_delegate: List of operators that should not be delegated
        """
        try:
            self.config = Target(config)
        except ValueError:
            raise ValueError(
                f"Config `{config}` is not a valid target. Must be one of `{Target.values()}`."
            )

        self.neutron_converter_flavor = neutron_converter_flavor

        assert (
            self.output_format is None
        ), f"Output format already set to f{self.output_format}"
        self.output_format = "tflite"
        self.compiler_flags = []

        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        if operators_not_to_delegate is not None:
            self.operators_not_to_delegate = [
                self._replace_colons(op) for op in operators_not_to_delegate
            ]

        return self

    def build(self):
        """
        Generate a list of compile spec objects from the builder
        """
        if self.output_format == "tflite":
            self.compile_spec += [
                CompileSpec("output_format", "tflite".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
                CompileSpec("target", self.config.value.encode()),
                CompileSpec(
                    "neutron_converter_flavor", self.neutron_converter_flavor.encode()
                ),
                CompileSpec(
                    "operators_not_to_delegate",
                    ",".join(self.operators_not_to_delegate).encode(),
                ),
            ]

        return self.compile_spec


def generate_neutron_compile_spec(
    config: str,  # The target platform. For example "imxrt700".
    neutron_converter_flavor: str,
    system_config: Optional[str] = None,
    extra_flags: Optional[str] = None,
    operators_not_to_delegate: Optional[List[str]] = None,
) -> List[CompileSpec]:
    return (
        NeutronCompileSpecBuilder()
        .neutron_compile_spec(
            config,
            neutron_converter_flavor,
            extra_flags=extra_flags,
            operators_not_to_delegate=operators_not_to_delegate,
        )
        .build()
    )


@final
class NeutronBackend(BackendDetails):

    @staticmethod
    def preprocess(  # noqa C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logging.info("NeutronBackend::preprocess")

        logging.debug(f"NeutronBackend preprocessing graph:\n{edge_program.graph}")

        output_format = ""
        compile_flags = []
        binary = bytes()
        target = ""
        neutron_converter_flavor = ""
        for spec in compile_spec:
            if spec.key == "output_format":
                output_format = spec.value.decode()
            if spec.key == "target":
                target = spec.value.decode()
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())
            if spec.key == "neutron_converter_flavor":
                neutron_converter_flavor = spec.value.decode()

        # Check that the output format is set in the compile spec
        if not output_format:
            raise RuntimeError("output format is required")

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                logging.debug(f"Operator to be processed: {node.target}")

        # Serialize and return the program.
        if output_format == "tflite":
            # We need to create custom model verifier with max_pool2d added as exception.
            # Otherwise, we get violation that this op is not part of ATen Core ops.
            edge_program._verifiers = [
                EXIREdgeDialectVerifier(
                    class_only=True,
                    core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
                )
            ]

            # Remove MaxPool-related "getitem" nodes from graph
            edge_program = NeutronPassManager(
                edge_program, [RemoveGetItemPass]
            ).transform()

            # Convert the edge program to TFLite.
            tflite_model, io_formats = EdgeProgramToIRConverter().convert_program(
                edge_program,
            )

            neutron_model = NeutronConverterManager().convert(
                tflite_model, target, neutron_converter_flavor
            )

            # Dump the tflite file if logging level is enabled
            if logging.root.isEnabledFor(logging.DEBUG):
                import os

                # Some of the nodes do not have delegation_tag, find any node with delegation tag.
                delegation_tag = None
                for n in list(edge_program.graph.nodes):
                    if "delegation_tag" in n.meta.keys():
                        delegation_tag = n.meta["delegation_tag"]
                        break
                assert delegation_tag is not None

                logging.debug(
                    f"Serializing converted graph with tag {delegation_tag} to {os.getcwd()}"
                )
                with open(f"{delegation_tag}_pure.et.tflite", "wb") as f:
                    f.write(bytes(tflite_model))
                with open(f"{delegation_tag}_neutron.et.tflite", "wb") as f:
                    f.write(bytes(neutron_model))

            binary = PayloadComposer().get_binary_payload(io_formats, neutron_model)
        else:
            raise RuntimeError(f"Unknown format {output_format}")

        return PreprocessResult(processed_bytes=binary)


class PayloadComposer:
    ALIGNMENT = 16

    def _padding_format_string_for_array(self, array: np.ndarray) -> str:
        """Create a padding format string for the given array, which will add 0s at the end for correct alignment.
        E.g. the string '10x' represents adding 10 bytes of '0' padding.
        """
        assert array.dtype == np.dtype("uint8")

        overflow = array.size % self.ALIGNMENT
        if overflow == 0:
            return ""

        # Overflow 1 means padding 15, so use `alignment - overflow` padding.
        return f"{self.ALIGNMENT - overflow}x"

    def _format_string_for_array(self, array: np.ndarray) -> str:
        """Create a format string which will represent the provided array. It also handles the necessary alignment.
        E.g. for array [1,2,3] we get '3s13x', because '3s' means string of 3 bytes, and `13x` means adding 13 bytes
         of '0' padding at the end (for 16B alignment).
        """
        assert array.dtype == np.dtype("uint8")

        return f"{array.size}s{self._padding_format_string_for_array(array)}"

    def _create_payload_header(self, io_formats, neutron_artifacts) -> np.ndarray:
        """
        Create bytes header for returned payload. It contains information about
        input and output tensor formats. Tensors are ordered based on graph signature
        of ExportedProgram. Header schema:

        +----------------------------+-----------------------------+------------------------+
        | Neutron inputs length (1B) | Neutron outputs length (1B) | Input args length (1B) |
        +----------------------------+-----------+-----------------+------------------------+
        | 1st input tensor format (1B)           | [nth* input tensor format (1B)]          |
        +----------------------------------------+------------------------------------------+
        | 1st output tensor format (1B)          | [nth* output tensor format (1B)]         |
        +----------------------------------------+------------------------------------------+
        | 1st input map (1B)                     | [nth* input map (1B)]                    |
        +----------------------------------------+------------------------------------------+
        | 1st output map (1B)                    | [nth* output map (1B)]                   |
        +----------------------------------------+------------------------------------------+

        :param io_formats: IO tensors formats.
        :return: Bytes representation of payload header.
        """
        inputs = io_formats["inputs"]
        outputs = io_formats["outputs"]

        assert (
            len(neutron_artifacts.input_indices) < 256
        ), "Models with more than 255 inputs are not supported."
        assert (
            len(neutron_artifacts.output_indices) < 256
        ), "Models with more than 255 outputs are not supported."

        header_data = [len(neutron_artifacts.input_indices)]
        header_data.append(len(neutron_artifacts.output_indices))
        header_data.append(len(inputs))

        for input_name in neutron_artifacts.input_names:
            try:
                header_data.append(
                    1
                    if inputs[input_name.decode()] == TensorFormat.CHANNELS_LAST
                    else 0
                )
            except KeyError:
                raise AssertionError(
                    f"Input tensor `{input_name.decode()}` not found in the converted model."
                )

        for output_name in neutron_artifacts.output_names:
            try:
                header_data.append(
                    1
                    if outputs[output_name.decode()] == TensorFormat.CHANNELS_LAST
                    else 0
                )
            except KeyError:
                raise AssertionError(
                    f"Output tensor `{output_name.decode()}` not found in the converted model."
                )

        header_data.extend(neutron_artifacts.input_indices)
        header_data.extend(neutron_artifacts.output_indices)

        # noinspection PyTypeChecker
        return np.array(header_data, dtype=np.uint8)

    def _pack_with_alignment(
        self, header: np.ndarray, neutron_artifacts: NeutronNodeArtifacts
    ) -> bytes:
        """
        Packs provided data into serialized binary data of the following C struct:
         struct NeutronBinary {
             uint8[] header;
             uint8[] microcode;
             uint8[] weights;
             uint8[] kernels;
         }
        The individual components must be aligned to 16 bytes.
        """

        return struct.pack(
            self._format_string_for_array(header)
            + self._format_string_for_array(neutron_artifacts.microcode)
            + self._format_string_for_array(neutron_artifacts.weights)
            + self._format_string_for_array(neutron_artifacts.kernels),
            header.tobytes(),
            neutron_artifacts.microcode.tobytes(),
            neutron_artifacts.weights.tobytes(),
            neutron_artifacts.kernels.tobytes(),
        )

    def get_binary_payload(self, io_formats, neutron_model) -> bytes:
        """
        Get binary payload for provided input/output tensor formats and neutron_model. Returned data have
        following structure:

        +----------------------------------------------------------------------------------------------------------------+
        |                                            16 bytes aligned blocks                                             |
        +================================================================================================================+
        |                                                     Header                                                     |
        +----------------------------------------------------------------------------------------------------------------+
        |                                                Neutron microcode                                               |
        +----------------------------------------------------------------------------------------------------------------+
        |                                                 Neutron weights                                                |
        +----------------------------------------------------------------------------------------------------------------+
        |                                                 Neutron kernels                                                |
        +----------------------------------------------------------------------------------------------------------------+

        Tensor format definition: '0x1' == CHANNELS_LAST, '0x0' == FORMATLESS (no format).

        :param io_formats: Dictionary with keys 'inputs' and 'outputs' that contains dictionaries
            mapping tensor name to TensorFormat.
        :param neutron_model: Neutron model with single NeutronGraph node.
        :return: 16 bytes aligned binary payload.
        """
        # Extract the Neutron microcode, weights and kernels from the Neutron Node in the `neutron_model`.
        neutron_artifacts = extract_artifacts_from_neutron_node(neutron_model)

        header = self._create_payload_header(io_formats, neutron_artifacts)

        return self._pack_with_alignment(header, neutron_artifacts)
