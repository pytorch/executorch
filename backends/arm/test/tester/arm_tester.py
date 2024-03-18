# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.arm.arm_backend import (
    generate_ethosu_compile_spec,
    generate_tosa_compile_spec,
)

from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test.tosautil.tosa_test_utils import (
    QuantizationParams,
    TosaProfile,
    TosaTestUtils,
)

from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import (
    Export,
    Partition,
    Quantize,
    ToEdge,
)

from executorch.exir import EdgeCompileConfig
from torch.export import ExportedProgram


class ArmBackendSelector(Enum):
    TOSA = "tosa"
    ETHOS_U55 = "ethos-u55"


class ArmTester(Tester):
    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        backend: ArmBackendSelector = ArmBackendSelector.TOSA,
        profile: TosaProfile = TosaProfile.BI,
        permute_memory_to_nhwc: bool = False,
    ):
        """
        Args:
            model (torch.nn.Module): The model to test
            inputs (Tuple[torch.Tensor]): The inputs to the model
            backend (ArmBackendSelector): The backend to use. E.g. TOSA or
                ETHOS_U55.
                TOSA: Lower to TOSA and test numerical correctness compared to
                    torch reference.
                ETHOS_U55: Lower to TOSA, then let Vela compile. Only
                    functional test, no numerical checks.
            profile (TosaProfile): The TOSA profile to use. Either
                TosaProfile.BI or TosaProfile.MI
            permute_memory_to_nhwc (bool) : flag for enabling the memory format
                permutation to nhwc as required by TOSA
        """
        self.tosa_test_util = None
        self.is_quantized = profile == TosaProfile.BI
        self.permute_memory_to_nhwc = permute_memory_to_nhwc

        if backend == ArmBackendSelector.TOSA:
            self.tosa_test_util = TosaTestUtils(profile=profile)
            # The spec below tiggers arm_backend.py to output two files:
            #   1) output.tosa
            #   2) desc.json
            # Saved on disk in self.tosa_test_util.intermediate_path
            self.compile_spec = generate_tosa_compile_spec(
                self.permute_memory_to_nhwc, self.tosa_test_util.intermediate_path
            )
        elif backend == ArmBackendSelector.ETHOS_U55:
            self.compile_spec = generate_ethosu_compile_spec(
                "ethos-u55-128", self.permute_memory_to_nhwc
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        super().__init__(model, inputs)

    def quantize(self, quantize_stage: Optional[Quantize] = None):
        if quantize_stage is None:
            quantize_stage = Quantize(
                ArmQuantizer(),
                get_symmetric_quantization_config(is_per_channel=False),
            )
        return super().quantize(quantize_stage)

    def to_edge(self, to_edge_stage: Optional[ToEdge] = None):
        if to_edge_stage is None:
            to_edge_stage = ToEdge(EdgeCompileConfig(_check_ir_validity=False))
        return super().to_edge(to_edge_stage)

    def partition(self, partition_stage: Optional[Partition] = None):
        if partition_stage is None:
            arm_partitioner = ArmPartitioner(compile_spec=self.compile_spec)
            partition_stage = Partition(arm_partitioner)
        return super().partition(partition_stage)

    def run_method(
        self, stage: Optional[str] = None, inputs: Optional[Tuple[torch.Tensor]] = None
    ):
        """
        This function runs the tosa_reference_model tool to get output data
        needed for comparison with the torch reference data.

        Args:
            stage: (Optional[str]): Allows you input a custom stage. Currently
                not used.
            inputs (Optional[Tuple[torch.Tensor]]): Allows you to input custom
                input data.

        Todo:
            * A lot of the stuff in this method should be broken out into a
              run_artifact() method on a ToExecutorch stage class.
            * See "TODO" inline below
        """
        assert (
            self.tosa_test_util is not None
        ), "self.tosa_test_util is not initialized, cannot use run_method()"
        inputs_to_run = inputs or self.inputs

        export_stage = self.stages[self.stage_name(Export)]

        (input_names, qp_input) = self._get_input_params(export_stage.artifact)
        (output_name, qp_output) = self._get_output_param(export_stage.artifact)

        # Calculate the reference output using the original module or the quant
        # module. self.quantization_scale is used by compare_outputs() to
        # calculate the tolerance
        self.quantization_scale = None if qp_output is None else qp_output.scale
        if self.is_quantized:
            module_for_ref = self.stages[self.stage_name(Quantize)].artifact
        else:
            module_for_ref = self.original_module
        self.reference_output = self._calculate_reference_output(
            module_for_ref, inputs_to_run
        )

        # Run the TOSA ref model to get the output tensor, which will be
        # compared to the torch output in compare_outputs()
        self.stage_output = self.tosa_test_util.run_tosa_ref_model(
            params_input=(input_names, qp_input),
            param_output=(output_name, qp_output),
            inputs=inputs_to_run,
            permute_memory_to_nhwc=self.permute_memory_to_nhwc,
        )

        return self

    def _get_input_params(
        self, program: ExportedProgram
    ) -> Tuple[str, Union[List[QuantizationParams], List[None]]]:
        """
        Get name and optionally quantization parameters for the inputs to this
        model.

        Args:
            program (ExportedProgram): The program to get input parameters from
        Returns:
            Tuple[str, Optional[QuantizationParams]]: A tuple containing the
                input node names and their quantization parameters.
        """
        input_names = []
        # E.g. bias and weights are 'placeholders' as well. This is used to
        # get only the use inputs.
        usr_inputs = program.graph_signature.user_inputs
        for node in program.graph.nodes:
            if node.op == "placeholder" and node.name in usr_inputs:
                input_names.append(node.name)
                continue

        if self.is_quantized:
            quant_params = []
            for node in program.graph.nodes:
                if (
                    node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    and node.args[0].name in input_names
                ):
                    qp = QuantizationParams(
                        node_name=node.args[0].name, scale=node.args[1], zp=node.args[2]
                    )
                    quant_params.append(qp)
                    if len(quant_params) == len(
                        input_names
                    ):  # break early if we have all the inputs quantized parameters
                        break
            assert len(quant_params) != 0, "Quantization paramerters not found"
            return (input_names, quant_params)
        else:
            return (input_names, len(input_names) * [None])  # return a list of None's

    def _get_output_param(
        self, program: ExportedProgram
    ) -> Tuple[str, Union[QuantizationParams, None]]:
        """
        Get name and optionally quantization parameters for the inputs to this
        model.

        Args:
            program (ExportedProgram): The program to get output parameters from.
        Returns:
            Tuple[str, Optional[QuantizationParams]]: A tuple containing the
                output node name and its quantization parameters.
        """
        output_node = None
        for node in program.graph.nodes:
            if node.op == "output":
                output_node = node
                break

        if self.is_quantized:
            quant_params = None
            for node in program.graph.nodes:
                if (
                    node.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    and node == output_node.args[0][0]
                ):
                    quant_params = QuantizationParams(
                        node_name=node.args[0].name, scale=node.args[1], zp=node.args[2]
                    )
                    break  # break early, there's only one output node
            assert quant_params is not None, "Quantization paramerters not found"
            return (output_node.name, quant_params)
        else:
            return (output_node.name, None)

    @staticmethod
    def _calculate_reference_output(
        module: Union[torch.fx.GraphModule, torch.nn.Module], inputs
    ) -> torch.Tensor:
        """
        Note: I'd prefer to use the base class method here, but since it use the
        exported program, I can't. The partitioner stage clears the state_dict
        of the exported program, which causes an issue when evaluating the
        module.
        """

        return module.forward(*inputs)
