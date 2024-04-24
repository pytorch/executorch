# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import numpy as np

import torch

from executorch.backends.arm.arm_backend import (
    get_intermediate_path,
    is_permute_memory,
    is_tosa,
)
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test.tosautil.tosa_test_utils import (
    QuantizationParams,
    TosaTestUtils,
)

from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import (
    Export,
    Partition as PartitionSuperclass,
    Quantize,
    ToEdge,
)

from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram
from torch.fx.node import Node


def _get_input_names(program: ExportedProgram) -> list[str]:
    """
    Get a list[str] with the names of the inputs to this model.

    Args:
        program (ExportedProgram): The program to get input names from.
    Returns:
        A list of strings with the names of the model input.
    """
    input_names = []

    # E.g. bias and weights are 'placeholders' as well. This is used to
    # get only the use inputs.
    usr_inputs = program.graph_signature.user_inputs
    for node in program.graph.nodes:
        if node.op == "placeholder" and node.name in usr_inputs:
            input_names.append(node.name)

    return input_names


def _get_input_quantization_params(
    program: ExportedProgram, input_names: list[str]
) -> list[QuantizationParams]:
    """
    Get input QuantizationParams in a program, maximum one per input to the program.
    Args:
        program (ExportedProgram): The program to get input quantization parameters from.
    Returns:
        list[QuantizationParams]: The found quantization parameters.
    Raises:
        RuntimeError if no quantization parameters are found.
    """

    quant_params = []
    num_inputs = len(input_names)
    for node in program.graph.nodes:
        if (
            node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
            and node.args[0].name in input_names
        ):
            qp = QuantizationParams(
                node_name=node.args[0].name, scale=node.args[1], zp=node.args[2]
            )
            quant_params.append(qp)
            if (
                len(quant_params) == num_inputs
            ):  # break early if we have all the inputs quantized parameters
                break
    if len(quant_params) == 0:
        raise RuntimeError("No Quantization parameters not found in exported model.")
    return quant_params


def _get_output_node(program: ExportedProgram) -> Node:
    """
    Get output node to this model.

    Args:
        program (ExportedProgram): The program to get output node from.
    Returns:
        The node that is the output of 'program'.
    """

    for node in program.graph.nodes:
        if node.op == "output":
            return node
    raise RuntimeError("No output node found.")


def _get_output_quantization_params(
    program: ExportedProgram, output_node: Node
) -> QuantizationParams:
    """
    Get output QuantizationParams from a program.
    Args:
        program (ExportedProgram): The program to get output quantization parameters from.
    Returns:
        QuantizationParams: The found quantization parameters.
    Raises:
        RuntimeError if no output quantization parameters are found.
    """

    quant_params = None
    for node in program.graph.nodes:
        if (
            node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            and node == output_node.args[0][0]
        ):
            quant_params = QuantizationParams(
                node_name=node.args[0].name, scale=node.args[1], zp=node.args[2]
            )
            break  # break early, there's only one output node
    if quant_params is None:
        raise RuntimeError("No Quantization parameters not found in exported model.")
    return quant_params


class Partition(PartitionSuperclass):
    def dump_artifact(self, path_to_dump: Optional[str]):
        super().dump_artifact(path_to_dump)
        from pprint import pformat

        to_print = None
        for spec in self.graph_module.lowered_module_0.compile_specs:
            if spec.key == "output_format":
                if spec.value == b"tosa":
                    tosa_fb = self.graph_module.lowered_module_0.processed_bytes
                    to_print = TosaTestUtils.dbg_tosa_fb_to_json(tosa_fb)
                    to_print = pformat(to_print, compact=True, indent=1)
                    to_print = f"\n TOSA deserialized: \n{to_print}"
                elif spec.value == b"vela":
                    vela_cmd_stream = self.graph_module.lowered_module_0.processed_bytes
                    to_print = str(vela_cmd_stream)
                    to_print = f"\n Vela command stream: \n{to_print}"
                break
        assert to_print is not None, "No TOSA nor Vela compile spec found"

        if path_to_dump:
            with open(path_to_dump, "a") as fp:
                fp.write(to_print)
        else:
            print(to_print)


class ArmTester(Tester):
    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        compile_spec: List[CompileSpec] = None,
    ):
        """
        Args:
            model (torch.nn.Module): The model to test
            inputs (Tuple[torch.Tensor]): The inputs to the model
            compile_spec (List[CompileSpec]): The compile spec to use
        """

        # Use the TosaTestUtils if you are using a TOSA backend
        self.tosa_test_util = None
        if is_tosa(compile_spec):
            intermediate_path = get_intermediate_path(compile_spec)
            self.tosa_test_util = TosaTestUtils(intermediate_path=intermediate_path)

        self.compile_spec = compile_spec

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

    def run_method_and_compare_outputs(
        self,
        stage: Optional[str] = None,
        inputs: Optional[Tuple[torch.Tensor]] = None,
        num_runs=1,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
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
        """
        assert (
            self.tosa_test_util is not None
        ), "self.tosa_test_util is not initialized, cannot use run_method()"

        number_of_runs = 1 if inputs is not None else num_runs
        stage = stage or self.cur

        export_stage = self.stages[self.stage_name(Export)]
        exported_program = export_stage.artifact
        input_names = _get_input_names(exported_program)
        output_node = _get_output_node(exported_program)
        output_name = output_node.name
        is_quantized = self.stages[self.stage_name(Quantize)] is not None

        if is_quantized:
            qp_input = _get_input_quantization_params(exported_program, input_names)
            qp_output = _get_output_quantization_params(exported_program, output_node)
        else:
            qp_input = [None] * len(input_names)
            qp_output = None

        # Calculate the reference output using the original module or the quant
        # module.
        quantization_scale = None
        if is_quantized:
            quantization_scale = qp_output.scale
            quantize_stage = self.stages[self.stage_name(Quantize)]
            module_for_ref = quantize_stage.artifact
            print(f"Comparing Stage {stage} with Stage {quantize_stage}")
        else:
            module_for_ref = self.original_module
            print(f"Comparing Stage {stage} with original module")

        # Loop inputs and compare TOSA ref model output with Torch reference
        # for each loop iteration.
        for run_iteration in range(number_of_runs):
            inputs_to_run = inputs if inputs else next(self.generate_random_inputs())
            input_shapes = [generated_input.shape for generated_input in inputs_to_run]
            print(f"Run {run_iteration} with input shapes: {input_shapes}")

            # Get Torch reference data...
            reference_output = self._calculate_reference_output(
                module_for_ref, inputs_to_run
            )

            # ...now get TOSA ref model data
            # Transpose input data which is on NCHW format to NHWC format,
            is_nhwc = is_permute_memory(self.compile_spec)
            if is_nhwc and len(inputs_to_run[0].shape) == 4:
                NHWC_Order = (0, 2, 3, 1)
                inputs_to_run = (np.transpose(inputs_to_run[0], NHWC_Order),)

            # Run the TOSA ref model to get the output tensor, which will be
            # compared to the torch output in compare_outputs()
            tosa_output = self.tosa_test_util.run_tosa_ref_model(
                params_input=(input_names, qp_input),
                param_output=(output_name, qp_output),
                inputs=inputs_to_run,
            )

            # Transpose back to NCHW format for comparison to torch output
            if is_nhwc and len(tosa_output.shape) == 4:
                NCHW_Order = (0, 3, 1, 2)
                tosa_output = (np.transpose(tosa_output, NCHW_Order),)

            stage_output = tosa_output

            # Output from running artifact at stage
            self._compare_outputs(
                reference_output, stage_output, quantization_scale, atol, rtol, qtol
            )

        return self

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
