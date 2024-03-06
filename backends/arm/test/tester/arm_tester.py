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
    Partition,
    Quantize,
    ToEdge,
)

from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram


class Partition(Partition):
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
        """
        assert (
            self.tosa_test_util is not None
        ), "self.tosa_test_util is not initialized, cannot use run_method()"
        inputs_to_run = inputs or self.inputs

        export_stage = self.stages[self.stage_name(Export)]

        is_quantized = self.stages["Quantize"] is not None
        (input_names, qp_input) = self._get_input_params(
            export_stage.artifact, is_quantized
        )
        (output_name, qp_output) = self._get_output_param(
            export_stage.artifact, is_quantized
        )

        self.qp_input = qp_input
        self.qp_output = qp_output

        # Calculate the reference output using the original module or the quant
        # module. self.quantization_scale is used by compare_outputs() to
        # calculate the tolerance
        self.quantization_scale = None
        if is_quantized:
            self.quantization_scale = qp_output.scale
            module_for_ref = self.stages[self.stage_name(Quantize)].artifact
        else:
            module_for_ref = self.original_module
        self.reference_output = self._calculate_reference_output(
            module_for_ref, inputs_to_run
        )

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

        self.stage_output = tosa_output

        return self

    def _get_input_params(
        self, program: ExportedProgram, is_quantized: bool
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

        if is_quantized:
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
        self, program: ExportedProgram, is_quantized: bool
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

        if is_quantized:
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

    def compare_outputs(self, atol=1e-03, rtol=1e-03, qtol=0):  # noqa (C901)
        try:
            super().compare_outputs(atol, rtol, qtol)
        except AssertionError as e:
            # Capture asertion error and print more info
            banner = "=" * 40 + "TOSA debug info" + "=" * 40
            print(banner)
            path_to_tosa_files = self.tosa_test_util.get_tosa_artifact_path()
            print(f"{self.qp_input=}")
            print(f"{self.qp_output=}")
            print(f"{path_to_tosa_files=}")
            import os

            torch.save(
                self.reference_output,
                os.path.join(path_to_tosa_files, "torch_ref_output.pt"),
            )
            print(f"{atol=}, {rtol=}, {qtol=}")
            analyze_diff(
                self.reference_output[0], self.stage_output[0], path_to_tosa_files
            )

            raise e


def analyze_diff(reference_output, backend_output, save_fig_path):
    """
    This function is used to visualize the difference between the reference
    output and the output. This is just a debug feature and should not be used
    in production code.
    """
    import os

    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ref_output = np.array(reference_output)
    backend_output = np.array(backend_output)
    # Make sure we have 4 dims
    while backend_output.ndim < 4:
        backend_output = np.reshape(ref_output, (1, *ref_output.shape))
        ref_output = np.reshape(backend_output, (1, *backend_output.shape))

    batches = ref_output.shape[0]
    channels = ref_output.shape[1]
    rows = int(channels // np.sqrt(channels))
    cols = rows
    remainder = int(channels % np.floor(np.sqrt(channels)))  # square layout
    fig = []
    axs = []
    for _ in range(batches):
        f, a = plt.subplots(rows + 1 if remainder > 0 else rows, cols, squeeze=False)
        fig.append(f)
        axs.append(a)
    for batch in range(batches):
        fig[batch].suptitle(f"Absolute diff per channel for batch {batch}")
        fig[batch].tight_layout()
        for row in range(rows):
            for col in range(cols):
                im = axs[batch][row][col].set_title(
                    f"Output channel {row * cols + col}"
                )
                err = np.abs(ref_output - backend_output)[batch][row * cols + col]
                im = axs[batch][row][col].imshow(err, interpolation="nearest")
                divider = make_axes_locatable(axs[batch][row][col])
                cax = divider.append_axes("right", size="10%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label("Abs error")
        for rem in range(remainder):
            im = axs[batch][rows][rem].set_title(f"Output channel {rows * cols + rem}")
            err = np.abs(ref_output - backend_output)[batch][rows * cols + rem]
            im = axs[batch][rows][rem].imshow(err, interpolation="nearest")
            divider = make_axes_locatable(axs[batch][rows][rem])
            cax = divider.append_axes("right", size="10%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Abs error")
        if remainder > 0:
            for col in range(cols - remainder):
                fig[batch].delaxes(axs[batch][rows][remainder + col])

    # Save diff plot to file
    for idx, fig in enumerate(plt.get_fignums()):
        plt.figure(fig)
        filename = os.path.join(save_fig_path, f"diff_{idx}.png")
        plt.savefig(filename)
