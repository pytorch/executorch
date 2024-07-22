# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from collections import Counter
from pprint import pformat
from typing import Any, List, Literal, Optional, Tuple, Union

import executorch.backends.xnnpack.test.tester.tester as tester

import numpy as np

import torch

from executorch.backends.arm.arm_backend import get_intermediate_path, is_permute_memory
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test.runner_utils import (
    _get_input_names,
    _get_input_quantization_params,
    _get_output_node,
    _get_output_quantization_params,
    dbg_tosa_fb_to_json,
    RunnerUtil,
)

from executorch.backends.xnnpack.test.tester import Tester
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.fx import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Partition(tester.Partition):
    def dump_artifact(self, path_to_dump: Optional[str]):
        super().dump_artifact(path_to_dump)

        to_print = None
        for spec in self.graph_module.lowered_module_0.compile_specs:
            if spec.key == "output_format":
                if spec.value == b"tosa":
                    tosa_fb = self.graph_module.lowered_module_0.processed_bytes
                    to_print = dbg_tosa_fb_to_json(tosa_fb)
                    to_print = pformat(to_print, compact=True, indent=1)
                    to_print = f"\n TOSA deserialized: \n{to_print}"
                elif spec.value == b"vela":
                    vela_cmd_stream = self.graph_module.lowered_module_0.processed_bytes
                    to_print = str(vela_cmd_stream)
                    to_print = f"\n Vela command stream: \n{to_print}"
                break
        assert to_print is not None, "No TOSA nor Vela compile spec found"
        _dump_str(to_print, path_to_dump)


class Serialize(tester.Serialize):
    def __init__(self, runner_util: RunnerUtil, timeout: int = 1):
        super().__init__()
        self.runner = runner_util
        self.runner.set_timeout(timeout)

    def run_artifact(self, inputs):
        return self.runner.run_corstone300(inputs)

    def dump_artifact(self, path_to_dump: Optional[str]):
        if not path_to_dump:
            path_to_dump = self.path + "/program.pte"
        super().dump_artifact(path_to_dump)


class ToExecutorch(tester.ToExecutorch):
    def __init__(
        self,
        tosa_test_util: RunnerUtil,
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        super().__init__(dynamic_shapes)
        self.tosa_test_util = tosa_test_util

    def run_artifact(self, inputs):
        tosa_output = self.tosa_test_util.run_tosa_ref_model(
            inputs=inputs,
        )
        return tosa_output


class InitialModel(tester.Stage):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def run(self, artifact, inputs=None) -> None:
        pass

    @property
    def artifact(self) -> torch.nn.Module:
        return self.model

    @property
    def graph_module(self) -> None:
        return None

    def artifact_str(self) -> str:
        return str(self.model)

    def run_artifact(self, inputs):
        return self.model.forward(*inputs)


class ArmTester(Tester):
    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        compile_spec: List[CompileSpec] = None,
    ):
        """
        Args:
            model (torch.nn.Module): The model to test
            example_inputs (Tuple[torch.Tensor]): Example inputs to the model
            compile_spec (List[CompileSpec]): The compile spec to use
        """

        # Initiate runner_util
        intermediate_path = get_intermediate_path(compile_spec)
        self.runner_util = RunnerUtil(intermediate_path=intermediate_path)

        self.compile_spec = compile_spec
        super().__init__(model, example_inputs)
        self.pipeline[self.stage_name(InitialModel)] = [
            self.stage_name(tester.Quantize),
            self.stage_name(tester.Export),
        ]

        # Initial model needs to be set as a *possible* but not yet added Stage, therefore add None entry.
        self.stages[self.stage_name(InitialModel)] = None
        self._run_stage(InitialModel(self.original_module))

    def quantize(self, quantize_stage: Optional[tester.Quantize] = None):
        if quantize_stage is None:
            quantize_stage = tester.Quantize(
                ArmQuantizer(),
                get_symmetric_quantization_config(is_per_channel=False),
            )
        return super().quantize(quantize_stage)

    def to_edge(
        self,
        to_edge_stage: Optional[tester.ToEdge] = None,
        config: Optional[EdgeCompileConfig] = None,
    ):
        if to_edge_stage is None:
            to_edge_stage = tester.ToEdge(config)
        else:
            if config is not None:
                to_edge_stage.edge_compile_conf = config

        # TODO(T182928844): Delegate dim order op to backend.
        to_edge_stage.edge_compile_conf._skip_dim_order = True
        return super().to_edge(to_edge_stage)

    def partition(self, partition_stage: Optional[Partition] = None):
        if partition_stage is None:
            arm_partitioner = ArmPartitioner(compile_spec=self.compile_spec)
            partition_stage = Partition(arm_partitioner)
        return super().partition(partition_stage)

    def to_executorch(self, to_executorch_stage: Optional[ToExecutorch] | None = None):
        if to_executorch_stage is None:
            to_executorch_stage = ToExecutorch(self.runner_util)
        return super().to_executorch(to_executorch_stage)

    def serialize(
        self, serialize_stage: Optional[Serialize] = None, timeout: int = 120
    ):
        if serialize_stage is None:
            serialize_stage = Serialize(self.runner_util, timeout=timeout)
        assert (
            get_intermediate_path(self.compile_spec) is not None
        ), "Can't dump serialized file when compile specs do not contain an artifact path."

        return (
            super()
            .serialize(serialize_stage)
            .dump_artifact(get_intermediate_path(self.compile_spec) + "/program.pte")
        )

    def run_method_and_compare_outputs(
        self,
        inputs: Optional[Tuple[torch.Tensor]] = None,
        stage: Optional[str] = None,
        num_runs=1,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
    ):
        """
        Compares the run_artifact output of 'stage' with the output of a reference stage.
        If the model is quantized, the reference stage is the Quantize stage output.
        Otherwise, the reference stage is the initial pytorch module.

        Asserts that the outputs are equal (within tolerances).
        Returns self to allow the function to be run in a test chain.

        Args:
            stage: (Optional[str]): The name of the stage to compare.
                The default is the latest run stage.
            inputs (Optional[Tuple[torch.Tensor]]): Allows you to input custom input data.
                The default is random data.
        """
        assert (
            self.runner_util is not None
        ), "self.tosa_test_util is not initialized, cannot use run_method()"
        assert (
            self.stages[self.stage_name(tester.Export)] is not None
        ), "To compare outputs, at least the Export stage needs to be run."

        stage = stage or self.cur
        test_stage = self.stages[stage]
        is_quantized = self.stages[self.stage_name(tester.Quantize)] is not None
        self.runner_util.init_run(
            self.stages[self.stage_name(tester.Export)].artifact, is_quantized
        )

        if is_quantized:
            reference_stage = self.stages[self.stage_name(tester.Quantize)]
            quantization_scale = self.runner_util.qp_output.scale
        else:
            reference_stage = self.stages[self.stage_name(InitialModel)]
            quantization_scale = None

        print(f"Comparing Stage {test_stage} with Stage {reference_stage}")
        is_nhwc = is_permute_memory(self.compile_spec)

        # Loop inputs and compare reference stage with the compared stage.
        for run_iteration in range(num_runs):
            reference_input = inputs if inputs else next(self.generate_random_inputs())
            if is_nhwc:
                test_input = self.transpose_data_format(reference_input, "NHWC")
            else:
                test_input = reference_input

            input_shapes = [
                generated_input.shape for generated_input in reference_input
            ]
            print(f"Run {run_iteration} with input shapes: {input_shapes}")

            reference_output = reference_stage.run_artifact(reference_input)
            test_output = (test_stage.run_artifact(test_input),)
            if is_nhwc:
                test_output = self.transpose_data_format(test_output, "NCHW")

            self._compare_outputs(
                reference_output, test_output, quantization_scale, atol, rtol, qtol
            )

        return self

    def get_graph(self, stage: str | None = None) -> Graph:
        if stage is None:
            stage = self.cur
        artifact = self.get_artifact(stage)
        if self.cur == self.stage_name(tester.ToEdge) or self.cur == self.stage_name(
            Partition
        ):
            graph = artifact.exported_program().graph
        elif self.cur == self.stage_name(tester.Export) or self.cur == self.stage_name(
            tester.Quantize
        ):
            graph = artifact.graph
        else:
            raise RuntimeError(
                "Can only get a graph from Quantize, ToEdge, Export, and Partition stages."
            )

        return graph

    def dump_operator_distribution(
        self, path_to_dump: Optional[str] = None
    ) -> ArmQuantizer:
        """Dump a dictionary with {operator: operator count} for the operators in the
        graph of the current stage.

        Returns self for daisy-chaining.
        """
        graph = self.get_graph(self.cur)
        op_dist = _get_operator_distribution(graph)
        to_print = self.cur + " operators: " + _format_dict(op_dist) + "\n"
        _dump_str(to_print, path_to_dump)
        return self

    def dump_dtype_distribution(
        self, path_to_dump: Optional[str] = None
    ) -> ArmQuantizer:
        """Dump a dictionary with {dtype: dtype count} for the dtypes of the nodes in the
        graph of the current stage.

        Returns self for daisy-chaining.
        """
        graph = self.get_graph(self.cur)
        op_dist = _get_dtype_distribution(graph)
        to_print = self.cur + " placeholder data types: " + _format_dict(op_dist) + "\n"
        _dump_str(to_print, path_to_dump)
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

    def transpose_data_format(
        self, data: Tuple[torch.Tensor], to: Literal["NHWC", "NCHW"]
    ):
        if to == "NCHW":
            dim_order = (0, 3, 1, 2)
        if to == "NHWC":
            dim_order = (0, 2, 3, 1)
        inputs_transposed = list(data)
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                inputs_transposed[i] = torch.Tensor(np.transpose(data[i], dim_order))
        return tuple(inputs_transposed)

    def _compare_outputs(
        self,
        reference_output,
        stage_output,
        quantization_scale=None,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
    ):
        try:
            super()._compare_outputs(
                reference_output, stage_output, quantization_scale, atol, rtol, qtol
            )
        except AssertionError as e:
            # Capture assertion error and print more info
            banner = "=" * 40 + "TOSA debug info" + "=" * 40
            logger.error(banner)
            path_to_tosa_files = self.runner_util.intermediate_path

            export_stage = self.stages.get(self.stage_name(tester.Export), None)
            if export_stage is not None:
                input_names = _get_input_names(export_stage.artifact)
                output_node = _get_output_node(export_stage.artifact)
                qp_input = _get_input_quantization_params(
                    export_stage.artifact, input_names
                )
                qp_output = _get_output_quantization_params(
                    export_stage.artifact, output_node
                )
                logger.error(f"{qp_input=}")
                logger.error(f"{qp_output=}")

            logger.error(f"{path_to_tosa_files=}")
            import os

            torch.save(
                stage_output,
                os.path.join(path_to_tosa_files, "torch_tosa_output.pt"),
            )
            torch.save(
                reference_output,
                os.path.join(path_to_tosa_files, "torch_ref_output.pt"),
            )
            logger.error(f"{atol=}, {rtol=}, {qtol=}")
            raise e


def _get_dtype_distribution(graph: Graph) -> dict:
    """Counts the occurences of placeholder data types in a graph.
    The result is a dict {'data type':'number of placeholders'}
    """
    return Counter(
        [
            node.meta["val"].dtype
            for node in list(graph.nodes)
            if node.op == "placeholder"
        ]
    )


def _get_operator_distribution(graph: Graph) -> dict[str, int]:
    """Counts the occurences of operator names in a graph.
    The result is a dict {'operator name':'number of nodes'}
    """
    return Counter(
        [str(node.target) for node in list(graph.nodes) if node.op == "call_function"]
    )


def _dump_str(to_print: str, path_to_dump: Optional[str] = None):
    if path_to_dump:
        with open(path_to_dump, "a") as fp:
            fp.write(to_print)
    else:
        print(to_print)


def _format_dict(to_print: dict) -> str:
    return pformat(to_print, compact=True, indent=1)
