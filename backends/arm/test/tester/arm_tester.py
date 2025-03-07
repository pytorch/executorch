# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import os
from collections import Counter
from pprint import pformat
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import executorch.backends.xnnpack.test.tester.tester as tester

import serializer.tosa_serializer as ts  # type: ignore[import-untyped]

import torch.fx
import torch.utils._pytree as pytree

from executorch.backends.arm.arm_backend import (
    get_intermediate_path,
    get_tosa_spec,
    is_ethosu,
    is_tosa,
)
from executorch.backends.arm.ethosu_partitioner import EthosUPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.runner_utils import (
    dbg_tosa_fb_to_json,
    get_elf_path,
    get_output_nodes,
    get_output_quantization_params,
    get_target_board,
    run_corstone,
    TosaReferenceModelDispatch,
)

from executorch.backends.arm.test.tester.analyze_output_utils import (
    dump_error_output,
    print_error_diffs,
)
from executorch.backends.arm.tosa_mapping import extract_tensor_meta
from executorch.backends.arm.tosa_partitioner import TOSAPartitioner

from executorch.backends.xnnpack.test.tester import Tester
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    ExportedProgram,
)
from executorch.exir.backend.backend_api import validation_disabled
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import _update_exported_program_graph_module

from tabulate import tabulate
from torch.export.graph_signature import ExportGraphSignature, InputSpec, OutputSpec
from torch.fx import Graph
from torch.utils._pytree import tree_flatten


logger = logging.getLogger(__name__)


def _dump_lowered_modules_artifact(
    path_to_dump: Optional[str],
    artifact: ExecutorchProgramManager,
    graph_module: torch.fx.GraphModule,
):
    output = "Formated Graph Signature:\n"
    output += _format_export_graph_signature(
        artifact.exported_program().graph_signature
    )

    def get_output_format(lowered_module) -> str | None:
        for spec in lowered_module.compile_specs:
            if spec.key == "output_format":
                return spec.value.decode()
        return None

    for node in graph_module.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lowered_module = getattr(graph_module, node.name)
            assert isinstance(
                lowered_module, LoweredBackendModule
            ), f"Attribute {node.name} must be of type LoweredBackendModule."

            output_format = get_output_format(lowered_module)
            if output_format == "tosa":
                tosa_fb = lowered_module.processed_bytes
                to_print = dbg_tosa_fb_to_json(tosa_fb)
                to_print = pformat(to_print, compact=True, indent=1)
                output += f"\nTOSA deserialized {node.name}: \n{to_print}\n"
            elif output_format == "vela":
                vela_cmd_stream = lowered_module.processed_bytes
                output += f"\nVela command stream {node.name}: \n{vela_cmd_stream}\n"
            else:
                logger.warning(
                    f"No TOSA nor Vela compile spec found in compile specs of {node.name}."
                )
                continue

    if not output:
        logger.warning("No output to print generated from artifact.")
        return

    _dump_str(output, path_to_dump)


class Partition(tester.Partition):
    def dump_artifact(self, path_to_dump: Optional[str]):
        super().dump_artifact(path_to_dump)
        _dump_lowered_modules_artifact(path_to_dump, self.artifact, self.graph_module)


class ToEdgeTransformAndLower(tester.ToEdgeTransformAndLower):
    def dump_artifact(self, path_to_dump: Optional[str]):
        super().dump_artifact(path_to_dump)
        _dump_lowered_modules_artifact(path_to_dump, self.artifact, self.graph_module)


class Serialize(tester.Serialize):
    def __init__(self, compile_spec: list[CompileSpec], timeout):
        super().__init__()
        self.timeout = timeout
        self.executorch_program_manager: ExecutorchProgramManager | None
        self.compile_spec = compile_spec

    def run(self, artifact: ExecutorchProgramManager, inputs=None) -> None:
        super().run(artifact, inputs)
        # Keep the entire ExecutorchProgramManager for execution.
        self.executorch_program_manager = artifact

    def run_artifact(self, inputs):
        if self.executorch_program_manager is None:
            raise RuntimeError(
                "Tried running artifact from Serialize stage without running the stage."
            )
        inputs_flattened, _ = tree_flatten(inputs)
        intermediate_path = get_intermediate_path(self.compile_spec)
        target_board = get_target_board(self.compile_spec)
        elf_path = get_elf_path(target_board)

        if not os.path.exists(elf_path):
            raise FileNotFoundError(
                f"Did not find build arm_executor_runner in path {elf_path}, run setup_testing.sh?"
            )

        return run_corstone(
            self.executorch_program_manager,
            inputs_flattened,
            intermediate_path,
            target_board,
            elf_path,
            self.timeout,
        )


class ToExecutorch(tester.ToExecutorch):
    def run_artifact(self, inputs):
        with TosaReferenceModelDispatch():
            return super().run_artifact(inputs)


class RunPasses(tester.RunPasses):

    def __init__(
        self,
        pass_list: Optional[List[Type[ExportPass]]] = None,
        pass_functions: Optional[List[Callable]] = None,
        passes_with_exported_program: Optional[List[Type[ExportPass]]] = None,
    ):
        """Passes are run in the order they are passed: first pass_list, second pass_functions,
        and lastly passes_with_exported_program."""
        self.pass_with_exported_program = passes_with_exported_program
        super().__init__(pass_list, pass_functions)

    def run(
        self, artifact: Union[EdgeProgramManager, ExportedProgram], inputs=None
    ) -> None:
        if self.pass_with_exported_program is not None:
            self.pass_functions = self.pass_functions or []  # type: ignore

            # pass_function list from superclass expects functions that take in
            # and return ExportedPrograms.
            # Create a wrapper to fit pass_with_exported_program into this.
            def wrap_ep_pass(ep_pass: Type[ExportPass]):
                def wrapped_ep_pass(ep: ExportedProgram) -> ExportedProgram:
                    pass_result = ep_pass(ep).call(ep.graph_module)
                    with validation_disabled():
                        return _update_exported_program_graph_module(
                            ep, pass_result.graph_module
                        )

                return wrapped_ep_pass

            self.pass_functions.extend(
                [wrap_ep_pass(ep_pass) for ep_pass in self.pass_with_exported_program]
            )
        super().run(artifact, inputs)


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
        example_inputs: Tuple,
        compile_spec: List[CompileSpec],
    ):
        """
        Args:
            model (torch.nn.Module): The model to test
            example_inputs (Tuple[torch.Tensor]): Example inputs to the model
            compile_spec (List[CompileSpec]): The compile spec to use
        """

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
            quantizer = None
            if is_tosa(self.compile_spec):
                tosa_spec = get_tosa_spec(self.compile_spec)
                quantizer = TOSAQuantizer(tosa_spec)
            elif is_ethosu(self.compile_spec):
                quantizer = EthosUQuantizer(self.compile_spec)
            quantize_stage = tester.Quantize(
                quantizer,
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

        return super().to_edge(to_edge_stage)

    def partition(self, partition_stage: Optional[Partition] = None):
        if partition_stage is None:
            if is_tosa(self.compile_spec):
                arm_partitioner = TOSAPartitioner(compile_spec=self.compile_spec)
            elif is_ethosu(self.compile_spec):
                arm_partitioner = EthosUPartitioner(compile_spec=self.compile_spec)
            else:
                raise ValueError("compile spec doesn't target any Arm Partitioner")
            partition_stage = Partition(arm_partitioner)
        return super().partition(partition_stage)

    def to_edge_transform_and_lower(
        self,
        to_edge_and_lower_stage: Optional[ToEdgeTransformAndLower] = None,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        if to_edge_and_lower_stage is None:
            if partitioners is None:
                arm_partitioner = None
                if is_tosa(self.compile_spec):
                    arm_partitioner = TOSAPartitioner(compile_spec=self.compile_spec)
                elif is_ethosu(self.compile_spec):
                    arm_partitioner = EthosUPartitioner(compile_spec=self.compile_spec)
                else:
                    raise ValueError("compile spec doesn't target any Arm Partitioner")
                partitioners = [arm_partitioner]
            to_edge_and_lower_stage = ToEdgeTransformAndLower(
                partitioners, edge_compile_config
            )
        else:
            if partitioners is not None:
                to_edge_and_lower_stage.partitioners = partitioners
            if edge_compile_config is not None:
                to_edge_and_lower_stage.edge_compile_conf = edge_compile_config
        return super().to_edge_transform_and_lower(to_edge_and_lower_stage)

    def to_executorch(self, to_executorch_stage: Optional[ToExecutorch] | None = None):
        if to_executorch_stage is None:
            to_executorch_stage = ToExecutorch()
        return super().to_executorch(to_executorch_stage)

    def serialize(
        self, serialize_stage: Optional[Serialize] = None, timeout: int = 480
    ):
        if serialize_stage is None:
            serialize_stage = Serialize(self.compile_spec, timeout)
        assert (
            get_intermediate_path(self.compile_spec) is not None
        ), "Can't dump serialized file when compile specs do not contain an artifact path."

        return super().serialize(serialize_stage)

    def is_quantized(self) -> bool:
        return self.stages[self.stage_name(tester.Quantize)] is not None

    def run_method_and_compare_outputs(
        self,
        inputs: Optional[Tuple[torch.Tensor]] = None,
        stage: Optional[str] = None,
        num_runs=1,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        error_callbacks=None,
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
        edge_stage = self.stages[self.stage_name(tester.ToEdge)]
        if edge_stage is None:
            edge_stage = self.stages[self.stage_name(tester.ToEdgeTransformAndLower)]
        assert (
            edge_stage is not None
        ), "To compare outputs, at least the ToEdge or ToEdgeTransformAndLower stage needs to be run."

        stage = stage or self.cur
        test_stage = self.stages[stage]
        is_quantized = self.is_quantized()

        if is_quantized:
            reference_stage = self.stages[self.stage_name(tester.Quantize)]
        else:
            reference_stage = self.stages[self.stage_name(InitialModel)]

        exported_program = self.stages[self.stage_name(tester.Export)].artifact
        output_nodes = get_output_nodes(exported_program)
        output_qparams = get_output_quantization_params(output_nodes)

        quantization_scales = []
        for node in output_qparams:
            quantization_scales.append(getattr(output_qparams[node], "scale", None))

        logger.info(
            f"Comparing Stage '{self.stage_name(test_stage)}' with Stage '{self.stage_name(reference_stage)}'"
        )

        # Loop inputs and compare reference stage with the compared stage.
        for run_iteration in range(num_runs):
            reference_input = inputs if inputs else next(self.generate_random_inputs())

            input_shapes = [
                generated_input.shape if hasattr(generated_input, "shape") else (1,)
                for generated_input in reference_input
            ]
            input_shape_str = ", ".join([str(list(i)) for i in input_shapes])
            logger.info(f"Run #{run_iteration}, input shapes: {input_shape_str}")

            reference_outputs, _ = pytree.tree_flatten(
                reference_stage.run_artifact(reference_input)
            )
            test_outputs, _ = pytree.tree_flatten(
                test_stage.run_artifact(reference_input)
            )

            for reference_output, test_output, quantization_scale in zip(
                reference_outputs, test_outputs, quantization_scales
            ):
                self._compare_outputs(
                    reference_output,
                    test_output,
                    quantization_scale,
                    atol,
                    rtol,
                    qtol,
                    error_callbacks,
                )

        return self

    def get_graph(self, stage: str | None = None) -> Graph:
        if stage is None:
            stage = self.cur
        artifact = self.get_artifact(stage)
        if (
            self.cur == self.stage_name(tester.ToEdge)
            or self.cur == self.stage_name(Partition)
            or self.cur == self.stage_name(ToEdgeTransformAndLower)
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
        self, path_to_dump: Optional[str] = None, print_table: bool = True
    ):
        """Dump the distribution of operators in the current stage.
        In the partition stage, additional information is included such as the number of
        delegates and the distribution of TOSA operators.
        Set parameter print_table to False to dump in a parseable format.


        Returns self for daisy-chaining.
        """
        line = "#" * 10
        to_print = f"{line} {self.cur.capitalize()} Operator Distribution {line}\n"

        if (
            self.cur
            in (
                self.stage_name(tester.Partition),
                self.stage_name(ToEdgeTransformAndLower),
            )
            and print_table
        ):
            graph_module = self.get_artifact().exported_program().graph_module
            if print_table:
                delegation_info = get_delegation_info(graph_module)
                op_dist = delegation_info.get_operator_delegation_dataframe()
            else:
                op_dist = dict(_get_operator_distribution(graph_module.graph))
            to_print += _format_dict(op_dist, print_table)
            to_print += "\n" + _get_tosa_operator_distribution(
                graph_module, print_table
            )
            to_print += "\n"
            to_print += delegation_info.get_summary()
        else:
            graph = self.get_graph(self.cur)
            op_dist = dict(_get_operator_distribution(graph))
            if print_table:
                op_dist = {
                    "Operator": list(op_dist),
                    "Count": [op_dist[key] for key in op_dist],
                }
            to_print += _format_dict(op_dist, print_table) + "\n"

        _dump_str(to_print, path_to_dump)

        return self

    def dump_dtype_distribution(
        self, path_to_dump: Optional[str] = None, print_table: bool = True
    ):
        """Dump a the distributions of dtypes of nodes and placeholders in the current stage.
        Set parameter print_table to False to dump in a parseable format.

        Returns self for daisy-chaining.
        """

        line = "#" * 10
        to_print = (
            f"{line} {self.cur.capitalize()} Placeholder Dtype Distribution {line}\n"
        )

        graph = self.get_graph(self.cur)
        dtype_dist_placeholders, dtype_dirst_tensors = _get_dtype_distribution(graph)
        all_dtypes = set(dtype_dist_placeholders.keys()) | set(
            dtype_dirst_tensors.keys()
        )
        if print_table:
            dtype_dist = {
                "Dtype": all_dtypes,
                "Placeholder Count": [
                    (
                        dtype_dist_placeholders[key]
                        if key in dtype_dist_placeholders
                        else 0
                    )
                    for key in all_dtypes
                ],
                "Tensor Count": [
                    (dtype_dirst_tensors[key] if key in dtype_dirst_tensors else 0)
                    for key in all_dtypes
                ],
            }
        else:
            dtype_dist = dict(dtype_dist_placeholders + dtype_dirst_tensors)
        to_print += _format_dict(dtype_dist, print_table) + "\n"
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

    def _compare_outputs(
        self,
        reference_output,
        stage_output,
        quantization_scale=None,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        error_callbacks=None,
    ):
        try:
            super()._compare_outputs(
                reference_output, stage_output, quantization_scale, atol, rtol, qtol
            )
        except AssertionError as e:
            if error_callbacks is None:
                error_callbacks = [print_error_diffs, dump_error_output]
            for callback in error_callbacks:
                callback(
                    self,
                    reference_output,
                    stage_output,
                    quantization_scale=None,
                    atol=1e-03,
                    rtol=1e-03,
                    qtol=0,
                )
            raise e


def _get_dtype_distribution(graph: Graph) -> tuple[dict, dict]:
    """Counts the occurences of placeholder and call_function dtypes in a graph.
    The result is a tuple of Counters (placeholder_distribution, call_function_distribution)
    """
    placeholder_dtypes = []
    call_function_dtypes = []
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholder_dtypes.append(str(node.meta["val"].dtype))
        if node.op == "call_function":
            if "val" in node.meta:
                dtype, _, _ = extract_tensor_meta(node.meta)
                call_function_dtypes.append(ts.DTypeNames[dtype])
    return Counter(placeholder_dtypes), Counter(call_function_dtypes)


def _get_operator_distribution(graph: Graph) -> dict[str, int]:
    """Counts the occurences of operator names in a graph.
    The result is a dict {'operator name':'number of nodes'}
    """
    return Counter(
        [str(node.target) for node in list(graph.nodes) if node.op == "call_function"]
    )


def _format_export_graph_signature(signature: ExportGraphSignature) -> str:
    def specs_dict(specs: list[InputSpec | OutputSpec], title: str):
        _dict: dict[str, list] = {title: [], "arg": [], "kind": [], "target": []}
        for i, spec in enumerate(specs):
            _dict[title].append(i)
            _dict["arg"].append(spec.arg)
            _dict["kind"].append(spec.kind)
            _dict["target"].append(spec.target if spec.target else "-")
        return _dict

    input_dict = specs_dict(signature.input_specs, "Inputs")
    output_dict = specs_dict(signature.output_specs, "Outputs")

    return f"{_format_dict(input_dict)}\n{_format_dict(output_dict)}"


def _get_tosa_operator_distribution(
    graph_module: torch.fx.GraphModule, print_table=False
) -> str:
    """Counts the occurences of operator names of all lowered modules containing
    a TOSA flatbuffer.
    The result is a string with the operator distribution or an error message.
    """
    op_list = []
    id = 0
    while lowered_module := getattr(graph_module, f"lowered_module_{id}", None):
        for spec in lowered_module.compile_specs:
            if spec.key != "output_format":
                continue
            if spec.value == b"tosa":
                tosa_fb = lowered_module.processed_bytes
                tosa_json = dbg_tosa_fb_to_json(tosa_fb)
                for region in tosa_json["regions"]:
                    for block in region["blocks"]:
                        op_list.extend(
                            [operator["op"] for operator in block["operators"]]
                        )
                break
            elif spec.value == b"vela":
                return "Can not get operator distribution for Vela command stream."
            else:
                return f"Unknown output format '{spec.value}'."
        id += 1
    if id == 0:
        return "No delegate with name 'lowered_module_0 found in graph module."
    op_dist = dict(Counter(op_list))
    op_dist = {
        "Operator": list(op_dist.keys()),
        "Count": [item[1] for item in op_dist.items()],
    }
    return "TOSA operators:\n" + _format_dict(dict(op_dist), print_table)


def _dump_str(to_print: str, path_to_dump: Optional[str] = None):
    if path_to_dump:
        with open(path_to_dump, "a") as fp:
            fp.write(to_print)
    else:
        logger.info(to_print)


def _format_dict(to_print: dict, print_table: bool = True) -> str:
    if isinstance(list(to_print.items())[0], Iterable) and print_table:
        return tabulate(
            to_print, headers="keys", tablefmt="fancy_grid", maxcolwidths=35
        )
    else:
        return pformat(to_print, compact=True, indent=1)
