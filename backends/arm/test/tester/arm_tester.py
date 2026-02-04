# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import logging

from collections import Counter, defaultdict
from pprint import pformat
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    no_type_check,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import executorch.backends.xnnpack.test.tester.tester as tester

import torch.fx
import torch.utils._pytree as pytree

import tosa_serializer as ts

from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import get_symmetric_quantization_config
from executorch.backends.arm.test.runner_utils import (
    dbg_tosa_fb_to_json,
    get_output_quantization_params,
    TosaReferenceModelDispatch,
)

from executorch.backends.arm.test.tester.analyze_output_utils import (
    dump_error_output,
    print_error_diffs,
)
from executorch.backends.arm.test.tester.quantize import ArmQuantize as Quantize
from executorch.backends.arm.test.tester.serialize import Serialize

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.mapping import extract_tensor_meta

from executorch.backends.arm.util._factory import (
    create_partitioner,
    create_quantizer,
    parse_compile_spec,
)
from executorch.backends.arm.vgf import VgfCompileSpec

from executorch.backends.test.harness.error_statistics import ErrorStatistics
from executorch.backends.test.harness.stages import Stage, StageType
from executorch.backends.xnnpack.test.tester import (
    Partition as XnnpackPartitionStage,
    Quantize as XnnpackQuantize,
    Tester,
    ToEdge as XnnpackToEdge,
    ToEdgeTransformAndLower as XnnpackToEdgeTransformAndLower,
    ToExecutorch as XnnpackToExecutorch,
)
from executorch.devtools.backend_debug import get_delegation_info

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    ExportedProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_api import validation_disabled
from executorch.exir.backend.operator_support import OperatorSupportBase
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassType
from executorch.exir.program._program import (
    _copy_module,
    _update_exported_program_graph_module,
)
from tabulate import tabulate  # type: ignore[import-untyped]

from torch.export.graph_signature import ExportGraphSignature, InputSpec, OutputSpec
from torch.fx import Graph

from torchao.quantization.pt2e.quantizer import QuantizationSpec, SharedQuantizationSpec
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

logger = logging.getLogger(__name__)


def _dump_lowered_modules_artifact(
    path_to_dump: Optional[str],
    artifact: Union[EdgeProgramManager, ExecutorchProgramManager],
    graph_module: torch.fx.GraphModule | None,
) -> None:
    if graph_module is None:
        logger.warning("No graph module available to dump lowered modules.")
        return

    output = "Formated Graph Signature:\n"
    output += _format_export_graph_signature(
        artifact.exported_program().graph_signature
    )

    for node in graph_module.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lowered_module = getattr(graph_module, node.name)
            assert isinstance(
                lowered_module, LoweredBackendModule
            ), f"Attribute {node.name} must be of type LoweredBackendModule."

            compile_spec = parse_compile_spec(lowered_module.compile_specs)
            if isinstance(compile_spec, TosaCompileSpec):
                tosa_fb = lowered_module.processed_bytes
                to_print = dbg_tosa_fb_to_json(tosa_fb)
                to_print = pformat(to_print, compact=True, indent=1)
                output += f"\nTOSA deserialized {node.name}: \n{to_print}\n"
            elif isinstance(compile_spec, EthosUCompileSpec):
                vela_cmd_stream = lowered_module.processed_bytes
                output += f"\nVela command stream {node.name}: \n{vela_cmd_stream!r}\n"
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
        artifact = cast(Optional[EdgeProgramManager], self.artifact)
        graph_module = cast(Optional[torch.fx.GraphModule], self.graph_module)
        if artifact is None:
            logger.warning(
                "Partition stage artifact missing; skipping lowered module dump."
            )
            return
        _dump_lowered_modules_artifact(path_to_dump, artifact, graph_module)


class ToEdgeTransformAndLower(tester.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        constant_methods: Optional[Dict[str, Any]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
    ):
        super().__init__(partitioners, edge_compile_config)
        self.constant_methods = constant_methods
        self.transform_passes = transform_passes

    def dump_artifact(self, path_to_dump: Optional[str]):
        super().dump_artifact(path_to_dump)
        artifact = cast(Optional[EdgeProgramManager], self.artifact)
        graph_module = cast(Optional[torch.fx.GraphModule], self.graph_module)
        if artifact is None:
            logger.warning(
                "ToEdgeTransformAndLower stage artifact missing; skipping lowered module dump."
            )
            return
        _dump_lowered_modules_artifact(path_to_dump, artifact, graph_module)

    def run(
        self, artifact: ExportedProgram, inputs=None, generate_etrecord: bool = False
    ) -> None:
        artifact_to_run = copy.deepcopy(artifact)
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact_to_run,
            transform_passes=self.transform_passes,
            compile_config=self.edge_compile_conf,
            partitioner=self.partitioners,
            constant_methods=self.constant_methods,
            generate_etrecord=generate_etrecord,
        )


class ToExecutorch(tester.ToExecutorch):
    def run_artifact(self, inputs):
        with TosaReferenceModelDispatch():
            return super().run_artifact(inputs)


class RunPasses(tester.RunPasses):
    @no_type_check
    def __init__(
        self,
        pass_list: Optional[List[Type[PassType]]] = None,
        pass_functions: Optional[List[Callable]] = None,
        passes_with_exported_program: Optional[List[Type[ExportPass]]] = None,
    ):
        """Passes are run in the order they are passed: first pass_list, second pass_functions,
        and lastly passes_with_exported_program."""
        self.pass_with_exported_program: Optional[List[Type[ExportPass]]] = (
            passes_with_exported_program
        )

        super().__init__(pass_list, pass_functions)

    def run(
        self, artifact: Union[EdgeProgramManager, ExportedProgram], inputs=None
    ) -> None:
        if self.pass_with_exported_program is not None:
            pass_functions = list(self.pass_functions or [])  # type: ignore[has-type]

            # pass_function list from superclass expects functions that take in
            # and return ExportedPrograms.
            # Create a wrapper to fit pass_with_exported_program into this.
            def wrap_ep_pass(ep_pass: Type[ExportPass]):
                def wrapped_ep_pass(ep: ExportedProgram) -> ExportedProgram:
                    pass_instance = ep_pass(ep)  # type: ignore[call-arg]
                    pass_result = pass_instance.call(ep.graph_module)
                    with validation_disabled():
                        return _update_exported_program_graph_module(
                            ep, pass_result.graph_module
                        )

                return wrapped_ep_pass

            pass_functions.extend(
                [wrap_ep_pass(ep_pass) for ep_pass in self.pass_with_exported_program]
            )
            self.pass_functions = pass_functions
        super().run(artifact, inputs)


class InitialModel(Stage):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def stage_type(self) -> StageType:
        return StageType.INITIAL_MODEL

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
        example_inputs: Tuple[Any, ...],
        compile_spec: ArmCompileSpec,
        tosa_ref_model_path: str | None = None,
        dynamic_shapes: Optional[Tuple[Any]] = None,
        constant_methods: Optional[Dict[str, Any]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
        use_portable_ops: bool = False,
        timeout: int = 600,
    ):
        """
        Args:
            model (torch.nn.Module): The model to test
            example_inputs (Tuple[torch.Tensor]): Example inputs to the model
            compile_spec (ArmCompileSpec): The compile spec to use
        """

        self.transform_passes = transform_passes
        self.constant_methods = constant_methods
        self.compile_spec = compile_spec
        super().__init__(model, example_inputs, dynamic_shapes)
        self.pipeline[StageType.INITIAL_MODEL] = [
            StageType.QUANTIZE,
            StageType.EXPORT,
        ]
        self.original_module.requires_grad_(False)

        # Initial model needs to be set as a *possible* but not yet added Stage, therefore add None entry.
        self.stages[StageType.INITIAL_MODEL] = cast(Stage, None)
        self._run_stage(InitialModel(self.original_module))
        self.use_portable_ops = use_portable_ops
        self.timeout = timeout

    @no_type_check
    def quantize(
        self,
        quantize_stage: Optional[XnnpackQuantize] = None,
    ):
        # Same stage type as parent but exposed via module alias
        if quantize_stage is None:
            quantizer = create_quantizer(self.compile_spec)
            quantize_stage = Quantize(
                quantizer,
                get_symmetric_quantization_config(),
            )
        return super().quantize(quantize_stage)

    @no_type_check
    def to_edge(
        self,
        to_edge_stage: Optional[XnnpackToEdge] = None,
        # Keep config keyword-only to avoid positional clashes with legacy calls.
        *,
        config: Optional[EdgeCompileConfig] = None,
    ):
        # Allow optional config override beyond base signature
        if to_edge_stage is None:
            to_edge_stage = tester.ToEdge(config)
        else:
            if config is not None:
                to_edge_stage.edge_compile_conf = config

        return super().to_edge(to_edge_stage)

    @no_type_check
    def partition(self, partition_stage: Optional[XnnpackPartitionStage] = None):
        # Accept Arm-specific partition stage subclass
        if partition_stage is None:
            arm_partitioner = create_partitioner(self.compile_spec)
            partition_stage = Partition(arm_partitioner)
        return super().partition(partition_stage)

    @no_type_check
    def to_edge_transform_and_lower(
        self,
        to_edge_and_lower_stage: Optional[XnnpackToEdgeTransformAndLower] = None,
        generate_etrecord: bool = False,
        # Force the optional tuning knobs to be keyword-only for readability/back-compat.
        *,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
    ):
        # Arm flow exposes extra stage wiring knobs
        if transform_passes is not None:
            raise RuntimeError(
                "transform passes are given to ArmTester at construction."
            )

        if to_edge_and_lower_stage is None:
            if partitioners is None:
                operator_checks = (
                    list(additional_checks) if additional_checks is not None else None
                )
                arm_partitioner = create_partitioner(self.compile_spec, operator_checks)
                partitioners = [arm_partitioner]
            to_edge_and_lower_stage = ToEdgeTransformAndLower(
                partitioners,
                edge_compile_config,
                constant_methods=self.constant_methods,
                transform_passes=self.transform_passes,
            )
        else:
            if partitioners is not None:
                to_edge_and_lower_stage.partitioners = partitioners
            if edge_compile_config is not None:
                to_edge_and_lower_stage.edge_compile_conf = edge_compile_config
        return super().to_edge_transform_and_lower(
            to_edge_and_lower_stage, generate_etrecord=generate_etrecord
        )

    @no_type_check
    def to_executorch(self, to_executorch_stage: Optional[XnnpackToExecutorch] = None):
        # Allow custom ExecuTorch stage subclass
        if to_executorch_stage is None:
            to_executorch_stage = ToExecutorch()
        return super().to_executorch(to_executorch_stage)

    @no_type_check
    def serialize(
        self,
        serialize_stage: Optional[Serialize] = None,
        # Keep timeout keyword-only so positional usage matches the base class.
        *,
        timeout: int = 480,
    ):
        if serialize_stage is None:
            serialize_stage = Serialize(
                compile_spec=self.compile_spec,
                module=self.original_module,
                use_portable_ops=self.use_portable_ops,
                timeout=self.timeout,
            )
        return super().serialize(serialize_stage)

    def is_quantized(self) -> bool:
        return self.stages[StageType.QUANTIZE] is not None

    def _get_input_and_stages(
        self, inputs, stage, reference_stage_type, run_eager_mode: bool
    ):
        if inputs is None and isinstance(stage, tuple):
            if all(isinstance(arg, torch.Tensor) for arg in stage):
                inputs = cast(Tuple[torch.Tensor, ...], stage)
                stage = None

        if not run_eager_mode:
            edge_stage = self.stages[StageType.TO_EDGE]
            if edge_stage is None:
                edge_stage = self.stages[StageType.TO_EDGE_TRANSFORM_AND_LOWER]
            assert (
                edge_stage is not None
            ), "To compare outputs, at least the ToEdge or ToEdgeTransformAndLower stage needs to be run."
        else:
            # Run models in eager mode. We do this when we want to check that the passes
            # are numerically accurate and the exported graph is correct.
            export_stage = self.stages[StageType.EXPORT]
            assert (
                export_stage is not None
            ), "To compare outputs in eager mode, the model must be at Export stage"

        stage = stage or self.cur
        if stage is None:
            raise RuntimeError("No stage has been executed yet.")
        test_stage = self.stages[stage]
        is_quantized = self.is_quantized()

        if is_quantized:
            reference_stage_type = reference_stage_type or StageType.QUANTIZE
        else:
            reference_stage_type = reference_stage_type or StageType.INITIAL_MODEL
        reference_stage = self.stages[reference_stage_type]

        return inputs, reference_stage, test_stage

    def run_method_and_compare_outputs(
        self,
        stage: Optional[StageType] = None,
        inputs: Optional[Tuple[torch.Tensor, ...]] = None,
        num_runs: int = 1,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 0,
        statistics_callback: Callable[[ErrorStatistics], None] | None = None,
        # Preserve positional compatibility while keeping new flags keyword-only.
        *,
        reference_stage_type: StageType | None = None,
        compare_callback: Optional[Callable[..., None]] = None,
        error_callbacks: Optional[Sequence[Callable[..., None]]] = None,
        run_eager_mode: bool = False,
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

        # backward-compatible ordering (accept inputs as the first positional argument)
        inputs, reference_stage, test_stage = self._get_input_and_stages(
            inputs, stage, reference_stage_type, run_eager_mode
        )

        exported_stage = self.stages[StageType.EXPORT]
        exported_program = cast(ExportedProgram, exported_stage.artifact)
        output_node = exported_program.graph_module.graph.output_node()
        output_qparams = get_output_quantization_params(output_node)

        quantization_params = []
        for node in output_qparams:
            quantization_params.append(output_qparams[node])

        logger.info(
            f"Comparing Stage '{test_stage.stage_type()}' with Stage '{reference_stage.stage_type()}'"
        )

        # Loop inputs and compare reference stage with the compared stage.
        number_of_runs = 1 if inputs is not None else num_runs

        for run_iteration in range(number_of_runs):
            reference_input = inputs if inputs else next(self.generate_random_inputs())

            # Avoid issues with inplace operators
            test_input = copy.deepcopy(reference_input)
            original_input = copy.deepcopy(reference_input)

            input_shapes = [
                generated_input.shape if hasattr(generated_input, "shape") else (1,)
                for generated_input in reference_input
            ]
            input_shape_str = ", ".join([str(list(i)) for i in input_shapes])
            logger.info(f"Run #{run_iteration}, input shapes: {input_shape_str}")

            reference_outputs, _ = pytree.tree_flatten(
                reference_stage.run_artifact(reference_input)
            )
            if run_eager_mode:
                # Run exported module directly
                eager_output, _ = self._calculate_reference_output(
                    exported_program, test_input
                )
                test_outputs, _ = pytree.tree_flatten(eager_output)
            else:
                # Run lowered model with target
                test_outputs, _ = pytree.tree_flatten(
                    test_stage.run_artifact(test_input)
                )

            logger.info(f"\n      Input: {original_input}")
            logger.info(f"\n Ref output: {reference_outputs}")
            logger.info(f"\nTest output: {test_outputs}")

            for reference_output, test_output, quantization_param in zip(
                reference_outputs, test_outputs, quantization_params
            ):
                quantization_scale = getattr(quantization_param, "scale", None)
                self._compare_outputs(
                    reference_output,
                    test_output,
                    quantization_scale,
                    atol,
                    rtol,
                    qtol,
                    statistics_callback=statistics_callback,
                    compare_callback=compare_callback,
                    error_callbacks=error_callbacks,
                    quantization_parameters=quantization_param,
                )

        return self

    def _get_output_qspec_from_node(
        self, node: torch.fx.Node
    ) -> QuantizationSpec | None:
        if Q_ANNOTATION_KEY not in node.meta:
            return None
        annotation = node.meta[Q_ANNOTATION_KEY]
        # If annotation.output_qspec is a SharedQuantizationSpec, we need to find
        # the actual QuantizationSpec from one of the inputs.
        if isinstance(annotation.output_qspec, SharedQuantizationSpec):
            # First try to find a non-shared qspec from the inputs.
            annotation_qspec = [
                qspec
                for qspec in annotation.input_qspec_map.values()
                if not isinstance(qspec, SharedQuantizationSpec)
            ]
            # If none of the inputs have a non-shared qspec, we need to
            # find the source node of the shared qspec.
            if len(annotation_qspec) == 0:
                edge_or_node = annotation.output_qspec.edge_or_node
                if isinstance(edge_or_node, tuple):
                    source_node = edge_or_node[0]
                else:
                    source_node = edge_or_node
                annotation_qspec = [source_node.meta[Q_ANNOTATION_KEY].output_qspec]
            annotation_qspec = annotation_qspec[0]
        else:
            annotation_qspec = annotation.output_qspec

        return annotation_qspec

    def _get_input_qspecs_from_node(
        self, node: torch.fx.Node
    ) -> List[QuantizationSpec | None]:
        if Q_ANNOTATION_KEY not in node.meta:
            return [None]
        annotation = node.meta[Q_ANNOTATION_KEY]
        input_qspec_map = annotation.input_qspec_map
        found_qspecs = []
        if len(input_qspec_map) == 0:
            return [None]
        for spec in input_qspec_map.values():
            # If spec is a SharedQuantizationSpec, we need to find
            # the actual QuantizationSpec.
            if isinstance(spec, SharedQuantizationSpec):
                # First try to find a non-shared qspec from the inputs.
                annotation_qspec = [
                    qspec
                    for qspec in input_qspec_map.values()
                    if not isinstance(qspec, SharedQuantizationSpec)
                ]
                # If none of the inputs have a non-shared qspec, we need to
                # find the source node of the shared qspec.
                if len(annotation_qspec) == 0:
                    edge_or_node = annotation.output_qspec.edge_or_node
                    if isinstance(edge_or_node, tuple):
                        source_node = edge_or_node[0]
                    else:
                        source_node = edge_or_node
                    annotation_qspec = [source_node.meta[Q_ANNOTATION_KEY].output_qspec]
                found_qspecs.append(annotation_qspec[0])
            else:
                found_qspecs.append(spec)

        return found_qspecs

    def _check_input_qspecs(self, graph: Graph, input_qspecs):
        if input_qspecs is None:
            return
        found_qspecs = []
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            annotation_qspec = self._get_output_qspec_from_node(node)
            found_qspecs.append(annotation_qspec)
        found_qspecs_counter = Counter(found_qspecs)
        for qspec in input_qspecs:
            # check that each expected qspec is found
            if qspec not in found_qspecs_counter:
                raise AssertionError(
                    f"Expected to find input quantization annotation {qspec}, but it was not found. "
                    f"Found annotations: {found_qspecs_counter}"
                )
            # check that number of occurrences of each qspec matches expected
            if found_qspecs_counter[qspec] != input_qspecs[qspec]:
                raise AssertionError(
                    f"Expected to find {input_qspecs[qspec]} instances of input quantization annotation {qspec}, but "
                    f"found {found_qspecs_counter[qspec]} instances."
                )

    def _check_output_qspecs(self, graph: Graph, output_qspecs):
        if output_qspecs is None:
            return
        found_qspecs = []
        output_node = graph.output_node()
        annotation_qspec = self._get_input_qspecs_from_node(output_node)
        found_qspecs.extend(annotation_qspec)
        found_qspecs_counter = Counter(found_qspecs)
        for qspec in output_qspecs:
            # check that each expected qspec is found
            if qspec not in found_qspecs_counter:
                raise AssertionError(
                    f"Expected to find output quantization annotation {qspec}, but it was not found. "
                    f"Found annotations: {found_qspecs_counter}"
                )
            # check that number of occurrences of each qspec matches expected
            if found_qspecs_counter[qspec] != output_qspecs[qspec]:
                raise AssertionError(
                    f"Expected to find {output_qspecs[qspec]} instances of output quantization annotation {qspec}, but "
                    f"found {found_qspecs_counter[qspec]} instances."
                )

    def _check_qspecs(self, graph: Graph, quantization_annotations):
        if quantization_annotations is None:
            return self

        quantization_annotations_found: List[Tuple[str, QuantizationSpec | None]] = []
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            quantization_annotations_found.append(
                (str(node.target), self._get_output_qspec_from_node(node))
            )

        # Counter: (target, qspec) -> count
        quantization_annotations_found_counter = Counter(quantization_annotations_found)
        # Convert counter to Dict[target, Dict[qspec, count]]
        quantization_annotations_found_dict: Dict[
            str, Dict[QuantizationSpec | None, int]
        ] = defaultdict(dict)
        for (target, qspec), count in quantization_annotations_found_counter.items():
            quantization_annotations_found_dict[target][qspec] = count

        for target, qspecs in quantization_annotations.items():
            # check if target is in found annotations
            if target not in quantization_annotations_found_dict:
                raise AssertionError(
                    f"Expected to find quantization annotation for operator {target}, but it was not found."
                )
            for qspec in qspecs:
                # check if qspec is in found annotations for target
                if qspec not in quantization_annotations_found_dict[target]:
                    raise AssertionError(
                        f"Expected to find quantization annotation {qspec} for operator {target}, but it was not found. "
                        f"Found annotations: {quantization_annotations_found_dict[target]}"
                    )
                # check that number of occurrences of each qspec matches expected
                if quantization_annotations_found_dict[target][qspec] != qspecs[qspec]:
                    raise AssertionError(
                        f"Expected to find {qspecs[qspec]} instances of quantization annotation {qspec} for operator "
                        f"{target}, but found {quantization_annotations_found_dict[target][qspec]} instances."
                    )

    def check_quantization_annotation(
        self,
        quantization_annotations: Optional[
            Dict[str, Dict[QuantizationSpec | None, int]]
        ] = None,
        input_qspecs: Optional[Dict[QuantizationSpec | None, int]] = None,
        output_qspecs: Optional[Dict[QuantizationSpec | None, int]] = None,
    ):
        """
        Check the quantization annotations in the graph of a quantized model.

        Args:
            quantization_annotations: A dictionary mapping operator names to a dictionary of
                QuantizationSpecs and their expected counts.
                If None, the check is skipped.
            input_qspecs: A dictionary of expected input QuantizationSpecs and their counts.
                If None, the check is skipped.
            output_qspecs: A dictionary of expected output QuantizationSpecs and their counts.
                If None, the check is skipped.

        Returns self for daisy-chaining.
        """
        if not self.is_quantized():
            raise RuntimeError(
                f"{self.check_quantization_annotation.__name__} should be called after quantization stage."
            )

        graph = self.get_graph(StageType.QUANTIZE)

        self._check_input_qspecs(graph, input_qspecs)
        self._check_output_qspecs(graph, output_qspecs)
        self._check_qspecs(graph, quantization_annotations)
        return self

    def get_graph(self, stage: StageType | None = None) -> Graph:
        if stage is None:
            stage = self.cur
        if stage is None:
            raise RuntimeError("No stage has been executed yet.")
        artifact = self.get_artifact(stage)
        if (
            self.cur == StageType.TO_EDGE
            or self.cur == StageType.PARTITION
            or self.cur == StageType.TO_EDGE_TRANSFORM_AND_LOWER
        ):
            graph = artifact.exported_program().graph
        elif self.cur == StageType.EXPORT or self.cur == StageType.QUANTIZE:
            graph = artifact.graph
        else:
            raise RuntimeError(
                "Can only get a graph from Quantize, ToEdge, Export, and Partition stages."
            )

        return graph

    def dump_operator_distribution(
        self,
        path_to_dump: Optional[str] = None,
        print_table: bool = True,
        include_dtypes: bool = True,
    ):
        """Dump the distribution of operators in the current stage.
        In the partition stage, additional information is included such as the number of
        delegates and the distribution of TOSA operators.
        Set parameter print_table to False to dump in a parseable format.


        Returns self for daisy-chaining.
        """
        line = "#" * 10
        to_print = f"\n{line} {self.cur} Operator Distribution {line}\n"

        if self.cur in (
            StageType.PARTITION,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        ):
            graph_module = self.get_artifact().exported_program().graph_module
            delegation_info = get_delegation_info(graph_module)
            op_dist = _get_tosa_operator_distribution(graph_module, include_dtypes)
            if print_table:
                aten_op_dist = delegation_info.get_operator_delegation_dataframe()
                to_print += "Aten operators:\n" + _format_dict(
                    dict(aten_op_dist), print_table
                )

                if include_dtypes:
                    op_dist_dict = {
                        "Operator": [op_type[0] for op_type, _ in op_dist],
                        "Dtype": [op_type[1] for op_type, _ in op_dist],
                        "Count": [count for _, count in op_dist],
                    }
                else:
                    op_dist_dict = {
                        "Operator": [op for op, _ in op_dist],
                        "Count": [count for _, count in op_dist],
                    }
            else:
                if include_dtypes:
                    op_dtype_dist_dict: Dict[str, Dict[str, int]] = defaultdict(dict)
                    for op_dtype, count in op_dist:
                        op = op_dtype[0]
                        dtype = op_dtype[1]
                        op_dtype_dist_dict[op].update({dtype: count})
                    op_dist_dict = dict(op_dtype_dist_dict)
                else:
                    op_dist_dict = dict(op_dist)  # type: ignore[arg-type]
            to_print += "\nTOSA operators:\n" + _format_dict(op_dist_dict, print_table)
            to_print += "\n" + delegation_info.get_summary()
        else:
            graph = self.get_graph(self.cur)
            if include_dtypes:
                op_dist = _get_operator_dtype_distribution(graph)
            else:
                op_dist = _get_operator_distribution(graph)
            if print_table:
                if include_dtypes:
                    op_dist_dict = {
                        "Operator": [op_dtype[0] for op_dtype, _ in op_dist],
                        "Dtype": [op_dtype[1] for op_dtype, _ in op_dist],
                        "Count": [count for _, count in op_dist],
                    }
                else:
                    op_dist_dict = {
                        "Operator": [op for op, _ in op_dist],
                        "Count": [count for _, count in op_dist],
                    }
            else:
                if include_dtypes:
                    op_dtype_dist_dict = defaultdict(dict)
                    for op_dtype, count in op_dist:
                        op = op_dtype[0]
                        dtype = op_dtype[1]
                        op_dtype_dist_dict[op].update({dtype: count})
                    op_dist_dict = dict(op_dtype_dist_dict)
                else:
                    op_dist_dict = dict(op_dist)  # type: ignore[arg-type]

            to_print += _format_dict(op_dist_dict, print_table) + "\n"

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
        to_print = f"{line} {self.cur} Placeholder Dtype Distribution {line}\n"

        graph = self.get_graph(self.cur)
        tosa_spec = self.compile_spec.tosa_spec
        dtype_dist_placeholders, dtype_dirst_tensors = _get_dtype_distribution(
            graph, tosa_spec
        )
        all_dtypes = set(dtype_dist_placeholders.keys()) | set(
            dtype_dirst_tensors.keys()
        )
        dtype_dist: dict[str, Any]
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
            combined_counts = dtype_dist_placeholders + dtype_dirst_tensors
            dtype_dist = {key: combined_counts[key] for key in combined_counts}
        to_print += _format_dict(dtype_dist, print_table) + "\n"
        _dump_str(to_print, path_to_dump)
        return self

    def run_transform_for_annotation_pipeline(
        self, stage: StageType | None = None
    ) -> torch.fx.GraphModule:
        """Run transform_for_annotation_pipeline on exported program to ensure
        passes do not break the initial model before quantization.

        There are caveats to this however. As we register buffers to the graph modules
        the resulting exported graph can fail. Use this only to compare numerical correctness
        in eager mode.

        Returns exported program with passes applied.
        """

        if stage is None:
            stage = self.cur
        if stage is None:
            raise RuntimeError("No stage has been executed yet.")
        # We need to clone the artifact in order to ensure that the state_dict is preserved after passes are run.
        artifact = self.get_artifact(stage)
        if self.cur == StageType.EXPORT:
            new_gm = ArmPassManager(
                self.compile_spec
            ).transform_for_annotation_pipeline(graph_module=artifact.graph_module)
        else:
            raise RuntimeError("Can only run passes on Export stage.")
        _copy_module(artifact.graph_module, new_gm)
        return artifact

    @staticmethod
    def _calculate_reference_output(
        program: ExportedProgram, inputs: Tuple[Any, ...]
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Note: I'd prefer to use the base class method here, but since it use the
        exported program, I can't. The partitioner stage clears the state_dict
        of the exported program, which causes an issue when evaluating the
        module.
        """

        module = program.module()
        return module.forward(*inputs), None

    @no_type_check
    def _compare_outputs(
        self,
        reference_output,
        stage_output,
        quantization_scale=None,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        statistics_callback: Callable[[ErrorStatistics], None] | None = None,
        # Extra debugging hooks are keyword-only to keep the signature stable.
        *,
        compare_callback: Optional[Callable[..., None]] = None,
        error_callbacks: Optional[Sequence[Callable[..., None]]] = None,
        quantization_parameters=None,
    ):
        # Accept extra error callback hook for debugging
        try:
            if compare_callback:
                compare_callback(
                    reference_output, stage_output, quantization_parameters
                )
            else:
                super()._compare_outputs(
                    reference_output,
                    stage_output,
                    quantization_scale,
                    atol,
                    rtol,
                    qtol,
                    statistics_callback=statistics_callback,
                )
        except AssertionError as e:
            callbacks = (
                list(error_callbacks)
                if error_callbacks is not None
                else [print_error_diffs, dump_error_output]
            )
            for callback in callbacks:
                callback(
                    self,
                    stage_output,
                    reference_output,
                    quantization_scale=quantization_scale,
                    atol=1e-03,
                    rtol=1e-03,
                    qtol=0,
                )
            raise e

    def check_dtype_count(self, dtype_dict: Dict[str, Dict[str, int]]):
        if self.cur in (
            StageType.PARTITION,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        ):
            graph_module = self.get_artifact().exported_program().graph_module
            op_dist = _get_tosa_operator_distribution(graph_module, include_dtypes=True)
            op_dist_dict: Dict[str, Dict[str, int]] = defaultdict(dict)
            for op_dtype, count in op_dist:
                if isinstance(op_dtype, str):
                    raise ValueError(
                        f"Expected {_get_tosa_operator_distribution.__name__} to return "
                        "Tuple[Tuple[str, str], int]."
                    )
                else:
                    op, dtype = op_dtype

                op_dist_dict[op].update({dtype: count})
            for op in dtype_dict.keys():
                if op not in op_dist_dict:
                    raise RuntimeError(f"Could not find op {op}.")
                for dtype, count in dtype_dict[op].items():
                    dtype_count = op_dist_dict[op].setdefault(dtype, 0)
                    if dtype_count != count:
                        raise RuntimeError(
                            f"Expected {count} occurencies of {op=}, {dtype=} but found {dtype_count}."
                        )

        else:

            raise NotImplementedError(f"Cannot check dtypes for stage {self.cur}")


def _get_dtype_distribution(
    graph: Graph, tosa_spec: TosaSpecification
) -> tuple[Counter[str], Counter[str]]:
    """Counts the occurences of placeholder and call_function dtypes in a graph.
    The result is a tuple of Counters (placeholder_distribution, call_function_distribution)
    """
    placeholder_dtypes: list[str] = []
    call_function_dtypes: list[str] = []
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholder_dtypes.append(str(node.meta["val"].dtype))
        if node.op == "call_function":
            if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
                dtype, _, _ = extract_tensor_meta(node.meta)
                call_function_dtypes.append(ts.DTypeNames[dtype])
    return Counter(placeholder_dtypes), Counter(call_function_dtypes)


def _get_operator_distribution(graph: Graph) -> List[Tuple[str, int]]:
    """Counts the occurences of operator names in a graph.
    The result is a sorted list [('operator name':'number of nodes')]
    """
    return sorted(
        Counter(
            [
                str(node.target)
                for node in list(graph.nodes)
                if node.op == "call_function"
            ]
        ).items()
    )


def _get_operator_dtype_distribution(graph: Graph) -> List[Tuple[Tuple[str, str], int]]:
    """Counts the occurences of operator names and dtype pairs in a graph.
    The result is a sorted list[(('operator name','dtype'),'number of nodes')]
    """
    target_dtype_pairs = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            dtype = str(node.meta["val"].dtype)
        else:
            dtype = "UNKNOWN"
        target_dtype_pairs.append((str(node.target), dtype))
    return sorted(Counter(target_dtype_pairs).items())


def _format_export_graph_signature(signature: ExportGraphSignature) -> str:
    def specs_dict(specs: Sequence[InputSpec | OutputSpec], title: str):
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
    graph_module: torch.fx.GraphModule, include_dtypes=False
) -> list[Tuple[str, int]] | list[Tuple[Tuple[str, str], int]]:
    """Counts the occurences of operator names of all lowered modules containing
    a TOSA flatbuffer.
    The result is a string with the operator distribution or an error message.
    """
    id = 0
    unknown_dtype_str = "UNKNOWN"
    op_list = []
    while lowered_module := getattr(graph_module, f"lowered_module_{id}", None):
        compile_spec = parse_compile_spec(lowered_module.compile_specs)
        if isinstance(compile_spec, TosaCompileSpec):
            tosa_fb = lowered_module.processed_bytes
            tosa_json = dbg_tosa_fb_to_json(tosa_fb)
            for region in tosa_json["regions"]:
                for block in region["blocks"]:
                    for operator in block["operators"]:
                        op = operator["op"]
                        if include_dtypes:
                            outputs = operator.get("outputs", [])
                            if outputs == []:
                                op_list.append((op, unknown_dtype_str))
                                continue
                            tensor_block = block.get("tensors", {})
                            tensors_with_matching_name = [
                                t for t in tensor_block if t["name"] == outputs[0]
                            ]
                            dtype = (
                                tensors_with_matching_name[0]["type"]
                                if len(tensors_with_matching_name) > 0
                                else unknown_dtype_str
                            )
                            op_list.append((op, dtype))
                        else:
                            op_list.append(op)

        elif isinstance(compile_spec, EthosUCompileSpec):
            raise NotImplementedError(
                "Can not get operator distribution for Vela command stream."
            )
        elif isinstance(compile_spec, VgfCompileSpec):
            raise NotImplementedError("Can not get operator distribution for VGF.")
        else:
            raise NotImplementedError(
                f"Unknown output format '{compile_spec.get_output_format()}'."
            )
        id += 1
    if id == 0:
        raise ValueError(
            "No delegate with name 'lowered_module_0 found in graph module."
        )
    return sorted(Counter(op_list).items())


def _dump_str(to_print: str, path_to_dump: Optional[str] = None):
    if path_to_dump:
        with open(path_to_dump, "a") as fp:
            fp.write(to_print)
    else:
        print(to_print)


def _format_dict(to_print: dict, print_table: bool = True) -> str:
    if isinstance(list(to_print.items())[0], Iterable) and print_table:
        return tabulate(
            to_print, headers="keys", tablefmt="fancy_grid", maxcolwidths=35
        )
    else:
        return pformat(to_print, compact=True, indent=1)
