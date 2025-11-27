# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from executorch.backends.test.harness.error_statistics import ErrorStatistics
from executorch.backends.test.harness.stages import (
    Export,
    Partition,
    Quantize,
    RunPasses,
    Serialize,
    Stage,
    StageType,
    ToEdge,
    ToEdgeTransformAndLower,
    ToExecutorch,
)
from executorch.exir.dim_order_utils import get_memory_format

from torch.export import ExportedProgram
from torch.testing import FileCheck


class Tester:
    """
    Base class for a backend tester. This class is not intended to be used directly. Instead,
    backends are expected to subclass it and provide implementations for backend-dependent
    stages.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        stage_classes: Dict[StageType, Callable] | None = None,
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        module.eval()

        self.stage_classes = stage_classes or Tester.default_stage_classes()
        self.original_module = module
        self.example_inputs = example_inputs
        self.dynamic_shapes = dynamic_shapes
        self.stages: Dict[StageType, Stage] = OrderedDict.fromkeys(list(StageType))
        self.pipeline = {
            StageType.QUANTIZE: [StageType.EXPORT],
            StageType.EXPORT: [
                StageType.RUN_PASSES,
                StageType.TO_EDGE,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            ],
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: [
                StageType.RUN_PASSES,
                StageType.TO_EXECUTORCH,
            ],
            StageType.TO_EDGE: [
                StageType.PARTITION,
                StageType.RUN_PASSES,
            ],
            StageType.RUN_PASSES: [
                StageType.PARTITION,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # TODO Make this Stage optional
            StageType.PARTITION: [StageType.TO_EXECUTORCH],
            StageType.TO_EXECUTORCH: [StageType.SERIALIZE],
            StageType.SERIALIZE: [],
        }

        # Current stage type
        self.cur: Optional[StageType] = None

        # Reference output from eager mode
        self.reference_output = None

        # Quantization scale from eager mode
        self.quantization_scale: Optional[float] = None

        # Artifact output from stage
        self.stage_output = None

    @staticmethod
    def default_stage_classes() -> Dict[StageType, Callable]:
        """
        Returns a map of StageType to default Stage implementation.
        """
        return {
            StageType.EXPORT: Export,
            StageType.QUANTIZE: Quantize,
            StageType.PARTITION: Partition,
            StageType.RUN_PASSES: RunPasses,
            StageType.SERIALIZE: Serialize,
            StageType.TO_EDGE: ToEdge,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
            StageType.TO_EXECUTORCH: ToExecutorch,
        }

    def _get_default_stage(self, stage_type: StageType, *args, **kwargs) -> Stage:
        stage_class = self.stage_classes.get(stage_type)
        if stage_class is None:
            raise RuntimeError(
                f"Attempted to instantiate a default implementation for stage {stage_type} but no default class was registered."
            )
        return stage_class(*args, **kwargs)

    def generate_random_inputs(self):
        # Get shapes of inputs
        input_shapes = []
        if self.dynamic_shapes is None:
            for tensor_arg in self.example_inputs:
                assert isinstance(tensor_arg, torch.Tensor)
                input_shapes.append(tensor_arg.shape)
        else:
            # Random shapes depending on dynamic shape constraint
            dim_name_to_size = {}
            for arg_idx in range(len(self.example_inputs)):
                assert isinstance(self.example_inputs[arg_idx], torch.Tensor)
                ex_shape = list(self.example_inputs[arg_idx].shape)
                dynamic_dim_spec = self.dynamic_shapes[arg_idx]
                for dim_idx, dim_spec in dynamic_dim_spec.items():
                    assert dim_idx < len(ex_shape)
                    if isinstance(dim_spec, torch.export.dynamic_shapes._DerivedDim):
                        # derived dims are of the form {0: 2 * torch.export.Dim() // 2}
                        # The root contains the min/max of the export dim and fn contains
                        # the function to compute the derived dim.
                        dim_spec = dim_spec.root
                        fn = dim_spec.fn
                    elif isinstance(dim_spec, torch.export.dynamic_shapes._Dim):
                        # Not derived dim so fn is just itself
                        def fn(x):
                            return x

                    else:
                        raise RuntimeError(
                            f"Expected Dynamic Dims to be of type _DerivedDim or _Dim but got {type(dim_spec)}"
                        )
                    dim_name = dim_spec.__name__
                    if dim_name not in dim_name_to_size:
                        upper_bound = min(
                            dim_spec.max, 1000
                        )  # unbounded int max is too large
                        lower_bound = (
                            dim_spec.min if dim_spec.min >= 2 else 1
                        )  # 0/1 specialization means dim_spec.min can never be 1
                        dim_name_to_size[dim_name] = fn(
                            random.randint(lower_bound, upper_bound)
                        )
                    ex_shape[dim_idx] = dim_name_to_size[dim_spec.__name__]
                input_shapes.append(torch.Size(ex_shape))
        # create random tensor inputs with the shapes given above:
        random_inputs = []
        for arg_idx in range(len(self.example_inputs)):
            memFormat = get_memory_format(
                list(self.example_inputs[arg_idx].dim_order())
            )
            random_inputs.append(
                torch.randn(input_shapes[arg_idx])
                .to(dtype=self.example_inputs[arg_idx].dtype)
                .to(memory_format=memFormat)
            )

        yield tuple(random_inputs)

    def _pre(self, stage):
        stage_type = stage.stage_type()
        assert stage_type in self.stages and not self.stages[stage_type]

        last_artifact = self.original_module
        if self.cur:
            assert self.cur in self.pipeline, f"Invalid state: {self.cur}"
            allowed_next_stages = self.pipeline[self.cur]
            assert (
                stage_type in allowed_next_stages
            ), f"Invalid next stage: {stage_type}"
            last_artifact = self.get_artifact()
        self.cur = stage_type
        return last_artifact

    def _post(self, stage):
        stage_type = stage.stage_type()
        assert stage_type in self.stages
        self.stages[stage_type] = stage

    def _run_stage(self, stage_instance, inputs=None, *args, **kwargs):
        assert isinstance(stage_instance, Stage)
        prev_stage_artifact = self._pre(stage_instance)
        stage_instance.run(prev_stage_artifact, inputs=inputs, *args, **kwargs)  # noqa
        self._post(stage_instance)
        return self

    # Stages
    def quantize(self, quantize_stage: Optional[Quantize] = None):
        return self._run_stage(
            quantize_stage or self._get_default_stage(StageType.QUANTIZE),
            self.example_inputs,
        )

    def export(self, export_stage: Optional[Export] = None):
        return self._run_stage(
            export_stage
            or self._get_default_stage(
                StageType.EXPORT, dynamic_shapes=self.dynamic_shapes
            ),
            self.example_inputs,
        )

    def to_edge(self, to_edge_stage: Optional[ToEdge] = None):
        if not to_edge_stage:
            to_edge_stage = self._get_default_stage(StageType.TO_EDGE)
        res = self._run_stage(to_edge_stage)
        return res

    def to_edge_transform_and_lower(
        self,
        to_edge_and_transform_stage: Optional[ToEdgeTransformAndLower] = None,
        generate_etrecord: bool = False,
    ):
        return self._run_stage(
            to_edge_and_transform_stage
            or self._get_default_stage(StageType.TO_EDGE_TRANSFORM_AND_LOWER),
            generate_etrecord=generate_etrecord,
        )

    def run_passes(self, run_passes_stage: Optional[RunPasses] = None):
        return self._run_stage(
            run_passes_stage or self._get_default_stage(StageType.RUN_PASSES)
        )

    def partition(self, partition_stage: Optional[Partition] = None):
        return self._run_stage(
            partition_stage or self._get_default_stage(StageType.PARTITION)
        )

    def to_executorch(self, to_executorch_stage: Optional[ToExecutorch] = None):
        return self._run_stage(
            to_executorch_stage or self._get_default_stage(StageType.TO_EXECUTORCH)
        )

    def serialize(self, serialize_stage: Optional[Serialize] = None):
        return self._run_stage(
            serialize_stage or self._get_default_stage(StageType.SERIALIZE)
        )

    # Util functions
    def dump_artifact(self, path: Optional[str] = None, stage: Optional[str] = None):
        stage = stage or self.cur
        self.stages[stage].dump_artifact(path)
        return self

    def get_artifact(self, stage: Optional[StageType] = None):
        stage = stage or self.cur
        return self.stages[stage].artifact

    def check(self, input: List[str]):
        for key in input:
            FileCheck().check(key).run(self.stages[self.cur].graph_module.code)
        return self

    def check_not(self, input: List[str]):
        for key in input:
            FileCheck().check_not(key).run(self.stages[self.cur].graph_module.code)
        return self

    def check_count(self, input: Dict[Any, int]):
        # TODO target checks similar to checkGraphModuleNodes()
        for key, count in input.items():
            FileCheck().check_count(key, count, exactly=True).run(
                self.stages[self.cur].graph_module.code
            )
        return self

    def check_node_count(self, input: Dict[Any, int]):
        # Count the occurances of each target in the graph.
        target_ops = [
            node.target
            for node in self.stages[self.cur].graph_module.graph.nodes
            if node.op == "call_function"
        ]
        op_counts = Counter(target_ops)

        for key, count in input.items():
            if count != op_counts[key]:
                print(f"Nodes: {op_counts}")
                raise AssertionError(
                    f"Expected {count} {key} nodes but found {op_counts[key]}."
                )

        return self

    def visualize(
        self, reuse_server: bool = True, stage: Optional[StageType] = None, **kwargs
    ):
        # import here to avoid importing model_explorer when it is not needed which is most of the time.
        from executorch.devtools.visualization import visualize

        visualize(self.get_artifact(stage), reuse_server=reuse_server, **kwargs)
        return self

    def run_method_and_compare_outputs(
        self,
        stage: Optional[StageType] = None,
        inputs: Optional[Tuple[torch.Tensor]] = None,
        num_runs=1,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        statistics_callback: Callable[[ErrorStatistics], None] | None = None,
    ):
        number_of_runs = 1 if inputs is not None else num_runs
        reference_stage = self.stages[StageType.EXPORT]

        stage = stage or self.cur

        for _ in range(number_of_runs):
            inputs_to_run = inputs if inputs else next(self.generate_random_inputs())

            # Reference output (and quantization scale)
            (
                reference_output,
                quantization_scale,
            ) = self._calculate_reference_output(
                reference_stage.artifact, inputs_to_run
            )

            # Output from running artifact at stage
            stage_output = self.stages[stage].run_artifact(inputs_to_run)
            self._compare_outputs(
                reference_output,
                stage_output,
                quantization_scale,
                atol,
                rtol,
                qtol,
                statistics_callback,
            )

        return self

    @staticmethod
    def _assert_outputs_equal(
        model_output,
        ref_output,
        atol=1e-03,
        rtol=1e-03,
        statistics_callback: Callable[[ErrorStatistics], None] | None = None,
    ):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the XNNPACK's backend, we relax the detal such that absolute tolerance is 1e-3. and
        relative tolerance is 1e-3. In the event that the computation was quantized, we
        further relax the tolerance to one quantized step (equal to the quantization scale).
        This allows the quantized value to differ by 1 between the reference and model output.
        """

        assert len(model_output) == len(ref_output)

        for i in range(len(model_output)):
            model = model_output[i]
            ref = ref_output[i]

            error_stats = ErrorStatistics.from_tensors(model, ref)
            if statistics_callback is not None:
                statistics_callback(error_stats)

            assert (
                ref.shape == model.shape
            ), f"Output {i} shape {model.shape} does not match reference output shape {ref.shape}"
            if model.dtype == torch.bool:
                assert torch.equal(model, ref), (
                    f"Output {i} (bool tensor) does not match reference output.\n"
                    f"\tShape: {model.shape}\n"
                    f"\tMismatched count: {(model != ref).sum().item()} / {model.numel()}\n"
                )
            else:
                assert torch.allclose(
                    model,
                    ref,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                ), (
                    f"Output {i} does not match reference output.\n"
                    f"\tGiven atol: {atol}, rtol: {rtol}.\n"
                    f"\tOutput tensor shape: {model.shape}, dtype: {model.dtype}\n"
                    f"\tDifference: max: {torch.max(model-ref)}, abs: {torch.max(torch.abs(model-ref))}, mean abs error: {torch.mean(torch.abs(model-ref).to(torch.double))}.\n"
                    f"\t-- Model vs. Reference --\n"
                    f"\t Numel: {model.numel()}, {ref.numel()}\n"
                    f"\tMedian: {model.median()}, {ref.median()}\n"
                    f"\t  Mean: {model.to(torch.double).mean()}, {ref.to(torch.double).mean()}\n"
                    f"\t   Max: {model.max()}, {ref.max()}\n"
                    f"\t   Min: {model.min()}, {ref.min()}\n"
                )

    @staticmethod
    def _compare_outputs(
        reference_output,
        stage_output,
        quantization_scale=None,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        statistics_callback: Callable[[ErrorStatistics], None] | None = None,
    ):
        """
        Compares the original of the original nn module with the output of the generated artifact.
        This requres calling run_method before calling compare_outputs. As that runs the generated
        artifact on the sample inputs and sets the stage output to be compared against the reference.
        """
        # Wrap both outputs as tuple, since executor output is always a tuple even if single tensor
        if isinstance(reference_output, torch.Tensor):
            reference_output = (reference_output,)
        elif isinstance(reference_output, OrderedDict):
            reference_output = tuple(reference_output.values())
        if isinstance(stage_output, torch.Tensor):
            stage_output = (stage_output,)

        # If a qtol is provided and we found an dequantization node prior to the output, relax the
        # atol by qtol quant units.
        if quantization_scale is not None:
            atol += quantization_scale * qtol

        Tester._assert_outputs_equal(
            stage_output,
            reference_output,
            atol=atol,
            rtol=rtol,
            statistics_callback=statistics_callback,
        )

    @staticmethod
    def _calculate_reference_output(
        program: ExportedProgram, inputs
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Execute the reference program and return the output. If the output comes from a dequantize node,
        return the quantization scale as well.
        """

        # Locate the output node.
        output_node = program.graph.output_node()

        # Look for a dequantization node in the output node args. Returned values are found in the first
        # argument of the output node.
        dequant_node = None
        for arg_node in output_node.args[0]:
            if (
                arg_node.op == "call_function"
                and arg_node.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                dequant_node = arg_node
                break

        scale = None
        if dequant_node is not None:
            original_target = dequant_node.target

            # Replace the dequant node with shim to intercept the quantization parameters.
            # It will be invoked when we evaluate the program to find the reference outputs.
            def dequant_shim(*args):
                nonlocal scale
                scale = args[1]
                result = original_target(*args)
                return result

            dequant_node.target = dequant_shim

        output = program.module()(*inputs)
        return output, scale
