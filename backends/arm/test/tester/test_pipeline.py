# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings as _warnings

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec

from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
    VgfQuantizer,
)
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester, RunPasses

from executorch.backends.arm.test.tester.quantize import ArmQuantize as Quantize
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)

from executorch.backends.arm.util._factory import create_quantizer
from executorch.exir.pass_base import ExportPass
from torch._export.pass_base import PassType
from torchao.quantization.pt2e.quantizer import QuantizationSpec

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=Tuple[Any, ...])
""" Generic type used for test data in the pipeline. Depends on which type the operator expects."""


def _require_tosa_version() -> str:
    version = conftest.get_option("tosa_version")
    if not isinstance(version, str):
        raise TypeError(f"TOSA version option must be a string, got {type(version)}.")
    return version


def _has_quantizable_inputs(test_data: T) -> bool:
    for data in test_data:
        if isinstance(data, torch.Tensor) and data.is_floating_point():
            return True
    return False


class PipelineStage:
    """Container for a pipeline stage (callable plus arguments)."""

    def __init__(self, func: Callable, id: str, *args, **kwargs):
        self.id: str = id
        self.func: Callable = func
        self.args = args
        self.kwargs = kwargs
        self.is_called = False

    def __call__(self):
        if not self.is_called:
            self.func(*self.args, **self.kwargs)
        else:
            raise RuntimeError(f"{self.id} called twice.")
        self.is_called = True

    def update(self, *args, **kwargs):
        if not self.is_called:
            self.args = args
            self.kwargs = kwargs
        else:
            raise RuntimeError(f"{self.id} args updated after being called.")


class BasePipelineMaker(Generic[T]):
    """
    The BasePiplineMaker defines a list of stages to be applied to a torch.nn.module for lowering it
    in the Arm backend. To be inherited and adjusted for particular targets. Importantly, the
    pipeline list can be modified before running the pipeline to support various pipeline extensions
    and debugging usecases.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data used for quantizing and testing the module.
        aten_ops: Aten dialect ops expected to be found in the graph after export.
        compile_spec: The compile spec used in the lowering process.
        exir_ops: Exir dialect ops expected to be found in the graph after to_edge if not using
                  use_edge_to_transform_and_lower.
        use_edge_to_transform_and_lower: Selects betweeen two possible routes for lowering:
                tester.to_edge_transform_and_lower()
            or
                tester.to_edge().check(exir_ops).partition()
    """

    @staticmethod
    def _normalize_ops(ops: str | Sequence[str] | None) -> list[str]:
        if ops is None:
            return []
        if isinstance(ops, str):
            return [ops]
        return list(ops)

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | Sequence[str] | None,
        compile_spec: ArmCompileSpec,
        exir_ops: str | Sequence[str] | None = None,
        use_to_edge_transform_and_lower: bool = True,
        dynamic_shapes: Optional[Tuple[Any]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
    ):

        self.tester = ArmTester(
            module,
            example_inputs=test_data,
            compile_spec=compile_spec,
            dynamic_shapes=dynamic_shapes,
            transform_passes=transform_passes,
        )

        self.aten_ops = self._normalize_ops(aten_ops)
        self.exir_ops = self._normalize_ops(exir_ops)
        self.test_data = test_data
        self._stages: list[PipelineStage] = []

        self.add_stage(self.tester.export)
        self.add_stage(self.tester.check, self.aten_ops, suffix="aten")
        if use_to_edge_transform_and_lower:
            self.add_stage(self.tester.to_edge_transform_and_lower)
        else:
            self.add_stage(self.tester.to_edge)
            self.add_stage(self.tester.check, self.exir_ops, suffix="exir")
            self.add_stage(self.tester.partition)
        self.add_stage(self.tester.check_not, self.exir_ops, suffix="exir")
        self.add_stage(
            self.tester.check_count,
            {"torch.ops.higher_order.executorch_call_delegate": 1},
            suffix="exir",
        )
        self.add_stage(self.tester.to_executorch)

    def add_stage(self, func: Callable, *args, **kwargs):
        """
        Adds a stage defined by a function with args and kwargs. By default appends to the pipeline.
        For stages which may be added multiple times to a pipeline, s.a. checks and debug stages,
        a suffix is appended with a dot to make sure every id is unique, e.g. check becomes check.0

        Special kwargs:
            pos : specifies position in pipeline to add stage at.
            suffix : specifies a custom suffix to identify non unique stages, instead of a number.
        """
        pipeline_length = len(self._stages)

        pos = -1
        if "pos" in kwargs:
            pos = kwargs.pop("pos")

        if pos < 0:
            pos = pipeline_length + (pos + 1)
        if not -pipeline_length <= pos <= pipeline_length:
            raise ValueError(
                f"Pos must be between [-{pipeline_length}, {pipeline_length}]"
            )

        stage_id = func.__name__
        suffix = None
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
            if stage_id == "dump_artifact":
                args = (*args, suffix)

        unique_stages = [
            "quantize",
            "export",
            "to_edge_transform_and_lower",
            "to_edge",
            "partition",
            "to_executorch",
            "serialize",
        ]
        id_list = [stage.id for stage in self._stages]
        if stage_id in unique_stages:
            if stage_id in id_list:
                raise RuntimeError(f"Tried adding {stage_id} to pipeline twice.")
        else:
            if suffix is None:
                stages_containing_stage_id = [
                    id for id in id_list if stage_id == id.split(".")[0]
                ]

                suffix = str(len(stages_containing_stage_id))

            if not suffix == "0":
                stage_id = stage_id + "." + suffix

                if stage_id in id_list:
                    raise ValueError("Suffix must be unique in pipeline")

        pipeline_stage = PipelineStage(func, stage_id, *args, **kwargs)
        self._stages.insert(pos, pipeline_stage)

        logger.debug(f"Added stage {stage_id} to {type(self).__name__}")

        return self

    @property
    def quantizer(self) -> TOSAQuantizer:
        quantize_pipeline_stage = self._stages[self.find_pos("quantize")]
        quantize_stage = quantize_pipeline_stage.args[0]
        if isinstance(quantize_stage, Quantize):
            quantizer = quantize_stage.quantizer
            if isinstance(quantizer, TOSAQuantizer):
                return quantizer
            else:
                raise RuntimeError(
                    f"Quantizer in pipeline was {type(quantizer).__name__}, not TOSAQuantizer as expected."
                )
        else:
            raise RuntimeError(
                f"First argument of quantize stage was {type(quantize_stage).__name__}, not Quantize as expected."
            )

    def pop_stage(self, identifier: int | str):
        """Removes and returns the stage at postion pos"""
        if isinstance(identifier, int):
            stage = self._stages.pop(identifier)
        elif isinstance(identifier, str):
            pos = self.find_pos(identifier)
            stage = self._stages.pop(pos)
        else:
            raise TypeError("identifier must be an int or str")

        logger.debug(f"Removed stage {stage.id} from {type(self).__name__}")

        return stage

    def find_pos(self, stage_id: str):
        """Returns the position of the stage id."""
        for i, stage in enumerate(self._stages):
            if stage.id == stage_id:
                return i

        raise Exception(f"Stage id {stage_id} not found in pipeline")

    def has_stage(self, stage_id: str):
        try:
            return self.find_pos(stage_id) >= 0
        except:
            return False

    def add_stage_after(self, stage_id: str, func: Callable, *args, **kwargs):
        """Adds a stage after the given stage id."""
        pos = self.find_pos(stage_id) + 1
        kwargs["pos"] = pos

        self.add_stage(func, *args, **kwargs)
        return self

    def dump_artifact(self, stage_id: str, suffix: str | None = None):
        """Adds a dump_artifact stage after the given stage id."""
        self.add_stage_after(stage_id, self.tester.dump_artifact, suffix=suffix)
        return self

    def dump_operator_distribution(self, stage_id: str, suffix: str | None = None):
        """Adds a dump_operator_distribution stage after the given stage id."""
        self.add_stage_after(
            stage_id, self.tester.dump_operator_distribution, suffix=suffix
        )
        return self

    def visualize(self, stage_id: str, suffix: str | None = None):
        """Adds a dump_operator_distribution stage after the given stage id."""
        self.add_stage_after(stage_id, self.tester.visualize, suffix=suffix)
        return self

    def change_args(self, stage_id: str, *args, **kwargs):
        """Updates the args to the given stage id."""
        pos = self.find_pos(stage_id)
        pipeline_stage = self._stages[pos]
        pipeline_stage.update(*args, **kwargs)
        return self

    def run(self):
        """Calls each stage in order."""
        stage_list = [stage.id for stage in self._stages]
        logger.info(f"Running pipeline with stages:\n {stage_list}.")

        for stage in self._stages:
            try:
                stage()
            except Exception as e:
                logger.error(f"\nFailure in stage <{stage.id}>: \n   {str(e)}")
                raise e


class TOSAPipelineMaker(BasePipelineMaker, Generic[T]):

    @staticmethod
    def is_tosa_ref_model_available():
        """Checks if the TOSA reference model is available."""
        # Not all deployments of ET have the TOSA reference model available.
        # Make sure we don't try to use it if it's not available.
        try:
            import tosa_reference_model  # type: ignore[import-not-found, import-untyped]

            # Check if the module has content
            return bool(dir(tosa_reference_model))
        except ImportError:
            return False

    def run(self):
        if (
            self.has_stage("run_method_and_compare_outputs")
            and not self.is_tosa_ref_model_available()
        ):
            _warnings.warn(
                "Warning: Skipping run_method_and_compare_outputs stage. TOSA reference model is not available."
            )
            self.pop_stage("run_method_and_compare_outputs")
        super().run()


class TosaPipelineINT(TOSAPipelineMaker, Generic[T]):
    """
    Lowers a graph to INT TOSA spec (with quantization) and tests it with the TOSA reference model.

    Attributes:
       module: The module which the pipeline is applied to.
       test_data: Data used for quantizing and testing the module.

       aten_ops: Aten dialect ops expected to be found in the graph after export.
       exir_ops: Exir dialect ops expected to be found in the graph after to_edge.
       if not using use_edge_to_transform_and_lower.

       run_on_tosa_ref_model: Set to true to test the tosa file on the TOSA reference model.

       tosa_version: A string for identifying the TOSA version, see common.get_tosa_compile_spec for
                     options.
       use_edge_to_transform_and_lower: Selects betweeen two possible ways of lowering the module.
       custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_op: str | List[str],
        exir_op: Optional[str | List[str]] = None,
        run_on_tosa_ref_model: bool = True,
        symmetric_io_quantization: bool = False,
        per_channel_quantization: bool = True,
        use_to_edge_transform_and_lower: bool = True,
        custom_path: str | None = None,
        tosa_debug_mode: Optional[ArmCompileSpec.DebugMode] = None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 1,
        dynamic_shapes: Optional[Tuple[Any]] = None,
        tosa_extensions: Optional[List[str]] = None,
        epsilon: float = 2**-12,
    ):
        if tosa_extensions is None:
            tosa_extensions = []
        tosa_profiles: dict[str, TosaSpecification] = {
            "1.0": TosaSpecification.create_from_string(
                "TOSA-1.0+INT" + "".join([f"+{ext}" for ext in tosa_extensions])
            ),
        }
        tosa_version = _require_tosa_version()

        compile_spec = common.get_tosa_compile_spec(
            tosa_profiles[tosa_version],
            custom_path=custom_path,
            tosa_debug_mode=tosa_debug_mode,
        )

        quantizer = TOSAQuantizer(tosa_profiles[tosa_version])
        # choose 16A8W quantization config when int16 extension is requested
        if "int16" in tosa_extensions:
            quantization_config = get_symmetric_a16w8_quantization_config(
                is_per_channel=per_channel_quantization, epsilon=epsilon
            )
        else:
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=per_channel_quantization
            )
        if symmetric_io_quantization:
            quantizer.set_io(quantization_config)
        quant_stage = Quantize(quantizer, quantization_config)

        super().__init__(
            module,
            test_data,
            aten_op,
            compile_spec,
            exir_op,
            use_to_edge_transform_and_lower,
            dynamic_shapes,
        )
        self.add_stage(self.tester.quantize, quant_stage, pos=0)

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )

        if _has_quantizable_inputs(test_data):
            # only add stages if we have quantizable input
            self.add_stage_after(
                "quantize",
                self.tester.check,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )
            self.add_stage_after(
                remove_quant_nodes_stage,
                self.tester.check_not,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )

        if run_on_tosa_ref_model:
            self.add_stage(
                self.tester.run_method_and_compare_outputs,
                atol=atol,
                rtol=rtol,
                qtol=qtol,
                inputs=self.test_data,
            )


class TosaPipelineFP(TOSAPipelineMaker, Generic[T]):
    """
    Lowers a graph to FP TOSA spec and tests it with the TOSA reference model.

    Attributes:
       module: The module which the pipeline is applied to.
       test_data: Data used for quantizing and testing the module.

       aten_ops: Aten dialect ops expected to be found in the graph after export.
       exir_ops: Exir dialect ops expected to be found in the graph after to_edge.
       if not using use_edge_to_transform_and_lower.

       run_on_tosa_ref_model: Set to true to test the tosa file on the TOSA reference model.

       tosa_version: A string for identifying the TOSA version, see common.get_tosa_compile_spec for
                     options.
       use_edge_to_transform_and_lower: Selects betweeen two possible ways of lowering the module.
       custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_op: str | List[str],
        exir_op: Optional[str | List[str]] = None,
        run_on_tosa_ref_model: bool = True,
        use_to_edge_transform_and_lower: bool = True,
        custom_path: str | None = None,
        tosa_debug_mode: Optional[ArmCompileSpec.DebugMode] = None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 0,
        dynamic_shapes: Optional[Tuple[Any]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
        tosa_extensions: Optional[List[str]] = None,
    ):
        if tosa_extensions is None:
            tosa_extensions = []
        tosa_profiles: dict[str, TosaSpecification] = {
            "1.0": TosaSpecification.create_from_string(
                "TOSA-1.0+FP" + "".join([f"+{ext}" for ext in tosa_extensions])
            ),
        }
        tosa_version = _require_tosa_version()

        compile_spec = common.get_tosa_compile_spec(
            tosa_profiles[tosa_version],
            custom_path=custom_path,
            tosa_debug_mode=tosa_debug_mode,
        )
        super().__init__(
            module,
            test_data,
            aten_op,
            compile_spec,
            exir_op,
            use_to_edge_transform_and_lower,
            dynamic_shapes=dynamic_shapes,
            transform_passes=transform_passes,
        )
        self.add_stage_after(
            "export",
            self.tester.check_not,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
            suffix="quant_nodes",
        )

        if run_on_tosa_ref_model:
            self.add_stage(
                self.tester.run_method_and_compare_outputs,
                atol=atol,
                rtol=rtol,
                qtol=qtol,
                inputs=self.test_data,
            )


class EthosU55PipelineINT(BasePipelineMaker, Generic[T]):
    """
    Lowers a graph to u55 INT TOSA spec and tests it on the Corstone300 FVP, if run_on_fvp is true.

    Attributes:
       module: The module which the pipeline is applied to.
       test_data: Data used for quantizing and testing the module.
       aten_ops: Aten dialect ops expected to be found in the graph after export.

       exir_ops: Exir dialect ops expected to be found in the graph after to_edge.
       if not using use_edge_to_transform_and_lower.
       run_on_fvp: Set to true to test the pte fileon a fvp simulator.
       use_edge_to_transform_and_lower: Selects betweeen two possible ways of lowering the module.
       custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | List[str],
        exir_ops: Optional[str | List[str]] = None,
        run_on_fvp: bool = True,
        symmetric_io_quantization: bool = False,
        per_channel_quantization: bool = True,
        a16w8_quantization: bool = False,
        use_to_edge_transform_and_lower: bool = True,
        custom_path: str | None = None,
        tosa_debug_mode: Optional[ArmCompileSpec.DebugMode] = None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 1,
        epsilon: float = 2**-12,
    ):
        compile_spec = common.get_u55_compile_spec(
            custom_path=custom_path,
            tosa_debug_mode=tosa_debug_mode,
        )
        quantizer = EthosUQuantizer(compile_spec)
        # choose int8 or int16 activation quantization
        if a16w8_quantization:
            quantization_config = get_symmetric_a16w8_quantization_config(
                is_per_channel=per_channel_quantization, epsilon=epsilon
            )
        else:
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=per_channel_quantization
            )
        if symmetric_io_quantization:
            quantizer.set_io(quantization_config)
        quant_stage = Quantize(quantizer, quantization_config)

        super().__init__(
            module,
            test_data,
            aten_ops,
            compile_spec,
            exir_ops,
            use_to_edge_transform_and_lower,
        )

        self.add_stage(self.tester.quantize, quant_stage, pos=0)

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )

        if _has_quantizable_inputs(test_data):
            # only add stages if we have quantizable input
            self.add_stage_after(
                "quantize",
                self.tester.check,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )
            self.add_stage_after(
                remove_quant_nodes_stage,
                self.tester.check_not,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )

        if run_on_fvp:
            self.add_stage(self.tester.serialize)
            self.add_stage(
                self.tester.run_method_and_compare_outputs,
                atol=atol,
                rtol=rtol,
                qtol=qtol,
                inputs=self.test_data,
            )


class EthosU85PipelineINT(BasePipelineMaker, Generic[T]):
    """
    Lowers a graph to u85 INT TOSA spec and tests it on the Corstone320 FVP, if run_on_fvp is true.

    Attributes:
       module: The module which the pipeline is applied to.
       test_data: Data used for quantizing and testing the module.
       aten_ops: Aten dialect ops expected to be found in the graph after export.

       exir_ops: Exir dialect ops expected to be found in the graph after to_edge if not using
                 use_edge_to_transform_and_lower.
       run_on_fvp: Set to true to test the pte fileon a fvp simulator.
       use_edge_to_transform_and_lower: Selects betweeen two possible ways of lowering the module.
       custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | List[str],
        exir_ops: str | List[str] | None = None,
        run_on_fvp: bool = True,
        symmetric_io_quantization: bool = False,
        per_channel_quantization: bool = True,
        a16w8_quantization: bool = False,
        use_to_edge_transform_and_lower: bool = True,
        custom_path: str | None = None,
        tosa_debug_mode: Optional[ArmCompileSpec.DebugMode] = None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 1,
        epsilon: float = 2**-12,
    ):
        compile_spec = common.get_u85_compile_spec(
            custom_path=custom_path,
            tosa_debug_mode=tosa_debug_mode,
        )
        quantizer = EthosUQuantizer(compile_spec)
        # choose int8 or int16 activation quantization
        if a16w8_quantization:
            quantization_config = get_symmetric_a16w8_quantization_config(
                is_per_channel=per_channel_quantization, epsilon=epsilon
            )
        else:
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=per_channel_quantization
            )
        if symmetric_io_quantization:
            quantizer.set_io(quantization_config)
        quant_stage = Quantize(quantizer, quantization_config)

        super().__init__(
            module,
            test_data,
            aten_ops,
            compile_spec,
            exir_ops,
            use_to_edge_transform_and_lower,
        )

        self.add_stage(self.tester.quantize, quant_stage, pos=0)

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )

        if _has_quantizable_inputs(test_data):
            # only add stages if we have quantizable input
            self.add_stage_after(
                "quantize",
                self.tester.check,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )
            self.add_stage_after(
                remove_quant_nodes_stage,
                self.tester.check_not,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )

        if run_on_fvp:
            self.add_stage(self.tester.serialize)
            self.add_stage(
                self.tester.run_method_and_compare_outputs,
                atol=atol,
                rtol=rtol,
                qtol=qtol,
                inputs=self.test_data,
            )


class PassPipeline(TOSAPipelineMaker, Generic[T]):
    """
    Runs single passes directly on an edge_program and checks operators before/after.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data used for quantizing and testing the module.
        tosa_version: The TOSA-version which to test for.

        ops_before_pass : Ops expected to be found in the graph before passes.
        ops_not_before_pass : Ops expected not to be found in the graph before passes.
        ops_after_pass : Ops expected to be found in the graph after passes.
        ops_notafter_pass : Ops expected not to be found in the graph after passes.

        pass_list: List of regular passes.
        pass_functions: List of functions applied directly to the exported program.
        passes_with_exported_program: List of passes initiated with an exported_program.
        custom_path : Path to dump intermediate artifacts such as tosa and pte to.

    Passes are run in order pass_list -> pass_functions -> passes_with_exported_program.
    See arm_tester.RunPasses() for more information.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        quantize: Optional[bool] = False,
        ops_before_pass: Optional[Dict[str, int]] = None,
        ops_not_before_pass: Optional[list[str]] = None,
        ops_after_pass: Optional[Dict[str, int]] = None,
        ops_not_after_pass: Optional[list[str]] = None,
        pass_list: Optional[List[Type[PassType]]] = None,
        pass_functions: Optional[List[Callable]] = None,
        passes_with_exported_program: Optional[List[Type[ExportPass]]] = None,
        custom_path: str | None = None,
        tosa_extensions: Optional[List[str]] = None,
    ):
        if tosa_extensions is None:
            tosa_extensions = []
        tosa_profiles: dict[str, TosaSpecification] = {
            "1.0": TosaSpecification.create_from_string(
                "TOSA-1.0+"
                + ("INT" if quantize else "FP")
                + "".join([f"+{ext}" for ext in tosa_extensions]),
            ),
        }
        tosa_version = _require_tosa_version()
        self.tosa_spec: TosaSpecification = tosa_profiles[tosa_version]

        compile_spec = common.get_tosa_compile_spec(
            self.tosa_spec, custom_path=custom_path
        )
        super().__init__(
            module,
            test_data,
            None,
            compile_spec,
            None,
            use_to_edge_transform_and_lower=False,
        )

        # Delete most of the pipeline
        self.pop_stage("check.exir")
        self.pop_stage("partition")
        self.pop_stage("check_not.exir")
        self.pop_stage("check_count.exir")
        self.pop_stage("to_executorch")
        self.pop_stage("check.aten")

        if quantize:
            self.add_stage(self.tester.quantize, pos=0)

        # Add checks/check_not's if given
        if ops_before_pass:
            self.add_stage(self.tester.check_count, ops_before_pass, suffix="before")
        if ops_not_before_pass:
            self.add_stage(self.tester.check_not, ops_not_before_pass, suffix="before")
        test_pass_stage = RunPasses(  # type: ignore[arg-type]
            pass_list, pass_functions, passes_with_exported_program  # type: ignore[arg-type]
        )  # Legacy pass APIs expose callable classes rather than ExportPass subclasses

        self.add_stage(self.tester.run_passes, test_pass_stage)

        if ops_after_pass:
            self.add_stage(self.tester.check_count, ops_after_pass, suffix="after")
        if ops_not_after_pass:
            self.add_stage(self.tester.check_not, ops_not_after_pass, suffix="after")
        self.add_stage(
            self.tester.run_method_and_compare_outputs,
            inputs=self.test_data,
        )

    def run(self):
        with TosaLoweringContext(self.tosa_spec):
            super().run()


class TransformAnnotationPassPipeline(TOSAPipelineMaker, Generic[T]):
    """
    Runs transform_for_annotation_pipeline passes directly on an exported program and checks output.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data used for testing the module.

        custom_path : Path to dump intermediate artifacts such as tosa and pte to.

    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        custom_path: str | None = None,
        tosa_extensions: Optional[List[str]] = None,
    ):
        if tosa_extensions is None:
            tosa_extensions = []
        tosa_profiles: dict[str, TosaSpecification] = {
            "1.0": TosaSpecification.create_from_string(
                "TOSA-1.0+INT" + "".join([f"+{ext}" for ext in tosa_extensions]),
            ),
        }
        tosa_version = _require_tosa_version()

        compile_spec = common.get_tosa_compile_spec(
            tosa_profiles[tosa_version], custom_path=custom_path
        )
        super().__init__(
            module,
            test_data,
            None,
            compile_spec,
            None,
            use_to_edge_transform_and_lower=True,
        )
        self.add_stage_after(
            "export", self.tester.run_transform_for_annotation_pipeline
        )

        # Delete most of the pipeline
        self.pop_stage("check_not.exir")
        self.pop_stage("check_count.exir")
        self.pop_stage("to_executorch")
        self.pop_stage("to_edge_transform_and_lower")
        self.pop_stage("check.aten")
        self.add_stage(
            self.tester.run_method_and_compare_outputs,
            inputs=test_data,
            run_eager_mode=True,
        )


class QuantizationPipeline(TOSAPipelineMaker, Generic[T]):
    """
    Runs quantization and checks that appropriate nodes are annotated with an expected
    quantization-spec.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data used for testing the module.
        quantizer: The quantizer to use for quantization.
        qspecs: Annotations to check for after quantization. A dict mapping
            operator names to a dict mapping QuantizationSpec (or None) to the number of times
            that spec should appear in the graph. A None QuantizationSpec indicates that the
            operator should not be quantized.
        input_qspecs: Annotations to check for after quantization on inputs.
        output_qspecs: Annotations to check for after quantization on outputs.
        custom_path : Path to dump intermediate artifacts to.

    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        quantizer: TOSAQuantizer,
        qspecs: Optional[Dict[str, Dict[QuantizationSpec | None, int]]] = None,
        input_qspecs: Optional[Dict[QuantizationSpec | None, int]] = None,
        output_qspecs: Optional[Dict[QuantizationSpec | None, int]] = None,
        custom_path: Optional[str] = None,
    ):
        tosa_spec = quantizer.tosa_spec
        compile_spec = common.get_tosa_compile_spec(tosa_spec, custom_path=custom_path)
        super().__init__(
            module,
            test_data,
            None,
            compile_spec,
            None,
            use_to_edge_transform_and_lower=True,
        )
        # TODO sort out typing
        quant_stage = Quantize(quantizer, quantization_config=quantizer.global_config)  # type: ignore[arg-type]
        self.add_stage(self.tester.quantize, quant_stage, pos=0)

        # Delete most of the pipeline
        self.pop_stage("check_count.exir")
        self.pop_stage("to_executorch")
        self.pop_stage("to_edge_transform_and_lower")
        self.pop_stage("check.aten")
        self.add_stage_after(
            "export",
            self.tester.check_quantization_annotation,
            qspecs,
            input_qspecs,
            output_qspecs,
        )


class OpNotSupportedPipeline(TOSAPipelineMaker, Generic[T]):
    """
    Runs the partitioner on a module and checks that ops are not delegated to test
    SupportedTOSAOperatorChecks.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data with a representative shape which the operator_check is performed on.
        tosa_version: The TOSA-version which to test for.

        non_delegated_ops : Exir ops expected not to be delegated.
        n_expected_delegates : Number of delegate calls (0 in the usual case).
        custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        non_delegated_ops: Dict[str, int],
        n_expected_delegates: int = 0,
        custom_path: str | None = None,
        quantize: Optional[bool] = False,
        u55_subset: Optional[bool] = False,
        tosa_extensions: Optional[List[str]] = None,
    ):
        if tosa_extensions is None:
            tosa_extensions = []
        tosa_profiles: dict[str, TosaSpecification] = {
            "1.0": TosaSpecification.create_from_string(
                "TOSA-1.0+"
                + ("INT" if quantize else "FP")
                + ("+u55" if u55_subset and quantize else "")
                + "".join([f"+{ext}" for ext in tosa_extensions]),
            ),
        }
        tosa_version = _require_tosa_version()

        tosa_spec = tosa_profiles[tosa_version]

        compile_spec = common.get_tosa_compile_spec(tosa_spec, custom_path=custom_path)
        super().__init__(
            module,
            test_data,
            [],
            compile_spec,
            [],
        )

        if tosa_spec.support_integer():
            quantizer = create_quantizer(compile_spec)
            quantizer.set_global(get_symmetric_quantization_config())
            quant_stage = Quantize(quantizer)
            self.add_stage(self.tester.quantize, quant_stage, pos=0)

        self.change_args("check_not.exir", [])
        self.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": n_expected_delegates,
                **non_delegated_ops,
            },
        )
        self.pop_stage("to_executorch")


class VgfPipeline(BasePipelineMaker, Generic[T]):
    """
    Lowers a graph based on TOSA spec (with or without quantization) and converts TOSA to VFG.

    Attributes:
       module: The module which the pipeline is applied to.
       test_data: Data used for quantizing and testing the module.

       aten_ops: Aten dialect ops expected to be found in the graph after export.
       exir_ops: Exir dialect ops expected to be found in the graph after to_edge.
       if not using use_edge_to_transform_and_lower.

       run_on_vulkan_runtime: Whether to test VGF output on VKML runtime.

       vgf_compiler_flags: Optional compiler flags.

       tosa_version: A string for identifying the TOSA version.

       use_edge_to_transform_and_lower: Selects betweeen two possible ways of lowering the module.
       custom_path : Path to dump intermediate artifacts such as tosa and pte to.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_op: str | List[str],
        exir_op: Optional[str | List[str]] = None,
        run_on_vulkan_runtime: bool = True,
        vgf_compiler_flags: Optional[str] = "",
        tosa_version: str = "TOSA-1.0+INT+FP",
        symmetric_io_quantization: bool = False,
        per_channel_quantization: bool = True,
        use_to_edge_transform_and_lower: bool = True,
        custom_path: str | None = None,
        tosa_debug_mode: Optional[ArmCompileSpec.DebugMode] = None,
        atol: float = 1e-03,
        rtol: float = 1e-03,
        qtol: int = 1,
        dynamic_shapes: Optional[Tuple[Any]] = None,
        transform_passes: Optional[
            Union[Sequence[PassType], Dict[str, Sequence[PassType]]]
        ] = None,
        tosa_extensions: Optional[List[str]] = None,
    ):

        if tosa_extensions is None:
            tosa_extensions = []
        tosa_spec = TosaSpecification.create_from_string(
            tosa_version + "".join([f"+{ext}" for ext in tosa_extensions])
        )
        compile_spec = common.get_vgf_compile_spec(
            tosa_spec,
            compiler_flags=vgf_compiler_flags,
            custom_path=custom_path,
            tosa_debug_mode=tosa_debug_mode,
        )

        super().__init__(
            module,
            test_data,
            aten_op,
            compile_spec,
            exir_op,
            use_to_edge_transform_and_lower,
            dynamic_shapes,
            transform_passes=transform_passes,
        )

        if tosa_spec.support_integer():
            quantizer = VgfQuantizer(compile_spec)
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=per_channel_quantization
            )
            if symmetric_io_quantization:
                quantizer.set_io(quantization_config)
            quant_stage = Quantize(quantizer, quantization_config)

            self.add_stage(self.tester.quantize, quant_stage, pos=0)

            remove_quant_nodes_stage = (
                "to_edge_transform_and_lower"
                if use_to_edge_transform_and_lower
                else "partition"
            )

            if _has_quantizable_inputs(test_data):
                # only add stages if we have quantizable input
                self.add_stage_after(
                    "quantize",
                    self.tester.check,
                    [
                        "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                        "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                    ],
                    suffix="quant_nodes",
                )
                self.add_stage_after(
                    remove_quant_nodes_stage,
                    self.tester.check_not,
                    [
                        "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                        "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                    ],
                    suffix="quant_nodes",
                )
        else:
            self.add_stage_after(
                "export",
                self.tester.check_not,
                [
                    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                    "torch.ops.quantized_decomposed.quantize_per_tensor.default",
                ],
                suffix="quant_nodes",
            )

        if run_on_vulkan_runtime:
            self.add_stage(self.tester.serialize)
            self.add_stage(
                self.tester.run_method_and_compare_outputs,
                atol=atol,
                rtol=rtol,
                qtol=qtol,
                inputs=self.test_data,
            )
        self.run_on_vulkan_runtime = run_on_vulkan_runtime

    # TODO: Remove once CI fully working
    def run(self):
        import pytest

        if self.run_on_vulkan_runtime:
            try:
                super().run()
            except FileNotFoundError as e:
                pytest.skip(f"VKML executor_runner not found - not built - skip {e}")
        else:
            super().run()
