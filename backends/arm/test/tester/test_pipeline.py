# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Generic, List, TypeVar

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec


logger = logging.getLogger(__name__)
T = TypeVar("T")
""" Generic type used for test data in the pipeline. Depends on which type the operator expects."""


class BasePipelineMaker(Generic[T]):
    """
    The BasePiplineMaker defines a list of stages to be applied to a torch.nn.module for lowering it in the Arm backend. To be inherited and adjusted for particular targets.
    Importantly, the pipeline list can be modified before running the pipeline to support various pipeline extensions and debugging usecases.

    Attributes:
        module: The module which the pipeline is applied to.
        test_data: Data used for quantizing and testing the module.
        aten_ops: Aten dialect ops expected to be found in the graph after export.
        exir_ops: Exir dialect ops expected to be found in the graph after to_edge.
        compile_spec: The compile spec used in the lowering process
        use_edge_to_transform_and_lower: Selects betweeen two possible routes for lowering the module:
                tester.to_edge_transform_and_lower()
            or
                tester.to_edge().check(exir_ops).partition()
    """

    class PipelineStage:
        """
        Helper class to store a pipeline stage as a function call + args for calling later on.

        Attributes:
            id: name of the function to be called, used for refering to stages in the pipeline
            func: handle to the function to be called
            args: args used when called
            kwargs: kwargs used when called
            is_called: keeps track of if the function has been called
        """

        def __init__(self, func, *args, **kwargs):
            self.id: str = func.__name__
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

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | List[str],
        exir_ops: str | List[str],
        compile_spec: List[CompileSpec],
        use_to_edge_transform_and_lower: bool = False,
    ):

        self.tester = ArmTester(
            module, example_inputs=test_data, compile_spec=compile_spec
        )

        self.aten_ops = aten_ops if isinstance(aten_ops, list) else [aten_ops]
        self.exir_ops = exir_ops if isinstance(exir_ops, list) else [exir_ops]
        self.test_data = test_data
        self._stages = []

        self.add_stage(-1, self.tester.export)
        self.add_stage(-1, self.tester.check, self.aten_ops)
        if use_to_edge_transform_and_lower:
            self.add_stage(-1, self.tester.to_edge_transform_and_lower)

        else:
            self.add_stage(-1, self.tester.to_edge)
            self.add_stage(-1, self.tester.check, self.exir_ops)
            self.add_stage(-1, self.tester.partition)
        self.add_stage(-1, self.tester.check_not, self.exir_ops)
        self.add_stage(
            -1,
            self.tester.check_count,
            {"torch.ops.higher_order.executorch_call_delegate": 1},
        )
        self.add_stage(-1, self.tester.to_executorch)

    def add_stage(self, pos: int, func: Callable, *args, **kwargs):
        """Adds a stage defined by a function with arguments to the pipeline at index pos. Pos wraps around the list for negative values."""
        pipeline_stage = self.PipelineStage(func, *args, **kwargs)
        pipeline_length = len(self._stages)

        if pos < 0:
            pos = pipeline_length + (pos + 1)

        if not -pipeline_length <= pos <= pipeline_length:
            raise ValueError(
                f"Pos must be between [-{pipeline_length}, {pipeline_length}]"
            )

        self._stages.insert(pos, pipeline_stage)

        logger.debug(f"Added stage {func.__name__} to {type(self).__name__}")

        return self

    def pop_stage(self, pos: int):
        """Removes and returns the stage at postion pos"""
        return self._stages.pop(pos)

    def find_pos(self, stage_id: str):
        """Returns the position of the stage id. Note that this only finds the first stage with the given id, i.e. it should only be used with unique stages."""
        for i, stage in enumerate(self._stages):
            if stage.id == stage_id:
                return i

        raise Exception(f"Stage id {stage_id} not found in pipeline")

    def add_stage_after(self, stage_id: str, func: Callable, *args, **kwargs):
        """Adds a stage after the given stage id. Note that this only finds the first stage with the given id, i.e. it should only be used with unique stages."""
        pos = self.find_pos(stage_id)
        self.add_stage(pos + 1, func, *args, **kwargs)
        return self

    def dump_artifact(self, stage_id: str):
        """Adds a dump_artifact stage after the given stage id. Note that this only finds the first stage with the given id, i.e. it should only be used with unique stages."""
        self.add_stage_after(stage_id, self.tester.dump_artifact)
        return self

    def dump_operator_distribution(self, stage_id: str):
        """Adds a dump_operator_distribution stage after the given stage id. Note that this only finds the first stage with the given id, i.e. it should only be used with unique stages."""
        self.add_stage_after(stage_id, self.tester.dump_operator_distribution)
        return self

    def change_args(self, stage_id: str, *args, **kwargs):
        """Updates the args to the given stage id. Note that this only finds the first stage with the given id, i.e. it should only be used with unique stages."""
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


class TosaPipelineBI(BasePipelineMaker, Generic[T]):
    """Lowers a graph to BI TOSA spec (with quantization) and tests it with the TOSA reference model."""

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: Any,
        aten_op: str,
        exir_op: str,
        tosa_version: str = "TOSA-0.80+BI",
        use_to_edge_transform_and_lower: bool = False,
    ):
        compile_spec = common.get_tosa_compile_spec(
            tosa_version,
        )
        super().__init__(
            module,
            test_data,
            aten_op,
            exir_op,
            compile_spec,
            use_to_edge_transform_and_lower,
        )
        self.add_stage(0, self.tester.quantize)
        self.add_stage_after(
            "quantize",
            self.tester.check,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )
        self.add_stage_after(
            remove_quant_nodes_stage,
            self.tester.check_not,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        self.add_stage(
            -1, self.tester.run_method_and_compare_outputs, inputs=self.test_data
        )


class TosaPipelineMI(BasePipelineMaker, Generic[T]):
    """Lowers a graph to MI TOSA spec and tests it with the TOSA reference model"""

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: Any,
        aten_op: str,
        exir_op: str,
        tosa_version: str = "TOSA-0.80+MI",
        use_to_edge_transform_and_lower: bool = False,
    ):
        compile_spec = common.get_tosa_compile_spec(
            tosa_version,
        )
        super().__init__(
            module,
            test_data,
            aten_op,
            exir_op,
            compile_spec,
            use_to_edge_transform_and_lower,
        )
        self.add_stage_after(
            "export",
            self.tester.check_not,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        self.add_stage(
            -1, self.tester.run_method_and_compare_outputs, inputs=self.test_data
        )


class EthosU55PipelineBI(BasePipelineMaker, Generic[T]):
    """Lowers a graph to u55 BI TOSA spec and tests it on the Corstone300 FVP, if run_on_fvp is true."""

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | List[str],
        exir_ops: str | List[str],
        run_on_fvp: bool = False,
        use_to_edge_transform_and_lower: bool = False,
    ):
        compile_spec = common.get_u55_compile_spec()
        super().__init__(
            module,
            test_data,
            aten_ops,
            exir_ops,
            compile_spec,
            use_to_edge_transform_and_lower,
        )
        self.add_stage(0, self.tester.quantize)
        self.add_stage_after(
            "quantize",
            self.tester.check,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )
        self.add_stage_after(
            remove_quant_nodes_stage,
            self.tester.check_not,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        if run_on_fvp:
            self.add_stage(-1, self.tester.serialize)
            self.add_stage(
                -1,
                self.tester.run_method_and_compare_outputs,
                qtol=1,
                inputs=self.test_data,
            )


class EthosU85PipelineBI(BasePipelineMaker, Generic[T]):
    """Lowers a graph to u85 BI TOSA spec and tests it on the Corstone320 FVP, if run_on_fvp is true."""

    def __init__(
        self,
        module: torch.nn.Module,
        test_data: T,
        aten_ops: str | List[str],
        exir_ops: str | List[str],
        run_on_fvp: bool = False,
        use_to_edge_transform_and_lower: bool = False,
    ):
        compile_spec = common.get_u85_compile_spec()
        super().__init__(
            module,
            test_data,
            aten_ops,
            exir_ops,
            compile_spec,
            use_to_edge_transform_and_lower,
        )
        self.add_stage(0, self.tester.quantize)
        self.add_stage_after(
            "quantize",
            self.tester.check,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        remove_quant_nodes_stage = (
            "to_edge_transform_and_lower"
            if use_to_edge_transform_and_lower
            else "partition"
        )
        self.add_stage_after(
            remove_quant_nodes_stage,
            self.tester.check_not,
            [
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
                "torch.ops.quantized_decomposed.quantize_per_tensor.default",
            ],
        )

        if run_on_fvp:
            self.add_stage(-1, self.tester.serialize)
            self.add_stage(
                -1,
                self.tester.run_method_and_compare_outputs,
                qtol=1,
                inputs=self.test_data,
            )
