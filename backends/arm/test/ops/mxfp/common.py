# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Callable, Generic, TypeVar

import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.test.tester.analyze_output_utils import (
    compare_rel_frobenius_and_cosine_similarity,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from executorch.backends.test.harness.stages import Stage, StageType

T = TypeVar("T", bound=tuple[Any, ...])


class ConvertToMXFP(Stage):
    def __init__(
        self,
        config: MXFPOpConfig,
        filter_fn: Callable[[torch.nn.Module, str], bool],
    ) -> None:
        self.config = config
        self.filter_fn = filter_fn
        self.converted_module: torch.nn.Module | None = None

    def stage_type(self) -> StageType:
        return StageType.QUANTIZE

    def run(self, artifact: torch.nn.Module, inputs=None) -> None:
        self.converted_module = copy.deepcopy(artifact)
        to_mxfp(self.converted_module, self.config, filter_fn=self.filter_fn)

    @property
    def artifact(self) -> torch.nn.Module:
        assert self.converted_module is not None
        return self.converted_module

    @property
    def graph_module(self) -> torch.nn.Module:
        assert self.converted_module is not None
        return self.converted_module

    def run_artifact(self, inputs):
        assert self.converted_module is not None
        return self.converted_module.forward(*inputs)


def _configure_mxfp_pipeline(
    pipeline: TosaPipelineFP | VgfPipeline,
    config: MXFPOpConfig,
    filter_fn: Callable[[torch.nn.Module, str], bool],
    frobenius_threshold: float | None,
    cosine_threshold: float | None,
) -> None:
    pipeline.add_stage(
        pipeline.tester.quantize,
        ConvertToMXFP(config, filter_fn),
        pos=0,
    )
    if pipeline.has_stage("run_method_and_compare_outputs"):
        compare_stage = pipeline._stages[
            pipeline.find_pos("run_method_and_compare_outputs")
        ]
        compare_stage.kwargs["reference_stage_type"] = StageType.INITIAL_MODEL
        compare_stage.kwargs["compare_callback"] = lambda ref, test, qparams: (
            compare_rel_frobenius_and_cosine_similarity(
                ref,
                test,
                qparams,
                frobenius_threshold=frobenius_threshold,
                cosine_threshold=cosine_threshold,
                clean_reference=False,
            )
        )


class MXFPTosaPipelineFP(TosaPipelineFP[T], Generic[T]):
    def __init__(
        self,
        *args,
        filter_fn: Callable[[torch.nn.Module, str], bool],
        frobenius_threshold: float | None,
        cosine_threshold: float | None,
        mxfp_config: MXFPOpConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        _configure_mxfp_pipeline(
            self,
            mxfp_config if mxfp_config is not None else MXFPOpConfig(),
            filter_fn,
            frobenius_threshold,
            cosine_threshold,
        )


class MXFPVgfPipeline(VgfPipeline[T], Generic[T]):
    def __init__(
        self,
        *args,
        filter_fn: Callable[[torch.nn.Module, str], bool],
        frobenius_threshold: float | None,
        cosine_threshold: float | None,
        mxfp_config: MXFPOpConfig | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("quantize", False)
        super().__init__(*args, **kwargs)
        _configure_mxfp_pipeline(
            self,
            mxfp_config if mxfp_config is not None else MXFPOpConfig(),
            filter_fn,
            frobenius_threshold,
            cosine_threshold,
        )
