# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tester that lowers with the Cadence AOT flow and runs on the x86 cadence_runner.

Build the runner with backends/cadence/build_cadence_runner.sh first, and run
from the repository root: Executor finds the runner by a relative path.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import torch
from executorch.backends.cadence.aot.compiler import (
    _lower_ep_to_cadence,
    apply_pre_edge_transform_passes,
    get_cadence_executorch_backend_config,
    get_fake_quant_model,
)
from executorch.backends.cadence.aot.pass_utils import CompileMode
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceQuantizer
from executorch.backends.cadence.aot.utils import get_default_memory_config
from executorch.backends.cadence.runtime.runtime import run_bundled
from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import (
    Quantize,
    Serialize,
    StageType,
    ToEdgeTransformAndLower,
    ToExecutorch,
)
from executorch.exir import ExecutorchProgramManager
from torch.export import ExportedProgram


class CadenceQuantize(Quantize):
    """Quantizes as the Cadence AOT flow does: it decomposes all but the
    quantizer's ops before prepare and convert."""

    def run(self, artifact: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        self.converted_graph = get_fake_quant_model(
            artifact, inputs, self.quantizer, self.calibration_samples
        )


class CadenceToEdgeTransformAndLower(ToEdgeTransformAndLower):
    def __init__(self, quantizer: CadenceQuantizer | None, mode: CompileMode) -> None:
        super().__init__()
        self.quantizer = quantizer
        self.mode = mode

    def run(
        self,
        artifact: ExportedProgram,
        inputs: tuple[Any, ...] | None = None,
        generate_etrecord: bool = False,
    ) -> None:
        # The Cadence passes rewrite the program in place, and the harness takes
        # its reference outputs from the Export stage's program.
        program = copy.deepcopy(artifact)
        if self.quantizer is not None:
            program = apply_pre_edge_transform_passes(program, self.quantizer)
        self.edge_dialect_program = _lower_ep_to_cadence(
            program, mode=self.mode, generate_etrecord=generate_etrecord
        )


class CadenceSerialize(Serialize):
    """Runs the program on cadence_runner and reads the outputs from its ETDump."""

    def run(self, artifact: ExecutorchProgramManager, inputs: Any = None) -> None:
        super().run(artifact, inputs)
        self.executorch_program = artifact

    def run_artifact(self, inputs: tuple[Any, ...]) -> Any:
        return run_bundled(self.executorch_program, inputs)


class CadenceTester(Tester):
    """Lowers with the Cadence AOT flow and runs the program on cadence_runner.

    quantizer drives both the quantize stage and the quantized-pattern fusion,
    which must use the same quantizer. mode drives both the Cadence passes and
    the memory planning.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: tuple[Any, ...],
        quantizer: CadenceQuantizer | None = None,
        mode: CompileMode = CompileMode.DEFAULT,
    ) -> None:
        super().__init__(
            module,
            example_inputs,
            stage_classes=Tester.default_stage_classes()
            | {
                StageType.QUANTIZE: partial(CadenceQuantize, quantizer),
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: partial(
                    CadenceToEdgeTransformAndLower, quantizer, mode
                ),
                StageType.TO_EXECUTORCH: partial(
                    ToExecutorch,
                    get_cadence_executorch_backend_config(
                        get_default_memory_config(), mode
                    ),
                ),
                StageType.SERIALIZE: CadenceSerialize,
            },
        )
