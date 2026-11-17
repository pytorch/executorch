# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.quantizer import EspNNQuantizer
from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import (
    Quantize,
    RunPasses,
    StageType,
    ToEdgeTransformAndLower,
)
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from torch.utils._pytree import tree_flatten


class EspRunPasses(RunPasses):
    def __init__(self):
        super().__init__(EspPassManager)

    def run(self, artifact: EdgeProgramManager, inputs=None) -> None:
        self.edge_or_aten_program = artifact.transform(EspPassManager())


class EspToEdgeTransformAndLower(ToEdgeTransformAndLower):
    def __init__(self):
        super().__init__(edge_compile_config=EdgeCompileConfig())

    def run(self, artifact, inputs=None, generate_etrecord: bool = False) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            transform_passes=EspPassManager(),
            generate_etrecord=generate_etrecord,
        )


class EspTester(Tester):
    def __init__(self, module, example_inputs, calibration_samples=None):
        stages = self.default_stage_classes()
        stages[StageType.QUANTIZE] = lambda: Quantize(
            EspNNQuantizer(), calibration_samples=calibration_samples
        )
        stages[StageType.RUN_PASSES] = EspRunPasses
        stages[StageType.TO_EDGE_TRANSFORM_AND_LOWER] = EspToEdgeTransformAndLower
        super().__init__(module, example_inputs, stages)

    def test_dialect(
        self,
        ops_before_transforms,
        ops_after_transforms,
        *,
        qtol=0,
        atol=1e-3,
        rtol=1e-3,
    ):
        return (
            self.quantize()
            .export()
            .to_edge()
            .check_node_count(ops_before_transforms)
            .run_passes()
            .check_node_count(ops_after_transforms)
            .run_method_and_compare_outputs(
                inputs=self.example_inputs, qtol=qtol, atol=atol, rtol=rtol
            )
        )

    def test_implementation(self, repo: Path, *, atol=0, rtol=0):
        build_path = os.environ.get("ESPRESSIF_BUILD_DIR")
        if not build_path:
            pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")

        self.quantize().export().to_edge().run_passes()
        expected, _ = tree_flatten(
            self.get_artifact().exported_program().module()(*self.example_inputs)
        )
        self.to_executorch().serialize()
        bundle = BundledProgram(
            self.get_artifact(StageType.TO_EXECUTORCH),
            [
                MethodTestSuite(
                    "forward", [MethodTestCase(self.example_inputs, expected)]
                )
            ],
        )

        build_dir = Path(build_path).resolve()
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        firmware_dir = build_dir / f"operator-library-{worker}"
        firmware_dir.mkdir(parents=True, exist_ok=True)
        model_path = firmware_dir / "test.bpte"
        model_path.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))
        commands = [
            (
                "build.log",
                600,
                [
                    "idf.py",
                    f"--project-dir={repo / 'examples/espressif/project'}",
                    f"--build-dir={firmware_dir}",
                    "-DIDF_TARGET=esp32s3",
                    f"-DSDKCONFIG={firmware_dir / 'sdkconfig'}",
                    "-DET_ESPRESSIF_OPS=ON",
                    "-DET_ESPNN_BACKEND=OFF",
                    f"-DET_BUILD_DIR_PATH={build_dir / 'runtime'}",
                    f"-DET_PTE_FILE_PATH={model_path}",
                    "-DET_BUNDLE_IO=ON",
                    "-DET_NUM_INFERENCES=1",
                    f"-DET_ATOL={atol}",
                    f"-DET_RTOL={rtol}",
                    "build",
                    "merge-bin",
                ],
            ),
            (
                "emulator.log",
                90,
                [
                    "esp-emu",
                    "--chip=esp32s3",
                    f"--firmware={firmware_dir / 'merged-binary.bin'}",
                    f"--elf={firmware_dir / 'executorch_esp_runner.elf'}",
                    "--net=user",
                    "--psram-size=8M",
                    "--timeout=60s",
                    "--exit-on=Program complete.",
                ],
            ),
            (
                "check.log",
                60,
                [
                    sys.executable,
                    str(repo / "examples/espressif/test_emulator.py"),
                    str(firmware_dir / "emulator.log"),
                ],
            ),
        ]
        for log_name, timeout, command in commands:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            (firmware_dir / log_name).write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
        return self


@dataclass
class EspTestCase:
    model: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    calibration_samples: list[tuple[torch.Tensor, ...]] | None = None


def ramp_tensor(start: float, end: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.linspace(start, end, math.prod(shape)).reshape(shape)
