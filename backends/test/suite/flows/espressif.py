# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Espressif operator-library flow for the backend test suite.

Each case is quantized with the Espressif Fused Quant recipe and lowered with
EspPassManager. Supported operators become direct ESP-NN calls, and the rest
run on portable kernels. The program then runs on an emulated ESP32 with esp-emu.

The serialize stage builds the ESP-IDF runner from examples/espressif/project
around a bundled program that carries the case's inputs, runs it in the
emulator and reads the outputs back from the runner's output dump. The suite
compares them with the quantized reference on the host.

Requirements, checked when the flow registers:
- ESPRESSIF_BUILD_DIR, holding an ExecuTorch runtime for the target in
  ESPRESSIF_BUILD_DIR/runtime, built with every portable kernel the cases
  need (see .ci/scripts/test_espressif.sh for the runtime build);
- idf.py from ESP-IDF and esp-emu on PATH.

ESPRESSIF_TARGET picks the chip (esp32s3 by default; esp32p4 and esp32s31
also run in esp-emu). ESPRESSIF_NN_KERNELS picks the ESP-NN build (optimized
by default, or ansi).

Comparison uses the suite's fixed atol of 1e-1, which is not tuned for int8,
so a mismatch is not by itself evidence of a lowering bug.
"""

import os
import re
from functools import partial
from pathlib import Path

import torch

from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import (
    EspExportFusedQuant,
    EspTester,
    EspToEdgeTransformAndLower,
    run_bundled_program_on_emulator,
)
from executorch.backends.test.harness.stages import Serialize, StageType
from executorch.backends.test.suite.flow import TestFlow
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.exir import ExecutorchProgramManager
from executorch.frontends.fused_quant.recipe import get_fused_quant_edge_compile_config
from torch.utils._pytree import tree_flatten

_REPO = Path(__file__).resolve().parents[4]

# The runner prints each element of each output tensor on its own line.
_OUTPUT_LINE = re.compile(
    r"^Output\[(\d+)\]\[(\d+)\]: \((int|float|char|bool)\) (\S+)(?: \((0x[0-9a-f]+)\))?",
    re.MULTILINE,
)
_PARSED_DTYPES = {
    torch.float32: "float",
    torch.int32: "int",
    torch.int8: "char",
    torch.bool: "bool",
}

# A runner with every portable kernel is larger than the example project's
# app partitions, so the flow gives it a 16 MB flash and one app partition.
_PARTITIONS = """\
# Name,   Type, SubType, Offset,  Size
nvs,      data, nvs,     0x10000, 0x6000
phy_init, data, phy,     ,        0x1000
factory,  app,  factory, 0x20000, 0xF00000
"""

# Runner lines that explain why it printed no outputs.
_RUNNER_FAILURE = re.compile(
    r"(?:Check failed|Missing operator|Memory allocation failed)[^\n]*"
)

# esp-emu gives the runner 8 MB of PSRAM. The 2 MB default method allocator
# pool is too small for the largest cases, such as test_linear_feature_sizes
# with its 2.5 MB memory-planned buffer.
_METHOD_POOL_BYTES = 4 * 1024 * 1024

# The bundle carries placeholder outputs. The runner's own check must pass,
# or it stops before "Program complete.", so its tolerance is set out of
# reach. The suite compares the real outputs.
_BUNDLE_ATOL = 1e30


class _ExportFusedQuant(EspExportFusedQuant):
    """Quantizes and exports in the suite's export step.

    The Fused Quant recipe quantizes during export, so the flow does not use a
    separate quantize stage. The exported program is also the suite's
    reference, so the emulator output is compared with the quantized model.
    """

    def __init__(self, dynamic_shapes=None):
        super().__init__(quantizer=None, calibration_samples=None)


def _output_specs(program: ExecutorchProgramManager) -> list[torch.Tensor]:
    graph = program.exported_program().graph
    outputs, _ = tree_flatten(graph.output_node().args[0])
    return [node.meta["val"] for node in outputs]


def _parse_outputs(log: str, specs: list[torch.Tensor]) -> list[torch.Tensor]:
    values: dict[int, dict[int, float]] = {i: {} for i in range(len(specs))}
    for match in _OUTPUT_LINE.finditer(log):
        index, element, kind, text, raw = match.groups()
        if kind == "bool":
            value = int(raw, 16) != 0
        elif kind == "float":
            value = float(text)
        else:
            value = int(text)
        values.setdefault(int(index), {})[int(element)] = value

    outputs = []
    for index, spec in enumerate(specs):
        if spec.dtype not in _PARSED_DTYPES:
            raise RuntimeError(
                f"The ESP32 runner does not print {spec.dtype} outputs (output {index})"
            )
        found = values.get(index, {})
        if len(found) != spec.numel():
            failures = _RUNNER_FAILURE.findall(log)
            raise RuntimeError(
                f"Output {index}: expected {spec.numel()} values in the emulator log, "
                f"found {len(found)}"
                + (f". The runner reported: {failures[-1]}" if failures else "")
            )
        flat = [found[i] for i in range(spec.numel())]
        outputs.append(torch.tensor(flat, dtype=spec.dtype).reshape(spec.shape))
    return outputs


class EspEmulatorSerialize(Serialize):
    """Serializes the program and runs it on an emulated ESP32."""

    def __init__(self, target: EspTarget, kernel_mode: EspKernelMode):
        super().__init__()
        self.target = target
        self.kernel_mode = kernel_mode
        self.program: ExecutorchProgramManager | None = None

    def run(self, artifact: ExecutorchProgramManager, inputs=None) -> None:
        super().run(artifact, inputs)
        self.program = artifact

    def run_artifact(self, inputs):
        specs = _output_specs(self.program)
        placeholders = [torch.zeros(spec.shape, dtype=spec.dtype) for spec in specs]
        bundle = BundledProgram(
            self.program,
            [MethodTestSuite("forward", [MethodTestCase(inputs, placeholders)])],
        )
        build_dir = Path(os.environ["ESPRESSIF_BUILD_DIR"]).resolve()
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        firmware_dir = (
            build_dir / f"suite-{worker}-{self.target.value}-{self.kernel_mode.value}"
        )
        firmware_dir.mkdir(parents=True, exist_ok=True)
        partitions = firmware_dir / "partitions_suite.csv"
        partitions.write_text(_PARTITIONS)
        log = run_bundled_program_on_emulator(
            bundle,
            repo=_REPO,
            build_dir=build_dir,
            firmware_dir=firmware_dir,
            target=self.target,
            kernel_mode=self.kernel_mode,
            atol=_BUNDLE_ATOL,
            check_outputs=False,
            emulator_timeout=300,
            extra_defaults=(
                "CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y\n"
                "CONFIG_PARTITION_TABLE_CUSTOM=y\n"
                f'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="{partitions}"\n'
            ),
            extra_cmake_args=(
                f"-DET_ESP_METHOD_ALLOCATOR_POOL_SIZE={_METHOD_POOL_BYTES}",
            ),
        )
        return _parse_outputs(log, specs)


class EspSuiteTester(EspTester):
    def __init__(
        self,
        module,
        example_inputs,
        *,
        target: EspTarget,
        kernel_mode: EspKernelMode,
    ):
        super().__init__(module, example_inputs, target=target, kernel_mode=kernel_mode)
        self.stage_classes[StageType.EXPORT] = _ExportFusedQuant
        self.stage_classes[StageType.TO_EDGE_TRANSFORM_AND_LOWER] = partial(
            EspToEdgeTransformAndLower,
            edge_compile_config=get_fused_quant_edge_compile_config(()),
            target=self.target,
            kernel_mode=self.kernel_mode,
        )
        self.stage_classes[StageType.SERIALIZE] = partial(
            EspEmulatorSerialize, self.target, self.kernel_mode
        )


def _create_espressif_flow() -> TestFlow:
    target = EspTarget(os.environ.get("ESPRESSIF_TARGET", EspTarget.ESP32S3.value))
    kernel_mode = EspKernelMode(
        os.environ.get("ESPRESSIF_NN_KERNELS", EspKernelMode.OPTIMIZED.value)
    )
    return TestFlow(
        f"espressif_{target.value}",
        backend="espressif",
        tester_factory=partial(EspSuiteTester, target=target, kernel_mode=kernel_mode),
        is_delegated=False,
        param_skip_reasons={
            "use_dynamic_shapes": {
                True: "The Espressif operator library lowers for a fixed shape."
            }
        },
    )


ESPRESSIF_TEST_FLOW = _create_espressif_flow()
