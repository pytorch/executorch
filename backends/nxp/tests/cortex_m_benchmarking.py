# Copyright 2025-2026 Arm Limited and/or its affiliates.
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMQuantize, CortexMTester
from executorch.backends.nxp.tests.utils import (
    process_input_sample,
    process_output_sample,
    read_prepared_samples,
    store_results,
)
from executorch.backends.test.harness.stages import StageType


class CortexMNXPBenchmarkTester(CortexMTester):
    def __init__(
        self,
        module,
        example_inputs,
        target_config: CortexMTargetConfig | None = None,
        timeout: int = 120,
    ):
        target_config = target_config or CortexMTargetConfig(
            cpu=CortexM.M33
        )  # set default to M33 for NXP boards
        super().__init__(module, example_inputs, target_config, timeout)

    def run_benchmark(
        self,
        calibration_samples,
        input_spec,
        output_spec,
        testing_dataset_dir,
        cpu_results_dir,
        npu_results_dir,
    ):
        quantization_stage = CortexMQuantize(calibration_samples=calibration_samples)

        self.quantize(quantization_stage)
        self.export()
        self.to_edge()
        self.run_passes()
        self.to_executorch()
        self.serialize()
        self.run_program(
            input_spec,
            output_spec,
            testing_dataset_dir,
            cpu_results_dir,
            npu_results_dir,
        )

        return self.stages[StageType.SERIALIZE].executorch_program_manager

    def run_program(
        self,
        input_spec,
        output_spec,
        testing_dataset_dir,
        cpu_results_dir,
        npu_results_dir,
    ):
        all_outputs = []

        for input_samples in read_prepared_samples(testing_dataset_dir, input_spec):
            current_input_samples = process_input_sample(input_spec, input_samples)

            # Run the model.
            output = self.stages[StageType.SERIALIZE].run_artifact(
                *current_input_samples
            )
            current_outputs = process_output_sample(output, output_spec)
            all_outputs.append(current_outputs)

        # Store all the results.
        store_results(all_outputs, cpu_results_dir, npu_results_dir)
