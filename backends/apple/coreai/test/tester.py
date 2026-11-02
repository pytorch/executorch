# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
from collections import OrderedDict

import pytest
import torch
from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.quantizer.quantizer import CoreAIQuantizer
from executorch.backends.test.harness import stages, Tester
from executorch.backends.test.harness.stages import StageType
from executorch.backends.test.harness.stages.stage import Stage
from executorch.exir import to_edge_transform_and_lower
from torch.utils._pytree import tree_flatten, tree_map


def _fp16_inputs(inputs):
    return tree_map(
        lambda value: (
            value.half()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        ),
        inputs,
    )


# FP16 rounding error grows with the network's activation scale rather than
# each element's value, so outputs near zero (e.g. cancelling logits) can carry
# error far above a per-element rtol. Allow a fraction of each output's largest
# magnitude instead.
_FP16_ATOL_FRACTION_OF_MAX = 0.02


def _fp16_atol(reference, atol):
    if not (isinstance(reference, torch.Tensor) and reference.is_floating_point()):
        return atol
    finite = reference[torch.isfinite(reference)].float()
    if finite.numel() == 0:
        return atol
    return max(atol, _FP16_ATOL_FRACTION_OF_MAX * finite.abs().max().item())


class CoreAIQuantize(Stage):
    def __init__(self, mode="w8a8", *, skip_if_unquantizable=False):
        if mode not in ("w8a8", "w8", "w4"):
            raise ValueError(f"Unsupported CoreAI quantization mode: {mode}")
        self.mode = mode
        self.skip_if_unquantizable = skip_if_unquantizable
        self.converted_graph = None

    def stage_type(self):
        return StageType.QUANTIZE

    def run(self, artifact, inputs):
        from coreai_opt.quantization import QuantizerConfig

        config = (
            QuantizerConfig()
            if self.mode == "w8a8"
            else getattr(QuantizerConfig.presets, self.mode)()
        )
        quantizer = CoreAIQuantizer(copy.deepcopy(artifact), config)
        prepared = quantizer.prepare(tuple(inputs))
        with quantizer.calibration_mode():
            prepared(*inputs)
        try:
            self.converted_graph = quantizer.convert()
        except ValueError as error:
            no_targets = (
                str(error) == "Model contains no fake quantization nodes to convert"
            )
            if self.skip_if_unquantizable and no_targets:
                pytest.skip(f"{self.mode}: no applicable quantization targets")
            raise

    @property
    def artifact(self):
        return self.converted_graph

    @property
    def graph_module(self):
        return self.converted_graph

    def run_artifact(self, inputs):
        return self.converted_graph(*inputs)


class CoreAILower(stages.ToEdgeTransformAndLower):
    def __init__(self, partitioners=None, edge_compile_config=None):
        super().__init__(
            default_partitioner_cls=CoreAIPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config or get_default_compile_config(),
        )

    def run(self, artifact, inputs=None, generate_etrecord=False):
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            partitioner=self.partitioners,
            transform_passes=get_default_passes(),
            generate_etrecord=generate_etrecord,
        )


class CoreAISerialize(stages.Serialize):
    def run_artifact(self, inputs):
        from executorch.extension.pybindings.portable_lib import (
            _is_available,
            _load_program_from_buffer,
            Verification,
        )

        if not _is_available("CoreAIBackend"):
            raise RuntimeError(
                "Build the Python runtime with EXECUTORCH_BUILD_COREAI=ON"
            )
        flattened, _ = tree_flatten(inputs)
        with tempfile.TemporaryDirectory(prefix="coreai-suite-") as cache:
            program = _load_program_from_buffer(
                self.buffer,
                program_verification=Verification.Minimal,
                backend_options={"CoreAIBackend": {"coreai_cache_dir": cache}},
            )
            try:
                method = program.load_method("forward")
                try:
                    return method.call(tuple(flattened), clone_outputs=True)
                finally:
                    del method
            finally:
                del program


class CoreAITester(Tester):
    def __init__(self, module, example_inputs, dynamic_shapes=None, fp16=False):
        self.fp16 = fp16
        if fp16:
            module = copy.deepcopy(module).half()
            example_inputs = _fp16_inputs(example_inputs)
        stage_classes = self.default_stage_classes() | {
            StageType.QUANTIZE: CoreAIQuantize,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: CoreAILower,
            StageType.SERIALIZE: CoreAISerialize,
        }
        super().__init__(
            module,
            example_inputs,
            stage_classes=stage_classes,
            dynamic_shapes=dynamic_shapes,
        )

    def run_method_and_compare_outputs(self, stage=None, inputs=None, **kwargs):
        if self.fp16 and inputs is not None:
            inputs = _fp16_inputs(inputs)
        return super().run_method_and_compare_outputs(
            stage=stage, inputs=inputs, **kwargs
        )

    def _compare_outputs(
        self,
        reference_output,
        stage_output,
        quantization_scale=None,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
        statistics_callback=None,
    ):
        if not self.fp16:
            return Tester._compare_outputs(
                reference_output,
                stage_output,
                quantization_scale,
                atol,
                rtol,
                qtol,
                statistics_callback,
            )
        if isinstance(reference_output, torch.Tensor):
            reference_output = (reference_output,)
        elif isinstance(reference_output, OrderedDict):
            reference_output = tuple(reference_output.values())
        if isinstance(stage_output, torch.Tensor):
            stage_output = (stage_output,)
        assert len(stage_output) == len(reference_output)
        for i, (actual, reference) in enumerate(zip(stage_output, reference_output)):
            try:
                Tester._compare_outputs(
                    reference,
                    actual,
                    quantization_scale,
                    _fp16_atol(reference, atol),
                    rtol,
                    qtol,
                    statistics_callback,
                )
            except AssertionError as error:
                # Each output is compared alone, so the harness numbers it 0.
                raise AssertionError(f"FP16 output {i}: {error}") from None
