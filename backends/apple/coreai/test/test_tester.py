# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import patch

import pytest
import torch
from executorch.backends.apple.coreai.test.tester import (
    _fp16_inputs,
    CoreAIQuantize,
    CoreAITester,
)
from executorch.backends.test.harness import Tester as BaseTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.lowered_backend_module import get_lowered_backend_modules


def test_fp16_inputs_preserve_integer_tensors():
    values = (torch.ones(2), torch.ones(2, dtype=torch.int64))
    converted = _fp16_inputs(values)
    assert converted[0].dtype == torch.float16
    assert converted[1] is values[1]
    assert values[0].dtype == torch.float32


def test_fp16_tester_does_not_mutate_model_or_inputs():
    model = torch.nn.Linear(32, 32).eval()
    inputs = (torch.randn(1, 32),)
    tester = CoreAITester(model, inputs, fp16=True)
    assert model.weight.dtype == torch.float32
    assert inputs[0].dtype == torch.float32
    assert tester.example_inputs[0].dtype == torch.float16
    assert tester.original_module.weight.dtype == torch.float16
    assert next(tester.generate_random_inputs())[0].dtype == torch.float16


def test_fp16_explicit_comparison_inputs():
    tester = CoreAITester(torch.nn.Identity(), (torch.ones(2),), fp16=True)
    with patch.object(BaseTester, "run_method_and_compare_outputs") as compare:
        tester.run_method_and_compare_outputs(inputs=(torch.ones(2),))
    assert compare.call_args.kwargs["inputs"][0].dtype == torch.float16


def test_serialized_execution_uses_private_cache_and_cloned_outputs():
    from pathlib import Path
    from types import SimpleNamespace

    from executorch.backends.apple.coreai.test.tester import CoreAISerialize

    roots = []
    output = [torch.ones(2)]

    def load(buffer, *, program_verification, backend_options):
        root = backend_options["CoreAIBackend"]["coreai_assets_dir"]
        assert Path(root).is_dir()
        roots.append(root)

        def call(inputs, *, clone_outputs):
            assert clone_outputs
            return output

        method = SimpleNamespace(call=call)
        return SimpleNamespace(load_method=lambda name: method)

    runtime = SimpleNamespace(
        _is_available=lambda name: name == "CoreAIBackend",
        _load_program_from_buffer=load,
        Verification=SimpleNamespace(Minimal=0),
    )
    stage = CoreAISerialize()
    stage.buffer = b"test"
    with patch.dict(
        "sys.modules", {"executorch.extension.pybindings.portable_lib": runtime}
    ):
        assert stage.run_artifact((torch.ones(2),)) is output
        assert stage.run_artifact((torch.ones(2),)) is output
    assert len(set(roots)) == 2
    assert all(not Path(root).exists() for root in roots)


def test_quantization_modes_are_explicit():
    for mode in ("w8a8", "w8", "w4"):
        assert CoreAIQuantize(mode).mode == mode
    with pytest.raises(ValueError, match="Unsupported"):
        CoreAIQuantize("unknown")


def test_all_coreai_flows_are_registered():
    from executorch.backends.test.suite.flows.coreai import COREAI_TEST_FLOWS

    assert {flow.name for flow in COREAI_TEST_FLOWS} == {
        "coreai_fp32",
        "coreai_fp16",
        "coreai_w8a8",
        "coreai_w8",
        "coreai_w4",
    }
    assert sum(flow.quantize for flow in COREAI_TEST_FLOWS) == 3


@pytest.mark.skipif(
    os.environ.get("COREAI_TEST_RUNTIME") != "1",
    reason="Set COREAI_TEST_RUNTIME=1 with a CoreAI-enabled Python runtime",
)
@pytest.mark.parametrize("mode", ["fp32", "fp16", "w8a8", "w8", "w4"])
def test_weighted_model_runs_on_coreai(mode):
    from executorch.extension.pybindings.portable_lib import _is_available

    assert _is_available("CoreAIBackend")
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32)
    ).eval()
    inputs = (torch.randn(1, 32),)
    tester = CoreAITester(model, inputs, fp16=mode == "fp16")
    if mode.startswith("w"):
        tester.quantize(CoreAIQuantize(mode))
        graph = tester.get_artifact(StageType.QUANTIZE)
        ops = [
            getattr(node.target, "__name__", str(node.target))
            for node in graph.graph.nodes
            if node.op == "call_function"
        ]
        assert any("constexpr_blockwise_shift_scale" in op for op in ops), ops
        if mode == "w8a8":
            assert "quantize" in ops and "dequantize" in ops, ops
        else:
            assert "quantize" not in ops and "dequantize" not in ops, ops
    tester.export().to_edge_transform_and_lower()
    graph = tester.get_artifact().exported_program().graph_module
    delegates = get_lowered_backend_modules(graph)
    assert delegates and all(
        module.backend_id == "CoreAIBackend" for module in delegates
    )
    tester.to_executorch().serialize().run_method_and_compare_outputs(
        inputs=inputs, atol=1e-1, rtol=4e-2
    )
