# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch
from executorch.backends.arm._passes import UnsqueezeBeforeRepeatPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

pipeline_input_t = Tuple[torch.Tensor, ...]
test_case_t = Tuple[
    pipeline_input_t,
    Dict[str, int],
    List[str],
]


class Repeat(torch.nn.Module):
    """Basic repeat model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(2, 2, 2, 2)

    test_data: Dict[str, test_case_t] = {
        "insert_view": (
            (torch.rand((2, 3, 4)),),
            {"aten_repeat_default": 3, "aten_view_copy_default": 4},
            [],
        ),
        "dont_insert_view": (
            (torch.rand((2, 3, 4, 1)),),
            {"aten_repeat_default": 3},
            ["aten_view_copy_default"],
        ),
    }


@common.parametrize("test_data", Repeat.test_data)
def test_unsqueeze_before_repeat_tosa_FP(test_data: test_case_t):
    """When rank(input) != number of repeated dimensions (=4 in Repeat module),
    insert view.
    """
    module = Repeat()
    data, ops_after_pass, ops_not_after_pass = test_data
    pipeline = PassPipeline[pipeline_input_t](
        module,
        data,
        quantize=False,
        ops_before_pass={"aten_repeat_default": 3},
        ops_not_before_pass=["aten_view_copy_default"],
        ops_after_pass=ops_after_pass,
        ops_not_after_pass=ops_not_after_pass,
        pass_list=[UnsqueezeBeforeRepeatPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()


class _RepeatBlock(torch.nn.Module):
    """Provide an actual named module boundary for exported FX provenance."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(2, 1, 1, 1)


class _NestedRepeat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.repeat_block = _RepeatBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.repeat_block(x)


def test_unsqueeze_before_repeat_preserves_provenance() -> None:
    """The new view retains repeat provenance, but gets its own
    shape/qparams.
    """
    model = _NestedRepeat().eval()
    inputs = (torch.randn(2, 3, 4),)
    ep = to_edge(
        torch.export.export(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    gm = ep.graph_module
    repeats = [
        n for n in gm.graph.nodes if n.target == exir_ops.edge.aten.repeat.default
    ]
    assert len(repeats) == 1
    repeat = repeats[0]

    original_stack = repeat.meta.get("nn_module_stack")
    assert original_stack, "Export must preserve Repeat module provenance"
    source_fn_stack = [("repeat_block", _RepeatBlock)]
    repeat.meta["source_fn_stack"] = source_fn_stack
    repeat.meta["stack_trace"] = "original repeat location"
    repeat.meta["debug_handle"] = 123456
    # These refer to Repeat's inputs/output, not to the new view. Inheriting
    # them would assign an incorrect quantization annotation to the view.
    repeat.meta["input_qparams"] = {0: "repeat-input-qparams"}
    repeat.meta["output_qparams"] = {0: "repeat-output-qparams"}

    result = UnsqueezeBeforeRepeatPass()(gm)
    assert result is not None
    assert result.modified
    repeats_after = [
        n
        for n in result.graph_module.graph.nodes
        if n.target == exir_ops.edge.aten.repeat.default
    ]
    assert len(repeats_after) == 1
    view = repeats_after[0].args[0]
    assert isinstance(view, torch.fx.Node)
    assert view.target == exir_ops.edge.aten.view_copy.default
    assert view.args[1] == [1, 2, 3, 4]

    assert view.meta.get("nn_module_stack") == original_stack
    assert view.meta.get("source_fn_stack") == source_fn_stack
    assert view.meta.get("debug_handle") == 123456
    assert "original repeat location" in view.meta.get("stack_trace", "")
    assert not view.meta.get("input_qparams")
    assert not view.meta.get("output_qparams")
    assert tuple(view.meta["val"].shape) == (1, 2, 3, 4)
    assert tuple(repeats_after[0].meta["val"].shape) == (2, 2, 3, 4)

    # Keep both semantics and idempotence unchanged.
    actual = result.graph_module(*inputs)
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    torch.testing.assert_close(actual, model(*inputs))
    repeated = UnsqueezeBeforeRepeatPass()(result.graph_module)
    assert repeated is not None
    assert not repeated.modified
