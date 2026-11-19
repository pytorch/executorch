# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for fix of Arm PT2E quantizer with export guards."""

import pytest
import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.arm_quantizer import (
    _strip_exported_program_guards,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e


class _ConvMean(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


class _NoOpGuard(torch.nn.Module):
    def forward(self, *args) -> None:
        return None


def _guard_calls(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        n for n in gm.graph.nodes if n.op == "call_module" and n.target == "_guards_fn"
    ]


def _ensure_guard_call(gm: torch.fx.GraphModule) -> None:
    """Force the real exported-graph shape when torch suppresses guards.

    Some PyTorch versions suppress _guards_fn if the caller's path contains
    'executorch'. Thus a test *inside this repo* needs an explicit fallback.

    """
    if _guard_calls(gm):
        return
    gm.add_module("_guards_fn", _NoOpGuard())
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    assert placeholders
    with gm.graph.inserting_after(placeholders[-1]):
        gm.graph.call_module("_guards_fn", args=tuple(placeholders))
    gm.recompile()


@pytest.mark.parametrize("use_composable_quantizer", [True, False])
@pytest.mark.parametrize("guarded", [True, False])
def test_ethosu_prepare_pt2e_accepts_exported_module_guards(
    use_composable_quantizer: bool, guarded: bool
) -> None:
    inputs = (torch.randn(1, 3, 16, 16),)
    ep = torch.export.export(_ConvMean().eval(), inputs)
    gm = ep.module() if guarded else ep.module(check_guards=False)
    if guarded:
        _ensure_guard_call(gm)
        assert _guard_calls(gm)
    else:
        assert not _guard_calls(gm)

    reference = gm(*inputs)
    quantizer = EthosUQuantizer(
        EthosUCompileSpec("ethos-u85-256"),
        use_composable_quantizer=use_composable_quantizer,
    )
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    prepared = prepare_pt2e(gm, quantizer)
    assert not _guard_calls(prepared)
    assert not hasattr(prepared, "_guards_fn")
    torch.testing.assert_close(prepared(*inputs), reference, atol=1e-4, rtol=1e-4)


def test_only_export_guard_call_is_removed() -> None:
    gm = torch.export.export(_ConvMean().eval(), (torch.randn(1, 3, 16, 16),)).module(
        check_guards=False
    )
    _ensure_guard_call(gm)
    gm.add_module("other_module", _NoOpGuard())
    guard = _guard_calls(gm)[0]
    with gm.graph.inserting_after(guard):
        gm.graph.call_module("other_module", args=())
    gm.recompile()

    _strip_exported_program_guards(gm)
    assert not _guard_calls(gm)
    assert not hasattr(gm, "_guards_fn")
    assert any(
        n.op == "call_module" and n.target == "other_module" for n in gm.graph.nodes
    )


def test_guard_with_used_output_is_not_discarded() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    guarded_result = graph.call_module("_guards_fn", args=(x,))
    graph.output(guarded_result)

    root = torch.nn.Module()
    root.add_module("_guards_fn", torch.nn.Identity())
    gm = torch.fx.GraphModule(root, graph)
    with pytest.raises(RuntimeError, match="its result is in use"):
        _strip_exported_program_guards(gm)
    assert len(_guard_calls(gm)) == 1
