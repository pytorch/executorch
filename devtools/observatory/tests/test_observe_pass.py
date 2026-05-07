# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from executorch.devtools.observatory import Observatory, observe_pass


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _IdentityPass(PassBase):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return PassResult(graph_module, False)


def _make_gm() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(_ToyModel().eval())


# --- Name deduplication in Observatory.collect() ---


def test_collect_dedup_names() -> None:
    Observatory.clear()
    gm = _make_gm()
    with Observatory.enable_context():
        Observatory.collect("x", gm)
        Observatory.collect("x", gm)
        Observatory.collect("x", gm)
    names = Observatory.list_collected()
    assert "x" in names
    assert "x #2" in names
    assert "x #3" in names
    assert len(names) == 3
    Observatory.clear()


def test_collect_no_dedup_for_unique_names() -> None:
    Observatory.clear()
    gm = _make_gm()
    with Observatory.enable_context():
        Observatory.collect("a", gm)
        Observatory.collect("b", gm)
    names = Observatory.list_collected()
    assert names == ["a", "b"]
    Observatory.clear()


# --- observe_pass: instance wrapper ---


def test_observe_pass_instance_both() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass())
    with Observatory.enable_context():
        result = observed(gm)
    assert isinstance(result, PassResult)
    names = Observatory.list_collected()
    assert "_IdentityPass/input" in names
    assert "_IdentityPass/output" in names
    Observatory.clear()


def test_observe_pass_instance_output_only() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass(), collect_input=False)
    with Observatory.enable_context():
        observed(gm)
    names = Observatory.list_collected()
    assert "_IdentityPass" in names
    assert len(names) == 1
    Observatory.clear()


def test_observe_pass_instance_input_only() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass(), collect_output=False)
    with Observatory.enable_context():
        observed(gm)
    names = Observatory.list_collected()
    assert "_IdentityPass" in names
    assert len(names) == 1
    Observatory.clear()


# --- observe_pass: class decorator ---


def test_observe_pass_class_decorator() -> None:
    @observe_pass
    class _DecoratedPass(PassBase):
        def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
            return PassResult(graph_module, False)

    Observatory.clear()
    gm = _make_gm()
    p = _DecoratedPass()
    with Observatory.enable_context():
        result = p(gm)
    assert isinstance(result, PassResult)
    names = Observatory.list_collected()
    assert "_DecoratedPass/input" in names
    assert "_DecoratedPass/output" in names
    Observatory.clear()


def test_observe_pass_parameterized_class_decorator() -> None:
    @observe_pass(name="Custom", collect_input=False)
    class _ParamPass(PassBase):
        def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
            return PassResult(graph_module, False)

    Observatory.clear()
    gm = _make_gm()
    with Observatory.enable_context():
        _ParamPass()(gm)
    names = Observatory.list_collected()
    assert "Custom" in names
    assert len(names) == 1
    Observatory.clear()


# --- observe_pass: function wrapper ---


def test_observe_pass_function() -> None:
    def my_pass_fn(gm: torch.fx.GraphModule) -> PassResult:
        return PassResult(gm, False)

    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(my_pass_fn)
    with Observatory.enable_context():
        result = observed(gm)
    assert isinstance(result, PassResult)
    names = Observatory.list_collected()
    assert "my_pass_fn/input" in names
    assert "my_pass_fn/output" in names
    Observatory.clear()


# --- Deduplication on repeated calls ---


def test_observe_pass_repeated_calls_dedup() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass())
    with Observatory.enable_context():
        observed(gm)
        observed(gm)
    names = Observatory.list_collected()
    assert "_IdentityPass/input" in names
    assert "_IdentityPass/output" in names
    assert "_IdentityPass/input #2" in names
    assert "_IdentityPass/output #2" in names
    assert len(names) == 4
    Observatory.clear()


# --- No-op outside context ---


def test_observe_pass_noop_outside_context() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass())
    result = observed(gm)
    assert isinstance(result, PassResult)
    assert len(Observatory.list_collected()) == 0
    Observatory.clear()


# --- PassResult preservation ---


def test_observe_pass_preserves_return_value() -> None:
    class _ModifyingPass(PassBase):
        def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
            return PassResult(graph_module, True)

    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_ModifyingPass())
    with Observatory.enable_context():
        result = observed(gm)
    assert result.modified is True
    assert result.graph_module is gm
    Observatory.clear()


# --- Custom name ---


def test_observe_pass_custom_name() -> None:
    Observatory.clear()
    gm = _make_gm()
    observed = observe_pass(_IdentityPass(), name="MyStep")
    with Observatory.enable_context():
        observed(gm)
    names = Observatory.list_collected()
    assert "MyStep/input" in names
    assert "MyStep/output" in names
    Observatory.clear()
