# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import operator
from collections.abc import Callable

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.tosa.dialect.ops.custom import (
    has_fake_tosa_impl,
    register_fake_tosa,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.fx.passes.infra.pass_base import PassResult
from torch.library import impl, Library, register_fake

_TEST_LIB: Library | None = None
_TEST_OPS_REGISTERED = False
_TEST_NAMESPACE = "arm_test_mylibrary"
_TEST_DOMAIN = "com.arm.test"


def _register_test_ops() -> None:
    global _TEST_LIB, _TEST_OPS_REGISTERED
    if _TEST_OPS_REGISTERED:
        return

    test_lib = torch.library.Library(_TEST_NAMESPACE, "DEF")
    _TEST_LIB = test_lib
    test_lib.define("test_op(Tensor x) -> Tensor")

    @impl(test_lib, "test_op", dispatch_key="CompositeExplicitAutograd")
    def _test_op_impl(x: torch.Tensor) -> torch.Tensor:
        return x + 7.0

    @register_fake(f"{_TEST_NAMESPACE}::test_op")
    def _test_op_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @register_fake_tosa("mylibrary.test_op")
    def _test_op_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        assert operator_name == "mylibrary.test_op"
        assert domain_name == _TEST_DOMAIN
        _ = implementation_attrs
        return [torch.empty_like(inputs[0])]

    @register_fake_tosa("mylibrary.add_replacement")
    def _add_replacement_tosa_fake_impl(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        assert operator_name == "mylibrary.add_replacement"
        assert domain_name == _TEST_DOMAIN
        _ = implementation_attrs
        return [torch.empty_like(inputs[0])]

    _TEST_OPS_REGISTERED = True


class _EncodeWrappedOpToTosaCustomPass(ArmPass):
    _passes_required_after = set()

    def __init__(
        self,
        operator_name: str,
        matcher: Callable[[object], bool],
    ) -> None:
        self._operator_name = operator_name
        self._matcher = matcher

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if not self._matcher(node.target):
                continue
            if not has_fake_tosa_impl(self._operator_name):
                raise RuntimeError(
                    f"tosa.CUSTOM fake impl is not registered for {self._operator_name}"
                )

            inputs = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
            payload = {
                "operator_name": self._operator_name,
                "binding_count": len(inputs),
            }
            impl_list = list(json.dumps(payload, sort_keys=True).encode("utf-8"))
            fake_outputs = [torch.empty_like(inputs[0].meta["val"])]

            with graph.inserting_before(node):
                tosa_custom = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=(inputs,),
                    kwargs={
                        "operator_name": self._operator_name,
                        "domain_name": _TEST_DOMAIN,
                        "implementation_attrs": impl_list,
                    },
                )
                tosa_custom.meta = dict(node.meta)
                tosa_custom.meta["val"] = fake_outputs

                output = graph.call_function(operator.getitem, args=(tosa_custom, 0))
                output.meta = dict(node.meta)
                output.meta["val"] = fake_outputs[0]

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


class _SingleCustomOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_mylibrary.test_op.default(x)


class _AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _AddAndMulModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x + y) * y


def _transform(module: torch.nn.Module, example_inputs: tuple, pass_: ArmPass):
    edge_model = to_edge(export(module, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        return edge_model.transform([pass_])


# Covers adding a brand new op and wrapping it as tosa.CUSTOM.
# Checks the rewrite emits the custom node plus the single-output getitem pattern.
def test_register_new_custom_op_rewrite_to_tosa_custom():
    _register_test_ops()
    transformed = _transform(
        _SingleCustomOpModule(),
        (torch.randn(2, 3),),
        _EncodeWrappedOpToTosaCustomPass(
            "mylibrary.test_op",
            lambda target: "arm_test_mylibrary" in str(target),
        ),
    )
    nodes = list(transformed.exported_program().graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    output_node = next(node for node in nodes if node.target == operator.getitem)

    assert custom_node.kwargs["operator_name"] == "mylibrary.test_op"
    assert custom_node.kwargs["domain_name"] == _TEST_DOMAIN
    assert output_node.args[0] == custom_node
    assert output_node.args[1] == 0


# Covers replacing an existing aten op instead of introducing a new one.
# Checks aten.add is removed and replaced by a tosa.CUSTOM node.
def test_replace_existing_aten_add_with_custom_op():
    _register_test_ops()
    transformed = _transform(
        _AddModule(),
        (torch.randn(2, 3), torch.randn(2, 3)),
        _EncodeWrappedOpToTosaCustomPass(
            "mylibrary.add_replacement",
            lambda target: target == exir_ops.edge.aten.add.Tensor,
        ),
    )
    nodes = list(transformed.exported_program().graph.nodes)

    assert not any(node.target == exir_ops.edge.aten.add.Tensor for node in nodes)
    assert any(node.target == exir_ops.backend.tosa.CUSTOM.default for node in nodes)


# Covers rewrite selectivity when the graph contains both target and non-target ops.
# Checks add is rewritten while unrelated ops like mul remain in the graph.
def test_rewrite_only_targets_intended_operator():
    _register_test_ops()
    transformed = _transform(
        _AddAndMulModule(),
        (torch.randn(2, 3), torch.randn(2, 3)),
        _EncodeWrappedOpToTosaCustomPass(
            "mylibrary.add_replacement",
            lambda target: target == exir_ops.edge.aten.add.Tensor,
        ),
    )
    nodes = list(transformed.exported_program().graph.nodes)

    assert not any(node.target == exir_ops.edge.aten.add.Tensor for node in nodes)
    assert any(node.target == exir_ops.edge.aten.mul.Tensor for node in nodes)


# Covers the failure path when no fake-TOSA implementation is registered.
# Checks the pass raises a clear error instead of producing a broken custom node.
def test_missing_fake_impl_fails_cleanly():
    _register_test_ops()
    with torch.no_grad():
        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
            exported = to_edge(export(_SingleCustomOpModule(), (torch.randn(2, 3),)))
            with pytest.raises(
                RuntimeError,
                match="tosa.CUSTOM fake impl is not registered for missing.test_op",
            ):
                _EncodeWrappedOpToTosaCustomPass(
                    "missing.test_op",
                    lambda target: "arm_test_mylibrary" in str(target),
                ).call(exported.exported_program().graph_module)


# Covers the current single-output custom-op convention.
# Checks tosa.CUSTOM keeps list-valued meta and getitem keeps the selected tensor meta.
def test_custom_op_rewrite_preserves_single_output_getitem_meta():
    _register_test_ops()
    transformed = _transform(
        _SingleCustomOpModule(),
        (torch.randn(2, 3),),
        _EncodeWrappedOpToTosaCustomPass(
            "mylibrary.test_op",
            lambda target: "arm_test_mylibrary" in str(target),
        ),
    )
    nodes = list(transformed.exported_program().graph.nodes)
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    output_node = next(node for node in nodes if node.target == operator.getitem)

    assert isinstance(custom_node.meta["val"], list)
    assert len(custom_node.meta["val"]) == 1
    assert tuple(output_node.meta["val"].shape) == tuple(
        custom_node.meta["val"][0].shape
    )
