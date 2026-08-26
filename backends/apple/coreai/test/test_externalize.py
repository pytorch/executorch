# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Externalized composites through the full ExecuTorch lowering path.

Externalization itself is coreai-torch's; these cover the parts ExecuTorch
adds. Composite behavior is not re-tested here, only that the prepared
submodules reach the delegate.
"""

import contextlib
import gc
import re
import unittest

import executorch.backends.apple.coreai.compiler.preprocess as preprocess_module

import torch
import torch.nn as nn
from coreai_torch import ExternalizeSpec, get_decomp_table
from coreai_torch.composite_ops import GatherMM, RMSNorm, SDPA

from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.externalize import (
    default_specs,
    externalize_modules,
    is_externalize_target,
    is_supported_target,
    lookup,
    register,
    spec_for,
)
from executorch.backends.apple.coreai.externalize.specs import _TARGETS, target_class
from executorch.backends.apple.coreai.partition.partitioner import (
    _OperatorsSupportedForCoreAIBackend,
)
from executorch.exir import to_edge_transform_and_lower


class TwoNorms(nn.Module):
    """Two RMSNorm instances with different eps, so per-call-site attrs matter."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.n0 = RMSNorm(dim)
        self.lin = nn.Linear(dim, dim)
        self.n1 = RMSNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.n1(self.lin(self.n0(x)))


class NormAndAttention(nn.Module):
    """Two different composites, so op names must resolve to distinct entries."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = SDPA()

    def forward(self, q, k, v):
        return self.attn(self.norm(q), k, v)


def _model():
    torch.manual_seed(0)
    model = TwoNorms().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))
    return model, (torch.randn(2, 8),)


def _mixed_model():
    torch.manual_seed(0)
    model = NormAndAttention().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))
    shape = (1, 2, 4, 8)
    return model, tuple(torch.randn(shape) for _ in range(3))


def _externalize(model, sample, specs=("rms_norm",)):
    return externalize_modules(
        model,
        [spec_for(name) for name in specs],
        export_fn=lambda m: torch.export.export(m, sample).run_decompositions(
            get_decomp_table()
        ),
    )


def _lower(ep, externalized):
    return to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[CoreAIPartitioner(externalized_modules=externalized)],
        compile_config=get_default_compile_config(),
    )


class ExternalizedLoweringTest(unittest.TestCase):
    def test_externalized_ops_land_inside_the_delegate(self) -> None:
        model, sample = _model()
        ep, externalized = _externalize(model, sample)
        self.assertEqual(len(externalized), 2)

        lowered = _lower(ep, externalized)
        graph_module = lowered.exported_program().graph_module
        leftover = [
            node.name
            for node in graph_module.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(leftover, [], "externalized op left outside the delegate")

    def test_produces_an_executable_program(self) -> None:
        model, sample = _model()
        ep, externalized = _externalize(model, sample)
        executorch_program = _lower(ep, externalized).to_executorch()
        self.assertGreater(len(executorch_program.buffer), 0)

    def test_emits_no_externalize_compile_spec(self) -> None:
        """The prepared submodules are build-time state, not runtime state.

        Serializing a handle to them would also make the .pte non-reproducible.
        """
        model, sample = _model()
        _, externalized = _externalize(model, sample)
        keys = {
            spec.key
            for spec in CoreAIPartitioner(
                externalized_modules=externalized
            ).delegation_spec.compile_specs
        }
        self.assertEqual(keys, set())

    def test_scales_remain_delegate_inputs(self) -> None:
        """The composite takes its scale as an argument, so it stays visible."""
        model, sample = _model()
        ep, _ = _externalize(model, sample)
        placeholders = {
            node.name for node in ep.graph.nodes if node.op == "placeholder"
        }
        self.assertTrue({"p_n0_weight", "p_n1_weight"} <= placeholders)


class PartitionerSupportTest(unittest.TestCase):
    def test_claims_only_prepared_ops(self) -> None:
        model, sample = _model()
        ep, externalized = _externalize(model, sample)
        targets = [
            node.target
            for node in ep.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertTrue(targets)
        for target in targets:
            self.assertTrue(is_supported_target(target, externalized))
            self.assertFalse(
                is_supported_target(target, []),
                "an op with no prepared submodule must not be claimed",
            )

    def test_ignores_ordinary_aten_targets(self) -> None:
        self.assertFalse(is_supported_target(torch.ops.aten.addmm.default, []))


class DefaultSpecsTest(unittest.TestCase):
    def test_covers_the_whole_composite_library(self) -> None:
        """Derived from the installed library, not a list that can go stale.

        ``_TARGETS`` is hand-written, so a composite added upstream is silently
        absent from ``default_specs``. ``RMSNorm`` is the only deliberate
        omission: it owns its scale, and ``RMSNormImpl`` is the leaf that takes
        it as an argument.
        """
        from coreai_torch import composite_ops

        library = set(composite_ops.__all__) - {"RMSNorm"}
        covered = {_TARGETS[spec.composite_op_name] for spec in default_specs()}
        self.assertEqual(
            covered,
            library,
            f"default_specs covers {sorted(covered)} but the installed library "
            f"exposes {sorted(library)}",
        )

    def test_every_target_class_resolves(self) -> None:
        """A renamed upstream class must fail here, not at export time."""
        for name in default_specs():
            with self.subTest(name.composite_op_name):
                self.assertTrue(isinstance(target_class(name.composite_op_name), type))

    def test_attrs_are_read_from_the_installed_classes(self) -> None:
        """Transcribing them would let the specs drift from the SDK."""
        expected = {
            "rms_norm": {"axes", "eps"},
            "rope": {"scale", "base", "dims", "interleaved"},
            "scaled_dot_product_attention": {"scale", "is_causal", "window_size"},
            "gather_mm": {"num_batch_axes"},
            "gated_delta_update": {"use_qk_l2_norm"},
        }
        for spec in default_specs():
            with self.subTest(spec.composite_op_name):
                self.assertEqual(
                    set(spec.composite_attrs), expected[spec.composite_op_name]
                )


class MixedCompositeTest(unittest.TestCase):
    """Distinct composites in one model must resolve to distinct submodules."""

    def setUp(self) -> None:
        self.model, self.sample = _mixed_model()
        self.ep, self.externalized = _externalize(
            self.model, self.sample, specs=("rms_norm", "scaled_dot_product_attention")
        )

    def test_each_composite_is_prepared_separately(self) -> None:
        self.assertEqual(len(self.externalized), 2)
        self.assertEqual(
            {module.composite_op_name for module in self.externalized},
            {"rms_norm", "scaled_dot_product_attention"},
        )
        # Op names are what the registry keys on, so they must not collide.
        self.assertEqual(len({m.op_name for m in self.externalized}), 2)

    def test_lookup_returns_the_matching_submodule_per_op(self) -> None:
        register(self.externalized)
        by_op = {module.op_name: module for module in self.externalized}
        for op_name, expected in by_op.items():
            with self.subTest(op_name):
                self.assertIs(lookup([op_name])[0], expected)

    def test_both_land_inside_the_delegate(self) -> None:
        lowered = _lower(self.ep, self.externalized)
        leftover = [
            node.name
            for node in lowered.exported_program().graph_module.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(leftover, [])

    def test_partitioner_claims_both(self) -> None:
        targets = [
            node.target
            for node in self.ep.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertTrue(is_supported_target(target, self.externalized))

    def test_claiming_requires_the_matching_submodule(self) -> None:
        """A partial list must not let the other composite into the delegate."""
        rms_only = [
            module
            for module in self.externalized
            if module.composite_op_name == "rms_norm"
        ]
        claimed = [
            is_supported_target(node.target, rms_only)
            for node in self.ep.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(sorted(claimed), [False, True])


class SharedInstance(nn.Module):
    """One composite instance invoked twice, as a block reuses a shared RoPE.

    Both call sites share an ``op_name`` (it names the submodule) and differ
    only in ``name``, so anything keyed on ``op_name`` alone collapses them.
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.norm(self.lin(self.norm(x)))


def _shared_instance_model():
    torch.manual_seed(0)
    model = SharedInstance().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))
    return model, (torch.randn(2, 8),)


class SharedInstanceTest(unittest.TestCase):
    """A submodule used at several call sites must lower at every one."""

    def setUp(self) -> None:
        model, sample = _shared_instance_model()
        self.ep, self.externalized = _externalize(model, sample)

    def test_one_prepared_module_per_call_site(self) -> None:
        self.assertEqual(len(self.externalized), 2)
        # op_name names the submodule, name names the call site.
        self.assertEqual(len({m.op_name for m in self.externalized}), 1)
        self.assertEqual(len({m.name for m in self.externalized}), 2)

    def test_registry_keeps_every_call_site(self) -> None:
        register(self.externalized)
        op_name = self.externalized[0].op_name
        found = lookup([op_name])
        self.assertEqual(len(found), 2, "a call site was dropped by the registry")
        self.assertEqual({id(m) for m in found}, {id(m) for m in self.externalized})

    def test_lowers_with_both_call_sites_delegated(self) -> None:
        lowered = _lower(self.ep, self.externalized)
        leftover = [
            node.name
            for node in lowered.exported_program().graph_module.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(leftover, [])
        self.assertGreater(len(lowered.to_executorch().buffer), 0)

    def test_preprocess_collects_each_module_once(self) -> None:
        """One module per call site, not one per node naming the same op.

        ``_externalized_modules`` reads an op name off every matching node, so
        N call sites of one submodule yield N copies of the same name. The
        converter keys its resolved nodes on ``id(ext)``, so duplicates collide
        and all but one call site loses its lowering. Fixing the registry alone
        would not catch this.
        """
        from executorch.backends.apple.coreai.compiler.preprocess import (
            _externalized_modules,
        )

        register(self.externalized)
        collected = _externalized_modules(self.ep)
        self.assertIsNotNone(collected)
        ids = [id(m) for m in collected]
        self.assertEqual(len(ids), len(set(ids)), "the same module was collected twice")
        self.assertEqual(
            set(ids),
            {id(m) for m in self.externalized},
            "every call site's module must be collected exactly once",
        )


class InteriorIndices(nn.Module):
    """Indices computed in-graph, as a gather after an argmax does.

    The narrowing pass only casts at the exported-program boundary, so an
    int64 produced here stays int64 all the way to the delegate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gather = GatherMM()

    def forward(self, lhs, rhs, scores):
        return self.gather(lhs, rhs, lhs_indices=scores.argmax(dim=-1))


class ExternalizedBoundaryDtypeTest(unittest.TestCase):
    """64-bit operands are cast at the boundary, not left outside the delegate.

    Core AI narrows 64-bit dtypes, and an externalized op cannot be lowered
    anywhere else, so the pass inserts an explicit cast rather than the
    partitioner declining the op.
    """

    def _externalized(self):
        torch.manual_seed(0)
        model = InteriorIndices().eval()
        sample = (torch.randn(4, 8, 8), torch.randn(4, 8, 8), torch.randn(4, 4))
        return _externalize(model, sample, specs=("gather_mm",))

    def test_interior_int64_operand_is_present_before_lowering(self):
        ep, _ = self._externalized()
        node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and is_externalize_target(n.target)
        )
        operands = [
            a.meta["val"]
            for a in node.args
            if hasattr(a, "meta") and isinstance(a.meta.get("val"), torch.Tensor)
        ]
        self.assertTrue(
            any(v.dtype == torch.int64 for v in operands),
            "fixture no longer produces an int64 operand",
        )

    def test_op_is_still_claimed(self):
        """Only coreai can lower it, so it must go in whatever its dtypes."""
        ep, externalized = self._externalized()
        node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and is_externalize_target(n.target)
        )
        support = _OperatorsSupportedForCoreAIBackend(externalized=externalized)
        self.assertTrue(support.is_node_supported({}, node))

    def test_no_64bit_crosses_the_delegate_boundary(self):
        ep, externalized = self._externalized()
        seen = {}
        original = preprocess_module._prepare_program_for_conversion

        def spy(edge_program):
            seen["dtypes"] = [
                node.meta["val"].dtype
                for node in edge_program.graph.nodes
                if node.op == "placeholder" and hasattr(node.meta.get("val"), "dtype")
            ]
            return original(edge_program)

        preprocess_module._prepare_program_for_conversion = spy
        try:
            lowered = _lower(ep, externalized)
            self.assertGreater(len(lowered.to_executorch().buffer), 0)
        finally:
            preprocess_module._prepare_program_for_conversion = original

        self.assertTrue(seen.get("dtypes"), "preprocess never ran")
        self.assertNotIn(torch.int64, seen["dtypes"])
        self.assertIn(torch.int32, seen["dtypes"], "the cast should be visible")


class UnpreparedExternalizedOpTest(unittest.TestCase):
    def test_missing_submodule_fails_at_partition(self):
        """Better than the "Missing out variants" error it would hit later."""
        model, sample = _model()
        ep, _ = _externalize(model, sample)
        with self.assertRaisesRegex(RuntimeError, "no prepared submodule"):
            _lower(ep, [])


class CustomLeaf(nn.Module):
    """A user-defined composite, following the library's leaf convention.

    Tensors it needs arrive as forward arguments, so they stay visible to
    ExecuTorch instead of being captured in the op closure.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * scale


class CustomLeafWrapper(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.impl = CustomLeaf()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(self.impl(x, self.weight))


def _externalize_custom(model, target_class, composite_op_name):
    sample = (torch.randn(2, 8),)
    spec = ExternalizeSpec(
        target_class=target_class,
        composite_op_name=composite_op_name,
        composite_attrs=["eps"],
    )
    ep, externalized = externalize_modules(
        model,
        [spec],
        export_fn=lambda m: torch.export.export(m, sample).run_decompositions(
            get_decomp_table()
        ),
    )
    return ep, externalized


@contextlib.contextmanager
def _capture_asset_ir():
    """Capture the Core AI IR ``preprocess`` produces for each delegate.

    The asset is built and consumed inside ``preprocess``, so the only way to
    assert on it from an end-to-end lowering is to observe it in flight.
    """
    captured = []
    original = preprocess_module._convert_to_aiprogram

    def spy(edge_program):
        program = original(edge_program)
        captured.append(str(program._mlir_module))
        return program

    preprocess_module._convert_to_aiprogram = spy
    try:
        yield captured
    finally:
        preprocess_module._convert_to_aiprogram = original


class CustomExternalizationTest(unittest.TestCase):
    """Externalization is not limited to coreai_torch.composite_ops.

    Nothing in the ExecuTorch path consults the composite library: the
    partitioner, registry and preprocess all key on the op name coreai-torch
    derives from the module path. No upstream test covers a user-defined
    composite, so the emitted IR is checked here rather than assumed.
    """

    def test_user_defined_leaf_reaches_the_delegate(self) -> None:
        torch.manual_seed(0)
        ep, externalized = _externalize_custom(
            CustomLeafWrapper().eval(), CustomLeaf, "my_leaf"
        )
        self.assertEqual(len(externalized), 1)
        self.assertEqual(externalized[0].composite_op_name, "my_leaf")

        lowered = _lower(ep, externalized)
        leftover = [
            node.name
            for node in lowered.exported_program().graph_module.graph.nodes
            if node.op == "call_function" and is_externalize_target(node.target)
        ]
        self.assertEqual(leftover, [])
        self.assertGreater(len(lowered.to_executorch().buffer), 0)

    def test_user_defined_composite_is_emitted_in_the_asset(self) -> None:
        """Landing in the delegate does not prove a composite was emitted."""
        torch.manual_seed(0)
        ep, externalized = _externalize_custom(
            CustomLeafWrapper().eval(), CustomLeaf, "my_emitted"
        )

        with _capture_asset_ir() as captured:
            _lower(ep, externalized)

        self.assertEqual(len(captured), 1, "expected exactly one delegate")
        ir = captured[0]
        self.assertIn('composite_declaration<"my_emitted"', ir)
        self.assertIn("noinline", ir)
        self.assertEqual(ir.count("coreai.invoke"), 1)

    def test_composite_declares_the_forward_arguments(self) -> None:
        """The declaration names come from the submodule's own signature."""
        torch.manual_seed(0)
        ep, externalized = _externalize_custom(
            CustomLeafWrapper().eval(), CustomLeaf, "my_declared"
        )

        with _capture_asset_ir() as captured:
            _lower(ep, externalized)

        declared = re.search(r"input_names = \[([^\]]*)\]", captured[0])
        self.assertIsNotNone(declared, "composite declared no inputs")
        self.assertEqual(declared.group(1), '"x", "scale"')
        self.assertIn("eps = ", captured[0], "composite attrs should be declared")


class NestedLeaf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class NestedParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.leaf = NestedLeaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaf(x) + 1.0


class NestedWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parent = NestedParent()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.parent(x)


class NestedExternalizationTest(unittest.TestCase):
    """An externalized submodule may itself contain an externalized one.

    The inner call site is inside the outer submodule's own program rather
    than the partitioned graph, so a lookup that only reads the graph it is
    converting will not find it.
    """

    def _externalize_nested(self):
        sample = (torch.randn(4, 8),)
        return externalize_modules(
            NestedWrapper().eval(),
            [
                ExternalizeSpec(target_class=NestedParent),
                ExternalizeSpec(target_class=NestedLeaf),
            ],
            export_fn=lambda m: torch.export.export(m, sample).run_decompositions(
                get_decomp_table()
            ),
        )

    def test_both_levels_are_prepared(self) -> None:
        _, externalized = self._externalize_nested()
        self.assertEqual(len(externalized), 2)

    def test_nested_externalization_lowers(self) -> None:
        ep, externalized = self._externalize_nested()
        _lower(ep, externalized).to_executorch()


class RegistryLifetimeTest(unittest.TestCase):
    """Prepared submodules must not outlive the caller's reference to them."""

    def test_entries_are_dropped_when_the_caller_lets_go(self) -> None:
        model, sample = _model()
        _, externalized = _externalize(model, sample)
        op_names = [module.op_name for module in externalized]

        partitioner = CoreAIPartitioner(externalized_modules=externalized)
        self.assertEqual(len(lookup(op_names)), len(op_names))

        del externalized, partitioner
        gc.collect()

        with self.assertRaisesRegex(KeyError, "no prepared submodule"):
            lookup(op_names)

    def test_entries_survive_while_the_partitioner_holds_them(self) -> None:
        """The partitioner keeps them alive for the duration of lowering."""
        model, sample = _model()
        _, externalized = _externalize(model, sample)
        op_names = [module.op_name for module in externalized]

        partitioner = CoreAIPartitioner(externalized_modules=externalized)
        del externalized
        gc.collect()

        self.assertEqual(len(lookup(op_names)), len(op_names))
        del partitioner


if __name__ == "__main__":
    unittest.main()
