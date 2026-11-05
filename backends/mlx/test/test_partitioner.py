#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX partitioner.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.backends.mlx.test.test_utils import get_mlx_node_counts
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.program import get_fake_program
from executorch.runtime import Runtime
from torch.export import export
from torch.export.graph_signature import InputKind, InputSpec, TensorArgument


class TestMLXPartitionerRejectsToEdge(unittest.TestCase):
    """MLXPartitioner must only be used via to_edge_transform_and_lower."""

    def test_to_edge_then_to_backend_raises(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        ep = export(M(), (torch.randn(4),), strict=False)
        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

        with self.assertRaises(RuntimeError) as ctx:
            edge.to_backend(MLXPartitioner())

        self.assertIn("to_edge_transform_and_lower", str(ctx.exception))


def _lower(model, inputs, transform_passes=None):
    return to_edge_transform_and_lower(
        export(model, inputs, strict=False),
        transform_passes=transform_passes,
        partitioner=[MLXPartitioner()],
    ).to_executorch()


def _delegate_count(program) -> int:
    return sum(
        1
        for node in program.exported_program().graph_module.graph.nodes
        if node.op == "call_function" and "executorch_call_delegate" in str(node.target)
    )


def _cpu_ops(program) -> set:
    """The ATen ops left outside every delegate, which run on the CPU."""
    return {
        str(node.target)
        for node in program.exported_program().graph_module.graph.nodes
        if node.op == "call_function" and str(node.target).startswith("aten.")
    }


def _run(model, inputs, transform_passes=None):
    """Lower, execute, and return the node counts, the delegate count and the error.

    The delegate count is returned so a test can tell "decomposed onto this backend"
    apart from "not lowered here at all", which a node count alone cannot show.
    """
    with torch.no_grad():
        ref = model(*inputs)
    program = _lower(model, inputs, transform_passes)
    delegates = _delegate_count(program)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pte"
        path.write_bytes(program.buffer)
        counts = get_mlx_node_counts(path)
        method = Runtime.get().load_program(path).load_method("forward")
        out = method.execute(list(inputs))[0]
    assert out.shape == ref.shape, (out.shape, ref.shape)
    diff = (out - ref).abs()
    return counts, delegates, diff.max().item() if diff.numel() else 0.0


class TestMLXPartitionerConstantInputs(unittest.TestCase):
    class OptionalInput(nn.Module):
        def forward(self, x, optional, y):
            return x + y

    def test_unused_constants_preserve_signature(self):
        for value in (None, True, 3, 1.5, "unused"):
            with self.subTest(value=value):
                ep = export(
                    self.OptionalInput(),
                    (torch.randn(4), value, torch.randn(4)),
                    strict=False,
                )
                signature = str(ep.graph_signature)
                placeholder = next(n for n in ep.graph.nodes if n.name == "optional")
                self.assertFalse(placeholder.users)

                builder = MLXProgramBuilder(ep)
                builder.check_support_only()
                self.assertTrue(
                    all(info.supported for info in builder.node_info.values())
                )
                built = MLXProgramBuilder(ep).build()
                self.assertEqual(built.num_input_tensors, 2)
                self.assertEqual(len(built.input_map), 2)
                self.assertEqual(str(ep.graph_signature), signature)

    def test_unused_constants_lower_and_execute(self):
        # The Python runtime does not accept string inputs; cover those above.
        for value in (None, True, 3, 1.5):
            with self.subTest(value=value):
                _, delegates, error = _run(
                    self.OptionalInput(),
                    (torch.randn(4), value, torch.randn(4)),
                )
                self.assertGreater(delegates, 0)
                self.assertLess(error, 1e-6)

    def test_string_constant_lowers(self):
        program = _lower(
            self.OptionalInput(),
            (torch.randn(4), "unused", torch.randn(4)),
        )
        self.assertGreater(_delegate_count(program), 0)

    def test_consumed_constant_still_raises(self):
        ep = export(
            self.OptionalInput(),
            (torch.randn(4), 3, torch.randn(4)),
            strict=False,
        )
        placeholder = next(n for n in ep.graph.nodes if n.name == "optional")
        add = next(n for n in ep.graph.nodes if n.op == "call_function")
        add.target = torch.ops.aten.add.Scalar
        add.args = (add.args[0], placeholder)
        ep.graph_module.recompile()
        ep.validate()

        for check_only in (True, False):
            with self.subTest(check_only=check_only):
                builder = MLXProgramBuilder(ep)
                with self.assertRaisesRegex(NotImplementedError, "ConstantArgument"):
                    if check_only:
                        builder.check_support_only()
                    else:
                        builder.build()


class TestMLXPartitionerUnsupportedDtypes(unittest.TestCase):
    def test_float64_tensor_list_stays_off_the_delegate(self):
        class Float64Cat(nn.Module):
            def forward(self, x, y):
                return x + torch.cat([y.sin(), y.cos()]).float()

        _, delegates, err = _run(
            Float64Cat().eval(),
            (torch.randn(8), torch.randn(4, dtype=torch.float64)),
        )
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-6)

    def test_int64_scatter_add_stays_off_the_delegate(self):
        class ScatterAdd(nn.Module):
            def forward(self, base, index, src):
                return base.scatter_add(0, index, src) + 1

        index = torch.tensor([0, 2, 2, 7])
        for dtype, scatters in ((torch.float32, 1), (torch.int64, 0)):
            with self.subTest(dtype=dtype):
                counts, delegates, err = _run(
                    ScatterAdd(),
                    (torch.zeros(8, dtype=dtype), index, torch.ones(4, dtype=dtype)),
                )
                self.assertEqual(counts.get("ScatterAddNode", 0), scatters)
                self.assertGreater(delegates, 0)
                self.assertEqual(err, 0)


class MoeDispatch(nn.Module):
    """The static dispatch of a mixture-of-experts block.

    Routing weights come from gather, an int64 histogram counts tokens per
    expert, tokens are scattered into expert slots, and index_add_ (which
    becomes index_put with accumulate) brings the results back.
    """

    def __init__(self, experts=4, hidden=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(hidden, experts, bias=False)
        self.weight = nn.Parameter(torch.randn(experts, hidden, hidden))

    def forward(self, x):
        tokens, hidden = x.shape
        experts = self.weight.shape[0]
        scores = self.router(x).softmax(-1)
        _, selected = torch.topk(scores, self.top_k, dim=-1)
        weights = scores.gather(1, selected)
        expert = selected.reshape(-1)
        order = torch.argsort(expert)
        token = torch.arange(tokens).repeat_interleave(self.top_k)[order]
        expert = expert[order]
        counts = torch.zeros(experts, dtype=torch.long).scatter_add(
            0, expert, torch.ones_like(expert)
        )
        slot = torch.arange(tokens * self.top_k) - (counts.cumsum(0) - counts)[expert]
        slots = torch.zeros(experts, tokens, hidden)
        slots[expert, slot] = x[token]
        y = torch.bmm(slots, self.weight)[expert, slot]
        out = torch.zeros(tokens, hidden)
        out.index_add_(0, token, weights.reshape(-1, 1)[order] * y)
        return x + out


class TestMLXPartitionerIndexPutAndGather(unittest.TestCase):
    def test_moe_dispatch_is_lowered(self):
        model, inputs = MoeDispatch().eval(), (torch.randn(6, 8),)
        program = _lower(model, inputs)
        # Only the int64 histogram stays on the CPU, between two delegates.
        self.assertEqual(_cpu_ops(program), {"aten.scatter_add.out"})
        self.assertEqual(_delegate_count(program), 2)
        method = Runtime.get().load_program(program.buffer).load_method("forward")
        err = (method.execute(list(inputs))[0] - model(*inputs)).abs().max()
        self.assertLess(err.item(), 1e-5)

    def test_cache_write_into_a_buffer(self):
        class Cache(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(2, 8, 4))

            def forward(self, pos, v):
                self.k[:, pos] = v
                return self.k * 1

        program = _lower(Cache().eval(), (torch.tensor([0]), torch.randn(2, 1, 4)))
        self.assertEqual(_cpu_ops(program), set())
        method = Runtime.get().load_program(program.buffer).load_method("forward")
        reference = Cache()
        # Each call must see the writes of the calls before it.
        for pos in (0, 3, 5):
            inputs = (torch.tensor([pos]), torch.randn(2, 1, 4))
            out = method.execute(list(inputs))[0]
            self.assertTrue(torch.equal(out, reference(*inputs)))
        # Eager raises on a position outside the cache; the delegate clamps it
        # to the nearest end, so the write stays inside the buffer.
        for pos, clamped in ((8, 7), (-9, 0)):
            v = torch.randn(2, 1, 4)
            out = method.execute([torch.tensor([pos]), v])[0]
            self.assertTrue(torch.equal(out, reference(torch.tensor([clamped]), v)))

    def test_gather_clamps_out_of_range_indices(self):
        class Gather(nn.Module):
            def forward(self, x, index):
                return torch.gather(x, 1, index)

        x = torch.arange(12.0).reshape(3, 4)
        program = _lower(Gather(), (x, torch.zeros(3, 2, dtype=torch.long)))
        method = Runtime.get().load_program(program.buffer).load_method("forward")
        # Eager raises on 9, -1 and -7. The delegate counts -1 from the end and
        # clamps 9 and -7 to the nearest end instead of reading outside x.
        out = method.execute([x, torch.tensor([[9, -1], [-7, 1], [3, 2]])])[0]
        clamped = torch.tensor([[3, 3], [0, 1], [3, 2]])
        self.assertTrue(torch.equal(out, torch.gather(x, 1, clamped)))

    def test_index_add_clamps_out_of_range_indices(self):
        class IndexAdd(nn.Module):
            def forward(self, x, index, values):
                return x.index_add(0, index, values)

        x, values = torch.randn(8, 4), torch.randn(3, 4)
        program = _lower(IndexAdd(), (x, torch.tensor([0, 1, 2]), values))
        self.assertEqual(_cpu_ops(program), set())
        method = Runtime.get().load_program(program.buffer).load_method("forward")
        # Eager raises on 9, -1 and -9. The delegate counts -1 from the end,
        # clamps 9 and -9 to the nearest end, and adds both rows aimed at row 7.
        out = method.execute([x, torch.tensor([9, -1, -9]), values])[0]
        expected = IndexAdd()(x, torch.tensor([7, 7, 0]), values)
        self.assertTrue(torch.allclose(out, expected))

    def test_empty_axis(self):
        # The delegate cannot return an empty tensor yet, so both models sum.
        class IndexPut(nn.Module):
            def forward(self, x, index, values):
                return torch.ops.aten.index_put.default(x, [index], values).sum(0)

        class Gather(nn.Module):
            def forward(self, x, index):
                return torch.gather(x, 1, index).sum(1)

        def run(model, *inputs):
            program = _lower(model, inputs)
            self.assertEqual(_cpu_ops(program), set())
            method = Runtime.get().load_program(program.buffer).load_method("forward")
            return method.execute(list(inputs))[0]

        # Writing or reading nothing on an empty axis works, as in eager.
        empty, no_index = torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
        out = run(IndexPut(), empty, no_index, empty)
        self.assertTrue(torch.equal(out, torch.zeros(4)))
        out = run(Gather(), torch.zeros(3, 0), torch.zeros(3, 0, dtype=torch.long))
        self.assertTrue(torch.equal(out, torch.zeros(3)))
        # Eager raises on a non-empty write, but there is nowhere to write,
        # so the delegate returns the input.
        out = run(IndexPut(), empty, torch.tensor([0]), torch.ones(1, 4))
        self.assertTrue(torch.equal(out, torch.zeros(4)))
        # Eager raises; with no element to clamp to, the delegate must too.
        with self.assertRaisesRegex(RuntimeError, "empty axis"):
            run(Gather(), torch.zeros(3, 0), torch.zeros(3, 2, dtype=torch.long))

    def test_unsupported_forms_run_on_the_cpu(self):
        class IndexPut(nn.Module):
            def __init__(self, accumulate=False):
                super().__init__()
                self.accumulate = accumulate

            def forward(self, x, index, values):
                # Some work before the refused op, so the program has a delegate.
                x = x.logical_not() if x.dtype == torch.bool else x + 1
                return torch.ops.aten.index_put.default(
                    x, [index], values, self.accumulate
                )

        class Gather(nn.Module):
            def forward(self, x, index):
                return torch.gather(x + 1, 0, index)

        index = torch.tensor([4, 1, 4])
        cases = {
            "int64": (
                IndexPut(),
                (torch.zeros(6, dtype=torch.long), index[1:], torch.tensor([7, 9])),
                "on torch.int64 is not supported",
            ),
            "bool_accumulate": (
                IndexPut(True),
                (
                    torch.ones(6, dtype=torch.bool),
                    index,
                    torch.ones(3, dtype=torch.bool),
                ),
                "accumulate on bool",
            ),
            "bool_mask": (
                IndexPut(),
                (
                    torch.zeros(6),
                    torch.tensor([1, 0, 1, 0, 0, 1]) == 1,
                    torch.tensor(2.0),
                ),
                "only int32 and int64 index tensors",
            ),
            # PyTorch lets a 0-d index gather from a 1-d input.
            "gather_0d_index": (
                Gather(),
                (torch.arange(5.0), torch.tensor(2)),
                "index of rank 0",
            ),
        }
        refused_ops = (torch.ops.aten.index_put.default, torch.ops.aten.gather.default)
        for name, (model, inputs, reason) in cases.items():
            with self.subTest(name):
                # Refused on purpose, not by a handler that crashed.
                builder = MLXProgramBuilder(get_fake_program(export(model, inputs)))
                builder.check_support_only()
                [info] = [
                    info
                    for node, info in builder.node_info.items()
                    if node.target in refused_ops
                ]
                self.assertFalse(info.supported)
                self.assertIn(reason, info.unsupported_reason)

                program = _lower(model, inputs)
                self.assertEqual(_delegate_count(program), 1)
                op = "gather" if isinstance(model, Gather) else "index_put"
                self.assertEqual(_cpu_ops(program), {f"aten.{op}.out"})
                method = (
                    Runtime.get().load_program(program.buffer).load_method("forward")
                )
                # Exact, and bool outputs cannot be subtracted.
                self.assertTrue(
                    torch.equal(method.execute(list(inputs))[0], model(*inputs))
                )


class Sdpa(nn.Module):
    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class GroupedSdpa(nn.Module):
    """Grouped key/value attention, where the repeat is unwrapped before the kernel."""

    def __init__(self, dim: int, is_causal: bool = False):
        super().__init__()
        self.dim = dim
        self.is_causal = is_causal

    def forward(self, q, k, v):
        k = k.repeat_interleave(2, dim=self.dim)
        v = v.repeat_interleave(2, dim=self.dim)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class TestMLXPartitionerSdpaShapes(unittest.TestCase):
    """The fused kernel takes rank 4, so other ranks are adapted or left alone."""

    def test_normalization_runs_before_preservation_check(self):
        observed_ranks = []

        class RankCheckingPartitioner(MLXPartitioner):
            def ops_to_not_decompose(self, ep):
                for node in ep.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target
                        == torch.ops.aten.scaled_dot_product_attention.default
                    ):
                        observed_ranks.append(
                            tuple(arg.meta["val"].dim() for arg in node.args[:3])
                        )
                return super().ops_to_not_decompose(ep)

        for shape in ((16, 64), (2, 16, 64), (1, 2, 16, 64)):
            with self.subTest(rank=len(shape)):
                observed_ranks.clear()
                inputs = tuple(torch.randn(shape) for _ in range(3))
                program = to_edge_transform_and_lower(
                    export(Sdpa(), inputs, strict=False),
                    partitioner=[RankCheckingPartitioner()],
                )
                self.assertTrue(observed_ranks)
                self.assertTrue(all(ranks == (4, 4, 4) for ranks in observed_ranks))
                self.assertEqual(_delegate_count(program), 1)

    def test_handler_requires_normalized_rank4_inputs(self):
        for shape in ((16, 64), (2, 16, 64), (1, 2, 16, 64)):
            with self.subTest(rank=len(shape)):
                ep = export(Sdpa(), tuple(torch.randn(shape) for _ in range(3)))
                sdpa = next(
                    node
                    for node in ep.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                )
                builder = MLXProgramBuilder(ep)
                builder.check_support_only()
                info = builder.node_info[sdpa]
                self.assertEqual(info.supported, len(shape) == 4)
                if len(shape) < 4:
                    self.assertIn("rank-4", info.unsupported_reason)

    def test_handler_keyword_options(self):
        inputs = tuple(torch.randn(1, 2, 3, 8) for _ in range(3))
        for scale in (None, 0.0, 0.25):
            with self.subTest(scale=scale):
                ep = export(Sdpa(), inputs)
                sdpa = next(
                    node
                    for node in ep.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                )
                sdpa.kwargs = {
                    "query": sdpa.args[0],
                    "key": sdpa.args[1],
                    "value": sdpa.args[2],
                    "dropout_p": 0.0,
                    "is_causal": False,
                    "scale": scale,
                    "enable_gqa": False,
                }
                sdpa.args = ()
                ep.graph_module.recompile()
                built = MLXProgramBuilder(ep).build()
                instructions = [
                    instr.op
                    for chain in built.instruction_chains
                    for instr in chain.instructions
                    if type(instr.op).__name__ == "SdpaNode"
                ]
                self.assertEqual(len(instructions), 1)
                self.assertEqual(
                    instructions[0].scale, 8**-0.5 if scale is None else scale
                )

    def test_rank4_is_unchanged(self):
        counts, _, err = _run(
            Sdpa(), tuple(torch.randn(2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 0)
        self.assertEqual(counts.get("SqueezeNode", 0), 0)
        self.assertLess(err, 1e-4)

    def test_rank3_is_lifted_once(self):
        counts, _, err = _run(Sdpa(), tuple(torch.randn(2, 16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertEqual(counts.get("SqueezeNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank2_is_lifted_twice(self):
        counts, _, err = _run(Sdpa(), tuple(torch.randn(16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 6)
        self.assertEqual(counts.get("SqueezeNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank5_is_decomposed_on_this_backend(self):
        # Folding the leading dimensions pairs the wrong operands once one of them
        # broadcasts a batch, so this decomposes rather than fusing.
        counts, delegates, err = _run(
            Sdpa(), tuple(torch.randn(2, 2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_unequal_batch_is_decomposed_on_this_backend(self):
        counts, delegates, err = _run(
            Sdpa(),
            (
                torch.randn(2, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
            ),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_zero_head_count_declines_instead_of_raising(self):
        # The head multiple test would divide by zero here, and raising from the
        # matcher aborts the whole export rather than declining this one node. Only
        # lowering is checked: a zero-size operand is not executable either way.
        program = _lower(
            Sdpa(),
            (
                torch.randn(1, 4, 8, 16),
                torch.randn(1, 0, 8, 16),
                torch.randn(1, 0, 8, 16),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pte"
            path.write_bytes(program.buffer)
            self.assertEqual(get_mlx_node_counts(path).get("SdpaNode", 0), 0)


class TestMLXPartitionerSharedConstants(unittest.TestCase):
    def test_weight_used_twice_in_one_delegate_and_once_in_another(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                y = self.linear(x) + self.linear(x * 2)
                # elu is not lowered, so the last use lands in a second delegate.
                return self.linear(torch.nn.functional.elu(y))

        _, delegates, err = _run(SharedLinear().eval(), (torch.randn(2, 4),))
        self.assertGreater(delegates, 1)
        self.assertLess(err, 1e-5)

    def test_second_placeholder_for_one_constant_raises(self):
        class TwoReads(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("c", torch.randn(4))

            def forward(self, x):
                return x + self.c, x * self.c

        ep = export(TwoReads(), (torch.randn(4),), strict=False)
        original = next(n for n in ep.graph.nodes if n.name == "b_c")
        with ep.graph.inserting_after(original):
            second = ep.graph.placeholder("b_c_second")
        second.meta = dict(original.meta)
        mul = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.mul.Tensor)
        mul.replace_input_with(original, second)
        ep.graph_signature.input_specs.insert(
            1, InputSpec(InputKind.BUFFER, TensorArgument("b_c_second"), "c", True)
        )

        for check_only in (True, False):
            with self.subTest(check_only=check_only):
                builder = MLXProgramBuilder(ep)
                with self.assertRaisesRegex(ValueError, "more than one placeholder"):
                    if check_only:
                        builder.check_support_only()
                    else:
                        builder.build()

    def test_delegates_share_the_bytes_of_a_shared_weight(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                # elu is not lowered, so the second use lands in another delegate.
                return self.linear(torch.nn.functional.elu(self.linear(x)))

        model = SharedLinear().eval()
        weight = model.linear.weight.detach().clone()
        program = to_edge_transform_and_lower(
            export(model, (torch.randn(2, 4),), strict=False),
            partitioner=[MLXPartitioner()],
        ).exported_program()
        delegates = [
            module
            for module in program.graph_module.children()
            if isinstance(module, LoweredBackendModule)
        ]

        self.assertEqual(len(delegates), 2)
        first, second = (
            {id(buffer) for buffer in delegate.named_data_store_output.buffers}
            for delegate in delegates
        )
        self.assertEqual(len(first), 2)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(model.linear.weight, weight))


class TestMLXPartitionerGroupedKeys(unittest.TestCase):
    """Repeated KV heads are fused only when the caller enables the shared passes."""

    def test_rank4_head_repeat_fusion_is_opt_in(self):
        for use_default_passes in (False, True):
            with self.subTest(use_default_passes=use_default_passes):
                counts, _, err = _run(
                    GroupedSdpa(dim=1),
                    (
                        torch.randn(2, 4, 16, 64),
                        torch.randn(2, 2, 16, 64),
                        torch.randn(2, 2, 16, 64),
                    ),
                    transform_passes=(
                        get_default_passes() if use_default_passes else None
                    ),
                )
                self.assertEqual(counts.get("SdpaNode", 0), 1)
                self.assertEqual(
                    counts.get("RepeatNode", 0), 0 if use_default_passes else 2
                )
                self.assertLess(err, 1e-4)

    def test_rank3_sequence_repeat_is_kept(self):
        # At rank 3 dim 1 is the key sequence, so absorbing the repeat would drop
        # half the keys. Without a mask that still sums correctly, which is what
        # makes it easy to miss; with a causal mask it is wrong by whole units.
        counts, _, err = _run(
            GroupedSdpa(dim=1, is_causal=True),
            (torch.randn(2, 16, 64), torch.randn(2, 8, 64), torch.randn(2, 8, 64)),
            transform_passes=get_default_passes(),
        )
        self.assertEqual(counts.get("RepeatNode", 0), 2)
        self.assertLess(err, 1e-4)


class TestMLXPartitionerSdpaCausal(unittest.TestCase):
    """MLX anchors a causal mask at the bottom right and torch at the top left."""

    def test_equal_lengths_stay_fused(self):
        counts, _, err = _run(
            Sdpa(is_causal=True), tuple(torch.randn(1, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank3_equal_lengths_are_lifted(self):
        counts, _, err = _run(
            Sdpa(is_causal=True), tuple(torch.randn(2, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertLess(err, 1e-4)

    def test_rank3_unequal_lengths_use_causal_correction(self):
        counts, delegates, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(2, 6, 64), torch.randn(2, 16, 64), torch.randn(2, 16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("SliceNode", 0), 2)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_rank2_unequal_lengths_use_causal_correction(self):
        counts, _, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(6, 64), torch.randn(16, 64), torch.randn(16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("SliceNode", 0), 2)
        self.assertLess(err, 1e-4)


class TestMLXPartitionerConstantBase(unittest.TestCase):
    def test_slice_scatter_into_a_buffer(self):
        class ScatterIntoBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("base", torch.zeros(4, 8))

            def forward(self, x):
                return torch.slice_scatter(self.base, x, 1, 0, 2) * 2

        counts, _, err = _run(ScatterIntoBuffer().eval(), (torch.randn(4, 2),))
        self.assertEqual(counts.get("SliceUpdateNode", 0), 1)
        self.assertLess(err, 1e-6)


class TestMLXPartitionerBoolIndex(unittest.TestCase):
    """The MLX runtime gathers only with integer indices."""

    class ConstantMask(nn.Module):
        def __init__(self, mask, persistent=None, axis=0, scale=True):
            super().__init__()
            if persistent is None:
                self.mask = mask
            else:
                self.register_buffer("mask", mask, persistent=persistent)
            self.axis = axis
            self.scale = scale

        def forward(self, x):
            y = (x + 1)[(slice(None),) * self.axis + (self.mask,)]
            if self.axis:
                # The leading dim's stride depends on how many elements the
                # mask selects, which exir spec propagation rejects, so keep it
                # in the delegate.
                y = y.sum(self.axis)
            return y * 2 if self.scale else y

    def test_constant_mask_becomes_positions(self):
        mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]) == 1
        cases = {
            "1d": (self.ConstantMask(mask), (8, 16)),
            "2d": (self.ConstantMask(mask.reshape(1, 8)), (1, 8, 16)),
            "3d": (self.ConstantMask(mask.reshape(2, 2, 2)), (2, 2, 2, 5)),
            "buffer": (self.ConstantMask(mask, persistent=False), (8, 16)),
            "axis_1": (self.ConstantMask(mask.reshape(2, 4), axis=1), (3, 2, 4, 5)),
        }
        for name, (model, shape) in cases.items():
            with self.subTest(name):
                _, delegates, err = _run(model.eval(), (torch.randn(shape),))
                self.assertEqual(delegates, 1)
                self.assertLess(err, 1e-5)

    def test_other_bool_indices_run_on_the_cpu(self):
        class ComputedMask(nn.Module):
            def __init__(self):
                super().__init__()
                self.ids = torch.tensor([[1, 7, 7, 2, 7, 3, 7, 4]])

            def forward(self, x):
                return (x + 1)[self.ids == 7] * 2

        class MaskInput(nn.Module):
            def forward(self, x, mask):
                return (x + 1)[mask] * 2

        class MaskWithIndex(nn.Module):
            def __init__(self, mask):
                super().__init__()
                self.mask = mask

            def forward(self, x, i):
                return (x + 1)[self.mask, i] * 2

        class MaskThenUpdate(nn.Module):
            def __init__(self, mask):
                super().__init__()
                self.register_buffer("mask", mask, persistent=False)

            def forward(self, x):
                y = (x + 1)[self.mask] * 2
                self.mask.copy_(x[:, :, 0] > 0)
                return y

        mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1]]) == 1
        x = torch.randn(1, 8, 16)
        # An all-false mask gives an empty result, so it is the graph output
        # here: an MLX op after it would have to write an empty output too.
        all_false = self.ConstantMask(torch.zeros(1, 8, dtype=torch.bool), scale=False)
        cases = {
            "computed_mask": (ComputedMask(), (x,), 2),
            "mask_input": (MaskInput(), (x, mask), 2),
            "mask_with_index": (MaskWithIndex(mask), (x, torch.tensor([3])), 2),
            "persistent_buffer": (self.ConstantMask(mask, persistent=True), (x,), 2),
            "all_false": (all_false, (x,), 1),
        }
        for name, (model, inputs, expected) in cases.items():
            with self.subTest(name):
                _, delegates, err = _run(model.eval(), inputs)
                self.assertEqual(delegates, expected)
                self.assertLess(err, 1e-6)

        with self.subTest("mutated_buffer"):
            # Not run: the eager reference call would overwrite the mask first.
            program = _lower(MaskThenUpdate(mask).eval(), (x,))
            self.assertEqual(_delegate_count(program), 2)

    def test_handler_refuses_masks_it_cannot_lower(self):
        mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]) == 1
        cases = {
            # A persistent buffer has no values in the fake copy, so the handler
            # must refuse it rather than fail reading it.
            "persistent_buffer": (
                self.ConstantMask(mask, persistent=True),
                "only a constant boolean mask",
            ),
            "all_false": (
                self.ConstantMask(torch.zeros(8, dtype=torch.bool)),
                "all-false",
            ),
        }
        for name, (model, reason) in cases.items():
            with self.subTest(name):
                # The same fake copy of the program that to_backend gives the
                # partitioner.
                ep = get_fake_program(export(model.eval(), (torch.randn(8, 16),)))
                index = next(
                    n for n in ep.graph.nodes if n.target == torch.ops.aten.index.Tensor
                )
                builder = MLXProgramBuilder(ep)
                builder.check_support_only()
                info = builder.node_info[index]
                self.assertFalse(info.supported)
                self.assertIn(reason, info.unsupported_reason)


class TestMLXPartitionerMixedSupport(unittest.TestCase):
    """An operator is preserved from decomposition per operator, not per call."""

    def test_supported_and_unsupported_calls_in_one_graph(self):
        class Mixed(nn.Module):
            def forward(self, a, b):
                x = torch.nn.functional.scaled_dot_product_attention(a, a, a)
                y = torch.nn.functional.scaled_dot_product_attention(b, b, b)
                return x.sum() + y.sum()

        # Without giving the whole operator back, the rank-5 call would be neither
        # lowered nor decomposed and this would raise a missing out variant.
        counts, delegates, err = _run(
            Mixed().eval(),
            (torch.randn(1, 4, 16, 64), torch.randn(2, 2, 4, 16, 64)),
        )
        # The cost of the coarse choice: the supported call is unfused as well.
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-3)

    def test_mixed_support_outside_attention(self):
        class TwoRolls(nn.Module):
            def forward(self, x):
                return torch.roll(x, 1, dims=0).sum() + torch.roll(x, 1).sum()

        _, delegates, err = _run(TwoRolls().eval(), (torch.randn(4, 8),))
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)


if __name__ == "__main__":
    unittest.main()
