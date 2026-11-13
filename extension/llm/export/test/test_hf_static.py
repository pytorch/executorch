# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph-level regressions for hf_static configuration and padding writes."""

import unittest
from types import SimpleNamespace

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.llm.export import hf_static
from executorch.extension.llm.export.cache_graph import (
    fold_position_tables,
    fold_weight_arithmetic,
)
from executorch.runtime import Runtime, Verification


class TestHfStaticGraph(unittest.TestCase):
    def test_weight_arithmetic_is_folded(self):
        # Gemma-style RMSNorm scale: (1 + weight) becomes one constant.
        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8))

            def forward(self, x):
                norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return norm * (1.0 + self.weight.float())

        gm = torch.export.export(Norm(), (torch.randn(2, 8),)).module()
        x = torch.randn(2, 8)
        expected = gm(x)
        fold_weight_arithmetic(gm)
        gm.recompile()
        last = next(n for n in gm.graph.nodes if n.op == "output").args[0][0]
        self.assertTrue(any(a.op == "get_attr" for a in last.args))
        torch.testing.assert_close(gm(x), expected)
        torch.export.export(gm, (x,))

    def _writes(self, valid, start=8):
        graph = torch.fx.Graph()
        cache, rows, position, count = [
            graph.placeholder(n) for n in ("cache", "rows", "position", "count")
        ]
        graph.output(cache)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        graph.erase_node(next(n for n in graph.nodes if n.op == "output"))
        out = hf_static._sliding_writes(gm, cache, rows, position, count, 6, 4)
        graph.output(out)
        gm.recompile()
        payloads = []

        class Observe(torch.fx.Interpreter):
            def run_node(self, node):
                if node.target == torch.ops.aten.index_copy_.default:
                    args, _ = self.fetch_args_kwargs_from_env(node)
                    payloads.append((args[2].clone(), args[3].clone()))
                return super().run_node(node)

        cache = torch.arange(12.0).reshape(1, 1, 6, 2)
        rows = torch.arange(100.0, 108.0).reshape(1, 1, 4, 2)
        result = Observe(gm).run(cache, rows, torch.tensor(start), torch.tensor(valid))
        return result, payloads

    def test_reinplace_write_chain_has_no_full_cache_copy(self):
        class Chain(torch.nn.Module):
            def forward(self, cache, rows, index):
                cache.index_copy_(2, index, rows)
                checksum = cache.sum()
                cache.index_copy_(2, index, rows + 1)
                return checksum, cache

        inputs = (torch.zeros(1, 1, 6, 2), torch.ones(1, 1, 2, 2), torch.tensor([1, 2]))
        program = torch.export.export(Chain(), inputs)
        manager = to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).to_executorch(
            ExecutorchBackendConfig(
                run_reinplace_pass=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            )
        )
        plan = manager.executorch_program.execution_plan[0]
        ops = [op.name for op in plan.operators]
        self.assertNotIn("aten::index_put", ops)
        self.assertNotIn("aten::copy_", ops)
        self.assertEqual(plan.outputs[0], plan.inputs[0])
        runtime = Runtime.get().load_program(
            manager.buffer, verification=Verification.Minimal
        )
        outputs = runtime.load_method("forward").execute(inputs)
        self.assertEqual(float(outputs[1]), 4.0)
        torch.testing.assert_close(
            inputs[0][..., 1:3, :], torch.full((1, 1, 2, 2), 2.0)
        )

    def test_full_context_position_table_probe(self):
        class Position(torch.nn.Module):
            def forward(self, positions, values):
                return values + torch.cos(positions.float())[..., None]

        inputs = (torch.arange(4).reshape(1, 4), torch.zeros(1, 4, 2))
        program = torch.export.export(Position(), inputs)
        gm = program.module()
        position = next(
            n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "positions"
        )
        fold_position_tables(gm, position, position, 4, torch.float32)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        torch.testing.assert_close(gm(*inputs), Position()(*inputs))

    def test_padding_collision_payloads_are_identical(self):
        for valid in (1, 2, 3, 4):
            with self.subTest(valid=valid):
                result, payloads = self._writes(valid)
                slots, rows = payloads[-1]
                for i in range(4):
                    for j in range(i + 1, 4):
                        if slots[i] == slots[j]:
                            torch.testing.assert_close(
                                rows[:, :, i], rows[:, :, j], rtol=0, atol=0
                            )
                expected = torch.cat(
                    (
                        torch.arange(12.0).reshape(1, 1, 6, 2)[:, :, valid:],
                        torch.arange(100.0, 108.0).reshape(1, 1, 4, 2)[:, :, :valid],
                    ),
                    2,
                )
                torch.testing.assert_close(result, expected)

    def test_prefill_longer_than_window_is_rejected(self):
        config = SimpleNamespace(num_hidden_layers=2, sliding_window=6)
        # The graph is deliberately absent: validation must precede tracing.
        with self.assertRaisesRegex(ValueError, "smallest sliding cache"):
            hf_static.specialize(None, 8, 32, config=config)

    def test_all_sliding_config_without_layer_types(self):
        config = SimpleNamespace(num_hidden_layers=2, sliding_window=6)
        self.assertEqual(hf_static.sliding_windows(config), [6, 6])
        config.layer_types = ["sliding_attention", "full_attention"]
        self.assertEqual(hf_static.sliding_windows(config), [6, None])

    def test_inconsistent_window_config_is_rejected(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            sliding_window=6,
            layer_types=["full_attention", "full_attention"],
        )
        with self.assertRaisesRegex(ValueError, "no sliding layer"):
            hf_static.specialize(None, 4, 32, config=config)

    def test_oversized_write_is_rejected(self):
        with self.assertRaises(AssertionError):
            hf_static._sliding_writes(None, None, None, None, None, 3, 4)


if __name__ == "__main__":
    unittest.main()
