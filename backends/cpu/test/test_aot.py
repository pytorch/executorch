# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from unittest import mock

import torch
from executorch.backends.cpu.partitioner import (
    CPU_DELEGATE_VERSION,
    CPUPartitioner,
    CPUSemanticOperators,
)
from executorch.backends.cpu.preprocess import CpuBackend
from executorch.backends.cpu.recipes import CPURecipeType
from executorch.backends.native.serialization import deserialize_graph, serialize_graph
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.schema import DelegateCall, Tensor
from executorch.export import export, ExportRecipe


class CPUAOTTest(unittest.TestCase):
    def test_convnext_block_native_payload_and_constants(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 7, padding=3, groups=4)
                self.norm = torch.nn.LayerNorm(4)
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                y = self.conv(x).permute(0, 2, 3, 1)
                y = torch.nn.functional.gelu(self.linear(self.norm(y)))
                return y.permute(0, 3, 1, 2) + x

        model = Block().eval()
        sample = (torch.randn(1, 4, 8, 8),)
        session = export(
            model, [sample], export_recipe=ExportRecipe.get_recipe(CPURecipeType.FP32)
        )
        program = session.get_executorch_program()
        self.assertEqual(
            [d.id for d in program.execution_plan[0].delegates], ["CpuBackend"]
        )
        self.assertEqual(
            [
                (spec.key, len(spec.value), int.from_bytes(spec.value, "little"))
                for spec in program.execution_plan[0].delegates[0].compile_specs
            ],
            [("cpu_delegate_version", 4, CPU_DELEGATE_VERSION)],
        )
        blob = bytes(program.backend_delegate_data[0].data)
        self.assertEqual(blob[4:8], b"NPTG")
        graph = deserialize_graph(blob)
        layer_norm = next(
            n
            for n in graph.nodes
            if n.target == "torch.ops.aten.native_layer_norm.default"
        )
        self.assertEqual(len(layer_norm.outputs), 3)
        self.assertIn("torch.ops.aten.linear.default", [n.target for n in graph.nodes])
        self.assertNotIn(
            "torch.ops.aten.addmm.default", [n.target for n in graph.nodes]
        )
        self.assertNotIn(
            "torch.ops.aten.permute.default", [n.target for n in graph.nodes]
        )
        torch.testing.assert_close(
            session.get_edge_program_manager().exported_program().module()(*sample),
            model(*sample),
        )
        self.assertGreater(len(session.get_pte_buffer()), 0)

    def test_semantic_partition_does_not_encode_provider_coverage(self):
        class Negate(torch.nn.Module):
            def forward(self, x):
                return -x

        edge = to_edge_transform_and_lower(
            torch.export.export(Negate(), (torch.ones(2),)),
            partitioner=[CPUPartitioner()],
        )
        program = edge.to_executorch().executorch_program
        self.assertEqual(
            [d.id for d in program.execution_plan[0].delegates], ["CpuBackend"]
        )
        graph = deserialize_graph(bytes(program.backend_delegate_data[0].data))
        self.assertIn(
            "torch.ops.aten.neg.default", [node.target for node in graph.nodes]
        )

    def test_delegate_boundary_layout_matches_program(self):
        class ChannelMean(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=1)

        class ChannelsLast(torch.nn.Module):
            def forward(self, x):
                return x.to(memory_format=torch.channels_last)

        for model, memory_format, delegates in (
            (ChannelMean(), torch.contiguous_format, ["CpuBackend"]),
            (ChannelMean(), torch.channels_last, []),
            (ChannelsLast(), torch.contiguous_format, []),
        ):
            with self.subTest(model=type(model).__name__, memory_format=memory_format):
                sample = torch.randn(2, 3, 4, 5).contiguous(memory_format=memory_format)
                edge = to_edge_transform_and_lower(
                    torch.export.export(model, (sample,)),
                    partitioner=[CPUPartitioner()],
                )
                program = edge.to_executorch().executorch_program
                plan = program.execution_plan[0]
                self.assertEqual([d.id for d in plan.delegates], delegates)
                if not delegates:
                    continue
                call = next(
                    instruction.instr_args
                    for chain in plan.chains
                    for instruction in chain.instructions
                    if isinstance(instruction.instr_args, DelegateCall)
                )
                data_index = plan.delegates[call.delegate_index].processed.index
                graph = deserialize_graph(
                    bytes(program.backend_delegate_data[data_index].data)
                )
                metadata = {value.name: value.meta for value in graph.tensor_values}
                for name, index in zip(
                    graph.inputs + graph.outputs, call.args, strict=True
                ):
                    with self.subTest(boundary=name):
                        tensor = plan.values[index].val
                        self.assertIsInstance(tensor, Tensor)
                        self.assertEqual(
                            metadata[name].dim_order,
                            tensor.dim_order,
                            "CPU payload layout must match the tensor passed to the delegate",
                        )

    def test_preprocess_reverts_dim_order_ops(self):
        class AllocateAndClone(torch.nn.Module):
            def forward(self, x):
                empty = torch.empty(x.shape, dtype=x.dtype, device=x.device)
                return empty, x.clone()

        edge = to_edge(
            torch.export.export(AllocateAndClone(), (torch.randn(2, 3),))
        ).exported_program()
        dim_order_targets = {
            exir_ops.edge.dim_order_ops._empty_dim_order.default,
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
        }
        for target in dim_order_targets:
            self.assertIn(target, {node.target for node in edge.graph.nodes})

        result = CpuBackend.preprocess(
            edge,
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                )
            ],
        )
        graph = deserialize_graph(bytes(result.processed_bytes))
        targets = {node.target for node in graph.nodes}
        self.assertTrue(
            {
                "torch.ops.aten.empty.memory_format",
                "torch.ops.aten.clone.default",
            }.issubset(targets)
        )
        self.assertFalse(
            any(target and "dim_order_ops" in target for target in targets)
        )

    def test_constants_have_recorded_readable_tail(self):
        model = torch.nn.Linear(2, 3).eval()
        result = CpuBackend.preprocess(
            torch.export.export(model, (torch.ones(1, 2),)),
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                )
            ],
        )
        data = result.data_store_output
        self.assertIsNotNone(data)
        for name, tensor in model.state_dict().items():
            entry = data.pte_data[name]
            buffer = data.buffers[entry.buffer_index]
            self.assertEqual(entry.alignment, 64)
            self.assertEqual(len(buffer), tensor.numel() * tensor.element_size() + 64)
            self.assertEqual(buffer[-64:], bytes(64))

    def test_preprocess_rejects_compile_spec_mismatch(self):
        ep = torch.export.export(torch.nn.Linear(2, 3), (torch.ones(1, 2),))
        for specs in (
            [],
            [CompileSpec("wrong_key", CPU_DELEGATE_VERSION.to_bytes(4, "little"))],
            [CompileSpec("cpu_delegate_version", b"static_fp32_contiguous_v1")],
            [
                CompileSpec(
                    "cpu_delegate_version",
                    (CPU_DELEGATE_VERSION + 1).to_bytes(4, "little"),
                )
            ],
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(3, "little"),
                )
            ],
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(5, "little"),
                )
            ],
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(4, "big"),
                )
            ],
            [
                CompileSpec(
                    "cpu_delegate_version",
                    CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                ),
                CompileSpec("extra", b""),
            ],
        ):
            with self.assertRaisesRegex(
                ValueError,
                f"CpuBackend requires delegate version {CPU_DELEGATE_VERSION}",
            ) as error:
                CpuBackend.preprocess(ep, specs)
            self.assertIn(
                f"got {[(spec.key, spec.value) for spec in specs]!r}",
                str(error.exception),
            )

    def test_preprocess_rejects_unsupported_constants(self):
        for dtype, device in (
            (torch.bfloat16, "cpu"),
            (torch.float64, "cpu"),
            (torch.float32, "meta"),
        ):
            with self.subTest(dtype=dtype, device=device):
                model = torch.nn.Linear(2, 3, dtype=dtype, device=device)
                ep = torch.export.export(
                    model, (torch.ones(1, 2, dtype=dtype, device=device),)
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"CpuBackend requires FP32 CPU constant weight, "
                    f"got dtype {dtype} on {device}",
                ):
                    CpuBackend.preprocess(
                        ep,
                        [
                            CompileSpec(
                                "cpu_delegate_version",
                                CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                            )
                        ],
                    )

    def test_preprocess_rejects_non_contiguous_constant(self):
        ep = torch.export.export(torch.nn.Linear(2, 3), (torch.ones(1, 2),))
        with mock.patch(
            "executorch.backends.cpu.preprocess.serialize_graph",
            return_value=(b"", {"weight": torch.ones(3, 2).t()}),
        ):
            with self.assertRaisesRegex(
                ValueError, "CpuBackend requires contiguous constant weight"
            ):
                CpuBackend.preprocess(
                    ep,
                    [
                        CompileSpec(
                            "cpu_delegate_version",
                            CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                        )
                    ],
                )

    def test_preprocess_rejects_non_contiguous_boundaries(self):
        class Transpose(torch.nn.Module):
            def forward(self, x):
                return x.t()

        for model, sample, boundary in (
            (torch.nn.ReLU(), torch.ones(2, 3).t(), "input"),
            (Transpose(), torch.ones(2, 3), "t"),
        ):
            with self.subTest(boundary=boundary):
                ep = torch.export.export(model, (sample,))
                with self.assertRaisesRegex(
                    ValueError,
                    f"CpuBackend requires contiguous boundary tensor {boundary}",
                ):
                    CpuBackend.preprocess(
                        ep,
                        [
                            CompileSpec(
                                "cpu_delegate_version",
                                CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                            )
                        ],
                    )

    def test_preprocess_preserves_input_meta(self):
        class Transpose(torch.nn.Module):
            def forward(self, x):
                return (x.t() * 2).contiguous()

        ep = torch.export.export(Transpose(), (torch.ones(2, 3),))
        before = {node: node.meta.get("val") for node in ep.graph.nodes}
        for fail_serialization in (False, True):
            with self.subTest(fail_serialization=fail_serialization), mock.patch(
                "executorch.backends.cpu.preprocess.serialize_graph",
                wraps=serialize_graph,
                side_effect=(
                    RuntimeError("serialization failed") if fail_serialization else None
                ),
            ) as serialize:
                specs = [
                    CompileSpec(
                        "cpu_delegate_version",
                        CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                    )
                ]
                if fail_serialization:
                    with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                        CpuBackend.preprocess(ep, specs)
                else:
                    CpuBackend.preprocess(ep, specs)
                serialized_module = serialize.call_args.args[0]
                self.assertIsNot(serialized_module, ep.graph_module)
                self.assertTrue(
                    any(
                        isinstance(val, torch.Tensor) and not val.is_contiguous()
                        for val in before.values()
                    )
                )
                for node in serialized_module.graph.nodes:
                    val = node.meta.get("val")
                    if isinstance(val, torch.Tensor):
                        self.assertTrue(val.is_contiguous())
                for node in ep.graph.nodes:
                    self.assertIs(node.meta.get("val"), before[node])

    def test_malformed_getitem_is_unsupported(self):
        graph = torch.fx.Graph()
        for args in ((), ((torch.ones(1),),), ((torch.ones(1),), 0)):
            with self.subTest(args=args):
                node = graph.call_function(operator.getitem, args)
                self.assertFalse(CPUSemanticOperators().is_node_supported({}, node))

    def test_missing_input_or_output_metadata_is_unsupported(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.neg.default, (x,))
        x.meta["val"] = node.meta["val"] = torch.ones(2)
        support = CPUSemanticOperators()
        self.assertTrue(support.is_node_supported({}, node))
        for related_node in (x, node):
            with self.subTest(node=related_node.name):
                val = related_node.meta.pop("val")
                self.assertFalse(support.is_node_supported({}, node))
                related_node.meta["val"] = val

    def test_getitem_validates_output_metadata(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(2)
        producer = graph.call_function(
            torch.ops.aten.native_layer_norm.default, (x, [2], None, None, 1e-5)
        )
        producer.meta["val"] = (torch.ones(2), torch.ones(1), torch.ones(1))
        node = graph.call_function(operator.getitem, (producer, 0))
        node.meta["val"] = producer.meta["val"][0]
        support = CPUSemanticOperators()
        self.assertTrue(support.is_node_supported({}, node))
        for val in (None, 1, (), torch.ones(2, dtype=torch.int32)):
            with self.subTest(val=val):
                node.meta["val"] = val
                self.assertFalse(support.is_node_supported({}, node))

    def test_unsupported_dtype_remains_in_executorch(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return x + x

        ep = torch.export.export(Add(), (torch.ones(2, dtype=torch.int32),))
        edge = to_edge_transform_and_lower(ep, partitioner=[CPUPartitioner()])
        self.assertEqual(
            len(edge.to_executorch().executorch_program.execution_plan[0].delegates), 0
        )

    def test_recipe_rejects_unknown_options(self):
        with self.assertRaisesRegex(ValueError, "Unexpected CPU recipe options"):
            ExportRecipe.get_recipe(CPURecipeType.FP32, precision="fp16")
