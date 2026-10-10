# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from unittest.mock import Mock, patch

import torch
from executorch.backends.samsung._passes.enn_pass_manager import EnnPassManager
from executorch.backends.samsung.builders.op_topk import TopKVisitor
from executorch.backends.samsung.partition.enn_partitioner import (
    EnnOperatorSupport,
    EnnPartitioner,
)
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.ops.test_topk import QuantizedTopK
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TopK(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, x):
        result = torch.topk(x, 300)
        if self.output == "indices":
            return result.indices
        if self.output == "values":
            return result.values
        return result


class TestTopKVisitor(unittest.TestCase):
    def _is_supported(self, program, node):
        with patch(
            "executorch.backends.samsung.partition.enn_partitioner.PyEnnWrapper.EnnWrapper",
            create=True,
        ), patch("executorch.backends.samsung.partition.enn_partitioner.EnnGraph"):
            support = EnnOperatorSupport(
                program, [gen_samsung_backend_compile_spec("E9955")]
            )
            return support.is_node_supported(None, node)

    def test_portable_quantized_topk_serializes(self):
        for output in ("indices", "values", "both"):
            with self.subTest(output=output):
                program = to_edge(
                    torch.export.export(
                        QuantizedTopK(k=30, output=output, quantize_values=True),
                        (torch.randperm(127).float().view(1, 127) * 0.1,),
                    ),
                    compile_config=EdgeCompileConfig(_skip_dim_order=True),
                )
                try:
                    serialized = program.to_executorch().buffer
                except RuntimeError as error:
                    self.fail(str(error))
                self.assertGreater(len(serialized), 0)

    def test_quantized_input_without_quantized_values_stays_on_cpu(self):
        for output in ("indices", "values", "both"):
            with self.subTest(output=output):
                program = to_edge(
                    torch.export.export(
                        QuantizedTopK(k=300, output=output), (torch.randn(1, 2100),)
                    ),
                    compile_config=EdgeCompileConfig(_skip_dim_order=True),
                ).exported_program()
                topk = next(
                    node
                    for node in program.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                self.assertNotIn("quantize_attrs", topk.meta)
                self.assertFalse(self._is_supported(program, topk))
                graph_module = EnnPassManager().transform_for_preprocess_pass(program)
                topk = next(
                    node
                    for node in graph_module.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                self.assertTrue(topk.args[0].meta.get("quantize_attrs"))
                self.assertFalse(self._is_supported(program, topk))

    def test_quantized_values_stay_on_cpu_before_and_after_preprocess(self):
        for output in ("values", "both"):
            with self.subTest(output=output):
                program = to_edge(
                    torch.export.export(
                        QuantizedTopK(k=300, output=output, quantize_values=True),
                        (torch.randn(1, 2100),),
                    ),
                    compile_config=EdgeCompileConfig(_skip_dim_order=True),
                ).exported_program()
                topk = next(
                    node
                    for node in program.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                self.assertNotIn("quantize_attrs", topk.meta)
                self.assertFalse(self._is_supported(program, topk))
                graph_module = EnnPassManager().transform_for_preprocess_pass(program)
                topk = next(
                    node
                    for node in graph_module.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                self.assertFalse(self._is_supported(program, topk))

    def test_output_qdq_stays_with_portable_topk(self):
        program = to_edge(
            torch.export.export(
                QuantizedTopK(k=300, quantize_values=True), (torch.randn(1, 2100),)
            ),
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        ).exported_program()
        topk = next(
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.aten.topk.default
        )
        values = next(node for node in topk.users if node.args[1] == 0)
        quantize = next(iter(values.users))
        dequantize = next(iter(quantize.users))
        for node in (topk, values, quantize, dequantize):
            with self.subTest(target=node.target):
                self.assertFalse(self._is_supported(program, node))

    def test_unquantized_topk_stays_on_cpu(self):
        for output in ("indices", "values", "both"):
            with self.subTest(output=output):
                program = to_edge(
                    torch.export.export(TopK(output), (torch.randn(1, 2100),)),
                    compile_config=EdgeCompileConfig(_skip_dim_order=True),
                ).exported_program()
                topk = next(
                    node
                    for node in program.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                self.assertFalse(self._is_supported(program, topk))

    def test_preserves_quantized_output_modes(self):
        for output in ("indices", "values", "both"):
            with self.subTest(output=output):
                program = to_edge(
                    torch.export.export(TopK(output), (torch.randn(1, 2100),)),
                    compile_config=EdgeCompileConfig(_skip_dim_order=True),
                ).exported_program()
                topk = next(
                    node
                    for node in program.graph.nodes
                    if node.target == exir_ops.edge.aten.topk.default
                )
                topk.meta["quantize_attrs"] = {
                    QuantConstants.QUANT_KEY.quant_dtype: torch.int8
                }
                graph = Mock()
                graph.define_tensor.side_effect = range(3)
                ids = {}
                self.assertTrue(TopKVisitor(program).define_node(topk, graph, ids))
                sdk_outputs = graph.define_op.call_args.args[3]
                self.assertEqual(len(sdk_outputs), 2 if output == "both" else 1)
                self.assertEqual(
                    graph.define_op.call_args.args[4],
                    {
                        "k_dims": 300,
                        "output": {"indices": "index", "values": "value"}.get(
                            output, "both"
                        ),
                        "axis": 1,
                        "quant_dtype": "AINT8",
                    },
                )
                for user, sdk_output in zip(topk.users, sdk_outputs):
                    self.assertEqual(ids[user], sdk_output)

    def test_getitems_follow_cpu_topk_and_downstream_ops_stay_delegated(self):
        class TopKAndAdd(torch.nn.Module):
            def forward(self, x):
                values, indices = torch.topk(x, 300)
                return values + 1, indices

        program = to_edge(
            torch.export.export(TopKAndAdd(), (torch.randn(1, 2100),)),
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        ).exported_program()
        with patch(
            "executorch.backends.samsung.partition.enn_partitioner.PyEnnWrapper.EnnWrapper",
            create=True,
        ), patch("executorch.backends.samsung.partition.enn_partitioner.EnnGraph"):
            partitions = EnnPartitioner(
                [gen_samsung_backend_compile_spec("E9955")]
            ).generate_partitions(program)
        delegated = {node for partition in partitions for node in partition.nodes}
        self.assertTrue(delegated)
        self.assertFalse(
            any(
                node.target in (exir_ops.edge.aten.topk.default, operator.getitem)
                for node in delegated
            )
        )
        self.assertTrue(
            any(node.target == exir_ops.edge.aten.add.Tensor for node in delegated)
        )


if __name__ == "__main__":
    unittest.main()
