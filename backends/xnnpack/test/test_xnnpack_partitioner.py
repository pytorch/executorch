# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    ExecutorchBackendConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.export.experimental import _export_forward_backward


class TestXnnpackPartitioner(unittest.TestCase):
    """Test cases for XnnpackPartitioner functionality and deprecation warnings."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    def test_deprecation_warning_for_to_backend_workflow(self):
        """
        Test that the deprecated to_edge + to_backend workflow shows a deprecation warning.
        """
        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        edge = to_edge(exported_model)
        partitioner = XnnpackPartitioner()

        edge.to_backend(partitioner)

        log_contents = log_capture_string.getvalue()
        self.assertIn("DEPRECATION WARNING", log_contents)
        self.assertIn("to_edge() + to_backend()", log_contents)
        self.assertIn("to_edge_transform_and_lower()", log_contents)

    def test_no_warning_for_to_edge_transform_and_lower_workflow(self):
        """
        Test that the recommended to_edge_transform_and_lower workflow does NOT show a deprecation warning.
        """

        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        partitioner = XnnpackPartitioner()

        to_edge_transform_and_lower(exported_model, partitioner=[partitioner])

        log_contents = log_capture_string.getvalue()
        self.assertNotIn("DEPRECATION WARNING", log_contents)

    def test_multi_method_partitioning_with_shared_weights(self):
        """
        Test that multi-method models with shared weights are correctly partitioned.
        Verify that:
        1. Both methods are fully lowered to XNNPACK.
        2. Constants are not duplicated between named data and constant buffers.
        3. Program executes correctly.
        """

        class MultiMethodModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)
                self.linear2 = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear2(F.sigmoid(self.linear(x)))

            def forward_2(self, x):
                return self.linear2(F.relu(self.linear(x)))

            def example_inputs(self):
                return (torch.randn(1, 8),)

        model = MultiMethodModel()

        # Get eager reference output.
        example_inputs = model.example_inputs()
        with torch.no_grad():
            fwd1_eager = model.forward(*example_inputs)
            fwd2_eager = model.forward_2(*example_inputs)

        # Export both methods
        ep_fwd = export(model, model.example_inputs(), strict=True)
        # Patch the forward, as export only traces the 'forward' method.
        model.forward = model.forward_2
        ep_fwd_2 = export(model, model.example_inputs(), strict=True)

        # Convert to edge and lower to executorch
        edge = to_edge({"forward": ep_fwd, "forward_2": ep_fwd_2})
        lowered = edge.to_backend(XnnpackPartitioner(force_fp32_dynamic_linear=True))
        executorch = lowered.to_executorch()

        # Check that graph is fully delegated.
        nodes_1 = list(lowered._edge_programs["forward"].graph.nodes)
        nodes_2 = list(lowered._edge_programs["forward_2"].graph.nodes)
        self.assertEqual(len(nodes_1), 5)
        self.assertEqual(len(nodes_2), 5)
        expected_node_names = [
            "x",
            "lowered_module_0",
            "executorch_call_delegate",
            "getitem",
            "output_1",
        ]
        for n in expected_node_names:
            self.assertTrue(any(node.name == n for node in nodes_1))
            self.assertTrue(any(node.name == n for node in nodes_2))

        # Check that weights are not duplicated.
        self.assertEqual(len(executorch._named_data.pte_data), 4)
        self.assertEqual(len(executorch._named_data.buffers), 4)
        self.assertEqual(len(executorch._named_data.external_data), 0)

        # Check that there are no constant buffers (besides the placeholder).
        self.assertEqual(len(executorch._emitter_output.program.constant_buffer), 1)

        # Check for model correctness.
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        fwd1_et = executorch_module.run_method("forward", example_inputs)
        fwd2_et = executorch_module.run_method("forward_2", example_inputs)
        self.assertTrue(torch.allclose(fwd1_eager, fwd1_et[0], 1e-3))
        self.assertTrue(torch.allclose(fwd2_eager, fwd2_et[0], 1e-3))

    def test_parametrized_weight_is_folded_before_partitioning(self):
        """
        A weight computed from parameters (here weight_norm) is folded into a
        constant before partitioning, so the convolution is delegated instead
        of falling back to the portable kernels with the weight computation.
        """

        class ParametrizedConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(4, 4, 3)
                )

            def forward(self, x):
                return self.conv(x)

        model = ParametrizedConv().eval()
        example_inputs = (torch.randn(1, 4, 8),)
        eager = model(*example_inputs)

        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        call_functions = [
            node
            for node in edge.exported_program().graph_module.graph.nodes
            if node.op == "call_function"
        ]
        delegates = [
            node
            for node in call_functions
            if node.target == torch.ops.higher_order.executorch_call_delegate
        ]
        self.assertEqual(len(delegates), 1)
        # The delegate call and the getitem on its output are all that is left.
        self.assertEqual(len(call_functions), 2)

        # The module keeps a pointer into the buffer rather than a copy, so the
        # program manager that owns the buffer has to outlive the module.
        executorch = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        self.assertTrue(
            torch.allclose(
                executorch_module.forward(example_inputs)[0],
                eager,
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_pre_decomposition_folding_keeps_quantization_primitives(self):
        """
        Folding must not touch the Q/DQ chain that convert_pt2e leaves on a
        quantized weight, or the weight would be dequantized at export time.
        """
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        model = self.SimpleModel().eval()
        example_inputs = (torch.randn(2, 10),)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(export(model, example_inputs).module(), quantizer)
        prepared(*example_inputs)
        converted = convert_pt2e(prepared)

        def quant_targets(ep):
            return sorted(
                str(node.target)
                for node in ep.graph.nodes
                if node.op == "call_function"
                and "quantized_decomposed" in str(node.target)
            )

        class GroupwiseLinear(torch.nn.Module):
            """A weight stored as int8 groups, the way 4-bit LLM exports do."""

            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randint(-8, 8, (8, 16), dtype=torch.int8)
                )
                self.register_buffer("scales", torch.rand(8, 2))
                self.register_buffer("zeros", torch.zeros(8, 2, dtype=torch.int8))

            def forward(self, x):
                weight = torch.ops.quantized_decomposed.dequantize_per_channel_group(
                    self.weight, self.scales, self.zeros, -8, 7, torch.int8, 8, x.dtype
                )
                return torch.nn.functional.linear(x, weight)

        for quantized, inputs in (
            (converted, example_inputs),
            (GroupwiseLinear(), (torch.randn(2, 16),)),
        ):
            exported = export(quantized, inputs)
            before = quant_targets(exported)
            self.assertGreater(len(before), 0)
            after = quant_targets(
                XnnpackPartitioner().transform_for_pre_decomposition(exported)
            )
            self.assertEqual(before, after)

    def test_pre_decomposition_folding_skips_factory_ops(self):
        """
        A scalar fill of a static shape stays an op, so that the fold does not
        turn it into a stored tensor. Every target in the skip set is covered,
        and each is checked to be what the edge-level pass skips for the same
        reason: a fill that decomposes to aten.full or aten.full_like.
        """
        fills = {
            torch.ops.aten.full.default: lambda p: torch.full((4, 8), 1.5),
            torch.ops.aten.new_full.default: lambda p: p.new_full((4, 8), 1.5),
            torch.ops.aten.ones.default: lambda p: torch.ones(4, 8),
            torch.ops.aten.new_ones.default: lambda p: p.new_ones((4, 8)),
            torch.ops.aten.zeros.default: lambda p: torch.zeros(4, 8),
            torch.ops.aten.new_zeros.default: lambda p: p.new_zeros((4, 8)),
            torch.ops.aten.full_like.default: lambda p: torch.full_like(p, 1.5),
            torch.ops.aten.ones_like.default: lambda p: torch.ones_like(p),
            torch.ops.aten.zeros_like.default: lambda p: torch.zeros_like(p),
        }
        self.assertEqual(
            set(fills), set(XnnpackPartitioner._CONSTANT_PROP_SKIP_TARGETS)
        )

        class Fill(torch.nn.Module):
            def __init__(self, fill):
                super().__init__()
                self.fill = fill
                self.p = torch.nn.Parameter(torch.randn(4, 8))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.fill(self.p))

        def call_targets(ep):
            return [
                node.target for node in ep.graph.nodes if node.op == "call_function"
            ]

        for target, fill in fills.items():
            exported = export(Fill(fill).eval(), (torch.randn(4, 8),))
            self.assertIn(target, call_targets(exported))
            decomposed = set(call_targets(exported.run_decompositions()))
            self.assertTrue(
                decomposed
                & {torch.ops.aten.full.default, torch.ops.aten.full_like.default},
                f"{target} does not decompose to a fill: {decomposed}",
            )

            folded = XnnpackPartitioner().transform_for_pre_decomposition(exported)
            self.assertIn(target, call_targets(folded), f"{target} was folded")
            self.assertEqual(len(folded.constants), 0)
            self.assertEqual(
                list(folded.graph_signature.inputs_to_parameters.values()), ["p"]
            )

    def test_pre_decomposition_folding_keeps_mutated_buffer(self):
        """
        A buffer the model writes in place, such as a KV cache, is not a
        constant. The ATen program handed to the hook lists no mutated buffers
        yet; the write is still an in-place copy_ on a view of the buffer.
        Folding that view would leave the write on a constant, and
        run_decompositions would then fail on the aliasing.
        """

        class Cache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(8, 8))
                self.register_buffer("cache", torch.zeros(4, 8))

            def forward(self, x):
                previous = self.cache[:2]
                self.cache[:2] = x
                return (previous + x) @ self.w.t()

        model = Cache().eval()
        example_inputs = (torch.randn(2, 8),)

        folded = XnnpackPartitioner().transform_for_pre_decomposition(
            export(model, example_inputs)
        )
        self.assertEqual(
            list(folded.graph_signature.buffers_to_mutate.values()), ["cache"]
        )
        self.assertEqual(len(folded.constants), 0)
        (weight,) = folded.graph_signature.inputs_to_parameters.values()
        self.assertRegex(weight, r"^w_prop_[0-9a-f]{8}$")

        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        self.assertEqual(
            list(edge.exported_program().graph_signature.buffers_to_mutate.values()),
            ["cache"],
        )
        executorch = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        # The initial state of a mutated buffer is not serialized, so the first
        # call only primes the cache on both sides. From then on the cache
        # carries state from one call to the next.
        x = torch.randn(2, 8)
        executorch_module.forward((x,))
        model(x)
        for _ in range(3):
            x = torch.randn(2, 8)
            self.assertTrue(
                torch.allclose(
                    executorch_module.forward((x,))[0],
                    model(x),
                    rtol=1e-5,
                    atol=1e-5,
                )
            )

    def test_pre_decomposition_folding_keeps_buffers_shared_across_methods(self):
        """
        A buffer one method only reads can be written by another method of
        the same program. The hook sees one method at a time, so buffers stay
        out of the fold and the reader keeps its shared allocation.
        """

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

        class Write(torch.nn.Module):
            def __init__(self, shared):
                super().__init__()
                self.shared = shared

            def forward(self, x):
                self.shared.state.copy_(x)
                return x

        class Read(torch.nn.Module):
            def __init__(self, shared):
                super().__init__()
                self.shared = shared

            def forward(self, x):
                return x + self.shared.state.sum()

        shared = Shared()
        x = torch.ones(4)
        edge = to_edge_transform_and_lower(
            {"write": export(Write(shared), (x,)), "read": export(Read(shared), (x,))},
            partitioner={
                "write": [XnnpackPartitioner()],
                "read": [XnnpackPartitioner()],
            },
        )
        read = edge.exported_program("read")
        self.assertEqual(
            list(read.graph_signature.inputs_to_buffers.values()), ["shared.state"]
        )
        self.assertEqual(len(read.constants), 0)

        program = edge.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True),
                emit_mutable_buffer_names=True,
            )
        ).executorch_program
        for plan in program.execution_plan:
            shared_names = [
                value.val.extra_tensor_info.fully_qualified_name
                for value in plan.values
                if getattr(value.val, "allocation_info", None) is not None
                and value.val.allocation_info.memory_id == 2
            ]
            self.assertEqual(shared_names, ["shared.state"], plan.name)

    def test_pre_decomposition_folding_skips_training_graphs(self):
        """
        A training graph keeps its parameters as inputs: the runtime hands
        them to the optimizer through the gradient and parameter outputs.
        """

        class Loss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 8))

            def forward(self, x):
                return (x @ self.weight.t()).sum()

        joint = _export_forward_backward(export(Loss(), (torch.randn(2, 8),)))
        self.assertIs(
            XnnpackPartitioner().transform_for_pre_decomposition(joint), joint
        )

        edge = to_edge_transform_and_lower(
            joint,
            partitioner=[
                XnnpackPartitioner(force_non_static_weights_for_f32_linear=True)
            ],
        )
        self.assertEqual(
            list(edge.exported_program().graph_signature.inputs_to_parameters.values()),
            ["weight"],
        )
        methods = [
            plan.name for plan in edge.to_executorch().executorch_program.execution_plan
        ]
        self.assertIn("__et_training_parameters_index_forward", methods)

    def test_pre_decomposition_folding_handles_view_of_parameter(self):
        """
        Before decomposition aten.t returns a view of the parameter, which
        keeps requires_grad. The fold registers a detached leaf; otherwise
        the retrace in to_edge clones it into a non-leaf that the delegate
        cannot deep-copy.
        """

        class MatmulT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(3, 4))

            def forward(self, x):
                return x @ self.w.t()

        model = MatmulT().eval()
        example_inputs = (torch.randn(2, 4),)
        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        self.assertEqual(
            [n.op for n in edge.exported_program().graph.nodes].count("get_attr"), 1
        )
        executorch = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        self.assertTrue(
            torch.allclose(
                executorch_module.forward(example_inputs)[0],
                model(*example_inputs),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_pre_decomposition_folding_keeps_external_weight_tags(self):
        """
        A weight tagged for an external file keeps its tag through the fold.
        The folded value is a parameter named after the weight, so the tag
        function sees the name and run_decompositions carries the custom
        meta to the delegate.
        """

        class TaggedWeights(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 4))
                self.lora_weight = torch.nn.Parameter(torch.randn(3, 4))

            def forward(self, x):
                return x @ self.weight.t() + x @ self.lora_weight.t()

        def gen_tag_fn(node):
            return "lora.ptd" if "lora" in node.name else "foundation.ptd"

        def sha256(tensor):
            return hashlib.sha256(
                tensor.detach().t().contiguous().numpy().tobytes()
            ).hexdigest()

        model = TaggedWeights().eval()
        example_inputs = (torch.randn(2, 4),)
        module = export(model, example_inputs).module()
        delegate_external_constants_pass_unlifted(module, gen_tag_fn)
        edge = to_edge_transform_and_lower(
            export(module, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        executorch = edge.to_executorch(
            ExecutorchBackendConfig(external_constants=gen_tag_fn)
        )
        self.assertEqual(
            {
                file: set(entries)
                for file, entries in executorch._named_data.external_data.items()
            },
            {
                "foundation.ptd": {sha256(model.weight)},
                "lora.ptd": {sha256(model.lora_weight)},
            },
        )
        self.assertEqual(len(executorch._named_data.pte_data), 0)

    def test_pre_decomposition_folding_names_folds_after_their_expression(self):
        """
        The external constant map is keyed by name and shared by the methods
        of a program. A folded value is named after its source and the
        expression, so two methods that fold the same parameter through
        different expressions write two entries, and two methods that fold
        the same expression share one.
        """

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3, 4))

        class Transpose(torch.nn.Module):
            def __init__(self, shared):
                super().__init__()
                self.shared = shared

            def forward(self, x):
                return x @ self.shared.w.t()

        class ScaledTranspose(Transpose):
            def forward(self, x):
                return x @ (self.shared.w * 14).t()

        shared = Shared()
        example_inputs = (torch.ones(2, 4),)
        partitioner = XnnpackPartitioner()
        programs = {
            name: partitioner.transform_for_pre_decomposition(
                export(model, example_inputs)
            )
            for name, model in (
                ("prefill", Transpose(shared)),
                ("decode", ScaledTranspose(shared)),
                ("scaled", ScaledTranspose(shared)),
            )
        }
        executorch = to_edge(programs).to_executorch(
            ExecutorchBackendConfig(external_constants=lambda node: "weights.ptd")
        )
        external_map = executorch._emitter_output.external_constant_map
        self.assertEqual(len(external_map["weights.ptd"]), 2)
        for name in external_map["weights.ptd"]:
            self.assertRegex(name, r"^shared\.w_prop_[0-9a-f]{8}$")
        self.assertEqual(len(executorch._emitter_output.external_constant_buffer), 2)
        # The program reads the folded values from the .ptd. With one name
        # for both folds the second write replaced the first, and one method
        # read the other's weight: 4 where eager gives 56, or the reverse.
        with tempfile.TemporaryDirectory() as directory:
            executorch.write_tensor_data_to_file(directory)
            data = (Path(directory) / "weights.ptd").read_bytes()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer, data)
        for name, expected in (("prefill", 4.0), ("decode", 56.0), ("scaled", 56.0)):
            output = executorch_module.run_method(name, example_inputs)[0]
            self.assertTrue(torch.equal(output, torch.full((2, 3), expected)), name)

    def test_pre_decomposition_folding_folds_gemm_weights_only(self):
        """
        Only a computed weight or bias of a GEMM-like op is folded: it makes
        the op partitionable. A parameter-only subgraph elsewhere unlocks no
        delegation and would only be executed at export time and stored, so
        an arange mask, an expand and a parameter-derived output stay ops.
        A program with no computed GEMM weight leaves the hook untouched.
        """

        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(4, 4, 3)
                )
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.row = torch.nn.Parameter(torch.randn(6))

            def forward(self, x):
                mask = torch.arange(6) < 3
                expanded = self.row.expand(64, 6)
                return self.conv(x) * self.scale, mask, expanded, self.scale * 2

        model = Mixed().eval()
        example_inputs = (torch.randn(1, 4, 8),)
        folded = XnnpackPartitioner().transform_for_pre_decomposition(
            export(model, example_inputs)
        )
        targets = [n.target for n in folded.graph.nodes if n.op == "call_function"]
        self.assertNotIn(torch.ops.aten._weight_norm.default, targets)
        self.assertIn(torch.ops.aten.arange.default, targets)
        self.assertIn(torch.ops.aten.expand.default, targets)
        self.assertEqual(targets.count(torch.ops.aten.mul.Tensor), 2)
        parameters = list(folded.graph_signature.inputs_to_parameters.values())
        self.assertEqual(set(parameters[:-1]), {"conv.bias", "scale", "row"})
        self.assertRegex(
            parameters[-1],
            r"^conv\.parametrizations\.weight\.original0_prop_[0-9a-f]{8}$",
        )

        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        executorch = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        for actual, expected in zip(
            executorch_module.forward(example_inputs), model(*example_inputs)
        ):
            self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))

        static = export(self.SimpleModel().eval(), (torch.randn(2, 10),))
        self.assertIs(
            XnnpackPartitioner().transform_for_pre_decomposition(static), static
        )

    def test_pre_decomposition_folding_does_not_duplicate_shared_weights(self):
        """
        A fold is applied only if it does not make the program larger. A
        parameter used both inside and outside the folded expression cannot
        be erased, so its fold would only add a copy: a tied embedding read
        by one lookup and one transposed matmul stays as it is, and so does a
        weight used twice.
        """

        class TiedEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 32)

            def forward(self, ids):
                hidden = self.embedding(ids)
                return hidden @ self.embedding.weight.t()

        class TwoUses(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(32, 32))

            def forward(self, x):
                return x @ self.w.t() + x @ self.w

        for model, example_inputs in (
            (TiedEmbedding().eval(), (torch.tensor([[1, 2, 3]]),)),
            (TwoUses().eval(), (torch.randn(2, 32),)),
        ):
            folded = XnnpackPartitioner().transform_for_pre_decomposition(
                export(model, example_inputs)
            )
            self.assertEqual(len(folded.state_dict), 1)
            self.assertTrue(
                any(
                    n.target is torch.ops.aten.t.default
                    for n in folded.graph.nodes
                    if n.op == "call_function"
                )
            )
            with_hook = to_edge_transform_and_lower(
                export(model, example_inputs), partitioner=[XnnpackPartitioner()]
            ).to_executorch()
            with mock.patch.object(
                XnnpackPartitioner,
                "transform_for_pre_decomposition",
                lambda self, exported_program: exported_program,
            ):
                without_hook = to_edge_transform_and_lower(
                    export(model, example_inputs), partitioner=[XnnpackPartitioner()]
                ).to_executorch()
            self.assertEqual(len(with_hook.buffer), len(without_hook.buffer))
            executorch_module = _load_for_executorch_from_buffer(with_hook.buffer)
            self.assertTrue(
                torch.allclose(
                    executorch_module.forward(example_inputs)[0],
                    model(*example_inputs),
                    rtol=1e-5,
                    atol=1e-5,
                )
            )

    def test_pre_decomposition_folding_keeps_scalar_item(self):
        """
        aten.item yields a Python float that its consumer takes directly, so
        there is no tensor to lift. The op stays and the export goes through.
        """

        class ScaleByItem(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 4))
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.register_buffer("offset", torch.tensor(1.0))

            def forward(self, x):
                weight = self.weight * self.scale.item() + self.offset.item()
                return torch.nn.functional.linear(x, weight)

        model = ScaleByItem().eval()
        example_inputs = (torch.randn(2, 4),)
        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        executorch = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        self.assertTrue(
            torch.allclose(
                executorch_module.forward(example_inputs)[0],
                model(*example_inputs),
                rtol=1e-5,
                atol=1e-5,
            )
        )
