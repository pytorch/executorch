# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from types import SimpleNamespace
from unittest import TestCase

import torch

from executorch.backends.vulkan.patterns import replace_all_fusable_subgraphs
from executorch.backends.vulkan.patterns.rms_norm import (
    RmsNormMatch,
    replace_rms_norm_with_fused_op,
)
from executorch.exir.dialects._ops import ops as exir_ops


def _fake_program(graph_module, **signature_overrides):
    signature = {
        "inputs_to_parameters": {},
        "inputs_to_buffers": {},
        "inputs_to_lifted_tensor_constants": {},
        "non_persistent_buffers": set(),
    }
    signature.update(signature_overrides)
    return SimpleNamespace(
        constants={},
        graph_module=graph_module,
        graph_signature=SimpleNamespace(**signature),
        state_dict={},
    )


def _make_rms_norm_graph(
    *,
    input_dtype=torch.float32,
    compute_dtype=torch.float32,
    square="pow",
    eps=1e-5,
    eps_is_tensor=False,
    alpha=1,
    mean_dims=None,
    keepdim=True,
    exponent=2,
    share_input=True,
    weight_shape=(1280,),
    weight_dtype=None,
    output_cast_after_scale=False,
    scale_dtype=None,
):
    graph = torch.fx.Graph()
    input_shape = (1, 4, 1280)

    def set_val(node, shape, dtype):
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    x = set_val(graph.placeholder("x"), input_shape, input_dtype)
    weight = set_val(
        graph.placeholder("weight"), weight_shape, weight_dtype or input_dtype
    )

    x_compute = x
    if input_dtype != compute_dtype:
        x_compute = set_val(
            graph.call_function(
                exir_ops.edge.aten._to_copy.default,
                args=(x,),
                kwargs={"dtype": compute_dtype},
            ),
            input_shape,
            compute_dtype,
        )

    square_input = x_compute
    if not share_input:
        square_input = set_val(graph.placeholder("other_x"), input_shape, compute_dtype)

    if square == "pow":
        squared = set_val(
            graph.call_function(
                exir_ops.edge.aten.pow.Tensor_Scalar,
                args=(square_input, exponent),
            ),
            input_shape,
            compute_dtype,
        )
    elif square == "mul":
        squared = set_val(
            graph.call_function(
                exir_ops.edge.aten.mul.Tensor,
                args=(square_input, square_input),
            ),
            input_shape,
            compute_dtype,
        )
    else:
        raise AssertionError(f"Unsupported square form: {square}")

    mean = set_val(
        graph.call_function(
            exir_ops.edge.aten.mean.dim,
            args=(squared, mean_dims or [-1], keepdim),
        ),
        (1, 4, 1),
        compute_dtype,
    )

    if eps_is_tensor:
        eps_arg = set_val(graph.placeholder("eps"), (), compute_dtype)
        add_target = exir_ops.edge.aten.add.Tensor
    else:
        eps_arg = eps
        add_target = exir_ops.edge.aten.add.Scalar

    added = set_val(
        graph.call_function(add_target, args=(mean, eps_arg, alpha)),
        (1, 4, 1),
        compute_dtype,
    )
    reciprocal_std = set_val(
        graph.call_function(exir_ops.edge.aten.rsqrt.default, args=(added,)),
        (1, 4, 1),
        compute_dtype,
    )
    normalized = set_val(
        graph.call_function(
            exir_ops.edge.aten.mul.Tensor, args=(x_compute, reciprocal_std)
        ),
        input_shape,
        compute_dtype,
    )

    normalized_for_output = normalized
    weight_for_output = weight
    if input_dtype != compute_dtype and not output_cast_after_scale:
        normalized_for_output = set_val(
            graph.call_function(
                exir_ops.edge.aten._to_copy.default,
                args=(normalized,),
                kwargs={"dtype": input_dtype},
            ),
            input_shape,
            input_dtype,
        )
    elif input_dtype != compute_dtype:
        scale_dtype = scale_dtype or compute_dtype
        weight_for_output = set_val(
            graph.call_function(
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                args=(weight,),
                kwargs={"dtype": scale_dtype, "dim_order": [0]},
            ),
            weight_shape,
            scale_dtype,
        )

    output_dtype = (
        scale_dtype or compute_dtype
        if output_cast_after_scale
        else weight_dtype or input_dtype
    )
    scaled = set_val(
        graph.call_function(
            exir_ops.edge.aten.mul.Tensor,
            args=(normalized_for_output, weight_for_output),
        ),
        input_shape,
        output_dtype,
    )
    output = scaled
    if output_cast_after_scale:
        output = set_val(
            graph.call_function(
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                args=(scaled,),
                kwargs={"dtype": input_dtype, "dim_order": [0, 1, 2]},
            ),
            input_shape,
            input_dtype,
        )
    graph.output((output,))
    graph_module = SimpleNamespace(graph=graph)
    return graph_module, scaled, eps_arg


class TestRmsNormMatch(TestCase):
    def test_matches_scalar_and_tensor_epsilon(self) -> None:
        cases = (
            _make_rms_norm_graph(),
            _make_rms_norm_graph(mean_dims=[2]),
            _make_rms_norm_graph(eps_is_tensor=True, square="mul"),
            _make_rms_norm_graph(
                input_dtype=torch.float16,
                compute_dtype=torch.float32,
            ),
            _make_rms_norm_graph(
                input_dtype=torch.float16,
                compute_dtype=torch.float32,
                output_cast_after_scale=True,
            ),
        )
        for _, output, _ in cases:
            with self.subTest(output=output):
                self.assertTrue(RmsNormMatch(output).match_found)

    def test_rejects_near_matches(self) -> None:
        cases = (
            _make_rms_norm_graph(alpha=2),
            _make_rms_norm_graph(mean_dims=[-2]),
            _make_rms_norm_graph(keepdim=False),
            _make_rms_norm_graph(exponent=3),
            _make_rms_norm_graph(share_input=False),
            _make_rms_norm_graph(eps=-1e-5),
            _make_rms_norm_graph(eps=math.inf),
            _make_rms_norm_graph(weight_shape=(1279,)),
            _make_rms_norm_graph(weight_dtype=torch.float16),
            _make_rms_norm_graph(compute_dtype=torch.float16),
            _make_rms_norm_graph(
                input_dtype=torch.float16,
                compute_dtype=torch.float32,
                output_cast_after_scale=True,
                scale_dtype=torch.float64,
            ),
        )
        for _, output, _ in cases:
            with self.subTest(output=output):
                self.assertFalse(RmsNormMatch(output).match_found)


class TestRmsNormReplacement(TestCase):
    def test_replaces_scalar_epsilon(self) -> None:
        graph_module, output, _ = _make_rms_norm_graph()
        match = RmsNormMatch(output)

        replaced = replace_rms_norm_with_fused_op(
            _fake_program(graph_module), graph_module, match
        )

        self.assertTrue(replaced)
        fused = [
            node
            for node in graph_module.graph.nodes
            if node.target == exir_ops.edge.et_vk.rms_norm.default
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].args[2], 1e-5)

    def test_replaces_lifted_constant_and_buffer_epsilon(self) -> None:
        for epsilon_kind in ("constant", "buffer"):
            graph_module, output, eps_node = _make_rms_norm_graph(eps_is_tensor=True)
            program = _fake_program(graph_module)
            if epsilon_kind == "constant":
                program.graph_signature.inputs_to_lifted_tensor_constants = {
                    eps_node.name: "epsilon"
                }
                program.constants["epsilon"] = torch.tensor(1e-6)
            else:
                program.graph_signature.inputs_to_buffers = {eps_node.name: "epsilon"}
                program.state_dict["epsilon"] = torch.tensor(1e-6)

            with self.subTest(epsilon_kind=epsilon_kind):
                replaced = replace_rms_norm_with_fused_op(
                    program, graph_module, RmsNormMatch(output)
                )
                self.assertTrue(replaced)
                fused = [
                    node
                    for node in graph_module.graph.nodes
                    if node.target == exir_ops.edge.et_vk.rms_norm.default
                ]
                self.assertEqual(len(fused), 1)
                self.assertAlmostEqual(fused[0].args[2], 1e-6)

    def test_dynamic_tensor_epsilon_remains_unfused(self) -> None:
        graph_module, _, _ = _make_rms_norm_graph(eps_is_tensor=True)

        replaced = replace_all_fusable_subgraphs(
            _fake_program(graph_module), graph_module
        )

        self.assertEqual(replaced, 0)
        self.assertFalse(
            any(
                node.target == exir_ops.edge.et_vk.rms_norm.default
                for node in graph_module.graph.nodes
            )
        )
