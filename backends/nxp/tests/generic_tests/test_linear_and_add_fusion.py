# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import numpy as np
import torch

from executorch.backends.nxp.aten_passes.fuse_linear_and_add_pass import (
    FuseLinearAndAddPass,
)
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.aten_passes.remove_nodes_with_known_outputs import (
    RemoveNodesWithKnownOutputs,
)
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from parameterized import parameterized


class LinearAddModule(torch.nn.Module):
    def __init__(
        self,
        fc_in_features: int,
        fc_out_features: int,
        bias: bool,
        artificial_bias_shape: list[int],
        alpha=1.0,
    ):
        super().__init__()
        self.fc_in_features = fc_in_features
        self.fc_out_features = fc_out_features
        self.bias = bias
        self.artificial_bias_shape = artificial_bias_shape
        self.alpha = alpha
        self.linear = torch.nn.Linear(fc_in_features, fc_out_features, bias=bias)
        self.eval()

    def forward(self, x):
        artificial_bias = torch.ones(self.artificial_bias_shape, dtype=torch.float32)
        x = self.linear(x)
        return torch.add(x, artificial_bias, alpha=self.alpha)


class LinearAddModuleReverseNodeOrder(torch.nn.Module):
    """The `ones` added by the `add` are only generated after the `linear` node."""

    def __init__(
        self,
        fc_in_features: int,
        fc_out_features: int,
        bias: bool,
        artificial_bias_shape: list[int],
    ):
        super().__init__()
        self.fc_in_features = fc_in_features
        self.fc_out_features = fc_out_features
        self.bias = bias
        self.artificial_bias_shape = artificial_bias_shape
        self.linear = torch.nn.Linear(fc_in_features, fc_out_features, bias=bias)
        self.eval()

    def forward(self, x):
        # The `ones` are generated after the `linear` call.
        x = self.linear(x)
        artificial_bias = torch.ones(self.artificial_bias_shape, dtype=torch.float32)
        return torch.add(x, artificial_bias)


class LinearAddModuleReverseInputOrder(torch.nn.Module):
    """The `add` has the output of the `linear` as its second input (which is the input multiplied by `alpha`)."""

    def __init__(
        self,
        fc_in_features: int,
        fc_out_features: int,
        bias: bool,
        artificial_bias_shape: list[int],
        alpha=1.0,
    ):
        super().__init__()
        self.fc_in_features = fc_in_features
        self.fc_out_features = fc_out_features
        self.bias = bias
        self.artificial_bias_shape = artificial_bias_shape
        self.alpha = alpha
        self.linear = torch.nn.Linear(fc_in_features, fc_out_features, bias=bias)
        self.eval()

    def forward(self, x):
        artificial_bias = torch.ones(self.artificial_bias_shape, dtype=torch.float32)
        x = self.linear(x)
        return torch.add(artificial_bias, x, alpha=self.alpha)  # Reversed input order.


class TestLinearAndAddFusing(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests.

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    @parameterized.expand(
        [
            ["2D", [4, 6]],
            ["4D", [4, 6, 8, 10]],
        ]
    )
    def test_linear_add_fusing__static__no_bias__valid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, False, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # The `add` has been removed.
        assert len(modified_nodes) == 5
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert "ones" in modified_nodes[3].args[2].name
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    @parameterized.expand(
        [
            ["2D", [8, 10]],
        ]
    )
    def test_linear_add_fusing__static__no_bias__invalid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(
            input_shape[-1], 5, False, [8, 5]  # Unsupported `linear` bias shape.
        )
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert len(original_nodes[3].args) == 2
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # Nothing changed.
        assert len(modified_nodes) == 6
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert modified_nodes[4].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    @parameterized.expand(
        [
            ["2D", [4, 6]],
            ["4D", [2, 3, 4, 5]],
        ]
    )
    def test_linear_add_fusing__static__bias__valid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, True, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 7
        assert original_nodes[3].target == torch.ops.aten.ones.default
        assert original_nodes[4].target == torch.ops.aten.linear.default
        assert len(original_nodes[4].args) == 3
        assert original_nodes[5].target == torch.ops.aten.add.Tensor

        # make sure the `add` and the `ones` were removed.
        assert len(modified_nodes) == 5
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.ones.default]
        )
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert "combined" in modified_nodes[3].args[2].name
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__static__no_bias__reverse_order(self):
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        # Use a module where the `bias` is generated after the `linear` node, which prevents the change.
        module = LinearAddModuleReverseNodeOrder(input_shape[-1], 5, False, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[2].target == torch.ops.aten.linear.default
        assert len(original_nodes[2].args) == 2
        assert (
            original_nodes[3].target == torch.ops.aten.ones.default
        )  # `ones` after `linear`.
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # The `add` has been removed.
        assert len(modified_nodes) == 5
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__static__bias__reverse_order(self):
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        # Use a module where the `bias` is generated after the `linear` node, which prevents the change.
        module = LinearAddModuleReverseNodeOrder(input_shape[-1], 5, True, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 7
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert len(original_nodes[3].args) == 3
        assert (
            original_nodes[4].target == torch.ops.aten.ones.default
        )  # `ones` after `linear`.
        assert original_nodes[5].target == torch.ops.aten.add.Tensor

        # The `add` and `ones` have been removed.
        assert len(modified_nodes) == 5
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.ones.default]
        )
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__static__alpha__no_bias(self):
        alpha = 2.34
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, False, [5], alpha=alpha)
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[2].target == torch.ops.aten.ones.default
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert len(original_nodes[3].args) == 2
        assert original_nodes[4].target == torch.ops.aten.add.Tensor
        assert original_nodes[4].kwargs["alpha"] == alpha

        # The `add` has been removed.
        assert len(modified_nodes) == 5
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__static__alpha__bias(self):
        alpha = 2.34
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, True, [5], alpha=alpha)
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 7
        assert original_nodes[3].target == torch.ops.aten.ones.default
        assert original_nodes[4].target == torch.ops.aten.linear.default
        assert len(original_nodes[4].args) == 3
        assert original_nodes[5].target == torch.ops.aten.add.Tensor
        assert original_nodes[5].kwargs["alpha"] == alpha

        # The `add` has been removed.
        assert len(modified_nodes) == 5
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert len(modified_nodes[3].args) == 3
        assert not graph_contains_any_of_ops(
            modified_module.graph, [torch.ops.aten.add.Tensor]
        )

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__static__alpha__reversed_add_inputs(self):
        alpha = 2.34
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        module = LinearAddModuleReverseInputOrder(
            input_shape[-1], 5, True, [5], alpha=alpha
        )
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec,
            [
                RemoveNodesWithKnownOutputs(),  # Make the added tensor static.
                FuseLinearAndAddPass(),
            ],
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 7
        assert original_nodes[3].target == torch.ops.aten.ones.default
        assert original_nodes[4].target == torch.ops.aten.linear.default
        assert len(original_nodes[4].args) == 3
        assert original_nodes[5].target == torch.ops.aten.add.Tensor
        assert (
            original_nodes[5].args[1] == original_nodes[4]
        )  # `linear` is the second input.
        assert original_nodes[5].kwargs["alpha"] == alpha

        # Nothing changed (except the `ones` was replaced by static data).
        assert len(modified_nodes) == 7
        assert modified_nodes[4].target == torch.ops.aten.linear.default
        assert len(modified_nodes[4].args) == 3
        assert modified_nodes[5].target == torch.ops.aten.add.Tensor
        assert (
            modified_nodes[5].args[1] == modified_nodes[4]
        )  # `linear` is the second input.
        assert modified_nodes[5].kwargs["alpha"] == alpha

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    @parameterized.expand(
        [
            ["2D", [4, 6]],
        ]
    )
    def test_linear_add_fusing__dynamic__no_bias__valid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, False, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec, [FuseLinearAndAddPass()]
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # Nothing changed.
        assert len(modified_nodes) == 6
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert modified_nodes[4].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    @parameterized.expand(
        [
            ["2D", [8, 10]],
        ]
    )
    def test_linear_add_fusing__dynamic__no_bias__invalid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(
            input_shape[-1], 5, False, [8, 5]  # Unsupported `linear` bias shape.
        )
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec, [FuseLinearAndAddPass()]
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # Nothing changed.
        assert len(modified_nodes) == 6
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert modified_nodes[4].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    @parameterized.expand(
        [
            ["2D", [4, 6]],
        ]
    )
    def test_linear_add_fusing__dynamic__bias__valid_shape(
        self, _, input_shape: list[int]
    ):
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, True, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec, [FuseLinearAndAddPass()]
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 7
        assert original_nodes[3].target == torch.ops.aten.ones.default
        assert original_nodes[4].target == torch.ops.aten.linear.default
        assert original_nodes[5].target == torch.ops.aten.add.Tensor

        # Nothing has changed, as the second bias is dynamic, so it cannot be added together with the first bias.
        assert len(modified_nodes) == 7
        assert modified_nodes[3].target == torch.ops.aten.ones.default
        assert modified_nodes[4].target == torch.ops.aten.linear.default
        assert modified_nodes[5].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__dynamic__reverse_order(self):
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        # Use a module where the `bias` is generated after the `linear` node, which prevents the change.
        module = LinearAddModuleReverseNodeOrder(input_shape[-1], 5, False, [5])
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec, [FuseLinearAndAddPass()]
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[2].target == torch.ops.aten.linear.default
        assert original_nodes[3].target == torch.ops.aten.ones.default
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # Nothing has changed.
        assert len(modified_nodes) == 6
        assert modified_nodes[2].target == torch.ops.aten.linear.default
        assert modified_nodes[3].target == torch.ops.aten.ones.default
        assert modified_nodes[4].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_linear_add_fusing__dynamic__alpha(self):
        alpha = 2.34
        input_shape = [4, 8]
        example_input = (torch.ones(input_shape),)

        module = LinearAddModule(input_shape[-1], 5, False, [5], alpha=alpha)
        program = torch.export.export(module, example_input, strict=True)
        original_module = program.module()

        modified_module = NeutronAtenPassManager(
            neutron_target_spec, [FuseLinearAndAddPass()]
        )(deepcopy(program.module())).graph_module

        # Make sure the module wasn't broken.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 6
        assert original_nodes[2].target == torch.ops.aten.ones.default
        assert original_nodes[3].target == torch.ops.aten.linear.default
        assert original_nodes[4].target == torch.ops.aten.add.Tensor

        # Nothing has changed.
        assert len(modified_nodes) == 6
        assert modified_nodes[2].target == torch.ops.aten.ones.default
        assert modified_nodes[3].target == torch.ops.aten.linear.default
        assert modified_nodes[4].target == torch.ops.aten.add.Tensor

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)
