# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.recipes.nxp_recipe_provider import (
    NEUTRON_RECIPE_CONFIG_KEY,
    NeutronRecipeConfig,
    NXPRecipeProvider,
)
from executorch.backends.nxp.recipes.nxp_recipe_types import NXPRecipeType
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.executors import (
    graph_contains_any,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.ops_aliases import ExecutorchDelegateCall
from executorch.export import export
from executorch.export.recipe import ExportRecipe
from torch._inductor.lowering import quantized_decomposed


class SimpleCNN(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.reshape(1, -1)
        x = x + x
        return x


INPUT_SHAPE = (1, 3, 8, 8)


def _run_export(
    model, rc, recipe_type=NXPRecipeType.INT8_PTQ_NEUTRON, input_shape=INPUT_SHAPE
):
    example_inputs = [(torch.randn(input_shape),)]
    recipe = NXPRecipeProvider().create_recipe(recipe_type, neutron_recipe_config=rc)
    return export(model, example_inputs=example_inputs, export_recipe=recipe)


def _get_graph(sess):
    return sess.get_edge_program_manager().exported_program().graph


def test_ptq_neutron_basic():
    """Baseline PTQ: whole model delegated, IO is quantized."""
    model = SimpleCNN()
    rc = NeutronRecipeConfig(INPUT_SHAPE)
    sess = _run_export(model, rc)
    graph = _get_graph(sess)

    assert graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])

    def is_cnn_op(n):
        return any(op in n.name.lower() for op in ["conv", "relu", "view", "add"])

    assert not graph_contains_any(graph, is_cnn_op)

    nodes = list(graph.nodes)
    assert nodes[2].target == quantized_decomposed.quantize_per_tensor.out
    assert nodes[-2].target == quantized_decomposed.dequantize_per_tensor.out


class TestInt8PTQNoDelegate:

    def test__basic(self):
        """INT8_PTQ_NO_DELEGATE: model is quantized but no delegate call appears."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        sess = _run_export(model, rc, recipe_type=NXPRecipeType.INT8_PTQ_NO_DELEGATE)
        graph = _get_graph(sess)

        assert not graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])

        def is_cnn_op(n):
            return any(op in n.name.lower() for op in ["conv", "relu", "view", "add"])

        # With no delegation, original ops should be visible in the graph.
        assert graph_contains_any(graph, is_cnn_op)

    def test__recipe_has_empty_partitioners(self):
        """INT8_PTQ_NO_DELEGATE recipe has an empty partitioner list."""
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NO_DELEGATE, neutron_recipe_config=rc
        )
        assert recipe.lowering_recipe.partitioners == []


class TestNeutronRecipeConfigFlags:
    def test_operators_not_to_delegate(self):
        """Ops listed in operators_not_to_delegate are not lowered to Neutron."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(
            INPUT_SHAPE, operators_not_to_delegate=["aten::convolution"]
        )
        sess = _run_export(model, rc)
        graph = _get_graph(sess)

        assert graph_contains_any_of_ops(
            graph, [torch.ops.aten.convolution.out]
        )  # Convolution was not delegated.
        assert graph_contains_any_of_ops(
            graph, [ExecutorchDelegateCall]
        )  # Other operators were delegated.

        def _is_relu_add_or_view(n: torch.fx.Node) -> bool:
            return any(op in n.name.lower() for op in ["relu", "add", "view"])

        assert not graph_contains_any(graph, _is_relu_add_or_view)

    def test_remove_quant_io_ops(self):
        """remove_quant_io_ops=True: no quantize op at the IO boundary."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, remove_quant_io_ops=True)
        sess = _run_export(model, rc)
        graph = _get_graph(sess)
        nodes = list(graph.nodes)

        real_nodes = [n for n in nodes if n.op not in ("placeholder", "output")]
        assert real_nodes[0].target != quantized_decomposed.quantize_per_tensor.out
        assert real_nodes[-1].target != quantized_decomposed.dequantize_per_tensor.out
        assert real_nodes[-1].meta["val"].dtype == torch.int8
        placeholder_nodes = [n for n in nodes if n.op == "placeholder"]
        assert placeholder_nodes[0].name == "x"  # Main input
        assert placeholder_nodes[0].meta["val"].dtype == torch.int8

    def test_use_quant_state_dict_false(self, mocker):
        """use_quant_state_dict=False: the NeutronPartitioner used during lowering has
        post_quantization_state_dict=None, confirmed by intercepting the constructor."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, use_quant_state_dict=False)

        captured = []
        original_init = NeutronPartitioner.__init__

        def capturing_init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            captured.append(self_)

        mocker.patch.object(NeutronPartitioner, "__init__", capturing_init)

        _run_export(model, rc)

        assert (
            len(captured) == 1
        ), "Expected exactly one NeutronPartitioner to be created."
        assert captured[0].post_quantization_state_dict is None

    def test_custom_delegation_options_explicit(self, mocker):
        """Explicitly provided CustomDelegationOptions are forwarded to NeutronPartitioner."""
        model = SimpleCNN()
        opts = CustomDelegationOptions()
        rc = NeutronRecipeConfig(INPUT_SHAPE, custom_delegation_options=opts)

        captured = []
        original_init = NeutronPartitioner.__init__

        def capturing_init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            captured.append(self_)

        mocker.patch.object(NeutronPartitioner, "__init__", capturing_init)
        _run_export(model, rc)

        assert len(captured) == 1
        assert captured[0].custom_delegation_options == opts

    def test_intermediates_dir(self, tmp_path):
        """intermediates_dir: intermediate compilation files are written to the directory."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, intermediates_dir=str(tmp_path))
        _run_export(model, rc)
        assert any(
            tmp_path.iterdir()
        ), "No intermediate files written to intermediates_dir."

    def test_fetch_constants_to_sram_flag(self, mocker):
        """fetch_constants_to_sram=True reaches the NeutronPartitioner used during export."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, fetch_constants_to_sram=True)

        captured = []
        original_init = NeutronPartitioner.__init__

        def capturing_init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            captured.append(self_)

        mocker.patch.object(NeutronPartitioner, "__init__", capturing_init)
        _run_export(model, rc)

        assert (
            len(captured) == 1
        ), "Expected exactly one NeutronPartitioner to be created."
        spec_map = {s.key: s.value.decode() for s in captured[0].delegation_spec[1]}
        assert spec_map["fetch_constants_to_sram"] == "True"

    def test_use_profiling_flag(self, mocker):
        """use_profiling=True reaches the NeutronPartitioner used during export."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, use_profiling=True)

        captured = []
        original_init = NeutronPartitioner.__init__

        def capturing_init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            captured.append(self_)

        mocker.patch.object(NeutronPartitioner, "__init__", capturing_init)
        _run_export(model, rc)

        assert (
            len(captured) == 1
        ), "Expected exactly one NeutronPartitioner to be created."
        spec_map = {s.key: s.value.decode() for s in captured[0].delegation_spec[1]}
        assert spec_map["use_profiling"] == "True"

    def test_dump_kernel_selection_code(self, tmp_path, monkeypatch):
        """dump_kernel_selection_code=True causes a kernel selection C file to be written."""
        monkeypatch.chdir(tmp_path)
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, dump_kernel_selection_code=True)
        _run_export(model, rc)
        assert (
            tmp_path / "_kernel_selection.c"
        ).exists(), "_kernel_selection.c was not created in the working directory."

    def test_custom_quantizer_fn(self):
        """get_quantizer_fn overrides the default NeutronQuantizer."""
        from executorch.backends.nxp.backend.neutron_target_spec import (
            NeutronTargetSpec,
        )
        from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer

        custom_quantizer_called = []

        def my_quantizer_fn():
            q = NeutronQuantizer(NeutronTargetSpec("imxrt700"))
            custom_quantizer_called.append(True)
            return q

        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, get_quantizer_fn=my_quantizer_fn)
        sess = _run_export(model, rc)
        assert custom_quantizer_called, "Custom quantizer factory was not called."
        assert sess.get_edge_program_manager() is not None

    def test_use_neutron_for_format_conversion_false(self):
        """use_neutron_for_format_conversion=False still produces a valid export."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, use_neutron_for_format_conversion=False)
        sess = _run_export(model, rc)
        assert sess.get_edge_program_manager() is not None

    def test_target_explicit(self):
        """Specifying fake target to make sure an error is raised."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE, target="FAKE")
        with pytest.raises(ValueError, match="`FAKE` is not a valid target"):
            _run_export(model, rc)


class TestInputSpecForms:
    def test__single_tuple(self):
        """input_spec as a plain shape tuple works."""
        model = SimpleCNN()
        sess = _run_export(model, NeutronRecipeConfig((1, 3, 8, 8)))
        assert sess.get_edge_program_manager() is not None

    def test__list_of_tuples(self):
        """input_spec as list of shape tuples works."""
        model = SimpleCNN()
        sess = _run_export(model, NeutronRecipeConfig([(1, 3, 8, 8)]))
        assert sess.get_edge_program_manager() is not None

    def test__model_input_spec(self):
        """input_spec as list of ModelInputSpec objects works."""
        model = SimpleCNN()
        sess = _run_export(model, NeutronRecipeConfig([ModelInputSpec((1, 3, 8, 8))]))
        assert sess.get_edge_program_manager() is not None

    def test__multi_input(self):
        """input_spec with multiple inputs (two tensors) works."""

        class AddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = AddModel()
        rc = NeutronRecipeConfig([(1, 3, 8, 8), (1, 3, 8, 8)])
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )
        example_inputs = [(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))]
        sess = export(model, example_inputs=example_inputs, export_recipe=recipe)
        assert sess.get_edge_program_manager() is not None


class TestErrorHandling:
    def test_create_recipe_missing_config_key(self):
        """create_recipe without neutron_recipe_config kwarg raises KeyError."""
        with pytest.raises(KeyError, match=NEUTRON_RECIPE_CONFIG_KEY):
            NXPRecipeProvider().create_recipe(NXPRecipeType.INT8_PTQ_NEUTRON)

    def test_create_recipe_invalid_recipe_type(self):
        """create_recipe with an unsupported recipe type returns None with a warning."""
        from executorch.export.recipe import RecipeType

        class FakeRecipeType(RecipeType):
            FAKE = "fake"

            @classmethod
            def get_backend_name(cls):
                return "fake_backend"

        rc = NeutronRecipeConfig(INPUT_SHAPE)
        result = NXPRecipeProvider().create_recipe(
            FakeRecipeType.FAKE, neutron_recipe_config=rc
        )
        assert result is None


class TestRecipeStructureValidation:

    def test_ptq_neutron_recipe_structure(self):
        """INT8_PTQ_NEUTRON recipe: correct quantizer and partitioner are set."""
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )
        assert recipe.quantization_recipe is not None
        assert len(recipe.quantization_recipe.quantizers) == 1
        assert recipe.lowering_recipe.partitioners is not None
        assert len(recipe.lowering_recipe.partitioners) == 1

    def test_ptq_neutron_recipe_name(self):
        """INT8_PTQ_NEUTRON recipe has the expected name."""
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )
        assert recipe.name == NXPRecipeType.INT8_PTQ_NEUTRON.value


class TestRecipeCombination:
    def test__chains_pre_partitioning_callbacks(self):
        """Combining two NXP recipes chains both pre_partitioning_callbacks."""
        recipe1 = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON,
            neutron_recipe_config=NeutronRecipeConfig(INPUT_SHAPE),
        )
        recipe2 = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON,
            neutron_recipe_config=NeutronRecipeConfig(INPUT_SHAPE),
        )
        combined = ExportRecipe.combine([recipe1, recipe2])
        assert combined.lowering_recipe.pre_partitioning_callback is not None
        # Calling the combined callback should not raise.
        combined.lowering_recipe.pre_partitioning_callback(None, {})

    def test__collects_post_partitioning_transforms(self):
        """Combining two NXP recipes collects post_partitioning_transforms from both."""
        recipe1 = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON,
            neutron_recipe_config=NeutronRecipeConfig(INPUT_SHAPE),
        )
        recipe2 = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON,
            neutron_recipe_config=NeutronRecipeConfig(INPUT_SHAPE),
        )
        n1 = len(recipe1.lowering_recipe.post_partitioning_transforms)
        n2 = len(recipe2.lowering_recipe.post_partitioning_transforms)
        combined = ExportRecipe.combine([recipe1, recipe2])
        assert len(combined.lowering_recipe.post_partitioning_transforms) == n1 + n2


class TestPostPartitioningTransforms:
    def test__executed(self):
        """post_partitioning_transforms are called after partitioning."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )

        transform_called = []

        def tracking_transform(epm):
            transform_called.append(True)
            return epm

        recipe.lowering_recipe.post_partitioning_transforms = [tracking_transform]

        example_inputs = [(torch.randn(INPUT_SHAPE),)]
        export(model, example_inputs=example_inputs, export_recipe=recipe)
        assert transform_called, "post_partitioning_transforms were not executed."
