# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn

from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    FuseBatchNormWithLinearPass,
)
from executorch.backends.nxp.aten_passes.simulated_linear_bn_fusion_passes import (
    AddSimulatedLinearBatchNormFusionQATPass,
    RemoveSimulatedLinearBatchNormFusionQATPass,
)
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.graph_utils import batch_norm_target_ops
from executorch.backends.nxp.backend.ops_aliases import ExecutorchDelegateCall
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.recipes.nxp_recipe_provider import (
    _histogram_observer_fix_pass,
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
from executorch.backends.nxp.tests.simple_models import ConvBatchNormModule
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    QuantizeFusedConvBnBiasAtenPass,
)
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
    # Skip alloc nodes (e.g. "alloc", "alloc_1") which also have op == "call_function".
    first_call = next(
        n for n in nodes if n.op == "call_function" and not n.name.startswith("alloc")
    )
    last_call = next(n for n in reversed(nodes) if n.op == "call_function")
    assert first_call.target == quantized_decomposed.quantize_per_tensor.out
    assert last_call.target == quantized_decomposed.dequantize_per_tensor.out


class TestInt8PTQNoDelegate:

    def test__basic(self):
        """INT8_PTQ_NO_DELEGATE: model is quantized but no delegate call is present in the graph."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        sess = _run_export(model, rc, recipe_type=NXPRecipeType.INT8_PTQ_NO_DELEGATE)
        graph = _get_graph(sess)

        assert not graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])

        def is_cnn_op(n):
            return any(op in n.name.lower() for op in ["conv", "relu", "view", "add"])

        # With no delegation, original ops should be visible in the graph.
        assert graph_contains_any(graph, is_cnn_op)


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


class TestEdgeManagerTransformPasses:
    def test__executed(self):
        """edge_manager_transform_passes are called after partitioning."""
        model = SimpleCNN()
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )

        transform_called = []

        def tracking_pass(epm):
            transform_called.append(True)
            return []

        recipe.lowering_recipe.edge_manager_transform_passes = [tracking_pass]

        example_inputs = [(torch.randn(INPUT_SHAPE),)]
        export(model, example_inputs=example_inputs, export_recipe=recipe)
        assert transform_called, "edge_manager_transform_passes were not executed."

    def test__qdq_pass_callable_returns_pass_manager(self, mocker):
        """_remove_additional_qdq_clusters returns a bare NeutronEdgePassManager, not a
        list containing one. EdgeProgramManagerTransformStage calls epm.transform(passes)
        directly, so a list-of-PassManager would be silently mis-applied."""
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )
        # remove_quant_io_ops=False (default): first callable is _remove_additional_qdq_clusters.
        qdq_callable = recipe.lowering_recipe.edge_manager_transform_passes[0]
        result = qdq_callable(mocker.MagicMock())
        assert isinstance(result, NeutronEdgePassManager)


# ---------------------------------------------------------------------------
# Helpers shared by QAT tests
# ---------------------------------------------------------------------------


def _noop_train_fn(model: torch.fx.GraphModule) -> None:
    """A no-op train_fn used by structural/unit tests that only inspect pass shape."""
    pass


def _minimal_train_fn(model: torch.fx.GraphModule, shape=(1, 3, 5, 5)) -> None:
    """Run a few SGD steps on random data so fake-quant observer statistics are populated.
    Used by end-to-end tests.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(torch.randn(shape))
        loss = out.sum()
        loss.backward()
        optimizer.step()


def _run_qat_export(model, train_fn=None, recipe_type=NXPRecipeType.INT8_QAT_NEUTRON):
    if train_fn is None:
        train_fn = _noop_train_fn
    rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=train_fn)
    return _run_export(model, rc, recipe_type=recipe_type)


# ---------------------------------------------------------------------------
# QAT recipe: end-to-end tests
# ---------------------------------------------------------------------------


# Both QAT recipe types must reject a missing train_fn.
@pytest.mark.parametrize(
    "recipe_type",
    [NXPRecipeType.INT8_QAT_NEUTRON, NXPRecipeType.INT8_QAT_NO_DELEGATE],
    ids=lambda r: r.value,
)
def test__qat_requires_train_fn(recipe_type):
    """Any QAT recipe raises ValueError when train_fn is absent from NeutronRecipeConfig."""
    rc = NeutronRecipeConfig(INPUT_SHAPE)  # train_fn=None (default)
    with pytest.raises(ValueError, match="train_fn"):
        NXPRecipeProvider().create_recipe(recipe_type, neutron_recipe_config=rc)


# Both _NO_DELEGATE recipe types (PTQ and QAT) must produce an empty partitioner list.
@pytest.mark.parametrize(
    "recipe_type",
    [NXPRecipeType.INT8_PTQ_NO_DELEGATE, NXPRecipeType.INT8_QAT_NO_DELEGATE],
    ids=lambda r: r.value,
)
def test__no_delegate_recipe_has_empty_partitioners(recipe_type):
    """Both NO_DELEGATE recipe types produce an empty partitioner list."""
    # train_fn is required by QAT recipes; PTQ ignores it, so always pass it.
    rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=_noop_train_fn)
    recipe = NXPRecipeProvider().create_recipe(recipe_type, neutron_recipe_config=rc)
    assert recipe.lowering_recipe.partitioners == []


class TestInt8QATNeutron:

    def test__basic(self):
        """INT8_QAT_NEUTRON: full export succeeds and the graph contains a delegate call."""
        model = SimpleCNN()
        sess = _run_qat_export(model)
        graph = _get_graph(sess)
        assert graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])

    def test__train_fn_is_called(self):
        """train_fn is invoked exactly once during the QAT export pipeline."""
        model = SimpleCNN()
        call_count = []

        def counting_train_fn(m):
            call_count.append(1)

        rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=counting_train_fn)
        _run_export(model, rc, recipe_type=NXPRecipeType.INT8_QAT_NEUTRON)
        assert (
            len(call_count) == 1
        ), f"Expected train_fn called once, got {len(call_count)}"

    def test__recipe_name(self):
        """INT8_QAT_NEUTRON recipe has the expected name."""
        rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=_noop_train_fn)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_QAT_NEUTRON, neutron_recipe_config=rc
        )
        assert recipe.name == NXPRecipeType.INT8_QAT_NEUTRON.value

    def test__recipe_structure(self):
        """INT8_QAT_NEUTRON recipe has is_qat=True, one quantizer, one partitioner."""
        rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=_noop_train_fn)
        recipe = NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_QAT_NEUTRON, neutron_recipe_config=rc
        )
        qr = recipe.quantization_recipe
        assert qr is not None
        assert qr.is_qat is True
        assert qr.train_fn is _noop_train_fn
        assert len(qr.quantizers) == 1
        assert recipe.lowering_recipe.partitioners is not None
        assert len(recipe.lowering_recipe.partitioners) == 1

    def test__io_is_quantized_by_default(self):
        """QAT export with default settings: IO boundary has quantize/dequantize ops."""
        model = SimpleCNN()
        sess = _run_qat_export(model)
        graph = _get_graph(sess)
        nodes = list(graph.nodes)
        # Skip alloc nodes (e.g. "alloc", "alloc_1") which also have op == "call_function".
        first_call = next(
            n
            for n in nodes
            if n.op == "call_function" and not n.name.startswith("alloc")
        )
        last_call = next(n for n in reversed(nodes) if n.op == "call_function")
        assert first_call.target == quantized_decomposed.quantize_per_tensor.out
        assert last_call.target == quantized_decomposed.dequantize_per_tensor.out


class TestInt8QATNoDelegate:

    def test__basic(self):
        """INT8_QAT_NO_DELEGATE: export succeeds without any delegate call."""
        model = SimpleCNN()
        sess = _run_qat_export(model, recipe_type=NXPRecipeType.INT8_QAT_NO_DELEGATE)
        graph = _get_graph(sess)
        assert not graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])


# ---------------------------------------------------------------------------
# QAT recipe: NXP-specific pass structure tests
# ---------------------------------------------------------------------------

# Both QAT recipe types are built from the same _build_quantization_recipe(is_qat=True)
# call, so their pass lists must be identical. The parametrization below makes this
# explicit and catches any accidental divergence.
_QAT_RECIPE_TYPES = [NXPRecipeType.INT8_QAT_NEUTRON, NXPRecipeType.INT8_QAT_NO_DELEGATE]


@pytest.mark.parametrize("recipe_type", _QAT_RECIPE_TYPES, ids=lambda r: r.value)
class TestQATNXPPasses:

    def _get_qat_recipe(self, recipe_type: NXPRecipeType) -> "ExportRecipe":
        rc = NeutronRecipeConfig(INPUT_SHAPE, train_fn=_noop_train_fn)
        return NXPRecipeProvider().create_recipe(recipe_type, neutron_recipe_config=rc)

    def test__post_prepare_passes_start_with_add_bn_fusion(self, recipe_type):
        """QAT post_prepare_passes: first pass is AddSimulatedLinearBatchNormFusionQATPass wrapper."""
        recipe = self._get_qat_recipe(recipe_type)
        qr = recipe.quantization_recipe
        assert qr.post_prepare_passes is not None
        # The first post-prepare pass must wrap AddSimulatedLinearBatchNormFusionQATPass.
        # We verify by inspecting the __qualname__ set by _wrap_exir_pass.
        first_pass = qr.post_prepare_passes[0]
        assert (
            AddSimulatedLinearBatchNormFusionQATPass.__name__ in first_pass.__qualname__
        )

    def test__post_prepare_passes_end_with_histogram_observer_fix(self, recipe_type):
        """QAT post_prepare_passes: last pass is _histogram_observer_fix_pass."""
        recipe = self._get_qat_recipe(recipe_type)
        qr = recipe.quantization_recipe
        assert qr.post_prepare_passes is not None
        last_pass = qr.post_prepare_passes[-1]
        assert last_pass is _histogram_observer_fix_pass

    def test__pre_convert_passes_include_remove_bn_fusion_and_fold(self, recipe_type):
        """QAT pre_convert_passes: contains RemoveSimulatedLinearBatchNormFusionQATPass
        followed by FuseBatchNormWithLinearPass (each applied once)."""
        recipe = self._get_qat_recipe(recipe_type)
        qr = recipe.quantization_recipe
        assert qr.pre_convert_passes is not None
        assert len(qr.pre_convert_passes) == 2, (
            "Expected 2 pre_convert passes (remove + fuse), "
            f"got {len(qr.pre_convert_passes)}"
        )
        qualnames = [p.__qualname__ for p in qr.pre_convert_passes]
        assert (
            qualnames[0]
            == f"_wrap_exir_pass({RemoveSimulatedLinearBatchNormFusionQATPass.__name__})"
        )
        assert (
            qualnames[1] == f"_wrap_exir_pass({FuseBatchNormWithLinearPass.__name__})"
        )

    def test__post_convert_passes_include_quant_fused_conv_bn_bias(self, recipe_type):
        """QAT post_convert_passes: contains QuantizeFusedConvBnBiasAtenPass wrapper."""
        recipe = self._get_qat_recipe(recipe_type)
        qr = recipe.quantization_recipe
        assert qr.post_convert_passes is not None
        assert len(qr.post_convert_passes) == 1
        assert f"_wrap_exir_pass({QuantizeFusedConvBnBiasAtenPass.__name__})" in (
            qr.post_convert_passes[0].__qualname__
        )


# ---------------------------------------------------------------------------
# PTQ recipe: pass structure tests (complement to TestQATNXPPasses above)
# ---------------------------------------------------------------------------


class TestPTQNXPPasses:
    """Verifies the pass structure of INT8_PTQ_NEUTRON recipes.

    These tests are separate from TestQATNXPPasses because the assertions are
    PTQ-specific and independent of which QAT recipe type is being tested.
    """

    def _get_ptq_recipe(self) -> "ExportRecipe":
        rc = NeutronRecipeConfig(INPUT_SHAPE)
        return NXPRecipeProvider().create_recipe(
            NXPRecipeType.INT8_PTQ_NEUTRON, neutron_recipe_config=rc
        )

    def test__no_pre_or_post_convert_passes(self):
        """PTQ recipe does not set pre_convert_passes or post_convert_passes."""
        qr = self._get_ptq_recipe().quantization_recipe
        assert qr.pre_convert_passes is None
        assert qr.post_convert_passes is None

    def test__post_prepare_passes_include_histogram_observer_fix(self):
        """PTQ recipe post_prepare_passes contains _histogram_observer_fix_pass."""
        qr = self._get_ptq_recipe().quantization_recipe
        assert qr.post_prepare_passes is not None
        assert _histogram_observer_fix_pass in qr.post_prepare_passes

    def test__post_prepare_passes_do_not_include_add_bn_fusion(self):
        """PTQ recipe post_prepare_passes must NOT contain AddSimulatedLinearBatchNormFusionQATPass."""
        qr = self._get_ptq_recipe().quantization_recipe
        for p in qr.post_prepare_passes or []:
            assert AddSimulatedLinearBatchNormFusionQATPass.__name__ not in getattr(
                p, "__qualname__", ""
            )


# ---------------------------------------------------------------------------
# QAT recipe: e2e test on a real model - equivalent to imperative QAT tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda b: "bias" if b else "no_bias"
)
class TestQATEquivalentToImperative:
    """Recipe-path QAT tests that mirror the imperative-path tests in test_batch_norm_fusion.py.

    The imperative reference is test_biasless_convbn_fusion_qat (and its bias=True variant).
    The recipe path must produce an equivalent result: the graph is delegated and the
    BN is fully fused away by the QAT passes.
    """

    # Use (1, 3, 5, 5) so that Conv2d(kernel_size=3) produces a (1, 3, 3, 3) feature map,
    # giving BatchNorm > 1 value per channel in training mode (QAT requires train mode).
    _CONVBN_INPUT_SHAPE = (1, 3, 5, 5)

    def test__convbn_qat_produces_delegate_call(self, bias):
        """INT8_QAT_NEUTRON on ConvBatchNormModule produces a delegate call.
        Equivalent imperative test: test_biasless_convbn_fusion_qat / test_batch_norm_conv_fusing
        in backends/nxp/tests/generic_tests/test_batch_norm_fusion.py."""
        model = ConvBatchNormModule(
            bias=bias,
            input_rank=len(self._CONVBN_INPUT_SHAPE),
            num_features=self._CONVBN_INPUT_SHAPE[1],
        )
        rc = NeutronRecipeConfig(
            self._CONVBN_INPUT_SHAPE,
            train_fn=_minimal_train_fn,
            use_neutron_for_format_conversion=False,
        )
        sess = _run_export(
            model,
            rc,
            recipe_type=NXPRecipeType.INT8_QAT_NEUTRON,
            input_shape=self._CONVBN_INPUT_SHAPE,
        )
        graph = _get_graph(sess)

        # Same assertion as the imperative path: the model is delegated.
        assert graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])

    def test__convbn_qat_bn_is_fused_away(self, bias):
        """INT8_QAT_NEUTRON on ConvBatchNormModule: BN is fused away by QAT passes.
        Equivalent imperative test: test_batch_norm_conv_fusing__full_pipeline__2d
        in backends/nxp/tests/generic_tests/test_batch_norm_fusion.py."""
        model = ConvBatchNormModule(
            bias=bias,
            input_rank=len(self._CONVBN_INPUT_SHAPE),
            num_features=self._CONVBN_INPUT_SHAPE[1],
        )
        rc = NeutronRecipeConfig(
            self._CONVBN_INPUT_SHAPE,
            train_fn=_minimal_train_fn,
            use_neutron_for_format_conversion=False,
        )
        sess = _run_export(
            model,
            rc,
            recipe_type=NXPRecipeType.INT8_QAT_NEUTRON,
            input_shape=self._CONVBN_INPUT_SHAPE,
        )
        # The edge program (before delegation) must not contain any BN ops.
        edge_graph = sess.get_edge_program_manager().exported_program().graph
        assert not graph_contains_any_of_ops(edge_graph, batch_norm_target_ops)
