import contextlib
import logging
import operator
import os
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock

import torch
from executorch.backends.qualcomm._passes import (
    AnnotateQuantAttrs,
    ConvertBmmToMatmul,
    ConvertFullLikeToFull,
    ConvertMhaToSha,
    DecomposeColIm,
    DecomposeWrapWithAutocast,
    ExpandBroadcastTensorShape,
    FoldQDQ,
    I64toI32,
    InsertIOQDQ,
    InsertReshapeForReduceOps,
    LayoutTransform,
    LiftConstantScalarOperands,
    RecomposeRmsNorm,
    Remove0DTensor,
    RemoveRedundancy,
    ResolveDebugHandle,
    TagPositionalSinCos,
)
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    to_dq_op,
    to_q_op,
)
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.op_custom_op import (
    _resolve_qnn_data_type,
    CustomOp,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.partition.qnn_partitioner import (
    QnnOperatorSupport,
    QnnPartitioner,
)
from executorch.backends.qualcomm.partition.utils import (
    filter_fn,
    get_skip_decomp_table,
)
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.recipes.qnn_recipe_provider import QNNRecipeProvider
from executorch.backends.qualcomm.recipes.qnn_recipe_types import QNNRecipeType
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchOpPackageInfo,
)
from executorch.backends.qualcomm.tests.models import (
    BroadcastAndMutate,
    CastToFloat,
    ConvBackward,
    FullLikeShapeOnly,
    HardSigmoid,
    Reciprocal,
    ScaledRmsNorm,
    TopKandIndex,
    Unfold,
    UpcastRmsNorm,
)
from executorch.backends.qualcomm.tests.tester import ToEdgeTransformAndLower
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_FALLBACK_NODE,
    QCOM_INSERTED_CAST,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import QnnManagerRegistry
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    qnn_edge_config,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.lowered_backend_module import get_lowered_backend_modules
from executorch.exir.program._program import _replace_aten_ops_with_transformed_ops
from torch.export.exported_program import OutputKind
from torch.export.graph_signature import ExportGraphSignature
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.library import Library
from torchao.quantization import IntxWeightOnlyConfig, PerAxis, quantize_
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class ConstantBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(4, 16))
        self.register_buffer("weight", torch.randn(16, 16))
        self.register_buffer("mask", torch.rand(4, 16) > 0.5)

    def forward(self, x):
        linear = torch.nn.functional.linear
        # QNN has no bfloat16 type, and a masked select has no static shape,
        # so these branches read nothing at runtime once those ops run on
        # CPU. Their linears are decomposed.
        cast = linear(self.table, self.weight).to(torch.bfloat16).float()
        masked = linear(self.table, self.weight)[self.mask].sum()
        # This linear feeds a delegated add both directly and through CPU
        # casts, so the partitioner splits it off to avoid a cycle. It is
        # decomposed.
        detour = linear(self.table, self.weight)
        detour = detour + detour.to(torch.bfloat16).float()
        # This linear reads only constants too, but it shares a partition with
        # the add that reads x, so it stays whole, even though that add
        # reaches another QNN op through CPU casts.
        delegated = linear(self.table, self.weight)
        y = x + delegated + cast + masked + detour
        return y.to(torch.bfloat16).float() + 1


class MaskedReader(ConstantBranches):
    def forward(self, x):
        # Nothing after the masked select reaches QNN: the sum reads a
        # data-dependent size, and the runtime asserts that export adds for
        # that size have no tensor output. So the linear shares the add's
        # partition and stays whole.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + linear, linear[self.mask].sum()


class SkippedRelu(ConstantBranches):
    def forward(self, x):
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + linear.relu()


class QuantizedCast(ConstantBranches):
    def forward(self, x):
        # A quantized linear stays whole. Quantized casts get a QNN type, so
        # here the whole model is delegated.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + linear.to(torch.bfloat16).float()


class NoRuntimeInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(4, 16))
        self.register_buffer("weight", torch.randn(16, 16))

    def forward(self, x):
        # x only reaches a CPU cast, so no partition reads a runtime value, and
        # the linear is decomposed.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x.to(torch.bfloat16), linear * 2


class CastSplit(NoRuntimeInput):
    def forward(self, x):
        # QNN takes every getitem, so the getitems of this CPU split can join
        # the add's partition. The linear would then close a cycle through the
        # split, so it is left out and decomposed.
        linear = torch.nn.functional.linear(self.table, self.weight)
        low, high = linear.to(torch.bfloat16).split(8, dim=-1)
        return x + linear, low, high


class ConstantOutput(NoRuntimeInput):
    def forward(self, x):
        # No path links this linear to the ops that read x, but the partitioner
        # also merges such ops, so the linear stays whole. The getitems of the
        # split stay with it, since QNN takes it.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + 1, *x.split(8, dim=-1), linear


class CastSplitInput(NoRuntimeInput):
    def forward(self, x):
        # Only the getitems of a CPU split read x, and the partitioner moves
        # them out of the linear's partition, so it reads no runtime value.
        # The linear is decomposed.
        low, high = x.to(torch.bfloat16).split(8, dim=-1)
        return low, high, torch.nn.functional.linear(self.table, self.weight)


class SplitAfterAdd(NoRuntimeInput):
    def forward(self, x):
        # These getitems come after a QNN op, so their partition cannot merge
        # with the add's. The linear joins theirs, which reads no runtime value
        # once they are moved out, so it is decomposed.
        low, high = (x + 1).to(torch.bfloat16).split(8, dim=-1)
        return low, high, torch.nn.functional.linear(self.table, self.weight)


class CastSplitLinked(NoRuntimeInput):
    def forward(self, x):
        # As above, but the linear feeds the add, so it stays whole.
        low, high = (x + 1).to(torch.bfloat16).split(8, dim=-1)
        linear = torch.nn.functional.linear(self.table, self.weight)
        return low, high, x + linear


class LinearOfInput(NoRuntimeInput):
    def forward(self, x):
        # This linear reads x through the add, so it stays whole even though it
        # reaches a QNN op through CPU casts.
        linear = torch.nn.functional.linear(x + 1, self.weight)
        return linear + linear.to(torch.bfloat16).float()


class ReluBeforeCast(NoRuntimeInput):
    def forward(self, x):
        # The linear reaches the CPU cast through a relu that reads no runtime
        # value either, so it is cut off and decomposed.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + linear.relu().to(torch.bfloat16).float()


class CondAfterLinear(NoRuntimeInput):
    def forward(self, x):
        # QNN has no node visitor for cond, so it runs on CPU and cuts the
        # linear off from the add. The linear is decomposed.
        linear = torch.nn.functional.linear(self.table, self.weight)
        branch = torch.cond(x.sum() > 0, torch.neg, torch.abs, (linear,))
        return x + branch


class ItemAfterLinear(NoRuntimeInput):
    def forward(self, x):
        # item has no tensor output, so it runs on CPU, and the partition of
        # the linear and the sum reads no runtime value. The linear is
        # decomposed.
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x * linear.sum().item()


class IntStackBesideLinear(NoRuntimeInput):
    def forward(self, x):
        # Both stay on CPU, but filter_fn already decomposes an int64 stack.
        linear = torch.nn.functional.linear(self.table, self.weight)
        indices = torch.stack([self.table.long(), self.table.long()])
        return x.to(torch.bfloat16), linear * 2, indices


class EluBesideLinear(NoRuntimeInput):
    def forward(self, x):
        # Both stay on CPU, but exir has no decomposition for elu.
        linear = torch.nn.functional.linear(self.table, self.weight)
        elu = torch.nn.functional.elu(self.table)
        return x.to(torch.bfloat16), linear * 2, elu * 2


class SharedConstant(NoRuntimeInput):
    def forward(self, x):
        # Like CastSplitLinked, but the linear reads a value that the add
        # reads.
        low, high = (x + 1).to(torch.bfloat16).split(8, dim=-1)
        table = self.table * 2
        linear = torch.nn.functional.linear(table, self.weight)
        return low, high, x + table, linear


class UnlinkedSplit(NoRuntimeInput):
    def forward(self, x):
        # No QNN op comes before this split, so its getitems cannot keep the
        # linear out of the add's partition. It stays whole.
        low, high = x.to(torch.bfloat16).split(8, dim=-1)
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + 1, low, high, linear


class MutatedTable(NoRuntimeInput):
    def forward(self, x):
        # The table changes on every run, so the linear reads a runtime value
        # and stays whole, although only a CPU cast reads it.
        self.table.mul_(2)
        linear = torch.nn.functional.linear(self.table, self.weight)
        return x + linear.to(torch.bfloat16).float()


class QuantizedWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(4, 16))
        self.linear = torch.nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return x + self.linear(self.table).to(torch.bfloat16).float()


class GroupQuantizedWeight(NoRuntimeInput):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.full((16, 2), 0.1))
        self.register_buffer("zero_points", torch.zeros(16, 2, dtype=torch.int8))

    def forward(self, x):
        group = (self.scales, self.zero_points, -8, 7, torch.int8, 8)
        weight = torch.ops.quantized_decomposed.quantize_per_channel_group(
            self.weight, *group
        )
        weight = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            weight, *group, torch.float32
        )
        linear = torch.nn.functional.linear(self.table, weight)
        return x + linear.to(torch.bfloat16).float()


@contextlib.contextmanager
def stubbed_qnn_backend():
    # Lowers without the QNN SDK: the backend accepts every op and compiles
    # nothing.
    with patch("executorch.backends.qualcomm.utils.utils.setup_qnn_sdk"), patch.object(
        QnnManagerRegistry, "get_or_create_qnn_manager", return_value=MagicMock()
    ), patch.object(
        QnnBackend,
        "preprocess_multimethod",
        side_effect=lambda programs, _: {
            method: [PreprocessResult(processed_bytes=b"qnn")] * len(partitions)
            for method, partitions in programs.items()
        },
    ):
        yield


# Read at import, because lowering can download the SDK and set QNN_SDK_ROOT
# for this process, too late for the loader to find the SDK libraries.
QNN_SDK_SET_UP = "QNN_SDK_ROOT" in os.environ


def skip_without_qnn_sdk(test, reason):
    # With the SDK set up the backend has to load, so a failure to load is a
    # real error that the caller reports, not a reason to skip.
    if not QNN_SDK_SET_UP:
        test.skipTest(f"QNN SDK not available: {reason}")


@contextlib.contextmanager
def stubbed_recipe_setup():
    # Builds QNN recipes without setting up the QNN SDK or changing torch
    # flags for the other tests.
    provider = "executorch.backends.qualcomm.recipes.qnn_recipe_provider"
    with patch(f"{provider}.setup_qnn_sdk"), patch(
        f"{provider}.disable_mkldnn_on_amd"
    ), patch("executorch.backends.qualcomm.utils.utils.setup_qnn_sdk"):
        yield


def stubbed_compiler_specs(backend_options):
    with patch("executorch.backends.qualcomm.utils.utils.setup_qnn_sdk"):
        return generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650, backend_options=backend_options
        )


def delegated_ops(edge):
    return [
        node.target
        for lowered in get_lowered_backend_modules(edge.exported_program().graph_module)
        for node in lowered.original_module.graph.nodes
    ]


class TestPasses(unittest.TestCase):
    def _build_context_loader_edge_program(self, op_name, check_ir_validity=True):
        graph_name = "forward"
        custom_op = Library(OpContextLoader.namespace, "FRAGMENT")
        self.addCleanup(custom_op._destroy)
        custom_op.define(f"{op_name}(Tensor[] inputs) -> Any")

        @torch.library.impl(
            custom_op, op_name, dispatch_key="CompositeExplicitAutograd"
        )
        def op_impl(inputs):
            return (torch.zeros((1, 2), device="meta", dtype=inputs[0].dtype),)

        class Model(torch.nn.Module):
            def forward(self, x):
                return getattr(
                    getattr(torch.ops, OpContextLoader.namespace), op_name
                ).default((x,))

        exported_program = torch.export.export(
            Model(), (torch.ones(1, 2),), strict=True
        )
        compile_config = (
            None if check_ir_validity else EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_program_manager = to_edge(
            {graph_name: exported_program},
            compile_config=compile_config,
        )
        edge_program = edge_program_manager._edge_programs[graph_name]
        context_loader_nodes = [
            node
            for node in edge_program.graph.nodes
            if node.op == "call_function"
            and OpContextLoader.namespace in str(node.target)
        ]
        return edge_program, context_loader_nodes

    def test_context_loader_edge_op_is_delegated(self):
        op_name = "ctx_loader_delegation"
        ctx_bin = b"qnn_context_binary"
        _, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=False
        )
        self.assertEqual(1, len(context_loader_nodes))
        context_loader_nodes[0].meta[OpContextLoader.meta_ctx_bin] = ctx_bin

        # A fully constructed QnnOperatorSupport needs a live QNN manager, so
        # drive is_node_supported with a mock self: the context-loader node is
        # force-passed without touching instance state beyond the log label.
        support = MagicMock()
        support.phase = "QnnPartitioner"
        self.assertTrue(
            QnnOperatorSupport.is_node_supported(support, None, context_loader_nodes[0])
        )

    def test_lift_constant_scalar_operands_skips_higher_order_ops(self):
        # A `with torch.no_grad()` block is captured as a higher-order op
        # (wrap_with_set_grad_enabled) whose target carries no schema.
        # LiftConstantScalarOperands reads node.target._schema, so it must skip
        # such nodes instead of crashing with AttributeError.
        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.no_grad():
                    y = x + 1
                return y * 2

        gm = torch.export.export(Model(), (torch.randn(3),), strict=True).graph_module
        schemaless = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and not hasattr(node.target, "_schema")
        ]
        self.assertTrue(schemaless, "expected a higher-order op in the graph")

        # Must not raise.
        LiftConstantScalarOperands().call(gm)

    def test_decompose_wrap_with_autocast_nested_regions(self):
        # A region that reads inputs and parameters names its placeholders
        # like arg2_1, arg0_1, arg1_1, not after the operands passed to it.
        # Both nested submodules are named submod_1, each in its own module.
        # The second linear runs in bfloat16 only because of the outer region.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    with torch.autocast("cpu", enabled=False):
                        y = self.linear(x)
                    z = self.linear(y * 2)
                return y + 1, z

        # A disabled region is copied, not retraced, so its nested region
        # is inlined on its own.
        class DisabledOuterModel(Model):
            def forward(self, x):
                with torch.autocast("cpu", enabled=False):
                    y = x * 2
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        z = self.linear(y)
                return y + 1, z

        x = torch.randn(4, 16)
        for model in (Model().eval(), DisabledOuterModel().eval()):
            with self.subTest(model=type(model).__name__):
                gm = torch.export.export(model, (x,)).module()

                DecomposeWrapWithAutocast()(gm)

                self.assertFalse(
                    any(
                        isinstance(
                            n.target, torch._higher_order_ops.wrap.WrapWithAutocast
                        )
                        for n in gm.graph.nodes
                    )
                )
                torch.testing.assert_close(gm(x), model(x))

    def test_export_pipeline_keeps_node_names_unique(self):
        # torch.export leaves the input name t_1 unrecorded, and renames
        # input * 2 to mul_1 to make room for the input mul without recording
        # that either. The decompositions would otherwise reuse those names.
        class Autocast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(8, 8))

            def forward(self, x, t_1):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    # The second transpose is named t_1 inside the region, and
                    # the decomposition copies it next to the input t_1.
                    y = x @ self.w.t() @ self.w.t()
                return y.float() + t_1

        class Attention(torch.nn.Module):
            # input is a Python builtin name. It has to survive the reservation
            # as it is, because the graph signature refers to inputs by name.
            def forward(self, input, k, v, mul):
                return (
                    torch.nn.functional.scaled_dot_product_attention(input * 2, k, v)
                    + mul
                )

        cases = (
            (
                Autocast(),
                (torch.randn(4, 8), torch.randn(4, 8)),
                torch._higher_order_ops.wrap.wrap_with_autocast,
            ),
            (
                Attention(),
                tuple(torch.randn(1, 2, 4, 8) for _ in range(4)),
                torch.ops.aten.scaled_dot_product_attention.default,
            ),
        )
        no_collision = []
        for model, inputs, decomposed_op in cases:
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model.eval(), inputs)
                self.assertIn(decomposed_op, [n.target for n in ep.graph.nodes])
                ep = get_qnn_pass_manager_cls()().transform_for_export_pipeline(ep)
                self.assertNotIn(decomposed_op, [n.target for n in ep.graph.nodes])
                # ep.module() copies the graph and renames duplicates, so lint
                # the graph the passes produced.
                ep.graph.lint()
                self.assertEqual(
                    [n.name for n in ep.graph.nodes if n.op == "placeholder"],
                    [spec.arg.name for spec in ep.graph_signature.input_specs],
                )
                torch.testing.assert_close(ep.module()(*inputs), model(*inputs))
                ep = torch.export.export(model.eval(), inputs)
                with patch(
                    "executorch.backends.qualcomm._passes.qnn_pass_manager."
                    "reserve_node_names"
                ) as reserve:
                    try:
                        get_qnn_pass_manager_cls()().transform_for_export_pipeline(
                            ep
                        ).graph.lint()
                    except Exception as e:
                        # The pass manager wraps the lint error in its own.
                        if "Node redefined name" not in str(e.__cause__ or e):
                            raise
                    else:
                        no_collision.append(type(model).__name__)
                    reserve.assert_called_once()
        # A skip, not a failure: a torch update that removes the collision is
        # fine. While any case still collides, the test still means something.
        message = (
            f"{no_collision} no longer collide without reserve_node_names, so "
            "those cases no longer test it; see "
            "https://github.com/pytorch/pytorch/issues/202927"
        )
        if len(no_collision) == len(cases):
            self.skipTest(message)
        if no_collision:
            logging.warning(message)

    def test_build_op_wrappers_returns_context_binary(self):
        op_name = "ctx_loader_build"
        ctx_bin = b"qnn_context_binary"
        edge_program, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=False
        )
        for node in context_loader_nodes:
            node.meta[OpContextLoader.meta_ctx_bin] = ctx_bin

        # For a graph whose only op is the context-binary loader, _build_op_wrappers
        # returns the stamped context binary directly, before any QNN compilation.
        result = QnnBackend._build_op_wrappers(
            edge_program,
            enable_tensor_dump=False,
            op_package_infos=[],
            use_mha2sha=False,
            backend_type=QnnExecuTorchBackendType.kHtpBackend,
        )
        self.assertEqual(ctx_bin, result)

    def test_context_loader_op_lowers_with_ir_validation(self):
        op_name = "ctx_loader_validation"

        # from_context_binary lowers with IR validity checks enabled (the
        # default). The context-loader custom op survives the edge verifier
        # because its namespace is not aten, so no validation is disabled and
        # the loader node is still present for downstream stamping.
        _, context_loader_nodes = self._build_context_loader_edge_program(
            op_name, check_ir_validity=True
        )
        self.assertEqual(1, len(context_loader_nodes))

    def test_skip_decomp_table_preserves_conv_backward(self):
        self.assertIn(
            torch.ops.aten.convolution_backward.default,
            get_skip_decomp_table(),
        )

    def test_conv_backward_composite_lowering(self):
        x = torch.randn(1, 3, 8, 8)
        grad_output = torch.randn(1, 4, 4, 4)
        edge_program = to_edge(
            torch.export.export(
                ConvBackward(flatten_grad_input=True, duplicate_backward=True),
                (grad_output, x),
                strict=True,
            )
        ).exported_program()
        graph_module = LayoutTransform(edge_program, insert_permute=True)(
            edge_program.graph_module
        ).graph_module

        conv_backwards = [
            node
            for node in graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.convolution_backward.default
        ]
        self.assertEqual(2, len(conv_backwards))
        conv_backward = conv_backwards[0]
        getitem = next(node for node in conv_backward.users if node.args[1] == 0)
        permutes = [
            node
            for node in graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.permute_copy.default
        ]
        self.assertTrue(any(node.args[0] is getitem for node in permutes))
        self.assertFalse(any(node.args[0] is conv_backward for node in permutes))

        forward_conv = next(
            node
            for node in graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.convolution.default
        )
        weight_node = forward_conv.args[1]
        nodes_to_wrappers = defaultdict(dict)
        visitors = get_node_visitors(edge_program)
        visitors["aten.convolution.default"].define_node(
            forward_conv, nodes_to_wrappers
        )
        forward_weight_wrapper = nodes_to_wrappers[weight_node.name][0]
        first_ops = visitors["aten.convolution_backward.default"].define_node(
            conv_backwards[0], nodes_to_wrappers
        )
        first_hwoi_wrapper = nodes_to_wrappers[f"{weight_node.name}_hwoi"][0]
        second_ops = visitors["aten.convolution_backward.default"].define_node(
            conv_backwards[1], nodes_to_wrappers
        )
        second_hwoi_wrapper = nodes_to_wrappers[f"{weight_node.name}_hwoi"][0]

        self.assertEqual(2, len(first_ops))
        self.assertEqual(1, len(second_ops))
        self.assertIs(forward_weight_wrapper, nodes_to_wrappers[weight_node.name][0])
        self.assertIn(f"{weight_node.name}_hwoi", nodes_to_wrappers)
        self.assertIs(first_hwoi_wrapper, second_hwoi_wrapper)

    def test_conv_backward_quantized_inputs_fallback(self):
        x = torch.randn(1, 3, 8, 8)
        grad_output = torch.randn(1, 4, 4, 4)
        edge_program = to_edge(
            torch.export.export(ConvBackward(), (grad_output, x), strict=True)
        ).exported_program()
        conv_backward = next(
            node
            for node in edge_program.graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.convolution_backward.default
        )
        visitor = get_node_visitors(edge_program)["aten.convolution_backward.default"]

        for input_index in (0, 2):
            with self.subTest(input_index=input_index):
                conv_backward.args[input_index].meta[QCOM_QUANT_ATTRS] = {}
                self.assertIsNone(visitor.define_node(conv_backward, defaultdict(dict)))
                conv_backward.args[input_index].meta.pop(QCOM_QUANT_ATTRS)

    def test_layout_transform_tuple_outputs(self):
        with self.subTest("multiple_getitems"):
            x = torch.randn(1, 3, 8, 8)
            grad_output = torch.randn(1, 4, 4, 4)
            edge_program = to_edge(
                torch.export.export(
                    ConvBackward(flatten_grad_input=True, return_grad_weight=True),
                    (grad_output, x),
                    strict=True,
                )
            ).exported_program()
            conv_backward = next(
                node
                for node in edge_program.graph_module.graph.nodes
                if node.target == exir_ops.edge.aten.convolution_backward.default
            )
            getitems = [
                node
                for node in conv_backward.users
                if node.target.__name__ == "getitem" and node.args[1] in (0, 1)
            ]
            # Synthetic metadata exercises LayoutTransform only; quantized
            # ConvBackward lowering remains unsupported.
            for index, getitem in enumerate(getitems):
                getitem.meta[QCOM_QUANT_ATTRS] = {"index": index}

            graph_module = LayoutTransform(edge_program, insert_permute=True)(
                edge_program.graph_module
            ).graph_module
            self.assertEqual(2, len(getitems))
            permutes_by_input = {
                node.args[0]: node
                for node in graph_module.graph.nodes
                if node.target == exir_ops.edge.aten.permute_copy.default
            }
            self.assertTrue(all(getitem in permutes_by_input for getitem in getitems))
            for getitem in getitems:
                self.assertEqual(
                    getitem.meta[QCOM_QUANT_ATTRS],
                    permutes_by_input[getitem].meta[QCOM_QUANT_ATTRS],
                )

        with self.subTest("none_output"):
            graph = torch.fx.Graph()
            conv_backward = graph.call_function(
                torch.ops.aten.convolution_backward.default,
                (),
            )
            conv_backward.meta["val"] = (torch.empty(1, 3, 4, 4), None, None)
            graph.call_function(operator.getitem, (conv_backward, 2))
            graph.call_function(operator.getitem, (conv_backward, 0))

            layout_transform = LayoutTransform(MagicMock())
            layout_transform.mark_as_transformed(conv_backward)

            self.assertEqual(
                (0, 2, 3, 1),
                conv_backward.meta[layout_transform.transformed_tag],
            )

        with self.subTest("invalid_tuple_user"):
            graph = torch.fx.Graph()
            tuple_node = graph.call_function(
                torch.ops.aten.convolution_backward.default,
                (),
            )
            tuple_node.meta["val"] = (torch.empty(1, 3, 4, 4),)
            getitem_node = graph.call_function(operator.getitem, (tuple_node, 0))
            graph.call_function(torch.ops.aten.alias.default, (tuple_node,))
            graph.output(getitem_node)
            graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
            layout_transform = LayoutTransform(MagicMock(), insert_permute=True)
            original_nodes = list(graph_module.graph.nodes)

            with self.assertRaisesRegex(AssertionError, "Expected tuple-output node"):
                layout_transform.insert_node(
                    graph_module, tuple_node, revert_layout=False
                )

            self.assertEqual(original_nodes, list(graph_module.graph.nodes))

    def _build_quantized_graph(self):
        """Build a quantized graph through AnnotateQuantAttrs + FoldQDQ."""

        class AddModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        module = AddModule().eval()
        sample_input = (torch.randn(1, 4),)

        exported = torch.export.export(module, sample_input, strict=True).module()
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*sample_input)
        qdq_module = convert_pt2e(prepared)

        edge_program = to_edge(
            torch.export.export(qdq_module, sample_input, strict=True)
        )
        ep = edge_program.exported_program()
        gm = ep.graph_module

        gm = AnnotateQuantAttrs(ep)(gm).graph_module
        gm = FoldQDQ(ep)(gm).graph_module
        return gm, ep

    def test_insert_io_qdq_handles_dequant_encoding(self):
        """InsertIOQDQ should not KeyError when a node with a dequantize
        encoding feeds the output node (e.g. pre-quantized LLM parameters)."""
        gm, ep = self._build_quantized_graph()

        # Wire b__frozen_param0 (which has dequantize encoding) to output,
        # simulating the LLM topology from github issue #17732.
        param_node = None
        output_node = None
        for n in gm.graph.nodes:
            if n.name == "b__frozen_param0":
                param_node = n
            if n.op == "output":
                output_node = n

        self.assertIsNotNone(param_node)
        old_args = output_node.args[0]
        output_node.args = (
            ((old_args,) if not isinstance(old_args, tuple) else old_args)
            + (param_node,),
        )
        gm.graph.lint()
        gm.recompile()

        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)

        dq_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and hasattr(n.target, "__name__")
            and "dequantize" in n.target.__name__
            and any(u.op == "output" for u in n.users.keys())
        ]
        self.assertGreaterEqual(len(dq_nodes), 1)

    def test_insert_io_qdq_no_revisit(self):
        """InsertIOQDQ must not revisit newly inserted nodes."""
        gm, ep = self._build_quantized_graph()

        node_count_before = len(list(gm.graph.nodes))
        pass_instance = InsertIOQDQ(ep)
        pass_instance._insert(gm)
        node_count_after = len(list(gm.graph.nodes))

        # AddModule with one input and one output should insert exactly
        # one quantize (input) and one dequantize (output) = +2 nodes.
        self.assertEqual(node_count_after, node_count_before + 2)

    def test_q_dq_map_pins_per_channel_group_pairs(self):
        """to_q_op / to_dq_op must map per_channel_group q<->dq to each other.

        ``to_dq_op`` short-circuits for targets already in ``dq_ops``, so the
        map is only consulted when converting across the pair; a wrong entry
        here would surface later as an assertion in ``insert_quant_node``.
        """
        quantize_per_channel_group = (
            exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default
        )
        dequantize_per_channel_group = (
            exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default
        )
        self.assertIs(to_q_op(dequantize_per_channel_group), quantize_per_channel_group)
        self.assertIs(
            to_dq_op(quantize_per_channel_group), dequantize_per_channel_group
        )
        self.assertIs(to_q_op(quantize_per_channel_group), quantize_per_channel_group)
        self.assertIs(
            to_dq_op(dequantize_per_channel_group), dequantize_per_channel_group
        )

    def test_insert_io_qdq_per_channel_group_dequantizes_output(self):
        """InsertIOQDQ must insert a dequantize_per_channel_group before output.

        A pre-quantized group-wise (e.g. int4 LLM) weight parameter that feeds
        the graph output takes the dequantize-before-output branch. The
        insert-quantize-after-input branch is skipped for parameters, so the
        output branch is the only place the encoding is resolved.
        """
        graph_module, exported_program = self._build_quantized_graph()

        parameter_node = next(
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder" and is_parameter(node, exported_program)
        )
        output_node = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        )

        # Annotate the parameter as a group-wise quantized weight.
        scales = torch.ones(4, 1)
        parameter_node.meta[QCOM_QUANT_ATTRS] = {
            QCOM_ENCODING: exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default,
            QCOM_SCALE: scales,
            "scales": scales,
            "zero_points": None,
            "quant_min": -8,
            "quant_max": 7,
            "dtype": torch.int8,
            "group_size": 1,
            "output_dtype": torch.float32,
        }

        existing_outputs = output_node.args[0]
        if not isinstance(existing_outputs, tuple):
            existing_outputs = (existing_outputs,)
        output_node.args = (existing_outputs + (parameter_node,),)
        graph_module.graph.lint()
        graph_module.recompile()

        InsertIOQDQ(exported_program)._insert(graph_module)

        dequantize_target = (
            exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default
        )
        dequantize_nodes_feeding_output = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == dequantize_target
            and any(user.op == "output" for user in node.users.keys())
        ]
        self.assertEqual(len(dequantize_nodes_feeding_output), 1)

    def test_make_qnn_per_block_config_rejects_asymmetric(self):
        """QNN blockwise expansion is symmetric-only; non-zero zero_points must
        raise instead of silently lowering to a symmetric encoding."""
        graph_module, exported_program = self._build_quantized_graph()
        parameter_node = next(
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder" and is_parameter(node, exported_program)
        )

        visitor = NodeVisitor({}, exported_program, enable_tensor_dump=False)
        quant_attrs = {
            QCOM_ENCODING: exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default,
            QCOM_ZERO_POINTS: torch.tensor([0, 1, 0, 0]),
        }
        with self.assertRaisesRegex(ValueError, "symmetric"):
            visitor.make_qnn_per_block_config(parameter_node, quant_attrs)

    def test_insert_reshape_for_argmax(self):
        class ArgmaxModule(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=None)

        mod = ArgmaxModule()

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        ep = torch.export.export(mod, (x,))
        # Run original module for reference
        ref = mod(x)

        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 0, "Reshape node not inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        InsertReshapeForReduceOps()(ep.graph_module)

        out = ep.graph_module(x)

        # Check graph structure: argmax should take a reshape as input
        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 1, "Reshape node should be inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        argmax_node = argmax_nodes[0]
        self.assertEqual(argmax_node.args[1], 0, "Argmax dim not set to 0")

        # Execute new graph and compare with reference
        out = ep.graph_module(x)
        self.assertTrue(
            torch.equal(*out, ref), f"Output mismatch: got {out}, expected {ref}"
        )

    def test_remove_0d_tensor_keeps_select_it_cannot_replace(self):
        class TableLookup(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(4.0))

            def forward(self, x):
                return x * self.table[2]

        class StackSelected(torch.nn.Module):
            def forward(self, x):
                return torch.stack([x[0], x[0]])

        class CastSelected(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor([3]))

            def forward(self, x):
                return x * self.scale[0]

        # A longer input is not the selected element, stacking the one-element
        # input instead of the 0-d element changes the shape, and a cast keeps
        # the 0-d layout in its arguments.
        for model, x in (
            (TableLookup(), torch.randn(4)),
            (StackSelected(), torch.randn(1)),
            (CastSelected(), torch.randn(4)),
        ):
            with self.subTest(model=type(model).__name__):
                ep = to_edge(torch.export.export(model, (x,), strict=True))
                ep = ep.exported_program()
                Remove0DTensor()(ep.graph_module)
                self.assertTrue(
                    any(
                        n.target == exir_ops.edge.aten.select_copy.int
                        for n in ep.graph.nodes
                    )
                )
                self.assertTrue(torch.equal(ep.module()(x), model(x)))

    def test_recompose_rms_norm_keeps_the_norm_dtype(self):
        class UpcastNorm(torch.nn.Module):  # transformers' Qwen2RMSNorm
            def __init__(self, dtype, compute_dtype=torch.float32):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(32, dtype=dtype))
                self.compute_dtype = compute_dtype

            def upcast(self, x):
                return x.to(self.compute_dtype)

            def forward(self, x):
                h = self.upcast(x)
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                return self.weight * h.to(x.dtype)

        class TypeAsNorm(torch.nn.Module):  # type_as, to.device, a square of casts
            def __init__(self, dtype):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(32, dtype=dtype))

            def forward(self, x):
                square = x.float() * x.float()
                rms = torch.rsqrt(square.mean(-1, keepdim=True) + 1e-6)
                h = x.to(x.device, torch.float32) * rms
                return h.type_as(x) * self.weight

        class T5Norm(UpcastNorm):  # transformers' T5LayerNorm, half weights
            def forward(self, x):
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + 1e-6)
                return self.weight * x.to(self.weight.dtype)

        class ContiguousUpcastNorm(UpcastNorm):
            def upcast(self, x):
                return x.to(self.compute_dtype, memory_format=torch.contiguous_format)

        class RoundTripNorm(UpcastNorm):
            def forward(self, x):
                return super().forward(x.half().float())

        class ShiftedNorm(UpcastNorm):
            def __init__(self, dtype):
                super().__init__(dtype)
                self.register_buffer("shift", torch.randn(32, dtype=dtype))

            def forward(self, x):
                h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return self.weight * (h + self.shift)

        # The norm is rebuilt on the nearest input in the output dtype, past
        # casts that keep it or widen it to float32 at most. When the whole norm
        # cannot be rebuilt, the HTP float16 specs that edge rows pass rebuild
        # its float32 normalization alone. A norm about to be quantized is
        # rebuilt whole or not at all. Edge graphs carry casts as
        # _to_dim_order_copy, or as _to_copy without dim order, and captured
        # graphs as to.dtype or, with a device, to.device.
        f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
        c, cl = torch.contiguous_format, torch.channels_last
        for model, x_dtype, layout, capture, dim_order, norm_inputs in (
            (UpcastNorm(f16), f16, c, False, True, [(f16, True, True)]),
            (UpcastNorm(f16), f16, cl, False, True, [(f16, True, True)]),
            (ContiguousUpcastNorm(f16), f16, cl, False, True, [(f32, False, False)]),
            (UpcastNorm(f16), f16, c, False, False, [(f16, True, True)]),
            (TypeAsNorm(f16), f16, c, False, True, [(f16, True, True)]),
            (T5Norm(f16), f16, c, False, True, [(f16, True, True)]),
            (UpcastNorm(f32), f16, c, False, True, [(f32, False, False)]),
            (UpcastNorm(f32, compute_dtype=f16), f32, c, False, True, []),
            (UpcastNorm(f32, compute_dtype=torch.float64), f32, c, False, True, []),
            (RoundTripNorm(f32), f32, c, False, True, [(f32, False, True)]),
            (UpcastNorm(bf16), bf16, c, False, True, [(f32, False, False)]),
            (UpcastNorm(bf16), f32, c, False, True, [(f32, True, False)]),
            (ShiftedNorm(f32), f32, c, False, True, [(f32, True, False)]),
            (UpcastNorm(bf16), bf16, c, True, True, [(bf16, True, True)]),
            (UpcastNorm(f32), f32, c, True, True, [(f32, True, True)]),
            (UpcastNorm(bf16), f32, c, True, True, [(f32, True, True)]),
            (UpcastNorm(f32), bf16, c, True, True, []),
            (UpcastNorm(f32, compute_dtype=torch.float64), f32, c, True, True, []),
            (ShiftedNorm(f32), f32, c, True, True, []),
            (ContiguousUpcastNorm(f16), f16, cl, True, True, []),
            (TypeAsNorm(bf16), bf16, c, True, True, [(bf16, True, True)]),
            (TypeAsNorm(f32), f32, c, True, True, [(f32, True, True)]),
            (T5Norm(bf16), bf16, c, True, True, [(bf16, True, True)]),
        ):
            with self.subTest(
                model=type(model).__name__,
                weight=model.weight.dtype,
                x=x_dtype,
                layout=layout,
                capture=capture,
                dim_order=dim_order,
            ):
                x = torch.randn(2, 4, 8, 32, dtype=x_dtype)
                self._check_recomposed_rms_norm(
                    model, x.to(memory_format=layout), capture, dim_order, norm_inputs
                )

    @staticmethod
    def _htp_compiler_specs(use_fp16):
        return stubbed_compiler_specs(generate_htp_compiler_spec(use_fp16=use_fp16))

    def _check_recomposed_rms_norm(self, model, x, capture, dim_order, norm_inputs):
        """
        Runs RecomposeRmsNorm on the captured or edge graph of `model`, checks
        the dtype of each rebuilt norm's input, whether it is the graph input
        and whether the norm keeps the weight, and compares the output with
        eager.
        """
        ep = torch.export.export(model, (x,), strict=True)
        if not capture:
            # As in to_edge_transform_and_lower_to_qnn, which lifts the exponent of pow.
            ep = get_qnn_pass_manager_cls()().transform_for_export_pipeline(ep)
            config = EdgeCompileConfig(_skip_dim_order=not dim_order)
            ep = to_edge(ep, compile_config=config).exported_program()
        graph_module = ep.module() if capture else ep.graph_module
        RecomposeRmsNorm(
            quantization_capture=capture,
            edge_program=None if capture else ep,
            compiler_specs=None if capture else self._htp_compiler_specs(True),
        )(graph_module)
        rms_norms = (
            exir_ops.edge.aten.rms_norm.default,
            torch.ops.aten.rms_norm.default,
        )
        self.assertEqual(
            [
                (
                    node.args[0].meta["val"].dtype,
                    node.args[0].op == "placeholder",
                    node.args[2] is not None,
                )
                for node in graph_module.graph.nodes
                if node.target in rms_norms
            ],
            norm_inputs,
        )
        module = graph_module if capture else ep.module()
        torch.testing.assert_close(module(x), model(x))

    @staticmethod
    def _rms_norm_kept_weights(graph_module):
        return [
            node.args[2] is not None
            for node in graph_module.graph.nodes
            if node.target
            in (exir_ops.edge.aten.rms_norm.default, torch.ops.aten.rms_norm.default)
        ]

    def test_recompose_rms_norm_only_takes_what_qnn_supports(self):
        class Norm(torch.nn.Module):  # a Llama-style norm
            def __init__(self, dtype=torch.bfloat16):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(32, dtype=dtype))
                self.eps = 1e-6

            def forward(self, x):
                h = x.float()
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
                return self.weight * h.to(x.dtype)

        class BufferEpsNorm(Norm):
            def __init__(self, dtype=torch.bfloat16):
                super().__init__(dtype)
                del self.eps
                self.register_buffer("eps", torch.tensor(1e-6))

        class ScaledNorm(torch.nn.Module):  # a norm times a scale or a gate
            def __init__(self, scale, dim=-1, keepdim=True, gate=False):
                super().__init__()
                self.dim, self.keepdim, self.gate = dim, keepdim, gate
                self.register_buffer("scale", scale)

            def forward(self, x):
                ms = x.pow(2).mean(self.dim, keepdim=self.keepdim)
                h = x * torch.rsqrt(ms + 1e-6)
                return h * (torch.sigmoid(x) if self.gate else self.scale)

        class TableNorm(ScaledNorm):  # a norm of a buffer, added to the input
            def __init__(self, scale):
                super().__init__(scale)
                self.register_buffer("table", torch.randn(4, 32))

            def forward(self, x):
                return x + super().forward(self.table)

        class StateNorm(ScaledNorm):  # a norm of a buffer the model updates
            def __init__(self):
                super().__init__(torch.tensor(2.0))
                self.register_buffer("state", torch.zeros(4, 32))

            def forward(self, x):
                self.state.fill_(1.0)
                return x + super().forward(self.state)

        # QNN's RmsNorm takes a mean over the last of 1 to 4 dims, kept as a
        # dim, with static sizes and a constant eps. Anything else stays
        # decomposed, since rms_norm has no CPU kernel. With the HTP float16
        # specs passed here, a norm whose weight is not one value per channel
        # has its normalization rebuilt alone. A norm of constants is rebuilt
        # here, since its result meets the input.
        bf16 = torch.bfloat16
        dynamic = {"x": {0: torch.export.Dim("n", min=2)}}
        scale, weight = torch.tensor(2.0), torch.randn(32)
        for i, (model, x, dynamic_shapes, kept_weights) in enumerate(
            (
                (Norm(), torch.randn(1, 2, 3, 4, 32, dtype=bf16), None, []),
                (ScaledNorm(weight), torch.randn(1, 2, 3, 4, 32), None, []),
                (BufferEpsNorm(), torch.randn(4, 32, dtype=bf16), None, []),
                (BufferEpsNorm(torch.float32), torch.randn(4, 32), None, []),
                (ScaledNorm(weight), torch.randn(4, 32), dynamic, []),
                (ScaledNorm(scale, gate=True), torch.randn(4, 32), dynamic, []),
                (ScaledNorm(scale), torch.randn(4, 32), None, [False]),
                (ScaledNorm(weight, dim=0), torch.randn(32, 32), None, []),
                (ScaledNorm(weight, dim=None), torch.randn(32, 32), None, []),
                (ScaledNorm(weight, keepdim=False), torch.randn(32, 32), None, []),
                (TableNorm(scale), torch.randn(4, 32), None, [False]),
                (TableNorm(weight), torch.randn(4, 32), None, [True]),
                (StateNorm(), torch.randn(4, 32), None, [False]),
            )
        ):
            with self.subTest(row=i, model=type(model).__name__):
                ep = torch.export.export(
                    model, (x,), dynamic_shapes=dynamic_shapes, strict=True
                )
                ep = to_edge(ep).exported_program()
                RecomposeRmsNorm(
                    edge_program=ep, compiler_specs=self._htp_compiler_specs(True)
                )(ep.graph_module)
                self.assertEqual(
                    self._rms_norm_kept_weights(ep.graph_module), kept_weights
                )
                torch.testing.assert_close(ep.module()(x), model(x))

    def test_recompose_rms_norm_rebuilds_a_normalization_for_htp_fp16_only(self):
        def manager_passes(compiler_specs):
            # As in to_edge_transform_and_lower_to_qnn, which passes the specs.
            return lambda ep: get_qnn_pass_manager_cls()().get_to_edge_transform_passes(
                ep, compiler_specs=compiler_specs
            )

        def fp16_recipe_passes():
            recipe = QNNRecipeProvider()._get_qnn_lowering_recipe(
                use_fp16=True, soc_model=QcomChipset.SM8650
            )
            return lambda ep: recipe.edge_transform_passes[0](None, ep)

        # The normalization alone is rebuilt to keep HTP's float16 precision
        # from overflowing. Elsewhere it stays decomposed, as every op in it
        # has a CPU kernel.
        x = torch.randn(4, 32)
        for name, get_passes, kept_weights in (
            ("htp_fp16", manager_passes(self._htp_compiler_specs(True)), [False]),
            ("htp_quantized", manager_passes(self._htp_compiler_specs(False)), []),
            ("none", manager_passes(None), []),
            ("fp16_recipe", fp16_recipe_passes(), [False]),
        ):
            with self.subTest(specs=name):
                ep = torch.export.export(ScaledRmsNorm(), (x,), strict=True)
                ep = to_edge(ep).exported_program()
                passes = get_passes(ep)
                next(p for p in passes if isinstance(p, RecomposeRmsNorm))(
                    ep.graph_module
                )
                self.assertEqual(
                    self._rms_norm_kept_weights(ep.graph_module), kept_weights
                )

    def test_to_edge_pipeline_rebuilds_a_normalization_for_htp_fp16(self):
        # export_llama runs the edge passes through this method, with the
        # compiler specs of its partitioner, on a program whose edge config
        # allows ops outside the core ATen set.
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        for use_fp16, kept_weights in ((True, [False]), (False, [])):
            with self.subTest(use_fp16=use_fp16):
                ep = to_edge(
                    torch.export.export(
                        UpcastRmsNorm(torch.bfloat16), (x,), strict=True
                    ),
                    compile_config=EdgeCompileConfig(_check_ir_validity=False),
                ).exported_program()
                ep = get_qnn_pass_manager_cls()().transform_for_to_edge_pipeline(
                    ep, compiler_specs=self._htp_compiler_specs(use_fp16)
                )
                self.assertEqual(
                    self._rms_norm_kept_weights(ep.graph_module), kept_weights
                )

    def test_recompose_rms_norm_leaves_a_norm_the_partitioner_skips(self):
        class Norm(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.register_buffer("weight", weight)

            def forward(self, x):
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return x * rms * self.weight

        class UpcastNorm(Norm):  # a Qwen-style norm
            def forward(self, x):
                h = x.float()
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                return self.weight * h.to(x.dtype)

        class SharedRsqrtNorm(Norm):  # a norm that also returns its rsqrt
            def forward(self, x):
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return x * rms * self.weight, rms

        class SharedCastNorms(UpcastNorm):  # norms of one float32 copy of x
            def forward(self, x):
                h = x.float()
                a = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                b = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
                return self.weight * a.to(x.dtype), self.weight * b.to(x.dtype)

        # The norm stays decomposed when the partitioner would skip rms_norm,
        # which has no CPU kernel, or a node the rebuild removes, by op or by
        # name. A skipped node the rebuild keeps, like the weight multiply of
        # a normalization rebuilt alone, an rsqrt used elsewhere too or a cast
        # another norm reads, does not matter. The FP16 recipe passes its skip
        # sets on.
        weighted, scaled = Norm(torch.randn(32)), Norm(torch.tensor(2.0))
        upcast = UpcastNorm(torch.randn(32, dtype=torch.float16))
        shared_rsqrt = SharedRsqrtNorm(torch.randn(32))
        shared_cast = SharedCastNorms(torch.randn(32, dtype=torch.float16))
        x32, x16 = torch.randn(4, 32), torch.randn(4, 32, dtype=torch.float16)
        rms_norm_op = {"aten.rms_norm.default"}
        cast_op = {"dim_order_ops._to_dim_order_copy.default"}
        cast = "dim_order_ops__to_dim_order_copy_default"
        for skip, model, x, skip_ids, skip_ops, kept_weights in (
            ("none", weighted, x32, None, None, [True]),
            ("none", scaled, x32, None, None, [False]),
            ("none", upcast, x16, None, None, [True]),
            ("rms_norm op", weighted, x32, None, rms_norm_op, []),
            ("rms_norm op", scaled, x32, None, rms_norm_op, []),
            ("rsqrt op", weighted, x32, None, {"aten.rsqrt.default"}, []),
            ("rsqrt id", scaled, x32, {"aten_rsqrt_default"}, None, []),
            ("rsqrt id", shared_rsqrt, x32, {"aten_rsqrt_default"}, None, [True]),
            ("cast op", upcast, x16, None, cast_op, [False]),
            ("cast id", shared_cast, x16, {cast}, None, [False, True]),
            ("weight mul id", weighted, x32, {"aten_mul_tensor_1"}, None, [False]),
        ):
            with self.subTest(skip=skip, model=type(model).__name__):
                ep = torch.export.export(model, (x,), strict=True)
                ep = to_edge(ep).exported_program()
                self.assertLessEqual(
                    skip_ids or set(), {node.name for node in ep.graph.nodes}
                )
                recipe = QNNRecipeProvider()._get_qnn_lowering_recipe(
                    use_fp16=True,
                    soc_model=QcomChipset.SM8650,
                    skip_node_id_set=skip_ids,
                    skip_node_op_set=skip_ops,
                )
                passes = recipe.edge_transform_passes[0](None, ep)
                next(p for p in passes if isinstance(p, RecomposeRmsNorm))(
                    ep.graph_module
                )
                self.assertEqual(
                    self._rms_norm_kept_weights(ep.graph_module), kept_weights
                )
                torch.testing.assert_close(ep.module()(x), model(x))

    def test_recompose_rms_norm_needs_a_runtime_value_on_htp_only(self):
        class TableNorm(torch.nn.Module):  # a norm of a buffer
            def __init__(self, add_input, update):
                super().__init__()
                self.add_input, self.update = add_input, update
                self.register_buffer("table", torch.randn(4, 32))
                self.weight = torch.nn.Parameter(torch.randn(32))

            def forward(self, x):
                if self.update:
                    self.table.fill_(1.0)
                t = self.table
                rms = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
                norm = self.weight * (t * rms)
                return x + norm if self.add_input else (x * 2, norm)

        # Only HTP drops a partition that reads nothing at runtime, so a norm of
        # constants that never meets a runtime value stays decomposed for HTP
        # and for a pass that does not know the backend. A buffer the model
        # updates is a runtime value, also in the program fresh from
        # torch.export that to_edge_transform_and_lower_to_qnn passes in.
        x = torch.randn(4, 32)
        for backend, backend_options, kept_alone in (
            ("htp", generate_htp_compiler_spec(use_fp16=True), []),
            ("gpu", generate_gpu_compiler_spec(), [True]),
            ("unknown", None, []),
        ):
            for add_input, update in ((True, False), (False, False), (False, True)):
                with self.subTest(backend=backend, add_input=add_input, update=update):
                    model = TableNorm(add_input, update)
                    exported = torch.export.export(model, (x,), strict=True)
                    ep = to_edge(exported).exported_program()
                    compiler_specs = None
                    if backend_options is not None:
                        compiler_specs = generate_qnn_executorch_compiler_spec(
                            soc_model=QcomChipset.SM8650,
                            backend_options=backend_options,
                        )
                    RecomposeRmsNorm(
                        edge_program=exported, compiler_specs=compiler_specs
                    )(ep.graph_module)
                    self.assertEqual(
                        self._rms_norm_kept_weights(ep.graph_module),
                        [True] if add_input or update else kept_alone,
                    )
                    torch.testing.assert_close(ep.module()(x), model(x))

    def test_recompose_rms_norm_meets_runtime_values_only_in_delegated_ops(self):
        class TableNorm(torch.nn.Module):  # a norm of a buffer, used with x
            def __init__(self, use):
                super().__init__()
                self.use = use
                self.register_buffer("table", torch.randn(4, 32))
                self.weight = torch.nn.Parameter(torch.randn(32))

            def forward(self, x):
                t = self.table
                rms = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
                norm = self.weight * (t * rms)
                return self.use(x, norm)

        # On CPU, the op that reads x leaves the norm in a partition of
        # constants, which HTP drops, and rms_norm has no CPU kernel.
        x = torch.randn(4, 32)
        for name, use, skip_ops, rebuilt in (
            ("add", lambda x, norm: x + norm, None, True),
            ("skipped sub", lambda x, norm: x - norm, {"aten.sub.Tensor"}, False),
            ("mask index", lambda x, norm: norm[x > 0], None, False),
            ("prod", lambda x, norm: x + norm.prod(-1, keepdim=True), None, False),
        ):
            with self.subTest(use=name):
                model = TableNorm(use)
                exported = torch.export.export(model, (x,), strict=True)
                ep = to_edge(exported).exported_program()
                RecomposeRmsNorm(
                    edge_program=exported,
                    compiler_specs=generate_qnn_executorch_compiler_spec(
                        soc_model=QcomChipset.SM8650,
                        backend_options=generate_htp_compiler_spec(use_fp16=True),
                    ),
                    skip_node_op_set=skip_ops,
                )(ep.graph_module)
                self.assertEqual(
                    self._rms_norm_kept_weights(ep.graph_module),
                    [True] if rebuilt else [],
                )
                torch.testing.assert_close(ep.module()(x), model(x))

    def test_tester_rebuilds_a_normalization_for_htp_fp16(self):
        # The backend test suite lowers through this stage, with HTP float16
        # specs by default.
        stage = ToEdgeTransformAndLower()
        sample_input = (torch.linspace(-1000, 1000, 128).reshape(4, 32),)
        try:
            stage.run(torch.export.export(ScaledRmsNorm(), sample_input, strict=True))
        except RuntimeError as e:
            if "Failed to initialize QNN backend" in str(e):
                skip_without_qnn_sdk(self, e)
            raise
        graph_module = stage.edge_dialect_program.exported_program().graph_module
        self.assertIn(
            exir_ops.edge.aten.rms_norm.default,
            [
                node.target
                for _, lowered in graph_module.named_children()
                for node in lowered.original_module.graph.nodes
            ],
        )

    def test_recompose_rms_norm_takes_no_runtime_weight(self):
        class InputScaledNorm(torch.nn.Module):
            def forward(self, x, scale):
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return x * rms * scale

        # QNN's RmsNorm needs a constant weight, so a graph input is not taken
        # as one. With HTP float16 specs, edge graphs rebuild the normalization
        # alone instead. Edge graphs need the program to tell the two apart.
        inputs = (torch.randn(4, 32), torch.randn(32))
        with self.assertRaises(AssertionError):
            RecomposeRmsNorm()(
                torch.export.export(InputScaledNorm(), inputs, strict=True).graph_module
            )
        for capture, kept_weights in ((False, [False]), (True, [])):
            with self.subTest(capture=capture):
                ep = torch.export.export(InputScaledNorm(), inputs, strict=True)
                if capture:
                    graph_module = ep.module()
                    RecomposeRmsNorm(quantization_capture=True)(graph_module)
                else:
                    ep = to_edge(ep).exported_program()
                    graph_module = ep.graph_module
                    RecomposeRmsNorm(
                        edge_program=ep, compiler_specs=self._htp_compiler_specs(True)
                    )(graph_module)
                self.assertEqual(
                    self._rms_norm_kept_weights(graph_module), kept_weights
                )
                module = graph_module if capture else ep.module()
                torch.testing.assert_close(module(*inputs), InputScaledNorm()(*inputs))

    def test_recompose_rms_norm_needs_the_norm_of_one_tensor(self):
        class AlmostNorm(torch.nn.Module):
            def __init__(self, scaled, squared):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(32))
                self.register_buffer("other", torch.rand(32, 32) + 0.5)
                self.scaled, self.squared = scaled, squared

            def forward(self, x):
                other = self.other
                squared = {"pow3": x.pow(3), "mixed": x * other}.get(
                    self.squared, x.pow(2)
                )
                scaled = other if self.scaled == "other" else x
                rms = torch.rsqrt(squared.mean(-1, keepdim=True) + 1e-6)
                return self.weight * (scaled * rms)

        # Each model looks like a norm, but the tensor it scales is not the
        # one it squares, or it does not square it, so it stays decomposed.
        # The export pipeline lifts the exponent of pow into a constant.
        for lift in (False, True):
            for scaled, squared in (("other", "pow2"), ("x", "pow3"), ("x", "mixed")):
                with self.subTest(scaled=scaled, squared=squared, lift=lift):
                    model = AlmostNorm(scaled, squared)
                    # Positive, so a cube or a product still has a real square root.
                    x = torch.rand(32, 32) + 0.5
                    ep = torch.export.export(model, (x,), strict=True)
                    if lift:
                        pm = get_qnn_pass_manager_cls()()
                        ep = pm.transform_for_export_pipeline(ep)
                    ep = to_edge(ep).exported_program()
                    RecomposeRmsNorm(
                        edge_program=ep, compiler_specs=self._htp_compiler_specs(True)
                    )(ep.graph_module)
                    self.assertNotIn(
                        exir_ops.edge.aten.rms_norm.default,
                        [node.target for node in ep.graph.nodes],
                    )
                    torch.testing.assert_close(ep.module()(x), model(x))

    def test_remove_redundancy_keeps_a_to_copy_that_changes_the_dtype(self):
        class Copy(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                y = x.to(self.dtype, memory_format=torch.contiguous_format)
                return y + y

        # Without dim order, a layout change is a _to_copy with memory_format,
        # which QNN drops. A _to_copy that also changes the dtype has to stay.
        x = torch.randn(2, 4, 8, 8).to(memory_format=torch.channels_last)
        for dtype, kept in ((torch.float32, False), (torch.float16, True)):
            with self.subTest(dtype=dtype):
                config = EdgeCompileConfig(_skip_dim_order=True)
                ep = to_edge(
                    torch.export.export(Copy(dtype), (x,), strict=True),
                    compile_config=config,
                ).exported_program()
                RemoveRedundancy(quantization_capture=False)(ep.graph_module)
                self.assertEqual(
                    exir_ops.edge.aten._to_copy.default
                    in [node.target for node in ep.graph.nodes],
                    kept,
                )
                self.assertEqual(ep.module()(x).dtype, dtype)

    def test_i64_to_i32_casts_a_top_of_graph_constant_after_placeholders(self):
        # DecomposeRemainder puts the int64 constant 3 at the top of the graph.
        class IntRemainder(torch.nn.Module):
            def forward(self, x, i):
                return x * (i % 3).to(x.dtype)

        sample_input = (torch.randn(2, 4), torch.randint(0, 9, (2, 4)))
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            to_edge_transform_and_lower_to_qnn(
                IntRemainder().eval(), sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "Failed to initialize QNN backend" in str(e):
                skip_without_qnn_sdk(self, e)
            raise

    def test_uint8_to_float32_cast_goes_through_int32_only_on_htp(self):
        quant_attrs = {
            QCOM_ENCODING: exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            QCOM_SCALE: 1 / 255,
            QCOM_ZERO_POINT: 0,
            QCOM_QUANT_MIN: 0,
            QCOM_QUANT_MAX: 255,
            QCOM_DTYPE: torch.uint8,
        }
        cases = [
            (torch.float32, QnnExecuTorchBackendType.kHtpBackend, None, 2),
            (torch.float32, QnnExecuTorchBackendType.kGpuBackend, None, 1),
            (torch.float16, QnnExecuTorchBackendType.kHtpBackend, None, 1),
            (torch.float32, QnnExecuTorchBackendType.kHtpBackend, quant_attrs, 1),
        ]
        for dtype, backend_type, cast_quant_attrs, expected in cases:
            with self.subTest(
                dtype=dtype, backend_type=backend_type, quantized=bool(cast_quant_attrs)
            ):
                edge_program = to_edge(
                    torch.export.export(
                        CastToFloat(dtype),
                        (torch.arange(4, dtype=torch.uint8),),
                        strict=True,
                    )
                ).exported_program()
                visitors = get_node_visitors(edge_program, backend_type=backend_type)
                cast = next(
                    node
                    for node in edge_program.graph.nodes
                    if node.op == "call_function" and node.args[0].op == "placeholder"
                )
                if cast_quant_attrs:
                    cast.meta[QCOM_QUANT_ATTRS] = cast_quant_attrs
                ops = visitors[cast.target.__name__].define_node(
                    cast, defaultdict(dict)
                )
                self.assertEqual(expected, len(ops))

    def test_convert_full_like_to_full(self):
        sample_input = (torch.randn(2, 3), torch.randn(2, 3))
        # An int fill on a float input must still give a float tensor.
        for dynamic, fill in ((False, 0.5), (False, 1), (True, 0.5)):
            with self.subTest(dynamic=dynamic, fill=fill):
                module = FullLikeShapeOnly(fill=fill)
                dynamic_shapes = (
                    {"x": {0: torch.export.Dim("n", min=2, max=4)}, "y": None}
                    if dynamic
                    else None
                )
                ep = to_edge(
                    torch.export.export(
                        module,
                        sample_input,
                        dynamic_shapes=dynamic_shapes,
                        strict=True,
                    )
                ).exported_program()
                graph_module = ConvertFullLikeToFull()(ep.graph_module).graph_module
                targets = {node.target for node in graph_module.graph.nodes}
                x = next(n for n in graph_module.graph.nodes if n.name == "x")
                # A dynamic shape has to be read at runtime. Otherwise the ops
                # that only gave full_like its shape are removed, so nothing
                # reads x.
                self.assertEqual(
                    exir_ops.edge.aten.full_like.default in targets, dynamic
                )
                self.assertEqual(
                    exir_ops.edge.aten.full.default in targets, not dynamic
                )
                self.assertEqual(len(x.users) > 0, dynamic)
                # The pass rewrites ep's graph in place.
                for result, expected in zip(
                    ep.module()(*sample_input), module(*sample_input)
                ):
                    torch.testing.assert_close(result, expected)

    def test_i64_to_i32_casts_any_int64_result(self):
        # A sum of bools and topk indices are int64, and masked_scatter needs
        # them to match the int64 input the pass turns into int32.
        class SumIntoInput(torch.nn.Module):
            def forward(self, x, k, mask):
                return (x > 0.5).sum(-1).masked_scatter(mask, k)

        class TopkIndicesIntoInput(torch.nn.Module):
            def forward(self, x, k, mask):
                return x.topk(1).indices.masked_scatter(mask[:, None], k)

        class LongIntoInput(torch.nn.Module):
            def forward(self, x, k, mask):
                return (x[:, 0] > 0.5).long().masked_scatter(mask, k)

        sample_input = (
            torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.1, 0.3]]),
            torch.tensor([3, 4]),
            torch.tensor([True, False]),
        )
        # Without dim order, a user's int64 cast is a _to_copy, like the index
        # casts the pass keeps int64.
        cases = (
            (SumIntoInput(), False),
            (TopkIndicesIntoInput(), False),
            (LongIntoInput(), True),
        )
        for module, skip_dim_order in cases:
            with self.subTest(type(module).__name__):
                ep = to_edge(
                    torch.export.export(module, sample_input, strict=True),
                    compile_config=EdgeCompileConfig(_skip_dim_order=skip_dim_order),
                ).exported_program()
                gm = I64toI32(ep)(ep.graph_module).graph_module
                torch.testing.assert_close(gm(*sample_input)[0], module(*sample_input))

    def test_i64_to_i32_keeps_int64_for_cpu_ops(self):
        # QNN has no builder for prod or for per-token quantization, so they
        # run on CPU: an int32 cast would wrap the product, and the CPU
        # quantize_per_token needs an int64 zero point. A QNN op, like the
        # index here, still reads the product as int32, and so does a CPU op
        # that mixes it with an int32 cast. The CPU max pool backward needs
        # int64 indices too.
        qd = torch.ops.quantized_decomposed
        index = exir_ops.edge.aten.index.Tensor
        quantize = exir_ops.edge.quantized_decomposed.quantize_per_token.default
        backward = exir_ops.edge.aten.max_pool2d_with_indices_backward.default

        class Prod(torch.nn.Module):
            def forward(self, x):
                return x.prod(-1)

        class ProdIndex(torch.nn.Module):
            def forward(self, x):
                return x[:, (x > 0).prod(-1)]

        class PerTokenQuant(torch.nn.Module):
            def forward(self, x):
                scale, zero_point = qd.choose_qparams_per_token_asymmetric(
                    x, torch.int8
                )
                return qd.quantize_per_token(
                    x, scale, zero_point, -128, 127, torch.int8
                )

        class ProdIntoInput(torch.nn.Module):
            def forward(self, x, mask, k):
                return (x > 0).prod(-1).masked_scatter(mask, k)

        class MaxPoolBackward(torch.nn.Module):
            def forward(self, x, grad):
                _, indices = torch.ops.aten.max_pool2d_with_indices(x, [2, 2])
                return torch.ops.aten.max_pool2d_with_indices_backward(
                    grad, x, [2, 2], [2, 2], [0, 0], [1, 1], False, indices
                )

        prod_input = (torch.tensor([[65536, 65536]], dtype=torch.int32),)
        pool_input = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 2, 2))
        mix_input = (
            torch.randn(2, 3),
            torch.tensor([True, False]),
            torch.tensor([3, 4]),
        )
        cases = (
            (Prod(), prod_input, None),
            (ProdIndex(), (torch.randn(2, 3),), (index, torch.int32)),
            (PerTokenQuant(), (torch.randn(2, 3),), (quantize, torch.int64)),
            (ProdIntoInput(), mix_input, None),
            (MaxPoolBackward(), pool_input, (backward, torch.int64)),
        )
        for module, sample_input, read in cases:
            with self.subTest(type(module).__name__):
                ep = to_edge(
                    torch.export.export(module, sample_input, strict=True)
                ).exported_program()
                gm = I64toI32(ep)(ep.graph_module).graph_module
                torch.testing.assert_close(gm(*sample_input)[0], module(*sample_input))
                if read is not None:
                    target, dtype = read
                    node = next(n for n in gm.graph.nodes if n.target == target)
                    reads = []
                    torch.fx.node.map_arg(node.args, reads.append)
                    integer_reads = [
                        arg.meta["val"].dtype
                        for arg in reads
                        if not arg.meta["val"].is_floating_point()
                    ]
                    self.assertEqual(integer_reads, [dtype])

    def test_i64_to_i32_keeps_int64_for_ops_kept_on_cpu(self):
        # QNN has builders for these ops, but the partitioner leaves them on
        # CPU, where an int32 cast would wrap the result: a skipped op and an
        # op that reads float64.
        class Sum(torch.nn.Module):
            def forward(self, x):
                return x.sum(-1)

        class LongOfDouble(torch.nn.Module):
            def forward(self, x):
                return (x.double() * x.double()).long()

        big = (torch.tensor([[2**30, 2**30]], dtype=torch.int32),)
        cases = (
            ("skipped op", Sum(), big, {"skip_node_op_set": {"aten.sum.dim_IntList"}}),
            (
                "skipped node",
                Sum(),
                big,
                {"skip_node_id_set": {"aten_sum_dim_int_list"}},
            ),
            ("float64 input", LongOfDouble(), (torch.tensor([1e5]),), {}),
        )
        for name, module, sample_input, kwargs in cases:
            with self.subTest(name):
                ep = to_edge(
                    torch.export.export(module, sample_input, strict=True)
                ).exported_program()
                gm = I64toI32(ep, **kwargs)(ep.graph_module).graph_module
                torch.testing.assert_close(gm(*sample_input)[0], module(*sample_input))

    def test_i64_to_i32_casts_back_multi_output_results(self):
        # topk of an int64 sum returns int64 values and indices.
        class SumTopk(torch.nn.Module):
            def forward(self, x):
                return (x > 0.5).sum(-1).topk(2)

        sample_input = (
            torch.tensor([[0.9, 0.9, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9]]),
        )
        module = SumTopk()
        ep = to_edge(
            torch.export.export(module, sample_input, strict=True)
        ).exported_program()
        gm = I64toI32(ep)(ep.graph_module).graph_module
        for result, expected in zip(gm(*sample_input), module(*sample_input)):
            torch.testing.assert_close(result, expected)

    def test_i64_to_i32_keeps_int64_indices(self):
        # gather and the 8-bit embedding run on CPU kernels that need an int64
        # index, even when the pass casts the op that computes it.
        import executorch.exir.passes._quant_patterns_and_replacements  # noqa: F401

        embedding_byte = exir_ops.edge.quantized_decomposed.embedding_byte.default
        gather = exir_ops.edge.aten.gather.default

        class Model(torch.nn.Module):
            def __init__(self, use_gather, view_tokens):
                super().__init__()
                self.use_gather = use_gather
                self.view_tokens = view_tokens
                self.register_buffer("weight", torch.ones(10, 4, dtype=torch.int8))
                self.register_buffer("scales", torch.ones(10))

            def forward(self, x, tokens):
                if self.use_gather:
                    return x.gather(1, x.argmax(-1, keepdim=True))
                if self.view_tokens:
                    tokens = tokens.view(-1)
                return torch.ops.quantized_decomposed.embedding_byte(
                    self.weight, self.scales, None, -128, 127, tokens
                )

        sample_input = (torch.randn(2, 3), torch.tensor([1, 5, 7]))
        for target, view_tokens in ((gather, False), (embedding_byte, True)):
            with self.subTest(target=target.__name__):
                module = Model(target == gather, view_tokens)
                ep = to_edge(
                    torch.export.export(module, sample_input, strict=True)
                ).exported_program()
                gm = I64toI32(ep, skip_node=frozenset({"tokens"}))(
                    ep.graph_module
                ).graph_module
                node = next(n for n in gm.graph.nodes if n.target == target)
                index = node.args[I64toI32.I64_IN_OPS[target][0]]
                self.assertEqual(torch.int64, index.meta["val"].dtype)

        # Token ids the pass skips stay int64 and need no cast.
        ep = to_edge(
            torch.export.export(Model(False, False), sample_input, strict=True)
        ).exported_program()
        gm = I64toI32(ep, skip_node=frozenset({"tokens"}))(ep.graph_module).graph_module
        node = next(n for n in gm.graph.nodes if n.target == embedding_byte)
        self.assertEqual("placeholder", node.args[5].op)

    def test_mha_to_sha(self):
        from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
        from executorch.examples.models.llama.model_args import ModelArgs
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            CausalAttentionMask,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaAttention,
        )

        # Initailize model config
        args = ModelArgs()
        args.max_seq_len = 128
        args.max_context_len = 128
        args.ar_len = 32
        args.use_kv_cache = True
        args.dim = 32
        args.n_heads = 8
        args.n_kv_heads = 8
        args.n_layers = 2
        args.head_dim = args.dim // args.n_heads
        mod = convert_linear_to_conv2d(LlamaAttention(0, args, True))

        # Prepare inputs
        hidden_states = torch.randn(args.max_batch_size, args.ar_len, args.dim)
        freqs_cos = torch.randn(args.ar_len, 1)
        freqs_sin = torch.randn(args.ar_len, 1)
        atten_mask = CausalAttentionMask(
            args.max_batch_size, args.ar_len, args.max_seq_len
        )
        k_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.head_dim,
            args.max_seq_len - args.ar_len,
        )

        v_cache = torch.zeros(
            args.max_batch_size,
            args.n_kv_heads,
            args.max_seq_len - args.ar_len,
            args.head_dim,
        )
        sample_input = (
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask.mask,
            k_cache,
            v_cache,
        )

        # Run original module for reference
        refs = mod(*sample_input)

        # Export the module and convert linear to conv2d
        edge_program = to_edge(torch.export.export(mod, sample_input))
        new_ep = edge_program.exported_program()

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # WQ, WK, WV, O
        self.assertTrue(len(conv_nodes) == 4, "Convolution nodes missing")

        # Convert MHA to SHA
        # This is a simplified version of what happens in the full pipeline to test the core functionality
        graph_module = RemoveRedundancy(quantization_capture=False)(
            new_ep.graph_module
        ).graph_module
        graph_module = ConvertBmmToMatmul()(graph_module).graph_module
        graph_module = ConvertMhaToSha(new_ep)(graph_module).graph_module

        conv_nodes = [
            n
            for n in new_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # Check graph structure: WQ, WK, WV should be converted to SHA
        self.assertTrue(len(conv_nodes) == 25, "Convolution nodes should be splited")

        # Execute new graph and compare with reference
        outs = graph_module(
            *new_ep.state_dict.values(), *new_ep.constants.values(), *sample_input
        )
        for i, (out, ref) in enumerate(zip(outs, refs)):
            self.assertTrue(
                torch.allclose(out, *ref, rtol=1e-6, atol=1e-6),
                f"Output {i} mismatch: got {out}, expected {ref}",
            )

    def test_decompose_im2col_matches_unfold(self):
        """DecomposeColIm's im2col decomposition must produce the exact same
        values as torch.nn.functional.unfold, across all three decomposition
        branches:
          - stride == kernel_size, square kernel -> pixel_unshuffle
          - stride == kernel_size, non-square kernel -> qnn_custom.space_to_depth
          - stride != kernel_size -> index_select gather + the tail above
        """
        cases = [
            ((2, 4, 8, 8), (2, 2), (2, 2), (0, 0)),  # stride == kernel_size, square
            ((2, 4, 9, 6), (3, 2), (3, 2), (0, 0)),  # stride == kernel_size, non-square
            (
                (2, 4, 8, 8),
                (2, 2),
                (1, 1),
                (0, 0),
            ),  # stride != kernel_size, square kernel
            (
                (1, 2, 7, 9),
                (2, 3),
                (1, 1),
                (0, 0),
            ),  # stride != kernel_size, non-square kernel
            (
                (1, 3, 10, 10),
                (3, 3),
                (2, 2),
                (0, 0),
            ),  # stride != kernel_size, overlapping
            (
                (2, 4, 6, 6),
                (2, 2),
                (2, 2),
                (1, 1),
            ),  # nonzero padding, stride == kernel_size
            (
                (2, 4, 6, 6),
                (2, 2),
                (1, 1),
                (1, 1),
            ),  # nonzero padding, stride != kernel_size
        ]
        for shape, kernel_size, stride, padding in cases:
            with self.subTest(
                shape=shape, kernel_size=kernel_size, stride=stride, padding=padding
            ):

                x = torch.randn(*shape)
                module = Unfold(kernel_size, stride, padding).eval()
                ref = module(x)
                exported_program = torch.export.export(module, (x,), strict=True)
                ep = to_edge(
                    exported_program,
                    compile_config=EdgeCompileConfig(
                        preserve_ops=[torch.ops.aten.im2col.default]
                    ),
                ).exported_program()
                self.assertTrue(
                    any(
                        n.target == exir_ops.edge.aten.im2col.default
                        for n in ep.graph_module.graph.nodes
                    ),
                    "im2col was decomposed in to_edge",
                )
                gm = DecomposeColIm()(ep.graph_module).graph_module
                gm.recompile()

                # Decomposition must actually have run: no im2col node left.
                self.assertFalse(
                    any(
                        n.target == exir_ops.edge.aten.im2col.default
                        for n in gm.graph.nodes
                    ),
                    "im2col node was not decomposed",
                )

                out = gm(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                self.assertTrue(
                    torch.allclose(ref, out),
                    f"unfold decomposition mismatch for kernel_size={kernel_size}, "
                    f"stride={stride}: got {out}, expected {ref}",
                )

    def test_decompose_im2col_fallback(self):
        """When stride < kernel_size, the index_select gather in
        DecomposeColIm duplicates elements by roughly
        (kernel_height/stride_height) * (kernel_width/stride_width).
        Above DecomposeColIm._MAX_GATHER_EXPANSION_FACTOR, the pass must skip
        decomposition rather than build an unbounded gather. Since QNN has no
        node visitor for aten.im2col, the un-decomposed node then falls back
        to CPU when lowered.
        """
        cases = [
            # factor == (8/2)*(8/2) <= MAX_GATHER_EXPANSION_FACTOR: decomposed.
            ((1, 1, 20, 20), (8, 8), (2, 2), True),
            # factor == (9/2)*(9/2) > MAX_GATHER_EXPANSION_FACTOR: falls back.
            ((1, 1, 20, 20), (9, 9), (2, 2), False),
        ]
        for shape, kernel_size, stride, should_decompose in cases:
            with self.subTest(shape=shape, kernel_size=kernel_size, stride=stride):

                x = torch.randn(*shape)
                module = Unfold(kernel_size, stride).eval()

                exported_program = torch.export.export(module, (x,), strict=True)
                ep = to_edge(
                    exported_program,
                    compile_config=EdgeCompileConfig(
                        preserve_ops=[torch.ops.aten.im2col.default]
                    ),
                ).exported_program()
                self.assertTrue(
                    any(
                        n.target == exir_ops.edge.aten.im2col.default
                        for n in ep.graph_module.graph.nodes
                    ),
                    "im2col was decomposed in to_edge",
                )
                gm = DecomposeColIm()(ep.graph_module).graph_module
                gm.recompile()

                im2col_present = any(
                    n.target == exir_ops.edge.aten.im2col.default
                    for n in gm.graph.nodes
                )
                self.assertNotEqual(
                    im2col_present,
                    should_decompose,
                    f"im2col {'should' if should_decompose else 'should NOT'} be "
                    f"decomposed for kernel_size={kernel_size}, stride={stride}",
                )

    def test_resolve_debug_handle(self):
        name_handle_map = {
            "aten_topk_default": 1,
            "getitem": 1,
            "getitem_1": 1,
            "aten_view_copy_default": 2,
            "aten__to_copy_default": 3,
            "aten_index_tensor": 4,
            "aten_add_tensor": 5,
        }
        module = TopKandIndex()  # noqa: F405
        sample_input = (torch.randn(3, 10),)

        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,  # Random soc_model
            backend_options=backend_options,
            dump_intermediate_outputs=True,
        )

        try:
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module,
                sample_input,
                compiler_spec,
                generate_etrecord=True,
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                skip_without_qnn_sdk(self, e)
            raise
        exec_prog_mgr = edge_prog_mgr.to_executorch()
        etrecord = exec_prog_mgr.get_etrecord()
        debug_handle_size = len(etrecord._debug_handle_map["forward"][0])
        self.assertEqual(
            len(name_handle_map),
            debug_handle_size,
            f"Number of handles does not match, expecting: {len(name_handle_map)}, but get: {debug_handle_size}",
        )
        after_edge_pass_ep = etrecord.graph_map["edge_after_transform/forward"]

        for node in after_edge_pass_ep.graph.nodes:
            if node.name in name_handle_map:
                expected_handle = name_handle_map.pop(node.name)
                node_handle = node.meta[DEBUG_HANDLE_KEY]
                self.assertEqual(
                    expected_handle,
                    node_handle,
                    f"{node.name} is expecting a handle id {expected_handle}, but got {node_handle}.",
                )
        self.assertEqual(
            len(name_handle_map),
            0,
            f"Following nodes did not find a match in the graph: {name_handle_map.keys()}",
        )

    def test_decompose_reciprocal_backend_aware(self):
        sample_input = (torch.tensor([2.0]),)
        target = torch.ops.aten.reciprocal.default
        decomposed_backends = (
            QnnExecuTorchBackendType.kHtpBackend,
            QnnExecuTorchBackendType.kGpuBackend,
            QnnExecuTorchBackendType.kLpaiBackend,
        )
        preserved_backends = (QnnExecuTorchBackendType.kUndefinedBackend,)

        for backend, should_decompose in [
            *[(b, True) for b in decomposed_backends],
            *[(b, False) for b in preserved_backends],
        ]:
            # The annotation pipeline is skipped for the GPU backend, as it does not support quantized data types
            pipelines = (
                ("export",)
                if backend == QnnExecuTorchBackendType.kGpuBackend
                else ("annotation", "export")
            )
            for pipeline in pipelines:
                with self.subTest(backend=backend, pipeline=pipeline):
                    ep = torch.export.export(Reciprocal(), sample_input, strict=True)
                    pm = get_qnn_pass_manager_cls(backend)()
                    if pipeline == "annotation":
                        pm.transform_for_annotation_pipeline(ep.graph_module)
                    else:
                        pm.transform_for_export_pipeline(ep)
                    has_target = any(
                        n.target == target for n in ep.graph_module.graph.nodes
                    )
                    self.assertNotEqual(
                        has_target,
                        should_decompose,
                        f"reciprocal {'should' if should_decompose else 'should NOT'} be decomposed for {backend.name}",
                    )

    def test_decompose_hardsigmoid_backend_aware(self):
        sample_input = (torch.tensor([2.0]),)
        target = torch.ops.aten.hardsigmoid.default
        decomposed_backends = (QnnExecuTorchBackendType.kLpaiBackend,)
        preserved_backends = (
            QnnExecuTorchBackendType.kGpuBackend,
            QnnExecuTorchBackendType.kHtpBackend,
            QnnExecuTorchBackendType.kUndefinedBackend,
        )

        for backend, should_decompose in [
            *[(b, True) for b in decomposed_backends],
            *[(b, False) for b in preserved_backends],
        ]:
            # The annotation pipeline is skipped for the GPU backend, as it does not support quantized data types
            pipelines = (
                ("export",)
                if backend == QnnExecuTorchBackendType.kGpuBackend
                else ("annotation", "export")
            )
            for pipeline in pipelines:
                with self.subTest(backend=backend, pipeline=pipeline):
                    ep = torch.export.export(HardSigmoid(), sample_input, strict=True)
                    pm = get_qnn_pass_manager_cls(backend)()
                    if pipeline == "annotation":
                        pm.transform_for_annotation_pipeline(ep.graph_module)
                    else:
                        pm.transform_for_export_pipeline(ep)
                    has_target = any(
                        n.target == target for n in ep.graph_module.graph.nodes
                    )
                    self.assertNotEqual(
                        has_target,
                        should_decompose,
                        f"hardsigmoid {'should' if should_decompose else 'should NOT'} be decomposed for {backend.name}",
                    )

    def test_partitioner_falls_back_on_op_without_visitor(self):
        """QnnOperatorSupport must reject an op that has no node visitor by returning
        False (CPU fallback), not by KeyError-ing on the node_visitors lookup.

        Uses aten.frac.default: it has no QNN node visitor and is not on any partition
        operator list, so it exercises the missing-visitor guard directly. (bitwise_not,
        the op that first surfaced this on Mamba2, is now in to_be_implemented_operator,
        which returns earlier and would not reach the guard.)
        """

        class FracModule(torch.nn.Module):
            def forward(self, x):
                return torch.frac(x) + x

        sample_input = (torch.randn(1, 4),)

        # Guard against a vacuous test: the visitor-less op must actually be present.
        exported = torch.export.export(FracModule().eval(), sample_input)
        self.assertTrue(
            any(
                node.op == "call_function" and "frac" in str(node.target)
                for node in exported.graph.nodes
            ),
            "expected aten.frac.default in the traced graph",
        )

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            # Must not raise KeyError: frac falls back to CPU; the rest may delegate.
            edge = to_edge_transform_and_lower_to_qnn(
                FracModule().eval(), sample_input, compiler_specs
            )
            edge.to_executorch()
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                skip_without_qnn_sdk(self, e)
            raise

    def test_stack_outside_pack_ranks_is_decomposed(self):
        class Stack(torch.nn.Module):
            def forward(self, x):
                return torch.stack([x, x]) + 1

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        # HTP Pack takes inputs of rank 1 to 4. Lowering must not fail with
        # "aten.stack ... was not decomposed or delegated".
        for shape in [(), (1, 2, 3, 4, 5)]:
            with self.subTest(input_rank=len(shape)):
                try:
                    to_edge_transform_and_lower_to_qnn(
                        Stack().eval(), (torch.randn(shape),), compiler_specs
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise

    def test_partitioner_keeps_constant_only_subgraph_on_cpu(self):
        class ConstantBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(4, 16))
                self.register_buffer("weight", torch.randn(16, 16))

            def forward(self, x):
                # With relu on CPU, table * 2 - 1 reads nothing at runtime, and
                # with the bfloat16 cast on CPU, neither does the linear. So the
                # linear must be decomposed for this to lower.
                linear = torch.nn.functional.linear(self.table, self.weight)
                cast = linear.to(torch.bfloat16).float()
                return x + torch.relu(self.table * 2 - 1) + cast

        class DetourBranch(ConstantBranch):
            def forward(self, x):
                # The linear reaches the add only through CPU casts, so it is
                # decomposed, but the partition of x + 1 takes its pieces.
                linear = torch.nn.functional.linear(self.table, self.weight)
                cast = (x + 1).to(torch.bfloat16).float()
                return linear.to(torch.bfloat16).float() + cast

        sample_input = (torch.randn(4, 16),)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        # HTP accepts the pieces of a decomposed linear that a partition takes.
        for model, takes_pieces in ((ConstantBranch(), False), (DetourBranch(), True)):
            with self.subTest(model=type(model).__name__):
                try:
                    edge = to_edge_transform_and_lower_to_qnn(
                        model.eval(),
                        sample_input,
                        compiler_specs,
                        skip_node_op_set={"aten.relu.default"},
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise
                edge.to_executorch()
                self.assertEqual(
                    takes_pieces, exir_ops.edge.aten.mm.default in delegated_ops(edge)
                )

    def test_htp_decomposes_kept_whole_ops_left_on_cpu(self):
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        sample_input = (torch.randn(4, 16),)
        with stubbed_qnn_backend():
            edge = to_edge_transform_and_lower_to_qnn(
                ConstantBranches().eval(), sample_input, compiler_specs
            )
            for model in (
                NoRuntimeInput(),
                CastSplit(),
                CastSplitInput(),
                SplitAfterAdd(),
                ReluBeforeCast(),
            ):
                with self.subTest(model=type(model).__name__):
                    to_edge_transform_and_lower_to_qnn(
                        model.eval(), sample_input, compiler_specs
                    )
        top_level_ops = [
            node.target
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function"
        ]
        self.assertNotIn(exir_ops.edge.aten.linear.default, top_level_ops)
        self.assertIn(exir_ops.edge.aten.linear.default, delegated_ops(edge))
        self.assertIn(exir_ops.edge.aten.add.Tensor, delegated_ops(edge))
        edge.to_executorch()

    def test_htp_keeps_linked_kept_whole_ops_whole(self):
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        sample_input = (torch.randn(4, 16),)
        for model in (
            MaskedReader(),
            ConstantOutput(),
            CastSplitLinked(),
            LinearOfInput(),
            SharedConstant(),
            UnlinkedSplit(),
            MutatedTable(),
        ):
            with self.subTest(model=type(model).__name__), stubbed_qnn_backend():
                edge = to_edge_transform_and_lower_to_qnn(
                    model.eval(), sample_input, compiler_specs
                )
                self.assertIn(exir_ops.edge.aten.linear.default, delegated_ops(edge))

    def test_htp_keeps_quantized_ops_whole(self):
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        sample_input = (torch.randn(4, 16),)
        quantized = prepare_pt2e(
            torch.export.export(QuantizedCast().eval(), sample_input).module(),
            QnnQuantizer(),
        )
        quantized(*sample_input)
        with stubbed_qnn_backend():
            edge = to_edge_transform_and_lower_to_qnn(
                convert_pt2e(quantized),
                sample_input,
                stubbed_compiler_specs(generate_htp_compiler_spec(use_fp16=False)),
            )
        self.assertIn(exir_ops.edge.aten.linear.default, delegated_ops(edge))

        # These ops are left on CPU, but stay whole. exir renames each op it
        # keeps whole while it filters, so the linear must still count as
        # quantized once the dequantize of its weight is renamed.
        weight_only = QuantizedWeight().eval()
        quantize_(
            weight_only,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
        )
        for model, quantized_ops in (
            (weight_only, {torch.ops.torchao.dequantize_affine.default}),
            (
                GroupQuantizedWeight().eval(),
                {
                    torch.ops.quantized_decomposed.quantize_per_channel_group.default,
                    torch.ops.quantized_decomposed.dequantize_per_channel_group.default,
                },
            ),
        ):
            with self.subTest(model=type(model).__name__):
                kept_whole_ops = _replace_aten_ops_with_transformed_ops(
                    "forward",
                    torch.export.export(model, sample_input),
                    {"forward": [QnnPartitioner(compiler_specs)]},
                )
                self.assertLessEqual(
                    {torch.ops.aten.linear.default, *quantized_ops}, kept_whole_ops
                )

    def test_kept_whole_filter_is_filter_fn_on_edge_and_gpu(self):
        sample_input = (torch.randn(4, 16),)
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        with stubbed_qnn_backend():
            edge = to_edge_transform_and_lower_to_qnn(
                ConstantBranches().eval(), sample_input, compiler_specs
            )
        # On the lowered program the filter must not let a kept-whole op left
        # on CPU through, so exir still reports it.
        _, keep_whole = QnnPartitioner(compiler_specs).ops_to_not_decompose(
            edge.exported_program()
        )
        self.assertIs(keep_whole, filter_fn)
        # GPU drops no partitions, so it keeps the same ops whole as before.
        _, keep_whole = QnnPartitioner(
            stubbed_compiler_specs(generate_gpu_compiler_spec())
        ).ops_to_not_decompose(
            torch.export.export(ConstantBranches().eval(), sample_input)
        )
        self.assertIs(keep_whole, filter_fn)

    def test_htp_kept_whole_filter_ignores_skips(self):
        # A skip of either kind that changed what is decomposed would rename
        # later nodes, so a skip_node_id_set entry could name a different node.
        # Skipping the relu cuts the linear off, so the model cannot be lowered.
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        ep = torch.export.export(SkippedRelu().eval(), (torch.randn(4, 16),))
        _, keep_whole = QnnPartitioner(compiler_specs).ops_to_not_decompose(ep)
        _, keep_whole_skipped = QnnPartitioner(
            compiler_specs, skip_node_op_set={"aten.relu.default"}
        ).ops_to_not_decompose(ep)
        self.assertEqual(
            [keep_whole(node) for node in ep.graph.nodes],
            [keep_whole_skipped(node) for node in ep.graph.nodes],
        )

    def test_htp_kept_whole_filter_counts_cond_and_item_as_cpu(self):
        # QNN's passes cannot lower either model yet, so the filter is checked
        # alone.
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        # With a skip_node_id_set, the renaming is worth a warning.
        for model, skip_node_id_set, level in (
            (CondAfterLinear(), None, "INFO"),
            (ItemAfterLinear(), {"linear"}, "WARNING"),
        ):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model.eval(), (torch.randn(4, 16),))
                _, keep_whole = QnnPartitioner(
                    compiler_specs, skip_node_id_set=skip_node_id_set
                ).ops_to_not_decompose(ep)
                linear = next(
                    node
                    for node in ep.graph.nodes
                    if node.target == torch.ops.aten.linear.default
                )
                with self.assertLogs(QnnPartitioner.__module__, "INFO") as logs:
                    self.assertFalse(keep_whole(linear))
                self.assertEqual(level, logs.records[-1].levelname)
                self.assertIn(linear.name, logs.output[-1])

    def test_htp_kept_whole_log_names_only_newly_decomposed_ops(self):
        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )
        cases = [
            (IntStackBesideLinear(), torch.ops.aten.stack.default),
            (EluBesideLinear(), torch.ops.aten.elu.default),
        ]
        for model, left_out in cases:
            with self.subTest(left_out=left_out):
                ep = torch.export.export(model.eval(), (torch.randn(4, 16),))
                _, keep_whole = QnnPartitioner(compiler_specs).ops_to_not_decompose(ep)
                nodes = {node.target: node for node in ep.graph.nodes}
                linear = nodes[torch.ops.aten.linear.default]
                with self.assertLogs(QnnPartitioner.__module__, "INFO") as logs:
                    self.assertFalse(keep_whole(linear))
                self.assertIn(f"'{linear.name}'", logs.output[-1])
                self.assertNotIn(f"'{nodes[left_out].name}'", logs.output[-1])

    def test_htp_kept_whole_estimate_avoids_quadratic_steps(self):
        class Chain(torch.nn.Module):
            def __init__(self, length):
                super().__init__()
                self.scales = torch.nn.ParameterList(
                    torch.rand(16) for _ in range(length)
                )

            def forward(self, x):
                # Each scale * 2 reads only constants, like the dequantize of a
                # quantized weight.
                for scale in self.scales:
                    x = torch.relu(x * (scale * 2) + x)
                return x

        compiler_specs = stubbed_compiler_specs(
            generate_htp_compiler_spec(use_fp16=True)
        )

        # Large models have tens of thousands of nodes and constants. The time
        # to partition grows with the square of the graph size, and so does
        # going over the input specs of the graph signature for each input, or
        # running the estimate for each node exir filters.
        def input_spec_reads(length):
            ep = torch.export.export(Chain(length), (torch.randn(4, 16),))
            input_specs = ep.graph_signature.input_specs
            nodes = [node for node in ep.graph.nodes if node.op == "call_function"]
            with patch.object(
                CapabilityBasedPartitioner,
                "propose_partitions",
                autospec=True,
                side_effect=CapabilityBasedPartitioner.propose_partitions,
            ) as partition, patch.object(
                ExportGraphSignature,
                "input_specs",
                new_callable=PropertyMock,
                return_value=input_specs,
                create=True,
            ) as read_input_specs:
                _, keep_whole = QnnPartitioner(compiler_specs).ops_to_not_decompose(ep)
                for node in nodes:
                    keep_whole(node)
                QnnPartitioner._drop_partitions_without_runtime_inputs(
                    [Partition(nodes=[node]) for node in nodes], ep
                )
            partition.assert_not_called()
            return read_input_specs.call_count

        reads = input_spec_reads(2)
        self.assertGreater(reads, 0)
        self.assertEqual(reads, input_spec_reads(4))

    def test_partitioner_keeps_integer_mutable_buffers_on_cpu(self):
        class Counter(torch.nn.Module):
            def __init__(self, step, dtype):
                super().__init__()
                self.step = step
                self.register_buffer("count", torch.zeros(2, dtype=dtype))
                self.register_buffer("total", torch.zeros(2))

            def forward(self, x):
                self.count.add_(self.step(x))
                self.total.add_(x.sum(-1))
                return x * 2, self.count + 0, self.total + 0

        sample_input = (torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.1, 0.3]]),)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        steps = {
            "one": lambda x: 1,
            "argmax": lambda x: x.argmax(-1),
            "sum_of_bools": lambda x: (x > 0.5).sum(-1),
        }
        for name, step in steps.items():
            for dtype in (torch.int64, torch.int32):
                with self.subTest(step=name, dtype=dtype):
                    try:
                        edge = to_edge_transform_and_lower_to_qnn(
                            Counter(step, dtype).eval(), sample_input, compiler_specs
                        )
                    except RuntimeError as e:
                        if "Failed to initialize QNN backend" in str(e):
                            skip_without_qnn_sdk(self, e)
                        raise
                    ep = edge.exported_program()
                    self.assertIn(
                        "count", ep.graph_signature.buffers_to_mutate.values()
                    )
                    mutated = [
                        lowered.original_module.graph_signature.buffers_to_mutate
                        for lowered in get_lowered_backend_modules(ep.graph_module)
                    ]
                    # Float buffers are still delegated.
                    self.assertEqual(
                        ["total"], [buffer for m in mutated for buffer in m.values()]
                    )

    def test_partitioner_drops_partitions_of_only_inserted_casts(self):
        # Only the I64toI32 mark counts, not the op: other passes insert
        # _to_copy too.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        casts = [
            graph.call_function(
                exir_ops.edge.aten._to_copy.default, (x,), {"dtype": torch.int32}
            )
            for _ in range(3)
        ]
        add = graph.call_function(exir_ops.edge.aten.add.Tensor, (casts[2], 1))
        casts[0].meta[QCOM_INSERTED_CAST] = True
        casts[2].meta[QCOM_INSERTED_CAST] = True
        partitions = [
            Partition(id=0, nodes=[casts[0]]),
            Partition(id=1, nodes=[casts[1]]),
            Partition(id=2, nodes=[casts[2], add]),
        ]
        self.assertEqual(
            [1, 2],
            [
                partition.id
                for partition in QnnPartitioner._drop_partitions_of_inserted_casts(
                    partitions
                )
            ],
        )

    def test_partitioner_keeps_partitions_of_inserted_casts_on_cpu(self):
        class SumIntoInput(torch.nn.Module):
            def forward(self, x, k, mask):
                return (x > 0.5).sum(-1).masked_scatter(mask, k)

        class UserCast(torch.nn.Module):
            def forward(self, x, k, mask):
                return x.to(torch.int32)

        # sum and masked_scatter stay on CPU, so the int32 casts I64toI32 adds
        # for k and for the sum would form a partition of their own. A cast
        # the model makes is still delegated.
        sample_input = (
            torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.1, 0.3]]),
            torch.tensor([3, 4]),
            torch.tensor([True, False]),
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        for module, delegated in (
            (SumIntoInput(), [["aten.gt.Tensor"]]),
            (UserCast(), [["dim_order_ops._to_dim_order_copy.default"]]),
        ):
            with self.subTest(type(module).__name__):
                try:
                    edge = to_edge_transform_and_lower_to_qnn(
                        module, sample_input, compiler_specs
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise
                ep = edge.exported_program()
                self.assertEqual(
                    delegated,
                    [
                        [
                            node.target.__name__
                            for node in lowered.original_module.graph.nodes
                            if node.op == "call_function"
                        ]
                        for lowered in get_lowered_backend_modules(ep.graph_module)
                    ],
                )
                torch.testing.assert_close(
                    ep.module()(*sample_input), module(*sample_input)
                )

    def test_partitioner_keeps_constants_above_rank_5_on_cpu(self):
        class Rank6Constant(torch.nn.Module):
            def __init__(self, use):
                super().__init__()
                self.use = use

            def forward(self, x):
                return x * 2, self.use(torch.zeros(1, 2, 1, 1, 3, 4))

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        # Either way the constant would be an output of the multiply's partition.
        uses = {"graph output": lambda c: c, "CPU op": lambda c: c.cumsum(5)}
        for name, use in uses.items():
            with self.subTest(use=name):
                try:
                    edge = to_edge_transform_and_lower_to_qnn(
                        Rank6Constant(use).eval(), (torch.randn(2, 3),), compiler_specs
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise
                self.assertIn(exir_ops.edge.aten.mul.Tensor, delegated_ops(edge))
                self.assertNotIn(exir_ops.edge.aten.full.default, delegated_ops(edge))

    def test_partitioner_keeps_bool_cat_of_static_tensor_on_cpu(self):
        class CatConstant(torch.nn.Module):
            def __init__(self, dtype, constant):
                super().__init__()
                self.dtype = dtype
                self.constant = constant
                self.register_buffer("ones", torch.ones(2, 2, dtype=dtype))

            def forward(self, x):
                return torch.cat([(x > 0).to(self.dtype), self.constant(self)], 1)

        constants = {
            "full": lambda m: torch.full((2, 2), 1, dtype=m.dtype),
            "buffer": lambda m: m.ones,
        }
        # Quantized first: some QNN SDKs crash on a quantized lowering that
        # follows an fp16 one in the same process. Quantized HTP graphs read
        # the constant right, so the cat stays.
        module = CatConstant(torch.bool, constants["full"]).eval()
        sample_input = (torch.randn(2, 3),)
        prepared = prepare_pt2e(
            torch.export.export(module, sample_input, strict=True).module(),
            QnnQuantizer(),
        )
        prepared(*sample_input)
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                convert_pt2e(prepared),
                sample_input,
                generate_qnn_executorch_compiler_spec(
                    soc_model=QcomChipset.SM8650,
                    backend_options=generate_htp_compiler_spec(use_fp16=False),
                ),
            )
        except RuntimeError as e:
            if "Failed to initialize QNN backend" in str(e):
                skip_without_qnn_sdk(self, e)
            raise
        self.assertIn(exir_ops.edge.aten.cat.default, delegated_ops(edge))

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        for dtype in (torch.bool, torch.float32):
            for name, constant in constants.items():
                with self.subTest(dtype=dtype, constant=name):
                    try:
                        edge = to_edge_transform_and_lower_to_qnn(
                            CatConstant(dtype, constant).eval(),
                            (torch.randn(2, 3),),
                            compiler_specs,
                        )
                    except RuntimeError as e:
                        if "Failed to initialize QNN backend" in str(e):
                            skip_without_qnn_sdk(self, e)
                        raise
                    self.assertIn(exir_ops.edge.aten.gt.Tensor, delegated_ops(edge))
                    self.assertEqual(
                        dtype != torch.bool,
                        exir_ops.edge.aten.cat.default in delegated_ops(edge),
                    )

    def test_partitioner_falls_back_on_bfloat16(self):
        class Bf16Region(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16).to(torch.bfloat16)

            def forward(self, x):
                return self.linear(x.to(torch.bfloat16)).to(torch.float32) + 1

        sample_input = (torch.randn(4, 16),)
        # Quantized first: some QNN SDKs crash on a quantized lowering that
        # follows an fp16 one in the same process.
        for quant_dtype, weight_only in (
            (QuantDtype.use_8a8w, False),
            (None, False),
            (None, True),
        ):
            with self.subTest(quant_dtype=quant_dtype, weight_only=weight_only):
                module = Bf16Region().eval()
                if weight_only:
                    quantize_(
                        module,
                        IntxWeightOnlyConfig(
                            weight_dtype=torch.int8, granularity=PerAxis(0)
                        ),
                    )
                if quant_dtype is not None:
                    quantizer = QnnQuantizer()
                    quantizer.set_default_quant_config(quant_dtype=quant_dtype)
                    module = prepare_pt2e(
                        torch.export.export(module, sample_input).module(), quantizer
                    )
                    module(*sample_input)
                    module = convert_pt2e(module)
                compiler_specs = generate_qnn_executorch_compiler_spec(
                    soc_model=QcomChipset.SM8650,
                    backend_options=generate_htp_compiler_spec(
                        use_fp16=quant_dtype is None
                    ),
                )
                try:
                    edge = to_edge_transform_and_lower_to_qnn(
                        module, sample_input, compiler_specs
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise
                cpu_nodes = [
                    node
                    for node in edge.exported_program().graph.nodes
                    if node.op == "call_function" and "delegate" not in str(node.target)
                ]
                cpu_dtypes = {
                    node.meta["val"].dtype
                    for node in cpu_nodes
                    if isinstance(node.meta.get("val"), torch.Tensor)
                }
                # Unquantized bfloat16 stays on CPU; quantized bfloat16 has a
                # QNN type and is delegated.
                self.assertEqual(torch.bfloat16 in cpu_dtypes, quant_dtype is None)
                # Portable kernels have no linear, so one left on CPU must be
                # decomposed.
                self.assertNotIn(
                    exir_ops.edge.aten.linear.default,
                    [node.target for node in cpu_nodes],
                )

    def test_partitioner_keeps_positional_sin_cos_in_float32(self):
        class Embedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, t, x):
                period = 4e-3 * 1000 ** torch.linspace(0, 1, 8)
                return torch.sin(t * (2 * torch.pi / period)) + torch.cos(
                    self.linear(x)
                )

        sample_input = (torch.rand(4, 1), torch.randn(4, 8))
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
        passes_job[TagPositionalSinCos][QCOM_PASS_ACTIVATE_KEY] = False
        for job in (None, passes_job):
            with self.subTest(tag_positional_sin_cos=job is None):
                try:
                    edge = to_edge_transform_and_lower_to_qnn(
                        Embedding().eval(),
                        sample_input,
                        compiler_specs,
                        passes_job=job,
                    )
                except RuntimeError as e:
                    if "Failed to initialize QNN backend" in str(e):
                        skip_without_qnn_sdk(self, e)
                    raise
                nodes = [
                    node
                    for node in edge.exported_program().graph.nodes
                    if node.op == "call_function"
                ]
                cpu_sin = [
                    node
                    for node in nodes
                    if node.target == exir_ops.edge.aten.sin.default
                ]
                # The sin of a positional angle and the mul computing the angle
                # run on CPU in float32 unless the pass is off, while the cos
                # of an activation stays delegated with the linear before it,
                # in one delegate either way.
                self.assertEqual(len(cpu_sin), int(job is None))
                for sin in cpu_sin:
                    self.assertEqual(sin.args[0].target, exir_ops.edge.aten.mul.Tensor)
                self.assertNotIn(
                    exir_ops.edge.aten.cos.default, [node.target for node in nodes]
                )
                self.assertEqual(
                    sum("call_delegate" in str(node.target) for node in nodes), 1
                )

    @staticmethod
    def _positional_edge_graph(
        module, sample_input, compiler_specs, backend_type, **kwargs
    ):
        # The graph after the QNN to_edge passes. Without the partitioner,
        # the ops it keeps whole, like stack and linear, are decomposed.
        pass_manager = get_qnn_pass_manager_cls(backend_type)()
        program = pass_manager.transform_for_export_pipeline(
            torch.export.export(
                module,
                sample_input,
                dynamic_shapes=kwargs.pop("dynamic_shapes", None),
                strict=True,
            )
        )
        return to_edge_transform_and_lower(
            program,
            transform_passes=pass_manager.get_to_edge_transform_passes(
                program, compiler_specs=compiler_specs, **kwargs
            ),
            compile_config=qnn_edge_config(),
        ).exported_program()

    @staticmethod
    def _computing_nodes(node):
        nodes, stack = {node}, list(node.all_input_nodes)
        while stack:
            arg = stack.pop()
            if arg not in nodes and arg.op == "call_function":
                nodes.add(arg)
                stack.extend(arg.all_input_nodes)
        return nodes

    @classmethod
    def _is_fully_tagged(cls, node):
        return all(
            arg.meta.get(QCOM_FALLBACK_NODE)
            for arg in cls._computing_nodes(node)
            if not isinstance(arg.meta["val"], torch.SymInt)
        )

    def test_tag_positional_sin_cos(self):
        class Embeddings(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("freqs", torch.linspace(1, 1000, 8))
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, t, mask, x):
                # An integer cumsum is exact, so it is followed as a position.
                positions = mask.cumsum(-1).unsqueeze(-1).float()
                # Offset positions, as a decoder step after a prefix builds them.
                offsets = (mask > 0).sum(-1) + torch.ones(4, dtype=torch.bool).cumsum(0)
                float_offsets = (mask > 0).sum(-1) + torch.ones(4).cumsum(0)
                return (
                    torch.sin(t * torch.linspace(1, 1000, 8)),
                    torch.cos(positions * self.freqs),
                    torch.cos(offsets.unsqueeze(-1).float() * self.freqs),
                    torch.cos(float_offsets.unsqueeze(-1) * self.freqs),
                    torch.sin(
                        t * (torch.arange(8) * -torch.log(torch.tensor(1e4))).exp()
                    ),
                    # Buffers, as large as a batch one angle, are not inputs,
                    # nor are integers built from constants, like linspace's.
                    torch.sin(t[:1] * self.freqs),
                    torch.sin(t[:1] * torch.linspace(1, 1000, 8)),
                    # Integer positions may read a mask as large as the angle.
                    torch.cos((x > 0).sum(-1, keepdim=True).float() * self.freqs),
                    torch.sin(self.linear(x) * self.freqs),
                    torch.sin(t),
                    torch.sin(t.view(-1)),
                    # Larger than t, but made by views only, so it holds no
                    # new angles.
                    torch.sin(t.expand(4, 8)),
                    torch.sin((t * self.freqs).half()),
                    # Integers computed from an activation stay delegated, and
                    # so does their angle.
                    torch.cos(
                        (self.linear(x).sum(-1) > 0).cumsum(-1).unsqueeze(-1).float()
                        * self.freqs
                    ),
                    torch.cos(
                        (x.sum(-1) > 0).cumsum(-1).unsqueeze(-1).float()
                        * torch.linspace(1, 1000, 16)
                    ),
                    # Not larger than its inputs, so not a positional angle.
                    torch.sin(x * 30),
                    torch.sin(torch.cat([t, t]) * 30),
                    torch.sin(mask.float() * 30),
                    torch.sin(mask * 30),
                )

        htp_fp16 = stubbed_compiler_specs(generate_htp_compiler_spec(use_fp16=True))
        htp, gpu = (
            QnnExecuTorchBackendType.kHtpBackend,
            QnnExecuTorchBackendType.kGpuBackend,
        )
        passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
        passes_job[TagPositionalSinCos][QCOM_PASS_ACTIVATE_KEY] = False
        sample_input = (
            torch.rand(4, 1),
            torch.ones(4, dtype=torch.int64),
            torch.randn(4, 8),
        )
        none = [False] * 19
        for name, compiler_specs, backend_type, quantized, expected in (
            ("htp fp16", htp_fp16, htp, None, [True] * 8 + none[8:]),
            ("quantized sin and cos", htp_fp16, htp, "sin_cos", none),
            ("quantized angles", htp_fp16, htp, "angle", none),
            (
                "htp quantized",
                stubbed_compiler_specs(generate_htp_compiler_spec(use_fp16=False)),
                htp,
                None,
                none,
            ),
            (
                "gpu",
                stubbed_compiler_specs(generate_gpu_compiler_spec()),
                gpu,
                None,
                none,
            ),
        ):
            with self.subTest(name):
                edge_program = self._positional_edge_graph(
                    Embeddings(),
                    sample_input,
                    compiler_specs,
                    backend_type,
                    passes_job=passes_job if quantized else None,
                )
                graph_module = edge_program.graph_module
                outputs = graph_module.graph.output_node().args[0]
                for node in outputs if quantized else []:
                    node = node.args[0] if quantized == "angle" else node
                    node.meta[QCOM_QUANT_ATTRS] = {}
                # The GPU pass manager does not list the pass, so run it here
                # to check that it leaves GPU graphs alone.
                if quantized or backend_type == gpu:
                    TagPositionalSinCos(edge_program, compiler_specs)(graph_module)
                # A tagged sin or cos takes its whole angle with it, integers
                # included, so the casts I64toI32 adds stay with their
                # producers.
                self.assertEqual(
                    [self._is_fully_tagged(node) for node in outputs], expected
                )
                for nodes in (outputs, [node.args[0] for node in outputs]):
                    self.assertEqual(
                        [node.meta.get(QCOM_FALLBACK_NODE, False) for node in nodes],
                        expected,
                    )
                # Ops that only the untagged outputs read, like the linear of
                # sin(linear(x) * freqs), stay delegated.
                tagged = {
                    node
                    for node in graph_module.graph.nodes
                    if node.meta.get(QCOM_FALLBACK_NODE)
                }
                kept = [
                    self._computing_nodes(n) for n, e in zip(outputs, expected) if e
                ]
                self.assertLessEqual(tagged, set().union(*kept))

    def test_tag_positional_sin_cos_dynamic_size(self):
        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.rand(8))

            def forward(self, x, m: int):
                n = x.shape[0]
                return (
                    torch.cos(torch.arange(n)[:, None] * self.inv_freq),
                    torch.cos(
                        torch.arange(n, dtype=torch.float32)[:, None] * self.inv_freq
                    ),
                    torch.cos(
                        torch.arange(m, dtype=torch.float32)[:, None] * self.inv_freq
                    ),
                    # Without a frequency axis, an int input counts as a size.
                    torch.sin(torch.arange(m) * 0.5),
                    torch.sin(torch.arange(n) * 0.5),
                    torch.cos(x * 30),
                )

        # With a dynamic size the aranges read a symbolic integer, a size or
        # an int input.
        graph_module = self._positional_edge_graph(
            Rotary(),
            (torch.rand(4, 8), 6),
            self._htp_compiler_specs(True),
            QnnExecuTorchBackendType.kHtpBackend,
            dynamic_shapes={
                "x": {0: torch.export.Dim("n", min=2, max=1024)},
                "m": torch.export.Dim.DYNAMIC(min=1, max=1024),
            },
        ).graph_module
        self.assertEqual(
            [
                self._is_fully_tagged(n)
                for n in graph_module.graph.output_node().args[0]
            ],
            [True, True, True, True, True, False],
        )

    def test_tag_positional_sin_cos_size_limit(self):
        class Fourier(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x[..., None] * torch.linspace(1, 1000, 16))

        class SlicedTable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.rand(64))

            def forward(self, x):
                table = torch.arange(32768).float()[:, None] * self.inv_freq
                return x + torch.cos(table[:16])

        def is_tagged(x, dynamic_shapes=None, module=Fourier):
            graph_module = self._positional_edge_graph(
                module(),
                (x,),
                self._htp_compiler_specs(True),
                QnnExecuTorchBackendType.kHtpBackend,
                dynamic_shapes=dynamic_shapes,
            ).graph_module
            (sin_cos,) = [
                node
                for node in graph_module.graph.nodes
                if node.target
                in (exir_ops.edge.aten.sin.default, exir_ops.edge.aten.cos.default)
            ]
            return self._is_fully_tagged(sin_cos)

        # 1024 positions times 256 is the most a tagged sin or cos may have.
        self.assertTrue(is_tagged(torch.rand(1024, 16)))
        self.assertFalse(is_tagged(torch.rand(1025, 16)))
        # The ops of the angle count too.
        self.assertFalse(is_tagged(torch.rand(16, 64), module=SlicedTable))
        # A dynamic size counts with its largest value, and one without a max
        # is left delegated with a debug message.
        with self.assertLogs(
            "executorch.backends.qualcomm._passes.tag_positional_sin_cos",
            level=logging.DEBUG,
        ) as logs:
            self.assertFalse(
                is_tagged(torch.rand(4, 16), {"x": {0: torch.export.Dim("n", min=2)}})
            )
        self.assertTrue(any("needs a max" in line for line in logs.output))
        for max_size, expected in ((1024, True), (1025, False)):
            self.assertEqual(
                is_tagged(
                    torch.rand(4, 16),
                    {"x": {0: torch.export.Dim("n", min=2, max=max_size)}},
                ),
                expected,
            )

    def test_tag_positional_sin_cos_mutable_buffer(self):
        class State(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4, 8))
                self.register_buffer("cache", torch.zeros(4, 8))
                self.register_buffer("parts", torch.zeros(4, 8))
                self.register_buffer("scaled", torch.zeros(4, 8))
                self.register_buffer("step", torch.zeros(1))
                self.register_buffer("freqs", torch.linspace(1, 1000, 8))

            def forward(self, x):
                outputs = (
                    torch.sin(self.state * 30),
                    torch.sin(self.cache * 30),
                    torch.sin(self.parts * 30),
                    torch.sin(self.scaled * 30),
                    torch.sin(self.step[:, None] * self.freqs),
                )
                self.state.add_(x)
                self.cache[:, :2] = x[:, :2]
                self.parts.split(2)[0].add_(x[:2])
                torch.mul(x, 2, out=self.scaled)
                self.step.add_(1)
                return outputs

        # A buffer the graph changes, directly, through a view or as an out=
        # argument, counts as an input, so sin(state * 30) is not larger than what it reads, while a
        # step counter spread over frequencies still is. The program fresh
        # from torch.export writes its buffers in place, and the edge program
        # lists them in its signature instead.
        compiler_specs = self._htp_compiler_specs(True)
        sample_input = (torch.rand(4, 8),)
        exported_program = self._positional_edge_graph(
            State(), sample_input, compiler_specs, QnnExecuTorchBackendType.kHtpBackend
        )
        edge_program = to_edge(
            torch.export.export(State(), sample_input, strict=True)
        ).exported_program()
        TagPositionalSinCos(edge_program, compiler_specs)(edge_program.graph_module)
        for program in (exported_program, edge_program):
            sin_nodes = [
                node
                for node in program.graph.nodes
                if node.target == exir_ops.edge.aten.sin.default
            ]
            self.assertEqual(
                [self._is_fully_tagged(node) for node in sin_nodes],
                [False, False, False, False, True],
            )

    def test_tag_positional_sin_cos_long_integer_chain(self):
        class Chain(torch.nn.Module):
            def forward(self, mask):
                positions = mask.cumsum(-1)
                for _ in range(1000):
                    positions = positions + 1
                return torch.sin(
                    positions.unsqueeze(-1).float() * torch.linspace(1, 1000, 8)
                )

        # Longer than the recursion limit once I64toI32 adds its casts.
        graph_module = self._positional_edge_graph(
            Chain(),
            (torch.ones(4, dtype=torch.int64),),
            self._htp_compiler_specs(True),
            QnnExecuTorchBackendType.kHtpBackend,
        ).graph_module
        self.assertTrue(
            self._is_fully_tagged(graph_module.graph.output_node().args[0][0])
        )

    def test_tag_positional_sin_cos_logs_without_compiler_specs(self):
        program = torch.export.export(
            torch.nn.Linear(8, 8), (torch.rand(4, 8),), strict=True
        )
        with self.assertLogs(
            "executorch.backends.qualcomm._passes.tag_positional_sin_cos",
            level=logging.DEBUG,
        ) as logs:
            TagPositionalSinCos(program)
        self.assertIn("no compiler_specs", logs.output[0])

    def test_to_edge_passes_keep_tag_positional_sin_cos_last(self):
        program = torch.export.export(
            torch.nn.Linear(8, 8), (torch.rand(4, 8),), strict=True
        )
        pass_manager = get_qnn_pass_manager_cls()()
        # A dep_table with no entry for it, like the base one, still runs it
        # last before ResolveDebugHandle, after I64toI32 and the other passes,
        # and so does one that orders a pass after ResolveDebugHandle.
        base_dep_table = get_qnn_pass_manager_cls(
            QnnExecuTorchBackendType.kGpuBackend
        ).get_passes_dependency_for_capture_program()
        late_dep_table = {RemoveRedundancy: [ResolveDebugHandle]}
        for name, dep_table in (
            ("default", None),
            ("base", base_dep_table),
            ("late", late_dep_table),
        ):
            with self.subTest(name):
                passes = pass_manager.get_to_edge_transform_passes(
                    program, dep_table=dep_table
                )
                self.assertEqual(
                    [type(p) for p in passes[-2:]],
                    [TagPositionalSinCos, ResolveDebugHandle],
                )

    @stubbed_recipe_setup()
    def test_qnn_recipe_can_turn_off_tag_positional_sin_cos(self):
        program = torch.export.export(
            torch.nn.Linear(8, 8), (torch.rand(4, 8),), strict=True
        )
        passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
        passes_job[TagPositionalSinCos][QCOM_PASS_ACTIVATE_KEY] = False
        passes = {}
        for job in (None, passes_job):
            recipe = QNNRecipeProvider().create_recipe(
                QNNRecipeType.FP16, soc_model=QcomChipset.SM8650, passes_job=job
            )
            edge_passes = recipe.lowering_recipe.edge_transform_passes[0](None, program)
            passes[job is None] = {type(p) for p in edge_passes}
        self.assertEqual(passes[True] - passes[False], {TagPositionalSinCos})
        self.assertEqual(passes[False] - passes[True], set())
        with self.assertRaisesRegex(ValueError, "must be a dict or None"):
            QNNRecipeProvider().create_recipe(
                QNNRecipeType.FP16, soc_model=QcomChipset.SM8650, passes_job=[]
            )
        with self.assertRaisesRegex(ValueError, "not by method name"):
            QNNRecipeProvider().create_recipe(
                QNNRecipeType.FP16,
                soc_model=QcomChipset.SM8650,
                passes_job={"forward": passes_job},
            )

    def test_partitioner_falls_back_on_data_dependent_shape(self):
        class MaskedStack(torch.nn.Module):
            def forward(self, x, mask):
                y = x[mask]
                return torch.stack([y, y], dim=-1).sum((0, 2)) + x

        sample_input = (torch.randn(8, 16), torch.randn(8) > 0)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                MaskedStack().eval(), sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "Failed to initialize QNN backend" in str(e):
                skip_without_qnn_sdk(self, e)
            raise
        cpu_ops = {
            node.target
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function"
        }
        self.assertIn(exir_ops.edge.aten.index.Tensor, cpu_ops)
        edge.to_executorch()

    def _assert_cpu_conv_keeps_its_weight_dequantize(
        self, quant_dtype, use_fp16, skip_node_id_set=None
    ):
        """A quantized conv kept off QNN must take its per-channel weight dequantize
        with it. FoldQDQ force-passes that dequantize, so it used to be lowered alone
        as a QNN Dequantize, which rejects per-channel encodings and fails context
        binary generation (#19908)."""

        class ConvDepthwiseConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = torch.nn.Conv2d(8, 16, 1)
                self.dw = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
                self.pw2 = torch.nn.Conv2d(16, 8, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.dw(torch.relu(self.pw1(x)))))

        sample_input = (torch.randn(1, 8, 16, 16),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=quant_dtype, is_conv_per_channel=True
        )
        prepared = prepare_pt2e(
            torch.export.export(ConvDepthwiseConv().eval(), sample_input).module(),
            quantizer,
        )
        prepared(*sample_input)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=use_fp16),
        )
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                convert_pt2e(prepared),
                sample_input,
                compiler_specs,
                skip_node_id_set=skip_node_id_set,
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                skip_without_qnn_sdk(self, e)
            raise

        # inspected before to_executorch(), which swaps in out-variant ops
        def cpu_nodes(target):
            return [
                node
                for node in edge.exported_program().graph.nodes
                if node.op == "call_function" and node.target == target
            ]

        convs = cpu_nodes(exir_ops.edge.aten.convolution.default)
        self.assertEqual([conv.name for conv in convs], ["aten_convolution_default_1"])
        dequants = cpu_nodes(
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        )
        self.assertTrue(dequants)
        for dequant in dequants:
            self.assertEqual(list(dequant.users), convs)
        edge.to_executorch()

    def test_skipped_conv_keeps_its_weight_dequantize_on_cpu(self):
        self._assert_cpu_conv_keeps_its_weight_dequantize(
            QuantDtype.use_16a8w,
            use_fp16=False,
            skip_node_id_set={"aten_convolution_default_1"},
        )

    def test_rejected_conv_keeps_its_weight_dequantize_on_cpu(self):
        # the HTP rejects fp16-activation/int8-weight depthwise conv
        self._assert_cpu_conv_keeps_its_weight_dequantize(
            QuantDtype.use_fp16a8w, use_fp16=True
        )

    def test_fp16a8w_conv_on_cpu_stays_fp16(self):
        """fp16a8w runs conv in fp16, but the HTP only takes it for 1x1 kernels, so a
        3x3 conv falls back to CPU. Its bias data and weight dequantize must be fp16
        as well, or to_executorch fails on a bias buffer that does not match its spec.
        """

        class ConvConvConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = torch.nn.Conv2d(8, 16, 1)
                self.k3 = torch.nn.Conv2d(16, 16, 3, padding=1)
                self.pw2 = torch.nn.Conv2d(16, 8, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.k3(torch.relu(self.pw1(x)))))

        sample_input = (torch.randn(1, 8, 16, 16),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_fp16a8w, is_conv_per_channel=True
        )
        prepared = prepare_pt2e(
            torch.export.export(ConvConvConv().eval(), sample_input).module(),
            quantizer,
        )
        prepared(*sample_input)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=False),
        )
        quantized = convert_pt2e(prepared)
        try:
            edge = to_edge_transform_and_lower_to_qnn(
                quantized, sample_input, compiler_specs
            )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                self.skipTest(f"QNN SDK not available: {e}")
            raise

        program = edge.exported_program()
        convs = [
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.aten.convolution.default
        ]
        self.assertEqual([conv.name for conv in convs], ["aten_convolution_default_1"])
        weight, bias = convs[0].args[1], convs[0].args[2]
        self.assertEqual(weight.kwargs.get("out_dtype"), torch.float16)
        bias_name = program.graph_signature.inputs_to_buffers[bias.name]
        self.assertEqual(program.state_dict[bias_name].dtype, torch.float16)
        # the cast must not reach back into the caller's module
        self.assertEqual(
            {t.dtype for t in quantized.state_dict().values() if t.is_floating_point()},
            {torch.float32},
        )
        edge.to_executorch()

    def test_lowering_accepts_exported_program_without_reexport(self):
        """to_edge_transform_and_lower_to_qnn must lower a caller-supplied
        ExportedProgram as captured, never re-exporting it.

        Callers that have already captured and patched a program - HF transformers'
        ExecuTorch exporter, for instance, which owns its own dynamic shapes and fx
        fixes - cannot express that through a module argument.

        A shape assertion alone does not discriminate here: re-exporting
        exported.module() reproduces the original graph, so the contract is checked
        by spying on torch.export.export and requiring that lowering never calls it.
        """

        class RetraceSensitive(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.narrow = False

            def forward(self, x):
                return torch.relu(x[:, :1] if self.narrow else x)

        module = RetraceSensitive().eval()
        exported = torch.export.export(module, (torch.randn(1, 4),), strict=True)

        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        with patch("torch.export.export", wraps=torch.export.export) as export_spy:
            try:
                edge = to_edge_transform_and_lower_to_qnn(
                    exported, None, compiler_specs
                )
            except RuntimeError as e:
                if "QNN" in str(e) or "qnn" in str(e):
                    skip_without_qnn_sdk(self, e)
                raise
            export_spy.assert_not_called()

        # The lowered graph is the one that was handed in, not a fresh capture.
        output_node = edge.exported_program("forward").graph.output_node()
        self.assertEqual(tuple(output_node.args[0][0].meta["val"].shape), (1, 4))

    def test_lowering_warns_when_inputs_ignored_for_exported_program(self):
        """inputs and dynamic_shapes cannot affect an already-captured program, so
        supplying them alongside one must warn rather than silently drop them."""

        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        sample_input = (torch.randn(1, 4),)
        exported = torch.export.export(Tiny().eval(), sample_input, strict=True)
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        try:
            with self.assertWarnsRegex(UserWarning, "inputs ignored"):
                to_edge_transform_and_lower_to_qnn(
                    exported, sample_input, compiler_specs
                )
        except RuntimeError as e:
            if "QNN" in str(e) or "qnn" in str(e):
                skip_without_qnn_sdk(self, e)
            raise

    def test_index_put_int64_value_not_quantized(self):
        """QNN's IndexPut annotator must skip a non-float (int64) value arg.

        Regression for MoE (Mixtral) routing, where index_put's value is an int64
        arange: annotating it makes quantize_per_tensor assert a float input, so
        to_executorch() fails. Only float tensors may be annotated.
        Exercised on both the HTP and LPAI annotators, which share the guard.
        """

        class IndexPutInt64Value(torch.nn.Module):
            def forward(self, x):
                buf = torch.zeros(4, dtype=torch.long)
                idx = torch.arange(
                    4, dtype=torch.int64
                )  # int64 value written by index_put
                buf = buf.index_put((torch.tensor([0, 1, 2, 3]),), idx)
                return x + buf.to(torch.float32)

        module = IndexPutInt64Value().eval()
        sample_input = (torch.randn(4),)

        def run_backend(backend):
            quantizer = QnnQuantizer(backend=backend)
            quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_8a8w)
            gm = (
                torch.export.export(module, sample_input)
                .run_decompositions({})
                .module()
            )
            prepared = prepare_pt2e(gm, quantizer)
            prepared(*sample_input)
            converted = convert_pt2e(prepared)
            # Re-export runs the quantize_per_tensor meta kernel; before the dtype
            # guard this raised "Expecting input to have dtype torch.float32" on the
            # int64 value.
            torch.export.export(converted, sample_input)

        # HTP is the core QNN quantizer backend; a failure here is a real regression,
        # not something to swallow. LPAI additionally needs quantized_aot_lib, so only
        # that missing optional dependency is allowed to skip the LPAI leg.
        run_backend(QnnExecuTorchBackendType.kHtpBackend)
        try:
            run_backend(QnnExecuTorchBackendType.kLpaiBackend)
        except Exception as e:
            if "quantized_aot_lib" in str(e):
                self.skipTest(f"LPAI quantizer unavailable: {e}")
            raise

    def test_expand_broadcast_preserves_rank0_input_mutation(self):
        """A rank-0 user input that is BOTH broadcast (needs rank promotion) AND mutated in
        place must keep its USER_INPUT_MUTATION write-back on the rank-0 value.

        Regression for the all-users rewrite: it redirected *every* consumer of the operand to
        the promoted (1,) view, including the in-place mutation, so the write-back became
        copy_(dst=(), src=(1,)) and to_executorch() failed with "expand: too few dimensions".
        Asserts the broadcast consumes the promoted view while the mutation path stays rank-0.
        """
        add = exir_ops.edge.aten.add.Tensor
        view_copy = exir_ops.edge.aten.view_copy.default

        exported = torch.export.export(
            BroadcastAndMutate().eval(),
            (torch.randn(1, 4), torch.tensor(0)),
            strict=True,
        )
        ep = to_edge(exported).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module

        broadcast = [
            n for n in gm.graph.nodes if n.target == add and n.meta["val"].dim() == 1
        ]
        self.assertTrue(broadcast)
        self.assertTrue(
            any(
                isinstance(a, torch.fx.Node) and a.target == view_copy
                for n in broadcast
                for a in n.args
            ),
            "the broadcast operand should be rank-promoted to a view_copy",
        )

        mutated = {
            spec.arg.name
            for spec in ep.graph_signature.output_specs
            if spec.kind == OutputKind.USER_INPUT_MUTATION
        }
        self.assertTrue(mutated)
        checked = 0
        for node in gm.graph.nodes:
            if node.name in mutated:
                checked += 1
                self.assertFalse(
                    any(
                        isinstance(a, torch.fx.Node) and a.target == view_copy
                        for a in node.args
                    ),
                    "rank promotion leaked into the USER_INPUT_MUTATION write-back path",
                )
        self.assertEqual(
            checked,
            len(mutated),
            "did not find every mutation-output node in the graph",
        )

    def test_expand_broadcast_promotes_per_consumer(self):
        """One tensor feeding two broadcasts of different output rank must be promoted
        independently per broadcast node, not shared. The all-users rewrite reshaped the
        operand once (to the first broadcast's rank) and redirected the second broadcast too,
        mis-shaping it."""
        add = exir_ops.edge.aten.add.Tensor
        view_copy = exir_ops.edge.aten.view_copy.default

        class TwoBroadcasts(torch.nn.Module):
            def forward(self, a2d, a1d, b):
                return (
                    a2d + b,
                    a1d + b,
                )  # b:(4,) -> rank-2 needs a view; rank-1 does not

        exported = torch.export.export(
            TwoBroadcasts().eval(),
            (torch.randn(3, 4), torch.randn(4), torch.randn(4)),
            strict=True,
        )
        ep = to_edge(exported).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module

        for node in gm.graph.nodes:
            if node.target != add:
                continue
            has_view = any(
                isinstance(a, torch.fx.Node) and a.target == view_copy
                for a in node.args
            )
            if node.meta["val"].dim() == 2:
                self.assertTrue(has_view, "rank-2 broadcast operand should be promoted")
            if node.meta["val"].dim() == 1:
                self.assertFalse(
                    has_view, "rank-1 broadcast operand should not be promoted"
                )

    def test_expand_broadcast_dedupes_shared_same_rank(self):
        """One operand feeding two broadcasts of the SAME output rank must be promoted with a
        single shared view_copy, not one per broadcast node.

        Regression for the per-node redirect (PR #21583): redirecting only the current node's
        input fixed the rank-0 mutation bug but created a separate view_copy for every broadcast
        consumer, so `a2d + b, a2d - b` (b:(4,)) produced two identical (1,4) reshapes. Deduping
        by (operand, new_rank) collapses them to one while keeping the rank-0 mutation path (whose
        operand is never rank-promoted) at rank-0 -- so this coexists with
        test_expand_broadcast_preserves_rank0_input_mutation.
        """
        add = exir_ops.edge.aten.add.Tensor
        sub = exir_ops.edge.aten.sub.Tensor
        view_copy = exir_ops.edge.aten.view_copy.default

        class TwoSameRankBroadcasts(torch.nn.Module):
            def forward(self, a2d, b):
                return a2d + b, a2d - b  # b:(4,) promoted to (1,4) for both

        exported = torch.export.export(
            TwoSameRankBroadcasts().eval(),
            (torch.randn(3, 4), torch.randn(4)),
            strict=True,
        )
        ep = to_edge(exported).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module

        views = [n for n in gm.graph.nodes if n.target == view_copy]
        self.assertEqual(
            len(views), 1, "the shared operand should yield exactly one view_copy"
        )
        shared = views[0]
        for target in (add, sub):
            broadcast = [n for n in gm.graph.nodes if n.target == target]
            self.assertTrue(broadcast)
            self.assertTrue(
                all(shared in n.args for n in broadcast),
                "both same-rank broadcasts should consume the shared view_copy",
            )

        exported = torch.export.export(
            BroadcastAndMutate().eval(),
            (torch.randn(1, 4), torch.tensor(0)),
            strict=True,
        )
        ep = to_edge(exported).exported_program()
        gm = ExpandBroadcastTensorShape()(ep.graph_module).graph_module

        mutated = {
            spec.arg.name
            for spec in ep.graph_signature.output_specs
            if spec.kind == OutputKind.USER_INPUT_MUTATION
        }
        self.assertTrue(mutated)
        for node in gm.graph.nodes:
            if node.name in mutated:
                self.assertFalse(
                    any(
                        isinstance(a, torch.fx.Node) and a.target == view_copy
                        for a in node.args
                    ),
                    "dedupe must not rank-promote the USER_INPUT_MUTATION write-back",
                )

    def _custom_op_node(self, arg, arg_name):
        """A CustomOp builder plus a single-arg node, driven through define_node so
        the branch dispatch is exercised rather than the type helper in isolation.
        Branch order matters: str must be matched before Iterable, since str is
        itself Iterable and would otherwise take the tensor-param path."""

        class _Arg:
            name = arg_name

        class _Schema:
            arguments = [_Arg()]

        class _Target:
            _schema = _Schema()

        node = MagicMock()
        node.name = "my_ops_foo_default"
        node.target = _Target()
        node.args = (arg,)

        info = QnnExecuTorchOpPackageInfo()
        info.custom_op_name = "my_ops.foo.default"
        info.op_package_name = "FooOpPackage"
        info.qnn_op_type_name = "Foo"
        return CustomOp(info, {}, None, False, True), node

    def test_custom_op_rejects_str_arg(self):
        """A str custom-op arg must name the argument and the missing binding.

        str is Iterable, so before the guard it reached the tensor-param branch and
        died on QNN_TENSOR_TYPE_MAP[type(arg[0])] with a bare KeyError. Strings are
        not a QNN limitation: QNN_DATATYPE_STRING exists and PyQnnManagerAdaptor.cpp
        already reads it; only AddScalarParam has no case for it.
        """
        builder, node = self._custom_op_node("bilinear", "soft_nms_method")
        with self.assertRaises(ValueError) as ctx:
            builder.define_node(node, {})
        message = str(ctx.exception)
        self.assertIn("soft_nms_method", message)
        self.assertIn("my_ops.foo.default", message)
        self.assertIn("QNN_DATATYPE_STRING", message)
        # A str must not be reported as a tensor param element type.
        self.assertNotIn("element type", message)

    def test_custom_op_rejects_unmapped_element_type(self):
        """A list whose element type has no QNN mapping -- str[] among them -- must
        report the element type rather than KeyError."""
        builder, node = self._custom_op_node(["linear", "gaussian"], "modes")
        with self.assertRaises(ValueError) as ctx:
            builder.define_node(node, {})
        self.assertIn("element type", str(ctx.exception))
        self.assertIn("modes", str(ctx.exception))

    def test_custom_op_rejects_empty_sequence(self):
        """An empty sequence arg used to IndexError on arg[0] inside define_node."""
        builder, node = self._custom_op_node([], "sizes")
        with self.assertRaises(ValueError) as ctx:
            builder.define_node(node, {})
        self.assertIn("sizes", str(ctx.exception))
        self.assertIn("empty", str(ctx.exception))

    def test_custom_op_resolves_supported_types(self):
        """Guard against a vacuous suite: the mapped scalar types still resolve.

        Checked at the helper rather than through define_node, because a valid arg
        carries on into output-tensor handling, which needs real tensor meta.
        """
        for py_type in (int, float, bool):
            self.assertIsNotNone(
                _resolve_qnn_data_type(py_type, "arg", "my_ops.foo.default")
            )

    def test_backend_bundle_cache_survives_an_expired_entry(self):
        """A backend bundle that expires must not poison the cache.

        QnnBackendUnifiedRegistry holds bundles by weak_ptr keyed on backend type.
        The stale key left behind by an expired bundle was never replaced, so once
        one bundle had come and gone every later request built a fresh backend --
        even while another bundle for that type was alive.
        """
        import tempfile

        import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
        from executorch.backends.qualcomm.partition.utils import (
            generate_qnn_executorch_option,
        )

        option = generate_qnn_executorch_option(
            generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8650,
                backend_options=generate_htp_compiler_spec(use_fp16=True),
            )
        )

        # QNN logs from C++, so capture at the fd level.
        with tempfile.TemporaryFile() as cap:
            saved = os.dup(2)
            os.dup2(cap.fileno(), 2)
            try:
                first = PyQnnManager.QnnManager(option)
                if first.InitBackend().value != 0:
                    skip_without_qnn_sdk(self, "QNN backend unavailable")
                    cap.seek(0)
                    self.fail(cap.read().decode("utf-8", "replace"))
                first.Destroy()
                del first

                kept = PyQnnManager.QnnManager(option)
                self.addCleanup(kept.Destroy)
                self.assertEqual(0, kept.InitBackend().value)

                later = PyQnnManager.QnnManager(option)
                self.addCleanup(later.Destroy)
                self.assertEqual(0, later.InitBackend().value)
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            cap.seek(0)
            log = cap.read().decode("utf-8", "replace")

        created = log.count("Creating new backend bundle")
        reused = log.count("Use cached backend bundle")
        self.assertEqual(2, created, f"expected 2 backend creations, got {created}")
        self.assertGreaterEqual(reused, 1, "third manager must reuse the live bundle")

    def test_is_graph_output_tolerates_users_without_a_name(self):
        """call_module targets are plain strings; reading __name__ raised."""
        from executorch.backends.qualcomm.builders.utils import is_graph_output

        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        # a call_module user, whose target is a str and so has no __name__
        graph.create_node("call_module", "_guards_fn", (placeholder,))
        graph.output((placeholder,))

        self.assertTrue(any(u.op == "call_module" for u in placeholder.users))
        self.assertTrue(is_graph_output(placeholder))
        self.assertTrue(is_graph_output(placeholder, 0))

    def test_is_graph_output_is_per_output_not_per_node(self):
        """Only the outputs a graph actually consumes may be published.

        A multi-output node used to be treated as a whole: one escaping output
        marked every output of that node as a graph output, so QNN published
        values nothing reads. The published set then no longer matched the one
        ExecuTorch bound, and the two are paired by position.
        """
        from executorch.backends.qualcomm.builders.utils import is_graph_output

        class OnlyValues(torch.nn.Module):
            def forward(self, x):
                # max.dim returns (values, indices); only values is returned.
                return torch.max(x, dim=1).values

        gm = torch.export.export(
            OnlyValues().eval(), (torch.randn(2, 3),), strict=True
        ).module()
        node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "max" in str(n.target)
        )
        getitems = {
            u.args[1]: u
            for u in node.users
            if getattr(u.target, "__name__", "") == "getitem"
        }
        self.assertEqual({0, 1}, set(getitems), "both outputs should be unpacked")
        self.assertTrue(getitems[0].users, "values reaches the graph output")
        self.assertFalse(getitems[1].users, "indices is dead")

        self.assertTrue(is_graph_output(node), "node does reach a graph output")
        self.assertTrue(is_graph_output(node, 0), "values is consumed")
        self.assertFalse(
            is_graph_output(node, 1),
            "indices has no consumer and must not be published",
        )

    def test_unconsumed_output_is_native_not_app_read(self):
        """The tensor type is what actually reaches QNN, so assert on it.

        An unconsumed output published as APP_READ makes the QNN graph declare
        more outputs than ExecuTorch binds, and the two are paired by position.
        """
        import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
        from executorch.backends.qualcomm.builders.node_visitor import NodeVisitor

        class OnlyValues(torch.nn.Module):
            def forward(self, x):
                return torch.max(x, dim=1).values

        edge_program = torch.export.export(
            OnlyValues().eval(), (torch.randn(2, 3),), strict=True
        )
        node = next(
            n
            for n in edge_program.graph_module.graph.nodes
            if n.op == "call_function" and "max" in str(n.target)
        )
        visitor = NodeVisitor({node: 0}, edge_program, enable_tensor_dump=False)

        native = PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE
        self.assertEqual(
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ,
            visitor.get_tensor_type(node, native, 0),
            "values is consumed and must be published",
        )
        self.assertEqual(
            native,
            visitor.get_tensor_type(node, native, 1),
            "indices has no consumer and must stay internal",
        )

    def test_constant_graph_output_is_written_by_an_op(self):
        """QNN only publishes a graph output that an op writes.

        A constant op's static tensor has no writer, so as a graph output it was
        left out and the delegate bound one output fewer than ExecuTorch passed.
        """
        import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

        class ConstantOut(torch.nn.Module):
            def __init__(self, make):
                super().__init__()
                self.make = make

            def forward(self, x):
                return x * 2, self.make(x)

        cases = (
            ("aten.full.default", lambda x: torch.full((2, 3), -1.0)),
            ("aten.full_like.default", lambda x: torch.full_like(x, -1.0)),
            ("aten.arange.start_step", lambda x: torch.arange(0, 3, 1.0)),
            ("scalar_tensor.default", lambda x: torch.scalar_tensor(3.0)),
        )
        for target, make in cases:
            with self.subTest(target):
                edge_program = torch.export.export(
                    ConstantOut(make), (torch.randn(2, 3),), strict=True
                )
                node = edge_program.graph.output_node().args[0][1]
                visitor = get_node_visitors(edge_program)[target]
                op = visitor.define_node(node, defaultdict(dict))
                self.assertIsNotNone(op)
                (output,) = op.GetOpWrapper().GetOutputTensors()
                self.assertEqual(
                    f"output_{node.name}@0",
                    PyQnnManager.PyQnnTensorWrapper(output).GetName(),
                )


# Attributes PT2E compares in _union_input_edge_with when deciding whether an
# input edge and its producer's output can share one observer.
_SHARING_ATTRS = (
    "dtype",
    "is_dynamic",
    "quant_min",
    "quant_max",
    "qscheme",
    "ch_axis",
    "scale",
    "zero_point",
)

# (quant_dtype, is_qat, act_symmetric) combinations whose default and per-channel
# activation specs are known not to match yet. Every 16-bit config sets
# quant_min/quant_max unconditionally while the per-channel config omits them when
# act_symmetric, so the two cannot share an observer. Fixing that changes the
# observed zero_point on a path that works today, so it is left for a follow-up.
_KNOWN_UNSHAREABLE = {
    (QuantDtype.use_16a16w, False, True),
    (QuantDtype.use_16a2w, False, True),
    (QuantDtype.use_16a4w, False, True),
    (QuantDtype.use_16a4w, True, True),
    (QuantDtype.use_16a4w_block, False, True),
    (QuantDtype.use_16a4w_block, True, True),
    (QuantDtype.use_16a8w, False, True),
    (QuantDtype.use_16a8w, True, True),
}

_Q_PER_TENSOR = {
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
}
_DQ_PER_TENSOR = {
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
}


def _count_requantize_clusters(graph_module):
    """Quantize nodes fed directly by a dequantize: two observers on one edge."""
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target in _Q_PER_TENSOR
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target in _DQ_PER_TENSOR
    )


class ConvReluConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


class LinearReluLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class TestActivationSpecSharing(unittest.TestCase):
    """A per-channel op's activation spec must match the default one.

    QNN uses two activation specs: the default config for most ops and the
    per-channel config for conv and linear. PT2E gives an edge one observer only
    when the producer's output qspec and the consumer's input qspec agree on every
    attribute in _SHARING_ATTRS -- observer_or_fake_quant_ctr is deliberately not
    compared, so this is about attributes, not spec identity or
    SharedQuantizationSpec. Any drift leaves two observers on one edge, which
    convert_pt2e materializes as a dequantize immediately followed by a quantize
    at every conv boundary.
    """

    def test_default_and_per_channel_act_specs_can_share_an_observer(self):
        from executorch.backends.qualcomm.quantizer.quantizer import QUANT_CONFIG_DICT
        from torchao.quantization.pt2e.prepare import _has_same_attr

        for (quant_dtype, is_qat), funcs in QUANT_CONFIG_DICT.items():
            default_fn, per_channel_fn = funcs[0], funcs[1]
            if default_fn is None or per_channel_fn is None:
                continue
            for act_symmetric in (False, True):
                combination = (quant_dtype, is_qat, act_symmetric)
                with self.subTest(combination=combination):
                    default_act = default_fn(
                        act_symmetric=act_symmetric
                    ).output_activation
                    per_channel_act = per_channel_fn(
                        act_symmetric=act_symmetric, ch_axis=0
                    ).input_activation
                    if default_act is None or per_channel_act is None:
                        # fp16a8w keeps activations in float.
                        continue
                    mismatched = [
                        attr
                        for attr in _SHARING_ATTRS
                        if not _has_same_attr(default_act, per_channel_act, attr)
                    ]
                    if combination in _KNOWN_UNSHAREABLE:
                        self.assertNotEqual(
                            mismatched,
                            [],
                            f"{combination} is listed in _KNOWN_UNSHAREABLE but now "
                            "matches; remove it from the list",
                        )
                        continue
                    self.assertEqual(
                        mismatched,
                        [],
                        f"{combination} cannot share an observer, mismatched on "
                        + ", ".join(
                            f"{attr}: {getattr(default_act, attr, None)} vs "
                            f"{getattr(per_channel_act, attr, None)}"
                            for attr in mismatched
                        ),
                    )

    def _requantize_counts(self, module_cls, example_inputs):
        """Redundant dequantize->quantize clusters under PTQ and QAT."""
        from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

        counts = {}
        for is_qat in (False, True):
            quantizer = QnnQuantizer()
            quantizer.set_default_quant_config(
                QuantDtype.use_8a8w,
                is_qat=is_qat,
                is_conv_per_channel=True,
                is_linear_per_channel=True,
            )
            module = module_cls()
            module = module.train() if is_qat else module.eval()
            exported = torch.export.export(module, example_inputs, strict=True).module()
            prepared = (
                prepare_qat_pt2e(exported, quantizer)
                if is_qat
                else prepare_pt2e(exported, quantizer)
            )
            prepared(*example_inputs)
            counts[is_qat] = _count_requantize_clusters(convert_pt2e(prepared))
        return counts

    def _assert_qat_matches_ptq(self, counts):
        self.assertEqual(counts[False], 0, "PTQ regressed")
        self.assertEqual(
            counts[True],
            counts[False],
            "QAT inserts a dequantize->quantize round trip that PTQ does not; the "
            "default and per-channel activation specs have drifted apart",
        )

    def test_8a8w_qat_does_not_double_quantize_conv_boundaries(self):
        """End-to-end guard for the same invariant.

        This model has no BatchNorm, so it isolates the activation-spec mismatch
        from anything conv-bn related. Before the act_symmetric split in
        get_8a8w_qnn_qat_config, QAT produced three redundant clusters here while
        PTQ produced none.
        """
        self._assert_qat_matches_ptq(
            self._requantize_counts(ConvReluConv, (torch.randn(2, 3, 8, 8),))
        )

    def test_8a8w_qat_does_not_double_quantize_linear_boundaries(self):
        """The conv case above never reaches is_linear_per_channel.

        Linear takes the same per-channel activation spec as conv, so the same
        drift shows up at linear boundaries -- but ConvReluConv has no aten.linear
        node, so nothing exercised that path.
        """
        self._assert_qat_matches_ptq(
            self._requantize_counts(LinearReluLinear, (torch.randn(2, 16),))
        )


class TestIoBindingCheck(unittest.TestCase):
    """_check_io_binding must fire when QNN publishes I/O the signature lacks.

    The runtime binds delegate arguments positionally against the tensor list in
    the context binary, so a graph that declares more bindable tensors than the
    program passes reads past the end of args. That has to be caught at lowering.
    """

    @staticmethod
    def _edge_program(num_inputs, num_outputs):
        signature = MagicMock()
        signature.user_inputs = [f"arg_{i}" for i in range(num_inputs)]
        signature.user_outputs = [f"out_{i}" for i in range(num_outputs)]
        edge_program = MagicMock()
        edge_program.graph_signature = signature
        return edge_program

    def _run(self, names, num_inputs, num_outputs):
        from executorch.backends.qualcomm import qnn_preprocess

        wrappers = {f"n{i}": {0: object()} for i in range(len(names))}
        by_id = {
            id(next(iter(v.values()))): name
            for v, name in zip(wrappers.values(), names)
        }

        class _StubWrapper:
            def __init__(self, wrapper):
                self._name = by_id[id(wrapper)]

            def GetName(self):
                return self._name

        with unittest.mock.patch.object(
            qnn_preprocess.PyQnnManager, "PyQnnTensorWrapper", _StubWrapper
        ):
            qnn_preprocess._check_io_binding(
                self._edge_program(num_inputs, num_outputs), wrappers
            )

    def test_matching_io_passes(self):
        self._run(
            ["input_0_x", "output_y", "conv2d_3", "_frozen_param0"],
            num_inputs=1,
            num_outputs=1,
        )

    def test_mutable_buffers_do_not_consume_arguments(self):
        # A mutable buffer is threaded through separately and is not a user input.
        self._run(
            ["input_0_x", "input_1_mutbuf_0_cache", "output_y"],
            num_inputs=1,
            num_outputs=1,
        )

    def test_extra_published_inputs_are_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(
                ["input_0_x", "input_1_stray", "input_2_stray", "output_y"],
                num_inputs=1,
                num_outputs=1,
            )
        message = str(ctx.exception)
        self.assertIn("QNN declares 3 graph inputs", message)
        self.assertIn("input_1_stray", message)

    def test_extra_published_outputs_are_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(
                ["input_0_x", "output_y", "output_stray"], num_inputs=1, num_outputs=1
            )
        self.assertIn("2 graph outputs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
