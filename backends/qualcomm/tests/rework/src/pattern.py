# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest
import torch

from executorch.backends.qualcomm import _passes
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    check_exception,
    EXCEPTION_FROM_PASSES,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

# Mechanism to not import but provide type hint below in this file.
if TYPE_CHECKING:
    from executorch.backends.qualcomm.tests.rework.passes.passes_helper import (
        Assertions,
        PassPipeline,
    )


def unpack_pass_fixtures(func):
    params = inspect.signature(func).parameters

    def wrapper(request, kwargs):
        extra_fixtures = set(params.keys()) - set(kwargs.keys())
        new_kwargs = {key: request.getfixturevalue(key) for key in extra_fixtures}
        return func(
            **{k: v for k, v in {**kwargs, **new_kwargs}.items() if k in params}
        )

    return wrapper


class SimpleModel(torch.nn.Module):
    """
    Reusable model for general testing purpose
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        kernel_sz = 32
        self.conv1 = torch.nn.Conv2d(
            kernel_sz, kernel_sz, kernel_size, padding=1, bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            kernel_sz, kernel_sz, kernel_size, padding=1, bias=True
        )
        self.conv3 = torch.nn.Conv2d(
            kernel_sz, kernel_sz, kernel_size, padding=1, bias=False
        )
        self.conv4 = torch.nn.Conv2d(
            kernel_sz, kernel_sz, kernel_size, padding=1, bias=False
        )
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm2d(kernel_sz)
        self.linear = torch.nn.Linear(4, 10)
        self.eval()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.batch_norm(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.relu(x4)
        y1 = self.conv3(y)
        y2 = self.batch_norm(y1)
        y3 = self.relu(y2)
        y4 = self.conv4(y3)
        y5 = self.relu(y4)
        z = torch.add(x5, y5)
        z1 = torch.permute(z, (0, 3, 2, 1))
        z2 = torch.mean(z1, [1, 2], True)
        z3 = torch.reshape(z2, (8, -1))
        z4 = self.linear(z3)
        z5 = self.hardtanh(z4)
        return z5


class AnnotateAvgPool1D:
    class _AvgPool1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            return self.pool(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = AnnotateAvgPool1D._AvgPool1d()
        inputs = (torch.randn(1, 4, 16),)
        target_pass = _passes.AnnotateAvgPool1D
        _ = assertions
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module
        partitions = get_source_partitions(gm.graph, target_pass._SOURCE_OPS)
        nodes = list(partitions.values())[0][0].nodes
        assert all(
            QCOM_QUANT_ATTRS in n.meta for n in nodes
        ), "avg_pool1d partition nodes should have QCOM_QUANT_ATTRS"


class AnnotateQuantAttrs:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = AnnotateQuantAttrs._Basic()
        inputs = (torch.randn(1, 4),)
        target_pass = _passes.AnnotateQuantAttrs
        _ = assertions
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module
        annotated = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and QCOM_QUANT_ATTRS in n.meta
        ]
        assert (
            len(annotated) > 0
        ), "op nodes should have QCOM_QUANT_ATTRS after AnnotateQuantAttrs"


class AnnotateStack:
    class _Basic(torch.nn.Module):
        def forward(self, x, y):
            return torch.stack([x, y], dim=0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = AnnotateStack._Basic()
        inputs = (
            torch.randn(
                4,
            ),
            torch.randn(
                4,
            ),
        )
        target_pass = _passes.AnnotateStack
        _ = assertions
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module
        partitions = get_source_partitions(gm.graph, target_pass._SOURCE_OPS)
        nodes = list(partitions.values())[0][0].nodes
        # TODO: update this once lpai starts to support pack
        if backend_type != QnnExecuTorchBackendType.kLpaiBackend:
            assert all(
                QCOM_QUANT_ATTRS in n.meta for n in nodes
            ), "stack partition nodes should have QCOM_QUANT_ATTRS"


class AnnotateUnbind:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.unbind(x, dim=0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = AnnotateUnbind._Basic()
        inputs = (torch.randn(4, 4),)
        target_pass = _passes.AnnotateUnbind
        _ = assertions
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module
        partitions = get_source_partitions(gm.graph, target_pass._SOURCE_OPS)
        nodes = list(partitions.values())[0][0].nodes
        # TODO: update this once lpai starts to support unpack
        if backend_type != QnnExecuTorchBackendType.kLpaiBackend:
            assert all(
                QCOM_QUANT_ATTRS in n.meta for n in nodes
            ), "unbind partition nodes should have QCOM_QUANT_ATTRS"


class BuildQuantIo:
    class _Basic(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO

        module = BuildQuantIo._Basic()
        inputs = (torch.randn(1, 4), torch.randn(1, 4))
        _ = assertions
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=None,
            quantizer=quantizer,
        ).graph_module
        # tag placeholders (IO) with QCOM_QUANTIZED_IO
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta[QCOM_QUANTIZED_IO] = torch.uint8

        gm = _passes.BuildQuantIo()(gm).graph_module
        for node in gm.graph.nodes:
            if node.op == "placeholder" and QCOM_QUANTIZED_IO in node.meta:
                assert (
                    node.meta["val"].dtype == torch.uint8
                ), "val.dtype should be updated to the QCOM_QUANTIZED_IO dtype"


class CanonicalizeConv:
    # --- Group 1: conv1d canonicalization (conv1d → unsqueeze + conv2d + squeeze) ---
    class _Conv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    class _ConvTranspose1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(4, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    # --- Group 2: transpose conv dilation (kernel dilated manually, dilation arg removed) ---
    class _ConvTranspose1dDilation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(4, 8, kernel_size=3, dilation=2)

        def forward(self, x):
            return self.conv(x)

    class _ConvTranspose2dDilation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(4, 8, kernel_size=3, dilation=2)

        def forward(self, x):
            return self.conv(x)

    class _ConvTranspose3dDilation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose3d(4, 8, kernel_size=3, dilation=2)

        def forward(self, x):
            return self.conv(x)

    @staticmethod
    def _assert_no_dilation(gm, targets):
        """Assert dilation arg has been removed from all matching nodes."""
        for node in gm.graph.nodes:
            if node.target in targets:
                assert len(node.args) <= 7 or all(
                    d == 1 for d in node.args[7]
                ), f"{node.target} dilation arg (index 7) should be all ones"

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.CanonicalizeConv
        _ = compile_spec

        # --- Group 1: conv1d → unsqueeze_copy + conv2d + squeeze_copy ---
        with subtests.test(msg="conv1d"):
            gm = pass_pipeline.lower_export_gm(
                module=CanonicalizeConv._Conv1d(),
                sample_input=(torch.randn(1, 4, 16),),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.conv1d.default)
            assertions.assert_target_count(gm, torch.ops.aten.conv2d.default, 1)
            assertions.assert_target_count(gm, torch.ops.aten.unsqueeze_copy.default, 1)
            assertions.assert_target_count(gm, torch.ops.aten.squeeze_copy.dims, 1)

        with subtests.test(msg="conv1d_transpose"):
            gm = pass_pipeline.lower_export_gm(
                module=CanonicalizeConv._ConvTranspose1d(),
                sample_input=(torch.randn(1, 4, 8),),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.conv_transpose1d.default)
            assertions.assert_target_count(gm, torch.ops.aten.conv_transpose2d.input, 1)
            assertions.assert_target_count(gm, torch.ops.aten.unsqueeze_copy.default, 1)
            assertions.assert_target_count(gm, torch.ops.aten.squeeze_copy.dims, 1)

        # --- Group 2: transpose conv dilation — kernel dilated, dilation arg removed ---
        with subtests.test(msg="conv_transpose1d_dilation"):
            gm = pass_pipeline.lower_export_gm(
                module=CanonicalizeConv._ConvTranspose1dDilation(),
                sample_input=(torch.randn(1, 4, 8),),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.conv_transpose1d.default)
            assertions.assert_target_count(gm, torch.ops.aten.conv_transpose2d.input, 1)
            CanonicalizeConv._assert_no_dilation(
                gm, {torch.ops.aten.conv_transpose2d.input}
            )

        with subtests.test(msg="conv_transpose2d_dilation"):
            gm = pass_pipeline.lower_export_gm(
                module=CanonicalizeConv._ConvTranspose2dDilation(),
                sample_input=(torch.randn(1, 4, 8, 8),),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_target_count(gm, torch.ops.aten.conv_transpose2d.input, 1)
            CanonicalizeConv._assert_no_dilation(
                gm, {torch.ops.aten.conv_transpose2d.input}
            )

        # TODO: update this once lpai starts to support transpose_conv3d
        if backend_type != QnnExecuTorchBackendType.kLpaiBackend:
            with subtests.test(msg="conv_transpose3d_dilation"):
                gm = pass_pipeline.lower_export_gm(
                    module=CanonicalizeConv._ConvTranspose3dDilation(),
                    sample_input=(torch.randn(1, 4, 4, 4, 4),),
                    target_pass=target_pass,
                    backend_type=backend_type,
                    quantizer=quantizer,
                )
                assertions.assert_target_count(
                    gm, torch.ops.aten.conv_transpose3d.input, 1
                )
                CanonicalizeConv._assert_no_dilation(
                    gm, {torch.ops.aten.conv_transpose3d.input}
                )


class ConvertBmmToMatmul:
    class _Basic(torch.nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.backends.qualcomm._passes.qnn_pass_manager import (
            get_qnn_pass_manager_cls,
        )

        passes_job = get_qnn_pass_manager_cls(backend_type).get_capture_program_passes()
        passes_job[_passes.ConvertBmmToMatmul][QCOM_PASS_ACTIVATE_KEY] = True
        module = ConvertBmmToMatmul._Basic()
        inputs = (torch.randn(2, 4, 8), torch.randn(2, 8, 4))
        target_pass = _passes.ConvertBmmToMatmul
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
            passes_job=passes_job,
        ).graph_module
        assertions.assert_no_target(gm, exir_ops.edge.aten.bmm.default)
        assertions.assert_target_count(gm, exir_ops.edge.aten.matmul.default, 1)


class ConvertLinearToConv2d:
    class _Basic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.linear(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = ConvertLinearToConv2d._Basic()
        inputs = (torch.randn(2, 8),)
        target_pass = _passes.ConvertLinearToConv2d
        _ = compile_spec
        gm = pass_pipeline.lower_export_gm(
            module=module,
            sample_input=inputs,
            target_pass=target_pass,
            backend_type=backend_type,
            quantizer=quantizer,
            convert_linear_to_conv2d=True,
        )
        assertions.assert_no_target(gm, torch.ops.aten.linear.default)
        assertions.assert_target_count(gm, torch.ops.aten.conv2d.default, 1)
        # rank-2 input: reshape×2 (input + output restore) + permute×2 (pre/post conv)
        assertions.assert_target_count(gm, torch.ops.aten.reshape.default, 2)
        assertions.assert_target_count(gm, torch.ops.aten.permute.default, 2)


class ConvertMhaToSha:
    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.backends.qualcomm.utils.utils import (
            convert_linear_to_conv2d as _cvt_linear,
        )
        from executorch.examples.models.llama.model_args import ModelArgs
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            CausalAttentionMask,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaAttention,
        )

        _ = assertions
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
        mod = _cvt_linear(LlamaAttention(0, args, True))

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

        # Phase 1: lower_edge_ep runs full edge pipeline (RemoveRedundancy,
        # ConvertBmmToMatmul, etc.); MHA→SHA is in preprocess so not yet run
        edge_ep = pass_pipeline.lower_edge_ep(
            module=mod,
            sample_input=sample_input,
            backend_type=backend_type,
            compile_spec=compile_spec,
            quantizer=quantizer,
        )
        conv_before = [
            n
            for n in edge_ep.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        assert (
            len(conv_before) == 4
        ), f"expected 4 conv nodes (WQ/WK/WV/O) before MHA→SHA, got {len(conv_before)}"

        # Phase 2: lower_preprocess_gm with use_mha2sha=True splits heads
        gm = pass_pipeline.lower_preprocess_gm(
            module=mod,
            sample_input=sample_input,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=_passes.ConvertMhaToSha,
            quantizer=quantizer,
            use_mha2sha=True,
        )
        conv_after = [
            n
            for n in gm.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        ]
        # n_heads×3 (WQ, WK, WV per head) + 1 (O) = 8×3+1 = 25
        assert (
            len(conv_after) == 25
        ), f"expected 25 conv nodes after MHA→SHA, got {len(conv_after)}"


class ConvertSquareToPow:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.square(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = ConvertSquareToPow._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.ConvertSquareToPow
        _ = compile_spec

        gm = pass_pipeline.lower_export_gm(
            module=module,
            sample_input=inputs,
            target_pass=target_pass,
            backend_type=backend_type,
            quantizer=quantizer,
        )
        assertions.assert_no_target(gm, torch.ops.aten.square.default)
        assertions.assert_target_count(gm, torch.ops.aten.pow.Tensor_Scalar, 1)
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.pow.Tensor_Scalar:
                assert node.args[1] == 2, "pow exponent should be 2"


class DecomposeAcos:
    class _Basic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.acos(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeAcos._Basic()
        inputs = (torch.rand(3, 4) * 2 - 1,)
        target_pass = _passes.DecomposeAcos

        if quantizer is None:
            gm = pass_pipeline.lower_edge_ep(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
        else:
            gm = pass_pipeline.lower_annotation_gm(
                module=module,
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
            )
        assertions.assert_target_count(
            gm, {torch.ops.aten.asin.default, exir_ops.edge.aten.asin.default}, 1
        )


class DecomposeAddmm:
    class _WithAlphaBeta(torch.nn.Module):
        def __init__(self, alpha=1, beta=1):
            super().__init__()
            self.alpha = alpha
            self.beta = beta

        def forward(self, bias, x, w):
            return torch.addmm(bias, x, w, alpha=self.alpha, beta=self.beta)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        # (alpha, beta, expected_mm, expected_add, expected_mul)
        cases = [
            ("default", 1, 1, 1, 1, 0),
            ("alpha_only", 2, 1, 1, 1, 1),
            ("beta_only", 1, 3, 1, 1, 1),
            ("alpha_and_beta", 2, 3, 1, 1, 2),
        ]
        inputs = (torch.randn(8), torch.randn(4, 3), torch.randn(3, 8))
        target_pass = _passes.DecomposeAddmm

        for label, alpha, beta, exp_mm, exp_add, exp_mul in cases:
            with subtests.test(msg=label):
                module = DecomposeAddmm._WithAlphaBeta(alpha=alpha, beta=beta)

                match backend_type:
                    case QnnExecuTorchBackendType.kGpuBackend:
                        gm = pass_pipeline.lower_edge_ep(
                            module=module,
                            sample_input=inputs,
                            backend_type=backend_type,
                            compile_spec=compile_spec,
                            target_pass=target_pass,
                        ).graph_module
                    case QnnExecuTorchBackendType.kHtpBackend:
                        if quantizer:
                            gm = pass_pipeline.lower_annotation_gm(
                                module=module,
                                sample_input=inputs,
                                target_pass=target_pass,
                                backend_type=backend_type,
                            )
                        else:
                            gm = pass_pipeline.lower_edge_ep(
                                module=module,
                                sample_input=inputs,
                                backend_type=backend_type,
                                compile_spec=compile_spec,
                                target_pass=target_pass,
                            ).graph_module
                    case QnnExecuTorchBackendType.kLpaiBackend:
                        gm = pass_pipeline.lower_annotation_gm(
                            module=module,
                            sample_input=inputs,
                            target_pass=target_pass,
                            backend_type=backend_type,
                        )
                    case _:
                        raise AssertionError(f"unhandled backend_type: {backend_type}")

                assertions.assert_no_target(
                    gm, {torch.ops.aten.addmm.default, exir_ops.edge.aten.addmm.default}
                )
                assertions.assert_target_count(
                    gm,
                    {torch.ops.aten.mm.default, exir_ops.edge.aten.mm.default},
                    exp_mm,
                )
                assertions.assert_target_count(
                    gm,
                    {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor},
                    exp_add,
                )
                assertions.assert_target_count(
                    gm,
                    {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor},
                    exp_mul,
                )


class DecomposeAny:
    class _WithDim(torch.nn.Module):
        def __init__(self, dim, keepdim=False):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.any(x, dim=self.dim, keepdim=self.keepdim)

    class _AllReduce(torch.nn.Module):
        def forward(self, x):
            return torch.any(x)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
    ):
        return pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
        ).graph_module

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randint(0, 2, (4, 4), dtype=torch.bool),)
        target_pass = _passes.DecomposeAny
        _ = quantizer

        # dim given, keepdim=False: _to_copy×2 + sum + ne, no flatten
        with subtests.test(msg="dim_keepdim_false"):
            gm = DecomposeAny._get_gm(
                DecomposeAny._WithDim(dim=0, keepdim=False),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
            )
            assertions.assert_no_target(gm, {exir_ops.edge.aten.any.dim})
            assertions.assert_target_count(gm, {exir_ops.edge.aten.sum.dim_IntList}, 1)
            assertions.assert_no_target(gm, {exir_ops.edge.aten.view_copy.default})

        # dim given, keepdim=True: same op set, sum preserves dimension
        with subtests.test(msg="dim_keepdim_true"):
            gm = DecomposeAny._get_gm(
                DecomposeAny._WithDim(dim=0, keepdim=True),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
            )
            assertions.assert_no_target(gm, {exir_ops.edge.aten.any.dim})
            assertions.assert_target_count(gm, {exir_ops.edge.aten.sum.dim_IntList}, 1)

        # dim=None (all-reduce): flatten inserted before cast+sum+ne
        with subtests.test(msg="dim_none"):
            gm = DecomposeAny._get_gm(
                DecomposeAny._AllReduce(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
            )
            assertions.assert_no_target(gm, {exir_ops.edge.aten.any.default})
            assertions.assert_target_count(gm, {exir_ops.edge.aten.sum.dim_IntList}, 1)
            assertions.assert_target_count(
                gm, {exir_ops.edge.aten.view_copy.default}, 1
            )


class DecomposeAtan2:
    class _Basic(torch.nn.Module):
        def forward(self, y, x):
            return torch.atan2(y, x)

    @staticmethod
    def _assert_decomposed(gm, assertions, extra_to_copy=0):
        assertions.assert_no_target(
            gm, {torch.ops.aten.atan2.default, exir_ops.edge.aten.atan2.default}
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.atan.default, exir_ops.edge.aten.atan.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.where.self, exir_ops.edge.aten.where.self}, 6
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.eq.Tensor, exir_ops.edge.aten.eq.Tensor}, 2
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.lt.Tensor, exir_ops.edge.aten.lt.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.ge.Tensor, exir_ops.edge.aten.ge.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.gt.Tensor, exir_ops.edge.aten.gt.Tensor}, 1
        )
        if extra_to_copy:
            assertions.assert_target_count(
                gm,
                {torch.ops.aten._to_copy.default, exir_ops.edge.aten._to_copy.default},
                extra_to_copy,
            )

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeAtan2._Basic()
        target_pass = _passes.DecomposeAtan2

        cases = [
            ("float_inputs", (torch.randn(3, 4), torch.randn(3, 4)), 0),
            (
                "int_inputs",
                (torch.randint(-5, 5, (3, 4)), torch.randint(-5, 5, (3, 4))),
                2,
            ),
        ]

        for label, inputs, extra_to_copy in cases:
            with subtests.test(msg=label):
                match backend_type:
                    case QnnExecuTorchBackendType.kGpuBackend:
                        gm = pass_pipeline.lower_edge_ep(
                            module=module,
                            sample_input=inputs,
                            backend_type=backend_type,
                            compile_spec=compile_spec,
                            target_pass=target_pass,
                        ).graph_module
                    case QnnExecuTorchBackendType.kHtpBackend:
                        if quantizer:
                            gm = pass_pipeline.lower_annotation_gm(
                                module=module,
                                sample_input=inputs,
                                target_pass=target_pass,
                                backend_type=backend_type,
                            )
                        else:
                            gm = pass_pipeline.lower_edge_ep(
                                module=module,
                                sample_input=inputs,
                                backend_type=backend_type,
                                compile_spec=compile_spec,
                                target_pass=target_pass,
                            ).graph_module
                    case QnnExecuTorchBackendType.kLpaiBackend:
                        gm = pass_pipeline.lower_annotation_gm(
                            module=module,
                            sample_input=inputs,
                            target_pass=target_pass,
                            backend_type=backend_type,
                        )
                    case _:
                        raise AssertionError(f"unhandled backend_type: {backend_type}")
                DecomposeAtan2._assert_decomposed(
                    gm, assertions, extra_to_copy=extra_to_copy
                )


class DecomposeBinaryAlpha:
    class _TensorAdd(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y, alpha=2)

    class _TensorSub(torch.nn.Module):
        def forward(self, x, y):
            return torch.sub(x, y, alpha=2)

    class _ConstantAdd(torch.nn.Module):
        # alpha with a constant second operand: pass folds alpha in-place (no mul inserted)
        def forward(self, x):
            return torch.add(x, 3.0, alpha=2)

    class _ConstantSub(torch.nn.Module):
        def forward(self, x):
            return torch.sub(x, 3.0, alpha=2)

    @staticmethod
    def _get_gm(module, inputs, target_pass, pass_pipeline, backend_type, quantizer):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.DecomposeBinaryAlpha
        tensor_inputs = (torch.randn(3, 4), torch.randn(3, 4))
        scalar_inputs = (torch.randn(3, 4),)
        _ = compile_spec

        # Tensor input2: mul.Scalar inserted, alpha kwarg removed
        with subtests.test(msg="add_tensor"):
            gm = DecomposeBinaryAlpha._get_gm(
                DecomposeBinaryAlpha._TensorAdd(),
                tensor_inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_target_count(gm, torch.ops.aten.mul.Scalar, 1)
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target in {
                    torch.ops.aten.add.Tensor,
                    exir_ops.edge.aten.add.Tensor,
                }:
                    assert "alpha" not in node.kwargs
                    # args[1] must now be the mul node (not raw input2)
                    mul_node = node.args[1]
                    assert mul_node.target == torch.ops.aten.mul.Scalar
                    assert mul_node.args[1] == 2  # alpha value

        with subtests.test(msg="sub_tensor"):
            gm = DecomposeBinaryAlpha._get_gm(
                DecomposeBinaryAlpha._TensorSub(),
                tensor_inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_target_count(gm, torch.ops.aten.mul.Scalar, 1)
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target in {
                    torch.ops.aten.sub.Tensor,
                    exir_ops.edge.aten.sub.Tensor,
                }:
                    assert "alpha" not in node.kwargs
                    mul_node = node.args[1]
                    assert mul_node.target == torch.ops.aten.mul.Scalar
                    assert mul_node.args[1] == 2  # alpha value

        # Constant input2: alpha folded into the constant in-place, no mul inserted
        with subtests.test(msg="add_constant"):
            gm = DecomposeBinaryAlpha._get_gm(
                DecomposeBinaryAlpha._ConstantAdd(),
                scalar_inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.mul.Scalar)
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target in {
                    torch.ops.aten.add.Tensor,
                    exir_ops.edge.aten.add.Tensor,
                }:
                    assert "alpha" not in node.kwargs
                    assert node.args[1] == 6.0  # 3.0 * 2

        with subtests.test(msg="sub_constant"):
            gm = DecomposeBinaryAlpha._get_gm(
                DecomposeBinaryAlpha._ConstantSub(),
                scalar_inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.mul.Scalar)
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target in {
                    torch.ops.aten.sub.Tensor,
                    exir_ops.edge.aten.sub.Tensor,
                }:
                    assert "alpha" not in node.kwargs
                    assert node.args[1] == 6.0  # 3.0 * 2


class DecomposeCDist:
    class _P2(torch.nn.Module):
        def forward(self, x, y):
            return torch.cdist(x, y)  # default p=2

    class _P1(torch.nn.Module):
        def forward(self, x, y):
            return torch.cdist(x, y, p=1)  # unsupported p-norm

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(4, 8), torch.randn(6, 8))
        target_pass = _passes.DecomposeCDist
        _ = compile_spec

        # p=2: decomposed into unsqueeze×2, sub, pow/mul, sum, sqrt
        with subtests.test(msg="p2"):
            gm = DecomposeCDist._get_gm(
                DecomposeCDist._P2(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(
                gm,
                {torch.ops.aten.cdist.default, torch.ops.aten._cdist_forward.default},
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.sub.Tensor, exir_ops.edge.aten.sub.Tensor}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.sqrt.default, exir_ops.edge.aten.sqrt.default}, 1
            )
            assertions.assert_target_count(
                gm,
                {torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList},
                1,
            )

        # p!=2: pass asserts p==2, wrapped by ExportPass into a generic Exception
        with subtests.test(msg="p1_unsupported"):
            with pytest.raises(  # noqa: B017
                Exception, check=check_exception(EXCEPTION_FROM_PASSES)
            ):
                DecomposeCDist._get_gm(
                    DecomposeCDist._P1(),
                    inputs,
                    target_pass,
                    pass_pipeline,
                    backend_type,
                    quantizer,
                )


class DecomposeColIm:
    class _Im2Col(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.unfold(x, kernel_size=2, stride=2)

    class _Col2Im(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.fold(
                x, output_size=(4, 4), kernel_size=2, stride=2
            )

    # im2col violation models — each breaks exactly one assertion in _decompose_im2col
    class _Im2ColStrideMismatch(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.unfold(x, kernel_size=2, stride=1)

    class _Im2ColNonSquareKernel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.unfold(x, kernel_size=(2, 3), stride=(2, 3))

    class _Im2ColDilation(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.unfold(x, kernel_size=2, stride=2, dilation=2)

    class _Im2ColPadding(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.unfold(x, kernel_size=2, stride=2, padding=1)

    # col2im violation models — each breaks exactly one assertion in _decompose_col2im
    class _Col2ImStrideMismatch(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.fold(
                x, output_size=(4, 4), kernel_size=2, stride=1
            )

    class _Col2ImNonSquareKernel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.fold(
                x, output_size=(4, 6), kernel_size=(2, 3), stride=(2, 3)
            )

    class _Col2ImDilation(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.fold(
                x, output_size=(6, 6), kernel_size=2, stride=2, dilation=2
            )

    class _Col2ImPadding(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.fold(
                x, output_size=(4, 4), kernel_size=2, stride=2, padding=1
            )

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.DecomposeColIm

        # im2col (unfold): pixel_unshuffle + view_copy
        with subtests.test(msg="im2col"):
            gm = pass_pipeline.lower_edge_ep(
                module=DecomposeColIm._Im2Col(),
                sample_input=(torch.randn(1, 4, 4, 4),),
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            assertions.assert_no_target(gm, exir_ops.edge.aten.im2col.default)
            assertions.assert_target_count(
                gm, exir_ops.edge.aten.pixel_unshuffle.default, 1
            )
            assertions.assert_target_count(gm, exir_ops.edge.aten.view_copy.default, 1)

        # col2im (fold): view_copy + pixel_shuffle
        with subtests.test(msg="col2im"):
            x = torch.nn.functional.unfold(
                torch.randn(1, 4, 4, 4), kernel_size=2, stride=2
            )
            gm = pass_pipeline.lower_edge_ep(
                module=DecomposeColIm._Col2Im(),
                sample_input=(x,),
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            assertions.assert_no_target(gm, exir_ops.edge.aten.col2im.default)
            assertions.assert_target_count(
                gm, exir_ops.edge.aten.pixel_shuffle.default, 1
            )
            assertions.assert_target_count(gm, exir_ops.edge.aten.view_copy.default, 1)

        # im2col assertion failures
        im2col_fail_cases = [
            (
                "im2col_stride_mismatch",
                DecomposeColIm._Im2ColStrideMismatch(),
                (torch.randn(1, 4, 4, 4),),
            ),
            (
                "im2col_non_square_kernel",
                DecomposeColIm._Im2ColNonSquareKernel(),
                (torch.randn(1, 4, 4, 6),),
            ),
            (
                "im2col_dilation_not_one",
                DecomposeColIm._Im2ColDilation(),
                (torch.randn(1, 4, 8, 8),),
            ),
            (
                "im2col_nonzero_padding",
                DecomposeColIm._Im2ColPadding(),
                (torch.randn(1, 4, 4, 4),),
            ),
        ]
        for label, module, inputs in im2col_fail_cases:
            with subtests.test(msg=label):
                with pytest.raises(  # noqa: B017
                    Exception, check=check_exception(EXCEPTION_FROM_PASSES)
                ):
                    pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                        quantizer=quantizer,
                    )

        # col2im assertion failures
        col2im_fail_cases = [
            (
                "col2im_stride_mismatch",
                DecomposeColIm._Col2ImStrideMismatch(),
                (
                    torch.nn.functional.unfold(
                        torch.randn(1, 4, 4, 4), kernel_size=2, stride=1
                    ),
                ),
            ),
            (
                "col2im_non_square_kernel",
                DecomposeColIm._Col2ImNonSquareKernel(),
                (
                    torch.nn.functional.unfold(
                        torch.randn(1, 4, 4, 6), kernel_size=(2, 3), stride=(2, 3)
                    ),
                ),
            ),
            (
                "col2im_dilation_not_one",
                DecomposeColIm._Col2ImDilation(),
                (
                    torch.nn.functional.unfold(
                        torch.randn(1, 4, 6, 6), kernel_size=2, stride=2, dilation=2
                    ),
                ),
            ),
            (
                "col2im_nonzero_padding",
                DecomposeColIm._Col2ImPadding(),
                (
                    torch.nn.functional.unfold(
                        torch.randn(1, 4, 4, 4), kernel_size=2, stride=2, padding=1
                    ),
                ),
            ),
        ]
        for label, module, inputs in col2im_fail_cases:
            with subtests.test(msg=label):
                with pytest.raises(  # noqa: B017
                    Exception, check=check_exception(EXCEPTION_FROM_PASSES)
                ):
                    pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                        quantizer=quantizer,
                    )


class DecomposeDiagonal:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.diagonal(x, offset=0, dim1=0, dim2=1)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeDiagonal._Basic()
        inputs = (torch.randn(4, 4),)
        target_pass = _passes.DecomposeDiagonal

        if quantizer is None:
            gm = pass_pipeline.lower_edge_ep(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
        else:
            gm = pass_pipeline.lower_annotation_gm(
                module=module,
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
            )
        # diagonal -> view + arange + index_select (dim1/dim2 already last two
        # dims, so no permute is inserted).
        assertions.assert_no_target(
            gm,
            {
                torch.ops.aten.diagonal.default,
                exir_ops.edge.aten.diagonal_copy.default,
            },
        )
        assertions.assert_target_count(
            gm,
            {
                torch.ops.aten.index_select.default,
                exir_ops.edge.aten.index_select.default,
            },
            1,
        )
        assertions.assert_target_count(
            gm,
            {torch.ops.aten.arange.start_step, exir_ops.edge.aten.arange.start_step},
            1,
        )
        assertions.assert_target_count(
            gm,
            {torch.ops.aten.view.default, exir_ops.edge.aten.view_copy.default},
            1,
        )


class DecomposeDivMode:
    class _DivMode(torch.nn.Module):
        def __init__(self, rounding_mode):
            super().__init__()
            self.rounding_mode = rounding_mode

        def forward(self, x, y):
            return torch.div(x, y, rounding_mode=self.rounding_mode)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 4), torch.randn(3, 4).abs() + 0.1)
        target_pass = _passes.DecomposeDivMode

        # rounding_mode=None: div.Tensor only, no rounding op
        with subtests.test(msg="none"):
            gm = DecomposeDivMode._get_gm(
                DecomposeDivMode._DivMode(rounding_mode=None),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.div.Tensor_mode, exir_ops.edge.aten.div.Tensor_mode}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.trunc.default, exir_ops.edge.aten.trunc.default}
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.floor.default, exir_ops.edge.aten.floor.default}
            )

        # rounding_mode="trunc": div.Tensor + trunc
        with subtests.test(msg="trunc"):
            gm = DecomposeDivMode._get_gm(
                DecomposeDivMode._DivMode(rounding_mode="trunc"),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.div.Tensor_mode, exir_ops.edge.aten.div.Tensor_mode}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.trunc.default, exir_ops.edge.aten.trunc.default}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.floor.default, exir_ops.edge.aten.floor.default}
            )

        # rounding_mode="floor": div.Tensor + floor
        with subtests.test(msg="floor"):
            gm = DecomposeDivMode._get_gm(
                DecomposeDivMode._DivMode(rounding_mode="floor"),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.div.Tensor_mode, exir_ops.edge.aten.div.Tensor_mode}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.trunc.default, exir_ops.edge.aten.trunc.default}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.floor.default, exir_ops.edge.aten.floor.default}, 1
            )


# since it's hard to detect the exact operators given different equations
# prefer to keep it simple here
class DecomposeEinsum:
    class _OuterProduct(torch.nn.Module):
        def forward(self, x, y):
            return torch.einsum("i,j->ij", x, y)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeEinsum._OuterProduct()
        inputs = (torch.randn(4), torch.randn(5))
        target_pass = _passes.DecomposeEinsum

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(gm, torch.ops.aten.einsum.default)


class DecomposeExpM1:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.expm1(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeExpM1._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeExpM1
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(gm, torch.ops.aten.special_expm1.default)
        assertions.assert_target_count(
            gm, {torch.ops.aten.exp.default, exir_ops.edge.aten.exp.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sub.Tensor, exir_ops.edge.aten.sub.Tensor}, 1
        )


class DecomposeFill:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.fill(x, 3.14)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeFill._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeFill

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm,
            {
                torch.ops.aten.fill.Scalar,
                torch.ops.aten.fill_.Scalar,
                exir_ops.edge.aten.fill.Scalar,
                exir_ops.edge.aten.fill_.Scalar,
            },
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.full.default, exir_ops.edge.aten.full.default}, 1
        )


class DecomposeFloorDivide:
    class _IntDiv(torch.nn.Module):
        def forward(self, x, y):
            return torch.floor_divide(x, y)

    class _FloatDiv(torch.nn.Module):
        def forward(self, x, y):
            return torch.floor_divide(x, y)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        _ = compile_spec
        target_pass = _passes.DecomposeFloorDivide

        # Integer inputs: pass fires — floor_divide decomposed into div + floor
        with subtests.test(msg="int"):
            gm = pass_pipeline.lower_export_gm(
                module=DecomposeFloorDivide._IntDiv(),
                sample_input=(
                    torch.randint(1, 10, (3, 4), dtype=torch.int32),
                    torch.randint(1, 5, (3, 4), dtype=torch.int32),
                ),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_no_target(gm, torch.ops.aten.floor_divide.default)
            assertions.assert_target_count(gm, torch.ops.aten.floor.default, 1)

        # Float inputs: pass does not fire — floor_divide preserved as-is
        with subtests.test(msg="float"):
            gm = pass_pipeline.lower_export_gm(
                module=DecomposeFloorDivide._FloatDiv(),
                sample_input=(torch.randn(3, 4), torch.rand(3, 4) + 0.1),
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=quantizer,
            )
            assertions.assert_target_count(gm, torch.ops.aten.floor_divide.default, 1)
            assertions.assert_no_target(gm, torch.ops.aten.floor.default)


class DecomposeGlu:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.glu(x, dim=-1)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeGlu._Basic()
        inputs = (torch.randn(3, 8),)
        target_pass = _passes.DecomposeGlu

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(gm, torch.ops.aten.glu.default)


class DecomposeHardsigmoid:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardsigmoid(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeHardsigmoid._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeHardsigmoid
        gm = pass_pipeline.lower_annotation_gm(
            module=module,
            sample_input=inputs,
            target_pass=target_pass,
            backend_type=backend_type,
        )
        assertions.assert_no_target(gm, torch.ops.aten.hardsigmoid.default)


class DecomposeHyperbolicVariants:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.sinh(x), torch.cosh(x), torch.asinh(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeHyperbolicVariants._Basic()
        inputs = (torch.randn(3, 4).clamp(-0.9, 0.9),)
        target_pass = _passes.DecomposeHyperbolicVariants

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        for op in (
            torch.ops.aten.sinh.default,
            torch.ops.aten.cosh.default,
            torch.ops.aten.asinh.default,
            exir_ops.edge.aten.sinh.default,
            exir_ops.edge.aten.cosh.default,
            exir_ops.edge.aten.asinh.default,
        ):
            assertions.assert_no_target(gm, op)
        # sinh: exp×2, neg×1, sub×1, mul×1
        # cosh: exp×2, neg×1, add×1, mul×1
        # asinh: mul×1(x²), add×1(x²+1), sqrt×1, add×1(x+sqrt), log×1
        assertions.assert_target_count(
            gm, {torch.ops.aten.exp.default, exir_ops.edge.aten.exp.default}, 4
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.neg.default, exir_ops.edge.aten.neg.default}, 2
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sub.Tensor, exir_ops.edge.aten.sub.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor}, 3
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 3
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sqrt.default, exir_ops.edge.aten.sqrt.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.log.default, exir_ops.edge.aten.log.default}, 1
        )


class DecomposeLinalgVectorNorm:
    class _Norm(torch.nn.Module):
        def __init__(self, ord, dim, keepdim=False):
            super().__init__()
            self.ord = ord
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.linalg.vector_norm(
                x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
            )

    @staticmethod
    def _get_gm(module, inputs, target_pass, pass_pipeline, backend_type, quantizer):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeLinalgVectorNorm
        no_target = {
            torch.ops.aten.linalg_vector_norm.default,
            exir_ops.edge.aten.linalg_vector_norm.default,
        }
        _ = compile_spec

        # ord=2, dim given: abs + pow×2 + sum
        with subtests.test(msg="ord2_with_dim"):
            gm = DecomposeLinalgVectorNorm._get_gm(
                DecomposeLinalgVectorNorm._Norm(ord=2, dim=1),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_target)
            assertions.assert_target_count(
                gm, {torch.ops.aten.abs.default, exir_ops.edge.aten.abs.default}, 1
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.pow.Tensor_Scalar,
                    exir_ops.edge.aten.pow.Tensor_Scalar,
                },
                2,
            )
            assertions.assert_target_count(
                gm,
                {torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList},
                1,
            )

        # ord=2, dim=None: flatten + abs + pow×2 + sum
        with subtests.test(msg="ord2_dim_none"):
            gm = DecomposeLinalgVectorNorm._get_gm(
                DecomposeLinalgVectorNorm._Norm(ord=2, dim=None),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_target)
            assertions.assert_target_count(
                gm, {torch.ops.aten.abs.default, exir_ops.edge.aten.abs.default}, 1
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.pow.Tensor_Scalar,
                    exir_ops.edge.aten.pow.Tensor_Scalar,
                },
                2,
            )
            assertions.assert_target_count(
                gm,
                {torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList},
                1,
            )
            # flatten inserts a view_copy or reshape in the edge graph
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.flatten.using_ints,
                    exir_ops.edge.aten.view_copy.default,
                },
                1,
            )

        # ord=inf: abs + amax (no pow, no sum)
        with subtests.test(msg="ord_inf"):
            gm = DecomposeLinalgVectorNorm._get_gm(
                DecomposeLinalgVectorNorm._Norm(ord=float("inf"), dim=1),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_target)
            assertions.assert_target_count(
                gm, {torch.ops.aten.abs.default, exir_ops.edge.aten.abs.default}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.amax.default, exir_ops.edge.aten.amax.default}, 1
            )
            assertions.assert_no_target(
                gm,
                {
                    torch.ops.aten.pow.Tensor_Scalar,
                    exir_ops.edge.aten.pow.Tensor_Scalar,
                },
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList}
            )

        # ord=-inf: abs + amin (no pow, no sum)
        with subtests.test(msg="ord_neg_inf"):
            gm = DecomposeLinalgVectorNorm._get_gm(
                DecomposeLinalgVectorNorm._Norm(ord=float("-inf"), dim=1),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_target)
            assertions.assert_target_count(
                gm, {torch.ops.aten.abs.default, exir_ops.edge.aten.abs.default}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.amin.default, exir_ops.edge.aten.amin.default}, 1
            )
            assertions.assert_no_target(
                gm,
                {
                    torch.ops.aten.pow.Tensor_Scalar,
                    exir_ops.edge.aten.pow.Tensor_Scalar,
                },
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList}
            )


class DecomposeLogVariants:
    class _Log2(torch.nn.Module):
        def forward(self, x):
            return torch.log2(x)

    class _Log10(torch.nn.Module):
        def forward(self, x):
            return torch.log10(x)

    class _Log1p(torch.nn.Module):
        def forward(self, x):
            return torch.log1p(x)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                    quantizer=quantizer,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                        quantizer=quantizer,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.rand(3, 4) + 0.1,)
        target_pass = _passes.DecomposeLogVariants

        # log2(x) = log(x) / log(2) → log×1 + div×1
        with subtests.test(msg="log2"):
            gm = DecomposeLogVariants._get_gm(
                DecomposeLogVariants._Log2(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.log2.default, exir_ops.edge.aten.log2.default}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.log.default, exir_ops.edge.aten.log.default}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor}
            )

        # log10(x) = log(x) / log(10) → log×1 + div×1
        with subtests.test(msg="log10"):
            gm = DecomposeLogVariants._get_gm(
                DecomposeLogVariants._Log10(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.log10.default, exir_ops.edge.aten.log10.default}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.log.default, exir_ops.edge.aten.log.default}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor}
            )

        # log1p(x) = log(1 + x) → add×1 + log×1
        with subtests.test(msg="log1p"):
            gm = DecomposeLogVariants._get_gm(
                DecomposeLogVariants._Log1p(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.log1p.default, exir_ops.edge.aten.log1p.default}
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.log.default, exir_ops.edge.aten.log.default}, 1
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.add.Tensor, exir_ops.edge.aten.add.Tensor}, 1
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}
            )


class DecomposeMaxPool3d:
    class _WithoutIndices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x):
            return self.pool(x)[0]

    class _WithIndices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        def forward(self, x):
            return self.pool(x)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(1, 4, 8, 8, 8),)
        target_pass = _passes.DecomposeMaxPool3d

        # return_indices=False: decomposed into permute×2, reshape×2, max_pool2d×2
        with subtests.test(msg="without_indices"):
            gm = DecomposeMaxPool3d._get_gm(
                DecomposeMaxPool3d._WithoutIndices(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm,
                {
                    torch.ops.aten.max_pool3d.default,
                    torch.ops.aten.max_pool3d_with_indices.default,
                    exir_ops.edge.aten.max_pool3d.default,
                    exir_ops.edge.aten.max_pool3d_with_indices.default,
                },
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.max_pool2d.default,
                    torch.ops.aten.max_pool2d_with_indices.default,
                    exir_ops.edge.aten.max_pool2d.default,
                    exir_ops.edge.aten.max_pool2d_with_indices.default,
                },
                2,
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.permute.default,
                    exir_ops.edge.aten.permute_copy.default,
                },
                2,
            )
            assertions.assert_target_count(
                gm,
                {torch.ops.aten.reshape.default, exir_ops.edge.aten.view_copy.default},
                2,
            )

        # return_indices=True: pass returns early (unsupported), graph unchanged
        with subtests.test(msg="with_indices"):
            gm = DecomposeMaxPool3d._get_gm(
                DecomposeMaxPool3d._WithIndices(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.max_pool3d_with_indices.default,
                    exir_ops.edge.aten.max_pool3d_with_indices.default,
                },
                1,
            )


class DecomposeMinMaxDim:
    class _Min(torch.nn.Module):
        def forward(self, x):
            val, idx = torch.min(x, dim=1)
            return val, idx

    class _Max(torch.nn.Module):
        def forward(self, x):
            val, idx = torch.max(x, dim=1)
            return val, idx

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeMinMaxDim

        # torch.min: min.dim×1 preserved (value), argmin×1 inserted (index)
        with subtests.test(msg="min"):
            gm = pass_pipeline.lower_edge_ep(
                module=DecomposeMinMaxDim._Min(),
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            assertions.assert_target_count(gm, exir_ops.edge.aten.min.dim, 1)
            assertions.assert_target_count(gm, exir_ops.edge.aten.argmin.default, 1)

        # torch.max: max.dim×1 preserved (value), argmax×1 inserted (index)
        with subtests.test(msg="max"):
            gm = pass_pipeline.lower_edge_ep(
                module=DecomposeMinMaxDim._Max(),
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            assertions.assert_target_count(gm, exir_ops.edge.aten.max.dim, 1)
            assertions.assert_target_count(gm, exir_ops.edge.aten.argmax.default, 1)


class DecomposePad:
    class _Reflect2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = torch.nn.ReflectionPad2d(1)

        def forward(self, x):
            return self.pad(x)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                    quantizer=quantizer,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                        quantizer=quantizer,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.DecomposePad

        # reflect 2d (4 padding values): pad.default → reflection_pad2d
        with subtests.test(msg="reflect2d"):
            gm = DecomposePad._get_gm(
                DecomposePad._Reflect2d(),
                (torch.randn(1, 4, 8, 8),),
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            assertions.assert_no_target(
                gm, {torch.ops.aten.pad.default, exir_ops.edge.aten.pad.default}
            )
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.reflection_pad2d.default,
                    exir_ops.edge.aten.reflection_pad2d.default,
                },
                1,
            )


class DecomposeReciprocal:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.reciprocal(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeReciprocal._Basic()
        inputs = (torch.rand(3, 4) + 0.1,)
        target_pass = _passes.DecomposeReciprocal
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                    quantizer=quantizer,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                        quantizer=quantizer,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(gm, torch.ops.aten.reciprocal.default)
        assertions.assert_target_count(gm, torch.ops.aten.div.Tensor, 1)


class DecomposeRemainder:
    class _TensorDiv(torch.nn.Module):
        def forward(self, x, y):
            return torch.remainder(x, y)

    class _ScalarDiv(torch.nn.Module):
        def forward(self, x):
            return torch.remainder(x, 3.0)

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        compile_spec,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    def _assert_decomposed(gm, assertions):
        no_target = {
            torch.ops.aten.remainder.Tensor,
            torch.ops.aten.remainder.Scalar,
            exir_ops.edge.aten.remainder.Tensor,
            exir_ops.edge.aten.remainder.Scalar,
        }
        assertions.assert_no_target(gm, no_target)
        # remainder(x, y) = x - floor(x / y) * y
        assertions.assert_target_count(
            gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.floor.default, exir_ops.edge.aten.floor.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sub.Tensor, exir_ops.edge.aten.sub.Tensor}, 1
        )

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.DecomposeRemainder

        # Tensor divisor: y is a graph node — no get_attr inserted
        with subtests.test(msg="tensor_div"):
            gm = DecomposeRemainder._get_gm(
                DecomposeRemainder._TensorDiv(),
                (torch.randn(3, 4), torch.rand(3, 4) + 0.5),
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            DecomposeRemainder._assert_decomposed(gm, assertions)
            const_attrs = [
                n
                for n in gm.graph.nodes
                if n.op == "get_attr" and "_remainder_const_" in n.target
            ]
            assert (
                len(const_attrs) == 0
            ), "tensor divisor should not produce get_attr const nodes"

        # Scalar divisor: y is a raw Python scalar — lifted to a get_attr const node in edge dialect
        with subtests.test(msg="scalar_div"):
            gm = DecomposeRemainder._get_gm(
                DecomposeRemainder._ScalarDiv(),
                (torch.randn(3, 4),),
                target_pass,
                pass_pipeline,
                backend_type,
                compile_spec,
                quantizer,
            )
            DecomposeRemainder._assert_decomposed(gm, assertions)
            const_attrs = [
                n
                for n in gm.graph.nodes
                if n.op == "get_attr" and "_remainder_const_" in n.target
            ]
            # getattr node only appears in cases using lower_edge_ep
            if len(const_attrs) > 0:
                assert (
                    len(const_attrs) == 1
                ), f"scalar divisor should produce exactly 1 get_attr const node, got {len(const_attrs)}"


class DecomposeRoll:
    class _WithDim(torch.nn.Module):
        def forward(self, x):
            return torch.roll(x, shifts=2, dims=1)

    class _WithoutDim(torch.nn.Module):
        def forward(self, x):
            # dims=None: flatten → cat two slices → view
            return torch.roll(x, shifts=2)

    class _MultiDim(torch.nn.Module):
        def forward(self, x):
            # two dims: each produces slice_copy×2 + cat×1
            return torch.roll(x, shifts=(2, 1), dims=(0, 1))

    @staticmethod
    def _get_gm(
        module,
        inputs,
        target_pass,
        pass_pipeline,
        backend_type,
        quantizer,
    ):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 8),)
        target_pass = _passes.DecomposeRoll
        no_roll = {torch.ops.aten.roll.default, exir_ops.edge.aten.roll.default}
        _ = compile_spec

        # single dim: slice_copy×2 + cat×1
        with subtests.test(msg="with_dim"):
            gm = DecomposeRoll._get_gm(
                DecomposeRoll._WithDim(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_roll)
            assertions.assert_target_count(
                gm, {torch.ops.aten.slice.Tensor, exir_ops.edge.aten.slice.Tensor}, 2
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.cat.default, exir_ops.edge.aten.cat.default}, 1
            )

        # no dim: flatten + slice_copy×2 + cat×1 + view/reshape
        with subtests.test(msg="without_dim"):
            gm = DecomposeRoll._get_gm(
                DecomposeRoll._WithoutDim(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_roll)
            assertions.assert_target_count(
                gm, {torch.ops.aten.slice.Tensor, exir_ops.edge.aten.slice.Tensor}, 2
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.cat.default, exir_ops.edge.aten.cat.default}, 1
            )
            # flatten inserts a view_copy/reshape before the slices
            assertions.assert_target_count(
                gm,
                {
                    torch.ops.aten.flatten.using_ints,
                    exir_ops.edge.aten.view_copy.default,
                },
                1,
            )

        # two dims: slice_copy×4 + cat×2
        with subtests.test(msg="multi_dim"):
            gm = DecomposeRoll._get_gm(
                DecomposeRoll._MultiDim(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            assertions.assert_no_target(gm, no_roll)
            assertions.assert_target_count(
                gm, {torch.ops.aten.slice.Tensor, exir_ops.edge.aten.slice.Tensor}, 4
            )
            assertions.assert_target_count(
                gm, {torch.ops.aten.cat.default, exir_ops.edge.aten.cat.default}, 2
            )


class DecomposeSelectScatter:
    class _Basic(torch.nn.Module):
        def forward(self, x, src):
            return x.select_scatter(src, dim=1, index=0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeSelectScatter._Basic()
        inputs = (torch.randn(3, 4, 5), torch.randn(3, 5))
        target_pass = _passes.DecomposeSelectScatter
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm,
            {
                torch.ops.aten.select_scatter.default,
                exir_ops.edge.aten.select_scatter.default,
            },
        )
        assertions.assert_target_count(
            gm,
            {
                torch.ops.aten.slice_scatter.default,
                exir_ops.edge.aten.slice_scatter.default,
            },
            1,
        )
        assertions.assert_target_count(
            gm,
            {
                torch.ops.aten.unsqueeze.default,
                exir_ops.edge.aten.unsqueeze_copy.default,
            },
            1,
        )


class DecomposeSilu:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeSilu._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeSilu

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    quantizer=quantizer,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        quantizer=quantizer,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm, {torch.ops.aten.silu.default, torch.ops.aten.silu_.default}
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sigmoid.default, exir_ops.edge.aten.sigmoid.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 1
        )


class DecomposeTan:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.tan(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeTan._Basic()
        inputs = (torch.rand(3, 4) * 2 - 1,)
        target_pass = _passes.DecomposeTan

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                    quantizer=quantizer,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                        quantizer=quantizer,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm, {torch.ops.aten.tan.default, exir_ops.edge.aten.tan.default}
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.sin.default, exir_ops.edge.aten.sin.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.cos.default, exir_ops.edge.aten.cos.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.div.Tensor, exir_ops.edge.aten.div.Tensor}, 1
        )


class DecomposeThreshold:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.threshold(x, threshold=0.5, value=0.0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeThreshold._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeThreshold
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm, {torch.ops.aten.threshold.default, torch.ops.aten.threshold_.default}
        )


class DecomposeTriu:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.triu(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeTriu._Basic()
        inputs = (torch.randn(4, 4),)
        target_pass = _passes.DecomposeTriu
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(gm, torch.ops.aten.triu.default)


class DecomposeTrunc:
    class _Basic(torch.nn.Module):
        def forward(self, x):
            return torch.trunc(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeTrunc._Basic()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeTrunc

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                ).graph_module
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_edge_ep(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        compile_spec=compile_spec,
                        target_pass=target_pass,
                    ).graph_module
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        assertions.assert_no_target(
            gm, {torch.ops.aten.trunc.default, exir_ops.edge.aten.trunc.default}
        )
        # trunc(x) = sign(x) * floor(abs(x))
        assertions.assert_target_count(
            gm, {torch.ops.aten.sign.default, exir_ops.edge.aten.sign.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.abs.default, exir_ops.edge.aten.abs.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.floor.default, exir_ops.edge.aten.floor.default}, 1
        )
        assertions.assert_target_count(
            gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 1
        )


class DecomposeVar:
    class _WithCorrection(torch.nn.Module):
        def forward(self, x):
            # correction=1 (Bessel's): scale = N/(N-1), adds mul + get_attr scale node
            return torch.var(x, dim=1, correction=1)

    class _NoCorrection(torch.nn.Module):
        def forward(self, x):
            # correction=0: biased variance, no scale step
            return torch.var(x, dim=1, correction=0)

    @staticmethod
    def _get_gm(module, inputs, target_pass, pass_pipeline, backend_type, quantizer):
        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    return pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    return pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                return pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")

    @staticmethod
    def _assert_common(gm, assertions):
        """Ops present in both correction and no-correction cases."""
        assertions.assert_no_target(
            gm,
            {
                torch.ops.aten.var.correction,
                torch.ops.aten.var.dim,
                exir_ops.edge.aten.var.correction,
                exir_ops.edge.aten.var.dim,
            },
        )
        # mean×2: mean_x (keepdim=True) + mean of squared diff
        assertions.assert_target_count(
            gm, {torch.ops.aten.mean.dim, exir_ops.edge.aten.mean.dim}, 2
        )
        # sub×1: x - mean_x
        assertions.assert_target_count(
            gm, {torch.ops.aten.sub.Tensor, exir_ops.edge.aten.sub.Tensor}, 1
        )

    @staticmethod
    @unpack_pass_fixtures
    def test(
        subtests,
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeVar
        _ = compile_spec

        # correction=1: mean×2 + sub×1 + mul×2 (diff² + scale) + get_attr scale const
        with subtests.test(msg="with_correction"):
            gm = DecomposeVar._get_gm(
                DecomposeVar._WithCorrection(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            DecomposeVar._assert_common(gm, assertions)
            # mul×2: diff*diff + var_mean * scale
            assertions.assert_target_count(
                gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 2
            )

        # correction=0: mean×2 + sub×1 + mul×1 (diff² only), no scale step
        with subtests.test(msg="no_correction"):
            gm = DecomposeVar._get_gm(
                DecomposeVar._NoCorrection(),
                inputs,
                target_pass,
                pass_pipeline,
                backend_type,
                quantizer,
            )
            DecomposeVar._assert_common(gm, assertions)
            # mul×1: diff*diff only, no scale multiplication
            assertions.assert_target_count(
                gm, {torch.ops.aten.mul.Tensor, exir_ops.edge.aten.mul.Tensor}, 1
            )


class DecomposeWrapWithAutocast:
    class _WithAutocast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule = torch.nn.ReLU()

        @torch.amp.autocast("cpu")
        def forward(self, arg0_1):
            return self.submodule(arg0_1)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = DecomposeWrapWithAutocast._WithAutocast()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.DecomposeWrapWithAutocast
        _ = compile_spec

        match backend_type:
            case QnnExecuTorchBackendType.kGpuBackend:
                gm = pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    target_pass=target_pass,
                )
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    gm = pass_pipeline.lower_annotation_gm(
                        module=module,
                        sample_input=inputs,
                        target_pass=target_pass,
                        backend_type=backend_type,
                    )
                else:
                    gm = pass_pipeline.lower_export_gm(
                        module=module,
                        sample_input=inputs,
                        backend_type=backend_type,
                        target_pass=target_pass,
                    )
            case QnnExecuTorchBackendType.kLpaiBackend:
                gm = pass_pipeline.lower_annotation_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")
        for node in gm.graph.nodes:
            assert not isinstance(
                node.target, torch._higher_order_ops.wrap.WrapWithAutocast
            ), "WrapWithAutocast should be decomposed"
        assertions.assert_target_count(
            gm, {torch.ops.aten.relu.default, exir_ops.edge.aten.relu.default}, 1
        )


class ExpandBroadcastTensorShape:
    class _Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class _Expand(torch.nn.Module):
        def forward(self, y):
            return y.expand(3, 4)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
        subtests,
    ):
        target_pass = _passes.ExpandBroadcastTensorShape
        cases = [
            (
                "add",
                ExpandBroadcastTensorShape._Add(),
                (torch.randn(3, 4), torch.randn(4)),
                exir_ops.edge.aten.add.Tensor,
            ),
            (
                "expand",
                ExpandBroadcastTensorShape._Expand(),
                (torch.randn(4),),
                exir_ops.edge.aten.expand_copy.default,
            ),
        ]
        for name, module, inputs, broadcast_target in cases:
            with subtests.test(msg=name):
                gm = pass_pipeline.lower_edge_ep(
                    module=module,
                    sample_input=inputs,
                    backend_type=backend_type,
                    compile_spec=compile_spec,
                    target_pass=target_pass,
                    quantizer=quantizer,
                ).graph_module
                assertions.assert_target_count(gm, broadcast_target, 1)
                # verify view_copy node exists as it does the broadcast.
                assertions.assert_target_count(
                    gm, exir_ops.edge.aten.view_copy.default, 1
                )


class FixedLinearKeepDim:
    class _Linear3D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 16)

        def forward(self, x):
            return self.linear(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        # Need an input tensor rank >= 3 to trigger the pass.
        module = FixedLinearKeepDim._Linear3D()
        inputs = (torch.randn(2, 4, 8),)
        target_pass = _passes.FixedLinearKeepDim
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module

        assertions.assert_target_count(gm, exir_ops.edge.aten.linear.default, 1)
        assertions.assert_target_count(gm, exir_ops.edge.aten.view_copy.default, 2)


class FoldQDQ:
    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
        subtests,
    ):
        module = SimpleModel()
        inputs = (torch.randn(1, 32, 8, 8), torch.randn(1, 32, 8, 8))
        target_pass = _passes.FoldQDQ
        q_targets = set(q_ops)
        dq_targets = set(dq_ops)

        # Stage 1: edge pipeline (force_fold=False).
        with subtests.test(msg="edge"):
            gm = pass_pipeline.lower_edge_ep(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            # Q nodes are always folded; activation DQ too.
            assertions.assert_no_target(gm, q_targets)
            if quantizer:
                # DQ node after weights are preserved during stage 1.
                # 9 dq node = conv1 (weight+bias) + conv2 (weight+bias)
                #     + conv3 (weight-only) + conv4 (weight-only)
                #     + linear (weight+bias) + batch_norm (weight, shared)
                assertions.assert_target_count(gm, dq_targets, 9)
            else:
                assertions.assert_no_target(gm, dq_targets)

        # Stage 2: preprocess pipeline (force_fold=True).
        with subtests.test(msg="preprocess"):
            gm = pass_pipeline.lower_preprocess_gm(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            )
            # Everything is force-folded after preprocess pass.
            assertions.assert_no_target(gm, q_targets)
            assertions.assert_no_target(gm, dq_targets)


class FuseConsecutiveCast:
    class _ConsecutiveCast(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.int32).to(torch.float) + 1.0

    # Both edge cast ops the pass recognises (op_map).
    _CAST_OPS = {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = FuseConsecutiveCast._ConsecutiveCast()
        inputs = (torch.randn(3, 4),)
        target_pass = _passes.FuseConsecutiveCast
        gm = pass_pipeline.lower_preprocess_gm(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        )

        # Multiple casting nodes fused to a single cast node
        assertions.assert_no_consecutive(gm, FuseConsecutiveCast._CAST_OPS)
        assertions.assert_target_count_at_most(gm, FuseConsecutiveCast._CAST_OPS, 1)


class FuseConsecutiveTranspose:
    class _ConsecutivePermute(torch.nn.Module):
        def forward(self, x):
            a = x.permute(2, 0, 1)
            b = a.permute(1, 2, 0)
            return b + 1.0

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = FuseConsecutiveTranspose._ConsecutivePermute()
        inputs = (torch.randn(3, 4, 5),)
        target_pass = _passes.FuseConsecutiveTranspose
        permute = exir_ops.edge.aten.permute_copy.default
        gm = pass_pipeline.lower_preprocess_gm(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        )

        # Multiple transpose nodes fused to a single transpose node
        assertions.assert_no_consecutive(gm, permute)
        assertions.assert_target_count_at_most(gm, permute, 1)


class I64toI32:
    class _ArgminViewSqueezeConv2D(torch.nn.Module):
        def __init__(self):
            # This model is mainly to test the PASS I64toI32
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            )

        def forward(self, x, y):
            argmin_out = torch.argmin(x, dim=0, keepdim=True)
            index_out = y[argmin_out]
            conv_out = self.conv(index_out)

            view_out = argmin_out.view(-1)
            squeeze_out = view_out.squeeze(-1)
            return squeeze_out, conv_out

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = I64toI32._ArgminViewSqueezeConv2D()
        inputs = (torch.randn(32), torch.randn(32, 3, 32, 32))
        target_pass = _passes.I64toI32
        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module

        # Quantized flow has an extra unnecessary cast after zero_point constant.
        # The reason is zero_point constant is i64, so the pass added a cast there.
        expected_casts = 2 if quantizer is None else 3
        assertions.assert_target_count(
            gm, exir_ops.edge.aten._to_copy.default, expected_casts
        )


class InsertIOQDQ:
    class _Basic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.relu(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        module = InsertIOQDQ._Basic()
        inputs = (torch.randn(1, 4),)
        target_pass = _passes.InsertIOQDQ
        gm = pass_pipeline.lower_preprocess_gm(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        )
        match backend_type:
            case QnnExecuTorchBackendType.kHtpBackend:
                if quantizer:
                    assertions.assert_target_count(
                        gm,
                        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                        1,
                    )
                    assertions.assert_target_count(
                        gm,
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                        1,
                    )
                else:
                    assertions.assert_no_target(
                        gm,
                        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                    )
                    assertions.assert_no_target(
                        gm,
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                    )
            case _:
                raise AssertionError(f"unhandled backend_type: {backend_type}")


class InsertRequantize:
    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.backends.qualcomm.export_utils import make_quantizer
        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_submodule_type_predicate,
            ModuleQConfig,
            QuantDtype,
        )
        from executorch.backends.qualcomm.tests.models import SimpleSubModules

        # Uses a custom quantizer to achieve mix precision, which triggers requantize
        mixed_quantizer = make_quantizer(
            backend=backend_type,
            submodule_qconfig_list=[
                (
                    get_submodule_type_predicate("Add"),
                    ModuleQConfig(QuantDtype.use_16a16w),
                ),
            ],
        )
        module = SimpleSubModules()
        inputs = (
            torch.randn(1, 4),
            torch.randn(1, 4),
            torch.randn(1, 4),
            torch.randn(1, 4),
        )
        target_pass = _passes.InsertRequantize
        gm = pass_pipeline.lower_preprocess_gm(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=mixed_quantizer,
        )

        # 1 requantize to_copy node: Add(16a) -> to_copy -> Mul(8a).
        assertions.assert_target_count(gm, exir_ops.edge.aten._to_copy.default, 1)


class InsertReshapeForReduceOps:
    class _ArgmaxAll(torch.nn.Module):
        def forward(self, x):
            return torch.argmax(x)  # dim=None -> flatten then argmax(0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        target_pass = _passes.InsertReshapeForReduceOps
        inputs = (torch.randn(3, 4),)
        argmax = torch.ops.aten.argmax.default
        reshape = torch.ops.aten.reshape.default

        # Mirror production stage selection: the annotation pipeline only runs
        # under prepare_pt2e (quantizer present). Without a quantizer (GPU,
        # htp_fp16) this pass runs in the export pipeline instead.
        if quantizer is None:
            gm = pass_pipeline.lower_export_gm(
                module=InsertReshapeForReduceOps._ArgmaxAll(),
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
                quantizer=None,
            )
        else:
            gm = pass_pipeline.lower_annotation_gm(
                module=InsertReshapeForReduceOps._ArgmaxAll(),
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
            )

        assertions.assert_target_count(gm, argmax, 1)
        assertions.assert_target_count(gm, reshape, 1)


class LayoutTransform:
    class _Conv1dRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 8, 3, padding=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
        subtests,
    ):
        from executorch.backends.qualcomm.utils.constants import QCOM_INSERTED_PERMUTE

        module = LayoutTransform._Conv1dRelu()
        inputs = (torch.randn(1, 3, 16),)
        target_pass = _passes.LayoutTransform
        conv = exir_ops.edge.aten.convolution.default
        permute = exir_ops.edge.aten.permute_copy.default

        # Phase 1 (edge stage): QCOM_AXIS_ORDER tagged on the conv node, no permute nodes inserted yet.
        with subtests.test(msg="edge"):
            gm = pass_pipeline.lower_edge_ep(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
            assertions.assert_meta_key_on_targets(gm, conv, QCOM_AXIS_ORDER)
            assertions.assert_no_target(gm, permute)

        # Phase 2 (preprocess stage): one permute inserted before and after
        # the conv node; every permute carries the QCOM_INSERTED_PERMUTE flag.
        # FP graph should look like: unsqueeze → permute → convolution → permute → squeeze → relu
        with subtests.test(msg="preprocess"):
            gm = pass_pipeline.lower_preprocess_gm(
                module=module,
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            )
            assertions.assert_target_count(gm, permute, 2)
            assertions.assert_meta_key_on_targets(gm, permute, QCOM_INSERTED_PERMUTE)


class LiftConstantScalarOperands:
    class _Pow(torch.nn.Module):
        def forward(self, x):
            return x**2

    class _LeakyRelu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.leaky_relu(x, negative_slope=0.05)

    class _Hardtanh(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)

    class _Where(torch.nn.Module):
        def forward(self, x):
            return torch.where(x > 0, 1.0, 0.0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
        subtests,
    ):
        target_pass = _passes.LiftConstantScalarOperands
        inputs = (torch.randn(3, 4),)

        # Mirror production stage selection: the annotation pipeline only runs
        # under prepare_pt2e (quantizer present). Without a quantizer (GPU,
        # htp_fp16) this pass runs in the export pipeline instead.
        def lower(module):
            if quantizer is None:
                return pass_pipeline.lower_export_gm(
                    module=module,
                    sample_input=inputs,
                    target_pass=target_pass,
                    backend_type=backend_type,
                    quantizer=None,
                )
            return pass_pipeline.lower_annotation_gm(
                module=module,
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
            )

        with subtests.test(msg="scalar_to_tensor"):
            gm = lower(LiftConstantScalarOperands._Pow())
            assertions.assert_no_target(gm, torch.ops.aten.pow.Tensor_Scalar)
            assertions.assert_target_count(gm, torch.ops.aten.pow.Tensor_Tensor, 1)

        with subtests.test(msg="leaky_relu_to_prelu"):
            gm = lower(LiftConstantScalarOperands._LeakyRelu())
            assertions.assert_no_target(gm, torch.ops.aten.leaky_relu.default)
            assertions.assert_target_count(gm, torch.ops.aten.prelu.default, 1)

        with subtests.test(msg="skip_list_hardtanh"):
            gm = lower(LiftConstantScalarOperands._Hardtanh())
            assertions.assert_target_count(gm, torch.ops.aten.hardtanh.default, 1)

        with subtests.test(msg="where_scalar_use_self_dtype"):
            gm = lower(LiftConstantScalarOperands._Where())
            assertions.assert_no_target(gm, torch.ops.aten.where.Scalar)
            assertions.assert_target_count(gm, torch.ops.aten.where.self, 1)
            where_node = next(
                n
                for n in gm.graph.nodes
                if n.op == "call_function" and n.target is torch.ops.aten.where.self
            )
            lifted = [
                a
                for a in where_node.args
                if isinstance(a, torch.fx.Node) and a.op == "get_attr"
            ]
            assert (
                len(lifted) == 2
            ), f"expected 2 lifted scalar get_attrs feeding where, found {len(lifted)}"
            for n in lifted:
                assert n.meta["val"].dtype == torch.float32, (
                    f"lifted scalar {n.name} expected float32, "
                    f"got {n.meta['val'].dtype} (use_self_dtype broken?)"
                )


class LpaiPartitionFallbackSupport:
    _SKIP_NODE_ID_SET = {"aten_add_tensor", "aten_mean_dim"}

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_BYPASS_NODE,
            QCOM_FALLBACK_NODE,
        )

        module = SimpleModel()
        inputs = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        target_pass = _passes.LpaiPartitionFallbackSupport

        gm = pass_pipeline.lower_edge_ep(
            module=module,
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
            skip_node_id_set=LpaiPartitionFallbackSupport._SKIP_NODE_ID_SET,
        ).graph_module

        #   Graph should look like
        #   prev -> Q[BP] -> DQ[FB] -> fallback source node[FB] -> Q[FB] -> DQ[BP] -> next
        for name in LpaiPartitionFallbackSupport._SKIP_NODE_ID_SET:
            assertions.assert_meta_key_on_named_node(gm, name, QCOM_FALLBACK_NODE)
            fallback_node = next(n for n in gm.graph.nodes if n.name == name)

            # Input side: -> Q -> DQ -> fallback_node
            # Q tagged as bypassed and DQ tagged as fallback.
            input_dqs = [
                i
                for i in fallback_node.all_input_nodes
                if i.op == "call_function" and i.target in dq_ops
            ]
            assert input_dqs, f"expected DQ inputs feeding {name}, found none"
            for dq in input_dqs:
                assertions.assert_meta_key_on_named_node(
                    gm, dq.name, QCOM_FALLBACK_NODE
                )
                q = dq.args[0]
                assertions.assert_meta_key_on_named_node(gm, q.name, QCOM_BYPASS_NODE)

            # Output side: fallback_node -> Q -> DQ ->
            # Q tagged as fallback and DQ tagged as bypassed.
            output_qs = [
                u
                for u in fallback_node.users
                if u.op == "call_function" and u.target in q_ops
            ]
            assert output_qs, f"expected Q consumers of {name}, found none"
            for q_out in output_qs:
                assertions.assert_meta_key_on_named_node(
                    gm, q_out.name, QCOM_FALLBACK_NODE
                )
                dq_downs = [
                    u
                    for u in q_out.users
                    if u.op == "call_function" and u.target in dq_ops
                ]
                assert dq_downs, f"expected DQ downstream of {q_out.name}, found none"
                for dq_out in dq_downs:
                    assertions.assert_meta_key_on_named_node(
                        gm, dq_out.name, QCOM_BYPASS_NODE
                    )


class RecomposePadMaxPool2d:
    class _MaxPoolPadded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=3, padding=1)

        def forward(self, x):
            return self.pool(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(1, 3, 8, 8),)
        target_pass = _passes.RecomposePadMaxPool2d
        maxpool = exir_ops.edge.aten.max_pool2d_with_indices.default
        const_pad = exir_ops.edge.aten.constant_pad_nd.default

        gm = pass_pipeline.lower_edge_ep(
            module=RecomposePadMaxPool2d._MaxPoolPadded(),
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=target_pass,
            quantizer=quantizer,
        ).graph_module
        assertions.assert_target_count(gm, const_pad, 1)
        mp_nodes = [
            n for n in gm.graph.nodes if n.op == "call_function" and n.target == maxpool
        ]
        assert len(mp_nodes) == 1, f"expected 1 max_pool2d, got {len(mp_nodes)}"
        assert tuple(mp_nodes[0].args[3]) == (0, 0), (
            f"expected max_pool2d padding to be (0, 0) after pass, "
            f"got {mp_nodes[0].args[3]}"
        )


class RecomposePixelUnshuffle:
    class _ManualPixelUnshuffle(torch.nn.Module):
        def forward(self, x):
            b, c, h, w = x.shape
            s = 2
            y = x.view(b, c, h // s, s, w // s, s)
            y = y.permute(0, 1, 3, 5, 2, 4)
            return y.reshape(b, c * s * s, h // s, w // s)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(1, 4, 8, 8),)
        target_pass = _passes.RecomposePixelUnshuffle
        if quantizer is None:
            gm = pass_pipeline.lower_edge_ep(
                module=RecomposePixelUnshuffle._ManualPixelUnshuffle(),
                sample_input=inputs,
                backend_type=backend_type,
                compile_spec=compile_spec,
                target_pass=target_pass,
                quantizer=quantizer,
            ).graph_module
        else:
            gm = pass_pipeline.lower_annotation_gm(
                module=RecomposePixelUnshuffle._ManualPixelUnshuffle(),
                sample_input=inputs,
                target_pass=target_pass,
                backend_type=backend_type,
            )

        assertions.assert_target_count(
            gm,
            {
                torch.ops.aten.pixel_unshuffle.default,
                exir_ops.edge.aten.pixel_unshuffle.default,
            },
            1,
        )
        assertions.assert_no_target(
            gm,
            {torch.ops.aten.view_copy.default, exir_ops.edge.aten.view_copy.default},
        )
        assertions.assert_no_target(
            gm,
            {
                torch.ops.aten.permute_copy.default,
                exir_ops.edge.aten.permute_copy.default,
            },
        )


class RecomposeRmsNorm:
    class _RmsNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return norm * self.weight

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        dim = 16
        inputs = (torch.randn(1, 4, 8, dim),)
        gm = pass_pipeline.lower_edge_ep(
            module=RecomposeRmsNorm._RmsNorm(dim),
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=_passes.RecomposeRmsNorm,
            quantizer=quantizer,
        ).graph_module
        assertions.assert_target_count(gm, exir_ops.edge.aten.rms_norm.default, 1)
        assertions.assert_no_target(gm, exir_ops.edge.aten.rsqrt.default)
        assertions.assert_no_target(gm, exir_ops.edge.aten.mean.dim)


class RemoveRedundancy:
    class _Clone(torch.nn.Module):
        def forward(self, x):
            return x.clone() + 1

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(3, 4),)
        gm = pass_pipeline.lower_edge_ep(
            module=RemoveRedundancy._Clone(),
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=_passes.RemoveRedundancy,
            quantizer=quantizer,
        ).graph_module
        assertions.assert_no_target(
            gm, exir_ops.edge.dim_order_ops._clone_dim_order.default
        )
        assertions.assert_target_count(gm, exir_ops.edge.aten.add.Tensor, 1)


class ReplaceArangeArgs:
    class _Arange(torch.nn.Module):
        def forward(self, x):
            r = torch.arange(x.shape[-1], dtype=torch.float32)
            return x + r

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        gm = pass_pipeline.lower_annotation_gm(
            module=ReplaceArangeArgs._Arange(),
            sample_input=(torch.randn(3, 4),),
            target_pass=_passes.ReplaceArangeArgs,
            backend_type=backend_type,
        )

        # arange.default disappears in favor of arange.start_step with the
        # float step-size that hints fp dtype.
        assertions.assert_no_target(gm, torch.ops.aten.arange.default)
        assertions.assert_target_count(gm, torch.ops.aten.arange.start_step, 1)


class ResolveDebugHandle:
    """
    Assigns each call_function node its own unique DEBUG_HANDLE_KEY meta so
    intermediate-tensor debugging can key on it. getitem nodes inherit the
    handle from their producing tuple-output node.

    Runs as the last edge pass (required — validated by _validate_edge_passes).

    Handle numbering is assigned by graph traversal order. The quantized flow
    prepends a dequantize node that takes handle 1, shifting every subsequent
    handle by one relative to the fp path; the test accounts for this offset.
    """

    class _TopKandIndex(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.idx_source = torch.rand(10, 3)

        def forward(self, x):
            a, b = torch.topk(x, 3)
            return a + self.idx_source[b]

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY

        # Base fp map. Extra DQ node for quantized flow due to self.idx_source.
        offset = 0 if quantizer is None else 1
        name_handle_map = {
            "aten_topk_default": 1 + offset,
            "getitem": 1 + offset,
            "getitem_1": 1 + offset,
            "aten_view_copy_default": 2 + offset,
            "aten_index_tensor": 3 + offset,
            "aten_add_tensor": 4 + offset,
        }
        if quantizer is not None:
            name_handle_map["quantized_decomposed_dequantize_per_tensor_default"] = 1

        inputs = (torch.randn(3, 10),)
        gm = pass_pipeline.lower_edge_ep(
            module=ResolveDebugHandle._TopKandIndex(),
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=_passes.ResolveDebugHandle,
            quantizer=quantizer,
        ).graph_module

        for node in gm.graph.nodes:
            if node.name in name_handle_map:
                expected = name_handle_map.pop(node.name)
                got = node.meta[DEBUG_HANDLE_KEY]
                assert (
                    expected == got
                ), f"{node.name} expected handle {expected}, got {got}"
        assert (
            len(name_handle_map) == 0
        ), f"nodes not found in graph: {list(name_handle_map)}"


class Remove0DTensor:
    class _ZeroDim(torch.nn.Module):
        def forward(self, x, y):
            return y + torch.select(x, 0, 0)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        quantizer,
        compile_spec,
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
        pass_pipeline: PassPipeline,
    ):
        inputs = (torch.randn(4), torch.randn(3, 4))
        gm = pass_pipeline.lower_edge_ep(
            module=Remove0DTensor._ZeroDim(),
            sample_input=inputs,
            backend_type=backend_type,
            compile_spec=compile_spec,
            target_pass=_passes.Remove0DTensor,
            quantizer=quantizer,
        ).graph_module

        assertions.assert_no_target(gm, exir_ops.edge.aten.select_copy.int)
        assertions.assert_target_count(gm, exir_ops.edge.aten.add.Tensor, 1)


class SeqMSE:
    class _ConvSingle(torch.nn.Module):
        def __init__(self, in_channel: int = 512, out_channel: int = 32):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channel, out_channel, kernel_size=(1, 1), padding=0
            )

        def forward(self, x):
            return self.conv(x)

    @staticmethod
    @unpack_pass_fixtures
    def test(
        backend_type: QnnExecuTorchBackendType,
        assertions: Assertions,
    ):
        from executorch.backends.qualcomm._passes.seq_mse import (
            SeqMSE as SeqMSEContext,
            SeqMseModule,
        )
        from executorch.backends.qualcomm.export_utils import make_quantizer
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        module = SeqMSE._ConvSingle()
        sample_input = (torch.randn(1, 512, 1, 32),)
        quantizer = make_quantizer(backend=backend_type)
        exported = torch.export.export(module, sample_input, strict=True).module()
        prepared = prepare_pt2e(exported, quantizer)

        assertions.assert_target_count(prepared, torch.ops.aten.conv2d.default, 1)

        def _seq_mse_call_modules(gm):
            return [
                n
                for n in gm.graph.nodes
                if n.op == "call_module"
                and isinstance(getattr(gm, n.target, None), SeqMseModule)
            ]

        with SeqMSEContext(prepared, num_candidates=10):
            inside = _seq_mse_call_modules(prepared)
            assert len(inside) == 1, (
                f"expected 1 SeqMseModule call_module inside context, "
                f"got {len(inside)}"
            )
            prepared(*sample_input)

        after = _seq_mse_call_modules(prepared)
        assert len(after) == 0, (
            f"expected 0 SeqMseModule call_module after context exit, "
            f"got {len(after)}"
        )
        # convert_pt2e to ensure it doesn't break the module.
        convert_pt2e(prepared)


class TagQuantIO:
    class _Relu(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    @staticmethod
    @unpack_pass_fixtures
    def test():
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_QUANT_ATTRS,
            QCOM_QUANT_ATTRS_MAP,
            QCOM_QUANTIZED_IO,
        )

        module = TagQuantIO._Relu()
        inputs = (torch.randn(3, 4),)
        ep = torch.export.export(module, inputs, strict=True)
        gm = ep.graph_module

        # Seed the graph with QCOM_QUANT_ATTRS on call_functions so the pass
        # populates QCOM_QUANT_ATTRS_MAP downstream.
        for n in gm.graph.nodes:
            if n.op == "call_function":
                n.meta[QCOM_QUANT_ATTRS] = {"scale": 0.1, "zero_point": 0}

        # Callback: tag placeholder inputs as uint8.
        def get_dtype(node):
            return torch.uint8 if node.op == "placeholder" else None

        _passes.TagQuantIO(get_dtype)(gm)

        # Trait 1: every input placeholder carries QCOM_QUANTIZED_IO with the
        # dtype the callback returned.
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        assert placeholders, "expected at least one placeholder"
        for n in placeholders:
            assert (
                QCOM_QUANTIZED_IO in n.meta
            ), f"{n.name} missing {QCOM_QUANTIZED_IO!r}"
            assert (
                n.meta[QCOM_QUANTIZED_IO] == torch.uint8
            ), f"expected {n.name} tagged uint8, got {n.meta[QCOM_QUANTIZED_IO]}"

        # Trait 2: every output node carries QCOM_QUANT_ATTRS_MAP with one entry
        # per returned tensor that carried QCOM_QUANT_ATTRS.
        outputs = [n for n in gm.graph.nodes if n.op == "output"]
        assert outputs, "expected at least one output"
        for n in outputs:
            assert (
                QCOM_QUANT_ATTRS_MAP in n.meta
            ), f"{n.name} missing {QCOM_QUANT_ATTRS_MAP}"
            num_returned = len(n.args[0])
            assert len(n.meta[QCOM_QUANT_ATTRS_MAP]) == num_returned, (
                f"expected {num_returned} entries in QCOM_QUANT_ATTRS_MAP, "
                f"got {len(n.meta[QCOM_QUANT_ATTRS_MAP])}"
            )
