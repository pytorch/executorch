# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import json
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
    QnnPassManager,
)

from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.debugger.utils import generate_optrace
from executorch.backends.qualcomm.tests.utils import (
    convert_pt2e,
    generate_context_binary,
    ModuleQConfig,
    prepare_pt2e,
    QuantDtype,
    TestQNN,
    validate_context_binary,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_ANNOTATION,
    QCOM_MODULE,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_QUANT_ATTRS_MAP,
    QCOM_QUANT_DTYPE,
    QCOM_SAMPLE_INPUTS,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    dump_context_from_pte,
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    is_qnn_sdk_version_less_than,
    PyQnnManagerAdaptor,
    rewrite_prepared_observer,
    skip_annotation,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)

from executorch.examples.qualcomm.utils import (
    make_quantizer,
    setup_common_args_and_variables,
)

from executorch.backends.qualcomm.tests.models import *  # noqa: F403

import os
import random

from collections import defaultdict
from typing import List

from executorch.backends.qualcomm._passes import (
    ExpandBroadcastTensorShape,
    FoldQDQ,
    TagQuantIO,
)
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.debugger.utils import DrawGraph
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.models.inception_v3 import InceptionV3Model
from executorch.examples.models.inception_v4 import InceptionV4Model

# from executorch.examples.models.mobilebert import MobileBertModelExample
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.examples.models.mobilenet_v3 import MV3Model
from executorch.examples.models.torchvision_vit.model import TorchVisionViTModel

from executorch.examples.models.wav2letter import Wav2LetterModel
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import disable_validation


class TestQNNFloatingPointOperator(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_abs(self):
        module = Abs()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_adaptive_avg_pool1d(self):
        module = AdaptiveAvgPool1D()  # noqa: F405
        sample_input = (torch.randn(1, 512, 7),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_adaptive_avg_pool2d(self):
        module = AdaptiveAvgPool2D()  # noqa: F405
        sample_input = (torch.randn(1, 512, 7, 7),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_alias(self):
        module = Alias()  # noqa: F405
        sample_input = (torch.randn(1, 10),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amax(self):
        modules = [
            AMax(dim=1, keepdim=False),  # noqa: F405
            AMax(dim=1, keepdim=True),  # noqa: F405
            AMax(),  # noqa: F405
        ]
        sample_input = (torch.randn(4, 4),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amax_conv(self):
        sample_input = (torch.randn(2, 3, 64, 64),)  # [batch, channels, height, width]
        module = AMaxFollowingConv2D(  # noqa: F405
            in_channels=3, out_channels=16, kernel_size=3, dim=-1, keepdim=False
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amin(self):
        modules = [
            AMin(dim=1, keepdim=False),  # noqa: F405
            AMin(dim=1, keepdim=True),  # noqa: F405
            AMin(),  # noqa: F405
        ]
        sample_input = (torch.randn(4, 4),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_any(self):
        modules = [Any(), Any(dim=[0, 1]), Any(dim=1, keepdim=True)]  # noqa: F405
        sample_input = (torch.randn(3, 3, 3) > 0,)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_arange(self):
        modules = [
            Arange(start=1, end=11, step=1, dtype=torch.int32),  # noqa: F405
        ]
        sample_input = (torch.randn(10),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_argmax(self):
        module = Argmax()  # noqa: F405
        sample_input = (torch.randn(16, 3, 4, 4),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_argmin(self):
        module = Argmin()  # noqa: F405
        sample_input = (torch.rand(3, 4),)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.expectedFailure
    def test_qnn_backend_asin(self):
        sample_input = (torch.rand(3, 4) * 2 - 1,)
        module = Asin()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_atan(self):
        sample_input = (torch.randn(3, 4),)
        module = Atan()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_avg_pool2d(self):
        modules = [
            AvgPoolModule((2, 2), (1, 1), (1, 1), False),  # noqa: F405
            AvgPoolModule((1280, 1280), (1280, 1280), (0, 0), True),  # noqa: F405
            AvgPoolModule((1280, 1280), (1280, 1280), (320, 320), True),  # noqa: F405
        ]  # noqa: F405
        sample_inputs = [
            (torch.randn(1, 3, 2, 2),),
            (torch.randn(1, 1280, 7, 7),),
            (torch.randn(1, 1280, 7, 7),),
        ]
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_inputs[i])

    def test_qnn_backend_batch_norm(self):
        modules = [BatchNorm(32), BatchNorm(32, False)]  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([4, 8, 32]), torch.randn([4, 32, 8]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cast(self):
        modules = [Cast(), CastMultiUsers()]  # noqa: F405
        sample_inputs = [
            (10 * torch.rand((9, 4, 5, 3)),),
            (torch.randint(0, 3, size=(3, 3)), torch.randn(3, 3)),
        ]
        for i, (module, sample_input) in enumerate(zip(modules, sample_inputs)):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cat(self):
        modules = [Cat2(), Cat3(), Cat4()]  # noqa: F405
        sample_input = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cdist(self):
        module = CDist()  # noqa: F405
        sample_input = (
            torch.randn(1, 125, 256),
            torch.randn(1, 2048, 256),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_chunk_single(self):
        module = Chunk()  # noqa: F405
        sample_input = (torch.randn(1, 1, 4, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        modules = [Clamp(), ClampMin(1e-10), ClampMax(1e10)]  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d(self):
        modules = [Conv1dSequential(), Conv1dSequential(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d(self):
        modules = [Conv2dSequential(), Conv2dSequential(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_channel_last(self):
        modules = [
            Conv2dSequential(channel_last=True),  # noqa: F405
            Conv2dSequential(bias=False, channel_last=True),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv_transpose1d(self):
        modules = [
            ConvTranspose1dSingle(),  # noqa: F405
            ConvTranspose1dSingle(bias=False),  # noqa: F405
            ConvTranspose1dSingle(dilation=2),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 33]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv_transpose2d(self):
        modules = [
            ConvTranspose2dSingle(),  # noqa: F405
            ConvTranspose2dSingle(bias=False),  # noqa: F405
            ConvTranspose2dSingle(dilation=2),  # noqa: F405
            ConvTranspose2dSingle(dilation=(2, 3)),  # noqa: F405
            ConvTranspose2dSingle(dilation=(2, 1)),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 33, 33]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cos(self):
        module = Cos()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cumsum(self):
        sample_input = ()
        test_comb = [
            {
                QCOM_MODULE: [CumSum()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(4),),
                    (torch.randint(0, 10, size=(4,)),),
                ],
            }
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_einsum_outer_product(self):
        module = EinsumOuterProduct()  # noqa: F405
        x = torch.randn(5)
        y = torch.randn(4)
        sample_input = (
            x,
            y,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_einsum_bilinear(self):
        module = EinsumBilinear()  # noqa: F405
        bn = torch.randn(2, 5)
        anm = torch.randn(3, 5, 4)
        bm = torch.randn(2, 4)
        sample_input = (
            bn,
            anm,
            bm,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_add(self):
        test_comb = [
            {
                QCOM_MODULE: [Add()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [AddConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [
                    AddConstantLong(),  # noqa: F405
                ],
                QCOM_SAMPLE_INPUTS: [(torch.randint(0, 10, size=(2, 3)),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_and(self):
        module = And(torch.tensor(1.7), torch.tensor(0.2))  # noqa: F405
        sample_input = (
            torch.tensor([1, 0, 1, 0], dtype=torch.bool),
            torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_ceil(self):
        torch.manual_seed(8)
        module = Ceil()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_div(self):
        eps = 1e-03
        torch.manual_seed(8)
        test_comb = [
            {
                QCOM_MODULE: [Div()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [DivConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_mul(self):
        test_comb = [
            {
                QCOM_MODULE: [Mul()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [MulConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [MulScalar()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_or(self):
        test_comb = [
            {
                QCOM_MODULE: OrBitWise(  # noqa: F405
                    torch.tensor(1.7), torch.tensor(0.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                    torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                ),
            },
            {
                QCOM_MODULE: OrOperator(  # noqa: F405
                    torch.tensor(1.5), torch.tensor(-1.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.full((3, 3), 1).triu(),
                    torch.full((3, 3), 1).tril(diagonal=0),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_element_wise_sqrt(self):
        modules = [Sqrt(), SqrtConstant()]  # noqa: F405
        for i, module in enumerate(modules):
            sample_input = (torch.rand([3, 1]),)
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_sub(self):
        test_comb = [
            {
                QCOM_MODULE: [Sub()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [SubConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    @unittest.expectedFailure
    def test_qnn_backend_elu(self):
        module = Elu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_embedding(self):
        module = Embedding()  # noqa: F405
        sample_input = (torch.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to(torch.int32),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_equal(self):
        test_comb = [
            {
                QCOM_MODULE: Equal(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: EqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_expand(self):
        modules = [ExpandAs(), ExpandCopy()]  # noqa: F405
        sample_inputs = [
            (torch.randn([3, 1]),),
            (torch.randn([4]),),
        ]
        passes_job = get_capture_program_passes()
        passes_job[ExpandBroadcastTensorShape][QCOM_PASS_ACTIVATE_KEY] = True
        index = 0
        for module in modules:
            for sample_input in sample_inputs:
                with self.subTest(i=index):
                    self.lower_module_and_test_output(
                        module, sample_input, passes_job=passes_job
                    )
                    index += 1

    def test_qnn_backend_expm1(self):
        sample_input = (torch.randn(3, 4, 5),)
        module = ExpM1()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_flip(self):
        sample_input = (torch.randn(3, 4, 5, 6),)
        module = Flip()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_floor(self):
        sample_input = (torch.randn(3, 4),)
        module = Floor()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_floor_divide(self):
        eps = 1e-03
        test_comb = [
            {
                QCOM_MODULE: [FloorDiv()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [FloorDivConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_fold(self):
        sample_input = (torch.randn(3, 512, 256),)
        module = Fold()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_full(self):
        shape = (1, 2, 3, 4)
        module = Full(0.5, shape)  # noqa: F405
        sample_input = (torch.randn(shape),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_full_like(self):
        module = FullLike(0.5)  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gather(self):
        modules = [
            Gather(),  # noqa: F405
            # TODO: resolve accuracy problem
            # GatherArgmin(),  # noqa: F405
            # TODO: There is a accuracy regression after 2.37
            # GatherWhere(),  # noqa: F405
        ]
        # shape = (2, 2, 3, 4)
        sample_inputs = [
            (
                torch.arange(128, dtype=torch.float32).view(64, 2),
                torch.ones(64, 2, dtype=torch.int64),
            ),
            # (torch.arange(128, dtype=torch.float32).view(64, 2),),
            # (torch.randn(shape), torch.randn(shape)),
        ]
        for i, (module, sample_input) in enumerate(zip(modules, sample_inputs)):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gelu(self):
        module = Gelu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_greater_equal(self):
        test_comb = [
            {
                QCOM_MODULE: GreaterEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: GreaterEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_greater_than(self):
        test_comb = [
            {
                QCOM_MODULE: GreaterThan(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: GreaterThanConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_group_norm(self):
        modules = [GroupNorm(), GroupNorm(bias=False)]  # noqa: F405
        sample_input = (torch.randn(3, 32, 56, 56),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardsigmoid(self):
        module = HardSigmoid()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardswish(self):
        module = HardSwish()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardtanh(self):
        module = HardTanh()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index(self):
        modules = [Index(0), Index(1), Index(2)]  # noqa: F405
        sample_input = (torch.randn([8, 172, 64]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index_copy(self):
        test_comb = [
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=1, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1024, 1, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2, 5], dtype=torch.int64),
                    torch.randn([1, 1024, 2, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=1, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1024, 1, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2, 5], dtype=torch.int64),
                    torch.randn([1, 1024, 2, 64]),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE],
                    test[QCOM_SAMPLE_INPUTS],
                    skip_mutable_buffer=test[QCOM_MODULE].skip_mutable_buffer,
                )

    def test_qnn_backend_index_put(self):
        test_comb = [
            {
                QCOM_MODULE: IndexPut(skip_mutable_buffer=False),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int32),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexPut(skip_mutable_buffer=True),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int32),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE],
                    test[QCOM_SAMPLE_INPUTS],
                    skip_mutable_buffer=test[QCOM_MODULE].skip_mutable_buffer,
                )

    def test_qnn_backend_index_select(self):
        module = IndexSelect(dim=1)  # noqa: F405
        sample_input = (
            torch.randn(2, 3, 4, 5),
            torch.tensor([0, 2]),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_instance_norm_2d(self):
        modules = [InstanceNorm2d(32), InstanceNorm2d(32, affine=False)]  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_bicubic(self):
        modules = [
            ResizeBicubic([2, 2], None, False),  # noqa: F405
            ResizeBicubic(None, [2, 2], False),  # noqa: F405
            ResizeBicubic([2, 2], None, True),  # noqa: F405
            ResizeBicubic(None, [2, 2], True),  # noqa: F405
        ]
        sample_input = (torch.randn(1, 4, 2, 2),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_bilinear_2d(self):
        module = ResizeBilinear2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_nearest_2d(self):
        module = ResizeNearest2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_up_sampling_nearest_2d_with_scale_factor(self):
        module = UpsampleNearest2D(scale_factor=2)  # noqa: F405
        sample_input = (torch.randn(1, 16, 72, 104),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_up_sampling_nearest_2d_with_size(self):
        module = UpsampleNearest2D(sizes=(144, 208))  # noqa: F405
        sample_input = (torch.randn(1, 16, 72, 104),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_layer_norm(self):
        modules = [LayerNorm(), LayerNorm(bias=False)]  # noqa: F405
        sample_input = (torch.randn(196, 768),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_leaky_relu(self):
        torch.manual_seed(8)
        test_comb = [
            {
                QCOM_MODULE: [LeakyReLUDefault()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [LeakyReLUCustom(0.05, inplace=False)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [LeakyReLUCustom(0.05, inplace=True)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_less_equal(self):
        test_comb = [
            {
                QCOM_MODULE: LessEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: LessEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_sign(self):
        module = Sign()  # noqa: F405
        sample_input = (torch.randn(3, 4),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_less_than(self):
        test_comb = [
            {
                QCOM_MODULE: LessThan(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: LessThanConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_linalg_vector_norm(self):
        modules = [
            LinalgVectorNorm(),  # noqa: F405
            LinalgVectorNorm(ord=3.5),  # noqa: F405
            LinalgVectorNorm(dim=1),  # noqa: F405
            LinalgVectorNorm(dim=1, keepdim=True),  # noqa: F405
        ]
        sample_input = (torch.randn([3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log(self):
        module = Log()  # noqa: F405
        sample_input = (torch.rand([1, 2, 3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_logical_not(self):
        module = LogicalNot()  # noqa: F405
        sample_input = (torch.rand([1, 2, 3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log_softmax(self):
        module = LogSoftmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_maximum(self):
        module = Maximum()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_max_dim(self):
        module = MaxDim()  # noqa: F405
        sample_input = (torch.randn(4, 10),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_max_pool2d(self):
        module = MaxPool2d()  # noqa: F405
        sample_input = (torch.randn(4, 3, 24, 24),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_mean_dim(self):
        modules = [MeanWKeppDim(), MeanWOKeppDim()]  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("failed to lower in QNN 2.26")
    def test_qnn_backend_mha(self):
        module = MultiheadAttention()  # noqa: F405
        sample_input = (torch.randn(1, 197, 96),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_minimum(self):
        module = Minimum()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_min_dim(self):
        module = MinDim()  # noqa: F405
        sample_input = (torch.randn(4, 10),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_neg(self):
        module = Neg()  # noqa: F405
        sample_input = (torch.randn(1, 4, 16, 16),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_not_equal(self):
        test_comb = [
            {
                QCOM_MODULE: NotEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: NotEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_pad(self):
        module = Pad()  # noqa: F405
        sample_input = (torch.randn([1, 8, 128]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_shuffle(self):
        module = PixelShuffle(2)  # noqa: F405
        sample_input = (torch.ones([2, 4, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_unshuffle(self):
        module = PixelUnshuffle(2)  # noqa: F405
        sample_input = (torch.ones([2, 2, 6, 6]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pow_tensor_scalar(self):
        module = PowTensorScalar()  # noqa: F405
        sample_input = (torch.rand([2, 4, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_prelu(self):
        test_comb = [
            {
                QCOM_MODULE: [PReLUDefault()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [PReLUPerChannel(5)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_relu(self):
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_relu6(self):
        module = Relu6()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_repeat(self):
        module = Repeat()  # noqa: F405
        sample_input = (torch.randn([2, 2, 2, 2]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_reshape(self):
        module = Reshape()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rms_norm(self):
        module = RmsNorm()  # noqa: F405
        sample_input = (torch.abs(torch.randn([1, 1, 1, 4])),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_roll(self):
        modules = [
            Roll(shifts=(3, 3), dims=(1, 2)),  # noqa: F405
            Roll(shifts=(70, 59), dims=(1, 2)),  # noqa: F405
            Roll(shifts=3),  # noqa: F405
            Roll(shifts=56 * 56 * 96 + 3),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 56, 56, 96]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_round(self):
        module = Round()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rsqrt(self):
        module = Rsqrt()  # noqa: F405
        sample_input = (torch.abs(torch.randn([3, 4])),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sdpa(self):
        modules = [
            ScaledDotProductAttention(),  # noqa: F405
            ScaledDotProductAttention(scale=0.5),  # noqa: F405
            ScaledDotProductAttention(scale=1.0),  # noqa: F405
        ]
        mask = torch.tril(torch.randn(1, 1, 100, 100))
        mask[mask == 0] = float("-inf")
        sample_input = (
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            mask,
        )
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sigmoid(self):
        module = Sigmoid()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sin(self):
        module = Sin()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_select_copy(self):
        module = SelectCopy()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_slice_copy(self):
        modules = [
            SliceCopyDefaultParameter(),  # noqa: F405
            SliceCopy(),  # noqa: F405
            SliceCopyWithStep(),  # noqa: F405
        ]
        sample_inputs = [
            (torch.randn([2, 1, 320, 512]),),
            (torch.randn([1, 512]), torch.randn([1, 8])),
            (torch.randn([1, 512]), torch.randn([1, 8])),
        ]
        for module, sample_input in zip(modules, sample_inputs):
            self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_slice_scatter(self):
        test_comb = [
            {
                QCOM_MODULE: [
                    SliceScatter(dim=0, start=3, end=5, step=1)  # noqa: F405
                ],
                QCOM_SAMPLE_INPUTS: [
                    (
                        torch.zeros(8, 8),
                        torch.ones(2, 8),
                    )
                ],
            },
            {
                QCOM_MODULE: [
                    SliceScatter(dim=1, start=2, end=6, step=2)  # noqa: F405
                ],
                QCOM_SAMPLE_INPUTS: [
                    (
                        (
                            torch.zeros(8, 8),
                            torch.ones(8, 2),
                        )
                    )
                ],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_stack(self):
        module = Stack()  # noqa: F405
        sample_input = (
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_softmax(self):
        modules = [Softmax(dim=1), Softmax(dim=-1)]  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_squared_relu(self):
        module = SquaredReLU()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_squeeze(self):
        module = Squeeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sum_int_list(self):
        module = SumIntList()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_tanh(self):
        module = Tanh()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unbind(self):
        module = Unbind()  # noqa: F405
        sample_input = (torch.randn([3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unfold(self):
        sample_input = (torch.randn(2, 128, 32, 32),)
        module = Unfold()  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unsqueeze(self):
        module = Unsqueeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view(self):
        module = View()  # noqa: F405
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_where(self):
        modules = [
            Where(),  # noqa: F405
            WhereConstant(torch.randn(3, 2), torch.randn(3, 2)),  # noqa: F405
            WhereConstantOther(),  # noqa: F405
            # TODO: There is a accuracy regression after 2.37
            # WhereConstantAll(),  # noqa: F405
            WhereConstantInf(),  # noqa: F405
        ]
        sample_inputs = [
            (torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)),
            (torch.randn(3, 2),),
            (torch.randn(3, 2),),
            # (torch.randn(3, 2),),
            (torch.randn(30, 20),),
        ]
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_inputs[i])

    def test_qnn_backend_element_wise_xor(self):
        test_comb = [
            {
                QCOM_MODULE: XorBitWise(  # noqa: F405
                    torch.tensor(1.7), torch.tensor(0.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                    torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                ),
            },
            {
                QCOM_MODULE: XorOperator(  # noqa: F405
                    torch.tensor(1.5), torch.tensor(-1.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.full((3, 3), 1).triu(),
                    torch.full((3, 3), 1).tril(diagonal=0),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                self.lower_module_and_test_output(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )

    def test_qnn_backend_masked_fill(self):
        module = MaskedFill()  # noqa: F405
        attn_mask = torch.ones((64, 49, 49), dtype=torch.float32)

        # Add some zero blocks to simulate the masking behavior
        for i in range(64):
            if i % 2 == 0:
                attn_mask[i, 35:, 35:] = 0

        sample_input = (attn_mask,)  # noqa: F405
        self.lower_module_and_test_output(module, sample_input)


class TestQNNFloatingPointModel(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_argmin_view_squeeze_conv2d(self):
        module = ArgminViewSqueezeConv2D()  # noqa: F405
        sample_input = (torch.randn(32), torch.randn(32, 3, 32, 32))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_causal_mask(self):
        module = CausalMask()  # noqa: F405
        sample_input = (torch.rand((1, 1, 1, 128)) < 0.5,)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_chunk_add(self):
        module = ChunkAdd()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn(1, 2, 4, 2),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d_relu_log_softmax(self):
        modules = [
            Conv1dReluLogSoftmax(dim=1),  # noqa: F405
            Conv1dReluLogSoftmax(dim=-1),  # noqa: F405
        ]
        sample_input = (torch.rand(1, 2, 28),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_argmin(self):
        module = Conv2dArgmin()  # noqa: F405
        sample_input = (torch.randn(16, 3, 4, 4),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_avg_pool2d(self):
        module = Conv2dAvgPool2d()  # noqa: F405
        sample_input = (torch.randn(16, 3, 16, 16),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_bn_hardtanh_mean(self):
        module = Conv2dBnHardtanhMean()  # noqa: F405
        sample_input = (torch.randn(1, 1, 6, 6),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_cat(self):
        module = Conv2dCat()  # noqa: F405
        sample_input = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_down_up_sample(self):
        module = Conv2dDownUpSample()  # noqa: F405
        sample_input = (torch.randn(1, 16, 224, 224),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_flip(self):
        module = Conv2dFlip()  # noqa: F405
        sample_input = (torch.randn(1, 16, 224, 224),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_max_pool2d(self):
        module = Conv2dMaxPool2d()  # noqa: F405
        sample_input = (torch.rand(1, 2, 14, 14),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_slice_copy(self):
        module = Conv2dSliceCopy()  # noqa: F405
        sample_input = (torch.randn([2, 1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_sum_reduce_dim(self):
        module = Conv2dSumReduceDim()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_topk(self):
        module = Conv2dTopK()  # noqa: F405
        sample_input = (torch.randn(1, 3, 32, 32),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_einsum_outer_product_relu(self):
        module = EinsumOuterProductRelu()  # noqa: F405
        x = torch.randn(5)
        y = torch.randn(4)
        sample_input = (
            x,
            y,
        )
        self.lower_module_and_test_output(module, sample_input)

    # TODO: Create a new UT class for passes specific checks
    def test_qnn_backend_lift_add_tensor(self):
        module = LiftAddTensor()  # noqa: F405
        sample_input = (torch.Tensor([1, 2, 3, 4]).to(torch.int32),)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("Fail because of bad accuracy")
    def test_qnn_backend_moe_feed_forward(self):
        from executorch.examples.models.llama.llama_transformer import MOEFeedForward
        from executorch.examples.models.llama.model_args import ModelArgs

        args = ModelArgs()
        args.dim = 32
        args.n_heads = 8
        args.n_layers = 2
        self.head_dim = args.dim // args.n_heads
        module = MOEFeedForward(args)  # noqa: F405
        sample_input = (
            torch.randint(low=0, high=100, size=(1, 32), dtype=torch.float32),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_unshuffle_math_equivalent(self):
        module = PixelUnshuffleMathEquivalent(2)  # noqa: F405
        sample_input = (torch.rand(2, 2, 6, 6),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_residual_block(self):
        module = ResidualBlockModule()  # noqa: F405
        sample_input = (torch.randn(1, 32, 28, 28),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_simple_model(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_topk_and_index(self):
        module = TopKandIndex()  # noqa: F405
        sample_input = (torch.randn(3, 10),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_zero_dim_tensor(self):
        module = ZeroDimTensor()  # noqa: F405
        sample_input = (torch.randn(1, 256, 125),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_example_models(self):
        # TODO Fix MobileBertModelExample and TorchVisionViTModel
        instances = [
            DeepLabV3ResNet101Model(),
            EdsrModel(),
            InceptionV3Model(),
            InceptionV4Model(),
            # The module of llama is changing frequently. Reopen it when it's stable
            MV2Model(),
            MV3Model(),
            # Fail during lowering Reopen once resolved
            # MobileBertModelExample(),
            # TorchVisionViTModel(),
            Wav2LetterModel(),
        ]
        expected_partitions = [
            1,
            1,
            1,
            1,
            1,
            1,
            # 1,
            # 1,
            1,
        ]
        # TODO: Due to trigger maximum recursion depth exceeded, need to check it.
        disable_validation()
        for i, instance in enumerate(instances):
            with self.subTest(i=i):
                module = instance.get_eager_model().eval()
                sample_input = instance.get_example_inputs()
                self.lower_module_and_test_output(
                    module,
                    sample_input,
                    expected_partitions=expected_partitions[i],
                    assert_output_equal=False,
                )


class TestQNNQuantizedOperator(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_16a4w_conv2d(self):
        modules = [Conv2dSingle(), Conv2dSingle(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    module, sample_input, quant_dtype=QuantDtype.use_16a4w
                )
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_conv2d_qat(self):
        modules = [Conv2dSingle(), Conv2dSingle(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                prepared = self.get_prepared_qat_module(module, sample_input)
                converted = self.get_converted_sgd_trained_module(
                    module, prepared, sample_input
                )
                self.lower_module_and_test_output(converted, sample_input)

    def test_qnn_backend_16a4w_layer_norm(self):
        module = LayerNorm()  # noqa: F405
        sample_input = (torch.randn(196, 768),)
        module = self.get_qdq_module(
            module,
            sample_input,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_per_channel_linear(self):
        module = Linear(use_bias=False)  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            is_linear_per_channel=True,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_per_channel_linear_with_bias(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            is_linear_per_channel=True,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_abs(self):
        module = Abs()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_adaptive_avg_pool1d(self):
        module = AdaptiveAvgPool1D()  # noqa: F405
        sample_input = (torch.randn(1, 512, 7),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_adaptive_avg_pool2d(self):
        module = AdaptiveAvgPool2D()  # noqa: F405
        sample_input = (torch.randn(1, 512, 7, 7),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_alias(self):
        module = Alias()  # noqa: F405
        sample_input = (torch.randn(1, 10),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amax(self):
        modules = [
            AMax(dim=1, keepdim=False),  # noqa: F405
            AMax(dim=1, keepdim=True),  # noqa: F405
            AMax(),  # noqa: F405
        ]
        sample_input = (torch.randn(4, 4),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amax_conv(self):
        sample_input = (torch.randn(2, 3, 64, 64),)  # [batch, channels, height, width]
        module = AMaxFollowingConv2D(  # noqa: F405
            in_channels=3, out_channels=16, kernel_size=3, dim=-1, keepdim=False
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_amin(self):
        modules = [
            AMin(dim=1, keepdim=False),  # noqa: F405
            AMin(dim=1, keepdim=True),  # noqa: F405
            AMin(),  # noqa: F405
        ]
        sample_input = (torch.randn(4, 4),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_any(self):
        modules = [Any(), Any(dim=[0, 1]), Any(dim=1, keepdim=True)]  # noqa: F405
        sample_input = (torch.randn(3, 3, 3) > 0,)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input, bypass_check=True)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_arange(self):
        modules = [
            Arange(start=1, end=6, step=0.5, dtype=torch.float32),  # noqa: F405
        ]
        sample_input = (torch.randn(10),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_argmax(self):
        module = Argmax()  # noqa: F405
        sample_input = (torch.randn(16, 3, 4, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_argmin(self):
        module = Argmin()  # noqa: F405
        sample_input = (torch.randn(16, 3, 4, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_asin(self):
        module = Asin()  # noqa: F405
        sample_input = (torch.rand([3, 4]) * 2 - 1,)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_atan(self):
        sample_input = (torch.randn(3, 4),)
        module = Atan()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_avg_pool2d(self):
        modules = [
            AvgPoolModule((2, 2), (1, 1), (1, 1), False),  # noqa: F405
            AvgPoolModule((1280, 1280), (1280, 1280), (0, 0), True),  # noqa: F405
            AvgPoolModule((1280, 1280), (1280, 1280), (320, 320), True),  # noqa: F405
        ]  # noqa: F405
        sample_inputs = [
            (torch.randn(1, 3, 2, 2),),
            (torch.randn(1, 1280, 7, 7),),
            (torch.randn(1, 1280, 7, 7),),
        ]
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_inputs[i])
                self.lower_module_and_test_output(module, sample_inputs[i])

    def test_qnn_backend_batch_norm(self):
        modules = [BatchNorm(32), BatchNorm(32, False)]  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([4, 8, 32]), torch.randn([4, 32, 8]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cast(self):
        module = CastMultiUsers()  # noqa: F405
        sample_input = (torch.randint(0, 3, size=(3, 3)), torch.randn(3, 3))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cat(self):
        modules = [Cat2(), Cat3(), Cat4()]  # noqa: F405
        sample_input = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cdist(self):
        module = CDist()  # noqa: F405
        sample_input = (
            torch.randn(1, 125, 256),
            torch.randn(1, 2048, 256),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_chunk_single(self):
        module = Chunk()  # noqa: F405
        sample_input = (torch.randn(1, 1, 4, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        modules = [Clamp(), ClampMin(1e-10), ClampMax(1e10)]  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d(self):
        modules = [Conv1dSequential(), Conv1dSequential(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d(self):
        modules = [Conv2dSequential(), Conv2dSequential(bias=False)]  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_block(self):
        o_ch, i_ch, kernel, padding = 32, 512, (1, 1), 0

        modules = [
            Conv2dSingle(  # noqa: F405
                bias=False,
                in_channel=i_ch,
                out_channel=o_ch,
                kernel_size=kernel,
                padding=padding,
            ),
            Conv2dSingle(  # noqa: F405
                in_channel=i_ch,
                out_channel=o_ch,
                kernel_size=kernel,
                padding=padding,
            ),
        ]

        sample_input = (torch.randn(1, i_ch, 1, o_ch),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                # update block size for convolution weight (OIHW)
                # channel dimension(O) is defaultly sliced in QNN
                # divide dimension(I) into 16 groups
                module = self.get_qdq_module(
                    module,
                    sample_input,
                    quant_dtype=QuantDtype.use_16a4w_block,
                    block_size_map={"conv2d": (1, 32, 1, 1)},
                )
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_channel_last(self):
        modules = [
            Conv2dSequential(channel_last=True),  # noqa: F405
            Conv2dSequential(bias=False, channel_last=True),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv_transpose1d(self):
        modules = [
            ConvTranspose1dSingle(),  # noqa: F405
            ConvTranspose1dSingle(bias=False),  # noqa: F405
            ConvTranspose1dSingle(dilation=2),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv_transpose2d(self):
        modules = [
            ConvTranspose2dSingle(),  # noqa: F405
            ConvTranspose2dSingle(bias=False),  # noqa: F405
            ConvTranspose2dSingle(dilation=(2, 3)),  # noqa: F405
            ConvTranspose2dSingle(dilation=(2, 1)),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 1, 3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cos(self):
        module = Cos()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cumsum(self):
        module = CumSum()  # noqa: F405
        sample_input = (torch.randn(4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_einsum_outer_product(self):
        module = EinsumOuterProduct()  # noqa: F405
        x = torch.randn(5)
        y = torch.randn(4)
        sample_input = (
            x,
            y,
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_einsum_bilinear(self):
        module = EinsumBilinear()  # noqa: F405
        bn = torch.randn(2, 5)
        anm = torch.randn(3, 5, 4)
        bm = torch.randn(2, 4)
        sample_input = (
            bn,
            anm,
            bm,
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_add(self):
        test_comb = [
            {
                QCOM_MODULE: [Add()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [AddConstantFloat(), AddConstantLong()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        gm = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(gm, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_and(self):
        module = And(torch.tensor(1.7), torch.tensor(0.2))  # noqa: F405
        sample_input = (
            torch.tensor([1, 0, 1, 0], dtype=torch.bool),
            torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_ceil(self):
        module = Ceil()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_div(self):
        eps = 1e-03
        torch.manual_seed(8)
        test_comb = [
            {
                QCOM_MODULE: [Div()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [DivConstantFloat(), DivConstantLong()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        gm = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(gm, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_mul(self):
        test_comb = [
            {
                QCOM_MODULE: [Mul()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [MulConstantFloat(), MulConstantLong()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [MulScalar()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        gm = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(gm, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_or(self):
        test_comb = [
            {
                QCOM_MODULE: OrBitWise(  # noqa: F405
                    torch.tensor(1.7), torch.tensor(0.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                    torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                ),
            },
            {
                QCOM_MODULE: OrOperator(  # noqa: F405
                    torch.tensor(1.5), torch.tensor(-1.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.full((3, 3), 1).triu(),
                    torch.full((3, 3), 1).tril(diagonal=0),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_element_wise_sqrt(self):
        modules = [Sqrt(), SqrtConstant()]  # noqa: F405
        for i, module in enumerate(modules):
            sample_input = (torch.rand([3, 1]),)
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_sub(self):
        test_comb = [
            {
                QCOM_MODULE: [Sub()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [SubConstantFloat(), SubConstantLong()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        gm = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(gm, sample_input)
                        index += 1

    def test_qnn_backend_elu(self):
        module = Elu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_embedding(self):
        modules = [Embedding(), Embedding()]  # noqa: F405
        sample_input = (torch.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to(torch.int32),)
        quant_dtype = [QuantDtype.use_8a8w, QuantDtype.use_16a4w]
        for i, qdtype in enumerate(quant_dtype):
            with self.subTest(i=i):
                modules[i] = self.get_qdq_module(
                    modules[i], sample_input, quant_dtype=qdtype
                )
                self.lower_module_and_test_output(modules[i], sample_input)

    def test_qnn_backend_equal(self):
        test_comb = [
            {
                QCOM_MODULE: Equal(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: EqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_expand(self):
        modules = [ExpandAs(), ExpandCopy()]  # noqa: F405
        sample_inputs = [
            (torch.randn([3, 1]),),
            (torch.randn([4]),),
        ]
        passes_job = get_capture_program_passes()
        passes_job[ExpandBroadcastTensorShape][QCOM_PASS_ACTIVATE_KEY] = True
        index = 0
        for module in modules:
            for sample_input in sample_inputs:
                with self.subTest(i=index):
                    module = self.get_qdq_module(module, sample_input)
                    self.lower_module_and_test_output(
                        module, sample_input, passes_job=passes_job
                    )
                    index += 1

    def test_qnn_backend_expm1(self):
        sample_input = (torch.randn(3, 4, 5),)
        module = ExpM1()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_flip(self):
        sample_input = (torch.randn(3, 4, 5, 6),)
        module = Flip()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_floor(self):
        sample_input = (torch.randn(3, 4),)
        module = Floor()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_floor_divide(self):
        eps = 1e-03
        test_comb = [
            {
                QCOM_MODULE: [FloorDiv()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                QCOM_MODULE: [FloorDivConstantFloat()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        gm = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(gm, sample_input)
                        index += 1

    def test_qnn_backend_fold(self):
        sample_input = (torch.randn(3, 512, 256),)
        module = Fold()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_full(self):
        shape = (1, 2, 3, 4)
        module = Full(0.5, shape)  # noqa: F405
        sample_input = (torch.randn(shape),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_full_like(self):
        module = FullLike(0.5)  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gather(self):
        modules = [
            Gather(),  # noqa: F405
            # TODO: resolve accuracy problem
            # GatherArgmin(),  # noqa: F405
            # TODO: There is a accuracy regression after 2.37
            # GatherWhere(),  # noqa: F405
        ]
        # shape = (2, 2, 3, 4)
        sample_inputs = [
            (
                torch.arange(128, dtype=torch.float32).view(64, 2),
                torch.ones(64, 2, dtype=torch.int64),
            ),
            # (torch.arange(128, dtype=torch.float32).view(64, 2),),
            # (torch.randn(shape), torch.randn(shape)),
        ]
        for i, (module, sample_input) in enumerate(zip(modules, sample_inputs)):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gelu(self):
        module = Gelu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_greater_equal(self):
        test_comb = [
            {
                QCOM_MODULE: GreaterEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: GreaterEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_greater_than(self):
        test_comb = [
            {
                QCOM_MODULE: GreaterThan(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: GreaterThanConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_group_norm(self):
        modules = [GroupNorm(), GroupNorm(bias=False)]  # noqa: F405
        sample_input = (torch.randn(3, 32, 56, 56),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardsigmoid(self):
        module = HardSigmoid()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardswish(self):
        module = HardSwish()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_hardtanh(self):
        module = HardTanh()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index(self):
        modules = [Index(0), Index(1), Index(2)]  # noqa: F405
        sample_input = (torch.randn([8, 172, 64]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index_copy(self):
        test_comb = [
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=1, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1024, 1, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=False
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2, 5], dtype=torch.int64),
                    torch.randn([1, 1024, 2, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=1, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1024, 1, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexCopy(  # noqa: F405
                    copy_dim=2, skip_mutable_buffer=True
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2, 5], dtype=torch.int64),
                    torch.randn([1, 1024, 2, 64]),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(
                    module,
                    test[QCOM_SAMPLE_INPUTS],
                    skip_mutable_buffer=test[QCOM_MODULE].skip_mutable_buffer,
                )

    def test_qnn_backend_index_put(self):
        test_comb = [
            {
                QCOM_MODULE: IndexPut(skip_mutable_buffer=False),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int32),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
            {
                QCOM_MODULE: IndexPut(skip_mutable_buffer=True),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([2], dtype=torch.int32),
                    torch.randn([1, 1, 12, 64]),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(
                    module,
                    test[QCOM_SAMPLE_INPUTS],
                    skip_mutable_buffer=test[QCOM_MODULE].skip_mutable_buffer,
                )

    def test_qnn_backend_index_select(self):
        module = IndexSelect(dim=1)  # noqa: F405
        sample_input = (
            torch.randn(2, 3, 4, 5),
            torch.tensor([0, 2]),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_instance_norm_2d(self):
        modules = [InstanceNorm2d(32), InstanceNorm2d(32, affine=False)]  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_bicubic(self):
        modules = [
            ResizeBicubic([2, 2], None, False),  # noqa: F405
            ResizeBicubic(None, [2, 2], False),  # noqa: F405
            ResizeBicubic([2, 2], None, True),  # noqa: F405
            ResizeBicubic(None, [2, 2], True),  # noqa: F405
        ]
        sample_input = (torch.randn(1, 4, 2, 2),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_bilinear_2d(self):
        module = ResizeBilinear2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_nearest_2d(self):
        module = ResizeNearest2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_up_sampling_nearest_2d_with_scale_factor(self):
        module = UpsampleNearest2D(scale_factor=2)  # noqa: F405
        sample_input = (torch.randn(1, 16, 72, 104),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_up_sampling_nearest_2d_with_size(self):
        module = UpsampleNearest2D(sizes=(144, 208))  # noqa: F405
        sample_input = (torch.randn(1, 16, 72, 104),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_layer_norm(self):
        modules = [LayerNorm(), LayerNorm(bias=False)]  # noqa: F405
        sample_input = (torch.randn(196, 768),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_leaky_relu(self):
        test_comb = [
            {
                QCOM_MODULE: [LeakyReLUDefault()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [LeakyReLUCustom(0.05)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [LeakyReLUCustom(0.05, inplace=True)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_less_equal(self):
        test_comb = [
            {
                QCOM_MODULE: LessEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: LessEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_less_than(self):
        test_comb = [
            {
                QCOM_MODULE: LessThan(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: LessThanConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_linalg_vector_norm(self):
        modules = [
            LinalgVectorNorm(),  # noqa: F405
            LinalgVectorNorm(ord=3.5),  # noqa: F405
            LinalgVectorNorm(dim=1),  # noqa: F405
            LinalgVectorNorm(dim=1, keepdim=True),  # noqa: F405
        ]
        sample_input = (torch.randn([3, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skipIf(is_qnn_sdk_version_less_than("2.30"), "UT pass after QNN 2.30")
    def test_qnn_backend_linear_block(self):
        modules = [
            Linear(use_bias=False),  # noqa: F405
            Linear(use_bias=True),  # noqa: F405
        ]

        sample_input = (torch.randn([3, 512]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                # update block size for linear weight (OI)
                # channel dimension(O) is defaultly sliced in QNN
                # divide dimension(I) into 16 groups
                module = self.get_qdq_module(
                    module,
                    sample_input,
                    quant_dtype=QuantDtype.use_16a4w_block,
                    block_size_map={"linear": (1, 32)},
                )
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_linear_qat(self):
        """
        Prototype to test qat model
        """
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 512]),)
        prepared = self.get_prepared_qat_module(module, sample_input)
        module = self.get_converted_sgd_trained_module(module, prepared, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log(self):
        module = Log()  # noqa: F405
        sample_input = (torch.rand([1, 2, 3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_logical_not(self):
        module = LogicalNot()  # noqa: F405
        sample_input = (torch.rand([1, 2, 3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log_softmax(self):
        module = LogSoftmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_maximum(self):
        module = Maximum()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_max_dim(self):
        module = MaxDim()  # noqa: F405
        sample_input = (torch.randn(4, 10),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_max_pool2d(self):
        module = MaxPool2d()  # noqa: F405
        sample_input = (torch.randn(4, 3, 24, 24),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_mean_dim(self):
        modules = [MeanWKeppDim(), MeanWOKeppDim()]  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_mha(self):
        module = MultiheadAttention()  # noqa: F405
        sample_input = (torch.randn(1, 197, 96),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_minimum(self):
        module = Minimum()  # noqa: F405
        sample_input = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_min_dim(self):
        module = MinDim()  # noqa: F405
        sample_input = (torch.randn(4, 10),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_neg(self):
        module = Neg()  # noqa: F405
        sample_input = (torch.randn(1, 4, 16, 16),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_not_equal(self):
        test_comb = [
            {
                QCOM_MODULE: NotEqual(),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
            },
            {
                QCOM_MODULE: NotEqualConstant(0.5),  # noqa: F405
                QCOM_SAMPLE_INPUTS: (torch.randn(1, 2, 3, 4),),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])

    def test_qnn_backend_pad(self):
        module = Pad()  # noqa: F405
        sample_input = (torch.randn([1, 8, 128]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_shuffle(self):
        module = PixelShuffle(2)  # noqa: F405
        sample_input = (torch.ones([2, 4, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_unshuffle(self):
        module = PixelUnshuffle(2)  # noqa: F405
        sample_input = (torch.ones([2, 2, 6, 6]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pow_tensor_scalar(self):
        module = PowTensorScalar()  # noqa: F405
        sample_input = (torch.rand([2, 4, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_prelu(self):
        test_comb = [
            {
                QCOM_MODULE: [PReLUDefault()],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
            {
                QCOM_MODULE: [PReLUPerChannel(5)],  # noqa: F405
                QCOM_SAMPLE_INPUTS: [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_relu(self):
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_relu6(self):
        module = Relu6()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_repeat(self):
        module = Repeat()  # noqa: F405
        sample_input = (torch.randn([2, 2, 2, 2]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_reshape(self):
        module = Reshape()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rms_norm(self):
        module = RmsNorm()  # noqa: F405
        sample_input = (torch.abs(torch.randn([1, 1, 1, 4])),)
        module = self.get_qdq_module(
            module, sample_input, quant_dtype=QuantDtype.use_16a4w
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_roll(self):
        modules = [
            Roll(shifts=(3, 3), dims=(1, 2)),  # noqa: F405
            Roll(shifts=(70, 59), dims=(1, 2)),  # noqa: F405
            Roll(shifts=3),  # noqa: F405
            Roll(shifts=56 * 56 * 96 + 3),  # noqa: F405
        ]
        sample_input = (torch.randn([1, 56, 56, 96]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_round(self):
        module = Round()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rsqrt(self):
        module = Rsqrt()  # noqa: F405
        sample_input = (torch.abs(torch.randn([3, 4])),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sdpa(self):
        modules = [
            ScaledDotProductAttention(),  # noqa: F405
            ScaledDotProductAttention(scale=0.5),  # noqa: F405
            ScaledDotProductAttention(scale=1.0),  # noqa: F405
        ]
        mask = torch.tril(torch.randn(1, 1, 100, 100))
        mask[mask == 0] = torch.finfo(torch.float32).min
        sample_input = (
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            mask,
        )
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    module, sample_input, quant_dtype=QuantDtype.use_16a8w
                )
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_select_copy(self):
        module = SelectCopy()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sigmoid(self):
        module = Sigmoid()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sign(self):
        module = Sign()  # noqa: F405
        sample_input = (torch.randn(3, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sin(self):
        module = Sin()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_slice_copy(self):
        modules = [
            SliceCopyDefaultParameter(),  # noqa: F405
            SliceCopy(),  # noqa: F405
            SliceCopyWithStep(),  # noqa: F405
        ]
        sample_inputs = [
            (torch.randn([2, 1, 320, 512]),),
            (torch.randn([1, 512]), torch.randn([1, 8])),
            (torch.randn([1, 512]), torch.randn([1, 8])),
        ]
        for module, sample_input in zip(modules, sample_inputs):
            module = self.get_qdq_module(module, sample_input)
            self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_slice_scatter(self):
        test_comb = [
            {
                QCOM_MODULE: [
                    SliceScatter(dim=0, start=3, end=5, step=1)  # noqa: F405
                ],
                QCOM_SAMPLE_INPUTS: [
                    (
                        torch.zeros(8, 8),
                        torch.ones(2, 8),
                    )
                ],
            },
            {
                QCOM_MODULE: [
                    SliceScatter(dim=1, start=2, end=6, step=2)  # noqa: F405
                ],
                QCOM_SAMPLE_INPUTS: [
                    (
                        (
                            torch.zeros(8, 8),
                            torch.ones(8, 2),
                        )
                    )
                ],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_softmax(self):
        modules = [Softmax(dim=1), Softmax(dim=-1)]  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_squared_relu(self):
        module = SquaredReLU()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_squeeze(self):
        module = Squeeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_stack(self):
        module = Stack()  # noqa: F405
        sample_input = (
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sum_int_list(self):
        module = SumIntList()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_tanh(self):
        module = Tanh()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unbind(self):
        module = Unbind()  # noqa: F405
        sample_input = (torch.randn([3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unfold(self):
        sample_input = (torch.randn(2, 128, 32, 32),)
        module = Unfold()  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unsqueeze(self):
        module = Unsqueeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view(self):
        module = View()  # noqa: F405
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_where(self):
        modules = [
            Where(),  # noqa: F405
            WhereConstant(torch.randn(3, 2), torch.randn(3, 2)),  # noqa: F405
            WhereConstantOther(),  # noqa: F405
            # TODO: There is a accuracy regression after 2.37
            # WhereConstantAll(),  # noqa: F405
            WhereConstantInf(),  # noqa: F405
        ]
        sample_inputs = [
            (torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)),
            (torch.randn(3, 2),),
            (torch.randn(3, 2),),
            # (torch.randn(3, 2),),
            (torch.randn(30, 20),),
        ]
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_inputs[i])
                self.lower_module_and_test_output(module, sample_inputs[i])

    def test_qnn_backend_masked_fill(self):
        module = MaskedFill()  # noqa: F405
        attn_mask = torch.ones((64, 49, 49), dtype=torch.float32)

        # Add some zero blocks to simulate the masking behavior
        for i in range(64):
            if i % 2 == 0:
                attn_mask[i, 35:, 35:] = 0

        sample_input = (attn_mask,)  # noqa: F405
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_xor(self):
        test_comb = [
            {
                QCOM_MODULE: XorBitWise(  # noqa: F405
                    torch.tensor(1.7), torch.tensor(0.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                    torch.tensor([1, 1, 0, 0], dtype=torch.bool),
                ),
            },
            {
                QCOM_MODULE: XorOperator(  # noqa: F405
                    torch.tensor(1.5), torch.tensor(-1.2)
                ),
                QCOM_SAMPLE_INPUTS: (
                    torch.full((3, 3), 1).triu(),
                    torch.full((3, 3), 1).tril(diagonal=0),
                ),
            },
        ]
        for i, test in enumerate(test_comb):
            with self.subTest(i=i):
                module = self.get_qdq_module(
                    test[QCOM_MODULE], test[QCOM_SAMPLE_INPUTS]
                )
                self.lower_module_and_test_output(module, test[QCOM_SAMPLE_INPUTS])


class TestQNNQuantizedModel(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_argmin_view_squeeze_conv2d(self):
        module = ArgminViewSqueezeConv2D()  # noqa: F405
        sample_input = (torch.randn(32), torch.randn(32, 3, 32, 32))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_causal_mask(self):
        module = CausalMask()  # noqa: F405
        sample_input = (torch.rand((1, 1, 1, 128)) < 0.5,)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_chunk_add(self):
        module = ChunkAdd()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn(1, 1, 4, 2),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d_relu_log_softmax(self):
        modules = [
            Conv1dReluLogSoftmax(dim=1),  # noqa: F405
            Conv1dReluLogSoftmax(dim=-1),  # noqa: F405
        ]
        sample_input = (torch.rand(1, 2, 28),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_argmin(self):
        module = Conv2dArgmin()  # noqa: F405
        sample_input = (torch.randn(16, 3, 4, 4),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_avg_pool2d(self):
        module = Conv2dAvgPool2d()  # noqa: F405
        sample_input = (torch.randn(16, 3, 16, 16),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_bn_hardtanh_mean(self):
        module = Conv2dBnHardtanhMean()  # noqa: F405
        sample_input = (torch.randn(1, 1, 6, 6),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_cat(self):
        module = Conv2dCat()  # noqa: F405
        sample_input = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_down_up_sample(self):
        module = Conv2dDownUpSample()  # noqa: F405
        sample_input = (torch.randn(1, 16, 224, 224),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_flip(self):
        module = Conv2dFlip()  # noqa: F405
        sample_input = (torch.randn(1, 16, 224, 224),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_max_pool2d(self):
        module = Conv2dMaxPool2d()  # noqa: F405
        sample_input = (torch.rand(1, 2, 14, 14),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_slice_copy(self):
        module = Conv2dSliceCopy()  # noqa: F405
        sample_input = (torch.randn([2, 1, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_sum_reduce_dim(self):
        module = Conv2dSumReduceDim()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d_topk(self):
        module = Conv2dTopK()  # noqa: F405
        sample_input = (torch.randn(1, 3, 32, 32),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_einsum_outer_product_relu(self):
        module = EinsumOuterProductRelu()  # noqa: F405
        x = torch.randn(5)
        y = torch.randn(4)
        sample_input = (
            x,
            y,
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skipIf(is_qnn_sdk_version_less_than("2.35"), "UT pass after QNN 2.35")
    def test_qnn_backend_masked_softmax(self):
        if self.enable_x86_64:
            self.skipTest(
                "At the moment, testing is only being conducted on the device."
            )
        module = MaskedSoftmax()  # noqa: F405
        kv_arange = torch.arange(128)
        reshaped_cache_position = torch.tensor([[0]])

        # Simplest and most efficient way to obtain a causal mask
        causal_mask = kv_arange <= reshaped_cache_position
        atten_mask = torch.full((1, 128), torch.tensor(-65535.0))
        atten_mask = atten_mask.masked_fill(causal_mask, 0)
        atten_mask = atten_mask[None, None, :, :].expand(1, -1, -1, -1)
        sample_input = (atten_mask, torch.randn([1, 1, 1, 128]))
        # Masked softmax is only support in quantized model
        module = self.get_qdq_module(
            module, sample_input, quant_dtype=QuantDtype.use_16a8w
        )
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            optrace=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module, sample_input, compiler_spec
            ).to_executorch()
            pte_path = f"{tmp_dir}/model.pte"
            with open(pte_path, "wb") as f:
                edge_prog_mgr.write_to_file(f)
            adb = self.get_adb_tool(pte_path)
            binaries_trace = generate_optrace(
                tmp_dir, self.chipset_table[self.model], adb, pte_path, [sample_input]
            )
            has_masked_softmax = False
            for _, (_, qhas) in binaries_trace.items():
                with open(qhas, "r") as qhas_file:
                    qhas_data = json.load(qhas_file)
                    for row in qhas_data["data"]["htp_op_types"]["data"]:
                        if "MaskedSoftmax" in row["op"]:
                            has_masked_softmax = True
            self.assertTrue(has_masked_softmax)

    @unittest.skip("UT pass before QNN 2.26, segfault during partitioner")
    def test_qnn_backend_moe_feed_forward(self):
        from executorch.examples.models.llama.llama_transformer import MOEFeedForward
        from executorch.examples.models.llama.model_args import ModelArgs

        args = ModelArgs()
        args.dim = 32
        args.n_heads = 8
        args.n_layers = 2
        self.head_dim = args.dim // args.n_heads
        module = MOEFeedForward(args)  # noqa: F405
        sample_input = (
            torch.randint(low=0, high=100, size=(1, 32), dtype=torch.float32),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_unshuffle_math_equivalent(self):
        module = PixelUnshuffleMathEquivalent(2)  # noqa: F405
        sample_input = (torch.rand(2, 2, 6, 6),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_residual_block(self):
        module = ResidualBlockModule()  # noqa: F405
        sample_input = (torch.randn(1, 32, 28, 28),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_simple_model(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_submodules(self):
        module = SimpleSubModules()  # noqa: F405
        sample_input = (
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
        )

        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_submodule_type_predicate,
        )

        submodule_qconfig_list = [
            (
                get_submodule_type_predicate("Add"),
                ModuleQConfig(QuantDtype.use_16a16w),
            )  # noqa: F405
        ]
        module = self.get_qdq_module(
            module,
            sample_input,
            submodule_qconfig_list=submodule_qconfig_list,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_topk_and_index(self):
        module = TopKandIndex()  # noqa: F405
        sample_input = (torch.randn(3, 10),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_zero_dim_tensor(self):
        module = ZeroDimTensor()  # noqa: F405
        sample_input = (torch.randn(1, 256, 125),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_example_models(self):
        instances = [
            {
                QCOM_MODULE: DeepLabV3ResNet101Model(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            {
                QCOM_MODULE: EdsrModel(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            {
                QCOM_MODULE: InceptionV3Model(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            {
                QCOM_MODULE: InceptionV4Model(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            # The module of llama is changing frequently. Reopen it when it's stable
            {
                QCOM_MODULE: MV2Model(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            {
                QCOM_MODULE: MV3Model(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            # only works on QNN 2.12 so far
            # { 'module': MobileBertModelExample(), 'annotation': (), QCOM_QUANT_DTYPE: QuantDtype.use_8a8w },
            {
                QCOM_MODULE: TorchVisionViTModel(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
            {
                QCOM_MODULE: Wav2LetterModel(),
                QCOM_ANNOTATION: (),
                QCOM_QUANT_DTYPE: QuantDtype.use_8a8w,
            },
        ]
        expected_partitions = [
            1,
            1,
            1,
            1,
            1,
            1,
            # For MobileBertModelExample
            # 1,
            1,
            1,
        ]
        # TODO: Due to trigger maximum recursion depth exceeded, need to check it.
        disable_validation()
        for i, instance in enumerate(instances):
            with self.subTest(i=i):
                module = instance[QCOM_MODULE].get_eager_model().eval()
                sample_input = instance[QCOM_MODULE].get_example_inputs()
                module = self.get_qdq_module(
                    module,
                    sample_input,
                    custom_quant_annotations=instance[QCOM_ANNOTATION],
                    quant_dtype=instance[QCOM_QUANT_DTYPE],
                )
                self.lower_module_and_test_output(
                    module,
                    sample_input,
                    expected_partitions=expected_partitions[i],
                    assert_output_equal=False,
                )


class TestQNNFloatingPointUtils(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
        )

    def test_qnn_backend_dump_intermediate_outputs(self):
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            dump_intermediate_outputs=True,
        )
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_intermediate_events=3,
        )

    def test_qnn_backend_skip_node_id(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=3,
            skip_node_id_set={"aten_add_tensor", "aten_mean_dim"},
        )

    def test_qnn_backend_skip_node_op(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=2,
            skip_node_op_set={"aten.add.Tensor"},
        )

    @unittest.expectedFailure
    def test_qnn_backend_spill_fill_buffer_size(self):
        # TODO: Fix self.assertNotEqual(0, max_sf_size)
        module = LargeTensorLinear()  # noqa: F405
        sample_input = (torch.randn(1, 256, 512),)
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        edge_prog = to_edge_transform_and_lower_to_qnn(
            module, sample_input, compiler_specs
        )

        max_sf_size = update_spill_fill_size(edge_prog.exported_program())
        self.assertNotEqual(0, max_sf_size)

    def test_qnn_backend_multi_contexts(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        pass_jobs = get_capture_program_passes()
        split_graph_pass, setting = self.split_graph(4)
        pass_jobs[split_graph_pass] = setting
        dep_table = get_passes_dependency_for_capture_program()
        dep_table[split_graph_pass] = [FoldQDQ]
        edge_prog = to_edge_transform_and_lower_to_qnn(
            module,
            sample_input,
            compiler_specs,
            dep_table=dep_table,
            passes_job=pass_jobs,
        )

        update_spill_fill_size(edge_prog.exported_program())
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_multi_contexts_composite(self):
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = CompositeDelegateModule(  # noqa: F405
            compiler_specs=compiler_specs,
            to_edge_transform_and_lower_method=to_edge_transform_and_lower_to_qnn,
        )
        sample_input = module.get_random_input()
        edge_prog = to_edge(
            torch.export.export(module, sample_input, strict=True),
        )
        update_spill_fill_size(edge_prog.exported_program())
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module.get_reference_module(), sample_input, exec_prog)

    def test_qnn_backend_multi_graphs(self):
        if self.enable_x86_64:
            self.skipTest("weight sharing is not supported on host machine")

        seq_conv = Conv2dSequential()  # noqa: F405
        # weight sharing
        modules = [seq_conv, seq_conv.second]
        sample_inputs = [(torch.randn([1, 1, 3, 3]),), (torch.randn([1, 3, 3, 3]),)]
        graph_names = ["seq_conv", "single_conv"]
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_weight_sharing=True,
        )
        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                graph_name=graph_name,
            )
            for graph_name in graph_names
        ]

        modules_dict = {}
        sample_inputs_dict = {}
        compiler_specs_dict = {}
        for i, graph_name in enumerate(graph_names):
            modules_dict[graph_name] = modules[i]
            sample_inputs_dict[graph_name] = sample_inputs[i]
            compiler_specs_dict[graph_name] = compiler_specs[i]
        delegated_program = to_edge_transform_and_lower_to_qnn(
            modules_dict, sample_inputs_dict, compiler_specs_dict
        )
        executorch_prog = delegated_program.to_executorch()
        for index, module in enumerate(modules):
            self.verify_output(
                module=module,
                sample_inputs=sample_inputs[index],
                executorch_prog=executorch_prog,
                method_index=index,
            )

    def test_qnn_backend_profile_op(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            profile=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_profile_events=30,
        )

    def test_qnn_backend_runtime_option_htp_performance(self):
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))

        def output_callback(log_msg, is_burst):
            msg = log_msg.stdout
            # Refer to HtpDevice.cpp for the following values
            min_voltage = (
                "coreVoltageCornerMin 160" if is_burst else "coreVoltageCornerMin 80"
            )
            self.assertTrue(min_voltage in msg, f"Expecting '{min_voltage} ' in log")

        burst_runtime_commands = (
            " --htp_performance_mode 2 --log_level 4"  # kHtpBurst, kLogLevelVerbose
        )
        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=burst_runtime_commands,
            output_callback=partial(output_callback, is_burst=True),
            save_inference_speed=True,
        )
        burst_speed = 1000 / self.inference_speed  # inferences per second

        power_saver_runtime_commands = " --htp_performance_mode 6 --log_level 4"  # kHtpHighPowerSaver, kLogLevelVerbose
        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=power_saver_runtime_commands,
            output_callback=partial(output_callback, is_burst=False),
            save_inference_speed=True,
        )
        power_saver_speed = 1000 / self.inference_speed  # inferences per second

        # Only need to ensure device burst is faster than high power saver
        if not self.enable_x86_64:
            self.assertGreater(
                burst_speed,
                power_saver_speed,
                f"Burst mode should be faster than high power saver mode, Burst: {burst_speed} inference / second, High Power Saver: {power_saver_speed} inference /second.",
            )

    def test_qnn_backend_runtime_option_log(self):
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        runtime_commands = " --log_level 4"  # kLogLevelVerbose

        def output_callback(log_msg):
            msg = log_msg.stdout
            # Check log prefix, different QNN version will have slightly different message format.
            self.assertTrue(
                any(
                    sub in msg
                    for sub in [
                        "[Qnn ExecuTorch]: QnnDsp <V>",
                        "[Qnn ExecuTorch]:  <V>",
                    ]
                ),
                "Expecting Verbose message in log",
            )

        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=runtime_commands,
            output_callback=output_callback,
        )

    def test_qnn_backend_runtime_option_profile(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            profile=False,  # Turn on using runtime command
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        runtime_commands = " --profile_level 2"  # kProfileDetailed
        # With same model, expected_profile events for this UT should match test_qnn_backend_profile_op
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_profile_events=30,
            extra_cmds=runtime_commands,
        )

    def test_qnn_backend_shared_buffer(self):
        TestQNN.shared_buffer = True
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
        )
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            shared_buffer=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
        )

    def test_qnn_backend_online_prepare(self):
        if self.enable_x86_64:
            self.skipTest("TODO: add online_prepare support on host platform")

        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(module, sample_input)

    @unittest.expectedFailure
    def test_qnn_backend_context_direct(self):
        # TODO: Fix QNN tools pairs with np 2.x
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = ContextBinaryExample()  # noqa: F405
            generate_context_binary(
                module=module,
                inputs=module.example_inputs(),
                quantized=False,
                artifact_dir=tmp_dir,
            )
            ctx_path = f"{tmp_dir}/model_ctx.bin"
            bundle_program = from_context_binary(ctx_path, "ctx_loader")
            self.verify_output(
                module,
                tuple(
                    torch.randn(size=v.shape, dtype=v.dtype)
                    for v in bundle_program["inputs"].values()
                ),
                bundle_program["edge_program_manager"].to_executorch(),
            )

    def test_qnn_backend_context_extraction(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        compiler_specs = [
            self.compiler_specs,
        ]
        validators = [validate_context_binary]

        for compiler_spec, validate in zip(compiler_specs, validators):
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module, sample_input, compiler_spec
            )
            lowered_module = edge_prog_mgr.exported_program().graph_module._modules[
                "lowered_module_0"
            ]
            qnn_mgr = PyQnnManagerAdaptor.QnnManager(
                lowered_module.compile_specs[0].value
            )
            qnn_mgr.Init()
            binary = qnn_mgr.StripProtocol(lowered_module.processed_bytes)
            validate(binary)

    def test_qnn_backend_dump_context_from_pte(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        compiler_specs = [
            self.compiler_specs,
        ]
        validators = [validate_context_binary]

        for compiler_spec, validate in zip(compiler_specs, validators):
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module, sample_input, compiler_spec
            ).to_executorch()

            with tempfile.TemporaryDirectory() as tmp_dir:
                pte_path = f"{tmp_dir}/model.pte"
                with open(pte_path, "wb") as f:
                    edge_prog_mgr.write_to_file(f)

                dump_context_from_pte(pte_path)
                binary_name = f"{tmp_dir}/forward_0.bin"
                self.assertTrue(os.path.isfile(binary_name))
                with open(binary_name, "rb") as f:
                    stripped_binary = f.read()
                    validate(stripped_binary)

    def test_qnn_backend_draw_graph(self):
        golden_data = """digraph test {
            rankdir=TB
            input_0_x_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightgreen">name: input_0_x_0</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            p_conv2_weight_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: p_conv2_weight_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [3, 3, 32, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            p_conv2_bias_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: p_conv2_bias_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_convolution_default_1_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_convolution_default_1_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_relu_default_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_relu_default_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_relu_default_1_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_relu_default_1_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            output_aten_add_tensor_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightgreen">name: output_aten_add_tensor_0</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            p_conv1_weight_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: p_conv1_weight_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [3, 3, 32, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            p_conv1_bias_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: p_conv1_bias_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_convolution_default_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_convolution_default_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            input_0_x_0 -> aten_convolution_default_1_0
            p_conv2_weight_0 -> aten_convolution_default_1_0
            p_conv2_bias_0 -> aten_convolution_default_1_0
            aten_convolution_default_0 -> aten_relu_default_0
            input_0_x_0 -> aten_convolution_default_0
            p_conv1_weight_0 -> aten_convolution_default_0
            p_conv1_bias_0 -> aten_convolution_default_0
            aten_convolution_default_1_0 -> aten_relu_default_1_0
            aten_relu_default_0 -> output_aten_add_tensor_0
            aten_relu_default_1_0 -> output_aten_add_tensor_0
        }
        """
        module = DrawGraphModel()  # noqa: F405
        sample_input = (torch.randn(1, 32, 28, 28),)
        # TODO: Figure out how to get the original graph module with the to_edge_transform_and_lowering_to_qnn API
        delegated_program = capture_program(module, sample_input)

        """
        This piece of code simulates the behavior of the final preprocessing step to obtain the op wrapper list.
        In practice, users need to set a breakpoint in the preprocessing step and use the DrawGraph tool to visualize the graph.
        """
        graph_module = QnnPassManager().transform_for_preprocess_pipeline(
            delegated_program.exported_program
        )
        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            delegated_program.exported_program, enable_tensor_dump=False
        )

        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        if isinstance(py_op_wrapper, List):
                            py_op_wrapper_list.extend(py_op_wrapper)
                        else:
                            py_op_wrapper_list.append(py_op_wrapper)
                elif node.op in [
                    "get_attr",
                    "placeholder",
                    "output",
                ]:
                    continue
        # random py_op_wrapper_list to check it's correctness
        random.shuffle(py_op_wrapper_list)
        DrawGraph("test", ".", py_op_wrapper_list, dot_string=True)
        test_file = os.path.join(".", "test.dot")
        with open(test_file, "r") as test:
            test_data = test.read()
        assert sorted(golden_data.split()) == sorted(
            test_data.split()
        ), "Generated .dot file does not match the golden file."

    def test_qnn_backend_generate_optrace(self):
        if self.enable_x86_64:
            self.skipTest(
                "At the moment, testing is only being conducted on the device."
            )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        backend_options = generate_htp_compiler_spec(use_fp16=True)

        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                online_prepare=True,
            ),
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                optrace=True,
            ),
        ]

        for compiler_spec in compiler_specs:
            with tempfile.TemporaryDirectory() as tmp_dir:
                edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module, sample_input, compiler_spec
                ).to_executorch()
                pte_path = f"{tmp_dir}/model.pte"
                with open(pte_path, "wb") as f:
                    edge_prog_mgr.write_to_file(f)

                adb = self.get_adb_tool(pte_path)
                binaries_trace = generate_optrace(
                    tmp_dir,
                    self.chipset_table[self.model],
                    adb,
                    pte_path,
                    [sample_input],
                )
                for _, (optrace, qhas) in binaries_trace.items():
                    with open(optrace, "r") as optrace_file:
                        optrace_data = json.load(optrace_file)
                        # {
                        #  header:
                        #    {
                        #     'header_version': {'major': x, 'minor': y, 'patch': z},
                        #     'version': {'major': x, 'minor': y, 'patch': z},
                        #     'artifact_type': 'OP_TRACE'
                        #    }
                        #  traceEvents:
                        #    {...}
                        # }
                        for row in optrace_data["traceEvents"]:
                            self.assertIn("pid", row)
                    with open(qhas, "r") as qhas_file:
                        qhas_data = json.load(qhas_file)
                        self.assertIn("data", qhas_data)


class TestQNNQuantizedUtils(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
        )

    def test_qnn_backend_dump_intermediate_outputs(self):
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            dump_intermediate_outputs=True,
        )
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_intermediate_events=5,
        )

    def test_qnn_backend_dynamic_shape(self):
        from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_DTYPE,
            QCOM_QUANT_ATTRS,
        )
        from executorch.exir.capture._config import ExecutorchBackendConfig

        module = Add()  # noqa: F405
        last_dim = torch.export.Dim("last_dim", min=1, max=8)
        dynamic_shapes = {"x": {3: last_dim}, "y": {3: last_dim}}
        # the tracing input in dynamic dimension should have maximun expected
        # value for QNN to be setup correctly
        input_shape = (1, 2, 3, last_dim.max)
        sample_input = (
            torch.randint(0, 2, input_shape, dtype=torch.float),
            torch.randint(0, 2, input_shape, dtype=torch.float),
        )
        module = self.get_qdq_module(
            module,
            sample_input,
            quant_dtype=QuantDtype.use_16a16w,
            dynamic_shapes=dynamic_shapes,
        )
        # only few ops with 16bit are supported with dynamic shape now
        # strip unsupported quantize / dequantize ops generated in preprocess
        pass_jobs = get_capture_program_passes()
        pass_jobs[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        pass_jobs[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = lambda n: (
            torch.uint16 if any(name in n.name for name in ["x", "y", "add"]) else None
        )
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module,
            sample_input,
            self.compiler_specs,
            dynamic_shapes=dynamic_shapes,
            passes_job=pass_jobs,
        )

        # collect encodings for ios
        input_encodings, output_encodings = [], []
        for n in edge_prog_mgr.exported_program().graph.nodes:
            if n.op == "placeholder":
                input_encodings.append(n.meta[QCOM_QUANT_ATTRS])
                input_encodings[-1][QCOM_DTYPE] = torch.uint16
            elif n.op == "output":
                output_encodings = n.meta[QCOM_QUANT_ATTRS_MAP].values()
                for output_encoding in output_encodings:
                    output_encoding[QCOM_DTYPE] = torch.uint16

        exec_prog = edge_prog_mgr.to_executorch(
            ExecutorchBackendConfig(passes=[BuildQuantIo()])
        )

        for dim in range(last_dim.min, last_dim.max + 1):
            with self.subTest(i=dim):
                input_shape = (1, 2, 3, dim)
                sample_input = (torch.rand(input_shape), torch.rand(input_shape))
                self.verify_output(
                    module,
                    sample_input,
                    exec_prog,
                    input_encodings=tuple(input_encodings),
                    output_encodings=tuple(output_encodings),
                    check_io_shape=True,
                )

    def test_qnn_backend_rewrite_prepared_observer(self):
        from torchao.quantization.pt2e import FixedQParamsObserver

        module = ReWriteObs()  # noqa: F405
        sample_input = (torch.randn([3, 1]),)
        module = torch.export.export(module, sample_input, strict=True).module()

        quantizer = make_quantizer()

        prepared = prepare_pt2e(module, quantizer)
        prepared(*sample_input)

        new_obs = FixedQParamsObserver(
            scale=0.004,
            zero_point=0,
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
        )

        rewrite_prepared_observer(prepared, {"activation_post_process_2": new_obs})
        self.assertTrue(
            prepared.activation_post_process_1
            == prepared.activation_post_process_2
            == new_obs
        )
        quantized_module = convert_pt2e(prepared)
        self.lower_module_and_test_output(quantized_module, sample_input)

    def test_qnn_backend_saver_backend(self):
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            saver=True,
        )
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)

        from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
            flatbuffer_to_option,
            option_to_flatbuffer,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            option = flatbuffer_to_option(TestQNN.compiler_specs[0].value)
            option.saver_output_dir = f"{tmp_dir}/saver_output"
            TestQNN.compiler_specs[0].value = option_to_flatbuffer(option)

            with self.assertRaises(SystemExit):
                self.lower_module_and_test_output(module, sample_input)
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/saver_output/params.bin"),
                "failed to find params.bin",
            )
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/saver_output/saver_output.c"),
                "failed to find saver_output.c",
            )

    def test_qnn_backend_skip_node_id_partitioner(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=3,
            skip_node_id_set={"aten_add_tensor", "aten_mean_dim"},
        )

    def test_qnn_backend_skip_node_id_quantizer(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))

        # define compile specs
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        # define quantizer
        quantizer = make_quantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, edge_prog_mgrs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            compiler_specs=compiler_specs,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_id_set={"conv2d"},
        )
        self.assertEqual(len(edge_prog_mgrs), 1)
        # lower all graph again, the skipped operators will be left in CPU
        exec_prog = to_edge(
            torch.export.export(graph_module, sample_input, strict=True),
        ).to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_skip_node_op_partitioner(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=2,
            skip_node_op_set={"aten.add.Tensor"},
        )

    def test_qnn_backend_skip_node_op_quantizer(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))

        # define compile specs
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        # define quantizer
        quantizer = make_quantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, edge_prog_mgrs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            compiler_specs=compiler_specs,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_op_set={torch.ops.aten.add.Tensor},
        )
        self.assertEqual(len(edge_prog_mgrs), 2)
        # lower all graph again, the skipped operators will be left in CPU
        exec_prog = exec_prog = to_edge(
            torch.export.export(graph_module, sample_input, strict=True),
        ).to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_spill_fill_buffer_size(self):
        module = LargeTensorLinear()  # noqa: F405
        sample_input = (torch.randn(1, 256, 512),)
        module = self.get_qdq_module(module, sample_input)
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        edge_prog = to_edge_transform_and_lower_to_qnn(
            module, sample_input, compiler_specs
        )

        max_sf_size = update_spill_fill_size(edge_prog.exported_program())
        self.assertNotEqual(0, max_sf_size)

    def test_qnn_backend_graph_level_mixed_precision(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))

        # define compile spec
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        # define quantizer
        quantizer = make_quantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, edge_prog_mgrs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            compiler_specs=compiler_specs,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_id_set={"add", "mean"},
            fallback_to_cpu=False,
        )
        self.assertEqual(len(edge_prog_mgrs), 5)
        # lower all graph again, the skipped operators will be delegated with fp16
        exec_prog = to_edge(
            torch.export.export(graph_module, sample_input, strict=True),
        ).to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_multi_contexts(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        pass_jobs = get_capture_program_passes()
        split_graph_pass, setting = self.split_graph(4)
        pass_jobs[split_graph_pass] = setting
        dep_table = get_passes_dependency_for_capture_program()
        dep_table[split_graph_pass] = [FoldQDQ]
        edge_prog = to_edge_transform_and_lower_to_qnn(
            module,
            sample_input,
            compiler_specs,
            dep_table=dep_table,
            passes_job=pass_jobs,
        )

        update_spill_fill_size(edge_prog.exported_program())
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_multi_contexts_composite(self):
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = CompositeDelegateModule(  # noqa: F405
            compiler_specs=compiler_specs,
            to_edge_transform_and_lower_method=to_edge_transform_and_lower_to_qnn,
            quantize_method=self.get_qdq_module,
        )
        sample_input = module.get_random_input()
        edge_prog = to_edge(
            torch.export.export(module, sample_input, strict=True),
        )
        update_spill_fill_size(edge_prog.exported_program())
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module.get_reference_module(), sample_input, exec_prog)

    def test_qnn_backend_multi_graphs(self):
        if self.enable_x86_64:
            self.skipTest("weight sharing is not supported on host machine")

        seq_conv = Conv2dSequential()  # noqa: F405
        # weight sharing
        modules = [seq_conv, seq_conv.second]
        sample_inputs = [(torch.randn([1, 1, 3, 3]),), (torch.randn([1, 3, 3, 3]),)]
        graph_names = ["seq_conv", "single_conv"]
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_weight_sharing=True,
        )
        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                graph_name=graph_name,
            )
            for graph_name in graph_names
        ]
        modules_dict = {}
        sample_inputs_dict = {}
        compiler_specs_dict = {}
        for i, graph_name in enumerate(graph_names):
            module_exported = torch.export.export(modules[i], sample_inputs[i]).module()
            module_prepared = prepare_pt2e(module_exported, make_quantizer())
            module_prepared(*sample_inputs[i])
            modules_dict[graph_name] = convert_pt2e(module_prepared)
            sample_inputs_dict[graph_name] = sample_inputs[i]
            compiler_specs_dict[graph_name] = compiler_specs[i]
        delegated_program = to_edge_transform_and_lower_to_qnn(
            modules_dict, sample_inputs_dict, compiler_specs_dict
        )

        executorch_prog = delegated_program.to_executorch()
        for index, module in enumerate(modules):
            self.verify_output(
                module=module,
                sample_inputs=sample_inputs[index],
                executorch_prog=executorch_prog,
                method_index=index,
            )

    def test_qnn_backend_profile_op(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            profile=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_profile_events=30,
        )

    def test_qnn_backend_runtime_option_htp_performance(self):
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)

        def output_callback(log_msg, is_burst):
            msg = log_msg.stdout
            # Refer to HtpDevice.cpp for the following values
            min_voltage = (
                "coreVoltageCornerMin 160" if is_burst else "coreVoltageCornerMin 80"
            )
            self.assertTrue(min_voltage in msg, f"Expecting '{min_voltage} ' in log")

        burst_runtime_commands = (
            " --htp_performance_mode 2 --log_level 4"  # kHtpBurst, kLogLevelVerbose
        )
        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=burst_runtime_commands,
            output_callback=partial(output_callback, is_burst=True),
            save_inference_speed=True,
        )
        burst_speed = 1000 / self.inference_speed  # num inference per second

        power_saver_runtime_commands = " --htp_performance_mode 6 --log_level 4"  # kHtpHighPowerSaver, kLogLevelVerbose
        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=power_saver_runtime_commands,
            output_callback=partial(output_callback, is_burst=False),
            save_inference_speed=True,
        )
        power_saver_speed = 1000 / self.inference_speed  # num inference per second

        # Only need to ensure device burst is faster than high power saver
        if not self.enable_x86_64:
            self.assertGreater(
                burst_speed,
                power_saver_speed,
                f"Burst mode should be faster than high power saver mode, Burst: {burst_speed} inference / second, High Power Saver: {power_saver_speed} inference /second.",
            )

    def test_qnn_backend_runtime_option_log(self):
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        runtime_commands = " --log_level 4"  # kLogLevelVerbose

        def output_callback(log_msg):
            msg = log_msg.stdout
            # Check log prefix, different QNN version will have slightly different message format.
            self.assertTrue(
                any(
                    sub in msg
                    for sub in [
                        "[Qnn ExecuTorch]: QnnDsp <V>",
                        "[Qnn ExecuTorch]:  <V>",
                    ]
                ),
                "Expecting Verbose message in log",
            )

        self.lower_module_and_test_output(
            module,
            sample_input,
            extra_cmds=runtime_commands,
            output_callback=output_callback,
        )

    def test_qnn_backend_runtime_option_profile(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            profile=False,  # Turn on using runtime command
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        runtime_commands = " --profile_level 2"  # kProfileDetailed
        # With same model, expected_profile events for this UT should match test_qnn_backend_profile_op
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_profile_events=30,
            extra_cmds=runtime_commands,
        )

    def test_qnn_backend_shared_buffer(self):
        TestQNN.shared_buffer = True
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            shared_buffer=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
        )

    def test_qnn_backend_online_prepare(self):
        if self.enable_x86_64:
            self.skipTest("TODO: add online_prepare support on host platform")

        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset_table[TestQNN.model],
            backend_options=backend_options,
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.expectedFailure
    def test_qnn_backend_context_direct(self):
        # TODO: Fix QNN tools pairs with np 2.x
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = ContextBinaryExample()  # noqa: F405
            generate_context_binary(
                module=module,
                inputs=module.example_inputs(),
                quantized=True,
                artifact_dir=tmp_dir,
            )
            ctx_path = f"{tmp_dir}/model_ctx.bin"
            bundle_program = from_context_binary(ctx_path, "ctx_loader")
            self.verify_output(
                module,
                tuple(
                    torch.randn(size=v.shape, dtype=v.dtype)
                    for v in bundle_program["inputs"].values()
                ),
                bundle_program["edge_program_manager"].to_executorch(),
            )

    def test_qnn_backend_context_extraction(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        compiler_specs = [
            self.compiler_specs,
        ]
        validators = [validate_context_binary]

        for compiler_spec, validate in zip(compiler_specs, validators):
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module, sample_input, compiler_spec
            )
            lowered_module = edge_prog_mgr.exported_program().graph_module._modules[
                "lowered_module_0"
            ]
            qnn_mgr = PyQnnManagerAdaptor.QnnManager(
                lowered_module.compile_specs[0].value
            )
            qnn_mgr.Init()
            binary = qnn_mgr.StripProtocol(lowered_module.processed_bytes)
            validate(binary)

    def test_qnn_backend_dump_context_from_pte(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        compiler_specs = [
            self.compiler_specs,
        ]
        validators = [validate_context_binary]

        for compiler_spec, validate in zip(compiler_specs, validators):
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module, sample_input, compiler_spec
            ).to_executorch()

            with tempfile.TemporaryDirectory() as tmp_dir:
                pte_path = f"{tmp_dir}/model.pte"
                with open(pte_path, "wb") as f:
                    edge_prog_mgr.write_to_file(f)

                dump_context_from_pte(pte_path)
                binary_name = f"{tmp_dir}/forward_0.bin"
                self.assertTrue(os.path.isfile(binary_name))
                with open(binary_name, "rb") as f:
                    stripped_binary = f.read()
                    validate(stripped_binary)

    def test_qnn_backend_draw_graph(self):
        golden_data = """digraph test {
            rankdir=TB
            aten_convolution_default_1_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_convolution_default_1_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_relu_default_1_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_relu_default_1_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            quantized_decomposed_quantize_per_tensor_default_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: quantized_decomposed_quantize_per_tensor_default_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 32, 28, 28]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            b__frozen_param2_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: b__frozen_param2_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [3, 3, 32, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            b__frozen_param3_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: b__frozen_param3_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            b__frozen_param0_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: b__frozen_param0_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [3, 3, 32, 32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            b__frozen_param1_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightpink">name: b__frozen_param1_0</TD></TR>
                        <TR><TD BGCOLOR="lightpink">data_type: Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32</TD></TR>
                        <TR><TD BGCOLOR="lightpink">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC</TD></TR>
                        <TR><TD BGCOLOR="lightpink">dims: [32]</TD></TR>
                        <TR><TD BGCOLOR="lightpink">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_convolution_default_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_convolution_default_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_relu_default_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_relu_default_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            aten_add_tensor_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="white">name: aten_add_tensor_0</TD></TR>
                        <TR><TD BGCOLOR="white">data_type: Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8</TD></TR>
                        <TR><TD BGCOLOR="white">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE</TD></TR>
                        <TR><TD BGCOLOR="white">dims: [1, 28, 28, 32]</TD></TR>
                        <TR><TD BGCOLOR="white">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            input_0_x_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightgreen">name: input_0_x_0</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">dims: [1, 32, 28, 28]</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            output_quantized_decomposed_dequantize_per_tensor_tensor_0 [label=<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                        <TR><TD BGCOLOR="lightgreen">name: output_quantized_decomposed_dequantize_per_tensor_tensor_0</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">data_type: Qnn_DataType_t.QNN_DATATYPE_FLOAT_32</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">tensor_type: Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">dims: [1, 32, 28, 28]</TD></TR>
                        <TR><TD BGCOLOR="lightgreen">quantization_encoding: Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED</TD></TR>
                    </TABLE>> color=black fillcolor=transparent shape=box style=rounded]
            quantized_decomposed_quantize_per_tensor_default_0 -> aten_convolution_default_1_0
            input_0_x_0 -> quantized_decomposed_quantize_per_tensor_default_0
            b__frozen_param2_0 -> aten_convolution_default_1_0
            b__frozen_param3_0 -> aten_convolution_default_1_0
            aten_convolution_default_1_0 -> aten_relu_default_1_0
            quantized_decomposed_quantize_per_tensor_default_0 -> aten_convolution_default_0
            b__frozen_param0_0 -> aten_convolution_default_0
            b__frozen_param1_0 -> aten_convolution_default_0
            aten_convolution_default_0 -> aten_relu_default_0
            aten_relu_default_0 -> aten_add_tensor_0
            aten_relu_default_1_0 -> aten_add_tensor_0
            aten_add_tensor_0 -> output_quantized_decomposed_dequantize_per_tensor_tensor_0
        }"""
        module = DrawGraphModel()  # noqa: F405
        sample_input = (torch.randn(1, 32, 28, 28),)
        module = self.get_qdq_module(module, sample_input)
        # TODO: Figure out how to get the original graph module with to_edge_transform_and_lowering_to_qnn
        delegated_program = capture_program(module, sample_input)

        """
        This piece of code simulates the behavior of the final preprocessing step to obtain the op wrapper list.
        In practice, users need to set a breakpoint in the preprocessing step and use the DrawGraph tool to visualize the graph.
        """
        graph_module = QnnPassManager().transform_for_preprocess_pipeline(
            delegated_program.exported_program
        )
        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            delegated_program.exported_program, enable_tensor_dump=False
        )

        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        if isinstance(py_op_wrapper, List):
                            py_op_wrapper_list.extend(py_op_wrapper)
                        else:
                            py_op_wrapper_list.append(py_op_wrapper)
                elif node.op in [
                    "get_attr",
                    "placeholder",
                    "output",
                ]:
                    continue
        # random py_op_wrapper_list to check it's correctness
        random.shuffle(py_op_wrapper_list)
        DrawGraph("test", ".", py_op_wrapper_list, dot_string=True)
        test_file = os.path.join(".", "test.dot")
        with open(test_file, "r") as test:
            test_data = test.read()
        assert sorted(golden_data.split()) == sorted(
            test_data.split()
        ), "Generated .dot file does not match the golden file."

    def test_qnn_backend_generate_optrace(self):
        if self.enable_x86_64:
            self.skipTest(
                "At the moment, testing is only being conducted on the device."
            )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        backend_options = generate_htp_compiler_spec(use_fp16=True)

        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                online_prepare=True,
            ),
            generate_qnn_executorch_compiler_spec(
                soc_model=self.chipset_table[TestQNN.model],
                backend_options=backend_options,
                optrace=True,
            ),
        ]

        for compiler_spec in compiler_specs:
            with tempfile.TemporaryDirectory() as tmp_dir:
                edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                    module, sample_input, compiler_spec
                ).to_executorch()
                pte_path = f"{tmp_dir}/model.pte"
                with open(pte_path, "wb") as f:
                    edge_prog_mgr.write_to_file(f)

                adb = self.get_adb_tool(pte_path)
                binaries_trace = generate_optrace(
                    tmp_dir,
                    self.chipset_table[self.model],
                    adb,
                    pte_path,
                    [sample_input],
                )
                for _, (optrace, qhas) in binaries_trace.items():
                    with open(optrace, "r") as optrace_file:
                        optrace_data = json.load(optrace_file)
                        # {
                        #  header:
                        #    {
                        #     'header_version': {'major': x, 'minor': y, 'patch': z},
                        #     'version': {'major': x, 'minor': y, 'patch': z},
                        #     'artifact_type': 'OP_TRACE'
                        #    }
                        #  traceEvents:
                        #    {...}
                        # }
                        for row in optrace_data["traceEvents"]:
                            self.assertIn("pid", row)
                    with open(qhas, "r") as qhas_file:
                        qhas_data = json.load(qhas_file)
                        self.assertIn("data", qhas_data)


class TestExampleLLMScript(TestQNN):
    def test_static_gemma3_1b(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--ptq",
            "16a4w_block",
            "--temperature",
            "0",
            "--decoder_model",
            "gemma3-1b",
            "--model_mode",
            "kv",
            "--max_seq_len",
            "1024",
            "--eval_perplexity",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
            "--enable_masked_softmax",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                if not self.compile_only:
                    self.assertLessEqual(msg["wiki_ppl"], 23)
                if not self.enable_x86_64:
                    pte_size = msg["pte_size"]
                    self.assertLessEqual(pte_size, 1_200_000_000)  # 1.2GB
                inference_speed_ref = {"SM8650": 70, "SM8750": 100}
                if (
                    not self.compile_only
                    and not self.enable_x86_64
                    and self.model in inference_speed_ref
                ):
                    self.assertGreaterEqual(
                        msg["inference_speed"], inference_speed_ref[self.model]
                    )

    def test_llama3_2_1b(self):
        if not self.required_envs():
            self.skipTest("missing required envs")
        assert (
            self.llama_artifacts is not None
        ), "Please provide path to llama artifacts"

        prompt = "What is the meaning of life?"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--checkpoint",
            f"{self.llama_artifacts}/consolidated.00.pth",
            "--params",
            f"{self.llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{self.llama_artifacts}/tokenizer.model",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--temperature",
            "0",
            "--decoder_model",
            "llama3_2",
            "--model_mode",
            "hybrid",
            "--prefill_ar_len",
            "32",
            "--max_seq_len",
            "512",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        golden_start_with = "<|start_header_id|>user<|end_header_id|>"
        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                if not self.compile_only:
                    model_out = msg["result"][0]
                    self.assertTrue(
                        model_out.startswith(golden_start_with),
                        f"Expected Output: {golden_start_with}. Actual Output: {model_out}",
                    )
                # x86 does not allow weight sharing, so we don't check pte size.
                # Inference speed on x86 is slow, so we only check when running on Android
                if not self.enable_x86_64:
                    pte_size = msg["pte_size"]
                    self.assertLessEqual(pte_size, 1_300_000_000)  # 1.3GB
                if not self.compile_only and not self.enable_x86_64:
                    self.assertGreaterEqual(msg["inference_speed"], 66)  # Lanai

    def test_llama_stories_260k(self):
        if not self.required_envs():
            self.skipTest("missing required envs")
        assert (
            self.llama_artifacts is not None
        ), "Please provide path to llama artifacts"

        prompt = "Once"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--checkpoint",
            f"{self.llama_artifacts}/stories260K.pt",
            "--params",
            f"{self.llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{self.llama_artifacts}/tokenizer.model",
            "--tokenizer_bin",
            f"{self.llama_artifacts}/tokenizer.bin",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--temperature",
            "0",
            "--decoder_model",
            "stories260k",
            "--model_mode",
            "hybrid",
            "--prefill_ar_len",
            "32",
            "--max_seq_len",
            "128",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        golden_start_with = "Once upon a time,"
        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                if not self.compile_only:
                    model_out = msg["result"][0]
                    print(f"Model CI result:{model_out[: len(golden_start_with)]}")
                    self.assertTrue(
                        model_out.startswith(golden_start_with),
                        f"Expected Output: {golden_start_with}. Actual Output: {model_out}",
                    )
                # x86 does not allow weight sharing, so we don't check pte size
                if not self.enable_x86_64:
                    pte_size = msg["pte_size"]
                    self.assertLessEqual(pte_size, 2_020_000)  # 2MB
                if not self.compile_only and not self.enable_x86_64:
                    self.assertGreaterEqual(msg["inference_speed"], 1600)  # Lanai

    def test_llama_stories_110m(self):
        if not self.required_envs():
            self.skipTest("missing required envs")
        assert (
            self.llama_artifacts is not None
        ), "Please provide path to llama artifacts"

        prompt = "Once"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--checkpoint",
            f"{self.llama_artifacts}/stories110M.pt",
            "--params",
            f"{self.llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{self.llama_artifacts}/tokenizer.model",
            "--tokenizer_bin",
            f"{self.llama_artifacts}/tokenizer.bin",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--temperature",
            "0",
            "--decoder_model",
            "stories110m",
            "--model_mode",
            "hybrid",
            "--prefill_ar_len",
            "32",
            "--max_seq_len",
            "128",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        golden_start_with = "Once upon a time,"
        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                if not self.compile_only:
                    model_out = msg["result"][0]
                    self.assertTrue(
                        model_out.startswith(golden_start_with),
                        f"Expected Output: {golden_start_with}. Actual Output: {model_out}",
                    )
                # x86 does not allow weight sharing, so we don't check pte size
                if not self.enable_x86_64:
                    pte_size = msg["pte_size"]
                    self.assertLessEqual(pte_size, 130_000_000)  # 130MB
                if not self.compile_only and not self.enable_x86_64:
                    self.assertGreaterEqual(msg["inference_speed"], 220)  # Lanai

    def test_static_phi4(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--decoder_model",
            "phi_4_mini",
            "--model_mode",
            "kv",
            "--max_seq_len",
            "1024",
            "--eval_perplexity",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])
            cmds.extend(
                [
                    "--quant_attrs_path",
                    f"{self.pre_gen_pte}/kv_llama_qnn_quant_attrs.json",
                ]
            )

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                inference_speed_ref = {"SM8650": 14, "SM8750": 19}
                self.assertLessEqual(msg["wiki_ppl"], 12)
                self.assertLessEqual(msg["pte_size"], 4_000_000_000)  # 4GB
                if self.model in inference_speed_ref:
                    self.assertGreaterEqual(
                        msg["inference_speed"], inference_speed_ref[self.model]
                    )

    def test_static_qwen2_5(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--decoder_model",
            "qwen2_5-0_5b",
            "--model_mode",
            "kv",
            "--max_seq_len",
            "1024",
            "--eval_perplexity",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                inference_speed_ref = {"SM8650": 115, "SM8750": 155}
                self.assertLessEqual(msg["wiki_ppl"], 15)
                self.assertLessEqual(msg["pte_size"], 600_000_000)  # 600MB
                if self.model in inference_speed_ref:
                    self.assertGreaterEqual(
                        msg["inference_speed"], inference_speed_ref[self.model]
                    )

    def test_static_qwen3(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--decoder_model",
            "qwen3-0_6b",
            "--model_mode",
            "kv",
            "--max_seq_len",
            "1024",
            "--eval_perplexity",
            "--tasks",
            "wikitext",
            "--limit",
            "1",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                inference_speed_ref = {"SM8650": 38, "SM8750": 56}
                self.assertLessEqual(msg["wiki_ppl"], 18)
                self.assertLessEqual(msg["pte_size"], 950_000_000)  # 950MB
                if self.model in inference_speed_ref:
                    self.assertGreaterEqual(
                        msg["inference_speed"], inference_speed_ref[self.model]
                    )

    def test_static_smollm2(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--decoder_model",
            "smollm2_135m",
            "--model_mode",
            "kv",
            "--temperature",
            "0",
            "--prefill_ar_len",
            "128",
            "--max_seq_len",
            "1024",
            "--eval_perplexity",
            "--task",
            "wikitext",
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertLessEqual(msg["wiki_ppl"], 25)
                self.assertGreaterEqual(msg["inference_speed"], 200)

    def test_qwen2_5(self):
        if not self.required_envs([]):
            self.skipTest("missing required envs")
        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/qwen2_5/qwen2_5.py",
            "--prompt",
            prompt,
            "--decoder_model",
            "qwen2.5_0.5B",
            "--ptq",
            "16a8w",
            "--enable_spinquant_r3",
            "--max_seq_len",
            "128",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
        ]
        if self.compile_only:
            cmds.extend(["--compile_only"])
        elif self.device:
            cmds.extend(["--device", self.device])
        if self.host:
            cmds.extend(["--host", self.host])
        elif self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])
        if self.pre_gen_pte:
            cmds.extend(["--pre_gen_pte", self.pre_gen_pte])

        golden_start_with = "My favourite condiment is iced tea."
        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                if not self.compile_only:
                    model_out = msg["result"][0]
                    self.assertTrue(
                        model_out.startswith(golden_start_with),
                        f"Expected Output: '{golden_start_with}' Actual Output: '{model_out}'",
                    )


class TestExampleOssScript(TestQNN):
    def test_albert(self):
        if not self.required_envs([self.sentence_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/albert.py",
            "--dataset",
            self.sentence_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["accuracy"], 0.95)

    def test_bert(self):
        if not self.required_envs([self.sentence_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/bert.py",
            "--dataset",
            self.sentence_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["accuracy"], 0.55)

    def test_conv_former(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/conv_former.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 70)
                self.assertGreaterEqual(msg["top_5"], 92)

    def test_cvt(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/cvt.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 70)
                self.assertGreaterEqual(msg["top_5"], 90)

    def test_deit(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/deit.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 76)
                self.assertGreaterEqual(msg["top_5"], 92)

    def test_dino_v2(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/dino_v2.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 62)
                self.assertGreaterEqual(msg["top_5"], 87)

    def test_distilbert(self):
        if not self.required_envs([self.sentence_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/distilbert.py",
            "--dataset",
            self.sentence_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["accuracy"], 0.48)

    def test_dit(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/dit.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 78)
                self.assertGreaterEqual(msg["top_5"], 92)

    def test_efficientnet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/efficientnet.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 61)
                self.assertGreaterEqual(msg["top_5"], 88)

    def test_efficientSAM(self):
        if not self.required_envs(
            [self.image_dataset, self.pretrained_weight, self.oss_repo]
        ):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/efficientSAM/efficientSAM.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--oss_repo",
            self.oss_repo,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                # test with efficient_sam_vitt.pt
                self.assertGreaterEqual(msg["MIoU"], 0.95)

    def test_esrgan(self):
        if not self.required_envs([self.oss_repo]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/esrgan.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--default_dataset",
            "--oss_repo",
            self.oss_repo,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["PSNR"], 23)
                self.assertGreaterEqual(msg["SSIM"], 0.85)

    def test_eurobert(self):
        if not self.required_envs([self.sentence_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/eurobert.py",
            "--dataset",
            self.sentence_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["accuracy"], 0.54)

    def test_fastvit(self):
        if not self.required_envs(
            [self.image_dataset, self.pretrained_weight, self.oss_repo]
        ):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/fastvit.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--oss_repo",
            self.oss_repo,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 60)
                self.assertGreaterEqual(msg["top_5"], 77)

    def test_fbnet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/fbnet.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 67)
                self.assertGreaterEqual(msg["top_5"], 88)

    def test_focalnet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/focalnet.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 54)
                self.assertGreaterEqual(msg["top_5"], 80)

    def test_gMLP(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/gMLP_image_classification.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 70)
                self.assertGreaterEqual(msg["top_5"], 88)

    @unittest.skip("Only outputs good accuracy in QNN 2.29")
    def test_mobilevit_v2(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/mobilevit_v2.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 50)
                self.assertGreaterEqual(msg["top_5"], 85)

    def test_mobilevit_v1(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/mobilevit_v1.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 76)
                self.assertGreaterEqual(msg["top_5"], 93)

    def test_pvt(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/pvt.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 65)
                self.assertGreaterEqual(msg["top_5"], 83)

    def test_regnet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        weights = ["regnet_y_400mf", "regnet_x_400mf"]
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/regnet.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        for weight in weights:
            p = subprocess.Popen(
                cmds + ["--weights", weight], stdout=subprocess.DEVNULL
            )
            with Listener((self.ip, self.port)) as listener:
                conn = listener.accept()
                p.communicate()
                msg = json.loads(conn.recv())
                if "Error" in msg:
                    self.fail(msg["Error"])
                else:
                    self.assertGreaterEqual(msg["top_1"], 64)
                    self.assertGreaterEqual(msg["top_5"], 86)

    def test_retinanet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/retinanet.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--dataset",
            self.image_dataset,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["mAP"], 0.6)

    def test_roberta(self):
        if not self.required_envs([self.sentence_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/roberta.py",
            "--dataset",
            self.sentence_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["accuracy"], 0.54)

    def test_squeezenet(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/squeezenet.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 27)
                self.assertGreaterEqual(msg["top_5"], 59)

    def test_ssd300_vgg16(self):
        if not self.required_envs([self.pretrained_weight, self.oss_repo]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/ssd300_vgg16.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--oss_repo",
            self.oss_repo,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["mAP"], 0.76)

    def test_swin_transformer(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/swin_transformer.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 71)
                self.assertGreaterEqual(msg["top_5"], 90)

    def test_t5(self):
        if not self.required_envs([self.qa_dataset]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/t5/t5.py",
            "--dataset",
            self.qa_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["f1"], 0.72)

    def test_whisper(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/whisper/whisper.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertLessEqual(msg["wer"], 0.25)


class TestExampleQaihubScript(TestQNN):
    def test_utils_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = ContextBinaryExample()  # noqa: F405
            generate_context_binary(
                module=module,
                inputs=module.example_inputs(),
                quantized=True,
                artifact_dir=tmp_dir,
            )
            ctx_path = f"{tmp_dir}/model_ctx.bin"
            fpath = f"{self.executorch_root}/examples/qualcomm/qaihub_scripts/utils/export.py"

            # do compilation
            compile_cmds = [
                "python",
                fpath,
                "compile",
                "-a",
                ctx_path,
                "-m",
                self.model,
                "-l",
                "False",
                "-b",
                self.build_folder,
                "-o",
                f"{tmp_dir}/output_pte",
            ]
            compile_process = subprocess.Popen(
                compile_cmds, stdout=subprocess.DEVNULL, cwd=self.executorch_root
            )
            output_pte_dir = f"{tmp_dir}/output_pte/model_ctx"
            compile_process.communicate()

            # check artifacts are correctly generated
            self.assertTrue(
                all(
                    [
                        Path(output_pte_dir).exists(),
                        Path(f"{output_pte_dir}/model_ctx.json").exists(),
                        Path(f"{output_pte_dir}/model_ctx.svg").exists(),
                    ]
                )
            )

            # prepare input files
            input_list, inputs = [], module.example_inputs()
            for name, tensor in inputs.items():
                tensor_path = f"{output_pte_dir}/{name}.pt"
                torch.save(tensor, tensor_path)
                input_list.append(tensor_path)

            # do execution
            output_data_dir = f"{tmp_dir}/output_data"
            execute_cmds = [
                "python",
                fpath,
                "execute",
                "-p",
                output_pte_dir,
                "-i",
                *input_list,
                "-s",
                self.device,
                "-z",
                "-b",
                self.build_folder,
                "-o",
                output_data_dir,
            ]
            if self.host is not None:
                execute_cmds.append(f"-H {self.host}")
            execute_process = subprocess.Popen(execute_cmds, cwd=self.executorch_root)
            execute_process.communicate()

            # read outputs
            with open(f"{output_pte_dir}/model_ctx.json", "r") as f:
                graph_info = json.load(f)

            device_output = []
            for output in graph_info["outputs"]:
                with open(f"{output_data_dir}/{output['name']}.pt", "rb") as f:
                    buffer = io.BytesIO(f.read())
                    device_output.append(torch.load(buffer, weights_only=False))

            # validate outputs
            golden_output = module.forward(inputs["x"], inputs["y"])
            self.atol, self.rtol = 1e-1, 1
            self._assert_outputs_equal(golden_output, device_output)

    def test_llama2_7b(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "Explain the rules of baseball"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/qaihub_scripts/llama/llama2/qaihub_llama2_7b.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--tokenizer_bin",
            f"{self.artifact_dir}/tokenizer.bin",
            "--context_binaries",
            f"{self.artifact_dir}",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                model_out = msg["result"]
                self.assertTrue(model_out.startswith(prompt))

    def test_llama3_8b(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "Explain the rules of baseball"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/qaihub_scripts/llama/llama3/qaihub_llama3_8b.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--tokenizer_model",
            f"{self.artifact_dir}/tokenizer.model",
            "--context_binaries",
            f"{self.artifact_dir}",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                model_out = msg["result"]
                expected_result = (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    + prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
                self.assertTrue(model_out.startswith(expected_result))

    def test_stable_diffusion(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        prompt = "a photo of an astronaut riding a horse on mars"
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/qaihub_scripts/stable_diffusion/qaihub_stable_diffusion.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--text_encoder_bin",
            f"{self.artifact_dir}/text_encoder.serialized.bin",
            "--unet_bin",
            f"{self.artifact_dir}/unet.serialized.bin",
            "--vae_bin",
            f"{self.artifact_dir}/vae.serialized.bin",
            "--vocab_json",
            f"{self.artifact_dir}/vocab.json",
            "--num_time_steps",
            "20",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            f"{prompt}",
            "--fix_latents",
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                # For the default settings and prompt, the expected results will be {PSNR: 23.258, SSIM: 0.852}
                self.assertGreaterEqual(msg["PSNR"], 20)
                self.assertGreaterEqual(msg["SSIM"], 0.8)


class TestExampleScript(TestQNN):
    def test_mobilenet_v2(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/mobilenet_v2.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 52)
                self.assertGreaterEqual(msg["top_5"], 84)

    def test_mobilenet_v3(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/mobilenet_v3.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 55)
                self.assertGreaterEqual(msg["top_5"], 81)

    def test_inception_v3(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/inception_v3.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 59)
                self.assertGreaterEqual(msg["top_5"], 78)

    def test_inception_v4(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/inception_v4.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 64)
                self.assertGreaterEqual(msg["top_5"], 85)

    def test_vit(self):
        if not self.required_envs([self.image_dataset]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/torchvision_vit.py",
            "--dataset",
            self.image_dataset,
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 69)
                self.assertGreaterEqual(msg["top_5"], 91)

    def test_edsr(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/edsr.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--default_dataset",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["PSNR"], 29)
                self.assertGreaterEqual(msg["SSIM"], 0.94)

    def test_deeplab_v3(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/deeplab_v3.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--download",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--seed",
            str(1126),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["PA"], 0.88)
                self.assertGreaterEqual(msg["MPA"], 0.79)
                self.assertGreaterEqual(msg["MIoU"], 0.67)

    @unittest.skip("dynamic shape inputs appear in recent torch.export.export")
    def test_mobilebert(self):
        if not self.required_envs([self.pretrained_weight]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/mobilebert_fine_tune.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--use_fp16",
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                cpu, htp = msg["CPU"], msg["HTP"]
                for k, v in cpu.items():
                    self.assertLessEqual(abs(v[0] - htp[k][0]), 2)

    @unittest.skip("eagar mode fake quant works well, need further investigation")
    def test_ptq_mobilebert(self):
        if not self.required_envs([self.pretrained_weight]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/mobilebert_fine_tune.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ptq",
            "16a16w",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                cpu, htp = msg["CPU"], msg["HTP"]
                for k, v in cpu.items():
                    self.assertLessEqual(abs(v[0] - htp[k][0]), 5)

    @unittest.skip("encountered undefined symbol in mainline, reopen once resolved")
    def test_wav2letter(self):
        if not self.required_envs([self.pretrained_weight]):
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/wav2letter.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--pretrained_weight",
            self.pretrained_weight,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertLessEqual(msg["wer"], 0.5)
                self.assertLessEqual(msg["cer"], 0.25)

    def test_export_example(self):
        if not self.required_envs([self.model_name]):
            self.skipTest("missing required envs")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmds = [
                "python",
                "qualcomm/scripts/export_example.py",
                "--model_name",
                self.model_name,
                "--output_folder",
                "{}/".format(tmp_dir),
                "--generate_etrecord",
            ]

            p = subprocess.Popen(
                cmds, stdout=subprocess.DEVNULL, cwd=f"{self.executorch_root}/examples"
            )
            p.communicate()
            self.assertTrue(
                Path("{0}/{1}.pte".format(tmp_dir, self.model_name)).exists()
            )


class TestUtilsScript(TestQNN):
    def required_envs(self, conditions=None) -> bool:
        conditions = [] if conditions is None else conditions
        return all(
            [
                self.executorch_root,
                self.artifact_dir,
                *conditions,
            ]
        )

    def test_custom_op(self):
        if not self.required_envs([self.op_package_dir]):
            self.skipTest("missing required envs")
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/custom_op/custom_ops_1.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--op_package_dir",
            self.op_package_dir,
            "--build_op_package",
        ]
        if self.host:
            cmds.extend(["--host", self.host])
        if self.enable_x86_64:
            cmds.extend(["--enable_x86_64"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            self.assertTrue(msg["is_close"])

    def test_debugger_generate_optrace(self):
        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/util_scripts/qairt_visualizer_demo.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--ip",
            self.ip,
            "--port",
            str(self.port),
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                for _, (optrace, qhas) in msg["binaries_trace"].items():
                    with open(optrace, "r") as optrace_file:
                        optrace_data = json.load(optrace_file)
                        for row in optrace_data:
                            self.assertIn("pid", row)
                    with open(qhas, "r") as qhas_file:
                        qhas_data = json.load(qhas_file)
                        self.assertIn("data", qhas_data)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_input = torch.randn(1, 2, 3, 4)
            ep = torch.export.export(Relu(), (sample_input,))  # noqa: F405
            torch.export.save(ep, f"{tmp_dir}/relu.pt2")
            torch.save(sample_input, f"{tmp_dir}/input_0_0.pt")
            with open(f"{tmp_dir}/input_list", "w") as f:
                f.write(f"{tmp_dir}/input_0_0.pt\n")

            # quantize
            cmds = [
                "python",
                "-m",
                "examples.qualcomm.util_scripts.cli",
                "quantize",
                "--artifact",
                f"{tmp_dir}/relu.pt2",
                "--output_folder",
                f"{tmp_dir}/q_out",
                "--input_list",
                f"{tmp_dir}/input_list",
            ]
            subprocess.run(cmds, stdout=subprocess.DEVNULL)
            self.assertTrue(os.path.isfile(f"{tmp_dir}/q_out/relu_quantized.pt2"))
            # compile
            cmds = [
                "python",
                "-m",
                "examples.qualcomm.util_scripts.cli",
                "compile",
                "--artifact",
                f"{tmp_dir}/q_out/relu_quantized.pt2",
                "--output_folder",
                f"{tmp_dir}/c_out",
                "--model",
                self.model,
            ]
            subprocess.run(cmds, stdout=subprocess.DEVNULL)
            self.assertTrue(os.path.isfile(f"{tmp_dir}/c_out/relu_quantized.pte"))
            self.assertTrue(os.path.isfile(f"{tmp_dir}/c_out/relu_quantized.svg"))
            # execute
            cmds = [
                "python",
                "-m",
                "examples.qualcomm.util_scripts.cli",
                "execute",
                "--artifact",
                f"{tmp_dir}/c_out/relu_quantized.pte",
                "--output_folder",
                f"{tmp_dir}/e_out",
                "--model",
                self.model,
                "--device",
                self.device,
                "--build_folder",
                self.build_folder,
                "--input_list",
                f"{tmp_dir}/input_list",
            ]
            subprocess.run(cmds, stdout=subprocess.DEVNULL)
            self.assertTrue(os.path.isfile(f"{tmp_dir}/e_out/output_0_0.pt"))


def setup_environment():
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-r",
        "--executorch_root",
        help="Root location of current repo",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--artifact_dir",
        help="Location for putting generated artifacts",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--image_dataset",
        help="Location for imagenet dataset",
        type=str,
    )
    parser.add_argument(
        "--qa_dataset",
        help="Location for QA dataset",
        type=str,
    )
    parser.add_argument(
        "--sentence_dataset",
        help="Location for sentence dataset",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help="Location for pretrained weighting",
        default="",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--model_name",
        help="Input the model to export",
        type=str,
    )
    parser.add_argument(
        "-P",
        "--enable_profile",
        help="Profile the performance of each operator with kProfileDetailed profile level",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--error_only",
        help="Emit log only when error happened",
        action="store_true",
    )
    parser.add_argument(
        "--oss_repo",
        help="Path to open source software model repository",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--op_package_dir",
        help="Path to operator package which generates from qnn-op-package-generator",
        default="",
        type=str,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated pte in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--llama_artifacts",
        help="A folder that contains: weight, tokenizer, and params.",
        type=str,
    )

    args, ns_args = parser.parse_known_args(namespace=unittest)
    TestQNN.host = args.host
    TestQNN.device = args.device
    TestQNN.model = args.model
    TestQNN.build_folder = args.build_folder
    TestQNN.executorch_root = args.executorch_root
    TestQNN.artifact_dir = args.artifact_dir
    TestQNN.image_dataset = args.image_dataset
    TestQNN.qa_dataset = args.qa_dataset
    TestQNN.sentence_dataset = args.sentence_dataset
    TestQNN.pretrained_weight = args.pretrained_weight
    TestQNN.model_name = args.model_name
    TestQNN.online_prepare = args.online_prepare
    TestQNN.enable_profile = args.enable_profile
    TestQNN.error_only = args.error_only
    TestQNN.oss_repo = args.oss_repo
    TestQNN.shared_buffer = args.shared_buffer
    TestQNN.enable_x86_64 = args.enable_x86_64
    TestQNN.dump_intermediate_outputs = args.dump_intermediate_outputs
    TestQNN.compile_only = args.compile_only
    TestQNN.pre_gen_pte = args.pre_gen_pte
    TestQNN.llama_artifacts = args.llama_artifacts
    TestQNN.op_package_dir = args.op_package_dir
    return sys.argv[:1] + ns_args


if __name__ == "__main__":
    ut_args = setup_environment()
    unittest.main(argv=ut_args)
