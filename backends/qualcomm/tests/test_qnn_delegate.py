# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import subprocess
import sys
import tempfile
import unittest
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from executorch.backends.qualcomm.tests.utils import (
    QnnPartitioner,
    QuantDtype,
    TestQNN,
    to_backend,
)

from executorch.backends.qualcomm.utils.utils import (
    canonicalize_program,
    capture_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)

from executorch.examples.qualcomm.scripts.utils import setup_common_args_and_variables

from executorch.backends.qualcomm.tests.models import *  # noqa: F403

from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.models.inception_v3 import InceptionV3Model
from executorch.examples.models.inception_v4 import InceptionV4Model

# from executorch.examples.models.llama2 import Llama2Model
from executorch.examples.models.mobilebert import MobileBertModelExample
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.examples.models.mobilenet_v3 import MV3Model
from executorch.examples.models.torchvision_vit.model import TorchVisionViTModel
from executorch.examples.models.wav2letter import Wav2LetterModel
from executorch.exir.backend.backend_api import disable_validation
from executorch.exir.program._program import EdgeCompileConfig, ExirExportedProgram


class TestQNNFloatingPointOperator(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            tensor_dump_output_path="",
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_arange(self):
        module = Arange(5)  # noqa: F405
        sample_input = (torch.randn(5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_avg_pool2d(self):
        module = AvgPoolModule()  # noqa: F405
        sample_input = (torch.randn(1, 3, 2, 2),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        sample_input = (torch.randn([4, 8, 32]), torch.randn([4, 32, 8]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cast(self):
        module = Cast()  # noqa: F405
        sample_input = (10 * torch.rand((9, 4, 5, 3)),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cat(self):
        modules = [Cat2(), Cat3(), Cat4()]  # noqa: F405
        sample_input = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        module = Clamp()  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d(self):
        module = Conv1DSequential()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d(self):
        module = Conv2DSequential()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_add(self):
        test_comb = [
            {
                "module": [Add()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [AddConstantFloat()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_ceil(self):
        module = Ceil()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_div(self):
        eps = 1e-03
        test_comb = [
            {
                "module": [Div()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                "module": [DivConstantFloat()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_mul(self):
        test_comb = [
            {
                "module": [Mul()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [MulConstantFloat()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
            {
                "module": [MulScalar()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    @unittest.skip("not yet implemented")
    def test_qnn_backend_element_wise_sqrt(self):
        modules = [Sqrt(), SqrtConstant()]  # noqa: F405
        sample_input = (torch.randn([3, 1]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_sub(self):
        test_comb = [
            {
                "module": [Sub()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [SubConstantFloat()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_embedding(self):
        module = Embedding()  # noqa: F405
        sample_input = (torch.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to(torch.int32),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_expand_copy(self):
        module = ExpandCopy()  # noqa: F405
        sample_input = (torch.randn([3, 1]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gelu(self):
        module = Gelu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
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

    def test_qnn_backend_interpolate(self):
        module = StaticResizeBilinear2DSizeModule()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_layer_norm(self):
        module = LayerNorm()  # noqa: F405
        sample_input = (torch.randn(196, 768),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log_softmax(self):
        module = LogSoftmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
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

    @unittest.skip("it will hang in runtime")
    def test_qnn_backend_mha(self):
        module = MultiheadAttention()  # noqa: F405
        sample_input = (torch.randn(1, 197, 96),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pad(self):
        module = Pad()  # noqa: F405
        sample_input = (torch.randn([1, 8, 128]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_shuffle(self):
        module = PixelShuffle()  # noqa: F405
        sample_input = (torch.ones([2, 4, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pow_tensor_scalar(self):
        module = PowTensorScalar()  # noqa: F405
        sample_input = (torch.rand([2, 4, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_relu(self):
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_reshape(self):
        module = Reshape()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rsqrt(self):
        module = Rsqrt()  # noqa: F405
        sample_input = (torch.abs(torch.randn([3, 4])),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sdpa(self):
        module = ScaledDotProductAttention()  # noqa: F405
        mask = torch.tril(torch.randn(1, 1, 100, 100))
        mask[mask == 0] = float("-inf")
        sample_input = (
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            mask,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sigmoid(self):
        module = Sigmoid()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_select_copy(self):
        module = SelectCopy()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_slice_copy(self):
        module = SliceCopy()  # noqa: F405
        sample_input = (
            torch.randn([1, 512]),
            torch.randn([1, 8]),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_stack(self):
        module = Stack()  # noqa: F405
        sample_input = (torch.randn([1, 2, 3, 4]), torch.randn([1, 2, 3, 4]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_softmax(self):
        module = Softmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_squeeze(self):
        module = Squeeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_tanh(self):
        module = Tanh()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.expectedFailure
    def test_qnn_backend_unbind(self):
        module = Unbind()  # noqa: F405
        sample_input = (torch.randn([3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_unsqueeze(self):
        module = Unsqueeze()  # noqa: F405
        sample_input = (torch.randn([1, 3, 3]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view(self):
        module = View()  # noqa: F405
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        self.lower_module_and_test_output(module, sample_input)


class TestQNNFloatingPointModel(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            tensor_dump_output_path="",
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_conv1d_relu_log_softmax(self):
        module = Conv1dReluLogSoftmax()  # noqa: F405
        sample_input = (torch.rand(1, 2, 28),)
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

    def test_qnn_backend_conv2d_max_pool2d(self):
        module = Conv2dMaxPool2d()  # noqa: F405
        sample_input = (torch.rand(1, 2, 14, 14),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_residual_block(self):
        module = ResidualBlockModule()  # noqa: F405
        sample_input = (torch.randn(1, 32, 28, 28),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_simple_model(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_example_models(self):
        instances = [
            DeepLabV3ResNet101Model(),
            EdsrModel(),
            InceptionV3Model(),
            InceptionV4Model(),
            # The module of llama is changing frequently. Reopen it when it's stable
            # Llama2Model(),
            MV2Model(),
            MV3Model(),
            MobileBertModelExample(),
            TorchVisionViTModel(),
            Wav2LetterModel(),
        ]
        expected_partitions = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
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
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            tensor_dump_output_path="",
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_16a4w_conv2d(self):
        module = Conv2DSingle()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        module = self.get_qdq_module(
            module, sample_input, quant_dtype=QuantDtype.use_16a4w
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_16a4w_per_channel_linear(self):
        module = Linear(use_bias=False)  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            is_linear_per_channel=True,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    # Is not enabled in the current qnn sdk release
    @unittest.expectedFailure
    def test_qnn_backend_16a4w_per_channel_linear_with_bias(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            is_linear_per_channel=True,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_arange(self):
        module = Arange(5)  # noqa: F405
        sample_input = (torch.randn(5),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_avg_pool2d(self):
        module = AvgPoolModule()  # noqa: F405
        sample_input = (torch.randn(1, 3, 2, 2),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        sample_input = (torch.randn([4, 8, 32]), torch.randn([4, 32, 8]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("not applicable")
    def test_qnn_backend_cast(self):
        module = Cast()  # noqa: F405
        sample_input = (10 * torch.rand((9, 4, 5, 3)),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cat(self):
        modules = [Cat2(), Cat3(), Cat4()]  # noqa: F405
        sample_input = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        module = Clamp()  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv1d(self):
        module = Conv1DSequential()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_conv2d(self):
        module = Conv2DSequential()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_add(self):
        test_comb = [
            {
                "module": [Add()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [AddConstantFloat(), AddConstantLong()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_ceil(self):
        module = Ceil()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_div(self):
        eps = 1e-03
        test_comb = [
            {
                "module": [Div()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), eps + torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), eps + torch.randn([4, 1])),
                ],
            },
            {
                "module": [DivConstantFloat(), DivConstantLong()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_mul(self):
        test_comb = [
            {
                "module": [Mul()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [MulConstantFloat(), MulConstantLong()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
            {
                "module": [MulScalar()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    @unittest.skip("not yet implemented")
    def test_qnn_backend_element_wise_sqrt(self):
        modules = [Sqrt(), SqrtConstant()]  # noqa: F405
        sample_input = (torch.randn([3, 1]),)
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_element_wise_sub(self):
        test_comb = [
            {
                "module": [Sub()],  # noqa: F405
                "sample_inputs": [
                    (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3)),
                    (torch.randn([2, 5, 1, 3]), torch.randn([4, 1])),
                ],
            },
            {
                "module": [SubConstantFloat(), SubConstantLong()],  # noqa: F405
                "sample_inputs": [(torch.randn(2, 5, 1, 3),)],
            },
        ]

        index = 0
        for comb in test_comb:
            for module in comb["module"]:
                for sample_input in comb["sample_inputs"]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_embedding(self):
        module = Embedding()  # noqa: F405
        sample_input = (torch.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to(torch.int32),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_expand_copy(self):
        module = ExpandCopy()  # noqa: F405
        sample_input = (torch.randn([3, 1]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_gelu(self):
        module = Gelu()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
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

    def test_qnn_backend_interpolate(self):
        module = StaticResizeBilinear2DSizeModule()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_layer_norm(self):
        module = LayerNorm()  # noqa: F405
        sample_input = (torch.randn(196, 768),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_log_softmax(self):
        module = LogSoftmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
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

    def test_qnn_backend_pad(self):
        module = Pad()  # noqa: F405
        sample_input = (torch.randn([1, 8, 128]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pixel_shuffle(self):
        module = PixelShuffle()  # noqa: F405
        sample_input = (torch.ones([2, 4, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_pow_tensor_scalar(self):
        module = PowTensorScalar()  # noqa: F405
        sample_input = (torch.rand([2, 4, 3, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_relu(self):
        module = Relu()  # noqa: F405
        sample_input = (torch.randn([2, 5, 1, 3]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_reshape(self):
        module = Reshape()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rsqrt(self):
        module = Rsqrt()  # noqa: F405
        sample_input = (torch.abs(torch.randn([3, 4])),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_sdpa(self):
        module = ScaledDotProductAttention()  # noqa: F405
        mask = torch.tril(torch.randn(1, 1, 100, 100))
        mask[mask == 0] = torch.finfo(torch.float32).min
        sample_input = (
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            mask,
        )
        module = self.get_qdq_module(module, sample_input)
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

    def test_qnn_backend_slice_copy(self):
        module = SliceCopy()  # noqa: F405
        sample_input = (
            torch.randn([1, 512]),
            torch.randn([1, 8]),
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_softmax(self):
        module = Softmax()  # noqa: F405
        sample_input = (torch.randn([1, 4, 8, 8]),)
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
        )
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_tanh(self):
        module = Tanh()  # noqa: F405
        sample_input = (torch.randn(2, 5, 1, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.expectedFailure
    def test_qnn_backend_unbind(self):
        module = Unbind()  # noqa: F405
        sample_input = (torch.randn([3, 3]),)
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


class TestQNNQuantizedModel(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=TestQNN.online_prepare,
            tensor_dump_output_path="",
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_conv1d_relu_log_softmax(self):
        module = Conv1dReluLogSoftmax()  # noqa: F405
        sample_input = (torch.rand(1, 2, 28),)
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

    def test_qnn_backend_conv2d_max_pool2d(self):
        module = Conv2dMaxPool2d()  # noqa: F405
        sample_input = (torch.rand(1, 2, 14, 14),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_example_models(self):
        instances = [
            {"module": DeepLabV3ResNet101Model(), "annotation": ()},
            {"module": EdsrModel(), "annotation": ()},
            {"module": InceptionV3Model(), "annotation": ()},
            {"module": InceptionV4Model(), "annotation": ()},
            # The module of llama is changing frequently. Reopen it when it's stable
            # {"module": Llama2Model(), "annotation": ()},
            {"module": MV2Model(), "annotation": ()},
            {"module": MV3Model(), "annotation": ()},
            # only works on QNN 2.12 so far
            # { 'module': MobileBertModelExample(), 'annotation': () },
            {"module": TorchVisionViTModel(), "annotation": ()},
            {"module": Wav2LetterModel(), "annotation": ()},
        ]
        expected_partitions = [
            1,
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
                module = instance["module"].get_eager_model().eval()
                sample_input = instance["module"].get_example_inputs()
                module = self.get_qdq_module(
                    module,
                    sample_input,
                    custom_quant_annotations=instance["annotation"],
                )
                self.lower_module_and_test_output(
                    module,
                    sample_input,
                    expected_partitions=expected_partitions[i],
                    assert_output_equal=False,
                )

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

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)
        # check if requantization work by reusing the 8bit qdq module
        module = self.get_qdq_module(
            module, sample_input, quant_dtype=QuantDtype.use_16a16w
        )
        self.lower_module_and_test_output(module, sample_input)


class TestQNNFloatingPointUtils(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1e-1
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
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

    def test_qnn_backend_multi_contexts(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        edge_prog = capture_program(module, sample_input)
        self.split_graph(edge_prog.exported_program.graph_module, 4)

        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)
        edge_prog.exported_program = to_backend(edge_prog.exported_program, partitioner)
        canonicalize_program(edge_prog.exported_program)
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_multi_contexts_composite(self):
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = CompositeDelegateModule(  # noqa: F405
            compiler_specs=compiler_specs,
            partitioner_type=QnnPartitioner,
            capture_method=capture_program,
            lowered_method=to_backend,
        )
        sample_input = module.get_random_input()
        edge_prog = ExirExportedProgram(
            torch.export.export(module, sample_input),
            after_to_edge_passes=False,
        ).to_edge(EdgeCompileConfig(_check_ir_validity=False))
        canonicalize_program(edge_prog.exported_program)
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module.get_reference_module(), sample_input, exec_prog)

    def test_qnn_backend_profile_op(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            profile=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=1,
            expected_profile_events=25,
        )

    def test_qnn_backend_shared_buffer(self):
        TestQNN.shared_buffer = True
        backend_options = generate_htp_compiler_spec(
            use_fp16=True,
        )
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
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
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(module, sample_input)


class TestQNNQuantizedUtils(TestQNN):
    # TODO: refactor to support different backends
    def setUp(self):
        TestQNN.atol = 1e-1
        TestQNN.rtol = 1
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
        )

    def test_qnn_backend_skip_node_id(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=3,
            skip_node_id_set={"aten_add_tensor", "aten_mean_dim"},
        )

    def test_qnn_backend_skip_node_op(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(
            module,
            sample_input,
            expected_partitions=2,
            skip_node_op_set={"aten.add.Tensor"},
        )

    def test_qnn_backend_multi_contexts(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        edge_prog = capture_program(module, sample_input)
        self.split_graph(edge_prog.exported_program.graph_module, 4)

        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)
        edge_prog.exported_program = to_backend(edge_prog.exported_program, partitioner)
        canonicalize_program(edge_prog.exported_program)
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_multi_contexts_composite(self):
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            use_dlbc=True,
            use_multi_contexts=True,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        module = CompositeDelegateModule(  # noqa: F405
            compiler_specs=compiler_specs,
            partitioner_type=QnnPartitioner,
            capture_method=capture_program,
            lowered_method=to_backend,
            quantize_method=self.get_qdq_module,
        )
        sample_input = module.get_random_input()
        edge_prog = ExirExportedProgram(
            torch.export.export(module, sample_input),
            after_to_edge_passes=False,
        ).to_edge(EdgeCompileConfig(_check_ir_validity=False))
        canonicalize_program(edge_prog.exported_program)
        exec_prog = edge_prog.to_executorch()
        self.verify_output(module.get_reference_module(), sample_input, exec_prog)

    def test_qnn_backend_profile_op(self):
        TestQNN.enable_profile = True
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
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
            expected_profile_events=26,
        )

    def test_qnn_backend_shared_buffer(self):
        TestQNN.shared_buffer = True
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
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
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)


class TestExampleOssScript(TestQNN):
    def required_envs(self, conditions=None) -> bool:
        conditions = [] if conditions is None else conditions
        return all(
            [
                self.executorch_root,
                self.artifact_dir,
                *conditions,
            ]
        )

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
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            self.assertGreaterEqual(msg["top_1"], 60)
            self.assertGreaterEqual(msg["top_5"], 90)


class TestExampleScript(TestQNN):
    def required_envs(self, conditions=None) -> bool:
        conditions = [] if conditions is None else conditions
        return all(
            [
                self.executorch_root,
                self.artifact_dir,
                *conditions,
            ]
        )

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
            self.assertGreaterEqual(msg["top_1"], 60)
            self.assertGreaterEqual(msg["top_5"], 80)

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
            self.assertGreaterEqual(msg["top_1"], 60)
            self.assertGreaterEqual(msg["top_5"], 80)

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
            self.assertGreaterEqual(msg["top_1"], 60)
            self.assertGreaterEqual(msg["top_5"], 80)

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
            self.assertGreaterEqual(msg["top_1"], 70)
            self.assertGreaterEqual(msg["top_5"], 90)

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
            self.assertGreaterEqual(msg["PNSR"], 25)
            self.assertGreaterEqual(msg["SSIM"], 0.8)

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
            self.assertGreaterEqual(msg["PA"], 0.85)
            self.assertGreaterEqual(msg["MPA"], 0.70)
            self.assertGreaterEqual(msg["MIoU"], 0.55)

    def test_dummy_llama2(self):
        self.skipTest(
            "The module of llama is changing frequently. Reopen it when it's stable"
        )
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/dummy_llama2.py",
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
            self.assertTrue(msg["is_close"])

    @unittest.expectedFailure
    def test_ptq_dummy_llama2(self):
        self.skipTest(
            "The module of llama is changing frequently. Reopen it when it's stable"
        )
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/scripts/dummy_llama2.py",
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
        if self.shared_buffer:
            cmds.extend(["--shared_buffer"])

        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            self.assertTrue(all(msg["is_close"]))

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
            cpu, htp = msg["CPU"], msg["HTP"]
            for k, v in cpu.items():
                self.assertLessEqual(abs(v[0] - htp[k][0]), 2)

    @unittest.expectedFailure
    def test_ptq_mobilebert(self):
        # TODO: 2 approaches to resolve accuracy issue
        # 1. fallback embedding layers:
        #    - skip annotation in quantizer (need PR to provide helper funciton)
        #    - skip operators in partitioner (use existent "skip_node_op_set")
        # 2. investigate different quantization configurations / mechanisms
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
            cpu, htp = msg["CPU"], msg["HTP"]
            for k, v in cpu.items():
                self.assertLessEqual(abs(v[0] - htp[k][0]), 5)

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
        "-o",
        "--online_prepare",
        help="Conduct on-device graph compilation",
        action="store_true",
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

    args, ns_args = parser.parse_known_args(namespace=unittest)
    TestQNN.host = args.host
    TestQNN.device = args.device
    TestQNN.model = args.model
    TestQNN.build_folder = args.build_folder
    TestQNN.executorch_root = args.executorch_root
    TestQNN.artifact_dir = args.artifact_dir
    TestQNN.image_dataset = args.image_dataset
    TestQNN.pretrained_weight = args.pretrained_weight
    TestQNN.model_name = args.model_name
    TestQNN.online_prepare = args.online_prepare
    TestQNN.enable_profile = args.enable_profile
    TestQNN.error_only = args.error_only
    TestQNN.shared_buffer = args.shared_buffer
    return sys.argv[:1] + ns_args


if __name__ == "__main__":
    ut_args = setup_environment()
    unittest.main(argv=ut_args)
