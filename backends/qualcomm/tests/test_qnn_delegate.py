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
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from executorch.backends.qualcomm.tests.utils import (
    generate_context_binary,
    QnnPartitioner,
    QnnQuantizer,
    QuantDtype,
    TestQNN,
    to_backend,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_ANNOTATION,
    QCOM_MODULE,
    QCOM_QUANT_DTYPE,
    QCOM_SAMPLE_INPUTS,
)

from executorch.backends.qualcomm.utils.utils import (
    canonicalize_program,
    capture_program,
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    skip_annotation,
)

from executorch.examples.qualcomm.utils import setup_common_args_and_variables

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
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import disable_validation


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
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
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

    def test_qnn_backend_batch_norm(self):
        module = BatchNorm(32)  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        torch.manual_seed(8)
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

    def test_qnn_backend_chunk_single(self):
        module = Chunk()  # noqa: F405
        sample_input = (torch.randn(1, 1, 4, 3),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        module = Clamp()  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
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
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

    def test_qnn_backend_element_wise_ceil(self):
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

    def test_qnn_backend_index(self):
        module = Index()  # noqa: F405
        sample_input = (torch.randn([8, 172, 64]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index_put(self):
        module = IndexPut()  # noqa: F405
        sample_input = (
            torch.tensor([2], dtype=torch.int32),
            torch.randn([1, 1, 12, 64]),
        )
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_bilinear_2d(self):
        module = ResizeBilinear2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_interpolate_nearest_2d(self):
        module = ResizeNearest2D()  # noqa: F405
        sample_input = (torch.randn(2, 3, 4, 5),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_layer_norm(self):
        module = LayerNorm()  # noqa: F405
        sample_input = (torch.randn(196, 768),)
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
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

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

    @unittest.skip("failed to lower in QNN 2.26")
    def test_qnn_backend_mha(self):
        module = MultiheadAttention()  # noqa: F405
        sample_input = (torch.randn(1, 197, 96),)
        self.lower_module_and_test_output(module, sample_input)

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

    def test_qnn_backend_reshape(self):
        module = Reshape()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_rms_norm(self):
        module = RmsNorm()  # noqa: F405
        sample_input = (torch.abs(torch.randn([1, 1, 1, 4])),)
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
        modules = [SliceCopy(), SliceCopyWithStep()]  # noqa: F405
        sample_input = (
            torch.randn([1, 512]),
            torch.randn([1, 8]),
        )
        for module in modules:
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
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_chunk_add(self):
        module = ChunkAdd()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn(1, 2, 4, 2),)
        self.lower_module_and_test_output(module, sample_input)

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

    def test_qnn_backend_conv2d_sum_reduce_dim(self):
        module = Conv2dSumReduceDim()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
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

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        torch.manual_seed(8)
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

    def test_qnn_backend_16a4w_linear(self):
        module = Linear()  # noqa: F405
        sample_input = (torch.randn([3, 4]),)
        module = self.get_qdq_module(
            module,
            sample_input,
            quant_dtype=QuantDtype.use_16a4w,
        )
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("segfault happens in QNN 2.26")
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

    def test_qnn_backend_batch_norm(self):
        module = BatchNorm(32)  # noqa: F405
        sample_input = (torch.randn([4, 32, 16, 16]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_bmm(self):
        module = Bmm()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([4, 8, 32]), torch.randn([4, 32, 8]))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_cat(self):
        modules = [Cat2(), Cat3(), Cat4()]  # noqa: F405
        sample_input = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        for i, module in enumerate(modules):
            with self.subTest(i=i):
                module = self.get_qdq_module(module, sample_input)
                self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_chunk_single(self):
        module = Chunk()  # noqa: F405
        sample_input = (torch.randn(1, 1, 4, 3),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_clamp(self):
        module = Clamp()  # noqa: F405
        sample_input = (torch.randn((9, 4, 5, 3)),)
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
                        module = self.get_qdq_module(module, sample_input)
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
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

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

    def test_qnn_backend_index(self):
        module = Index()  # noqa: F405
        sample_input = (torch.randn([8, 172, 64]),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    def test_qnn_backend_index_put(self):
        module = IndexPut()  # noqa: F405
        sample_input = (
            torch.tensor([2], dtype=torch.int32),
            torch.randn([1, 1, 12, 64]),
        )
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

    def test_qnn_backend_layer_norm(self):
        module = LayerNorm()  # noqa: F405
        sample_input = (torch.randn(196, 768),)
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
        ]

        index = 0
        for comb in test_comb:
            for module in comb[QCOM_MODULE]:
                for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                    with self.subTest(i=index):
                        module = self.get_qdq_module(module, sample_input)
                        self.lower_module_and_test_output(module, sample_input)
                        index += 1

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
        modules = [SliceCopy(), SliceCopyWithStep()]  # noqa: F405
        sample_input = (
            torch.randn([1, 512]),
            torch.randn([1, 8]),
        )
        for module in modules:
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
            dump_intermediate_outputs=TestQNN.dump_intermediate_outputs,
            profile=TestQNN.enable_profile,
            shared_buffer=TestQNN.shared_buffer,
        )

    def test_qnn_backend_chunk_add(self):
        module = ChunkAdd()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn(1, 1, 4, 2),)
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

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

    def test_qnn_backend_conv2d_sum_reduce_dim(self):
        module = Conv2dSumReduceDim()  # noqa: F405
        sample_input = (torch.randn([1, 1, 3, 3]),)
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

    def test_qnn_backend_view_permute_matmul(self):
        module = ViewPermuteMatMul()  # noqa: F405
        torch.manual_seed(8)
        sample_input = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
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
            # {QCOM_MODULE: Llama2Model(), QCOM_ANNOTATION: (), QCOM_QUANT_DTYPE: QuantDtype.use_8a8w},
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
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
        )

    def test_qnn_backend_dump_intermediate_outputs(self):
        backend_options = generate_htp_compiler_spec(use_fp16=True)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
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
        edge_prog = to_edge(
            torch.export.export(module, sample_input),
        )
        canonicalize_program(edge_prog.exported_program())
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
            expected_profile_events=24,
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
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("segfault happens in recent torch.export.export")
    def test_qnn_backend_context_direct(self):
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
            backend_options = generate_htp_compiler_spec(use_fp16=True)
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=self.arch_table[TestQNN.model],
                backend_options=backend_options,
                is_from_context_binary=True,
            )
            lowered_module = to_backend(
                "QnnBackend", bundle_program["edge_program"], compiler_specs
            )
            self.verify_output(
                module,
                tuple(
                    torch.randn(size=v.shape, dtype=v.dtype)
                    for v in bundle_program["inputs"].values()
                ),
                lowered_module,
            )


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

    def test_qnn_backend_dump_intermediate_outputs(self):
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        TestQNN.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
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

        # define partitioner
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)
        # define quantizer
        quantizer = QnnQuantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, exported_progs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            partitioner=partitioner,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_id_set={"conv2d"},
        )
        self.assertEqual(len(exported_progs), 1)
        # lower all graph again, the skipped operators will be left in CPU
        exec_prog = to_edge(
            torch.export.export(graph_module, sample_input),
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

        # define partitioner
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)
        # define quantizer
        quantizer = QnnQuantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, exported_progs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            partitioner=partitioner,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_op_set={torch.ops.aten.add.Tensor},
        )
        self.assertEqual(len(exported_progs), 2)
        # lower all graph again, the skipped operators will be left in CPU
        exec_prog = exec_prog = to_edge(
            torch.export.export(graph_module, sample_input),
        ).to_executorch()
        self.verify_output(module, sample_input, exec_prog)

    def test_qnn_backend_graph_level_mixed_precision(self):
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))

        # define partitioner
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.arch_table[TestQNN.model],
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)
        # define quantizer
        quantizer = QnnQuantizer()

        # define calibration method
        def calibrator(gm):
            gm(*sample_input)

        # get partially lowererd graph module
        graph_module, exported_progs = skip_annotation(
            nn_module=module,
            quantizer=quantizer,
            partitioner=partitioner,
            sample_input=sample_input,
            calibration_cb=calibrator,
            fp_node_id_set={"add", "mean"},
            fallback_to_cpu=False,
        )
        self.assertEqual(len(exported_progs), 5)
        # lower all graph again, the skipped operators will be delegated with fp16
        exec_prog = to_edge(
            torch.export.export(graph_module, sample_input),
        ).to_executorch()
        self.verify_output(module, sample_input, exec_prog)

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
        edge_prog = to_edge(
            torch.export.export(module, sample_input),
        )
        canonicalize_program(edge_prog.exported_program())
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
            expected_profile_events=25,
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
            online_prepare=True,
        )
        module = SimpleModel()  # noqa: F405
        sample_input = (torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))
        module = self.get_qdq_module(module, sample_input)
        self.lower_module_and_test_output(module, sample_input)

    @unittest.skip("segfault happens in recent torch.export.export")
    def test_qnn_backend_context_direct(self):
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
            backend_options = generate_htp_compiler_spec(use_fp16=False)
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=self.arch_table[TestQNN.model],
                backend_options=backend_options,
                is_from_context_binary=True,
            )
            lowered_module = to_backend(
                "QnnBackend", bundle_program["edge_program"], compiler_specs
            )
            self.verify_output(
                module,
                tuple(
                    torch.randn(size=v.shape, dtype=v.dtype)
                    for v in bundle_program["inputs"].values()
                ),
                lowered_module,
            )


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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 60)
                self.assertGreaterEqual(msg["top_5"], 90)

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
                self.assertGreaterEqual(msg["top_5"], 90)

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
                    self.assertGreaterEqual(msg["top_1"], 60)
                    self.assertGreaterEqual(msg["top_5"], 85)

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
                self.assertGreaterEqual(msg["mAP"], 0.70)

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
                self.assertGreaterEqual(msg["top_5"], 85)

    def test_esrgan(self):
        if not self.required_envs():
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
                self.assertGreaterEqual(msg["PSNR"], 24)
                self.assertGreaterEqual(msg["SSIM"], 0.8)

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
                self.assertGreaterEqual(msg["top_1"], 40)
                self.assertGreaterEqual(msg["top_5"], 70)


class TestExampleQaihubScript(TestQNN):

    def required_envs(self, conditions=None) -> bool:
        conditions = [] if conditions is None else conditions
        return all(
            [
                self.executorch_root,
                self.artifact_dir,
                *conditions,
            ]
        )

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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 60)
                self.assertGreaterEqual(msg["top_5"], 80)

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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["top_1"], 65)
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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["PSNR"], 25)
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
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                self.assertGreaterEqual(msg["PA"], 0.85)
                self.assertGreaterEqual(msg["MPA"], 0.70)
                self.assertGreaterEqual(msg["MIoU"], 0.55)

    def test_stories_single_llama(self):
        if not self.required_envs():
            self.skipTest("missing required envs")

        cmds = [
            "python",
            f"{self.executorch_root}/examples/qualcomm/oss_scripts/llama2/llama.py",
            "--artifact",
            self.artifact_dir,
            "--build_folder",
            self.build_folder,
            "--device",
            self.device,
            "--model",
            self.model,
            "--checkpoint",
            f"{self.artifact_dir}/stories110M.pt",
            "--params",
            f"{self.artifact_dir}/params.json",
            "--tokenizer_model",
            f"{self.artifact_dir}/tokenizer.model",
            "--tokenizer_bin",
            f"{self.artifact_dir}/tokenizer.bin",
            "--ip",
            self.ip,
            "--port",
            str(self.port),
            "--prompt",
            "Once",
            "--ptq",
            "16a4w",
            "--temperature",
            "0",
        ]
        if self.host:
            cmds.extend(["--host", self.host])

        golden_start_with = "Once upon a time,"
        p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
        with Listener((self.ip, self.port)) as listener:
            conn = listener.accept()
            p.communicate()
            msg = json.loads(conn.recv())
            if "Error" in msg:
                self.fail(msg["Error"])
            else:
                model_out = msg["result"][0]
                self.assertTrue(model_out.startswith(golden_start_with))

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
    parser.add_argument(
        "--oss_repo",
        help="Path to open source software model repository",
        type=str,
    )
    parser.add_argument(
        "-x",
        "--enable_x86_64",
        help="Enable unittest to be executed on x86_64 platform",
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
    TestQNN.oss_repo = args.oss_repo
    TestQNN.shared_buffer = args.shared_buffer
    TestQNN.enable_x86_64 = args.enable_x86_64
    TestQNN.dump_intermediate_outputs = args.dump_intermediate_outputs
    return sys.argv[:1] + ns_args


if __name__ == "__main__":
    ut_args = setup_environment()
    unittest.main(argv=ut_args)
