# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from executorch.backends.qualcomm.tests.rework.src.oss import *  # noqa: F403


# -- TestExampleScript (examples/qualcomm/scripts/) --


def test_mobilenet_v2(request):
    MobileNetV2.test(request)  # noqa: F405


def test_mobilenet_v3(request):
    MobileNetV3.test(request)  # noqa: F405


def test_inception_v3(request):
    InceptionV3.test(request)  # noqa: F405


def test_inception_v4(request):
    InceptionV4.test(request)  # noqa: F405


def test_vit(request):
    Vit.test(request)  # noqa: F405


def test_edsr(request):
    Edsr.test(request)  # noqa: F405


def test_deeplab_v3(request):
    DeepLabV3.test(request)  # noqa: F405


@pytest.mark.skip("dynamic shape inputs appear in recent torch.export.export")
def test_mobilebert(request):
    MobileBert.test(request)  # noqa: F405


@pytest.mark.skip("eagar mode fake quant works well, need further investigation")
def test_ptq_mobilebert(request):
    PtqMobileBert.test(request)  # noqa: F405


@pytest.mark.skip("encountered undefined symbol in mainline, reopen once resolved")
def test_wav2letter(request):
    Wav2Letter.test(request)  # noqa: F405


def test_export_example(request):
    ExportExample.test(request)  # noqa: F405


# -- TestExampleOssScript (examples/qualcomm/oss_scripts/) --


def test_albert(request):
    Albert.test(request)  # noqa: F405


def test_bert(request):
    Bert.test(request)  # noqa: F405


def test_conv_former(request):
    ConvFormer.test(request)  # noqa: F405


def test_convnext_small(request):
    ConvNextSmall.test(request)  # noqa: F405


def test_cvt(request):
    Cvt.test(request)  # noqa: F405


def test_deit(request):
    Deit.test(request)  # noqa: F405


def test_depthanything_v2_small(request):
    DepthAnythingV2Small.test(request)  # noqa: F405


def test_dino_v2(request):
    DinoV2.test(request)  # noqa: F405


def test_distilbert(request):
    Distilbert.test(request)  # noqa: F405


def test_dit(request):
    Dit.test(request)  # noqa: F405


def test_efficientnet(request):
    EfficientNet.test(request)  # noqa: F405


def test_efficientSAM(request):
    EfficientSAM.test(request)  # noqa: F405


def test_esrgan(request):
    Esrgan.test(request)  # noqa: F405


@pytest.mark.skip("Bad accuracy on source model since transformers v5")
def test_eurobert(request):
    Eurobert.test(request)  # noqa: F405


def test_fastvit(request):
    FastVit.test(request)  # noqa: F405


def test_fbnet(request):
    FbNet.test(request)  # noqa: F405


def test_focalnet(request):
    FocalNet.test(request)  # noqa: F405


def test_gMLP(request):
    GMLP.test(request)  # noqa: F405


def test_maxvit_t(request):
    MaxVitT.test(request)  # noqa: F405


def test_mobilevit_v2(request):
    MobileViTV2.test(request)  # noqa: F405


def test_mobilevit_v1(request):
    MobileViTV1.test(request)  # noqa: F405


def test_pvt(request):
    Pvt.test(request)  # noqa: F405


def test_regnet(request):
    RegNet.test(request)  # noqa: F405


def test_retinanet(request):
    RetinaNet.test(request)  # noqa: F405


def test_roberta(request):
    Roberta.test(request)  # noqa: F405


def test_squeezenet(request):
    SqueezeNet.test(request)  # noqa: F405


def test_ssd300_vgg16(request):
    Ssd300Vgg16.test(request)  # noqa: F405


def test_swin_transformer(request):
    SwinTransformer.test(request)  # noqa: F405


def test_swin_v2_t(request):
    SwinV2T.test(request)  # noqa: F405


def test_t5(request):
    T5.test(request)  # noqa: F405


def test_vit_b_16(request):
    VitB16.test(request)  # noqa: F405


def test_whisper(request):
    Whisper.test(request)  # noqa: F405
