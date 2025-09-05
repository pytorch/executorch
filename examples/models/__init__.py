# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class Model(str, Enum):
    Mul = "mul"
    Linear = "linear"
    Add = "add"
    AddMul = "add_mul"
    Softmax = "softmax"
    Dl3 = "dl3"
    Edsr = "edsr"
    EmformerTranscribe = "emformer_transcribe"
    EmformerPredict = "emformer_predict"
    EmformerJoin = "emformer_join"
    Llama2 = "llama2"
    Llama = "llama"
    Llama32VisionEncoder = "llama3_2_vision_encoder"
    Lstm = "lstm"
    MobileBert = "mobilebert"
    Mv2 = "mv2"
    Mv2Untrained = "mv2_untrained"
    Mv3 = "mv3"
    Vit = "vit"
    W2l = "w2l"
    Ic3 = "ic3"
    Ic4 = "ic4"
    ResNet18 = "resnet18"
    ResNet50 = "resnet50"
    Llava = "llava"
    EfficientSam = "efficient_sam"
    Qwen25 = "qwen2_5_1_5b"
    Phi4Mini = "phi_4_mini"
    SmolLM2 = "smollm2"

    def __str__(self) -> str:
        return self.value


class Backend(str, Enum):
    XnnpackQuantizationDelegation = "xnnpack-quantization-delegation"
    CoreMlExportOnly = "coreml"
    CoreMlExportAndTest = "coreml-test"  # AOT export + test with runner

    def __str__(self) -> str:
        return self.value


MODEL_NAME_TO_MODEL = {
    str(Model.Mul): ("toy_model", "MulModule"),
    str(Model.Linear): ("toy_model", "LinearModule"),
    str(Model.Add): ("toy_model", "AddModule"),
    str(Model.AddMul): ("toy_model", "AddMulModule"),
    str(Model.Softmax): ("toy_model", "SoftmaxModule"),
    str(Model.Dl3): ("deeplab_v3", "DeepLabV3ResNet50Model"),
    str(Model.Edsr): ("edsr", "EdsrModel"),
    str(Model.EmformerTranscribe): ("emformer_rnnt", "EmformerRnntTranscriberModel"),
    str(Model.EmformerPredict): ("emformer_rnnt", "EmformerRnntPredictorModel"),
    str(Model.EmformerJoin): ("emformer_rnnt", "EmformerRnntJoinerModel"),
    str(Model.Llama2): ("llama", "Llama2Model"),
    str(Model.Llama): ("llama", "Llama2Model"),
    str(Model.Llama32VisionEncoder): ("llama3_2_vision", "FlamingoVisionEncoderModel"),
    # TODO: This take too long to export on both Linux and MacOS (> 6 hours)
    # "llama3_2_text_decoder": ("llama3_2_vision", "Llama3_2Decoder"),
    str(Model.Lstm): ("lstm", "LSTMModel"),
    str(Model.MobileBert): ("mobilebert", "MobileBertModelExample"),
    str(Model.Mv2): ("mobilenet_v2", "MV2Model"),
    str(Model.Mv2Untrained): ("mobilenet_v2", "MV2UntrainedModel"),
    str(Model.Mv3): ("mobilenet_v3", "MV3Model"),
    str(Model.Vit): ("torchvision_vit", "TorchVisionViTModel"),
    str(Model.W2l): ("wav2letter", "Wav2LetterModel"),
    str(Model.Ic3): ("inception_v3", "InceptionV3Model"),
    str(Model.Ic4): ("inception_v4", "InceptionV4Model"),
    str(Model.ResNet18): ("resnet", "ResNet18Model"),
    str(Model.ResNet50): ("resnet", "ResNet50Model"),
    str(Model.Llava): ("llava", "LlavaModel"),
    str(Model.EfficientSam): ("efficient_sam", "EfficientSAM"),
    str(Model.Qwen25): ("qwen2_5", "Qwen2_5Model"),
    str(Model.Phi4Mini): ("phi_4_mini", "Phi4MiniModel"),
    str(Model.SmolLM2): ("smollm2", "SmolLM2Model"),
}

__all__ = [
    "MODEL_NAME_TO_MODEL",
]
