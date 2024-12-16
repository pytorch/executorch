# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

MODEL_NAME_TO_MODEL = {
    "mul": ("toy_model", "MulModule"),
    "linear": ("toy_model", "LinearModule"),
    "add": ("toy_model", "AddModule"),
    "add_mul": ("toy_model", "AddMulModule"),
    "softmax": ("toy_model", "SoftmaxModule"),
    "dl3": ("deeplab_v3", "DeepLabV3ResNet50Model"),
    "edsr": ("edsr", "EdsrModel"),
    "emformer_transcribe": ("emformer_rnnt", "EmformerRnntTranscriberModel"),
    "emformer_predict": ("emformer_rnnt", "EmformerRnntPredictorModel"),
    "emformer_join": ("emformer_rnnt", "EmformerRnntJoinerModel"),
    "llama2": ("llama", "Llama2Model"),
    "llama": ("llama", "Llama2Model"),
    "llama3_2_vision_encoder": ("llama3_2_vision", "FlamingoVisionEncoderModel"),
    # TODO: This take too long to export on both Linux and MacOS (> 6 hours)
    # "llama3_2_text_decoder": ("llama3_2_vision", "Llama3_2Decoder"),
    "lstm": ("lstm", "LSTMModel"),
    "mobilebert": ("mobilebert", "MobileBertModelExample"),
    "mv2": ("mobilenet_v2", "MV2Model"),
    "mv2_untrained": ("mobilenet_v2", "MV2UntrainedModel"),
    "mv3": ("mobilenet_v3", "MV3Model"),
    "vit": ("torchvision_vit", "TorchVisionViTModel"),
    "w2l": ("wav2letter", "Wav2LetterModel"),
    "ic3": ("inception_v3", "InceptionV3Model"),
    "ic4": ("inception_v4", "InceptionV4Model"),
    "resnet18": ("resnet", "ResNet18Model"),
    "resnet50": ("resnet", "ResNet50Model"),
    "llava": ("llava", "LlavaModel"),
    "efficient_sam": ("efficient_sam", "EfficientSAM"),
}

__all__ = [
    "MODEL_NAME_TO_MODEL",
]
