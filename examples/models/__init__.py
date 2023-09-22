# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .model_factory import EagerModelFactory

MODEL_NAME_TO_MODEL = {
    "mul": ("toy_model", "MulModule"),
    "linear": ("toy_model", "LinearModule"),
    "add": ("toy_model", "AddModule"),
    "add_mul": ("toy_model", "AddMulModule"),
    "dl3": ("deeplab_v3", "DeepLabV3ResNet50Model"),
    "edsr": ("edsr", "EdsrModel"),
    "emformer_transcribe": ("emformer_rnnt", "EmformerRnntTranscriberModel"),
    "emformer_predict": ("emformer_rnnt", "EmformerRnntPredictorModel"),
    "emformer_join": ("emformer_rnnt", "EmformerRnntJoinerModel"),
    "mobilebert": ("mobilebert", "MobileBertModelExample"),
    "mv2": ("mobilenet_v2", "MV2Model"),
    "mv3": ("mobilenet_v3", "MV3Model"),
    "vit": ("torchvision_vit", "TorchVisionViTModel"),
    "w2l": ("wav2letter", "Wav2LetterModel"),
    "ic3": ("inception_v3", "InceptionV3Model"),
    "ic4": ("inception_v4", "InceptionV4Model"),
    "resnet18": ("resnet", "ResNet18Model"),
    "resnet50": ("resnet", "ResNet50Model"),
}

__all__ = [
    "EagerModelFactory",
    "MODEL_NAME_TO_MODEL",
]
