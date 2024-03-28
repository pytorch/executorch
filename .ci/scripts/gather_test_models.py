#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import os
from typing import Any

from examples.models import MODEL_NAME_TO_MODEL
from examples.xnnpack import MODEL_NAME_TO_OPTIONS

# MODEL_NAME_TO_MODEL = {
#     "mul": ("toy_model", "MulModule"),
#     "linear": ("toy_model", "LinearModule"),
#     "add": ("toy_model", "AddModule"),
#     "add_mul": ("toy_model", "AddMulModule"),
#     "softmax": ("toy_model", "SoftmaxModule"),
#     "dl3": ("deeplab_v3", "DeepLabV3ResNet50Model"),
#     "edsr": ("edsr", "EdsrModel"),
#     "emformer_transcribe": ("emformer_rnnt", "EmformerRnntTranscriberModel"),
#     "emformer_predict": ("emformer_rnnt", "EmformerRnntPredictorModel"),
#     "emformer_join": ("emformer_rnnt", "EmformerRnntJoinerModel"),
#     "llama2": ("llama2", "Llama2Model"),
#     "mobilebert": ("mobilebert", "MobileBertModelExample"),
#     "mv2": ("mobilenet_v2", "MV2Model"),
#     "mv2_untrained": ("mobilenet_v2", "MV2UntrainedModel"),
#     "mv3": ("mobilenet_v3", "MV3Model"),
#     "vit": ("torchvision_vit", "TorchVisionViTModel"),
#     "w2l": ("wav2letter", "Wav2LetterModel"),
#     "ic3": ("inception_v3", "InceptionV3Model"),
#     "ic4": ("inception_v4", "InceptionV4Model"),
#     "resnet18": ("resnet", "ResNet18Model"),
#     "resnet50": ("resnet", "ResNet50Model"),
#     "llava_encoder": ("llava_encoder", "LlavaModel"),
# }

# from dataclasses import dataclass
# @dataclass
# class XNNPACKOptions(object):
#     quantization: bool
#     delegation: bool
#
# MODEL_NAME_TO_OPTIONS = {
#     "linear": XNNPACKOptions(True, True),
#     "add": XNNPACKOptions(True, True),
#     "add_mul": XNNPACKOptions(True, True),
#     "dl3": XNNPACKOptions(True, True),
#     "ic3": XNNPACKOptions(True, True),
#     "ic4": XNNPACKOptions(True, True),
#     "mv2": XNNPACKOptions(True, True),
#     "mv3": XNNPACKOptions(True, True),
#     "resnet18": XNNPACKOptions(True, True),
#     "resnet50": XNNPACKOptions(True, True),
#     "vit": XNNPACKOptions(False, True),
#     "w2l": XNNPACKOptions(False, True),
#     "edsr": XNNPACKOptions(True, True),
#     "mobilebert": XNNPACKOptions(False, True),  # T170286473
#     "llama2": XNNPACKOptions(False, True),
# }

DEFAULT_RUNNERS = {
    "linux": "linux.2xlarge",
    "macos": "macos-m1-stable",
}
CUSTOM_RUNNERS = {
    "linux": {
        # This one runs OOM on smaller runner, the root cause is unclear (T163016365)
        "w2l": "linux.12xlarge",
        "ic4": "linux.12xlarge",
        "resnet50": "linux.12xlarge",
        "llava_encoder": "linux.4xlarge",
        # This one causes timeout on smaller runner, the root cause is unclear (T161064121)
        "dl3": "linux.12xlarge",
        "emformer_join": "linux.12xlarge",
    }
}

DEFAULT_TIMEOUT = 90
CUSTOM_TIMEOUT = {
    # Just some examples on how custom timeout can be set
    "linux": {
        "mobilebert": 90,
    },
    "macos": {
        "mobilebert": 90,
    },
}


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Gather all models to test on CI for the target OS")
    parser.add_argument(
        "--target-os",
        type=str,
        default="linux",
        help="the target OS",
    )
    parser.add_argument(
        "-e",
        "--event",
        type=str,
        choices=["pull_request", "push"],
        required=True,
        help="GitHub CI Event. See https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#on",
    )

    return parser.parse_args()


def set_output(name: str, val: Any) -> None:
    """
    Set the GitHb output so that it can be accessed by other jobs
    """
    print(f"Setting {val} to GitHub output")

    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def model_should_run_on_event(model: str, event: str) -> bool:
    """
    A helper function to decide whether a model should be tested on an event (pull_request/push)
    We put higher priority and fast models to pull request and rest to push.
    """
    if event == "pull_request":
        return model in ["add", "ic3", "mv2", "mv3", "resnet18", "vit", "llava_encoder"]
    return True

def model_should_run_on_target_os(model: str, target_os: str) -> bool:
    """
    A helper function to decide whether a model should be tested on a target os (linux/macos).
    For example, a big model can be disabled in macos due to the limited macos resources.
    """
    if target_os == "macos":
        return model not in ["llava_encoder",]
    return True

def export_models_for_ci() -> dict[str, dict]:
    """
    This gathers all the example models that we want to test on GitHub OSS CI
    """
    args = parse_args()
    target_os = args.target_os
    event = args.event

    # This is the JSON syntax for configuration matrix used by GitHub
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    models = {"include": []}

    # Add MobileNet v3 for BUCK2 E2E validation (linux only)
    if target_os == "linux":
        for backend in ["portable", "xnnpack-quantization-delegation"]:
            record = {
                "build-tool": "buck2",
                "model": "mv3",
                "backend": backend,
                "runner": "linux.2xlarge",
                "timeout": DEFAULT_TIMEOUT,
            }
            models["include"].append(record)

    # Add all models for CMake E2E validation
    # CMake supports both linux and macos
    for name, backend in itertools.product(
        MODEL_NAME_TO_MODEL.keys(), ["portable", "xnnpack"]
    ):
        if not model_should_run_on_event(name, event):
            continue

        if not model_should_run_on_target_os(name, target_os):
            continue

        if backend == "xnnpack":
            if name not in MODEL_NAME_TO_OPTIONS:
                continue
            if MODEL_NAME_TO_OPTIONS[name].quantization:
                backend += "-quantization"

            if MODEL_NAME_TO_OPTIONS[name].delegation:
                backend += "-delegation"

        record = {
            "build-tool": "cmake",
            "model": name,
            "backend": backend,
            "runner": DEFAULT_RUNNERS.get(target_os, "linux.2xlarge"),
            "timeout": DEFAULT_TIMEOUT,
        }

        # Set the custom timeout if needed
        if target_os in CUSTOM_TIMEOUT and name in CUSTOM_TIMEOUT[target_os]:
            record["timeout"] = CUSTOM_TIMEOUT[target_os].get(name, DEFAULT_TIMEOUT)

        # NB: Some model requires much bigger Linux runner to avoid
        # running OOM. The team is investigating the root cause
        if target_os in CUSTOM_RUNNERS and name in CUSTOM_RUNNERS.get(target_os, {}):
            record["runner"] = CUSTOM_RUNNERS[target_os][name]

        models["include"].append(record)

    set_output("models", json.dumps(models))


if __name__ == "__main__":
    export_models_for_ci()
