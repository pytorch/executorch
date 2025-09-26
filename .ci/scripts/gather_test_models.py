#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# WARNING: The CI runner logic should directly be in the corresponding yml files
# This file will be deleted once the reference in periodic.yml is deleted.

import itertools
import json
import os
from typing import Any

from examples.models import MODEL_NAME_TO_MODEL
from examples.xnnpack import MODEL_NAME_TO_OPTIONS, QuantType

DEFAULT_RUNNERS = {
    "linux": "linux.2xlarge",
    "macos": "macos-m1-14",
}
CUSTOM_RUNNERS = {
    "linux": {
        # This one runs OOM on smaller runner, the root cause is unclear (T163016365)
        "w2l": "linux.4xlarge.memory",
        "ic4": "linux.4xlarge.memory",
        "resnet50": "linux.4xlarge.memory",
        "llava": "linux.4xlarge.memory",
        "llama3_2_vision_encoder": "linux.4xlarge.memory",
        "llama3_2_text_decoder": "linux.4xlarge.memory",
        # This one causes timeout on smaller runner, the root cause is unclear (T161064121)
        "dl3": "linux.4xlarge.memory",
        "emformer_join": "linux.4xlarge.memory",
        "emformer_predict": "linux.4xlarge.memory",
        "phi_4_mini": "linux.4xlarge.memory",
    }
}

DEFAULT_TIMEOUT = 90
CUSTOM_TIMEOUT = {
    # Just some examples on how custom timeout can be set
    "linux": {
        "mobilebert": 90,
        "emformer_predict": 360,
        "llama3_2_text_decoder": 360,
    },
    "macos": {
        "mobilebert": 90,
        "emformer_predict": 360,
        "llama3_2_text_decoder": 360,
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
        choices=["pull_request", "push", "schedule"],
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
        return model in ["mv3", "vit"]
    elif event == "push":
        # These are super slow. Only run it periodically
        return model not in ["dl3", "edsr", "emformer_predict"]
    else:
        return True


def model_should_run_on_target_os(model: str, target_os: str) -> bool:
    """
    A helper function to decide whether a model should be tested on a target os (linux/macos).
    For example, a big model can be disabled in macos due to the limited macos resources.
    """
    if target_os == "macos":
        # Disabled in macos due to limited resources, and should stay that way even if
        # we otherwise re-enable.
        return model not in ["llava"]
    # Disabled globally because we have test-llava-runner-linux that does a more
    # comprehensive E2E test of llava.
    return model not in ["llava"]


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
            if MODEL_NAME_TO_OPTIONS[name].quantization != QuantType.NONE:
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
