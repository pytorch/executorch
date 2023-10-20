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

# NB: Skip buck2 on MacOS to cut down the number of combinations we
# need to run there as the number of MacOS runner is limited. Buck2
# build and test has already been covered on Linux
BUILD_TOOLS = {
    "buck2": {"linux"},
    "cmake": {"linux", "macos"},
}
DEFAULT_RUNNERS = {
    "linux": "linux.2xlarge",
    "macos": "macos-m1-12",
}
CUSTOM_RUNNERS = {
    "linux": {
        # This one runs OOM on smaller runner, the root cause is unclear (T163016365)
        "w2l": "linux.12xlarge",
        "ic4": "linux.12xlarge",
        "resnet50": "linux.12xlarge",
        # This one causes timeout on smaller runner, the root cause is unclear (T161064121)
        "dl3": "linux.12xlarge",
        "emformer_join": "linux.12xlarge",
    }
}


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Gather all models to test on CI for the target OS")
    parser.add_argument(
        "--target-os",
        type=str,
        choices=["linux", "macos"],
        default="linux",
        help="the target OS",
    )
    parser.add_argument(
        "-e",
        "--event",
        type=str,
        choices=["pull_request", "push"],
        required=True,
        help=f"GitHub CI Event. See https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#on",
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
        return model in ["add", "ic3", "mv2", "mv3", "resnet18", "vit"]
    elif event == "push":
        return True
    return False


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
    for (name, build_tool, q_config, d_config) in itertools.product(MODEL_NAME_TO_MODEL.keys(), BUILD_TOOLS.keys(), [False, True], [False, True]):
        if not model_should_run_on_event(name, event):
            continue

        if q_config and ((not name in MODEL_NAME_TO_OPTIONS) or (not MODEL_NAME_TO_OPTIONS[name].quantization)):
            continue

        if d_config and ((not name in MODEL_NAME_TO_OPTIONS) or (not MODEL_NAME_TO_OPTIONS[name].delegation)):
            continue

        if target_os not in BUILD_TOOLS[build_tool]:
            continue

        record = {
            "build-tool": build_tool,
            "model": name,
            "xnnpack_quantization": q_config,
            "xnnpack_delegation": d_config,
            "runner": DEFAULT_RUNNERS.get(target_os, "linux.2xlarge"),
            # demo_backend_delegation test only supports add_mul model
            "demo_backend_delegation": name == "add_mul",
        }

        # NB: Some model requires much bigger Linux runner to avoid
        # running OOM. The team is investigating the root cause
        if target_os in CUSTOM_RUNNERS and name in CUSTOM_RUNNERS.get(
            target_os, {}
        ):
            record["runner"] = CUSTOM_RUNNERS[target_os][name]

        models["include"].append(record)

    set_output("models", json.dumps(models))


if __name__ == "__main__":
    export_models_for_ci()
