#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def export_models_for_ci() -> None:
    """
    This gathers all the example models that we want to test on GitHub OSS CI
    """
    args = parse_args()
    target_os = args.target_os

    # This is the JSON syntax for configuration matrix used by GitHub
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    models = {"include": []}
    for name in MODEL_NAME_TO_MODEL.keys():
        quantization_configs = {
            False,
            name in MODEL_NAME_TO_OPTIONS and MODEL_NAME_TO_OPTIONS[name].quantization,
        }
        delegation_configs = {
            False,
            name in MODEL_NAME_TO_OPTIONS and MODEL_NAME_TO_OPTIONS[name].delegation,
        }
        for build_tool in BUILD_TOOLS.keys():
            if target_os not in BUILD_TOOLS[build_tool]:
                continue

            for q_config in quantization_configs:
                for d_config in delegation_configs:
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
