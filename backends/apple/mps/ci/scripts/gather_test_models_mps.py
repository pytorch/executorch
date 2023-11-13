#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import json
import os
from typing import Any

from examples.models import MODEL_NAME_TO_MODEL
from examples.xnnpack import MODEL_NAME_TO_OPTIONS

BUILD_TOOLS = {
    "cmake": {"macos-14"},
}
DEFAULT_RUNNERS = {
    "macos-14": "macos-executorch",
}

def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Gather all models to test on CI for macOS MPS delegate")
    parser.add_argument(
        "--target-os",
        type=str,
        choices=["macos-14"],
        default="macos-14",
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
        delegation_configs = {
            name in MODEL_NAME_TO_OPTIONS and MODEL_NAME_TO_OPTIONS[name].delegation,
        }
        for build_tool in BUILD_TOOLS.keys():
            if target_os not in BUILD_TOOLS[build_tool]:
                continue

            record = {
                "build-tool": build_tool,
                "model": name,
                "runner": DEFAULT_RUNNERS.get(target_os),
            }

            models["include"].append(record)

    set_output("models", json.dumps(models))


if __name__ == "__main__":
    export_models_for_ci()
