# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from typing import List


@cache
def _unsafe_get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise RuntimeError(f"environment variable '{key}' is not set")
    return value


@cache
def _repository_root_dir() -> str:
    return os.path.join(
        _unsafe_get_env("GITHUB_WORKSPACE"),
        _unsafe_get_env("REPOSITORY"),
    )


# For some reason, we are unable to see the entire repo in the python path.
# So manually add it.
sys.path.append(_repository_root_dir())
from examples.models import Backend, Model


@dataclass
class ModelTest:
    model: Model
    backend: Backend


def run_tests(model_tests: List[ModelTest]) -> None:
    # Why are we doing this envvar shenanigans? Since we build the testers, which
    # uses buck, we cannot run as root. This is a sneaky of getting around that
    # test.
    #
    # This can be reverted if either:
    #   - We remove usage of buck in our builds
    #   - We stop running the Docker image as root: https://github.com/pytorch/test-infra/issues/5091
    envvars = os.environ.copy()
    envvars.pop("HOME")

    for model_test in model_tests:
        subprocess.run(
            [
                os.path.join(_repository_root_dir(), ".ci/scripts/test_model.sh"),
                str(model_test.model),
                # What to build `executor_runner` with for testing.
                "cmake",
                str(model_test.backend),
            ],
            env=envvars,
            check=True,
            cwd=_repository_root_dir(),
        )
