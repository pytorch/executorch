# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
from functools import lru_cache
from typing import List
from dataclasses import dataclass
from enum import Enum


class Model(str, Enum):
    Mv3 = "mv3"

    def __str__(self) -> str:
       return self.value

class Backend(str, Enum):
    XnnpackQuantizationDelegation = "xnnpack-quantization-delegation"

    def __str__(self) -> str:
       return self.value

@dataclass
class ModelTest:
    model: Model
    backend: Backend


@lru_cache()
def _repository_root_dir() -> str:
  workspace_dir = os.getenv("GITHUB_WORKSPACE")
  if workspace_dir is None:
    print("GITHUB_WORKSPACE is not set")
    sys.exit(1)

  repository_dir = os.getenv("REPOSITORY")
  if repository_dir is None:
    print("REPOSITORY is not set")
    sys.exit(1)

  return os.path.join(workspace_dir, repository_dir)


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
