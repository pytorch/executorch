# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path

SUITES_PATH = Path(__file__).with_name("suites.json")
SUITES = json.loads(SUITES_PATH.read_text())
CACHE_ROOT = (
    Path(
        os.environ.get(
            "EXECUTORCH_EVAL_CACHE",
            str(
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "executorch-evals"
            ),
        )
    )
    .expanduser()
    .resolve()
)
HARNESS_CACHE = CACHE_ROOT / "terminal-bench"
TASK_ROOT = HARNESS_CACHE / "tasks" / SUITES["dataset"]["revision"]
